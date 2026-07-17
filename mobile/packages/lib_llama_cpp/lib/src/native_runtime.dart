import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:lib_llama_cpp_ffi/lib_llama_cpp_ffi.dart';
import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';

import 'llama_command.dart';
import 'llama_content.dart';
import 'llama_response.dart';
import 'llama_state.dart';
import 'llama_tool.dart';
import 'tool_aware_streaming.dart';

final class NativeLlamaRuntime {
  NativeLlamaRuntime({required LlamaCppLibraryDescriptor library})
    : this._fromDynamicLibrary(_openDynamicLibrary(library));

  NativeLlamaRuntime._fromDynamicLibrary(DynamicLibrary library)
    : _bindings = LlamaCppBindings(library),
      _wrapper = _LibLlamaCppWrapper(library) {
    _bindings.llama_backend_init();
  }

  final LlamaCppBindings _bindings;
  final _LibLlamaCppWrapper _wrapper;
  _LoadedModel? _loaded;

  LlamaState loadModel(LlamaLoadModelCommand command) {
    _disposeLoaded();

    final file = File(command.modelPath);
    if (!file.existsSync()) {
      throw NativeLlamaException(
        'Model file does not exist: ${command.modelPath}',
      );
    }

    final path = command.modelPath.toNativeUtf8();
    Pointer<llama_model> model = nullptr;
    Pointer<llama_context> context = nullptr;
    Pointer<Void> mediaContext = nullptr;

    try {
      final modelParams = _bindings.llama_model_default_params();
      modelParams.n_gpu_layers = command.gpuLayerCount ?? 0;

      model = _bindings.llama_model_load_from_file(
        path.cast<Char>(),
        modelParams,
      );
      if (model == nullptr) {
        throw NativeLlamaException(
          'Failed to load model: ${command.modelPath}',
        );
      }

      final contextParams = _bindings.llama_context_default_params();
      contextParams.n_ctx = command.contextSize ?? 0;
      final threads = math.max(1, Platform.numberOfProcessors - 1);
      contextParams.n_threads = threads;
      contextParams.n_threads_batch = threads;

      context = _bindings.llama_new_context_with_model(model, contextParams);
      if (context == nullptr) {
        throw NativeLlamaException(
          'Failed to create llama.cpp context for: ${command.modelPath}',
        );
      }

      final vocab = _bindings.llama_model_get_vocab(model);
      if (vocab == nullptr) {
        throw NativeLlamaException(
          'Failed to read model vocabulary: ${command.modelPath}',
        );
      }

      final chatTemplate = _modelChatTemplate(model);
      var supportsImage = false;
      var supportsAudio = false;
      if (command.mmprojPath != null) {
        final mmprojFile = File(command.mmprojPath!);
        if (!mmprojFile.existsSync()) {
          throw NativeLlamaException(
            'Multimodal projector file does not exist: ${command.mmprojPath}',
          );
        }
        mediaContext = _wrapper.mediaInit(
          mmprojPath: command.mmprojPath!,
          model: model,
          useGpu: command.mmprojUseGpu,
          imageMinTokens: command.imageMinTokens,
          imageMaxTokens: command.imageMaxTokens,
        );
        supportsImage = _wrapper.mediaSupportsVision(mediaContext);
        supportsAudio = _wrapper.mediaSupportsAudio(mediaContext);
      }

      _loaded = _LoadedModel(
        modelPath: command.modelPath,
        model: model,
        context: context,
        vocab: vocab,
        mmprojPath: command.mmprojPath,
        mediaContext: mediaContext,
        supportsImage: supportsImage,
        supportsAudio: supportsAudio,
      );

      return LlamaState(
        modelPath: command.modelPath,
        isModelLoaded: true,
        capabilities: LlamaModelCapabilities(
          text: true,
          chatTemplate: chatTemplate != null,
          image: supportsImage,
          audio: supportsAudio,
          tools: chatTemplate != null,
        ),
      );
    } catch (_) {
      if (mediaContext != nullptr) {
        _wrapper.mediaFree(mediaContext);
      }
      if (context != nullptr) {
        _bindings.llama_free(context);
      }
      if (model != nullptr) {
        _bindings.llama_model_free(model);
      }
      rethrow;
    } finally {
      calloc.free(path);
    }
  }

  Iterable<LlamaResponse> generate(LlamaGenerateCommand command) sync* {
    final loaded = _loaded;
    if (loaded == null) {
      throw const NativeLlamaException(
        'Cannot generate before a model is loaded.',
      );
    }

    if (command.maxTokens != null && command.maxTokens! <= 0) {
      return;
    }

    final promptTokens = _tokenize(loaded.vocab, command.prompt);
    if (promptTokens.isEmpty) {
      throw const NativeLlamaException('Prompt produced no tokens.');
    }

    final contextSize = _bindings.llama_n_ctx(loaded.context);
    if (promptTokens.length >= contextSize) {
      throw NativeLlamaException(
        'Prompt token count ${promptTokens.length} exceeds context size $contextSize.',
      );
    }

    _decodeTokens(loaded.context, promptTokens);
    yield* _sampleFromEvaluatedPrompt(
      loaded: loaded,
      command: command,
      initialTokenCount: promptTokens.length,
    );
  }

  Iterable<LlamaResponse> generateMessages(
    LlamaGenerateMessagesCommand command,
  ) sync* {
    final loaded = _loaded;
    if (loaded == null) {
      throw const NativeLlamaException(
        'Cannot generate before a model is loaded.',
      );
    }
    if (command.hasMedia && loaded.mmprojPath == null) {
      throw const NativeLlamaException(
        'Image and audio inputs require a multimodal projector.',
      );
    }

    final mediaInputs = <_MediaInput>[];
    final templateResult = _wrapper.applyChatTemplate(
      model: loaded.model,
      request: _chatTemplateRequest(command, mediaInputs),
    );
    final samplingCommand = LlamaGenerateCommand(
      prompt: templateResult.prompt,
      maxTokens: command.maxTokens,
      temperature: command.temperature,
      topP: command.topP,
      stop: [...templateResult.additionalStops, ...command.stop],
    );

    final initialTokenCount = _evaluateMessagePrompt(
      loaded: loaded,
      prompt: templateResult.prompt,
      mediaInputs: mediaInputs,
    );

    final grammar = _samplingGrammarFor(templateResult);
    yield* streamToolAwareMessageResponses(
      sampled: _sampleFromEvaluatedPrompt(
        loaded: loaded,
        command: samplingCommand,
        initialTokenCount: initialTokenCount,
        grammar: grammar,
        grammarLazy: grammar != null && templateResult.grammarLazy,
        grammarTriggers: grammar == null
            ? const []
            : templateResult.grammarTriggers,
      ),
      command: command,
      parseChatOutput: (text, {required isPartial}) => _wrapper.parseChatOutput(
        text: text,
        format: templateResult.format,
        generationPrompt: templateResult.generationPrompt,
        parser: templateResult.parser,
        isPartial: isPartial,
      ),
    );
  }

  String? _samplingGrammarFor(_ChatTemplateResult templateResult) {
    if (templateResult.grammar.isEmpty) {
      return null;
    }

    // llama.cpp's Gemma4 tool grammar sampler is currently not usable from
    // this runtime: non-lazy grammar can throw through FFI, while lazy grammar
    // can stay in trigger-wait mode and hide tool behavior. Keep the tool-aware
    // prompt and output parser, but avoid installing either sampler path.
    if (templateResult.format == 'peg-gemma4') {
      return null;
    }

    return templateResult.grammar;
  }

  Iterable<LlamaResponse> _sampleFromEvaluatedPrompt({
    required _LoadedModel loaded,
    required LlamaGenerateCommand command,
    required int initialTokenCount,
    String? grammar,
    bool grammarLazy = false,
    List<_GrammarTrigger> grammarTriggers = const [],
  }) sync* {
    final contextSize = _bindings.llama_n_ctx(loaded.context);
    final maxTokens = command.maxTokens ?? contextSize - initialTokenCount;
    if (maxTokens <= 0) {
      return;
    }

    final sampler = _createSampler(
      loaded.vocab,
      command,
      grammar: grammar,
      grammarLazy: grammarLazy,
      grammarTriggers: grammarTriggers,
    );
    final stopMatcher = _StopMatcher(command.stop);
    var generated = 0;
    // Carries an incomplete trailing UTF-8 sequence from one token to the
    // next, so characters split across token boundaries decode correctly.
    final pendingBytes = <int>[];

    try {
      while (generated < maxTokens) {
        final token = _bindings.llama_sampler_sample(
          sampler,
          loaded.context,
          -1,
        );
        if (_bindings.llama_vocab_is_eog(loaded.vocab, token)) {
          break;
        }

        _bindings.llama_sampler_accept(sampler, token);
        pendingBytes.addAll(_tokenToBytes(loaded.vocab, token));
        final cut = _utf8TailStart(pendingBytes);
        final piece = cut == 0
            ? ''
            : utf8.decode(pendingBytes.sublist(0, cut), allowMalformed: true);
        pendingBytes.removeRange(0, cut);
        final delta = piece.isEmpty ? '' : stopMatcher.add(piece);
        if (delta.isNotEmpty) {
          yield LlamaTokenResponse(text: delta, index: generated);
        }
        generated += 1;

        if (stopMatcher.isStopped) {
          break;
        }
        if (generated >= maxTokens) {
          break;
        }
        if (initialTokenCount + generated >= contextSize) {
          throw NativeLlamaException(
            'Generation exceeded context size $contextSize.',
          );
        }

        _decodeTokens(loaded.context, [token]);
      }

      // A truncated character can be left over when generation ends mid-
      // sequence (maxTokens cut). Decode it leniently rather than drop bytes.
      if (pendingBytes.isNotEmpty) {
        final leftover = stopMatcher.add(
          utf8.decode(pendingBytes, allowMalformed: true),
        );
        pendingBytes.clear();
        if (leftover.isNotEmpty) {
          yield LlamaTokenResponse(text: leftover, index: generated);
        }
      }

      final tail = stopMatcher.flush();
      if (tail.isNotEmpty) {
        yield LlamaTokenResponse(text: tail, index: generated);
      }
    } finally {
      _bindings.llama_sampler_free(sampler);
    }
  }

  int _evaluateMessagePrompt({
    required _LoadedModel loaded,
    required String prompt,
    required List<_MediaInput> mediaInputs,
  }) {
    if (mediaInputs.isEmpty) {
      final promptTokens = _tokenize(loaded.vocab, prompt);
      if (promptTokens.isEmpty) {
        throw const NativeLlamaException('Prompt produced no tokens.');
      }

      final contextSize = _bindings.llama_n_ctx(loaded.context);
      if (promptTokens.length >= contextSize) {
        throw NativeLlamaException(
          'Prompt token count ${promptTokens.length} exceeds context size $contextSize.',
        );
      }

      _decodeTokens(loaded.context, promptTokens);
      return promptTokens.length;
    }

    if (loaded.mediaContext == nullptr) {
      throw const NativeLlamaException(
        'Image and audio inputs require a multimodal projector.',
      );
    }
    if (mediaInputs.any((input) => input.isImage) && !loaded.supportsImage) {
      throw const NativeLlamaException(
        'The loaded multimodal projector does not support image input.',
      );
    }
    if (mediaInputs.any((input) => input.isAudio) && !loaded.supportsAudio) {
      throw const NativeLlamaException(
        'The loaded multimodal projector does not support audio input.',
      );
    }

    final blobs = <Pointer<Void>>[];
    try {
      for (final input in mediaInputs) {
        blobs.add(_wrapper.mediaBlob(loaded.mediaContext, input));
      }
      return _wrapper.mediaEvalPrompt(
        mediaContext: loaded.mediaContext,
        llamaContext: loaded.context,
        prompt: prompt,
        blobs: blobs,
        nBatch: _bindings.llama_n_batch(loaded.context),
      );
    } finally {
      for (final blob in blobs) {
        _wrapper.mediaBlobFree(blob);
      }
    }
  }

  Map<String, Object?> _chatTemplateRequest(
    LlamaGenerateMessagesCommand command,
    List<_MediaInput> mediaInputs,
  ) {
    return {
      'messages': [
        for (final message in command.messages)
          _messageToJson(message, mediaInputs),
      ],
      if (command.tools.isNotEmpty)
        'tools': [for (final tool in command.tools) _toolToJson(tool)],
      'tool_choice': _toolChoiceToJson(command.toolChoice),
      'parallel_tool_calls': command.parallelToolCalls,
      'add_generation_prompt': true,
      'use_jinja': true,
    };
  }

  Map<String, Object?> _messageToJson(
    LlamaMessage message,
    List<_MediaInput> mediaInputs,
  ) {
    final json = <String, Object?>{
      'role': message.role,
      'content': _contentToJson(message.content, mediaInputs),
    };

    if (message.toolCalls.isNotEmpty) {
      json['tool_calls'] = [
        for (final toolCall in message.toolCalls)
          {
            'id': toolCall.id,
            'type': 'function',
            'function': {
              'name': toolCall.name,
              'arguments': toolCall.arguments,
            },
          },
      ];
    }
    if (message.toolCallId != null) {
      json['tool_call_id'] = message.toolCallId;
    }
    if (message.name != null) {
      json['name'] = message.name;
    }

    return json;
  }

  Object _contentToJson(Object content, List<_MediaInput> mediaInputs) {
    if (content is String) {
      return content;
    }
    if (content is List<LlamaContentPart>) {
      return [
        for (final part in content) _contentPartToJson(part, mediaInputs),
      ];
    }
    throw ArgumentError.value(
      content,
      'content',
      'Unsupported message content',
    );
  }

  Map<String, String> _contentPartToJson(
    LlamaContentPart part,
    List<_MediaInput> mediaInputs,
  ) {
    if (part is LlamaTextPart) {
      return {'type': 'text', 'text': part.text};
    }
    if (part.isMedia) {
      final index = mediaInputs.length;
      mediaInputs.add(_MediaInput(index: index, part: part));
      return {'type': 'media_marker', 'text': _MediaInput.mediaMarker};
    }
    throw ArgumentError.value(part, 'part', 'Unsupported content part');
  }

  Map<String, Object?> _toolToJson(LlamaTool tool) {
    return {
      'type': 'function',
      'function': {
        'name': tool.name,
        if (tool.description.isNotEmpty) 'description': tool.description,
        'parameters': tool.parameters,
      },
    };
  }

  Object _toolChoiceToJson(LlamaToolChoice toolChoice) {
    if (toolChoice.mode == 'tool' && toolChoice.name != null) {
      return {
        'type': 'function',
        'function': {'name': toolChoice.name},
      };
    }
    return toolChoice.mode;
  }

  void disposeModel() {
    _disposeLoaded();
  }

  void close() {
    _disposeLoaded();
  }

  List<int> _tokenize(Pointer<llama_vocab> vocab, String text) {
    final bytes = utf8.encode(text);
    final textPointer = calloc<Uint8>(bytes.length + 1);

    for (var i = 0; i < bytes.length; i += 1) {
      textPointer[i] = bytes[i];
    }
    textPointer[bytes.length] = 0;

    Pointer<llama_token> tokenPointer = nullptr;
    try {
      var capacity = math.max(8, bytes.length + 8);
      tokenPointer = calloc<llama_token>(capacity);
      var count = _bindings.llama_tokenize(
        vocab,
        textPointer.cast<Char>(),
        bytes.length,
        tokenPointer,
        capacity,
        true,
        true,
      );

      if (count < 0) {
        calloc.free(tokenPointer);
        capacity = -count;
        tokenPointer = calloc<llama_token>(capacity);
        count = _bindings.llama_tokenize(
          vocab,
          textPointer.cast<Char>(),
          bytes.length,
          tokenPointer,
          capacity,
          true,
          true,
        );
      }

      if (count < 0) {
        throw NativeLlamaException(
          'Failed to tokenize prompt: llama.cpp returned $count.',
        );
      }

      return [for (var i = 0; i < count; i += 1) tokenPointer[i]];
    } finally {
      if (tokenPointer != nullptr) {
        calloc.free(tokenPointer);
      }
      calloc.free(textPointer);
    }
  }

  void _decodeTokens(Pointer<llama_context> context, List<int> tokens) {
    if (tokens.isEmpty) {
      return;
    }

    final batchSize = _bindings.llama_n_batch(context);
    if (batchSize <= 0) {
      throw NativeLlamaException('Invalid llama.cpp batch size $batchSize.');
    }

    for (var offset = 0; offset < tokens.length; offset += batchSize) {
      final chunkLength = math.min(batchSize, tokens.length - offset);
      final tokenPointer = calloc<llama_token>(chunkLength);
      try {
        for (var i = 0; i < chunkLength; i += 1) {
          tokenPointer[i] = tokens[offset + i];
        }

        final batch = _bindings.llama_batch_get_one(tokenPointer, chunkLength);
        final result = _bindings.llama_decode(context, batch);
        if (result != 0) {
          throw NativeLlamaException('llama_decode failed with code $result.');
        }
      } finally {
        calloc.free(tokenPointer);
      }
    }
  }

  Pointer<llama_sampler> _createSampler(
    Pointer<llama_vocab> vocab,
    LlamaGenerateCommand command, {
    String? grammar,
    bool grammarLazy = false,
    List<_GrammarTrigger> grammarTriggers = const [],
  }) {
    final sampler = _bindings.llama_sampler_chain_init(
      _bindings.llama_sampler_chain_default_params(),
    );
    if (sampler == nullptr) {
      throw const NativeLlamaException('Failed to create llama.cpp sampler.');
    }

    void add(Pointer<llama_sampler> child, String name) {
      if (child == nullptr) {
        _bindings.llama_sampler_free(sampler);
        throw NativeLlamaException('Failed to create llama.cpp $name sampler.');
      }
      _bindings.llama_sampler_chain_add(sampler, child);
    }

    if (grammar != null && grammar.isNotEmpty && grammarLazy) {
      add(
        _createLazyGrammarSampler(vocab, grammar, grammarTriggers),
        'lazy grammar',
      );
    } else if (grammar != null && grammar.isNotEmpty) {
      final grammarPointer = grammar.toNativeUtf8();
      final rootPointer = 'root'.toNativeUtf8();
      try {
        add(
          _bindings.llama_sampler_init_grammar(
            vocab,
            grammarPointer.cast<Char>(),
            rootPointer.cast<Char>(),
          ),
          'grammar',
        );
      } finally {
        calloc.free(rootPointer);
        calloc.free(grammarPointer);
      }
    }

    final temperature = command.temperature;
    final topP = command.topP;
    if (temperature == null && topP == null) {
      add(_bindings.llama_sampler_init_greedy(), 'greedy');
      return sampler;
    }

    if (topP != null && topP > 0 && topP < 1) {
      add(_bindings.llama_sampler_init_top_p(topP, 1), 'top-p');
    }
    if (temperature != null && temperature > 0) {
      add(_bindings.llama_sampler_init_temp(temperature), 'temperature');
      add(
        _bindings.llama_sampler_init_dist(LLAMA_DEFAULT_SEED),
        'distribution',
      );
    } else {
      add(_bindings.llama_sampler_init_greedy(), 'greedy');
    }

    return sampler;
  }

  Pointer<llama_sampler> _createLazyGrammarSampler(
    Pointer<llama_vocab> vocab,
    String grammar,
    List<_GrammarTrigger> triggers,
  ) {
    final grammarPointer = grammar.toNativeUtf8();
    final rootPointer = 'root'.toNativeUtf8();
    final stringPointers = <Pointer<Utf8>>[];
    Pointer<Pointer<Char>> triggerStrings = nullptr;
    Pointer<llama_token> triggerTokens = nullptr;

    try {
      final patternTriggers = triggers
          .where(
            (trigger) =>
                trigger.type == 'pattern' || trigger.type == 'pattern_full',
          )
          .map((trigger) => trigger.value)
          .where((value) => value.isNotEmpty)
          .toList(growable: false);
      final wordTriggers = triggers
          .where((trigger) => trigger.type == 'word')
          .map((trigger) => trigger.value)
          .where((value) => value.isNotEmpty)
          .toList(growable: false);
      final tokenTriggers = triggers
          .where((trigger) => trigger.type == 'token')
          .map((trigger) => trigger.token)
          .where((token) => token != LLAMA_TOKEN_NULL)
          .toList(growable: false);

      if (patternTriggers.isNotEmpty) {
        triggerStrings = _allocateNativeStringArray(
          patternTriggers,
          stringPointers,
        );
        triggerTokens = _allocateNativeTokenArray(tokenTriggers);
        return _bindings.llama_sampler_init_grammar_lazy_patterns(
          vocab,
          grammarPointer.cast<Char>(),
          rootPointer.cast<Char>(),
          triggerStrings,
          patternTriggers.length,
          triggerTokens,
          tokenTriggers.length,
        );
      }

      triggerStrings = _allocateNativeStringArray(wordTriggers, stringPointers);
      triggerTokens = _allocateNativeTokenArray(tokenTriggers);
      return _bindings.llama_sampler_init_grammar_lazy(
        vocab,
        grammarPointer.cast<Char>(),
        rootPointer.cast<Char>(),
        triggerStrings,
        wordTriggers.length,
        triggerTokens,
        tokenTriggers.length,
      );
    } finally {
      if (triggerTokens != nullptr) {
        calloc.free(triggerTokens);
      }
      if (triggerStrings != nullptr) {
        calloc.free(triggerStrings);
      }
      for (final pointer in stringPointers) {
        calloc.free(pointer);
      }
      calloc.free(rootPointer);
      calloc.free(grammarPointer);
    }
  }

  Pointer<Pointer<Char>> _allocateNativeStringArray(
    List<String> values,
    List<Pointer<Utf8>> stringPointers,
  ) {
    if (values.isEmpty) {
      return nullptr;
    }

    final pointer = calloc<Pointer<Char>>(values.length);
    for (var i = 0; i < values.length; i += 1) {
      final stringPointer = values[i].toNativeUtf8();
      stringPointers.add(stringPointer);
      pointer[i] = stringPointer.cast<Char>();
    }
    return pointer;
  }

  Pointer<llama_token> _allocateNativeTokenArray(List<int> tokens) {
    if (tokens.isEmpty) {
      return nullptr;
    }

    final pointer = calloc<llama_token>(tokens.length);
    for (var i = 0; i < tokens.length; i += 1) {
      pointer[i] = tokens[i];
    }
    return pointer;
  }

  // PATCHED (vendored): return raw bytes instead of a decoded String.
  // BPE models routinely split one multi-byte UTF-8 character across two
  // tokens; decoding each token in isolation threw
  // "FormatException: Unfinished UTF-8 octet sequence" and killed the whole
  // generation. The sampling loop now carries incomplete tails across tokens.
  Uint8List _tokenToBytes(Pointer<llama_vocab> vocab, int token) {
    var capacity = 32;
    Pointer<Char> buffer = calloc<Char>(capacity);

    try {
      var length = _bindings.llama_token_to_piece(
        vocab,
        token,
        buffer,
        capacity,
        0,
        false,
      );
      if (length < 0) {
        calloc.free(buffer);
        capacity = -length;
        buffer = calloc<Char>(capacity);
        length = _bindings.llama_token_to_piece(
          vocab,
          token,
          buffer,
          capacity,
          0,
          false,
        );
      }

      if (length < 0) {
        throw NativeLlamaException(
          'Failed to convert token $token to text: llama.cpp returned $length.',
        );
      }

      return Uint8List.fromList(buffer.cast<Uint8>().asTypedList(length));
    } finally {
      calloc.free(buffer);
    }
  }

  /// Index where an incomplete trailing UTF-8 sequence starts, or b.length if
  /// the buffer ends on a complete character. Only the last 3 bytes can be an
  /// incomplete tail (lead byte + up to 2 of 3 continuations).
  static int _utf8TailStart(List<int> b) {
    final n = b.length;
    for (var i = n - 1; i >= 0 && i >= n - 3; i -= 1) {
      final byte = b[i];
      if (byte & 0x80 == 0) return n; // ASCII: everything before is complete
      if (byte & 0xC0 == 0xC0) {
        // Lead byte: how many bytes does its sequence need?
        final need = (byte & 0xF8) == 0xF0
            ? 4
            : (byte & 0xF0) == 0xE0
                ? 3
                : 2;
        return (n - i) < need ? i : n;
      }
      // Continuation byte (10xxxxxx): keep scanning backwards.
    }
    return n;
  }

  String? _modelChatTemplate(Pointer<llama_model> model) {
    final template = _bindings.llama_model_chat_template(model, nullptr);
    if (template == nullptr) {
      return null;
    }
    return template.cast<Utf8>().toDartString();
  }

  void _disposeLoaded() {
    final loaded = _loaded;
    if (loaded == null) {
      return;
    }

    if (loaded.mediaContext != nullptr) {
      _wrapper.mediaFree(loaded.mediaContext);
    }
    _bindings.llama_free(loaded.context);
    _bindings.llama_model_free(loaded.model);
    _loaded = null;
  }

  static DynamicLibrary _openDynamicLibrary(LlamaCppLibraryDescriptor library) {
    try {
      return switch (library.resolution) {
        LlamaCppLibraryResolution.process => DynamicLibrary.process(),
        LlamaCppLibraryResolution.executable => DynamicLibrary.executable(),
        LlamaCppLibraryResolution.path => DynamicLibrary.open(library.path!),
        LlamaCppLibraryResolution.lookupName => DynamicLibrary.open(
          library.lookupName!,
        ),
      };
    } on Object catch (error) {
      throw NativeLlamaException('Failed to open llama.cpp library: $error');
    }
  }
}

final class NativeLlamaException implements Exception {
  const NativeLlamaException(this.message);

  final String message;

  @override
  String toString() => message;
}

final class _LoadedModel {
  const _LoadedModel({
    required this.modelPath,
    required this.model,
    required this.context,
    required this.vocab,
    required this.mmprojPath,
    required this.mediaContext,
    required this.supportsImage,
    required this.supportsAudio,
  });

  final String modelPath;
  final Pointer<llama_model> model;
  final Pointer<llama_context> context;
  final Pointer<llama_vocab> vocab;
  final String? mmprojPath;
  final Pointer<Void> mediaContext;
  final bool supportsImage;
  final bool supportsAudio;
}

final class _StopMatcher {
  _StopMatcher(List<String> stop)
    : _stop = stop.where((item) => item.isNotEmpty).toList(),
      _maxStopLength = stop
          .where((item) => item.isNotEmpty)
          .fold<int>(0, (max, item) => math.max(max, item.length));

  final List<String> _stop;
  final int _maxStopLength;
  final StringBuffer _buffer = StringBuffer();
  var _emittedLength = 0;
  var isStopped = false;

  String add(String text) {
    if (isStopped) {
      return '';
    }

    _buffer.write(text);
    final value = _buffer.toString();
    final stopIndex = _firstStopIndex(value);
    if (stopIndex != null) {
      isStopped = true;
      final delta = value.substring(_emittedLength, stopIndex);
      _emittedLength = stopIndex;
      return delta;
    }

    if (_maxStopLength == 0) {
      final delta = value.substring(_emittedLength);
      _emittedLength = value.length;
      return delta;
    }

    final safeEnd = math.max(0, value.length - _maxStopLength + 1);
    if (safeEnd <= _emittedLength) {
      return '';
    }

    final delta = value.substring(_emittedLength, safeEnd);
    _emittedLength = safeEnd;
    return delta;
  }

  String flush() {
    if (isStopped) {
      return '';
    }

    final value = _buffer.toString();
    if (_emittedLength >= value.length) {
      return '';
    }

    final delta = value.substring(_emittedLength);
    _emittedLength = value.length;
    return delta;
  }

  int? _firstStopIndex(String value) {
    int? result;
    for (final stop in _stop) {
      final index = value.indexOf(stop);
      if (index >= 0 && (result == null || index < result)) {
        result = index;
      }
    }
    return result;
  }
}

final class _ChatTemplateResult {
  const _ChatTemplateResult({
    required this.prompt,
    required this.grammar,
    required this.grammarLazy,
    required this.format,
    required this.generationPrompt,
    required this.parser,
    required this.grammarTriggers,
    required this.additionalStops,
  });

  factory _ChatTemplateResult.fromJson(Map<String, Object?> json) {
    return _ChatTemplateResult(
      prompt: json['prompt'] as String? ?? '',
      grammar: json['grammar'] as String? ?? '',
      grammarLazy: json['grammar_lazy'] as bool? ?? false,
      format: json['format'] as String? ?? 'content-only',
      generationPrompt: json['generation_prompt'] as String? ?? '',
      parser: json['parser'] as String? ?? '',
      grammarTriggers: (json['grammar_triggers'] as List? ?? const [])
          .map(_GrammarTrigger.fromJsonObject)
          .whereType<_GrammarTrigger>()
          .toList(),
      additionalStops: (json['additional_stops'] as List? ?? const [])
          .whereType<String>()
          .toList(),
    );
  }

  final String prompt;
  final String grammar;
  final bool grammarLazy;
  final String format;
  final String generationPrompt;
  final String parser;
  final List<_GrammarTrigger> grammarTriggers;
  final List<String> additionalStops;
}

final class _GrammarTrigger {
  const _GrammarTrigger({
    required this.type,
    required this.value,
    required this.token,
  });

  static _GrammarTrigger? fromJsonObject(Object? value) {
    if (value is! Map<Object?, Object?>) {
      return null;
    }
    final json = value.map((key, value) => MapEntry(key.toString(), value));
    return _GrammarTrigger(
      type: json['type'] as String? ?? '',
      value: json['value'] as String? ?? '',
      token: json['token'] as int? ?? LLAMA_TOKEN_NULL,
    );
  }

  final String type;
  final String value;
  final int token;
}

final class _MediaInput {
  const _MediaInput({required this.index, required this.part});

  static const mediaMarker = '<__media__>';

  final int index;
  final LlamaContentPart part;

  String get id => 'media_$index';

  bool get isImage {
    final value = part;
    return value is LlamaImageFilePart || value is LlamaImageBytesPart;
  }

  bool get isAudio {
    final value = part;
    return value is LlamaAudioFilePart || value is LlamaAudioBytesPart;
  }
}

final class _LibLlamaCppWrapper {
  _LibLlamaCppWrapper(this._library);

  final DynamicLibrary _library;

  _ChatTemplateResult applyChatTemplate({
    required Pointer<llama_model> model,
    required Map<String, Object?> request,
  }) {
    return _withWrapper('chat template formatting', () {
      final error = calloc<Pointer<Char>>();
      Pointer<Void> handle = nullptr;
      try {
        handle = _chatTemplatesInit(
          model,
          nullptr.cast<Char>(),
          nullptr.cast<Char>(),
          nullptr.cast<Char>(),
          error,
        );
        if (handle == nullptr) {
          throw NativeLlamaException(
            _errorMessage(error, 'Failed to initialize chat templates.'),
          );
        }

        final requestPointer = jsonEncode(request).toNativeUtf8();
        try {
          final result = _chatTemplatesApplyJson(
            handle,
            requestPointer.cast<Char>(),
            error,
          );
          if (result == nullptr) {
            throw NativeLlamaException(
              _errorMessage(error, 'Failed to apply chat template.'),
            );
          }
          final decoded = jsonDecode(_takeString(result));
          if (decoded is! Map<String, Object?>) {
            throw const NativeLlamaException(
              'Chat template wrapper returned invalid JSON.',
            );
          }
          return _ChatTemplateResult.fromJson(decoded);
        } finally {
          calloc.free(requestPointer);
        }
      } finally {
        if (handle != nullptr) {
          _chatTemplatesFree(handle);
        }
        _freeError(error);
        calloc.free(error);
      }
    });
  }

  Map<String, Object?> parseChatOutput({
    required String text,
    required String format,
    required String generationPrompt,
    required String parser,
    bool isPartial = false,
  }) {
    return _withWrapper('chat output parsing', () {
      final request = {
        'text': text,
        'format': format,
        'generation_prompt': generationPrompt,
        'parser': parser,
        'parse_tool_calls': true,
        'is_partial': isPartial,
      };
      final requestPointer = jsonEncode(request).toNativeUtf8();
      final error = calloc<Pointer<Char>>();
      try {
        final result = _chatParseJson(requestPointer.cast<Char>(), error);
        if (result == nullptr) {
          throw NativeLlamaException(
            _errorMessage(error, 'Failed to parse chat output.'),
          );
        }
        final decoded = jsonDecode(_takeString(result));
        if (decoded is! Map<String, Object?>) {
          throw const NativeLlamaException(
            'Chat parser wrapper returned invalid JSON.',
          );
        }
        return decoded;
      } finally {
        _freeError(error);
        calloc.free(error);
        calloc.free(requestPointer);
      }
    });
  }

  Pointer<Void> mediaInit({
    required String mmprojPath,
    required Pointer<llama_model> model,
    required bool useGpu,
    required int? imageMinTokens,
    required int? imageMaxTokens,
  }) {
    return _withWrapper('multimodal projector loading', () {
      final pathPointer = mmprojPath.toNativeUtf8();
      final options = <String, Object?>{
        'use_gpu': useGpu,
        'media_marker': _MediaInput.mediaMarker,
        'warmup': false,
      };
      if (imageMinTokens != null) {
        options['image_min_tokens'] = imageMinTokens;
      }
      if (imageMaxTokens != null) {
        options['image_max_tokens'] = imageMaxTokens;
      }
      final optionsPointer = jsonEncode(options).toNativeUtf8();
      final error = calloc<Pointer<Char>>();
      try {
        final result = _mediaInit(
          pathPointer.cast<Char>(),
          model,
          optionsPointer.cast<Char>(),
          error,
        );
        if (result == nullptr) {
          throw NativeLlamaException(
            _errorMessage(error, 'Failed to load multimodal projector.'),
          );
        }
        return result;
      } finally {
        _freeError(error);
        calloc.free(error);
        calloc.free(optionsPointer);
        calloc.free(pathPointer);
      }
    });
  }

  void mediaFree(Pointer<Void> mediaContext) {
    if (mediaContext == nullptr) {
      return;
    }
    _withWrapper('multimodal cleanup', () {
      _mediaFree(mediaContext);
    });
  }

  bool mediaSupportsVision(Pointer<Void> mediaContext) {
    return _withWrapper('multimodal capability detection', () {
      return _mediaSupportsVision(mediaContext);
    });
  }

  bool mediaSupportsAudio(Pointer<Void> mediaContext) {
    return _withWrapper('multimodal capability detection', () {
      return _mediaSupportsAudio(mediaContext);
    });
  }

  Pointer<Void> mediaBlob(Pointer<Void> mediaContext, _MediaInput input) {
    final part = input.part;
    if (part is LlamaImageFilePart) {
      return _mediaBlobFromFile(mediaContext, part.path, input.id);
    }
    if (part is LlamaAudioFilePart) {
      return _mediaBlobFromFile(mediaContext, part.path, input.id);
    }
    if (part is LlamaImageBytesPart) {
      return _mediaBlobFromBytes(mediaContext, part.bytes, input.id);
    }
    if (part is LlamaAudioBytesPart) {
      return _mediaBlobFromBytes(mediaContext, part.bytes, input.id);
    }
    throw ArgumentError.value(part, 'part', 'Unsupported media part');
  }

  void mediaBlobFree(Pointer<Void> blob) {
    if (blob == nullptr) {
      return;
    }
    _withWrapper('media cleanup', () {
      _mediaBlobFree(blob);
    });
  }

  int mediaEvalPrompt({
    required Pointer<Void> mediaContext,
    required Pointer<llama_context> llamaContext,
    required String prompt,
    required List<Pointer<Void>> blobs,
    required int nBatch,
  }) {
    return _withWrapper('multimodal prompt evaluation', () {
      final promptPointer = prompt.toNativeUtf8();
      final blobArray = calloc<Pointer<Void>>(blobs.length);
      final newPast = calloc<Int32>();
      final error = calloc<Pointer<Char>>();
      try {
        for (var i = 0; i < blobs.length; i += 1) {
          blobArray[i] = blobs[i];
        }
        final result = _mediaEvalPrompt(
          mediaContext,
          llamaContext,
          promptPointer.cast<Char>(),
          blobArray,
          blobs.length,
          0,
          0,
          nBatch,
          true,
          true,
          true,
          newPast,
          error,
        );
        if (result != 0) {
          throw NativeLlamaException(
            _errorMessage(
              error,
              'Failed to evaluate multimodal prompt: mtmd returned $result.',
            ),
          );
        }
        return newPast.value;
      } finally {
        _freeError(error);
        calloc.free(error);
        calloc.free(newPast);
        calloc.free(blobArray);
        calloc.free(promptPointer);
      }
    });
  }

  Pointer<Void> _mediaBlobFromFile(
    Pointer<Void> mediaContext,
    String path,
    String id,
  ) {
    return _withWrapper('media file decoding', () {
      final pathPointer = path.toNativeUtf8();
      final idPointer = id.toNativeUtf8();
      final error = calloc<Pointer<Char>>();
      try {
        final result = _mediaBlobFromFileNative(
          mediaContext,
          pathPointer.cast<Char>(),
          idPointer.cast<Char>(),
          error,
        );
        if (result == nullptr) {
          throw NativeLlamaException(
            _errorMessage(error, 'Failed to decode media file.'),
          );
        }
        return result;
      } finally {
        _freeError(error);
        calloc.free(error);
        calloc.free(idPointer);
        calloc.free(pathPointer);
      }
    });
  }

  Pointer<Void> _mediaBlobFromBytes(
    Pointer<Void> mediaContext,
    List<int> bytes,
    String id,
  ) {
    return _withWrapper('media byte decoding', () {
      final bytesPointer = calloc<Uint8>(bytes.length);
      final idPointer = id.toNativeUtf8();
      final error = calloc<Pointer<Char>>();
      try {
        for (var i = 0; i < bytes.length; i += 1) {
          bytesPointer[i] = bytes[i];
        }
        final result = _mediaBlobFromEncodedBytes(
          mediaContext,
          bytesPointer,
          bytes.length,
          idPointer.cast<Char>(),
          error,
        );
        if (result == nullptr) {
          throw NativeLlamaException(
            _errorMessage(error, 'Failed to decode encoded media bytes.'),
          );
        }
        return result;
      } finally {
        _freeError(error);
        calloc.free(error);
        calloc.free(idPointer);
        calloc.free(bytesPointer);
      }
    });
  }

  T _withWrapper<T>(String feature, T Function() body) {
    try {
      return body();
    } on ArgumentError catch (error) {
      throw NativeLlamaException(
        'The loaded llama.cpp library does not include lib_llama_cpp native '
        'wrapper support for $feature: $error',
      );
    }
  }

  String _takeString(Pointer<Char> pointer) {
    try {
      return pointer.cast<Utf8>().toDartString();
    } finally {
      _stringFree(pointer);
    }
  }

  String _errorMessage(Pointer<Pointer<Char>> errorPointer, String fallback) {
    final pointer = errorPointer.value;
    if (pointer == nullptr) {
      return fallback;
    }

    try {
      final raw = pointer.cast<Utf8>().toDartString();
      final decoded = jsonDecode(raw);
      if (decoded is Map && decoded['message'] is String) {
        return decoded['message'] as String;
      }
      return raw;
    } finally {
      _stringFree(pointer);
      errorPointer.value = nullptr;
    }
  }

  void _freeError(Pointer<Pointer<Char>> errorPointer) {
    if (errorPointer.value != nullptr) {
      _stringFree(errorPointer.value);
      errorPointer.value = nullptr;
    }
  }

  late final _StringFreeDart _stringFree = _library
      .lookupFunction<_StringFreeNative, _StringFreeDart>(
        'lib_llama_cpp_string_free',
      );

  late final _ChatTemplatesInitDart _chatTemplatesInit = _library
      .lookupFunction<_ChatTemplatesInitNative, _ChatTemplatesInitDart>(
        'lib_llama_cpp_chat_templates_init',
      );

  late final _ChatTemplatesFreeDart _chatTemplatesFree = _library
      .lookupFunction<_ChatTemplatesFreeNative, _ChatTemplatesFreeDart>(
        'lib_llama_cpp_chat_templates_free',
      );

  late final _ChatTemplatesApplyJsonDart _chatTemplatesApplyJson = _library
      .lookupFunction<
        _ChatTemplatesApplyJsonNative,
        _ChatTemplatesApplyJsonDart
      >('lib_llama_cpp_chat_templates_apply_json');

  late final _ChatParseJsonDart _chatParseJson = _library
      .lookupFunction<_ChatParseJsonNative, _ChatParseJsonDart>(
        'lib_llama_cpp_chat_parse_json',
      );

  late final _MediaInitDart _mediaInit = _library
      .lookupFunction<_MediaInitNative, _MediaInitDart>(
        'lib_llama_cpp_media_init',
      );

  late final _MediaFreeDart _mediaFree = _library
      .lookupFunction<_MediaFreeNative, _MediaFreeDart>(
        'lib_llama_cpp_media_free',
      );

  late final _MediaSupportsDart _mediaSupportsVision = _library
      .lookupFunction<_MediaSupportsNative, _MediaSupportsDart>(
        'lib_llama_cpp_media_supports_vision',
      );

  late final _MediaSupportsDart _mediaSupportsAudio = _library
      .lookupFunction<_MediaSupportsNative, _MediaSupportsDart>(
        'lib_llama_cpp_media_supports_audio',
      );

  late final _MediaBlobFromFileDart _mediaBlobFromFileNative = _library
      .lookupFunction<_MediaBlobFromFileNative, _MediaBlobFromFileDart>(
        'lib_llama_cpp_media_blob_from_file',
      );

  late final _MediaBlobFromBytesDart _mediaBlobFromEncodedBytes = _library
      .lookupFunction<_MediaBlobFromBytesNative, _MediaBlobFromBytesDart>(
        'lib_llama_cpp_media_blob_from_encoded_bytes',
      );

  late final _MediaBlobFreeDart _mediaBlobFree = _library
      .lookupFunction<_MediaBlobFreeNative, _MediaBlobFreeDart>(
        'lib_llama_cpp_media_blob_free',
      );

  late final _MediaEvalPromptDart _mediaEvalPrompt = _library
      .lookupFunction<_MediaEvalPromptNative, _MediaEvalPromptDart>(
        'lib_llama_cpp_media_eval_prompt',
      );
}

typedef _StringFreeNative = Void Function(Pointer<Char>);
typedef _StringFreeDart = void Function(Pointer<Char>);

typedef _ChatTemplatesInitNative =
    Pointer<Void> Function(
      Pointer<llama_model>,
      Pointer<Char>,
      Pointer<Char>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );
typedef _ChatTemplatesInitDart =
    Pointer<Void> Function(
      Pointer<llama_model>,
      Pointer<Char>,
      Pointer<Char>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );

typedef _ChatTemplatesFreeNative = Void Function(Pointer<Void>);
typedef _ChatTemplatesFreeDart = void Function(Pointer<Void>);

typedef _ChatTemplatesApplyJsonNative =
    Pointer<Char> Function(
      Pointer<Void>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );
typedef _ChatTemplatesApplyJsonDart =
    Pointer<Char> Function(
      Pointer<Void>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );

typedef _ChatParseJsonNative =
    Pointer<Char> Function(Pointer<Char>, Pointer<Pointer<Char>>);
typedef _ChatParseJsonDart =
    Pointer<Char> Function(Pointer<Char>, Pointer<Pointer<Char>>);

typedef _MediaInitNative =
    Pointer<Void> Function(
      Pointer<Char>,
      Pointer<llama_model>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );
typedef _MediaInitDart =
    Pointer<Void> Function(
      Pointer<Char>,
      Pointer<llama_model>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );

typedef _MediaFreeNative = Void Function(Pointer<Void>);
typedef _MediaFreeDart = void Function(Pointer<Void>);

typedef _MediaSupportsNative = Bool Function(Pointer<Void>);
typedef _MediaSupportsDart = bool Function(Pointer<Void>);

typedef _MediaBlobFromFileNative =
    Pointer<Void> Function(
      Pointer<Void>,
      Pointer<Char>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );
typedef _MediaBlobFromFileDart =
    Pointer<Void> Function(
      Pointer<Void>,
      Pointer<Char>,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );

typedef _MediaBlobFromBytesNative =
    Pointer<Void> Function(
      Pointer<Void>,
      Pointer<Uint8>,
      Size,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );
typedef _MediaBlobFromBytesDart =
    Pointer<Void> Function(
      Pointer<Void>,
      Pointer<Uint8>,
      int,
      Pointer<Char>,
      Pointer<Pointer<Char>>,
    );

typedef _MediaBlobFreeNative = Void Function(Pointer<Void>);
typedef _MediaBlobFreeDart = void Function(Pointer<Void>);

typedef _MediaEvalPromptNative =
    Int32 Function(
      Pointer<Void>,
      Pointer<llama_context>,
      Pointer<Char>,
      Pointer<Pointer<Void>>,
      Size,
      Int32,
      Int32,
      Int32,
      Bool,
      Bool,
      Bool,
      Pointer<Int32>,
      Pointer<Pointer<Char>>,
    );
typedef _MediaEvalPromptDart =
    int Function(
      Pointer<Void>,
      Pointer<llama_context>,
      Pointer<Char>,
      Pointer<Pointer<Void>>,
      int,
      int,
      int,
      int,
      bool,
      bool,
      bool,
      Pointer<Int32>,
      Pointer<Pointer<Char>>,
    );
