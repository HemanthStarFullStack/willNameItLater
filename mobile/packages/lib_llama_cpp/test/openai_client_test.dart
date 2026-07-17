import 'package:flutter_test/flutter_test.dart';
import 'package:lib_llama_cpp/lib_llama_cpp.dart';
import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

final class FakeLibLlamaCppPlatform extends LibLlamaCppPlatform
    with MockPlatformInterfaceMixin {
  var resolveCount = 0;

  @override
  Future<LlamaCppLibraryDescriptor> resolveLibrary({
    LlamaCppLibraryRequest request = const LlamaCppLibraryRequest(),
  }) async {
    resolveCount += 1;
    return const LlamaCppLibraryDescriptor(
      resolution: LlamaCppLibraryResolution.lookupName,
      lookupName: 'libtest_llama.so',
      capabilities: {LlamaCppLibraryCapability.cpu},
    );
  }
}

final class ScriptedLlamaEngine implements LlamaEngine {
  ScriptedLlamaEngine(this.responses);

  final List<LlamaResponse> responses;
  final commands = <LlamaCommand>[];

  @override
  Stream<LlamaResponse> transform(
    Stream<LlamaCommand> input, {
    LlamaState initialState = const LlamaState.empty(),
    LlamaCppLibraryRequest libraryRequest = const LlamaCppLibraryRequest(),
  }) async* {
    commands.addAll(await input.toList());
    yield* Stream<LlamaResponse>.fromIterable(responses);
  }
}

void main() {
  group('LlamaOpenAIClient model registry', () {
    test('unknown model fails with model_not_found', () async {
      final platform = FakeLibLlamaCppPlatform();
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: LibLlamaCpp(platform: platform),
      );

      await expectLater(
        client.responses.create(model: 'missing', input: 'Hello'),
        throwsA(
          isA<LlamaOpenAIException>()
              .having((error) => error.code, 'code', 'model_not_found')
              .having((error) => error.param, 'param', 'model'),
        ),
      );
      expect(platform.resolveCount, 0);
    });
  });

  group('responses.create', () {
    test('maps streamed tokens into a completed response', () async {
      final engine = ScriptedLlamaEngine([
        const LlamaReadyResponse(
          library: LlamaCppLibraryDescriptor(
            resolution: LlamaCppLibraryResolution.lookupName,
            lookupName: 'libtest_llama.so',
            capabilities: {LlamaCppLibraryCapability.cpu},
          ),
        ),
        const LlamaStateChangedResponse(
          state: LlamaState(
            modelPath: '/models/local.gguf',
            isModelLoaded: true,
          ),
        ),
        const LlamaTokenResponse(text: 'Hello', index: 0),
        const LlamaTokenResponse(text: ' world', index: 1),
        const LlamaDoneResponse(),
      ]);
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: engine,
      );

      final response = await client.responses.create(
        model: 'local',
        input: 'Write one sentence.',
        maxOutputTokens: 16,
        temperature: 0.7,
        topP: 0.9,
        stop: const ['</s>'],
      );

      expect(response.status, 'completed');
      expect(response.outputText, 'Hello world');
      expect(
        engine.commands,
        contains(
          const LlamaGenerateCommand(
            prompt: 'Write one sentence.',
            maxTokens: 16,
            temperature: 0.7,
            topP: 0.9,
            stop: ['</s>'],
          ),
        ),
      );
    });

    test('store true fails explicitly', () async {
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: LibLlamaCpp(platform: FakeLibLlamaCppPlatform()),
      );

      await expectLater(
        client.responses.create(model: 'local', input: 'Hello', store: true),
        throwsA(
          isA<LlamaOpenAIException>()
              .having((error) => error.code, 'code', 'unsupported_parameter')
              .having((error) => error.param, 'param', 'store'),
        ),
      );
    });

    test('maps engine errors to OpenAI exceptions', () async {
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: ScriptedLlamaEngine([
          const LlamaErrorResponse(message: 'runtime failed'),
        ]),
      );

      await expectLater(
        client.responses.create(model: 'local', input: 'Hello'),
        throwsA(
          isA<LlamaOpenAIException>()
              .having((error) => error.code, 'code', 'generation_failed')
              .having((error) => error.message, 'message', 'runtime failed')
              .having((error) => error.type, 'type', 'server_error'),
        ),
      );
    });

    test('unsupported input shape fails explicitly', () async {
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: LibLlamaCpp(platform: FakeLibLlamaCppPlatform()),
      );

      await expectLater(
        client.responses.create(model: 'local', input: const ['Hello']),
        throwsA(
          isA<LlamaOpenAIException>()
              .having((error) => error.code, 'code', 'unsupported_parameter')
              .having((error) => error.param, 'param', 'input'),
        ),
      );
    });

    test('typed input items are accepted as response input', () async {
      final engine = ScriptedLlamaEngine([
        const LlamaTokenResponse(text: 'Hello back', index: 0),
      ]);
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: engine,
      );

      final response = await client.responses.create(
        model: 'local',
        input: const [LlamaResponseInputItem(role: 'user', content: 'Hello')],
      );

      expect(response.outputText, 'Hello back');
      expect(
        engine.commands,
        contains(const LlamaGenerateCommand(prompt: 'Hello')),
      );
    });
  });

  group('responses.stream', () {
    test('emits token deltas and a completed event', () async {
      final engine = ScriptedLlamaEngine([
        const LlamaTokenResponse(text: 'Hi', index: 0),
        const LlamaTokenResponse(text: ' there', index: 1),
        const LlamaDoneResponse(),
      ]);
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: engine,
      );

      final events = await client.responses
          .stream(model: 'local', input: 'Hello', maxOutputTokens: 4)
          .toList();

      expect(events.first.type, 'response.created');
      expect(
        events.whereType<LlamaResponseOutputTextDelta>().map(
          (event) => event.delta,
        ),
        ['Hi', ' there'],
      );
      expect(
        events.whereType<LlamaResponseCompleted>().single.response.outputText,
        'Hi there',
      );
    });

    test('emits failed event when the engine reports an error', () async {
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: ScriptedLlamaEngine([
          const LlamaErrorResponse(message: 'runtime failed'),
        ]),
      );

      final events = await client.responses
          .stream(model: 'local', input: 'Hello', maxOutputTokens: 4)
          .toList();

      expect(events.first.type, 'response.created');
      expect(events.last, isA<LlamaResponseFailed>());
      expect(
        (events.last as LlamaResponseFailed).error.code,
        'generation_failed',
      );
    });
  });

  group('chat.completions.create', () {
    test('maps chat messages through responses', () async {
      final engine = ScriptedLlamaEngine([
        const LlamaTokenResponse(text: 'Hello from llama.cpp', index: 0),
      ]);
      final client = LlamaOpenAIClient(
        models: {
          'local': const LlamaModelConfig(modelPath: '/models/local.gguf'),
        },
        engine: engine,
      );

      final completion = await client.chat.completions.create(
        model: 'local',
        messages: [const LlamaChatMessage(role: 'user', content: 'Hello')],
        maxTokens: 4,
      );

      expect(completion.choices.single.message.role, 'assistant');
      expect(completion.choices.single.message.content, 'Hello from llama.cpp');
      expect(
        engine.commands,
        contains(const LlamaGenerateCommand(prompt: 'Hello', maxTokens: 4)),
      );
    });
  });
}
