/// Thin wrapper around llama.cpp: chat + JSON judge — the mobile port of
/// llm.py (Ollama). One persistent inference session: the OpenAI-style client
/// in lib_llama_cpp reloads the model on EVERY request (LoadModel…Dispose per
/// call), which would make the multi-call pipeline reload a 2 GB model several
/// times per turn. Driving the command stream directly loads the model once
/// and reuses it, and always goes through the chat-template path
/// (GenerateMessages) — bare-prompt generation produced gibberish.
library;

import 'dart:async';
import 'dart:convert';

import 'package:lib_llama_cpp/lib_llama_cpp.dart';

/// Hide chain-of-thought. Reasoning models (Qwen3, DeepSeek R1) emit
/// `<think>…</think>` before the answer; recomputing from the full buffer
/// handles tags split across stream chunks, and an unclosed <think> (still
/// reasoning) shows a hint instead of leaking half a thought.
String stripThink(String raw) {
  var s = raw.replaceAll(RegExp(r'<think>[\s\S]*?</think>'), '').trim();
  final open = s.indexOf('<think>');
  if (open >= 0) s = s.substring(0, open).trim();
  // Template-forced thinkers (Qwen3.5): <think> is opened in the PROMPT, so
  // the output carries only the closing tag — keep what follows it.
  final close = s.lastIndexOf('</think>');
  if (close >= 0) s = s.substring(close + '</think>'.length).trim();
  if (s.isEmpty && raw.contains('think>')) return '💭 thinking…';
  return s;
}

class Llm {
  /// Explicit native library path. Unused on Android (the plugin resolves its
  /// bundled .so); the desktop test harness points this at the Windows CPU
  /// DLL so the full brain runs on the laptop.
  final String? libraryPath;

  Llm({this.libraryPath});

  final _commands = StreamController<LlamaCommand>();
  StreamSubscription<LlamaResponse>? _sub;

  // One generation at a time; the pipeline is sequential anyway.
  Future<void> _busy = Future.value();
  Completer<String>? _turn;
  final _buf = StringBuffer();
  final _toolCalls = <LlamaToolCall>[];
  void Function(String partial)? _onPartial;

  /// Template/sampler-level thinking control. Qwen3.5-era hybrid reasoners
  /// ignore the old /no_think text switch, so the vendored engine closes the
  /// template's think block and bans the <think> token instead.
  /// null = model default; false = never think (right for a phone pipeline).
  bool? enableThinking;

  bool get ready => _sub != null;

  /// Loads the model and completes only once the engine confirms it (or
  /// errors). Without this, a load-time failure arrived before any chat was
  /// pending and was silently dropped — the next chat() then hung forever.
  Future<void> load(String modelPath, {int contextSize = 4096}) async {
    await _sub?.cancel();
    final lib = libraryPath;
    final loaded = Completer<void>();
    _sub = const LibLlamaCpp()
        .transform(
      _commands.stream,
      libraryRequest: lib == null
          ? const LlamaCppLibraryRequest()
          : LlamaCppLibraryRequest(preferredPath: lib),
    )
        .listen((r) {
      switch (r) {
        case LlamaTokenResponse(:final text):
          _buf.write(text);
          _onPartial?.call(stripThink(_buf.toString()));
        case LlamaDoneResponse():
          _turn?.complete(_buf.toString());
          _turn = null;
        case LlamaErrorResponse(:final message):
          if (!loaded.isCompleted) {
            loaded.completeError(StateError(message));
          }
          _turn?.completeError(StateError(message));
          _turn = null;
        case LlamaStateChangedResponse(:final state):
          if (state.isModelLoaded && !loaded.isCompleted) loaded.complete();
        case LlamaToolCallResponse(:final toolCall):
          _toolCalls.add(toolCall);
        case LlamaReadyResponse() || LlamaEmbedResponse():
          break;
      }
    });
    _commands.add(
      LlamaLoadModelCommand(modelPath: modelPath, contextSize: contextSize),
    );
    await loaded.future;
  }

  /// One chat call. `onPartial` streams the visible (think-stripped) text —
  /// used only for the final user-facing answer; judge calls skip it.
  Future<String> chat(
    String prompt, {
    String? system,
    double temperature = 0.2,
    int maxTokens = 512,
    void Function(String partial)? onPartial,
  }) {
    final run = _busy.then((_) async {
      _buf.clear();
      _onPartial = onPartial;
      final done = Completer<String>();
      _turn = done;
      _commands.add(LlamaGenerateMessagesCommand(
        messages: [
          if (system != null) LlamaMessage(role: 'system', content: system),
          LlamaMessage(role: 'user', content: prompt),
        ],
        temperature: temperature,
        topP: 0.9,
        maxTokens: maxTokens,
        enableThinking: enableThinking,
      ));
      final raw = await done.future;
      _onPartial = null;
      return stripThink(raw).trim();
    });
    // Keep the queue alive even when a call fails.
    _busy = run.then((_) {}, onError: (_) {});
    return run;
  }

  /// One call with native tool definitions — the chat template declares the
  /// tools in the model's trained format. Returns the raw text AND any calls
  /// the engine parser recognized (this pin predates FunctionGemma's syntax,
  /// so the caller usually parses the raw text itself).
  Future<(String, List<LlamaToolCall>)> chatWithTools(
    String prompt, {
    required List<LlamaTool> tools,
    String? system,
    double temperature = 0,
    int maxTokens = 128,
    List<String> stop = const [],
  }) {
    final run = _busy.then((_) async {
      _buf.clear();
      _toolCalls.clear();
      final done = Completer<String>();
      _turn = done;
      _commands.add(LlamaGenerateMessagesCommand(
        messages: [
          if (system != null) LlamaMessage(role: 'system', content: system),
          LlamaMessage(role: 'user', content: prompt),
        ],
        tools: tools,
        temperature: temperature,
        maxTokens: maxTokens,
        stop: stop,
      ));
      final raw = await done.future;
      return (raw, List<LlamaToolCall>.unmodifiable(_toolCalls));
    });
    _busy = run.then((_) {}, onError: (_) {});
    return run;
  }

  /// Ask for strict JSON and parse it, with the same best-effort fallback as
  /// llm.py (no grammar-constrained decoding here, so the regex rescue does
  /// more work — prompts already demand "Output only JSON").
  Future<Map<String, dynamic>> chatJson(String prompt, {String? system}) async {
    String txt;
    try {
      txt = await chat(prompt, system: system, temperature: 0, maxTokens: 256);
    } catch (_) {
      return {};
    }
    try {
      final v = jsonDecodeSafe(txt);
      if (v != null) return v;
    } catch (_) {}
    return {};
  }
}

Map<String, dynamic>? jsonDecodeSafe(String txt) {
  Map<String, dynamic>? tryParse(String s) {
    try {
      final v = json.decode(s);
      return v is Map<String, dynamic> ? v : null;
    } catch (_) {
      return null;
    }
  }

  final direct = tryParse(txt);
  if (direct != null) return direct;
  final m = RegExp(r'\{[\s\S]*\}').firstMatch(txt);
  return m == null ? null : tryParse(m.group(0)!);
}
