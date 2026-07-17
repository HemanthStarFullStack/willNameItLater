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
  if (s.isEmpty && raw.contains('<think>')) return '💭 thinking…';
  return s;
}

class Llm {
  final _commands = StreamController<LlamaCommand>();
  StreamSubscription<LlamaResponse>? _sub;

  // One generation at a time; the pipeline is sequential anyway.
  Future<void> _busy = Future.value();
  Completer<String>? _turn;
  final _buf = StringBuffer();
  void Function(String partial)? _onPartial;

  /// Appended to system prompts for Qwen3-style hybrid thinkers (documented
  /// soft switch). R1-style models can't be switched off; stripThink handles
  /// their output instead.
  String noThinkSuffix = '';

  bool get ready => _sub != null;

  Future<void> load(String modelPath, {int contextSize = 4096}) async {
    await _sub?.cancel();
    _sub = const LibLlamaCpp().transform(_commands.stream).listen((r) {
      switch (r) {
        case LlamaTokenResponse(:final text):
          _buf.write(text);
          _onPartial?.call(stripThink(_buf.toString()));
        case LlamaDoneResponse():
          _turn?.complete(_buf.toString());
          _turn = null;
        case LlamaErrorResponse(:final message):
          _turn?.completeError(StateError(message));
          _turn = null;
        case LlamaReadyResponse() ||
              LlamaStateChangedResponse() ||
              LlamaToolCallResponse():
          break;
      }
    });
    _commands.add(
      LlamaLoadModelCommand(modelPath: modelPath, contextSize: contextSize),
    );
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
          if (system != null)
            LlamaMessage(role: 'system', content: '$system$noThinkSuffix'),
          LlamaMessage(role: 'user', content: prompt),
        ],
        temperature: temperature,
        topP: 0.9,
        maxTokens: maxTokens,
      ));
      final raw = await done.future;
      _onPartial = null;
      return stripThink(raw).trim();
    });
    // Keep the queue alive even when a call fails.
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
