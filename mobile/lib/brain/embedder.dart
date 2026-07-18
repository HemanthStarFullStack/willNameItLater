/// On-device text embeddings — EmbeddingGemma-300M GGUF through the same
/// llama.cpp engine that runs chat (vendored LlamaEmbedCommand), replacing the
/// 2020-era MiniLM/ONNX stack entirely. Vectors are pooled + L2-normalised by
/// the engine (pooling type comes from the GGUF metadata), 768-dim.
///
/// Runs as its own persistent inference session (one isolate per model), so
/// chat generation and retrieval never contend for a context.
library;

import 'dart:async';

import 'package:lib_llama_cpp/lib_llama_cpp.dart';

class Embedder {
  /// Explicit native library path — used by the desktop test harness; null on
  /// Android (the plugin resolves its bundled .so).
  final String? libraryPath;

  Embedder({this.libraryPath});

  final _commands = StreamController<LlamaCommand>();
  StreamSubscription<LlamaResponse>? _sub;

  Future<void> _busy = Future.value();
  Completer<List<List<double>>>? _turn;
  List<List<double>>? _pending;

  bool get ready => _sub != null;

  /// Loads the embedding model; completes only once the engine confirms the
  /// load (or errors), so failures surface here instead of hanging callers.
  Future<void> load(String modelPath) async {
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
        case LlamaEmbedResponse(:final embeddings):
          _pending = embeddings;
        case LlamaDoneResponse():
          final got = _pending;
          _pending = null;
          if (got != null) _turn?.complete(got);
          _turn = null;
        case LlamaErrorResponse(:final message):
          if (!loaded.isCompleted) {
            loaded.completeError(StateError(message));
          }
          _turn?.completeError(StateError(message));
          _turn = null;
        case LlamaStateChangedResponse(:final state):
          if (state.isModelLoaded && !loaded.isCompleted) loaded.complete();
        case LlamaTokenResponse() ||
              LlamaReadyResponse() ||
              LlamaToolCallResponse():
          break;
      }
    });
    // Small context: facts and queries are short; keeps the session light
    // alongside the chat model.
    _commands.add(
      LlamaLoadModelCommand(modelPath: modelPath, contextSize: 1024),
    );
    await loaded.future;
  }

  Future<List<List<double>>> embedBatch(List<String> texts) {
    final run = _busy.then((_) async {
      final done = Completer<List<List<double>>>();
      _turn = done;
      _commands.add(LlamaEmbedCommand(texts: texts));
      return done.future;
    });
    _busy = run.then((_) {}, onError: (_) {});
    return run;
  }

  Future<List<double>> embed(String text) async =>
      (await embedBatch([text.isEmpty ? ' ' : text])).first;
}
