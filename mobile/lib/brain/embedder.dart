/// On-device text embeddings — the mobile stand-in for llm.embed()
/// (nomic-embed via Ollama on the desktop prototype). MiniLM-L6-v2 over ONNX
/// Runtime via fonnx: built-in WordPiece tokenizer, 384-dim vectors, runs on
/// CPU/NNAPI. Note for calibration: cosine thresholds tuned on nomic-embed
/// (RAG gate 0.52 etc.) sit at slightly different values on MiniLM — see
/// Config in pipeline.dart, re-measure there before trusting the gates.
library;

import 'package:fonnx/models/minilml6v2/mini_lm_l6_v2.dart';

class Embedder {
  MiniLmL6V2? _model;

  bool get ready => _model != null;

  void load(String modelPath) {
    _model = MiniLmL6V2.load(modelPath);
  }

  /// Embed one text. Long inputs are chunked by the tokenizer (256-token cap);
  /// facts and queries are short, so the first chunk is the text.
  Future<List<double>> embed(String text) async {
    final model = _model;
    if (model == null) throw StateError('Embedder not loaded');
    final chunks = MiniLmL6V2.tokenizer.tokenize(text.isEmpty ? ' ' : text);
    final vec = await model.getEmbeddingAsVector(chunks.first.tokens);
    return List<double>.from(vec);
  }
}
