/// The 'parent' router: decides which expert memory a note or question
/// belongs to — the port of router.py.
library;

import 'dart:math' as math;

import 'embedder.dart';
import 'llm.dart';
import 'store.dart';

/// Seeded "life bucket" expert memories (config.py CATEGORIES).
const kCategories = [
  'Health', 'Money', 'Work', 'Learning', 'Home', 'Personal', 'Travel',
];

class Router {
  final Llm llm;
  final Embedder embedder;
  final MemoryStore store;

  Router(this.llm, this.embedder, this.store);

  /// Pick the best category for a new note (may propose a new one).
  Future<String> routeIngest(String text) async {
    final cats = kCategories.join(', ');
    final out = await llm.chatJson(
      'Classify this note into exactly ONE category from: [$cats]. '
      'If none truly fit, propose a short new category name.\n\n'
      'Note: "$text"\n\n'
      'Respond JSON: {"category": "<name>"}',
      system: 'You are a precise classifier. Output only JSON.',
    );
    var cat = ((out['category'] as String?) ?? 'Personal').trim();
    if (cat.length > 30) cat = cat.substring(0, 30);
    return cat.isEmpty ? 'Personal' : cat;
  }

  /// Unit-normalised mean embedding per existing expert memory.
  Map<String, List<double>> _centroids() {
    final byCat = <String, List<List<double>>>{};
    for (final x in store.facts) {
      byCat.putIfAbsent(x.category, () => []).add(x.embedding);
    }
    final out = <String, List<double>>{};
    byCat.forEach((cat, embs) {
      final dim = embs.first.length;
      final c = List<double>.filled(dim, 0);
      for (final e in embs) {
        for (var i = 0; i < dim; i++) {
          c[i] += e[i];
        }
      }
      var norm = 0.0;
      for (var i = 0; i < dim; i++) {
        c[i] /= embs.length;
        norm += c[i] * c[i];
      }
      norm = math.sqrt(norm);
      if (norm == 0) norm = 1;
      out[cat] = [for (final v in c) v / norm];
    });
    return out;
  }

  Future<List<double>> _embedUnit(String text) async {
    final q = await embedder.embed(text);
    var norm = 0.0;
    for (final v in q) {
      norm += v * v;
    }
    norm = math.sqrt(norm);
    if (norm == 0) norm = 1;
    return [for (final v in q) v / norm];
  }

  /// Pick which expert memory should answer; null means search everything.
  /// Centroid cosine instead of an LLM call — deterministic and ~free; only
  /// routes when one expert clearly wins, unsure -> ALL (always safe, the
  /// caller falls back to a global search anyway).
  Future<String?> routeQuery(String query) async {
    final cents = _centroids();
    if (cents.length < 2) return null;
    final q = await _embedUnit(query);
    final scored = cents.entries
        .map((e) => (score: _dotList(q, e.value), cat: e.key))
        .toList()
      ..sort((a, b) => b.score.compareTo(a.score));
    final best = scored[0], runnerUp = scored[1];
    if (best.score >= 0.45 && best.score - runnerUp.score >= 0.05) {
      return best.cat;
    }
    return null;
  }

  /// A brand-new bucket must earn its existence: if the note sits close to an
  /// existing expert's centroid, file it there instead of fragmenting the
  /// memories ("Fitness" appearing next to "Health").
  Future<String> snapToExisting(String text, String proposed) async {
    final cents = _centroids();
    if (cents.isEmpty) return proposed;
    final q = await _embedUnit(text);
    var bestCat = proposed;
    var best = 0.0;
    cents.forEach((cat, c) {
      final s = _dotList(q, c);
      if (s > best) {
        best = s;
        bestCat = cat;
      }
    });
    return best >= 0.45 ? bestCat : proposed;
  }
}

double _dotList(List<double> a, List<double> b) {
  var s = 0.0;
  for (var i = 0; i < a.length && i < b.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}
