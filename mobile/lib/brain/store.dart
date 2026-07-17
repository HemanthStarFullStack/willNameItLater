/// On-device memory store: hybrid (keyword + vector) search over saved notes —
/// the port of store.py. Each memory is {id, category, text, source,
/// embedding}, persisted to memories.json so it survives restarts.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'embedder.dart';

class Fact {
  int id;
  String category;
  String text;
  String source;
  List<double> embedding;

  /// Raw cosine attached by search (transient — not persisted), so callers can
  /// drop distractor notes; mirrors the `_cos` key in the Python store.
  double cos = 0;

  Fact(this.id, this.category, this.text, this.source, this.embedding);

  Map<String, dynamic> toJson() => {
        'id': id,
        'category': category,
        'text': text,
        'source': source,
        'embedding': embedding,
      };

  static Fact fromJson(Map<String, dynamic> j) => Fact(
        j['id'] as int,
        j['category'] as String,
        j['text'] as String,
        (j['source'] ?? '') as String,
        [for (final e in (j['embedding'] as List)) (e as num).toDouble()],
      );
}

double _dot(List<double> a, List<double> b) {
  var s = 0.0;
  for (var i = 0; i < a.length && i < b.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

double _norm(List<double> a) {
  final n = math.sqrt(_dot(a, a));
  return n == 0 ? 1.0 : n;
}

double cosine(List<double> a, List<double> b) =>
    _dot(a, b) / (_norm(a) * _norm(b));

List<String> _tok(String s) =>
    s.toLowerCase().split(RegExp(r'\s+')).where((w) => w.isNotEmpty).toList();

/// BM25Okapi over the pool — same defaults as the rank_bm25 package the
/// prototype uses (k1=1.5, b=0.75).
List<double> _bm25Scores(List<List<String>> docs, List<String> query) {
  const k1 = 1.5, b = 0.75;
  final n = docs.length;
  if (n == 0) return const [];
  final avgdl = docs.fold<int>(0, (s, d) => s + d.length) / n;
  final df = <String, int>{};
  for (final d in docs) {
    for (final w in d.toSet()) {
      df[w] = (df[w] ?? 0) + 1;
    }
  }
  return [
    for (final d in docs)
      query.fold<double>(0, (s, w) {
        final f = d.where((x) => x == w).length;
        if (f == 0) return s;
        final idf = math.log((n - df[w]! + 0.5) / (df[w]! + 0.5) + 1);
        return s +
            idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / avgdl));
      }),
  ];
}

class MemoryStore {
  final Embedder embedder;
  final String dataFile;
  final List<Fact> facts = [];

  MemoryStore(this.embedder, this.dataFile);

  Future<void> load() async {
    final f = File(dataFile);
    if (!f.existsSync()) return;
    try {
      final raw = json.decode(await f.readAsString()) as List;
      facts
        ..clear()
        ..addAll(raw.map((e) => Fact.fromJson(e as Map<String, dynamic>)));
    } catch (_) {
      // Corrupt file -> start fresh rather than crash the app.
    }
  }

  Future<void> _save() async {
    final f = File(dataFile);
    await f.parent.create(recursive: true);
    await f.writeAsString(json.encode([for (final x in facts) x.toJson()]));
  }

  Future<int> add(String text, String category, {String source = ''}) async {
    final emb = await embedder.embed(text);
    final id = facts.isEmpty
        ? 1
        : facts.map((x) => x.id).reduce(math.max) + 1;
    facts.add(Fact(id, category, text, source, emb));
    await _save();
    return id;
  }

  Future<bool> update(int id, String text) async {
    for (final x in facts) {
      if (x.id == id) {
        x.text = text;
        x.embedding = await embedder.embed(text);
        await _save();
        return true;
      }
    }
    return false;
  }

  Future<bool> delete(int id) async {
    final before = facts.length;
    facts.removeWhere((x) => x.id == id);
    if (facts.length < before) {
      await _save();
      return true;
    }
    return false;
  }

  Map<String, int> categories() {
    final c = <String, int>{};
    for (final x in facts) {
      c[x.category] = (c[x.category] ?? 0) + 1;
    }
    return c;
  }

  List<Fact> factsIn([String? category]) => [
        for (final x in facts)
          if (category == null || x.category == category) x,
      ];

  /// Hybrid retrieval: fuse vector-cosine and BM25 rankings via Reciprocal
  /// Rank Fusion. Returns the fused top-k with each fact's raw cosine set on
  /// `.cos`; `topScore` (best raw cosine) is the calibrated "is any saved note
  /// actually relevant?" signal the chat layer gates RAG with.
  Future<(List<Fact>, double)> search(String query,
      {String? category, int k = 8}) async {
    var pool = factsIn(category);
    if (pool.isEmpty) pool = List.of(facts);
    if (pool.isEmpty) return (const <Fact>[], 0.0);

    final q = await embedder.embed(query);
    final cosines = {for (final x in pool) x.id: cosine(q, x.embedding)};

    final vecRanked = List.of(pool)
      ..sort((a, b) => cosines[b.id]!.compareTo(cosines[a.id]!));

    final scores = _bm25Scores([for (final x in pool) _tok(x.text)], _tok(query));
    final order = List<int>.generate(pool.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    final bmRanked = [for (final i in order) pool[i]];

    final rr = <int, double>{};
    for (var rank = 0; rank < vecRanked.length; rank++) {
      rr[vecRanked[rank].id] = (rr[vecRanked[rank].id] ?? 0) + 1 / (60 + rank);
    }
    for (var rank = 0; rank < bmRanked.length; rank++) {
      rr[bmRanked[rank].id] = (rr[bmRanked[rank].id] ?? 0) + 1 / (60 + rank);
    }

    final fused = List.of(pool)
      ..sort((a, b) => (rr[b.id] ?? 0).compareTo(rr[a.id] ?? 0));
    final out = fused.take(k).toList();
    for (final x in out) {
      x.cos = cosines[x.id]!;
    }
    final top =
        out.isEmpty ? 0.0 : out.map((x) => x.cos).reduce(math.max);
    return (out, top);
  }
}
