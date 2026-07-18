/// Embedder gate: EmbeddingGemma-300M GGUF vs Qwen3-Embedding-0.6B GGUF via
/// the vendored LlamaEmbedCommand. Every query must retrieve its matching
/// seed fact top-1, and the margins printed here calibrate the pipeline
/// thresholds (kRagThreshold etc.) for whichever model wins.
@Timeout(Duration(minutes: 10))
library;

import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:ondevice_ai/brain/embedder.dart';

import 'harness.dart';

const _facts = [
  'My blood group is O positive.',
  "I'm allergic to penicillin.",
  'My health insurance policy is HLT-889241, renews every March.',
  'Home wifi password is bluefalcon77.',
  'My manager is Priya; standup is 10am daily on weekdays.',
  'Passport number X1234567, expires in 2029.',
  "I'm learning Flutter and Dart this year.",
];

// (query, index of the fact it must hit)
const _queries = [
  ('what is my blood group?', 0),
  ('which medicine am I allergic to?', 1),
  ('when does my insurance renew?', 2),
  ("what's the wifi password at home?", 3),
  ('what time is my standup?', 4),
  ('when does my passport expire?', 5),
  ('what am I studying these days?', 6),
];

// Off-topic queries — their best cosine marks where the RAG gate must close.
const _offTopic = [
  'what is the weather in Hyderabad today?',
  'who won the cricket world cup?',
  'write me a poem about the sea',
];

double _dot(List<double> a, List<double> b) {
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

void main() {
  setUpAll(registerWindowsPlatform);

  for (final model in const [
    'embeddinggemma-300m-qat-Q8_0.gguf',
    'Qwen3-Embedding-0.6B-Q8_0.gguf',
  ]) {
    test('retrieval quality: $model', () async {
      final e = Embedder(libraryPath: windowsDll);
      await e.load(modelPath(model));

      final factVecs = await e.embedBatch(_facts);
      // Vectors must be L2-normalised and non-degenerate.
      for (final v in factVecs) {
        expect(math.sqrt(_dot(v, v)), closeTo(1.0, 1e-3));
      }
      expect(_dot(factVecs[0], factVecs[3]).abs(), lessThan(0.99),
          reason: 'all facts collapsed to one vector');

      var hitMin = 1.0, marginMin = 1.0;
      for (final (q, want) in _queries) {
        final qv = await e.embed(q);
        final scored = [
          for (var i = 0; i < _facts.length; i++) (cos: _dot(qv, factVecs[i]), i: i)
        ]..sort((a, b) => b.cos.compareTo(a.cos));
        final top = scored.first;
        // ignore: avoid_print
        print('  "$q" -> [${top.i}] ${top.cos.toStringAsFixed(3)} '
            '(runner-up ${scored[1].cos.toStringAsFixed(3)})');
        expect(top.i, want, reason: '"$q" retrieved "${_facts[top.i]}"');
        hitMin = math.min(hitMin, top.cos);
        marginMin = math.min(marginMin, top.cos - scored[1].cos);
      }

      var offMax = -1.0;
      for (final q in _offTopic) {
        final qv = await e.embed(q);
        for (final f in factVecs) {
          offMax = math.max(offMax, _dot(qv, f));
        }
      }
      // ignore: avoid_print
      print('  CALIBRATION $model: on-topic min=$hitMin '
          'margin-min=$marginMin off-topic max=$offMax');
      expect(hitMin, greaterThan(offMax),
          reason: 'no threshold can separate on-topic from off-topic');
    });
  }
}
