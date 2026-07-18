/// Full-brain end-to-end on the desktop: the EXACT production pipeline
/// (routing, hybrid retrieval, verify, remember/forget) with the real GGUF
/// embedder and chat model — the same code path the phone runs.
@Timeout(Duration(minutes: 12))
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:ondevice_ai/brain/embedder.dart';
import 'package:ondevice_ai/brain/llm.dart';
import 'package:ondevice_ai/brain/pipeline.dart';
import 'package:ondevice_ai/brain/router.dart' as brain;
import 'package:ondevice_ai/brain/store.dart';

import 'harness.dart';

const _seeds = [
  ('My blood group is O positive.', 'Health'),
  ("I'm allergic to penicillin.", 'Health'),
  ('My health insurance policy is HLT-889241, renews every March.', 'Money'),
  ('Home wifi password is bluefalcon77.', 'Home'),
  ('My manager is Priya; standup is 10am daily on weekdays.', 'Work'),
  ('Passport number X1234567, expires in 2029.', 'Personal'),
  ("I'm learning Flutter and Dart this year.", 'Learning'),
];

void main() {
  setUpAll(registerWindowsPlatform);

  test('memory Q&A through the real pipeline (Qwen3.5-2B)', () async {
    final dir = Directory.systemTemp.createTempSync('brain_e2e');
    addTearDown(() => dir.deleteSync(recursive: true));

    final embedder = Embedder(libraryPath: windowsDll);
    await embedder.load(modelPath('embeddinggemma-300m-qat-Q8_0.gguf'));
    final llm = Llm(libraryPath: windowsDll)..enableThinking = false;
    await llm.load(modelPath('Qwen3.5-2B-Q4_K_M.gguf'));

    final store = MemoryStore(embedder, '${dir.path}/memories.json');
    for (final (text, cat) in _seeds) {
      await store.add(text, cat);
    }
    final pipeline =
        Pipeline(llm, brain.Router(llm, embedder, store), store);

    // Direct memory lookup.
    final r1 = await pipeline.chat('what is my blood group?', const []);
    // ignore: avoid_print
    print('Q1 -> ${r1.answer}  [${r1.footer}]');
    expect(r1.answer.toLowerCase(), contains('o positive'));

    final r2 = await pipeline.chat("what's my wifi password?", const []);
    // ignore: avoid_print
    print('Q2 -> ${r2.answer}  [${r2.footer}]');
    expect(r2.answer, contains('bluefalcon77'));

    // Auto-remember a new fact, then retrieve it.
    final r3 = await pipeline.chat('My bike service is due in August.', const []);
    // ignore: avoid_print
    print('R3 -> ${r3.answer}  [${r3.footer}]');
    expect(r3.footer, contains('saved'),
        reason: 'fact statement should be auto-saved');

    final r4 = await pipeline.chat('when is my bike service due?', const []);
    // ignore: avoid_print
    print('Q4 -> ${r4.answer}  [${r4.footer}]');
    expect(r4.answer.toLowerCase(), contains('august'));

    // Forget flow.
    final r5 = await pipeline.chat('forget my wifi password', const []);
    // ignore: avoid_print
    print('R5 -> ${r5.answer}  [${r5.footer}]');
    expect(r5.answer.toLowerCase(), contains('forgot'));
    expect(
        store.facts.any((f) => f.text.contains('bluefalcon77')), isFalse);
  });
}
