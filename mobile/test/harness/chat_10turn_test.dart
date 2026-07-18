/// 13-turn single-session conversation through the real pipeline — verifies
/// every tool fires in one continuous chat: memory retrieval, follow-up
/// rewrite, auto-save, correction/update, web search, general chat, forget.
@Timeout(Duration(minutes: 25))
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

  test('13-turn conversation uses every tool accurately', () async {
    final dir = Directory.systemTemp.createTempSync('brain_10turn');
    addTearDown(() => dir.deleteSync(recursive: true));

    final embedder = Embedder(libraryPath: windowsDll);
    await embedder.load(modelPath('embeddinggemma-300m-qat-Q8_0.gguf'));
    final llm = Llm(libraryPath: windowsDll)..enableThinking = false;
    await llm.load(modelPath('Qwen3.5-2B-Q4_K_M.gguf'));

    final store = MemoryStore(embedder, '${dir.path}/memories.json');
    for (final (text, cat) in _seeds) {
      await store.add(text, cat);
    }
    final pipeline = Pipeline(llm, brain.Router(llm, embedder, store), store);

    final history = <({String role, String content})>[];
    var n = 0;
    Future<ChatReply> turn(String msg) async {
      final sw = Stopwatch()..start();
      final r = await pipeline.chat(msg, List.unmodifiable(history));
      n++;
      // ignore: avoid_print
      print('T$n (${sw.elapsed.inSeconds}s) "$msg"\n'
          '   -> ${r.answer}\n   [${r.footer}]');
      history
        ..add((role: 'user', content: msg))
        ..add((role: 'assistant', content: r.answer));
      return r;
    }

    // 1-2: direct memory + adjacent question, same session.
    var r = await turn('What is my blood group?');
    expect(r.answer.toLowerCase(), contains('o positive'));
    expect(r.footer, contains('🧠'));

    r = await turn('And what am I allergic to?');
    expect(r.answer.toLowerCase(), contains('penicillin'));

    // 3-4: memory then pronoun follow-up (rewrite tool).
    r = await turn('What is my health insurance policy number?');
    expect(r.answer, contains('HLT-889241'));

    r = await turn('When does it renew?');
    expect(r.answer.toLowerCase(), contains('march'),
        reason: 'follow-up rewrite should resolve "it" -> insurance');

    // 5-6: auto-save a new fact, then retrieve it.
    r = await turn('My gym membership expires in December.');
    expect(r.footer, contains('💾'), reason: 'statement should auto-save');

    r = await turn('When does my gym membership expire?');
    expect(r.answer.toLowerCase(), contains('december'));

    // 7-9: memory, correction/update, re-ask.
    r = await turn('Who is my manager?');
    expect(r.answer.toLowerCase(), contains('priya'));

    r = await turn('My manager is now Raj.');
    // update cue -> replace-or-save path; either way it must persist.

    r = await turn('Who is my manager?');
    expect(r.answer.toLowerCase(), contains('raj'),
        reason: 'correction should have replaced Priya');

    // 10: web tool (no memory match + web hint words).
    r = await turn('What is the weather forecast for Hyderabad today?');
    // ignore: avoid_print
    print('   web footer check: ${r.footer}');
    expect(r.footer, contains('🌐'), reason: 'should route to web search');

    // 11: general chat.
    r = await turn('Tell me a short joke.');
    expect(r.footer, contains('💬'));
    expect(r.answer.trim(), isNotEmpty);

    // 12: forget tool.
    r = await turn('Forget my gym membership.');
    expect(r.footer, contains('🗑️'));
    expect(store.facts.any((f) => f.text.toLowerCase().contains('gym')),
        isFalse, reason: 'gym fact must be deleted');

    // 13: general-knowledge accuracy (chat prompt forbids arithmetic by
    // design, so ask a factual question instead).
    r = await turn('What is the capital of France?');
    expect(r.answer, contains('Paris'));

    // ignore: avoid_print
    print('ALL $n TURNS OK · ${store.facts.length} facts in store');
  });
}
