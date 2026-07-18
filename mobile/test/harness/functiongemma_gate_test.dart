/// Gate: FunctionGemma-270M as the agentic router, tested against the
/// PRODUCTION contract — it only dispatches the reminder tools; every other
/// intent must fall through (no false positives) to the classic pipeline.
@Timeout(Duration(minutes: 8))
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:ondevice_ai/brain/agent.dart';
import 'package:ondevice_ai/brain/reminders.dart';

import 'harness.dart';

// (message, expected reminder tool or null = must fall through)
const _cases = <(String, String?)>[
  ('remind me to take my pills at 9pm', 'set_reminder'),
  ('remind me to call mom tomorrow at 8am', 'set_reminder'),
  ('set a reminder for my gym session at 6:30', 'set_reminder'),
  ('what reminders do I have?', 'list_reminders'),
  ('show my reminders', 'list_reminders'),
  ('what is my blood group?', null),
  ('My manager is now Raj.', null),
  // "forget …" never reaches the router: the pipeline's deterministic forget
  // guard runs first (the 270M false-fires list_reminders on those).
  ("I'm allergic to penicillin.", null),
  ('what is the weather in Hyderabad today?', null),
  ('tell me a short joke', null),
];

void main() {
  setUpAll(registerWindowsPlatform);

  test('reminder routing + zero false positives', () async {
    final router = ToolRouter(libraryPath: windowsDll);
    await router.load(modelPath('functiongemma-270m-it-Q8_0.gguf'));

    var ok = 0;
    for (final (msg, want) in _cases) {
      final call = await router.route(msg);
      // Production only dispatches these two; anything else falls through.
      final dispatched = (call?.name == 'set_reminder' ||
              call?.name == 'list_reminders')
          ? call!.name
          : null;
      final pass = dispatched == want;
      if (pass) ok++;
      // ignore: avoid_print
      print('${pass ? 'PASS' : 'FAIL'}  "$msg" -> ${call ?? 'no call'}'
          '${pass ? '' : '  (wanted $want)'}');
    }
    // ignore: avoid_print
    print('ROUTING $ok/${_cases.length}');
    expect(ok, _cases.length,
        reason: 'production contract: exact reminder routing, no false fires');
  });

  test('time parsing heuristics', () {
    final now = DateTime(2026, 7, 18, 20, 0);
    expect(parseWhen('9pm', now: now), DateTime(2026, 7, 18, 21, 0));
    expect(parseWhen('21:30', now: now), DateTime(2026, 7, 18, 21, 30));
    // Earlier than now -> tomorrow.
    expect(parseWhen('8am', now: now), DateTime(2026, 7, 19, 8, 0));
    expect(parseWhen('tomorrow 9am', now: now), DateTime(2026, 7, 19, 9, 0));
    expect(parseWhen('in 20 minutes', now: now),
        DateTime(2026, 7, 18, 20, 20));
    expect(parseWhen('in 2 hours', now: now), DateTime(2026, 7, 18, 22, 0));
    expect(parseWhen('whenever', now: now), isNull);
  });
}
