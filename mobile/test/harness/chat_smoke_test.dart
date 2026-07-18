/// Engine + model-architecture smoke tests on the Windows host.
/// Gate for catalog entry: a model must load and produce coherent text here
/// (same llama.cpp pin as the Android .so) before it ships in the app.
///
/// Run: flutter test test/harness/chat_smoke_test.dart
@Timeout(Duration(minutes: 15))
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:ondevice_ai/brain/llm.dart';

import 'harness.dart';

void main() {
  setUpAll(registerWindowsPlatform);

  for (final model in const [
    'Qwen3.5-0.8B-Q4_K_M.gguf',
    'Qwen3.5-2B-Q4_K_M.gguf',
    'gemma-3-1b-it-Q4_K_M.gguf',
    'Qwen3.5-4B-Q4_K_M.gguf',
    'gemma-4-E2B-it-Q4_K_M.gguf',
  ]) {
    test('loads and generates: $model', () async {
      final llm = Llm(libraryPath: windowsDll);
      // Qwen3.5 is a hybrid thinker and ignores the /no_think text switch —
      // disable thinking at the template level, same as the app does.
      if (model.startsWith('Qwen3.5')) llm.enableThinking = false;
      await llm.load(modelPath(model), contextSize: 2048);
      final out = await llm.chat(
        'Reply with exactly: HARNESS OK',
        maxTokens: 64,
      );
      expect(out, isNotEmpty, reason: 'no output — arch unsupported?');
      expect(out.toUpperCase(), contains('HARNESS OK'),
          reason: 'incoherent output: "$out"');

      // A grounded-answer shape check with a memory-style prompt.
      final answer = await llm.chat(
        'NOTES:\n- My blood group is O positive.\n\n'
        'Question: what is my blood group?\n'
        'Answer briefly using ONLY the notes.',
        temperature: 0.2,
        maxTokens: 64,
      );
      expect(answer.toLowerCase(), contains('o positive'),
          reason: 'model failed the memory-answer shape: "$answer"');
    });
  }
}
