/// Fast probe: does the engine DLL load + model load succeed at all?
/// Surfaces the exact engine error instead of a silent hang.
@Timeout(Duration(minutes: 3))
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:ondevice_ai/brain/llm.dart';

import 'harness.dart';

void main() {
  setUpAll(registerWindowsPlatform);

  test('engine + Qwen3.5-0.8B load', () async {
    final llm = Llm(libraryPath: windowsDll);
    await llm.load(modelPath('Qwen3.5-0.8B-Q4_K_M.gguf'), contextSize: 2048);
  });
}
