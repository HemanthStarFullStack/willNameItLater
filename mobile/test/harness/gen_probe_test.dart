/// Verbose generation probe: drives the command stream directly and prints
/// every response type, so a stall shows exactly where it happens.
@Timeout(Duration(minutes: 4))
library;

import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:lib_llama_cpp/lib_llama_cpp.dart';

import 'harness.dart';

void main() {
  setUpAll(registerWindowsPlatform);

  test('raw generate: Qwen3.5-0.8B', () async {
    final commands = StreamController<LlamaCommand>();
    final done = Completer<void>();
    var tokens = 0;
    final buf = StringBuffer();
    final sw = Stopwatch()..start();

    final sub = const LibLlamaCpp()
        .transform(
      commands.stream,
      libraryRequest: LlamaCppLibraryRequest(preferredPath: windowsDll),
    )
        .listen((r) {
      switch (r) {
        case LlamaTokenResponse(:final text):
          tokens++;
          buf.write(text);
        case LlamaDoneResponse():
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] DONE ($tokens tokens): '
              '<<<${buf.toString()}>>>');
          if (!done.isCompleted) done.complete();
        case LlamaErrorResponse(:final message):
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] ERROR: $message');
          if (!done.isCompleted) done.completeError(StateError(message));
        case LlamaStateChangedResponse(:final state):
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] state: loaded=${state.isModelLoaded}');
        case LlamaReadyResponse():
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] engine ready');
        case LlamaToolCallResponse() || LlamaEmbedResponse():
          break;
      }
    });

    commands.add(LlamaLoadModelCommand(
      modelPath: modelPath('Qwen3.5-0.8B-Q4_K_M.gguf'),
      contextSize: 2048,
    ));
    commands.add(LlamaGenerateMessagesCommand(
      messages: [
        LlamaMessage(role: 'user', content: 'Say hello in five words.'),
      ],
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 400,
      enableThinking: false,
    ));

    await done.future.timeout(const Duration(minutes: 3));
    expect(tokens, greaterThan(0));
    await sub.cancel();
    await commands.close();
  });
}
