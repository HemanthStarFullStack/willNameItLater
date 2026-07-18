/// Raw probe: what does the engine actually emit when tools are attached to a
/// FunctionGemma request? Prints every event verbatim.
@Timeout(Duration(minutes: 4))
library;

import 'dart:async';

import 'package:flutter_test/flutter_test.dart';
import 'package:lib_llama_cpp/lib_llama_cpp.dart';

import 'harness.dart';

const _tools = [
  LlamaTool(
    name: 'search_memory',
    description:
        'Look up a fact the user previously saved (their personal data).',
    parameters: {
      'type': 'object',
      'properties': {
        'query': {'type': 'string', 'description': 'what to look up'},
      },
      'required': ['query'],
    },
  ),
  LlamaTool(
    name: 'set_reminder',
    description: 'Set a reminder for the user at a given time.',
    parameters: {
      'type': 'object',
      'properties': {
        'text': {'type': 'string', 'description': 'what to remind about'},
        'time': {'type': 'string', 'description': 'when, as the user said it'},
      },
      'required': ['text', 'time'],
    },
  ),
];

void main() {
  setUpAll(registerWindowsPlatform);

  test('raw tool call: FunctionGemma', () async {
    final commands = StreamController<LlamaCommand>();
    var done = Completer<void>();
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
          buf.write(text);
        case LlamaToolCallResponse(:final toolCall):
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] TOOLCALL: '
              '${toolCall.name} args=${toolCall.arguments}');
        case LlamaDoneResponse():
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] DONE text=<<<$buf>>>');
          buf.clear();
          if (!done.isCompleted) done.complete();
        case LlamaErrorResponse(:final message):
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] ERROR: $message');
          if (!done.isCompleted) done.completeError(StateError(message));
        case LlamaStateChangedResponse(:final state):
          // ignore: avoid_print
          print('[${sw.elapsedMilliseconds}ms] loaded=${state.isModelLoaded}');
        case LlamaReadyResponse() || LlamaEmbedResponse():
          break;
      }
    });

    commands.add(LlamaLoadModelCommand(
      modelPath: modelPath('functiongemma-270m-it-Q8_0.gguf'),
      contextSize: 2048,
    ));

    for (final msg in [
      'remind me to take my pills at 9pm',
      'what is my blood group?',
    ]) {
      // ignore: avoid_print
      print('--- "$msg" (toolChoice auto) ---');
      done = Completer<void>();
      commands.add(LlamaGenerateMessagesCommand(
        messages: [LlamaMessage(role: 'user', content: msg)],
        tools: _tools,
        temperature: 0,
        maxTokens: 200,
      ));
      await done.future.timeout(const Duration(minutes: 1));

      // ignore: avoid_print
      print('--- "$msg" (toolChoice required) ---');
      done = Completer<void>();
      commands.add(LlamaGenerateMessagesCommand(
        messages: [LlamaMessage(role: 'user', content: msg)],
        tools: _tools,
        toolChoice: LlamaToolChoice.required,
        temperature: 0,
        maxTokens: 200,
      ));
      await done.future.timeout(const Duration(minutes: 1));
    }

    await sub.cancel();
    await commands.close();
  });
}
