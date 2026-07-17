import 'dart:convert';

import 'package:lib_llama_cpp/src/llama_command.dart';
import 'package:lib_llama_cpp/src/llama_tool.dart';
import 'package:lib_llama_cpp/src/tool_call_fallback.dart';
import 'package:test/test.dart';

void main() {
  test('builds a forced tool call when no structured call was parsed', () {
    final toolCall = forcedToolCallFallback(
      LlamaGenerateMessagesCommand(
        messages: const [
          LlamaMessage(
            role: 'user',
            content:
                'Use the lookup_weather tool for Paris. Do not answer directly.',
          ),
        ],
        tools: const [
          LlamaTool(
            name: 'lookup_weather',
            description: 'Look up current weather for a city.',
            parameters: {
              'type': 'object',
              'properties': {
                'city': {'type': 'string', 'description': 'City name.'},
              },
              'required': ['city'],
            },
          ),
        ],
        toolChoice: const LlamaToolChoice.tool('lookup_weather'),
      ),
    );

    expect(toolCall, isNotNull);
    expect(toolCall!.id, 'call_0');
    expect(toolCall.name, 'lookup_weather');
    expect(jsonDecode(toolCall.arguments), {'city': 'Paris'});
  });

  test('does not fabricate calls for automatic tool choice', () {
    final toolCall = forcedToolCallFallback(
      LlamaGenerateMessagesCommand(
        messages: const [LlamaMessage(role: 'user', content: 'Hello.')],
        tools: const [
          LlamaTool(name: 'lookup_weather', parameters: {'type': 'object'}),
        ],
      ),
    );

    expect(toolCall, isNull);
  });
}
