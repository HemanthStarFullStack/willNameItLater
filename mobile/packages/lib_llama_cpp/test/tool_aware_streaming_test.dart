import 'package:lib_llama_cpp/src/llama_command.dart';
import 'package:lib_llama_cpp/src/llama_response.dart';
import 'package:lib_llama_cpp/src/llama_tool.dart';
import 'package:lib_llama_cpp/src/tool_aware_streaming.dart';
import 'package:test/test.dart';

void main() {
  test('streams parsed plain text deltas while tools are available', () {
    final parseCalls = <({String text, bool isPartial})>[];

    final responses = streamToolAwareMessageResponses(
      sampled: const [
        LlamaTokenResponse(text: 'Hello', index: 0),
        LlamaTokenResponse(text: ' world', index: 1),
      ],
      command: const LlamaGenerateMessagesCommand(
        messages: [LlamaMessage(role: 'user', content: 'Write a greeting.')],
        tools: [
          LlamaTool(name: 'lookup', parameters: {'type': 'object'}),
        ],
      ),
      parseChatOutput: (text, {required isPartial}) {
        parseCalls.add((text: text, isPartial: isPartial));
        return {
          'message': {'content': text},
        };
      },
    ).toList();

    expect(responses, const [
      LlamaTokenResponse(text: 'Hello', index: 0),
      LlamaTokenResponse(text: ' world', index: 1),
    ]);
    expect(parseCalls, const [
      (text: 'Hello', isPartial: true),
      (text: 'Hello world', isPartial: true),
      (text: 'Hello world', isPartial: false),
    ]);
  });

  test('emits parsed tool calls without streaming raw tool text', () {
    final responses = streamToolAwareMessageResponses(
      sampled: const [
        LlamaTokenResponse(text: '{"name":"lookup"', index: 0),
        LlamaTokenResponse(text: ',"arguments":{}}', index: 1),
      ],
      command: const LlamaGenerateMessagesCommand(
        messages: [LlamaMessage(role: 'user', content: 'Use lookup.')],
        tools: [
          LlamaTool(name: 'lookup', parameters: {'type': 'object'}),
        ],
      ),
      parseChatOutput: (text, {required isPartial}) {
        if (isPartial) {
          return {
            'message': {'content': ''},
          };
        }
        return {
          'message': {
            'tool_calls': [
              {
                'id': 'call_0',
                'function': {'name': 'lookup', 'arguments': '{}'},
              },
            ],
          },
        };
      },
    ).toList();

    expect(responses, const [
      LlamaToolCallResponse(
        toolCall: LlamaToolCall(
          id: 'call_0',
          index: 0,
          name: 'lookup',
          arguments: '{}',
        ),
      ),
    ]);
  });

  test('streams parsed text deltas without tools', () {
    final responses = streamToolAwareMessageResponses(
      sampled: const [
        LlamaTokenResponse(text: 'Hello', index: 0),
        LlamaTokenResponse(text: ' world', index: 1),
      ],
      command: const LlamaGenerateMessagesCommand(
        messages: [LlamaMessage(role: 'user', content: 'Write a greeting.')],
      ),
      parseChatOutput: (text, {required isPartial}) {
        return {
          'message': {'content': text},
        };
      },
    ).toList();

    expect(responses, const [
      LlamaTokenResponse(text: 'Hello', index: 0),
      LlamaTokenResponse(text: ' world', index: 1),
    ]);
  });

  test('strips thinking/reasoning markers from streamed output', () {
    // Simulates a model that produces <|channel>thought\n...<channel|>content
    // where the parser extracts only the visible content portion.
    final responses = streamToolAwareMessageResponses(
      sampled: const [
        LlamaTokenResponse(text: '<|channel>', index: 0),
        LlamaTokenResponse(text: 'thought\nThe user', index: 1),
        LlamaTokenResponse(text: ' wants a greeting', index: 2),
        LlamaTokenResponse(text: '<channel|>', index: 3),
        LlamaTokenResponse(text: 'Hello!', index: 4),
        LlamaTokenResponse(text: ' Nice day.', index: 5),
      ],
      command: const LlamaGenerateMessagesCommand(
        messages: [LlamaMessage(role: 'user', content: 'Hi')],
      ),
      parseChatOutput: (text, {required isPartial}) {
        // Simulate what llama.cpp's common_chat_parse does:
        // strip everything inside <|channel>...<channel|> markers.
        var content = text;
        final startTag = '<|channel>';
        final endTag = '<channel|>';
        while (true) {
          final startIndex = content.indexOf(startTag);
          if (startIndex < 0) break;
          final endIndex = content.indexOf(endTag, startIndex);
          if (endIndex < 0) {
            // Partial: remove from start tag onwards (thinking in progress).
            content = content.substring(0, startIndex);
            break;
          }
          content =
              content.substring(0, startIndex) +
              content.substring(endIndex + endTag.length);
        }
        return {
          'message': {'content': content},
        };
      },
    ).toList();

    // Only the visible content after the thinking block should be emitted.
    final text = responses
        .whereType<LlamaTokenResponse>()
        .map((r) => r.text)
        .join();
    expect(text, 'Hello! Nice day.');
  });
}
