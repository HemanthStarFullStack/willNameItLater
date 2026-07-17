import 'dart:convert';

import 'package:lib_llama_cpp/lib_llama_cpp.dart';
import 'package:test/test.dart';

void main() {
  group('OpenAI JSON helpers', () {
    test('converts OpenAI function tools', () {
      final tool = LlamaTool.fromJson({
        'type': 'function',
        'function': {
          'name': 'lookup_weather',
          'description': 'Lookup weather by city.',
          'parameters': {
            'type': 'object',
            'properties': {
              'city': {'type': 'string'},
            },
          },
        },
      });

      expect(tool.name, 'lookup_weather');
      expect(tool.description, 'Lookup weather by city.');
      expect(tool.parameters['type'], 'object');
      expect(tool.toJson(), {
        'type': 'function',
        'function': {
          'name': 'lookup_weather',
          'description': 'Lookup weather by city.',
          'parameters': {
            'type': 'object',
            'properties': {
              'city': {'type': 'string'},
            },
          },
        },
      });
    });

    test('accepts MCP-style input schema aliases for tools', () {
      final tool = LlamaTool.fromJson({
        'name': 'search_docs',
        'description': 'Search local docs.',
        'inputSchema': {
          'type': 'object',
          'properties': {
            'query': {'type': 'string'},
          },
        },
      });

      expect(tool.name, 'search_docs');
      expect(tool.parameters['type'], 'object');
      expect((tool.parameters['properties'] as Map)['query'], {
        'type': 'string',
      });
    });

    test('converts tool choices and tool calls', () {
      expect(LlamaToolChoice.fromJson('required'), LlamaToolChoice.required);

      final choice = LlamaToolChoice.fromJson({
        'type': 'function',
        'function': {'name': 'lookup_weather'},
      });
      expect(choice, const LlamaToolChoice.tool('lookup_weather'));
      expect(choice.toJson(), {
        'type': 'function',
        'function': {'name': 'lookup_weather'},
      });

      final call = LlamaToolCall.fromJson({
        'id': 'call_1',
        'index': 2,
        'type': 'function',
        'function': {
          'name': 'lookup_weather',
          'arguments': {'city': 'Taipei'},
        },
      });

      expect(call.id, 'call_1');
      expect(call.index, 2);
      expect(call.name, 'lookup_weather');
      expect(jsonDecode(call.arguments), {'city': 'Taipei'});
      expect(call.toJson(), {
        'id': 'call_1',
        'index': 2,
        'type': 'function',
        'function': {
          'name': 'lookup_weather',
          'arguments': '{"city":"Taipei"}',
        },
      });
    });

    test('converts chat messages with tool calls', () {
      final message = LlamaMessage.fromJson({
        'role': 'assistant',
        'content': null,
        'tool_calls': [
          {
            'id': 'call_1',
            'type': 'function',
            'function': {
              'name': 'lookup_weather',
              'arguments': '{"city":"Taipei"}',
            },
          },
        ],
      });

      expect(message.role, 'assistant');
      expect(message.content, '');
      expect(message.toolCalls.single.name, 'lookup_weather');
      expect(message.toJson(), {
        'role': 'assistant',
        'content': '',
        'tool_calls': [
          {
            'id': 'call_1',
            'index': 0,
            'type': 'function',
            'function': {
              'name': 'lookup_weather',
              'arguments': '{"city":"Taipei"}',
            },
          },
        ],
      });
    });

    test('converts response and chat input item maps', () {
      final json = {
        'role': 'tool',
        'content': '{"temperature":72}',
        'tool_call_id': 'call_1',
        'name': 'lookup_weather',
      };

      final responseItem = LlamaResponseInputItem.fromJson(json);
      final chatMessage = LlamaChatMessage.fromJson(json);

      expect(responseItem.toolCallId, 'call_1');
      expect(responseItem.toJson(), json);
      expect(chatMessage.name, 'lookup_weather');
      expect(chatMessage.toJson(), json);
    });

    test('converts text content parts', () {
      final message = LlamaMessage.fromJson({
        'role': 'user',
        'content': [
          {'type': 'text', 'text': 'hello'},
          {'type': 'input_text', 'text': ' world'},
        ],
      });

      final parts = message.content as List<LlamaContentPart>;
      expect(llamaContentToPlainText(parts), 'hello world');
      expect(message.toJson(), {
        'role': 'user',
        'content': [
          {'type': 'text', 'text': 'hello'},
          {'type': 'text', 'text': ' world'},
        ],
      });
    });
  });
}
