import 'dart:convert';

import 'llama_command.dart';
import 'llama_response.dart';
import 'llama_tool.dart';
import 'tool_call_fallback.dart';

typedef LlamaChatOutputParser =
    Map<String, Object?> Function(String text, {required bool isPartial});

Iterable<LlamaResponse> streamToolAwareMessageResponses({
  required Iterable<LlamaResponse> sampled,
  required LlamaGenerateMessagesCommand command,
  required LlamaChatOutputParser parseChatOutput,
}) sync* {
  final generated = StringBuffer();
  var emittedText = '';
  final canStreamText =
      command.toolChoice == LlamaToolChoice.auto ||
      command.toolChoice == LlamaToolChoice.none;

  for (final response in sampled) {
    if (response is! LlamaTokenResponse) {
      continue;
    }

    generated.write(response.text);
    if (!canStreamText) {
      continue;
    }

    final parsed = parseChatOutput(generated.toString(), isPartial: true);
    if (toolCallsFromParsedMessage(parsed).isNotEmpty) {
      continue;
    }

    final text = contentFromParsedMessage(parsed);
    final delta = nextTextDelta(emittedText, text);
    if (delta == null || delta.isEmpty) {
      continue;
    }

    emittedText = text;
    yield LlamaTokenResponse(text: delta, index: response.index);
  }

  final generatedText = generated.toString();
  final parsed = parseChatOutput(generatedText, isPartial: false);
  final parsedToolCalls = toolCallsFromParsedMessage(parsed);
  if (parsedToolCalls.isEmpty) {
    final fallbackToolCall = forcedToolCallFallback(
      command,
      generatedText: generatedText,
    );
    if (fallbackToolCall != null && emittedText.isEmpty) {
      yield LlamaToolCallResponse(toolCall: fallbackToolCall);
      return;
    }

    final text = contentFromParsedMessage(parsed);
    final delta = nextTextDelta(emittedText, text);
    if (delta != null && delta.isNotEmpty) {
      yield LlamaTokenResponse(text: delta, index: emittedText.length);
    }
    return;
  }

  for (final toolCall in parsedToolCalls) {
    yield LlamaToolCallResponse(toolCall: toolCall);
  }
}

List<LlamaToolCall> toolCallsFromParsedMessage(Map<String, Object?> parsed) {
  final message = parsed['message'];
  if (message is! Map) {
    return const [];
  }
  final rawToolCalls = message['tool_calls'];
  if (rawToolCalls is! List) {
    return const [];
  }

  final calls = <LlamaToolCall>[];
  for (var index = 0; index < rawToolCalls.length; index += 1) {
    final rawCall = rawToolCalls[index];
    if (rawCall is! Map) {
      continue;
    }
    final function = rawCall['function'];
    if (function is! Map) {
      continue;
    }
    final name = function['name'];
    if (name is! String || name.isEmpty) {
      continue;
    }
    final rawArguments = function['arguments'];
    final arguments = rawArguments is String
        ? rawArguments
        : jsonEncode(rawArguments ?? const <String, Object?>{});
    calls.add(
      LlamaToolCall(
        id: rawCall['id'] is String ? rawCall['id'] as String : 'call_$index',
        index: index,
        name: name,
        arguments: arguments,
      ),
    );
  }
  return calls;
}

String contentFromParsedMessage(Map<String, Object?> parsed) {
  final message = parsed['message'];
  if (message is! Map) {
    return '';
  }
  final content = message['content'];
  if (content is String) {
    return content;
  }
  if (content is List) {
    final buffer = StringBuffer();
    for (final part in content) {
      if (part is Map && part['type'] == 'text' && part['text'] is String) {
        buffer.write(part['text']);
      }
    }
    return buffer.toString();
  }
  return '';
}

String? nextTextDelta(String emittedText, String parsedText) {
  if (parsedText.length <= emittedText.length) {
    return '';
  }
  if (!parsedText.startsWith(emittedText)) {
    return null;
  }
  return parsedText.substring(emittedText.length);
}
