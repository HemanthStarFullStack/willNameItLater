import 'dart:convert';

import 'llama_command.dart';
import 'llama_content.dart';
import 'llama_tool.dart';

LlamaToolCall? forcedToolCallFallback(
  LlamaGenerateMessagesCommand command, {
  String generatedText = '',
}) {
  final name = command.toolChoice.name;
  if (command.toolChoice.mode != 'tool' || name == null || name.isEmpty) {
    return null;
  }

  LlamaTool? forcedTool;
  for (final tool in command.tools) {
    if (tool.name == name) {
      forcedTool = tool;
      break;
    }
  }
  if (forcedTool == null) {
    return null;
  }

  return LlamaToolCall(
    id: 'call_0',
    index: 0,
    name: forcedTool.name,
    arguments: jsonEncode(
      _fallbackArguments(
        forcedTool,
        _fallbackText(command.messages, generatedText),
      ),
    ),
  );
}

Map<String, Object?> _fallbackArguments(LlamaTool tool, String text) {
  final properties = _stringKeyedMapOrNull(tool.parameters['properties']);
  final required = _stringList(tool.parameters['required']);
  if (properties == null || required.isEmpty) {
    return const {};
  }

  final arguments = <String, Object?>{};
  for (final name in required) {
    final property = _stringKeyedMapOrNull(properties[name]);
    if (property == null || property['type'] != 'string') {
      continue;
    }
    final value = _inferStringArgument(name, text);
    if (value != null && value.isNotEmpty) {
      arguments[name] = value;
    }
  }
  return arguments;
}

String _fallbackText(List<LlamaMessage> messages, String generatedText) {
  final buffer = StringBuffer();
  for (final message in messages) {
    if (message.role != 'user') {
      continue;
    }
    final text = llamaContentToPlainText(message.content).trim();
    if (text.isNotEmpty) {
      if (buffer.isNotEmpty) {
        buffer.write('\n');
      }
      buffer.write(text);
    }
  }
  final generated = generatedText.trim();
  if (generated.isNotEmpty) {
    if (buffer.isNotEmpty) {
      buffer.write('\n');
    }
    buffer.write(generated);
  }
  return buffer.toString();
}

String? _inferStringArgument(String name, String text) {
  if (text.isEmpty) {
    return null;
  }

  final escapedName = RegExp.escape(name);
  final namedPattern = RegExp(
    '\\b$escapedName\\b\\s*(?:is|=|:)\\s*["\']?([^"\'.,;\\n]+)',
    caseSensitive: false,
  );
  final namedMatch = namedPattern.firstMatch(text);
  if (namedMatch != null) {
    return _cleanArgument(namedMatch.group(1));
  }

  final prepositionPattern = RegExp(
    r'''\b(?:for|in|near|at)\s+([A-Za-z][A-Za-z0-9 _.'-]{0,80}?)(?=\s*(?:[.?!,;:]|$))''',
    caseSensitive: false,
  );
  final prepositionMatch = prepositionPattern.firstMatch(text);
  if (prepositionMatch != null) {
    return _cleanArgument(prepositionMatch.group(1));
  }

  return null;
}

String? _cleanArgument(String? value) {
  final cleaned = value
      ?.trim()
      .replaceAll(RegExp(r"""^[`"']+|[`"']+$"""), '')
      .trim();
  return cleaned == null || cleaned.isEmpty ? null : cleaned;
}

List<String> _stringList(Object? value) {
  if (value is! List) {
    return const [];
  }
  return [
    for (final item in value)
      if (item is String) item,
  ];
}

Map<String, Object?>? _stringKeyedMapOrNull(Object? value) {
  if (value is! Map) {
    return null;
  }
  return {
    for (final entry in value.entries)
      if (entry.key is String) entry.key as String: entry.value,
  };
}
