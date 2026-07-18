import 'llama_content.dart';
import 'llama_tool.dart';

sealed class LlamaCommand {
  const LlamaCommand();
}

final class LlamaLoadModelCommand extends LlamaCommand {
  const LlamaLoadModelCommand({
    required this.modelPath,
    this.contextSize,
    this.gpuLayerCount,
    this.mmprojPath,
    this.mmprojUseGpu = false,
    this.imageMinTokens,
    this.imageMaxTokens,
  });

  final String modelPath;
  final int? contextSize;
  final int? gpuLayerCount;
  final String? mmprojPath;
  final bool mmprojUseGpu;
  final int? imageMinTokens;
  final int? imageMaxTokens;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaLoadModelCommand &&
            other.modelPath == modelPath &&
            other.contextSize == contextSize &&
            other.gpuLayerCount == gpuLayerCount &&
            other.mmprojPath == mmprojPath &&
            other.mmprojUseGpu == mmprojUseGpu &&
            other.imageMinTokens == imageMinTokens &&
            other.imageMaxTokens == imageMaxTokens;
  }

  @override
  int get hashCode => Object.hash(
    modelPath,
    contextSize,
    gpuLayerCount,
    mmprojPath,
    mmprojUseGpu,
    imageMinTokens,
    imageMaxTokens,
  );

  @override
  String toString() {
    return 'LlamaLoadModelCommand('
        'modelPath: $modelPath, '
        'contextSize: $contextSize, '
        'gpuLayerCount: $gpuLayerCount, '
        'mmprojPath: $mmprojPath, '
        'mmprojUseGpu: $mmprojUseGpu, '
        'imageMinTokens: $imageMinTokens, '
        'imageMaxTokens: $imageMaxTokens'
        ')';
  }
}

final class LlamaGenerateCommand extends LlamaCommand {
  const LlamaGenerateCommand({
    required this.prompt,
    this.maxTokens,
    this.temperature,
    this.topP,
    this.stop = const [],
  });

  final String prompt;
  final int? maxTokens;
  final double? temperature;
  final double? topP;
  final List<String> stop;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaGenerateCommand &&
            other.prompt == prompt &&
            other.maxTokens == maxTokens &&
            other.temperature == temperature &&
            other.topP == topP &&
            _listEquals(other.stop, stop);
  }

  @override
  int get hashCode =>
      Object.hash(prompt, maxTokens, temperature, topP, Object.hashAll(stop));

  @override
  String toString() {
    return 'LlamaGenerateCommand('
        'prompt: $prompt, '
        'maxTokens: $maxTokens, '
        'temperature: $temperature, '
        'topP: $topP, '
        'stop: $stop'
        ')';
  }
}

final class LlamaMessage {
  const LlamaMessage({
    required this.role,
    required this.content,
    this.toolCalls = const [],
    this.toolCallId,
    this.name,
  });

  factory LlamaMessage.fromJson(Map<String, Object?> json) {
    final rawToolCalls = json['tool_calls'] ?? json['toolCalls'];
    final toolCalls = <LlamaToolCall>[];
    if (rawToolCalls is List) {
      for (var index = 0; index < rawToolCalls.length; index += 1) {
        final rawToolCall = rawToolCalls[index];
        if (rawToolCall is Map) {
          toolCalls.add(
            LlamaToolCall.fromJson(
              _stringKeyedMap(rawToolCall, 'tool call'),
              index: index,
            ),
          );
        }
      }
    }

    return LlamaMessage(
      role: _requiredString(json, 'role'),
      content: llamaContentFromJson(json['content']),
      toolCalls: toolCalls,
      toolCallId:
          _optionalString(json['tool_call_id']) ??
          _optionalString(json['toolCallId']),
      name: _optionalString(json['name']),
    );
  }

  final String role;
  final Object content;
  final List<LlamaToolCall> toolCalls;
  final String? toolCallId;
  final String? name;

  bool get hasMedia => llamaContentHasMedia(content);

  Map<String, Object?> toJson() {
    return {
      'role': role,
      'content': llamaContentToJson(content),
      if (toolCalls.isNotEmpty)
        'tool_calls': [for (final toolCall in toolCalls) toolCall.toJson()],
      if (toolCallId != null) 'tool_call_id': toolCallId,
      if (name != null) 'name': name,
    };
  }
}

final class LlamaGenerateMessagesCommand extends LlamaCommand {
  const LlamaGenerateMessagesCommand({
    required this.messages,
    this.maxTokens,
    this.temperature,
    this.topP,
    this.stop = const [],
    this.tools = const [],
    this.toolChoice = LlamaToolChoice.auto,
    this.parallelToolCalls = false,
    this.enableThinking,
  });

  final List<LlamaMessage> messages;
  final int? maxTokens;
  final double? temperature;
  final double? topP;
  final List<String> stop;
  final List<LlamaTool> tools;
  final LlamaToolChoice toolChoice;
  final bool parallelToolCalls;

  /// Vendored addition: forwarded to the chat template as
  /// `chat_template_kwargs.enable_thinking` — turns hybrid reasoners'
  /// <think> blocks off (false) or on (true); null leaves the model default.
  final bool? enableThinking;

  bool get hasMedia => messages.any((message) => message.hasMedia);
}

/// Vendored addition: compute pooled embeddings for [texts] with the loaded
/// model (use an embedding GGUF such as EmbeddingGemma or Qwen3-Embedding —
/// their metadata carries the pooling type). Replies with one
/// [LlamaEmbedResponse] followed by [LlamaDoneResponse].
final class LlamaEmbedCommand extends LlamaCommand {
  const LlamaEmbedCommand({required this.texts});

  final List<String> texts;

  @override
  String toString() => 'LlamaEmbedCommand(${texts.length} texts)';
}

final class LlamaDisposeCommand extends LlamaCommand {
  const LlamaDisposeCommand();

  @override
  bool operator ==(Object other) {
    return other is LlamaDisposeCommand;
  }

  @override
  int get hashCode => runtimeType.hashCode;

  @override
  String toString() => 'LlamaDisposeCommand()';
}

bool _listEquals<T>(List<T> first, List<T> second) {
  if (identical(first, second)) {
    return true;
  }
  if (first.length != second.length) {
    return false;
  }

  for (var i = 0; i < first.length; i += 1) {
    if (first[i] != second[i]) {
      return false;
    }
  }
  return true;
}

String _requiredString(Map<String, Object?> json, String key) {
  final value = json[key];
  if (value is String && value.isNotEmpty) {
    return value;
  }
  throw ArgumentError.value(json, 'json', 'Expected non-empty string "$key"');
}

String? _optionalString(Object? value) {
  return value is String ? value : null;
}

Map<String, Object?> _stringKeyedMap(Object? value, String name) {
  if (value is! Map) {
    throw ArgumentError.value(value, name, 'Expected a JSON object');
  }

  return {
    for (final entry in value.entries)
      if (entry.key is String) entry.key as String: entry.value,
  };
}
