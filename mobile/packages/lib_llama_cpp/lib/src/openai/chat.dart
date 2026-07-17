import '../llama_command.dart';
import '../llama_tool.dart';
import 'responses.dart';

final class LlamaChatResource {
  LlamaChatResource({required LlamaResponsesResource responses})
    : completions = LlamaChatCompletionsResource(responses: responses);

  final LlamaChatCompletionsResource completions;
}

final class LlamaChatCompletionsResource {
  const LlamaChatCompletionsResource({
    required LlamaResponsesResource responses,
  }) : _responses = responses;

  final LlamaResponsesResource _responses;

  Future<LlamaChatCompletion> create({
    required String model,
    required List<LlamaChatMessage> messages,
    int? maxTokens,
    double? temperature,
    double? topP,
    List<String> stop = const [],
    List<LlamaTool> tools = const [],
    LlamaToolChoice toolChoice = LlamaToolChoice.auto,
    bool parallelToolCalls = false,
  }) async {
    final response = await _responses.create(
      model: model,
      input: [
        for (final message in messages)
          LlamaResponseInputItem(
            role: message.role,
            content: message.content,
            toolCalls: message.toolCalls,
            toolCallId: message.toolCallId,
            name: message.name,
          ),
      ],
      maxOutputTokens: maxTokens,
      temperature: temperature,
      topP: topP,
      stop: stop,
      tools: tools,
      toolChoice: toolChoice,
      parallelToolCalls: parallelToolCalls,
    );

    return LlamaChatCompletion(
      id: 'chatcmpl_${DateTime.now().toUtc().microsecondsSinceEpoch}',
      createdAt: DateTime.now().toUtc(),
      model: model,
      choices: [
        LlamaChatChoice(
          index: 0,
          message: LlamaChatMessage(
            role: 'assistant',
            content: response.outputText,
            toolCalls: response.toolCalls,
          ),
          finishReason: response.toolCalls.isEmpty ? 'stop' : 'tool_calls',
        ),
      ],
    );
  }
}

final class LlamaChatMessage {
  const LlamaChatMessage({
    required this.role,
    required this.content,
    this.toolCalls = const [],
    this.toolCallId,
    this.name,
  });

  factory LlamaChatMessage.fromJson(Map<String, Object?> json) {
    final message = LlamaMessage.fromJson(json);
    return LlamaChatMessage(
      role: message.role,
      content: message.content,
      toolCalls: message.toolCalls,
      toolCallId: message.toolCallId,
      name: message.name,
    );
  }

  final String role;
  final Object content;
  final List<LlamaToolCall> toolCalls;
  final String? toolCallId;
  final String? name;

  LlamaMessage toLlamaMessage() {
    return LlamaMessage(
      role: role,
      content: content,
      toolCalls: toolCalls,
      toolCallId: toolCallId,
      name: name,
    );
  }

  Map<String, Object?> toJson() {
    return toLlamaMessage().toJson();
  }
}

final class LlamaChatCompletion {
  const LlamaChatCompletion({
    required this.id,
    required this.createdAt,
    required this.model,
    required this.choices,
  });

  final String id;
  final DateTime createdAt;
  final String model;
  final List<LlamaChatChoice> choices;
}

final class LlamaChatChoice {
  const LlamaChatChoice({
    required this.index,
    required this.message,
    required this.finishReason,
  });

  final int index;
  final LlamaChatMessage message;
  final String finishReason;
}
