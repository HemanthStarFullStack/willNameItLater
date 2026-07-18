import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';

import 'llama_state.dart';
import 'llama_tool.dart';

sealed class LlamaResponse {
  const LlamaResponse();
}

final class LlamaReadyResponse extends LlamaResponse {
  const LlamaReadyResponse({required this.library});

  final LlamaCppLibraryDescriptor library;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaReadyResponse && other.library == library;
  }

  @override
  int get hashCode => library.hashCode;

  @override
  String toString() => 'LlamaReadyResponse(library: $library)';
}

final class LlamaStateChangedResponse extends LlamaResponse {
  const LlamaStateChangedResponse({required this.state});

  final LlamaState state;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaStateChangedResponse && other.state == state;
  }

  @override
  int get hashCode => state.hashCode;

  @override
  String toString() => 'LlamaStateChangedResponse(state: $state)';
}

final class LlamaTokenResponse extends LlamaResponse {
  const LlamaTokenResponse({required this.text, required this.index});

  final String text;
  final int index;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaTokenResponse &&
            other.text == text &&
            other.index == index;
  }

  @override
  int get hashCode => Object.hash(text, index);

  @override
  String toString() => 'LlamaTokenResponse(text: $text, index: $index)';
}

final class LlamaToolCallResponse extends LlamaResponse {
  const LlamaToolCallResponse({required this.toolCall});

  final LlamaToolCall toolCall;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaToolCallResponse && other.toolCall == toolCall;
  }

  @override
  int get hashCode => toolCall.hashCode;

  @override
  String toString() => 'LlamaToolCallResponse(toolCall: $toolCall)';
}

/// Vendored addition: pooled, L2-normalised embeddings for a batch of texts —
/// the reply to [LlamaEmbedCommand]. One vector per input text, in order.
final class LlamaEmbedResponse extends LlamaResponse {
  const LlamaEmbedResponse({required this.embeddings});

  final List<List<double>> embeddings;

  @override
  String toString() =>
      'LlamaEmbedResponse(${embeddings.length} vectors of '
      '${embeddings.isEmpty ? 0 : embeddings.first.length} dims)';
}

final class LlamaErrorResponse extends LlamaResponse {
  const LlamaErrorResponse({required this.message});

  final String message;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaErrorResponse && other.message == message;
  }

  @override
  int get hashCode => message.hashCode;

  @override
  String toString() => 'LlamaErrorResponse(message: $message)';
}

final class LlamaDoneResponse extends LlamaResponse {
  const LlamaDoneResponse();

  @override
  bool operator ==(Object other) {
    return other is LlamaDoneResponse;
  }

  @override
  int get hashCode => runtimeType.hashCode;

  @override
  String toString() => 'LlamaDoneResponse()';
}
