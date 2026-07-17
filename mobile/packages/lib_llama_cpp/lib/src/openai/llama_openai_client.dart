import '../lib_llama_cpp.dart';
import 'chat.dart';
import 'errors.dart';
import 'model_config.dart';
import 'responses.dart';

final class LlamaOpenAIClient {
  LlamaOpenAIClient({
    required Map<String, LlamaModelConfig> models,
    LlamaEngine engine = const LibLlamaCpp(),
  }) : _models = Map.unmodifiable(models) {
    responses = LlamaResponsesResource(
      resolveModel: _resolveModel,
      engine: engine,
    );
    chat = LlamaChatResource(responses: responses);
  }

  final Map<String, LlamaModelConfig> _models;
  late final LlamaResponsesResource responses;
  late final LlamaChatResource chat;

  LlamaModelConfig _resolveModel(String model) {
    final config = _models[model];
    if (config == null) {
      throw LlamaOpenAIException(
        code: 'model_not_found',
        message: 'Model "$model" is not registered.',
        param: 'model',
      );
    }
    return config;
  }
}
