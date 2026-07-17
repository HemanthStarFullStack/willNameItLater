final class LlamaModelCapabilities {
  const LlamaModelCapabilities({
    this.text = true,
    this.chatTemplate = false,
    this.image = false,
    this.audio = false,
    this.tools = false,
  });

  const LlamaModelCapabilities.empty()
    : text = false,
      chatTemplate = false,
      image = false,
      audio = false,
      tools = false;

  final bool text;
  final bool chatTemplate;
  final bool image;
  final bool audio;
  final bool tools;

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaModelCapabilities &&
            other.text == text &&
            other.chatTemplate == chatTemplate &&
            other.image == image &&
            other.audio == audio &&
            other.tools == tools;
  }

  @override
  int get hashCode => Object.hash(text, chatTemplate, image, audio, tools);

  @override
  String toString() {
    return 'LlamaModelCapabilities('
        'text: $text, '
        'chatTemplate: $chatTemplate, '
        'image: $image, '
        'audio: $audio, '
        'tools: $tools'
        ')';
  }
}

final class LlamaState {
  const LlamaState({
    this.modelPath,
    this.isModelLoaded = false,
    this.capabilities = const LlamaModelCapabilities.empty(),
  });

  const LlamaState.empty()
    : modelPath = null,
      isModelLoaded = false,
      capabilities = const LlamaModelCapabilities.empty();

  final String? modelPath;
  final bool isModelLoaded;
  final LlamaModelCapabilities capabilities;

  LlamaState copyWith({
    String? modelPath,
    bool? isModelLoaded,
    LlamaModelCapabilities? capabilities,
  }) {
    return LlamaState(
      modelPath: modelPath ?? this.modelPath,
      isModelLoaded: isModelLoaded ?? this.isModelLoaded,
      capabilities: capabilities ?? this.capabilities,
    );
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        other is LlamaState &&
            other.modelPath == modelPath &&
            other.isModelLoaded == isModelLoaded &&
            other.capabilities == capabilities;
  }

  @override
  int get hashCode => Object.hash(modelPath, isModelLoaded, capabilities);

  @override
  String toString() {
    return 'LlamaState('
        'modelPath: $modelPath, '
        'isModelLoaded: $isModelLoaded, '
        'capabilities: $capabilities'
        ')';
  }
}
