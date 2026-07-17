final class LlamaModelConfig {
  const LlamaModelConfig({
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
}
