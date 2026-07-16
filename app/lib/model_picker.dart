/// On-device model picker.
///
/// Given how much memory a device can actually give one app, choose the
/// largest model whose runtime footprint fits with headroom. Pure logic — no
/// Flutter, no dart:io — so it runs under plain `dart run` and drops straight
/// into the Flutter app later. The device-RAM number is injected by the
/// caller (Flutter reads it per-platform); the *decision* lives here where it
/// can be tested without a device.
///
/// Self-check:  dart run --enable-asserts app/lib/model_picker.dart
library;

class ModelSpec {
  final String id; // short display name
  final String file; // GGUF filename to download / load
  final double params; // billions of parameters
  final int quantBits; // e.g. 4 for Q4_K_M
  final String url; // where to fetch it (HuggingFace GGUF repo)

  const ModelSpec(this.id, this.file, this.params, this.quantBits, this.url);

  /// Rough runtime footprint in GB. Quantised weights are params * bits/8 GB;
  /// add 40% for the KV cache (a few k of context), activations and the
  /// llama.cpp runtime. Deliberately generous — over-estimating means we pick
  /// a model that *stays alive* instead of one iOS jetsams mid-sentence.
  double get footprintGB => params * (quantBits / 8) * 1.4;

  @override
  String toString() => '$id (${params}B Q$quantBits, ~${footprintGB.toStringAsFixed(1)}GB)';
}

/// Small -> large. Q4 GGUF builds that are known to run on-device via
/// llama.cpp. Add/trim freely; the picker just needs them sorted by size.
const List<ModelSpec> catalog = [
  ModelSpec('qwen2.5-0.5b', 'qwen2.5-0.5b-instruct-q4_k_m.gguf', 0.5, 4,
      'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF'),
  ModelSpec('qwen2.5-1.5b', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', 1.5, 4,
      'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF'),
  ModelSpec('llama-3.2-3b', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', 3.0, 4,
      'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF'),
  ModelSpec('phi-3.5-mini', 'Phi-3.5-mini-instruct-Q4_K_M.gguf', 3.8, 4,
      'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF'),
  ModelSpec('qwen2.5-7b', 'qwen2.5-7b-instruct-q4_k_m.gguf', 7.0, 4,
      'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF'),
  ModelSpec('llama-3.1-8b', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', 8.0, 4,
      'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF'),
];

/// Pick the largest model that fits the memory an app can safely use.
///
/// [deviceRamGB] is the device's *total* physical RAM. [reserveFactor] is the
/// fraction of that a single app may spend before the OS starts killing it —
/// iOS foreground apps get roughly half of physical RAM (jetsam), so 0.5 is a
/// safe default. Returns null when nothing fits (device too small).
ModelSpec? pickModel(double deviceRamGB, {double reserveFactor = 0.5}) {
  final budget = deviceRamGB * reserveFactor;
  final fits = catalog.where((m) => m.footprintGB <= budget).toList()
    ..sort((a, b) => b.params.compareTo(a.params)); // largest first
  return fits.isEmpty ? null : fits.first;
}

void main() {
  // Ladder: the pick should grow with the device, and refuse when nothing fits.
  assert(pickModel(8)!.id == 'phi-3.5-mini', pickModel(8).toString()); // M2 iPad
  assert(pickModel(16)!.id == 'llama-3.1-8b', pickModel(16).toString());
  assert(pickModel(6)!.id == 'phi-3.5-mini', pickModel(6).toString());
  assert(pickModel(4)!.id == 'qwen2.5-1.5b', pickModel(4).toString());
  assert(pickModel(2)!.id == 'qwen2.5-0.5b', pickModel(2).toString());
  assert(pickModel(0.4) == null, 'tiny device should get nothing');
  // Bigger device never picks a smaller model than a smaller device.
  assert(pickModel(16)!.params >= pickModel(8)!.params);

  for (final ram in [2, 4, 8, 16]) {
    print('${ram}GB device -> ${pickModel(ram.toDouble())}');
  }
  print('OK: model picker self-check passed');
}
