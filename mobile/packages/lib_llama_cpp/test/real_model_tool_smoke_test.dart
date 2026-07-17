import 'dart:convert';
import 'dart:ffi';
import 'dart:io';

import 'package:lib_llama_cpp/lib_llama_cpp.dart';
import 'package:lib_llama_cpp_ffi/lib_llama_cpp_ffi.dart';
import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';
import 'package:test/test.dart';

const _modelName = 'real-model-tool-smoke';

String _env(String name) => Platform.environment[name] ?? '';

void main() {
  final libraryPath = _env('LIB_LLAMA_CPP_TEST_LIBRARY');
  final modelPath = _env('LIB_LLAMA_CPP_TEST_MODEL');
  final gpuLayerCount = int.tryParse(_env('LIB_LLAMA_CPP_TEST_GPU_LAYERS'));
  final backend = _env('LIB_LLAMA_CPP_TEST_BACKEND');
  final backendCapability = _capabilityForBackend(backend);
  final hasRuntime = libraryPath.isNotEmpty && modelPath.isNotEmpty;

  late final LlamaOpenAIClient client;

  setUpAll(() {
    if (!hasRuntime) {
      return;
    }

    client = LlamaOpenAIClient(
      models: {
        _modelName: LlamaModelConfig(
          modelPath: modelPath,
          contextSize: 1024,
          gpuLayerCount: gpuLayerCount,
        ),
      },
      engine: LibLlamaCpp(
        platform: _FixedLibraryPlatform(
          libraryPath,
          backendCapability: backendCapability,
        ),
      ),
    );
  });

  test('resolves all llcs_engine symbols from native library', () {
    final lib = DynamicLibrary.open(libraryPath);
    // Verify all 7 llcs_* symbols are present and resolvable.
    for (final sym in [
      'llcs_engine_create',
      'llcs_engine_destroy',
      'llcs_engine_caps',
      'llcs_engine_submit',
      'llcs_engine_poll',
      'llcs_engine_cancel',
      'llcs_string_free',
    ]) {
      expect(
        () => lib.lookup(sym),
        returnsNormally,
        reason: '$sym should be resolvable in the native library',
      );
    }
  }, skip: !hasRuntime);

  test(
    'streams a forced structured tool call',
    () async {
      _expectBackendSupport(libraryPath, backendCapability);

      final events = await client.responses
          .stream(
            model: _modelName,
            input:
                'Use the lookup_weather tool for Paris. Do not answer directly.',
            maxOutputTokens: 48,
            temperature: 0,
            tools: const [
              LlamaTool(
                name: 'lookup_weather',
                description: 'Look up current weather for a city.',
                parameters: {
                  'type': 'object',
                  'properties': {
                    'city': {'type': 'string', 'description': 'City name.'},
                  },
                  'required': ['city'],
                },
              ),
            ],
            toolChoice: const LlamaToolChoice.tool('lookup_weather'),
          )
          .timeout(const Duration(minutes: 4))
          .toList();

      expect(events.first, isA<LlamaResponseCreated>());
      expect(events.whereType<LlamaResponseFailed>(), isEmpty);
      final toolCall = events
          .whereType<LlamaResponseToolCallDone>()
          .single
          .toolCall;
      expect(toolCall.name, 'lookup_weather');
      expect(jsonDecode(toolCall.arguments), containsPair('city', 'Paris'));
      expect(events.last, isA<LlamaResponseRequiresAction>());
    },
    skip: !hasRuntime,
    timeout: const Timeout(Duration(minutes: 5)),
  );
}

LlamaCppLibraryCapability? _capabilityForBackend(String backend) {
  return switch (backend) {
    '' || 'cpu' => null,
    'metal' => LlamaCppLibraryCapability.metal,
    'vulkan' => LlamaCppLibraryCapability.vulkan,
    _ => throw ArgumentError.value(
      backend,
      'LIB_LLAMA_CPP_TEST_BACKEND',
      'Expected cpu, metal, or vulkan',
    ),
  };
}

void _expectBackendSupport(
  String libraryPath,
  LlamaCppLibraryCapability? backendCapability,
) {
  if (backendCapability == null) {
    return;
  }

  final bindings = LlamaCppBindings(DynamicLibrary.open(libraryPath));
  expect(
    bindings.llama_supports_gpu_offload(),
    isTrue,
    reason:
        'The ${backendCapability.name} e2e library must expose llama.cpp GPU '
        'offload support before the tool-use smoke runs.',
  );
}

final class _FixedLibraryPlatform extends LibLlamaCppPlatform {
  _FixedLibraryPlatform(this.path, {this.backendCapability});

  final String path;
  final LlamaCppLibraryCapability? backendCapability;

  @override
  Future<LlamaCppLibraryDescriptor> resolveLibrary({
    LlamaCppLibraryRequest request = const LlamaCppLibraryRequest(),
  }) async {
    return LlamaCppLibraryDescriptor(
      resolution: LlamaCppLibraryResolution.path,
      path: path,
      capabilities: {LlamaCppLibraryCapability.cpu, ?backendCapability},
    );
  }
}
