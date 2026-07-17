import 'dart:async';

import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';

import 'inference_isolate.dart';
import 'llama_command.dart';
import 'llama_response.dart';
import 'llama_state.dart';

abstract interface class LlamaEngine {
  Stream<LlamaResponse> transform(
    Stream<LlamaCommand> commands, {
    LlamaState initialState = const LlamaState.empty(),
    LlamaCppLibraryRequest libraryRequest = const LlamaCppLibraryRequest(),
  });
}

final class LibLlamaCpp implements LlamaEngine {
  const LibLlamaCpp({LibLlamaCppPlatform? platform}) : _platform = platform;

  final LibLlamaCppPlatform? _platform;

  @override
  Stream<LlamaResponse> transform(
    Stream<LlamaCommand> commands, {
    LlamaState initialState = const LlamaState.empty(),
    LlamaCppLibraryRequest libraryRequest = const LlamaCppLibraryRequest(),
  }) async* {
    final platform = _platform ?? LibLlamaCppPlatform.instance;
    late final LlamaCppLibraryDescriptor library;
    try {
      library = await platform.resolveLibrary(request: libraryRequest);
    } on Object catch (error) {
      yield LlamaErrorResponse(
        message: 'Failed to resolve llama.cpp library: $error',
      );
      return;
    }

    late final InferenceIsolate actor;
    try {
      actor = await InferenceIsolate.spawn(
        library: library,
        initialState: initialState,
      );
    } on Object catch (error) {
      yield LlamaErrorResponse(
        message: 'Failed to start llama.cpp inference isolate: $error',
      );
      return;
    }

    yield LlamaReadyResponse(library: library);

    try {
      await for (final command in commands) {
        yield* actor.dispatch(command);
        if (command is LlamaDisposeCommand) {
          break;
        }
      }
    } finally {
      actor.close();
    }
  }
}
