import 'dart:async';
import 'dart:isolate';

import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';

import 'llama_command.dart';
import 'llama_response.dart';
import 'llama_state.dart';
import 'native_runtime.dart';

final class InferenceIsolate {
  InferenceIsolate._({
    required Isolate isolate,
    required SendPort commandPort,
    required StreamSubscription<Object?> subscription,
  }) : _isolate = isolate,
       _commandPort = commandPort,
       _subscription = subscription;

  final Isolate _isolate;
  final SendPort _commandPort;
  final StreamSubscription<Object?> _subscription;
  final Map<int, StreamController<LlamaResponse>> _pending = {};
  var _nextRequestId = 0;
  var _isClosed = false;

  static Future<InferenceIsolate> spawn({
    required LlamaCppLibraryDescriptor library,
    required LlamaState initialState,
  }) async {
    final ready = Completer<SendPort>();
    final receivePort = ReceivePort();
    late final StreamSubscription<Object?> subscription;
    late final InferenceIsolate actor;

    subscription = receivePort.listen((message) {
      if (message is SendPort) {
        ready.complete(message);
        return;
      }

      if (message is _ResponseEnvelope) {
        final controller = actor._pending[message.requestId];
        if (controller == null) {
          return;
        }
        if (message.response != null) {
          controller.add(message.response!);
        }
        if (message.isDone) {
          actor._pending.remove(message.requestId);
          unawaited(controller.close());
        }
      }
    });

    final isolate = await Isolate.spawn(
      _runInferenceWorker,
      _StartMessage(
        replyPort: receivePort.sendPort,
        library: library,
        initialState: initialState,
      ),
      debugName: 'lib_llama_cpp_inference',
    );

    final commandPort = await ready.future;
    actor = InferenceIsolate._(
      isolate: isolate,
      commandPort: commandPort,
      subscription: subscription,
    );
    return actor;
  }

  Stream<LlamaResponse> dispatch(LlamaCommand command) {
    if (_isClosed) {
      return Stream<LlamaResponse>.value(
        const LlamaErrorResponse(message: 'Inference isolate is closed.'),
      );
    }

    final requestId = _nextRequestId++;
    final controller = StreamController<LlamaResponse>();
    _pending[requestId] = controller;
    _commandPort.send(_CommandEnvelope(requestId: requestId, command: command));
    return controller.stream;
  }

  void close() {
    if (_isClosed) {
      return;
    }
    _isClosed = true;
    _commandPort.send(const _ShutdownMessage());
    for (final controller in _pending.values) {
      unawaited(controller.close());
    }
    _pending.clear();
    unawaited(_subscription.cancel());
    _isolate.kill(priority: Isolate.immediate);
  }
}

final class _StartMessage {
  const _StartMessage({
    required this.replyPort,
    required this.library,
    required this.initialState,
  });

  final SendPort replyPort;
  final LlamaCppLibraryDescriptor library;
  final LlamaState initialState;
}

final class _CommandEnvelope {
  const _CommandEnvelope({required this.requestId, required this.command});

  final int requestId;
  final LlamaCommand command;
}

final class _ResponseEnvelope {
  const _ResponseEnvelope({
    required this.requestId,
    this.response,
    this.isDone = false,
  });

  final int requestId;
  final LlamaResponse? response;
  final bool isDone;
}

final class _ShutdownMessage {
  const _ShutdownMessage();
}

void _runInferenceWorker(_StartMessage start) {
  var state = start.initialState;
  NativeLlamaRuntime? runtime;
  String? startupError;
  try {
    runtime = NativeLlamaRuntime(library: start.library);
  } on NativeLlamaException catch (error) {
    startupError = error.message;
  } on Object catch (error) {
    startupError = 'Failed to initialize llama.cpp runtime: $error';
  }

  final receivePort = ReceivePort();
  start.replyPort.send(receivePort.sendPort);

  void send(int requestId, LlamaResponse response) {
    start.replyPort.send(
      _ResponseEnvelope(requestId: requestId, response: response),
    );
  }

  void done(int requestId) {
    start.replyPort.send(_ResponseEnvelope(requestId: requestId, isDone: true));
  }

  receivePort.listen((message) {
    if (message is _ShutdownMessage) {
      runtime?.close();
      receivePort.close();
      return;
    }

    if (message is! _CommandEnvelope) {
      return;
    }

    final command = message.command;
    switch (command) {
      case LlamaLoadModelCommand():
        if (startupError != null || runtime == null) {
          send(message.requestId, LlamaErrorResponse(message: startupError!));
        } else {
          try {
            state = runtime.loadModel(command);
            send(message.requestId, LlamaStateChangedResponse(state: state));
          } on NativeLlamaException catch (error) {
            send(message.requestId, LlamaErrorResponse(message: error.message));
          } on Object catch (error) {
            send(
              message.requestId,
              LlamaErrorResponse(message: 'Failed to load model: $error'),
            );
          }
        }
      case LlamaGenerateCommand():
        if (!state.isModelLoaded || runtime == null) {
          send(
            message.requestId,
            const LlamaErrorResponse(
              message: 'Cannot generate before a model is loaded.',
            ),
          );
        } else {
          try {
            for (final response in runtime.generate(command)) {
              send(message.requestId, response);
            }
            // Vendored fix: signal end-of-generation explicitly. The stock
            // package only closed the per-request envelope stream, which a
            // persistent multi-command session cannot observe.
            send(message.requestId, const LlamaDoneResponse());
          } on NativeLlamaException catch (error) {
            send(message.requestId, LlamaErrorResponse(message: error.message));
          } on Object catch (error) {
            send(
              message.requestId,
              LlamaErrorResponse(message: 'Generation failed: $error'),
            );
          }
        }
      case LlamaGenerateMessagesCommand():
        if (!state.isModelLoaded || runtime == null) {
          send(
            message.requestId,
            const LlamaErrorResponse(
              message: 'Cannot generate before a model is loaded.',
            ),
          );
        } else {
          try {
            for (final response in runtime.generateMessages(command)) {
              send(message.requestId, response);
            }
            // Vendored fix: explicit end-of-generation (see generate case).
            send(message.requestId, const LlamaDoneResponse());
          } on NativeLlamaException catch (error) {
            send(message.requestId, LlamaErrorResponse(message: error.message));
          } on Object catch (error) {
            send(
              message.requestId,
              LlamaErrorResponse(message: 'Generation failed: $error'),
            );
          }
        }
      case LlamaEmbedCommand():
        if (!state.isModelLoaded || runtime == null) {
          send(
            message.requestId,
            const LlamaErrorResponse(
              message: 'Cannot embed before a model is loaded.',
            ),
          );
        } else {
          try {
            send(message.requestId, runtime.embed(command));
            send(message.requestId, const LlamaDoneResponse());
          } on NativeLlamaException catch (error) {
            send(message.requestId, LlamaErrorResponse(message: error.message));
          } on Object catch (error) {
            send(
              message.requestId,
              LlamaErrorResponse(message: 'Embedding failed: $error'),
            );
          }
        }
      case LlamaDisposeCommand():
        runtime?.disposeModel();
        state = const LlamaState.empty();
        send(message.requestId, LlamaStateChangedResponse(state: state));
        send(message.requestId, const LlamaDoneResponse());
    }

    done(message.requestId);
  });
}

void unawaited(Future<void> future) {}
