/// Desktop test harness: runs the real llama.cpp engine (same package pin as
/// Android) on the Windows host, so models, embedders, and the full brain
/// pipeline can be validated on the laptop — the x86_64 emulator cannot run
/// any GGUF session (arm64-only .so), so this is the AI-layer test bed.
///
/// Model files live in test/harness/models/ (gitignored). Download them with:
///   dart run test/harness/fetch_models.dart
library;

import 'dart:io';

import 'package:lib_llama_cpp_platform_interface/lib_llama_cpp_platform_interface.dart';
import 'package:lib_llama_cpp_windows/lib_llama_cpp_windows.dart';

/// The CPU DLL shipped inside the pub-cache windows platform package.
final windowsDll = () {
  final pubCache =
      Platform.environment['PUB_CACHE'] ??
          '${Platform.environment['LOCALAPPDATA']}\\Pub\\Cache';
  final path =
      '$pubCache\\hosted\\pub.dev\\lib_llama_cpp_windows-0.7.3'
      '\\windows\\prebuilt\\x64\\lib_llama_cpp_windows.dll';
  if (!File(path).existsSync()) {
    throw StateError('Windows llama DLL not found at $path — run flutter pub get.');
  }
  return path;
}();

/// Registers the Windows platform implementation (normally done by the
/// Flutter plugin registrar, which plain `flutter test` skips).
void registerWindowsPlatform() {
  LibLlamaCppPlatform.instance = LibLlamaCppWindows();
}

final modelsDir = '${Directory.current.path}\\test\\harness\\models';

String modelPath(String file) {
  final p = '$modelsDir\\$file';
  if (!File(p).existsSync()) {
    throw StateError(
        'Model $file missing — run: dart run test/harness/fetch_models.dart');
  }
  return p;
}
