import 'dart:io';

import 'package:test/test.dart';

void main() {
  group('package metadata', () {
    test('Apple podspec versions match package versions', () {
      final root = _repoRoot();
      final packages = {
        'lib_llama_cpp_ios': (
          pubspec: root / 'packages/lib_llama_cpp_ios/pubspec.yaml',
          podspec:
              root / 'packages/lib_llama_cpp_ios/ios/lib_llama_cpp_ios.podspec',
        ),
        'lib_llama_cpp_macos': (
          pubspec: root / 'packages/lib_llama_cpp_macos/pubspec.yaml',
          podspec:
              root /
              'packages/lib_llama_cpp_macos/macos/lib_llama_cpp_macos.podspec',
        ),
      };

      for (final entry in packages.entries) {
        final pubspecVersion = _pubspecVersion(entry.value.pubspec);
        final podspecVersion = _podspecVersion(entry.value.podspec);

        expect(
          podspecVersion,
          pubspecVersion,
          reason:
              '${entry.key} podspec version must match its pubspec version.',
        );
      }
    });

    test('Apple SwiftPM manifests use Flutter plugin package layout', () {
      final root = _repoRoot();
      final installScript =
          (root / '.github/scripts/install-prebuilt-packages.sh')
              .readAsStringSync();
      final releaseWorkflow = (root / '.github/workflows/release.yml')
          .readAsStringSync();
      final packages = {
        'lib_llama_cpp_ios': (
          platform: 'ios',
          minimumPlatform: '.iOS("13.0")',
          podspec:
              root / 'packages/lib_llama_cpp_ios/ios/lib_llama_cpp_ios.podspec',
        ),
        'lib_llama_cpp_macos': (
          platform: 'macos',
          minimumPlatform: '.macOS("10.15")',
          podspec:
              root /
              'packages/lib_llama_cpp_macos/macos/lib_llama_cpp_macos.podspec',
        ),
      };

      for (final entry in packages.entries) {
        final packageName = entry.key;
        final platform = entry.value.platform;
        final swiftPackageRoot =
            root / 'packages/$packageName/$platform/$packageName';
        final manifest = File('${swiftPackageRoot.path}/Package.swift');
        final flutterFrameworkRoot =
            root / 'packages/$packageName/$platform/FlutterFramework';
        final flutterFrameworkManifest = File(
          '${flutterFrameworkRoot.path}/Package.swift',
        );
        final flutterFrameworkStub = File(
          '${flutterFrameworkRoot.path}/Sources/FlutterFramework/FlutterFramework.swift',
        );
        final legacyManifest =
            root / 'packages/$packageName/$platform/Package.swift';
        final prebuiltPath = '$packageName/Frameworks/$packageName.xcframework';

        expect(
          manifest.existsSync(),
          isTrue,
          reason:
              '$packageName must expose SwiftPM at $platform/$packageName/Package.swift.',
        );

        final manifestContents = manifest.readAsStringSync();
        expect(manifestContents, contains('name: "$packageName"'));
        expect(manifestContents, contains(entry.value.minimumPlatform));
        expect(
          manifestContents,
          contains(
            '.library(name: "${packageName.replaceAll('_', '-')}", '
            'targets: ["$packageName"])',
          ),
        );
        expect(
          manifestContents,
          contains(
            '.package(name: "FlutterFramework", path: "../FlutterFramework")',
          ),
        );
        expect(
          flutterFrameworkManifest.existsSync(),
          isTrue,
          reason:
              '$packageName must publish $platform/FlutterFramework/Package.swift '
              'for direct SwiftPM package resolution.',
        );
        expect(
          flutterFrameworkManifest.readAsStringSync(),
          contains('name: "FlutterFramework"'),
        );
        expect(
          flutterFrameworkStub.existsSync(),
          isTrue,
          reason:
              '$packageName must publish a FlutterFramework stub source file.',
        );
        expect(manifestContents, contains('.binaryTarget('));
        expect(
          manifestContents,
          contains('path: "Frameworks/$packageName.xcframework"'),
        );

        final podspecContents = entry.value.podspec.readAsStringSync();
        expect(podspecContents, contains(prebuiltPath));
        expect(legacyManifest.readAsStringSync(), contains(prebuiltPath));
        expect(
          installScript,
          contains('packages/$packageName/$platform/$packageName/Frameworks'),
        );
        expect(
          releaseWorkflow,
          contains(
            "--exclude='packages/$packageName/$platform/$packageName/Frameworks'",
          ),
        );
      }
    });

    test('Apple source manifests exclude llama.cpp server entrypoint', () {
      final root = _repoRoot();
      const serverMain = 'llama_cpp_sources/llama.cpp/tools/server/main.cpp';
      final manifests = [
        root / 'packages/lib_llama_cpp_ios/ios/lib_llama_cpp_ios.podspec',
        root / 'packages/lib_llama_cpp_ios/ios/Package.swift',
        root / 'packages/lib_llama_cpp_macos/macos/lib_llama_cpp_macos.podspec',
        root / 'packages/lib_llama_cpp_macos/macos/Package.swift',
      ];

      for (final manifest in manifests) {
        final contents = manifest.readAsStringSync();
        final exclusionBlock = manifest.path.endsWith('.podspec')
            ? contents.indexOf('s.exclude_files')
            : contents.indexOf('exclude: [');

        expect(exclusionBlock, isNonNegative);
        expect(
          contents.indexOf(serverMain, exclusionBlock),
          isNonNegative,
          reason:
              '${manifest.path} must not compile llama.cpp server/main.cpp.',
        );
      }
    });

    test('facade package re-exports the local server package', () {
      final root = _repoRoot();
      final pubspec = (root / 'packages/lib_llama_cpp/pubspec.yaml')
          .readAsStringSync();
      final library = (root / 'packages/lib_llama_cpp/lib/lib_llama_cpp.dart')
          .readAsStringSync();

      expect(pubspec, contains('  lib_llama_cpp_server: ^'));
      expect(
        library,
        contains(
          "export 'package:lib_llama_cpp_server/lib_llama_cpp_server.dart';",
        ),
      );
    });
  });
}

Directory _repoRoot() {
  var current = Directory.current;
  while (true) {
    if (File('${current.path}/melos.yaml').existsSync() &&
        Directory('${current.path}/packages/lib_llama_cpp').existsSync()) {
      return current;
    }

    final parent = current.parent;
    if (parent.path == current.path) {
      throw StateError(
        'Could not find repository root from ${Directory.current.path}',
      );
    }
    current = parent;
  }
}

String _pubspecVersion(File file) {
  final match = RegExp(
    r'^version:\s*([^\s]+)\s*$',
    multiLine: true,
  ).firstMatch(file.readAsStringSync());
  if (match == null) {
    throw StateError('Could not find version in ${file.path}');
  }
  return match.group(1)!;
}

String _podspecVersion(File file) {
  final match = RegExp(
    r'''s\.version\s*=\s*['"]([^'"]+)['"]''',
  ).firstMatch(file.readAsStringSync());
  if (match == null) {
    throw StateError('Could not find s.version in ${file.path}');
  }
  return match.group(1)!;
}

extension on Directory {
  File operator /(String child) => File('$path/$child');
}
