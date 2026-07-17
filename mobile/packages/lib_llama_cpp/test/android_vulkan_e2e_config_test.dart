import 'dart:io';

import 'package:test/test.dart';

void main() {
  group('Android Vulkan e2e configuration', () {
    test('workflow provisions Vulkan tooling and enables Android Vulkan e2e', () {
      final root = _repoRoot();
      final workflow = (root / '.github/workflows/e2e.yml').readAsStringSync();
      final androidJob = _workflowJob(workflow, 'android-real-model-smoke');

      expect(androidJob, contains('runs-on: ubuntu-latest'));
      expect(androidJob, contains('ANDROID_ABI: x86_64'));
      expect(androidJob, contains('ANDROID_PLATFORM: android-28'));
      expect(androidJob, contains('LIB_LLAMA_CPP_ENABLE_VULKAN: ON'));
      expect(androidJob, contains("LIB_LLAMA_CPP_TEST_GPU_LAYERS: '1'"));
      expect(androidJob, contains('LIB_LLAMA_CPP_EXAMPLE_TEST_BACKEND: cpu'));
      expect(
        androidJob,
        contains("LIB_LLAMA_CPP_EXAMPLE_TEST_GPU_LAYERS: '0'"),
      );
      expect(androidJob, contains("GGML_VK_VISIBLE_DEVICES: '0'"));
      expect(
        androidJob,
        contains(
          'sudo apt-get install -y clang cmake glslc libvulkan-dev mesa-vulkan-drivers ninja-build spirv-headers vulkan-tools',
        ),
      );
      expect(androidJob, contains('vulkan-host-headers'));
      expect(
        androidJob,
        contains(r'echo "VULKAN_SDK=$RUNNER_TEMP/vulkan-host-headers"'),
      );
      expect(androidJob, contains('ln -sfn /usr/include/vk_video'));
      expect(androidJob, contains('SPIRV-HeadersConfig.cmake'));
      expect(androidJob, contains('share/cmake/SPIRV-Headers'));
      expect(androidJob, contains('echo "LIBGL_ALWAYS_SOFTWARE=1"'));
      expect(androidJob, contains('VK_ICD_FILENAMES='));
      expect(androidJob, contains('storageBuffer16BitAccess'));
      expect(androidJob, contains('vulkaninfo --summary || true'));
      expect(androidJob, contains('api-level: 35'));
      expect(androidJob, contains('target: default'));
      expect(androidJob, contains('arch: x86_64'));
      expect(
        androidJob,
        contains(
          'pre-emulator-launch-script: mkdir -p "\$HOME/.android" && printf "Vulkan = on\\nGLDirectMem = on\\n" > "\$HOME/.android/advancedFeatures.ini"',
        ),
      );
      expect(
        androidJob,
        contains(
          'emulator-options: -no-window -gpu lavapipe -no-snapshot -noaudio -no-boot-anim -no-metrics -feature Vulkan,GLDirectMem',
        ),
      );
      expect(androidJob, contains('Enable Linux KVM'));
      expect(androidJob, contains('disable-linux-hw-accel: false'));
      expect(
        androidJob,
        contains('script: bash .github/scripts/android-real-model-smoke.sh'),
      );
      expect(
        androidJob,
        contains('Run Android Vulkan and example E2E harness'),
      );
    });

    test('release workflow keeps pub package prebuilts CPU-only', () {
      final root = _repoRoot();
      final workflow = (root / '.github/workflows/release.yml')
          .readAsStringSync();
      final publishWorkflow = (root / '.github/workflows/publish.yml')
          .readAsStringSync();
      final androidJob = _workflowJob(workflow, 'native-prebuilt-android');
      final appleJob = _workflowJob(workflow, 'native-prebuilt-apple');
      final windowsJob = _workflowJob(workflow, 'native-prebuilt-windows');

      expect(androidJob, contains('ANDROID_PLATFORM: android-24'));
      expect(androidJob, isNot(contains('LIB_LLAMA_CPP_ENABLE_VULKAN: ON')));
      expect(appleJob, contains('LIB_LLAMA_CPP_ENABLE_METAL: OFF'));
      expect(windowsJob, isNot(contains('Install Vulkan SDK')));
      expect(
        workflow,
        contains('LIB_LLAMA_CPP_ANDROID_PACKAGE_ABIS: arm64-v8a'),
      );
      expect(
        publishWorkflow,
        contains('LIB_LLAMA_CPP_ANDROID_PACKAGE_ABIS: arm64-v8a'),
      );
      expect(
        workflow,
        contains('.github/scripts/build-native-prebuilt.sh android'),
      );
      expect(
        publishWorkflow,
        contains('Download CPU native prebuilts from GitHub release'),
      );
      expect(
        _workflowJob(publishWorkflow, 'publish-facade'),
        contains('- wait-for-server'),
      );
      expect(
        workflow,
        contains('Dispatch accelerated native prebuilt release assets'),
      );
    });

    test('accelerated prebuilt workflow builds release GPU assets', () {
      final root = _repoRoot();
      final workflow = (root / '.github/workflows/accelerated-prebuilts.yml')
          .readAsStringSync();

      expect(workflow, contains('LIB_LLAMA_CPP_ENABLE_METAL: ON'));
      expect(workflow, contains('LIB_LLAMA_CPP_ENABLE_VULKAN: ON'));
      expect(workflow, contains('LIB_LLAMA_CPP_ENABLE_CUDA: ON'));
      expect(workflow, contains('lib_llama_cpp-prebuilt-metal-'));
      expect(workflow, contains('lib_llama_cpp-prebuilt-vulkan-linux-'));
      expect(workflow, contains('lib_llama_cpp-prebuilt-vulkan-android-'));
      expect(workflow, contains('lib_llama_cpp-prebuilt-vulkan-windows-'));
      expect(workflow, contains('lib_llama_cpp-prebuilt-cuda-linux-'));
      expect(workflow, contains('lib_llama_cpp-prebuilt-cuda-windows-'));
      expect(workflow, contains('- cuda-windows'));
      expect(
        _workflowJob(workflow, 'cuda-windows'),
        contains("inputs.backend == 'cuda-windows'"),
      );
      expect(
        _workflowJob(workflow, 'cuda-windows'),
        contains('timeout-minutes: 240'),
      );
      expect(workflow, contains(r'gh release upload "v${VERSION}"'));
    });

    test('release prebuilt download hook supports accelerator variants', () {
      final root = _repoRoot();
      final script = (root / '.github/scripts/download-release-prebuilts.sh')
          .readAsStringSync();

      expect(script, contains('cpu|metal|vulkan-linux|vulkan-android'));
      expect(script, contains('vulkan-windows|cuda-linux|cuda-windows'));
      expect(
        script,
        contains(
          'archive="lib_llama_cpp-prebuilt-\${variant}-\${version}.tar.gz"',
        ),
      );
    });

    test('Android Flutter platform build compiles plugin with Vulkan', () {
      final root = _repoRoot();
      final workflow = (root / '.github/workflows/e2e.yml').readAsStringSync();
      final platformJob = _workflowJob(workflow, 'flutter-platform-builds');

      expect(platformJob, contains('flutter build apk --debug'));
      expect(
        platformJob,
        contains('Install Android Vulkan plugin build dependencies'),
      );
      expect(
        platformJob,
        contains(
          'sudo apt-get install -y cmake glslc libvulkan-dev ninja-build spirv-headers',
        ),
      );
      expect(platformJob, contains('vulkan-host-headers'));
      expect(platformJob, contains('echo "VULKAN_SDK='));
      expect(platformJob, contains('ln -sfn /usr/include/vk_video'));
      expect(platformJob, contains('SPIRV-HeadersConfig.cmake'));
      expect(platformJob, contains('share/cmake/SPIRV-Headers'));
      expect(platformJob, contains('echo "LIB_LLAMA_CPP_ENABLE_VULKAN=ON"'));
      expect(platformJob, contains('echo "ANDROID_PLATFORM=android-28"'));
      expect(platformJob, contains('echo "ANDROID_ABIS=x86_64"'));
      expect(platformJob, contains('Setup Android SDK'));
      expect(platformJob, contains('sdkmanager "platform-tools"'));
      expect(platformJob, contains('platforms;android-36'));
      expect(platformJob, contains('ndk;27.0.12077973'));
      expect(platformJob, contains('ANDROID_NDK_HOME='));
    });

    test('Android native smoke build can compile llama.cpp with Vulkan', () {
      final root = _repoRoot();
      final script = (root / '.github/scripts/build-android-llama-cli.sh')
          .readAsStringSync();

      expect(script, contains('LIB_LLAMA_CPP_ENABLE_VULKAN'));
      expect(script, contains('-DGGML_VULKAN=\$enable_vulkan'));
      expect(script, contains('default_android_platform="android-28"'));
      expect(script, contains('-DVulkan_INCLUDE_DIR='));
      expect(script, contains('-DCMAKE_CXX_FLAGS='));
      expect(script, contains('vulkan-host-headers'));
      expect(script, contains('apply_llama_cpp_ci_patches'));
      expect(script, contains('llama-vulkan-core-16bit-storage.patch'));
      expect(script, contains('--unidiff-zero'));
      expect(script, contains('for include_name in vulkan spirv vk_video'));
      expect(script, contains('-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH'));
      expect(script, contains('-DSPIRV-Headers_DIR='));
      expect(script, contains('VULKAN_HPP_TYPESAFE_CONVERSION=1'));
      expect(script, contains('-DVulkan_LIBRARY='));
      expect(script, contains('android_api_level'));
      expect(script, contains('/\${android_api_level}/libvulkan.so'));
    });

    test('Android native smoke requires real Vulkan layer offload', () {
      final root = _repoRoot();
      final script = (root / '.github/scripts/android-real-model-smoke.sh')
          .readAsStringSync();

      expect(script, isNot(contains('adb kill-server')));
      expect(script, contains('adb devices -l'));
      expect(script, contains('sys.boot_completed'));
      expect(script, contains('guest_env_prefix='));
      expect(script, contains(r'GGML_VK_VISIBLE_DEVICES=$(printf'));
      expect(script, contains('flutter test'));
      expect(script, contains('integration_test/e2e_harness_test.dart'));
      expect(script, contains('LIB_LLAMA_CPP_TEST_BACKEND='));
      expect(script, contains('LIB_LLAMA_CPP_EXAMPLE_TEST_BACKEND'));
      expect(script, contains('LIB_LLAMA_CPP_EXAMPLE_TEST_GPU_LAYERS'));
      expect(
        script,
        contains(
          "grep -Eq 'assigned to device Vulkan|offloaded [1-9][0-9]*/[0-9]+ layers to GPU'",
        ),
      );
      expect(script, isNot(contains("grep -Eq 'ggml_vulkan")));
    });

    test('Android prebuilt build resolves versioned NDK Vulkan library', () {
      final root = _repoRoot();
      final script = (root / '.github/scripts/build-native-prebuilt.sh')
          .readAsStringSync();

      expect(script, contains('android_api_level'));
      expect(script, contains('/\${android_api_level}/libvulkan.so'));
      expect(script, contains('-DCMAKE_CXX_FLAGS='));
      expect(script, contains('vulkan-host-headers'));
      expect(script, contains('apply_llama_cpp_ci_patches'));
      expect(script, contains('llama-vulkan-core-16bit-storage.patch'));
      expect(script, contains('--unidiff-zero'));
      expect(script, contains('for include_name in vulkan spirv vk_video'));
      expect(script, contains('-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH'));
      expect(script, contains('-DSPIRV-Headers_DIR='));
      expect(script, contains('VULKAN_HPP_TYPESAFE_CONVERSION=1'));
    });

    test('publish package installs a size-bounded Android ABI set', () {
      final root = _repoRoot();
      final script = (root / '.github/scripts/install-prebuilt-packages.sh')
          .readAsStringSync();
      final readme = (root / 'packages/lib_llama_cpp_android/README.md')
          .readAsStringSync();

      expect(script, contains('LIB_LLAMA_CPP_ANDROID_PACKAGE_ABIS:-arm64-v8a'));
      expect(script, contains('android_package_abi_list'));
      expect(script, contains('Installed Android package ABIs'));
      expect(script, contains('require_path "\${prebuilt_dir}/android/x86_64'));
      expect(readme, contains('pub.dev releases'));
      expect(readme, contains('`arm64-v8a`'));
      expect(readme, contains('GitHub release CPU archive'));
      expect(readme, contains('`x86_64`'));
    });

    test('Android Gradle build resolves versioned NDK Vulkan library', () {
      final root = _repoRoot();
      final gradle =
          (root / 'packages/lib_llama_cpp_android/android/build.gradle')
              .readAsStringSync();

      expect(gradle, contains('androidApiLevel'));
      expect(gradle, contains('vulkanEnabled = sourceVulkanEnabled'));
      expect(
        gradle,
        contains('minSdkVersion = (vulkanEnabled || nnapiEnabled) ? 28 : 24'),
      );
      expect(gradle, contains('minSdk = minSdkVersion'));
      expect(gradle, contains('/\${androidApiLevel}/libvulkan.so'));
      expect(gradle, contains('-DCMAKE_CXX_FLAGS='));
      expect(gradle, contains('-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH'));
      expect(gradle, contains('-DSPIRV-Headers_DIR='));
      expect(gradle, contains('VULKAN_HPP_TYPESAFE_CONVERSION=1'));
    });

    test('example Android app raises minSdk for Vulkan e2e', () {
      final root = _repoRoot();
      final gradle = (root / 'example/android/app/build.gradle.kts')
          .readAsStringSync();

      expect(gradle, contains('LIB_LLAMA_CPP_ENABLE_VULKAN'));
      expect(
        gradle,
        contains(
          'minSdk = if (libLlamaCppVulkanEnabled || libLlamaCppNnapiEnabled) 28 else flutter.minSdkVersion',
        ),
      );
    });

    test('example e2e harness requests GPU layers when configured', () {
      final root = _repoRoot();
      final testSource =
          (root / 'example/integration_test/e2e_harness_test.dart')
              .readAsStringSync();
      final harnessSource = (root / 'example/lib/main.dart').readAsStringSync();

      expect(harnessSource, contains('LIB_LLAMA_CPP_TEST_GPU_LAYERS'));
      expect(harnessSource, contains('LIB_LLAMA_CPP_TEST_MODEL_ASSET'));
      expect(harnessSource, contains('gpuLayerCount: gpuLayerCount'));
      expect(
        harnessSource,
        contains('backendCapability: config.backendCapability'),
      );
      expect(
        testSource.indexOf('await runner.expectRequiredBackendSupport();'),
        lessThan(testSource.indexOf('client = await runner.createClient();')),
      );
    });

    test('sandboxed Apple workflow passes the model as an example asset', () {
      final root = _repoRoot();
      final workflow = (root / '.github/workflows/e2e.yml').readAsStringSync();
      final realModelJob = _workflowJob(workflow, 'real-model-smoke');
      final iosJob = _workflowJob(workflow, 'ios-real-model-smoke');

      expect(realModelJob, contains('assets/e2e/model.gguf'));
      expect(realModelJob, contains('LIB_LLAMA_CPP_TEST_MODEL_ASSET'));
      expect(iosJob, contains('assets/e2e/model.gguf'));
      expect(iosJob, contains('LIB_LLAMA_CPP_TEST_MODEL_ASSET'));
    });

    test('Gemma 4 E2B workflow exercises server API through llama server', () {
      final root = _repoRoot();
      final workflow = (root / '.github/workflows/e2e.yml').readAsStringSync();
      final gemmaJob = _workflowJob(workflow, 'gemma4-e2b-api-e2e');

      expect(gemmaJob, contains('Run Gemma 4 E2B server API E2E'));
      expect(gemmaJob, contains('LIB_LLAMA_CPP_TEST_MODEL_ALIAS=gemma4-e2b'));
      expect(
        gemmaJob,
        contains(
          'dart test packages/lib_llama_cpp_server/test/native_server_e2e_test.dart -r expanded',
        ),
      );
      expect(gemmaJob, contains('Run Gemma 4 E2B multimodal and tool E2E'));
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

extension on Directory {
  File operator /(String child) => File('$path/$child');
}

String _workflowJob(String workflow, String jobName) {
  final marker = '\n  $jobName:\n';
  final start = workflow.indexOf(marker);
  if (start == -1) {
    throw StateError('Could not find workflow job $jobName.');
  }

  final bodyStart = start + marker.length;
  final nextJob = RegExp(
    r'\n  [A-Za-z0-9_-]+:\n',
  ).firstMatch(workflow.substring(bodyStart));
  final end = nextJob == null ? workflow.length : bodyStart + nextJob.start;
  return workflow.substring(start, end);
}
