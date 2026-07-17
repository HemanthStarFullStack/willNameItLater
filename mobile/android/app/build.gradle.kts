plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.ondevice_ai"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.ondevice_ai"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = 28 // lib_llama_cpp_android requires 28 (Android 9+)
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // Signing with the debug key so the APK installs without a keystore.
            signingConfig = signingConfigs.getByName("debug")
            // R8 shrinks classes.dex ~9 MB -> ~3 MB; JNI-reached classes and
            // compile-time-only annotations are handled in proguard-rules.pro.
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    // Compress the native .so libs inside the APK so it fits under the 30 MB
    // chat/upload limit (~35 MB -> ~15 MB). Android extracts them at install
    // time — a bit more storage, no runtime downside. Still a normal,
    // directly-installable APK (no unzip step).
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

// fonnx pulls onnxruntime-extensions (custom ops for exotic models) but only
// touches it behind an isOrtExtensionsEnabled flag that MiniLM never sets —
// ~6 MB of native libs we never load. Excluded to fit delivery size limits.
configurations.all {
    exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime-extensions-android")
}
