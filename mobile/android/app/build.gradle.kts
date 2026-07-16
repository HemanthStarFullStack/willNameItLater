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
            // We don't need R8 shrinking for a test build; leave it off so the
            // build can't trip on missing-class references from native deps.
            isMinifyEnabled = false
            isShrinkResources = false
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
