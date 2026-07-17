# ONNX Runtime is driven from native code via JNI — R8 can't see those calls,
# so keep the whole Java API surface.
-keep class ai.onnxruntime.** { *; }

# The extensions AAR is excluded in build.gradle.kts (dead code for MiniLM);
# fonnx still references it from the disabled isOrtExtensionsEnabled branch.
-dontwarn ai.onnxruntime.extensions.**

# Compile-time-only annotations referenced by transitive deps but not packaged.
# R8 treats these as "missing classes" and fails the build without dontwarn.
-dontwarn com.google.auto.value.**
-dontwarn com.google.errorprone.annotations.**
-dontwarn javax.annotation.**
-dontwarn org.checkerframework.**
-dontwarn com.google.j2objc.annotations.**
