allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Some plugins (e.g. fonnx) hardcode an old compileSdk (33) but pull deps that
// require 34/35, breaking the build. Bump every Android plugin module to the
// app's compileSdk so modern ONNX Runtime resolves cleanly. Done reflectively
// because the AGP types aren't on this root build script's classpath.
fun bumpCompileSdk(p: Project) {
    val android = p.extensions.findByName("android") ?: return
    runCatching {
        android.javaClass
            .getMethod("compileSdkVersion", Integer.TYPE)
            .invoke(android, 36)
    }
}
subprojects {
    // evaluationDependsOn(":app") above may have evaluated a project already —
    // afterEvaluate would throw on those, so apply directly in that case.
    if (state.executed) bumpCompileSdk(this) else afterEvaluate { bumpCompileSdk(this) }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
