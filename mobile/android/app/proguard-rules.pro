# Compile-time-only annotations referenced by transitive deps but not
# packaged; R8 fails on them as "missing classes" without dontwarn.
-dontwarn com.google.errorprone.annotations.**
-dontwarn javax.annotation.**
