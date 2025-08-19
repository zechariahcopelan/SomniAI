plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android") version "1.9.25"
    //id("kotlin-kapt")
    id("kotlin-parcelize")
    id("org.jetbrains.kotlin.plugin.serialization") version "1.9.25"
}


configurations.all {
    resolutionStrategy {
        force(
            "org.jetbrains.kotlin:kotlin-stdlib:1.9.25",
            "org.jetbrains.kotlin:kotlin-stdlib-jdk7:1.9.25",
            "org.jetbrains.kotlin:kotlin-stdlib-jdk8:1.9.25",
            "org.jetbrains.kotlin:kotlin-stdlib-common:1.9.25"
        )
    }
}

android {
    namespace = "com.example.somniai"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.somniai"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Room schema export directory
        //kapt {
        //    arguments {
        //        arg("room.schemaLocation", "$projectDir/schemas")
        //    }
        //}

        // Network security configuration
        manifestPlaceholders["usesCleartextTraffic"] = false
    }

    buildTypes {
        debug {
            isDebuggable = true
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            buildConfigField("String", "API_BASE_URL", "\"https://api-dev.somniai.com/\"")
            buildConfigField("boolean", "ENABLE_LOGGING", "true")
        }

        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            buildConfigField("String", "API_BASE_URL", "\"https://api.somniai.com/\"")
            buildConfigField("boolean", "ENABLE_LOGGING", "false")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
        freeCompilerArgs += listOf(
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi",
            "-opt-in=kotlinx.serialization.ExperimentalSerializationApi"
        )
    }

    buildFeatures {
        viewBinding = true
        dataBinding = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            excludes += "/META-INF/versions/9/previous-compilation-data.bin"
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)

    // ========== CORE ANDROID DEPENDENCIES ==========
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.fragment:fragment-ktx:1.6.2")
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")

    // ========== NAVIGATION COMPONENTS ==========
    implementation("androidx.navigation:navigation-fragment-ktx:2.7.5")
    implementation("androidx.navigation:navigation-ui-ktx:2.7.5")

    // ========== DATABASE (ROOM) ==========
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    //kapt("androidx.room:room-compiler:2.6.1")

    // ========== LIFECYCLE COMPONENTS ==========
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-service:2.7.0")
    implementation("androidx.lifecycle:lifecycle-process:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-common-java8:2.7.0")

    // ========== NETWORKING DEPENDENCIES ==========

    // HTTP Client (Retrofit + OkHttp) - FIXED VERSIONS
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.retrofit2:converter-moshi:2.9.0")
    // REMOVED: converter-kotlinx-serialization (doesn't exist in 2.9.0)

    // OkHttp Core and Extensions - FIXED VERSIONS
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.11.0")
    // REMOVED: okhttp-bom (causing conflicts)
    // REMOVED: okhttp-ws (deprecated)
    // REMOVED: okhttp-tls (not needed for basic functionality)

    // WebSocket Support - Alternative implementation
    implementation("org.java-websocket:Java-WebSocket:1.5.4")

    // JSON Processing
    implementation("com.squareup.moshi:moshi:1.15.0")
    implementation("com.squareup.moshi:moshi-kotlin:1.15.0")
    //kapt("com.squareup.moshi:moshi-kotlin-codegen:1.15.0")

    // Kotlin Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-core:1.6.2")

    // Gson for JSON (Alternative to Moshi)
    implementation("com.google.code.gson:gson:2.10.1")

    // Image Loading for Charts and Media
    implementation("io.coil-kt:coil:2.5.0")
    implementation("io.coil-kt:coil-svg:2.5.0")

    // ========== REAL-TIME & BACKGROUND PROCESSING ==========

    // Coroutines for Async Processing
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")

    // WorkManager for Background Tasks
    implementation("androidx.work:work-runtime-ktx:2.9.0")

    // Foreground Service Management
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.core:core:1.12.0")

    // Local Broadcast Manager
    implementation("androidx.localbroadcastmanager:localbroadcastmanager:1.1.0")

    // ========== DATA PROCESSING & ANALYTICS ==========

    // Charts Library (MPAndroidChart)
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")

    // Date/Time Handling
    implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.5.0")

    // Math and Statistical Processing
    implementation("org.apache.commons:commons-math3:3.6.1")

    // ========== SECURITY & ENCRYPTION ==========

    // Android Security Crypto
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // Biometric Authentication
    implementation("androidx.biometric:biometric:1.1.0")

    // ========== AI/ML INTEGRATION ==========

    // TensorFlow Lite for On-device ML - FIXED VERSIONS
    implementation("org.tensorflow:tensorflow-lite:2.13.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.3")
// Exclude conflicting modules
    configurations.all {
        exclude(group = "org.tensorflow", module = "tensorflow-lite-api")
        exclude(group = "org.tensorflow", module = "tensorflow-lite-support-api")
    }

    // ========== DEBUGGING & DEVELOPMENT ==========

    // Network Request/Response Debugging (Debug builds only)
    debugImplementation("com.github.chuckerteam.chucker:library:4.0.0")
    releaseImplementation("com.github.chuckerteam.chucker:library-no-op:4.0.0")

    // Memory Leak Detection (Debug builds only)
    debugImplementation("com.squareup.leakcanary:leakcanary-android:2.12")

    // ========== PREFERENCES & SETTINGS ==========

    // DataStore for Modern Preferences
    implementation("androidx.datastore:datastore-preferences:1.0.0")
    implementation("androidx.datastore:datastore-core:1.0.0")

    // ========== NOTIFICATION & COMMUNICATION ==========

    // Firebase Cloud Messaging (Optional for push notifications)
    // implementation("com.google.firebase:firebase-messaging:23.4.0")

    // ========== PERMISSIONS & SYSTEM INTEGRATION ==========

    // Permission Handling
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.fragment:fragment-ktx:1.6.2")

    // ========== DEPENDENCY INJECTION (Optional but Recommended) ==========

    // Hilt for Dependency Injection
    implementation("com.google.dagger:hilt-android:2.48")
    //kapt("com.google.dagger:hilt-compiler:2.48")

    // ========== TESTING DEPENDENCIES ==========

    // Unit Testing
    testImplementation(libs.junit)
    testImplementation("org.mockito:mockito-core:5.8.0")
    testImplementation("org.mockito.kotlin:mockito-kotlin:5.2.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    testImplementation("androidx.arch.core:core-testing:2.2.0")

    // Room Testing
    testImplementation("androidx.room:room-testing:2.6.1")

    // Network Testing
    testImplementation("com.squareup.okhttp3:mockwebserver:4.11.0")
    testImplementation("org.json:json:20231013")

    // Android Instrumentation Testing
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test:rules:1.5.0")
    androidTestImplementation("androidx.test.espresso:espresso-contrib:3.5.1")
    androidTestImplementation("androidx.work:work-testing:2.9.0")

    // UI Testing
    androidTestImplementation("androidx.test.espresso:espresso-intents:3.5.1")
    androidTestImplementation("androidx.navigation:navigation-testing:2.7.5")
}