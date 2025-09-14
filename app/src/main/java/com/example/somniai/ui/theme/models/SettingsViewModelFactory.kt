package com.example.somniai.ui.theme.models


import android.content.Context
import android.content.SharedPreferences
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.somniai.ai.AIConfiguration
import com.example.somniai.ai.AIPerformanceMonitor
import com.example.somniai.ai.AIResponseParser
import com.example.somniai.ai.ParsingPerformanceAnalytics
import com.example.somniai.data.repository.UserPreferencesRepository
import com.example.somniai.data.repository.SleepSessionRepository
import com.example.somniai.data.repository.InsightRepository
import com.example.somniai.data.*
import com.example.somniai.utils.SecurityUtils
import com.example.somniai.utils.BackupManager
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import java.util.concurrent.TimeUnit

/**
 * ViewModelFactory for SettingsViewModel in SomniAI
 *
 * Advanced factory providing comprehensive dependency injection for sophisticated
 * settings management, AI configuration optimization, user preference handling,
 * privacy controls, performance tuning, and system customization. Integrates
 * seamlessly with the complete SomniAI ecosystem including AI processing pipeline,
 * performance monitoring, and enterprise-grade security features.
 */
class SettingsViewModelFactory(
    private val context: Context,
    private val userPreferencesRepository: UserPreferencesRepository,
    private val sleepSessionRepository: SleepSessionRepository,
    private val insightRepository: InsightRepository,
    private val sharedPreferences: SharedPreferences,
    private val aiConfiguration: AIConfiguration = AIConfiguration,
    private val aiPerformanceMonitor: AIPerformanceMonitor = AIPerformanceMonitor,
    private val aiResponseParser: AIResponseParser? = null,
    private val parsingAnalytics: ParsingPerformanceAnalytics? = null,
    private val securityUtils: SecurityUtils? = null,
    private val backupManager: BackupManager? = null,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val mainDispatcher: CoroutineDispatcher = Dispatchers.Main,
    private val settingsConfiguration: SettingsConfiguration = SettingsConfiguration.default(),
    private val enableAdvancedSettings: Boolean = true,
    private val enableAIConfiguration: Boolean = true,
    private val enablePerformanceSettings: Boolean = true,
    private val enablePrivacyControls: Boolean = true,
    private val enableBackupRestore: Boolean = true
) : ViewModelProvider.Factory {

    companion object {
        private const val TAG = "SettingsViewModelFactory"

        // Settings configuration constants
        private const val SETTINGS_CACHE_SIZE = 50
        private const val SETTINGS_SYNC_INTERVAL = 10 * 60 * 1000L // 10 minutes
        private const val BACKUP_RETENTION_DAYS = 30L
        private const val PERFORMANCE_SAMPLE_RATE = 0.05f // 5% sampling

        // Privacy and security settings
        private const val DEFAULT_DATA_RETENTION_DAYS = 365L
        private const val MIN_BACKUP_INTERVAL_HOURS = 6L
        private const val MAX_EXPORT_FILE_SIZE_MB = 100L

        // AI and performance settings
        private const val DEFAULT_AI_CONFIDENCE_THRESHOLD = 0.7f
        private const val DEFAULT_PERFORMANCE_MONITORING_ENABLED = true
        private const val DEFAULT_ANALYTICS_ENABLED = true

        /**
         * Create factory with standard dependencies for production use
         */
        fun create(
            context: Context,
            userPreferencesRepository: UserPreferencesRepository,
            sleepSessionRepository: SleepSessionRepository,
            insightRepository: InsightRepository
        ): SettingsViewModelFactory {
            val sharedPrefs = context.getSharedPreferences("somniai_settings_prefs", Context.MODE_PRIVATE)

            return SettingsViewModelFactory(
                context = context,
                userPreferencesRepository = userPreferencesRepository,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = insightRepository,
                sharedPreferences = sharedPrefs,
                settingsConfiguration = SettingsConfiguration.fromPreferences(sharedPrefs)
            )
        }

        /**
         * Create factory with comprehensive AI and security features
         */
        fun createWithFullFeatures(
            context: Context,
            userPreferencesRepository: UserPreferencesRepository,
            sleepSessionRepository: SleepSessionRepository,
            insightRepository: InsightRepository,
            aiResponseParser: AIResponseParser,
            parsingAnalytics: ParsingPerformanceAnalytics,
            securityUtils: SecurityUtils,
            backupManager: BackupManager
        ): SettingsViewModelFactory {
            val sharedPrefs = context.getSharedPreferences("somniai_settings_prefs", Context.MODE_PRIVATE)

            return SettingsViewModelFactory(
                context = context,
                userPreferencesRepository = userPreferencesRepository,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = insightRepository,
                sharedPreferences = sharedPrefs,
                aiResponseParser = aiResponseParser,
                parsingAnalytics = parsingAnalytics,
                securityUtils = securityUtils,
                backupManager = backupManager,
                settingsConfiguration = SettingsConfiguration.fromPreferences(sharedPrefs),
                enableAdvancedSettings = true,
                enableAIConfiguration = true,
                enablePerformanceSettings = true,
                enablePrivacyControls = true,
                enableBackupRestore = true
            )
        }

        /**
         * Create factory for testing with controlled dependencies
         */
        fun createForTesting(
            context: Context,
            userPreferencesRepository: UserPreferencesRepository,
            testDispatcher: CoroutineDispatcher,
            enableFeatures: Boolean = false
        ): SettingsViewModelFactory {
            // Create minimal test repositories
            val testSleepRepo = object : SleepSessionRepository {
                override suspend fun getAllSessions(): List<SleepSession> = emptyList()
                override suspend fun getSessionById(id: Long): SleepSession? = null
                override suspend fun insertSession(session: SleepSession): Long = -1L
                override suspend fun updateSession(session: SleepSession) {}
                override suspend fun deleteSession(id: Long) {}
                override suspend fun getSessionsInDateRange(startDate: Long, endDate: Long): List<SleepSession> = emptyList()
            }

            val testInsightRepo = object : InsightRepository {
                override suspend fun getAllInsights(): List<SleepInsight> = emptyList()
                override suspend fun getInsightsForSession(sessionId: Long): List<SleepInsight> = emptyList()
                override suspend fun insertInsight(insight: SleepInsight): Long = -1L
                override suspend fun updateInsight(insight: SleepInsight) {}
                override suspend fun deleteInsight(insightId: String) {}
            }

            val sharedPrefs = context.getSharedPreferences("test_settings_prefs", Context.MODE_PRIVATE)

            return SettingsViewModelFactory(
                context = context,
                userPreferencesRepository = userPreferencesRepository,
                sleepSessionRepository = testSleepRepo,
                insightRepository = testInsightRepo,
                sharedPreferences = sharedPrefs,
                ioDispatcher = testDispatcher,
                mainDispatcher = testDispatcher,
                enableAdvancedSettings = enableFeatures,
                enableAIConfiguration = enableFeatures,
                enablePerformanceSettings = enableFeatures,
                enablePrivacyControls = enableFeatures,
                enableBackupRestore = enableFeatures,
                settingsConfiguration = SettingsConfiguration.createTestConfiguration()
            )
        }

        /**
         * Create factory with AI configuration focus
         */
        fun createForAIConfiguration(
            context: Context,
            userPreferencesRepository: UserPreferencesRepository,
            aiResponseParser: AIResponseParser,
            parsingAnalytics: ParsingPerformanceAnalytics
        ): SettingsViewModelFactory {
            val sharedPrefs = context.getSharedPreferences("somniai_ai_settings", Context.MODE_PRIVATE)

            // Create minimal repositories for AI-focused settings
            val minimalSleepRepo = object : SleepSessionRepository {
                override suspend fun getAllSessions(): List<SleepSession> = emptyList()
                override suspend fun getSessionById(id: Long): SleepSession? = null
                override suspend fun insertSession(session: SleepSession): Long = -1L
                override suspend fun updateSession(session: SleepSession) {}
                override suspend fun deleteSession(id: Long) {}
                override suspend fun getSessionsInDateRange(startDate: Long, endDate: Long): List<SleepSession> = emptyList()
            }

            val minimalInsightRepo = object : InsightRepository {
                override suspend fun getAllInsights(): List<SleepInsight> = emptyList()
                override suspend fun getInsightsForSession(sessionId: Long): List<SleepInsight> = emptyList()
                override suspend fun insertInsight(insight: SleepInsight): Long = -1L
                override suspend fun updateInsight(insight: SleepInsight) {}
                override suspend fun deleteInsight(insightId: String) {}
            }

            return SettingsViewModelFactory(
                context = context,
                userPreferencesRepository = userPreferencesRepository,
                sleepSessionRepository = minimalSleepRepo,
                insightRepository = minimalInsightRepo,
                sharedPreferences = sharedPrefs,
                aiResponseParser = aiResponseParser,
                parsingAnalytics = parsingAnalytics,
                enableAdvancedSettings = false,
                enableAIConfiguration = true,
                enablePerformanceSettings = true,
                enablePrivacyControls = false,
                enableBackupRestore = false,
                settingsConfiguration = SettingsConfiguration.createAIFocusedConfiguration()
            )
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return when {
            modelClass.isAssignableFrom(SettingsViewModel::class.java) -> {
                createSettingsViewModel() as T
            }
            else -> {
                throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
            }
        }
    }

    /**
     * Create SettingsViewModel with comprehensive dependency injection
     */
    private fun createSettingsViewModel(): SettingsViewModel {
        // Initialize preference manager
        val preferenceManager = SettingsPreferenceManager(
            repository = userPreferencesRepository,
            sharedPreferences = sharedPreferences,
            configuration = settingsConfiguration,
            performanceMonitor = aiPerformanceMonitor,
            enableCaching = settingsConfiguration.enablePreferenceCaching,
            cacheSize = SETTINGS_CACHE_SIZE
        )

        // Initialize AI configuration manager (if enabled)
        val aiConfigurationManager = if (enableAIConfiguration) {
            SettingsAIConfigurationManager(
                aiConfiguration = aiConfiguration,
                parser = aiResponseParser,
                analytics = parsingAnalytics,
                performanceMonitor = aiPerformanceMonitor,
                defaultConfidenceThreshold = DEFAULT_AI_CONFIDENCE_THRESHOLD,
                enableAdvancedAISettings = enableAdvancedSettings
            )
        } else null

        // Initialize privacy controls manager (if enabled)
        val privacyControlsManager = if (enablePrivacyControls) {
            SettingsPrivacyControlsManager(
                securityUtils = securityUtils,
                performanceMonitor = aiPerformanceMonitor,
                dataRetentionDays = DEFAULT_DATA_RETENTION_DAYS,
                enableDataEncryption = settingsConfiguration.enableDataEncryption,
                enableAnonymization = settingsConfiguration.enableAnonymization
            )
        } else null

        // Initialize backup and restore manager (if enabled)
        val backupRestoreManager = if (enableBackupRestore && backupManager != null) {
            SettingsBackupRestoreManager(
                backupManager = backupManager,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = insightRepository,
                performanceMonitor = aiPerformanceMonitor,
                maxBackupSizeMB = MAX_EXPORT_FILE_SIZE_MB,
                retentionDays = BACKUP_RETENTION_DAYS,
                minBackupIntervalHours = MIN_BACKUP_INTERVAL_HOURS
            )
        } else null

        // Initialize performance settings manager (if enabled)
        val performanceSettingsManager = if (enablePerformanceSettings) {
            SettingsPerformanceManager(
                performanceMonitor = aiPerformanceMonitor,
                parsingAnalytics = parsingAnalytics,
                configuration = settingsConfiguration.performanceConfiguration,
                enablePerformanceMonitoring = DEFAULT_PERFORMANCE_MONITORING_ENABLED,
                enableAnalytics = DEFAULT_ANALYTICS_ENABLED,
                sampleRate = PERFORMANCE_SAMPLE_RATE
            )
        } else null

        // Initialize theme and appearance manager
        val themeAppearanceManager = SettingsThemeAppearanceManager(
            context = context,
            sharedPreferences = sharedPreferences,
            configuration = settingsConfiguration.themeConfiguration,
            enableDynamicTheming = settingsConfiguration.enableDynamicTheming,
            enableAccessibilityFeatures = settingsConfiguration.enableAccessibilityFeatures
        )

        // Initialize notification settings manager
        val notificationSettingsManager = SettingsNotificationManager(
            context = context,
            sharedPreferences = sharedPreferences,
            configuration = settingsConfiguration.notificationConfiguration,
            enableSmartNotifications = settingsConfiguration.enableSmartNotifications,
            performanceMonitor = aiPerformanceMonitor
        )

        // Initialize data export manager
        val dataExportManager = SettingsDataExportManager(
            context = context,
            sleepSessionRepository = sleepSessionRepository,
            insightRepository = insightRepository,
            performanceMonitor = aiPerformanceMonitor,
            supportedFormats = settingsConfiguration.exportFormats,
            maxFileSizeMB = MAX_EXPORT_FILE_SIZE_MB
        )

        // Initialize settings sync manager
        val settingsSyncManager = SettingsSyncManager(
            preferenceManager = preferenceManager,
            performanceMonitor = aiPerformanceMonitor,
            syncInterval = SETTINGS_SYNC_INTERVAL,
            enableCloudSync = settingsConfiguration.enableCloudSync,
            enableCrossDeviceSync = settingsConfiguration.enableCrossDeviceSync
        )

        // Initialize settings validation manager
        val settingsValidationManager = SettingsValidationManager(
            configuration = settingsConfiguration,
            performanceMonitor = aiPerformanceMonitor,
            validationRules = settingsConfiguration.validationRules,
            enableStrictValidation = settingsConfiguration.enableStrictValidation
        )

        // Initialize settings analytics tracker
        val settingsAnalyticsTracker = SettingsAnalyticsTracker(
            performanceMonitor = aiPerformanceMonitor,
            parsingAnalytics = parsingAnalytics,
            enabled = enablePerformanceSettings && DEFAULT_ANALYTICS_ENABLED
        )

        // Return fully configured ViewModel
        return SettingsViewModel(
            // Core repositories
            userPreferencesRepository = userPreferencesRepository,
            sleepSessionRepository = sleepSessionRepository,
            insightRepository = insightRepository,

            // Settings management components
            preferenceManager = preferenceManager,
            aiConfigurationManager = aiConfigurationManager,
            privacyControlsManager = privacyControlsManager,
            backupRestoreManager = backupRestoreManager,

            // UI and experience managers
            themeAppearanceManager = themeAppearanceManager,
            notificationSettingsManager = notificationSettingsManager,
            performanceSettingsManager = performanceSettingsManager,

            // Data and sync managers
            dataExportManager = dataExportManager,
            settingsSyncManager = settingsSyncManager,

            // Utility components
            settingsValidationManager = settingsValidationManager,
            settingsAnalyticsTracker = settingsAnalyticsTracker,

            // System dependencies
            context = context.applicationContext,
            sharedPreferences = sharedPreferences,
            ioDispatcher = ioDispatcher,
            mainDispatcher = mainDispatcher,

            // Configuration and flags
            settingsConfiguration = settingsConfiguration,
            enableAdvancedSettings = enableAdvancedSettings,
            enableAIConfiguration = enableAIConfiguration,
            enablePerformanceSettings = enablePerformanceSettings,
            enablePrivacyControls = enablePrivacyControls,
            enableBackupRestore = enableBackupRestore
        )
    }

    /**
     * Validate factory configuration and all dependencies
     */
    fun validateConfiguration(): FactoryValidationResult {
        val issues = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        // Validate core repositories
        try {
            userPreferencesRepository.javaClass
        } catch (e: Exception) {
            issues.add("UserPreferencesRepository is not properly initialized")
        }

        try {
            sleepSessionRepository.javaClass
        } catch (e: Exception) {
            issues.add("SleepSessionRepository is not properly initialized")
        }

        try {
            insightRepository.javaClass
        } catch (e: Exception) {
            issues.add("InsightRepository is not properly initialized")
        }

        // Validate AI components (if enabled)
        if (enableAIConfiguration) {
            if (aiResponseParser == null) {
                warnings.add("AI configuration enabled but AIResponseParser is null - some features will be limited")
            }
            if (parsingAnalytics == null) {
                warnings.add("AI configuration enabled but ParsingPerformanceAnalytics is null")
            }
        }

        // Validate privacy controls (if enabled)
        if (enablePrivacyControls && securityUtils == null) {
            warnings.add("Privacy controls enabled but SecurityUtils is null - security features will be limited")
        }

        // Validate backup/restore (if enabled)
        if (enableBackupRestore && backupManager == null) {
            warnings.add("Backup/restore enabled but BackupManager is null - backup features will be disabled")
        }

        // Validate configuration parameters
        if (settingsConfiguration.maxCacheSize <= 0) {
            issues.add("Settings cache size must be positive")
        }

        if (settingsConfiguration.syncIntervalMs <= 0) {
            issues.add("Settings sync interval must be positive")
        }

        return FactoryValidationResult(
            isValid = issues.isEmpty(),
            issues = issues,
            configurationSummary = mapOf(
                "enableAdvancedSettings" to enableAdvancedSettings,
                "enableAIConfiguration" to enableAIConfiguration,
                "enablePerformanceSettings" to enablePerformanceSettings,
                "enablePrivacyControls" to enablePrivacyControls,
                "enableBackupRestore" to enableBackupRestore,
                "warnings" to warnings
            )
        )
    }

    /**
     * Get comprehensive factory configuration summary
     */
    fun getConfigurationSummary(): Map<String, Any> {
        return mapOf(
            "factoryType" to "SettingsViewModelFactory",
            "version" to "1.0.0",
            "enabledFeatures" to mapOf(
                "advancedSettings" to enableAdvancedSettings,
                "aiConfiguration" to enableAIConfiguration,
                "performanceSettings" to enablePerformanceSettings,
                "privacyControls" to enablePrivacyControls,
                "backupRestore" to enableBackupRestore
            ),
            "availableComponents" to mapOf(
                "aiResponseParser" to (aiResponseParser != null),
                "parsingAnalytics" to (parsingAnalytics != null),
                "securityUtils" to (securityUtils != null),
                "backupManager" to (backupManager != null)
            ),
            "settingsConfiguration" to settingsConfiguration.toMap(),
            "dependenciesCount" to getDependenciesCount(),
            "memoryFootprint" to getEstimatedMemoryFootprint()
        )
    }

    private fun getDependenciesCount(): Int {
        var count = 3 // Core repositories
        if (aiResponseParser != null) count++
        if (parsingAnalytics != null) count++
        if (securityUtils != null) count++
        if (backupManager != null) count++
        return count
    }

    private fun getEstimatedMemoryFootprint(): String {
        val baseSize = 80 // KB base factory size
        val cacheSize = (settingsConfiguration.maxCacheSize * 1) // KB estimated per cached setting
        val featuresSize = (getEnabledFeaturesCount() * 20) // KB per enabled feature
        val totalKB = baseSize + cacheSize + featuresSize

        return when {
            totalKB < 1024 -> "${totalKB}KB"
            else -> "${totalKB / 1024}MB"
        }
    }

    private fun getEnabledFeaturesCount(): Int {
        return listOf(
            enableAdvancedSettings,
            enableAIConfiguration,
            enablePerformanceSettings,
            enablePrivacyControls,
            enableBackupRestore
        ).count { it }
    }
}

// ========== SETTINGS CONFIGURATION ==========

/**
 * Comprehensive configuration class for settings functionality
 */
data class SettingsConfiguration(
    val maxCacheSize: Int = SETTINGS_CACHE_SIZE,
    val syncIntervalMs: Long = SETTINGS_SYNC_INTERVAL,
    val enablePreferenceCaching: Boolean = true,
    val enableCloudSync: Boolean = false,
    val enableCrossDeviceSync: Boolean = false,
    val enableDataEncryption: Boolean = true,
    val enableAnonymization: Boolean = true,
    val enableDynamicTheming: Boolean = true,
    val enableAccessibilityFeatures: Boolean = true,
    val enableSmartNotifications: Boolean = true,
    val enableStrictValidation: Boolean = true,
    val performanceConfiguration: PerformanceConfiguration = PerformanceConfiguration.default(),
    val themeConfiguration: ThemeConfiguration = ThemeConfiguration.default(),
    val notificationConfiguration: NotificationConfiguration = NotificationConfiguration.default(),
    val validationRules: SettingsValidationRules = SettingsValidationRules.default(),
    val exportFormats: List<ExportFormat> = listOf(ExportFormat.JSON, ExportFormat.CSV, ExportFormat.PDF)
) {
    companion object {
        fun default() = SettingsConfiguration()

        fun fromPreferences(prefs: SharedPreferences): SettingsConfiguration {
            return SettingsConfiguration(
                maxCacheSize = prefs.getInt("settings_cache_size", SETTINGS_CACHE_SIZE),
                syncIntervalMs = prefs.getLong("settings_sync_interval", SETTINGS_SYNC_INTERVAL),
                enablePreferenceCaching = prefs.getBoolean("settings_enable_caching", true),
                enableCloudSync = prefs.getBoolean("settings_enable_cloud_sync", false),
                enableDataEncryption = prefs.getBoolean("settings_enable_encryption", true),
                enableAnonymization = prefs.getBoolean("settings_enable_anonymization", true),
                enableDynamicTheming = prefs.getBoolean("settings_enable_dynamic_theme", true),
                enableSmartNotifications = prefs.getBoolean("settings_enable_smart_notifications", true)
            )
        }

        fun createTestConfiguration(): SettingsConfiguration {
            return SettingsConfiguration(
                maxCacheSize = 10,
                syncIntervalMs = 1000L,
                enablePreferenceCaching = false,
                enableCloudSync = false,
                enableCrossDeviceSync = false,
                enableDataEncryption = false,
                enableAnonymization = false,
                enableDynamicTheming = false,
                enableSmartNotifications = false,
                enableStrictValidation = false
            )
        }

        fun createAIFocusedConfiguration(): SettingsConfiguration {
            return SettingsConfiguration(
                maxCacheSize = 30,
                enablePreferenceCaching = true,
                enableCloudSync = false,
                enableDataEncryption = true,
                performanceConfiguration = PerformanceConfiguration(
                    enablePerformanceMonitoring = true,
                    enableAnalytics = true,
                    sampleRate = 0.1f
                )
            )
        }
    }

    fun toMap(): Map<String, Any> {
        return mapOf(
            "maxCacheSize" to maxCacheSize,
            "syncIntervalMs" to syncIntervalMs,
            "enablePreferenceCaching" to enablePreferenceCaching,
            "enableCloudSync" to enableCloudSync,
            "enableDataEncryption" to enableDataEncryption,
            "enableDynamicTheming" to enableDynamicTheming,
            "enableSmartNotifications" to enableSmartNotifications
        )
    }
}

// ========== SUPPORTING CONFIGURATION CLASSES ==========

data class PerformanceConfiguration(
    val enablePerformanceMonitoring: Boolean = true,
    val enableAnalytics: Boolean = true,
    val sampleRate: Float = PERFORMANCE_SAMPLE_RATE,
    val enableMemoryOptimization: Boolean = true,
    val enableBatteryOptimization: Boolean = true
) {
    companion object {
        fun default() = PerformanceConfiguration()
    }
}

data class ThemeConfiguration(
    val defaultTheme: String = "system",
    val enableCustomColors: Boolean = true,
    val enableAnimations: Boolean = true,
    val enableReducedMotion: Boolean = false,
    val fontScale: Float = 1.0f
) {
    companion object {
        fun default() = ThemeConfiguration()
    }
}

data class NotificationConfiguration(
    val enableInsightNotifications: Boolean = true,
    val enableProgressNotifications: Boolean = true,
    val enableReminderNotifications: Boolean = true,
    val quietHoursStart: Int = 22, // 10 PM
    val quietHoursEnd: Int = 7,    // 7 AM
    val notificationPriority: String = "normal"
) {
    companion object {
        fun default() = NotificationConfiguration()
    }
}

data class SettingsValidationRules(
    val requireValidEmail: Boolean = false,
    val requireSecurePasswords: Boolean = true,
    val validateDataRanges: Boolean = true,
    val enforcePrivacyPolicies: Boolean = true,
    val maxExportFileSizeMB: Long = MAX_EXPORT_FILE_SIZE_MB
) {
    companion object {
        fun default() = SettingsValidationRules()
    }
}

// ========== COMPONENT PLACEHOLDER CLASSES ==========

class SettingsPreferenceManager(
    private val repository: UserPreferencesRepository,
    private val sharedPreferences: SharedPreferences,
    private val configuration: SettingsConfiguration,
    private val performanceMonitor: AIPerformanceMonitor,
    private val enableCaching: Boolean,
    private val cacheSize: Int
)

class SettingsAIConfigurationManager(
    private val aiConfiguration: AIConfiguration,
    private val parser: AIResponseParser?,
    private val analytics: ParsingPerformanceAnalytics?,
    private val performanceMonitor: AIPerformanceMonitor,
    private val defaultConfidenceThreshold: Float,
    private val enableAdvancedAISettings: Boolean
)

class SettingsPrivacyControlsManager(
    private val securityUtils: SecurityUtils?,
    private val performanceMonitor: AIPerformanceMonitor,
    private val dataRetentionDays: Long,
    private val enableDataEncryption: Boolean,
    private val enableAnonymization: Boolean
)

class SettingsBackupRestoreManager(
    private val backupManager: BackupManager,
    private val sleepSessionRepository: SleepSessionRepository,
    private val insightRepository: InsightRepository,
    private val performanceMonitor: AIPerformanceMonitor,
    private val maxBackupSizeMB: Long,
    private val retentionDays: Long,
    private val minBackupIntervalHours: Long
)

class SettingsPerformanceManager(
    private val performanceMonitor: AIPerformanceMonitor,
    private val parsingAnalytics: ParsingPerformanceAnalytics?,
    private val configuration: PerformanceConfiguration,
    private val enablePerformanceMonitoring: Boolean,
    private val enableAnalytics: Boolean,
    private val sampleRate: Float
)

class SettingsThemeAppearanceManager(
    private val context: Context,
    private val sharedPreferences: SharedPreferences,
    private val configuration: ThemeConfiguration,
    private val enableDynamicTheming: Boolean,
    private val enableAccessibilityFeatures: Boolean
)

class SettingsNotificationManager(
    private val context: Context,
    private val sharedPreferences: SharedPreferences,
    private val configuration: NotificationConfiguration,
    private val enableSmartNotifications: Boolean,
    private val performanceMonitor: AIPerformanceMonitor
)

class SettingsDataExportManager(
    private val context: Context,
    private val sleepSessionRepository: SleepSessionRepository,
    private val insightRepository: InsightRepository,
    private val performanceMonitor: AIPerformanceMonitor,
    private val supportedFormats: List<ExportFormat>,
    private val maxFileSizeMB: Long
)

class SettingsSyncManager(
    private val preferenceManager: SettingsPreferenceManager,
    private val performanceMonitor: AIPerformanceMonitor,
    private val syncInterval: Long,
    private val enableCloudSync: Boolean,
    private val enableCrossDeviceSync: Boolean
)

class SettingsValidationManager(
    private val configuration: SettingsConfiguration,
    private val performanceMonitor: AIPerformanceMonitor,
    private val validationRules: SettingsValidationRules,
    private val enableStrictValidation: Boolean
)

class SettingsAnalyticsTracker(
    private val performanceMonitor: AIPerformanceMonitor,
    private val parsingAnalytics: ParsingPerformanceAnalytics?,
    private val enabled: Boolean
)