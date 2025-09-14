package com.example.somniai.ui.theme.models


import android.content.Context
import android.content.SharedPreferences
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.somniai.ai.AIPerformanceMonitor
import com.example.somniai.ai.AIResponseParser
import com.example.somniai.ai.ParsingPerformanceAnalytics
import com.example.somniai.data.repository.SleepSessionRepository
import com.example.somniai.data.repository.InsightRepository
import com.example.somniai.data.repository.UserPreferencesRepository
import com.example.somniai.data.*
import com.example.somniai.utils.TimeUtils
import com.example.somniai.utils.ValidationUtils
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import java.util.concurrent.TimeUnit

/**
 * ViewModelFactory for HistoryViewModel in SomniAI
 *
 * Sophisticated factory providing comprehensive dependency injection for advanced
 * sleep session history management, AI-powered insights processing, data filtering,
 * pagination optimization, and performance monitoring. Integrates seamlessly with
 * the SomniAI analytics pipeline and session management ecosystem.
 */
class HistoryViewModelFactory(
    private val context: Context,
    private val sleepSessionRepository: SleepSessionRepository,
    private val insightRepository: InsightRepository,
    private val userPreferencesRepository: UserPreferencesRepository,
    private val sharedPreferences: SharedPreferences,
    private val aiPerformanceMonitor: AIPerformanceMonitor = AIPerformanceMonitor,
    private val aiResponseParser: AIResponseParser? = null,
    private val parsingAnalytics: ParsingPerformanceAnalytics? = null,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO,
    private val mainDispatcher: CoroutineDispatcher = Dispatchers.Main,
    private val historyConfiguration: HistoryConfiguration = HistoryConfiguration.default(),
    private val performanceTrackingEnabled: Boolean = true,
    private val advancedFilteringEnabled: Boolean = true,
    private val aiInsightProcessingEnabled: Boolean = true
) : ViewModelProvider.Factory {

    companion object {
        private const val TAG = "HistoryViewModelFactory"

        // History configuration constants
        private const val DEFAULT_PAGE_SIZE = 20
        private const val DEFAULT_CACHE_SIZE = 200
        private const val DEFAULT_PREFETCH_DISTANCE = 10
        private const val HISTORY_CACHE_TTL_HOURS = 2L
        private const val MAX_HISTORY_ITEMS = 1000

        // Performance optimization settings
        private const val BATCH_PROCESSING_SIZE = 50
        private const val BACKGROUND_SYNC_INTERVAL = 15 * 60 * 1000L // 15 minutes
        private const val DATA_QUALITY_THRESHOLD = 0.7f

        // Filter and search settings
        private const val SEARCH_DEBOUNCE_MS = 300L
        private const val MAX_SEARCH_RESULTS = 100
        private const val DATE_RANGE_LIMIT_DAYS = 365L

        /**
         * Create factory with standard dependencies for production use
         */
        fun create(
            context: Context,
            sleepSessionRepository: SleepSessionRepository,
            insightRepository: InsightRepository,
            userPreferencesRepository: UserPreferencesRepository
        ): HistoryViewModelFactory {
            val sharedPrefs = context.getSharedPreferences("somniai_history_prefs", Context.MODE_PRIVATE)

            return HistoryViewModelFactory(
                context = context,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = insightRepository,
                userPreferencesRepository = userPreferencesRepository,
                sharedPreferences = sharedPrefs,
                historyConfiguration = HistoryConfiguration.fromPreferences(sharedPrefs)
            )
        }

        /**
         * Create factory with enhanced AI processing for advanced history analysis
         */
        fun createWithAIProcessing(
            context: Context,
            sleepSessionRepository: SleepSessionRepository,
            insightRepository: InsightRepository,
            userPreferencesRepository: UserPreferencesRepository,
            aiResponseParser: AIResponseParser,
            parsingAnalytics: ParsingPerformanceAnalytics
        ): HistoryViewModelFactory {
            val sharedPrefs = context.getSharedPreferences("somniai_history_prefs", Context.MODE_PRIVATE)

            return HistoryViewModelFactory(
                context = context,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = insightRepository,
                userPreferencesRepository = userPreferencesRepository,
                sharedPreferences = sharedPrefs,
                aiResponseParser = aiResponseParser,
                parsingAnalytics = parsingAnalytics,
                historyConfiguration = HistoryConfiguration.fromPreferences(sharedPrefs),
                aiInsightProcessingEnabled = true,
                advancedFilteringEnabled = true
            )
        }

        /**
         * Create factory optimized for testing scenarios
         */
        fun createForTesting(
            context: Context,
            sleepSessionRepository: SleepSessionRepository,
            insightRepository: InsightRepository,
            userPreferencesRepository: UserPreferencesRepository,
            testDispatcher: CoroutineDispatcher,
            enablePerformanceTracking: Boolean = false
        ): HistoryViewModelFactory {
            val sharedPrefs = context.getSharedPreferences("test_history_prefs", Context.MODE_PRIVATE)

            return HistoryViewModelFactory(
                context = context,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = insightRepository,
                userPreferencesRepository = userPreferencesRepository,
                sharedPreferences = sharedPrefs,
                ioDispatcher = testDispatcher,
                mainDispatcher = testDispatcher,
                performanceTrackingEnabled = enablePerformanceTracking,
                advancedFilteringEnabled = false,
                aiInsightProcessingEnabled = false,
                historyConfiguration = HistoryConfiguration.createTestConfiguration()
            )
        }

        /**
         * Create factory with minimal dependencies for lightweight usage
         */
        fun createMinimal(
            context: Context,
            sleepSessionRepository: SleepSessionRepository
        ): HistoryViewModelFactory {
            // Create minimal repositories for basic functionality
            val minimalInsightRepo = object : InsightRepository {
                override suspend fun getAllInsights(): List<SleepInsight> = emptyList()
                override suspend fun getInsightsForSession(sessionId: Long): List<SleepInsight> = emptyList()
                override suspend fun insertInsight(insight: SleepInsight): Long = -1L
                override suspend fun updateInsight(insight: SleepInsight) {}
                override suspend fun deleteInsight(insightId: String) {}
            }

            val minimalPrefsRepo = object : UserPreferencesRepository {
                override suspend fun getUserPreferences(): UserPreferences = UserPreferences.default()
                override suspend fun updatePreference(key: String, value: Any) {}
                override suspend fun getPreference(key: String, defaultValue: Any): Any = defaultValue
            }

            val sharedPrefs = context.getSharedPreferences("somniai_history_minimal", Context.MODE_PRIVATE)

            return HistoryViewModelFactory(
                context = context,
                sleepSessionRepository = sleepSessionRepository,
                insightRepository = minimalInsightRepo,
                userPreferencesRepository = minimalPrefsRepo,
                sharedPreferences = sharedPrefs,
                performanceTrackingEnabled = false,
                advancedFilteringEnabled = false,
                aiInsightProcessingEnabled = false,
                historyConfiguration = HistoryConfiguration.createMinimalConfiguration()
            )
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return when {
            modelClass.isAssignableFrom(HistoryViewModel::class.java) -> {
                createHistoryViewModel() as T
            }
            else -> {
                throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
            }
        }
    }

    /**
     * Create HistoryViewModel with comprehensive dependency injection
     */
    private fun createHistoryViewModel(): HistoryViewModel {
        // Initialize session data manager
        val sessionDataManager = SessionDataManager(
            repository = sleepSessionRepository,
            configuration = historyConfiguration,
            performanceMonitor = aiPerformanceMonitor,
            enableCaching = historyConfiguration.enableCaching,
            cacheSize = historyConfiguration.cacheSize,
            cacheTtl = historyConfiguration.cacheTtlMs
        )

        // Initialize advanced filtering engine
        val filteringEngine = HistoryFilteringEngine(
            enabled = advancedFilteringEnabled,
            configuration = historyConfiguration.filterConfiguration,
            performanceMonitor = aiPerformanceMonitor,
            maxResults = MAX_SEARCH_RESULTS,
            debounceMs = SEARCH_DEBOUNCE_MS
        )

        // Initialize pagination controller
        val paginationController = HistoryPaginationController(
            pageSize = historyConfiguration.pageSize,
            prefetchDistance = historyConfiguration.prefetchDistance,
            maxItems = MAX_HISTORY_ITEMS,
            performanceMonitor = aiPerformanceMonitor
        )

        // Initialize AI insights processor (if enabled)
        val aiInsightsProcessor = if (aiInsightProcessingEnabled && aiResponseParser != null) {
            HistoryAIInsightsProcessor(
                parser = aiResponseParser,
                analytics = parsingAnalytics,
                performanceMonitor = aiPerformanceMonitor,
                enableBatchProcessing = historyConfiguration.enableBatchProcessing,
                batchSize = BATCH_PROCESSING_SIZE
            )
        } else null

        // Initialize data quality analyzer
        val dataQualityAnalyzer = HistoryDataQualityAnalyzer(
            qualityThreshold = DATA_QUALITY_THRESHOLD,
            performanceMonitor = aiPerformanceMonitor,
            enableValidation = historyConfiguration.enableDataValidation,
            validationRules = historyConfiguration.validationRules
        )

        // Initialize background sync manager
        val backgroundSyncManager = HistoryBackgroundSyncManager(
            repository = sleepSessionRepository,
            syncInterval = BACKGROUND_SYNC_INTERVAL,
            performanceMonitor = aiPerformanceMonitor,
            enableBackgroundSync = historyConfiguration.enableBackgroundSync
        )

        // Initialize export/import manager
        val exportImportManager = HistoryExportImportManager(
            context = context,
            repository = sleepSessionRepository,
            performanceMonitor = aiPerformanceMonitor,
            supportedFormats = historyConfiguration.exportFormats
        )

        // Initialize search and analytics manager
        val searchAnalyticsManager = HistorySearchAnalyticsManager(
            performanceMonitor = aiPerformanceMonitor,
            parsingAnalytics = parsingAnalytics,
            enabled = performanceTrackingEnabled && advancedFilteringEnabled
        )

        // Initialize performance tracker for history operations
        val performanceTracker = HistoryPerformanceTracker(
            monitor = aiPerformanceMonitor,
            enabled = performanceTrackingEnabled,
            trackingConfiguration = historyConfiguration.performanceTracking
        )

        // Initialize cache manager for session data
        val cacheManager = HistoryCacheManager(
            context = context,
            maxCacheSize = historyConfiguration.cacheSize,
            cacheTtl = historyConfiguration.cacheTtlMs,
            enableSmartCaching = historyConfiguration.enableSmartCaching
        )

        // Initialize configuration manager
        val configurationManager = HistoryConfigurationManager(
            sharedPreferences = sharedPreferences,
            defaultConfiguration = historyConfiguration,
            enableDynamicConfiguration = historyConfiguration.enableDynamicConfiguration
        )

        // Return fully configured ViewModel
        return HistoryViewModel(
            // Core repositories
            sleepSessionRepository = sleepSessionRepository,
            insightRepository = insightRepository,
            userPreferencesRepository = userPreferencesRepository,

            // History management components
            sessionDataManager = sessionDataManager,
            filteringEngine = filteringEngine,
            paginationController = paginationController,
            aiInsightsProcessor = aiInsightsProcessor,

            // Data quality and sync
            dataQualityAnalyzer = dataQualityAnalyzer,
            backgroundSyncManager = backgroundSyncManager,
            exportImportManager = exportImportManager,

            // Analytics and performance
            searchAnalyticsManager = searchAnalyticsManager,
            performanceTracker = performanceTracker,

            // Utility components
            cacheManager = cacheManager,
            configurationManager = configurationManager,

            // System dependencies
            context = context.applicationContext,
            sharedPreferences = sharedPreferences,
            ioDispatcher = ioDispatcher,
            mainDispatcher = mainDispatcher,

            // Configuration
            historyConfiguration = historyConfiguration,
            performanceTrackingEnabled = performanceTrackingEnabled,
            advancedFilteringEnabled = advancedFilteringEnabled,
            aiInsightProcessingEnabled = aiInsightProcessingEnabled
        )
    }

    /**
     * Validate factory configuration and dependencies
     */
    fun validateConfiguration(): FactoryValidationResult {
        val issues = mutableListOf<String>()

        // Validate core repositories
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

        try {
            userPreferencesRepository.javaClass
        } catch (e: Exception) {
            issues.add("UserPreferencesRepository is not properly initialized")
        }

        // Validate AI processing components (if enabled)
        if (aiInsightProcessingEnabled) {
            if (aiResponseParser == null) {
                issues.add("AI insight processing enabled but AIResponseParser is null")
            }
            if (parsingAnalytics == null) {
                issues.add("AI insight processing enabled but ParsingPerformanceAnalytics is null")
            }
        }

        // Validate configuration parameters
        if (historyConfiguration.pageSize <= 0) {
            issues.add("Page size must be positive")
        }

        if (historyConfiguration.cacheSize <= 0) {
            issues.add("Cache size must be positive")
        }

        if (historyConfiguration.cacheTtlMs <= 0) {
            issues.add("Cache TTL must be positive")
        }

        if (historyConfiguration.prefetchDistance < 0) {
            issues.add("Prefetch distance cannot be negative")
        }

        return FactoryValidationResult(
            isValid = issues.isEmpty(),
            issues = issues,
            configurationSummary = mapOf(
                "pageSize" to historyConfiguration.pageSize,
                "cacheSize" to historyConfiguration.cacheSize,
                "aiProcessingEnabled" to aiInsightProcessingEnabled,
                "advancedFilteringEnabled" to advancedFilteringEnabled,
                "performanceTrackingEnabled" to performanceTrackingEnabled
            )
        )
    }

    /**
     * Get factory configuration summary for debugging and monitoring
     */
    fun getConfigurationSummary(): Map<String, Any> {
        return mapOf(
            "factoryType" to "HistoryViewModelFactory",
            "version" to "1.0.0",
            "aiInsightProcessingEnabled" to aiInsightProcessingEnabled,
            "advancedFilteringEnabled" to advancedFilteringEnabled,
            "performanceTrackingEnabled" to performanceTrackingEnabled,
            "aiProcessingAvailable" to (aiResponseParser != null),
            "historyConfiguration" to historyConfiguration.toMap(),
            "dependenciesCount" to getDependenciesCount(),
            "memoryFootprint" to getEstimatedMemoryFootprint()
        )
    }

    private fun getDependenciesCount(): Int {
        var count = 3 // Core repositories
        if (aiResponseParser != null) count++
        if (parsingAnalytics != null) count++
        return count
    }

    private fun getEstimatedMemoryFootprint(): String {
        val baseSize = 100 // KB base factory size
        val cacheSize = (historyConfiguration.cacheSize * 2) // KB estimated per cached item
        val totalKB = baseSize + cacheSize

        return when {
            totalKB < 1024 -> "${totalKB}KB"
            else -> "${totalKB / 1024}MB"
        }
    }
}

// ========== HISTORY CONFIGURATION ==========

/**
 * Configuration class for history functionality
 */
data class HistoryConfiguration(
    val pageSize: Int = DEFAULT_PAGE_SIZE,
    val cacheSize: Int = DEFAULT_CACHE_SIZE,
    val prefetchDistance: Int = DEFAULT_PREFETCH_DISTANCE,
    val cacheTtlMs: Long = TimeUnit.HOURS.toMillis(HISTORY_CACHE_TTL_HOURS),
    val enableCaching: Boolean = true,
    val enableBatchProcessing: Boolean = true,
    val enableBackgroundSync: Boolean = true,
    val enableDataValidation: Boolean = true,
    val enableSmartCaching: Boolean = true,
    val enableDynamicConfiguration: Boolean = true,
    val filterConfiguration: FilterConfiguration = FilterConfiguration.default(),
    val performanceTracking: PerformanceTrackingConfiguration = PerformanceTrackingConfiguration.default(),
    val validationRules: ValidationRules = ValidationRules.default(),
    val exportFormats: List<ExportFormat> = listOf(ExportFormat.JSON, ExportFormat.CSV)
) {
    companion object {
        fun default() = HistoryConfiguration()

        fun fromPreferences(prefs: SharedPreferences): HistoryConfiguration {
            return HistoryConfiguration(
                pageSize = prefs.getInt("history_page_size", DEFAULT_PAGE_SIZE),
                cacheSize = prefs.getInt("history_cache_size", DEFAULT_CACHE_SIZE),
                prefetchDistance = prefs.getInt("history_prefetch_distance", DEFAULT_PREFETCH_DISTANCE),
                cacheTtlMs = prefs.getLong("history_cache_ttl", TimeUnit.HOURS.toMillis(HISTORY_CACHE_TTL_HOURS)),
                enableCaching = prefs.getBoolean("history_enable_caching", true),
                enableBatchProcessing = prefs.getBoolean("history_enable_batch", true),
                enableBackgroundSync = prefs.getBoolean("history_enable_sync", true),
                enableDataValidation = prefs.getBoolean("history_enable_validation", true)
            )
        }

        fun createTestConfiguration(): HistoryConfiguration {
            return HistoryConfiguration(
                pageSize = 10,
                cacheSize = 50,
                prefetchDistance = 3,
                cacheTtlMs = 1000L,
                enableCaching = false,
                enableBatchProcessing = false,
                enableBackgroundSync = false,
                enableDataValidation = false,
                enableSmartCaching = false,
                enableDynamicConfiguration = false
            )
        }

        fun createMinimalConfiguration(): HistoryConfiguration {
            return HistoryConfiguration(
                pageSize = 20,
                cacheSize = 20,
                prefetchDistance = 5,
                enableCaching = true,
                enableBatchProcessing = false,
                enableBackgroundSync = false,
                enableDataValidation = false,
                enableSmartCaching = false
            )
        }
    }

    fun toMap(): Map<String, Any> {
        return mapOf(
            "pageSize" to pageSize,
            "cacheSize" to cacheSize,
            "prefetchDistance" to prefetchDistance,
            "cacheTtlMs" to cacheTtlMs,
            "enableCaching" to enableCaching,
            "enableBatchProcessing" to enableBatchProcessing,
            "enableBackgroundSync" to enableBackgroundSync,
            "enableDataValidation" to enableDataValidation,
            "enableSmartCaching" to enableSmartCaching
        )
    }
}

// ========== SUPPORTING CONFIGURATION CLASSES ==========

data class FilterConfiguration(
    val enableDateRangeFilter: Boolean = true,
    val enableQualityFilter: Boolean = true,
    val enableDurationFilter: Boolean = true,
    val enableTextSearch: Boolean = true,
    val maxSearchResults: Int = MAX_SEARCH_RESULTS,
    val searchDebounceMs: Long = SEARCH_DEBOUNCE_MS,
    val dateRangeLimitDays: Long = DATE_RANGE_LIMIT_DAYS
) {
    companion object {
        fun default() = FilterConfiguration()
    }
}

data class PerformanceTrackingConfiguration(
    val trackLoadTimes: Boolean = true,
    val trackSearchPerformance: Boolean = true,
    val trackCacheHitRates: Boolean = true,
    val trackMemoryUsage: Boolean = true,
    val sampleRate: Float = 0.1f // 10% sampling
) {
    companion object {
        fun default() = PerformanceTrackingConfiguration()
    }
}

data class ValidationRules(
    val minSessionDuration: Long = 30 * 60 * 1000L, // 30 minutes
    val maxSessionDuration: Long = 24 * 60 * 60 * 1000L, // 24 hours
    val requiredFields: List<String> = listOf("startTime", "duration", "sleepEfficiency"),
    val qualityScoreRange: ClosedFloatingPointRange<Float> = 0f..10f
) {
    companion object {
        fun default() = ValidationRules()
    }
}

enum class ExportFormat {
    JSON, CSV, XML, PDF
}

// ========== COMPONENT PLACEHOLDER CLASSES ==========
// These would be fully implemented based on specific requirements

class SessionDataManager(
    private val repository: SleepSessionRepository,
    private val configuration: HistoryConfiguration,
    private val performanceMonitor: AIPerformanceMonitor,
    private val enableCaching: Boolean,
    private val cacheSize: Int,
    private val cacheTtl: Long
)

class HistoryFilteringEngine(
    private val enabled: Boolean,
    private val configuration: FilterConfiguration,
    private val performanceMonitor: AIPerformanceMonitor,
    private val maxResults: Int,
    private val debounceMs: Long
)

class HistoryPaginationController(
    private val pageSize: Int,
    private val prefetchDistance: Int,
    private val maxItems: Int,
    private val performanceMonitor: AIPerformanceMonitor
)

class HistoryAIInsightsProcessor(
    private val parser: AIResponseParser,
    private val analytics: ParsingPerformanceAnalytics?,
    private val performanceMonitor: AIPerformanceMonitor,
    private val enableBatchProcessing: Boolean,
    private val batchSize: Int
)

class HistoryDataQualityAnalyzer(
    private val qualityThreshold: Float,
    private val performanceMonitor: AIPerformanceMonitor,
    private val enableValidation: Boolean,
    private val validationRules: ValidationRules
)

class HistoryBackgroundSyncManager(
    private val repository: SleepSessionRepository,
    private val syncInterval: Long,
    private val performanceMonitor: AIPerformanceMonitor,
    private val enableBackgroundSync: Boolean
)

class HistoryExportImportManager(
    private val context: Context,
    private val repository: SleepSessionRepository,
    private val performanceMonitor: AIPerformanceMonitor,
    private val supportedFormats: List<ExportFormat>
)

class HistorySearchAnalyticsManager(
    private val performanceMonitor: AIPerformanceMonitor,
    private val parsingAnalytics: ParsingPerformanceAnalytics?,
    private val enabled: Boolean
)

class HistoryPerformanceTracker(
    private val monitor: AIPerformanceMonitor,
    private val enabled: Boolean,
    private val trackingConfiguration: PerformanceTrackingConfiguration
)

class HistoryCacheManager(
    private val context: Context,
    private val maxCacheSize: Int,
    private val cacheTtl: Long,
    private val enableSmartCaching: Boolean
)

class HistoryConfigurationManager(
    private val sharedPreferences: SharedPreferences,
    private val defaultConfiguration: HistoryConfiguration,
    private val enableDynamicConfiguration: Boolean
)