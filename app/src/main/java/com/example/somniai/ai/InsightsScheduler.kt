package com.example.somniai.ai

import android.content.Context
import android.content.SharedPreferences
import android.os.BatteryManager
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.work.*
import com.example.somniai.data.*
import com.example.somniai.repository.SleepRepository
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*
import kotlin.random.Random

/**
 * Advanced AI-powered insights generation scheduler
 *
 * Features:
 * - Intelligent scheduling based on user sleep patterns and engagement
 * - Machine learning-driven optimization for insight timing
 * - Comprehensive analytics integration with priority scoring
 * - Advanced retry mechanisms with circuit breaker patterns
 * - Performance monitoring and health diagnostics
 * - Battery and resource optimization
 * - Context-aware scheduling with environmental factors
 * - Personalized insight delivery optimization
 * - A/B testing support for scheduling strategies
 * - Comprehensive error handling and recovery
 * - Real-time scheduler health monitoring
 * - Advanced work prioritization and batching
 * - Integration with comprehensive analytics models
 */
class InsightsScheduler(
    private val context: Context,
    private val repository: SleepRepository,
    private val aiInsightsEngine: AIInsightsEngine,
    private val preferences: SharedPreferences,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "InsightsScheduler"

        // Scheduling configuration
        private const val MIN_INTERVAL_BETWEEN_INSIGHTS_MINUTES = 30L
        private const val MAX_DAILY_INSIGHTS = 15
        private const val MAX_WEEKLY_INSIGHTS = 50
        private const val INSIGHT_COOLDOWN_PERIOD_HOURS = 2L

        // Machine learning parameters
        private const val ENGAGEMENT_LEARNING_WINDOW_DAYS = 30
        private const val OPTIMAL_TIMING_LEARNING_SAMPLES = 100
        private const val PERSONALIZATION_CONFIDENCE_THRESHOLD = 0.7f

        // Performance monitoring
        private const val HEALTH_CHECK_INTERVAL_MINUTES = 15L
        private const val PERFORMANCE_METRICS_RETENTION_DAYS = 7
        private const val ERROR_RATE_THRESHOLD = 0.15f
        private const val LATENCY_THRESHOLD_MS = 5000L

        // Battery optimization
        private const val LOW_BATTERY_THRESHOLD = 20
        private const val CRITICAL_BATTERY_THRESHOLD = 10
        private const val BATTERY_OPTIMIZED_SCHEDULING_THRESHOLD = 30

        // Work tags and identifiers
        private const val WORK_TAG_PREFIX = "somniai_insights"
        private const val WORK_TAG_POST_SESSION = "${WORK_TAG_PREFIX}_post_session"
        private const val WORK_TAG_DAILY_ANALYSIS = "${WORK_TAG_PREFIX}_daily"
        private const val WORK_TAG_WEEKLY_ANALYSIS = "${WORK_TAG_PREFIX}_weekly"
        private const val WORK_TAG_MONTHLY_ANALYSIS = "${WORK_TAG_PREFIX}_monthly"
        private const val WORK_TAG_PERSONALIZED = "${WORK_TAG_PREFIX}_personalized"
        private const val WORK_TAG_PREDICTIVE = "${WORK_TAG_PREFIX}_predictive"
        private const val WORK_TAG_EMERGENCY = "${WORK_TAG_PREFIX}_emergency"
        private const val WORK_TAG_MAINTENANCE = "${WORK_TAG_PREFIX}_maintenance"
        private const val WORK_TAG_HEALTH_CHECK = "${WORK_TAG_PREFIX}_health"

        // Preference keys
        private const val PREF_SCHEDULER_CONFIG = "scheduler_config_v2"
        private const val PREF_USER_ENGAGEMENT_DATA = "user_engagement_data"
        private const val PREF_OPTIMAL_TIMING_DATA = "optimal_timing_data"
        private const val PREF_PERFORMANCE_METRICS = "performance_metrics"
        private const val PREF_ERROR_TRACKING = "error_tracking"
        private const val PREF_SCHEDULER_HEALTH = "scheduler_health"
        private const val PREF_A_B_TEST_CONFIG = "ab_test_config"
        private const val PREF_PERSONALIZATION_DATA = "personalization_data"
        private const val PREF_INSIGHT_HISTORY = "insight_history"
        private const val PREF_LAST_HEALTH_CHECK = "last_health_check"
        private const val PREF_CIRCUIT_BREAKER_STATE = "circuit_breaker_state"

        // Circuit breaker configuration
        private const val CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
        private const val CIRCUIT_BREAKER_TIMEOUT_MINUTES = 30L
        private const val CIRCUIT_BREAKER_RESET_TIMEOUT_MINUTES = 10L
    }

    // Core dependencies
    private val workManager = WorkManager.getInstance(context)
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
    private val scope = CoroutineScope(dispatcher + SupervisorJob())

    // State management
    private val _schedulerState = MutableStateFlow<SchedulerState>(SchedulerState.INITIALIZING)
    val schedulerState: StateFlow<SchedulerState> = _schedulerState.asStateFlow()

    private val _schedulerHealth = MutableStateFlow<SchedulerHealth>(SchedulerHealth.UNKNOWN)
    val schedulerHealth: StateFlow<SchedulerHealth> = _schedulerHealth.asStateFlow()

    private val _performanceMetrics = MutableLiveData<SchedulerPerformanceMetrics>()
    val performanceMetrics: LiveData<SchedulerPerformanceMetrics> = _performanceMetrics

    // Advanced features
    private val engagementTracker = UserEngagementTracker()
    private val timingOptimizer = OptimalTimingAnalyzer()
    private val priorityCalculator = InsightPriorityCalculator()
    private val circuitBreaker = CircuitBreaker()
    private val performanceMonitor = PerformanceMonitor()
    private val batteryOptimizer = BatteryOptimizer()

    // Caching and optimization
    private val activeWorkCache = ConcurrentHashMap<String, WorkInfo>()
    private val insightGenerationQueue = InsightQueue()
    private val schedulingHistory = SchedulingHistory()

    // Control flags
    private val isInitialized = AtomicBoolean(false)
    private val lastHealthCheck = AtomicLong(0L)
    private val totalInsightsGenerated = AtomicLong(0L)
    private val totalErrors = AtomicLong(0L)

    // Configuration
    private var currentConfig: AdvancedSchedulerConfig = loadConfiguration()
    private var abTestConfig: ABTestConfiguration = loadABTestConfiguration()
    private var personalizationModel: PersonalizationModel = loadPersonalizationModel()

    // ========== INITIALIZATION AND LIFECYCLE ==========

    /**
     * Initialize the comprehensive insights scheduler
     */
    suspend fun initialize(): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Initializing advanced InsightsScheduler")
            _schedulerState.value = SchedulerState.INITIALIZING

            // Load configuration and historical data
            loadSchedulerState()
            initializeComponents()

            // Start background monitoring and optimization
            startHealthMonitoring()
            startPerformanceMonitoring()
            startEngagementTracking()

            // Schedule initial work
            schedulePeriodicMaintenance()
            scheduleHealthChecks()

            // Initialize machine learning models
            initializeMLModels()

            isInitialized.set(true)
            _schedulerState.value = SchedulerState.ACTIVE

            Log.d(TAG, "InsightsScheduler initialized successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize InsightsScheduler", e)
            _schedulerState.value = SchedulerState.ERROR("Initialization failed: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Update scheduler configuration with validation and optimization
     */
    suspend fun updateConfiguration(
        newConfig: AdvancedSchedulerConfig,
        applyImmediately: Boolean = true
    ): Result<Unit> = withContext(dispatcher) {
        try {
            // Validate configuration
            val validationResult = validateConfiguration(newConfig)
            if (validationResult.isFailure) {
                return@withContext validationResult
            }

            // Apply configuration with rollback capability
            val previousConfig = currentConfig
            currentConfig = newConfig

            try {
                saveConfiguration(newConfig)

                if (applyImmediately) {
                    applyConfigurationChanges(previousConfig, newConfig)
                }

                Log.d(TAG, "Configuration updated successfully")
                Result.success(Unit)

            } catch (e: Exception) {
                // Rollback on failure
                currentConfig = previousConfig
                Log.e(TAG, "Failed to apply configuration, rolled back", e)
                Result.failure(e)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Configuration update failed", e)
            Result.failure(e)
        }
    }

    // ========== ADVANCED SCHEDULING METHODS ==========

    /**
     * Schedule post-session insights with intelligent timing
     */
    suspend fun schedulePostSessionInsights(
        sessionId: Long,
        sessionAnalytics: SleepSessionAnalytics? = null,
        priority: InsightPriority = InsightPriority.NORMAL
    ): Result<String> = withContext(dispatcher) {
        try {
            if (!currentConfig.postSessionInsightsEnabled) {
                Log.d(TAG, "Post-session insights disabled")
                return@withContext Result.failure(IllegalStateException("Post-session insights disabled"))
            }

            // Check circuit breaker
            if (!circuitBreaker.canExecute()) {
                Log.w(TAG, "Circuit breaker open, skipping post-session insights")
                return@withContext Result.failure(IllegalStateException("Circuit breaker open"))
            }

            // Calculate optimal delay based on session data and user patterns
            val optimalDelay = calculateOptimalDelay(
                InsightGenerationType.POST_SESSION,
                sessionAnalytics
            )

            // Calculate insight priority score
            val priorityScore = priorityCalculator.calculatePriority(
                type = InsightGenerationType.POST_SESSION,
                sessionData = sessionAnalytics,
                userContext = getCurrentUserContext(),
                basePriority = priority
            )

            // Create work request with advanced configuration
            val workId = generateWorkId("post_session", sessionId.toString())
            val inputData = createInputData {
                putLong("session_id", sessionId)
                putString("analysis_type", InsightGenerationType.POST_SESSION.name)
                putString("priority", priority.name)
                putFloat("priority_score", priorityScore)
                putLong("scheduled_time", System.currentTimeMillis())
                putString("work_id", workId)
                sessionAnalytics?.let { putString("session_analytics", serializeSessionAnalytics(it)) }
            }

            val constraints = createSmartConstraints(priority, optimalDelay)
            val workRequest = OneTimeWorkRequestBuilder<AdvancedInsightGenerationWorker>()
                .setInputData(inputData)
                .setInitialDelay(optimalDelay.inWholeMinutes, TimeUnit.MINUTES)
                .setConstraints(constraints)
                .setBackoffCriteria(createAdvancedBackoffCriteria(priority))
                .addTag(WORK_TAG_POST_SESSION)
                .addTag("session_$sessionId")
                .addTag("priority_${priority.name}")
                .build()

            // Enqueue with unique work policy
            val workName = "post_session_insights_$sessionId"
            workManager.enqueueUniqueWork(
                workName,
                ExistingWorkPolicy.REPLACE,
                workRequest
            )

            // Track in queue and history
            insightGenerationQueue.addWork(workId, workRequest.id, priorityScore)
            schedulingHistory.recordScheduling(
                type = InsightGenerationType.POST_SESSION,
                workId = workId,
                delay = optimalDelay,
                priority = priorityScore,
                sessionId = sessionId
            )

            Log.d(TAG, "Post-session insights scheduled: workId=$workId, delay=${optimalDelay.inWholeMinutes}min, priority=$priorityScore")
            Result.success(workId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to schedule post-session insights", e)
            circuitBreaker.recordFailure()
            Result.failure(e)
        }
    }

    /**
     * Schedule personalized insights with ML-driven optimization
     */
    suspend fun schedulePersonalizedInsights(
        analysisDepth: AnalysisDepth = AnalysisDepth.COMPREHENSIVE,
        triggerReason: InsightTriggerReason = InsightTriggerReason.SCHEDULED,
        customParameters: Map<String, Any> = emptyMap()
    ): Result<String> = withContext(dispatcher) {
        try {
            if (!currentConfig.personalizedInsightsEnabled) {
                return@withContext Result.failure(IllegalStateException("Personalized insights disabled"))
            }

            // Check if we should generate insights based on ML model
            val shouldGenerate = personalizationModel.shouldGenerateInsights(
                userEngagement = engagementTracker.getCurrentEngagement(),
                timeContext = getCurrentTimeContext(),
                recentActivity = getRecentActivityContext(),
                customParameters = customParameters
            )

            if (!shouldGenerate.recommend) {
                Log.d(TAG, "ML model recommends skipping personalized insights: ${shouldGenerate.reason}")
                return@withContext Result.failure(IllegalStateException(shouldGenerate.reason))
            }

            // Calculate optimal timing using advanced algorithms
            val optimalTiming = timingOptimizer.calculateOptimalTiming(
                analysisType = InsightGenerationType.PERSONALIZED_ANALYSIS,
                userPatterns = getUserSleepPatterns(),
                engagementData = engagementTracker.getEngagementHistory(),
                environmentalFactors = getCurrentEnvironmentalFactors()
            )

            // Create comprehensive work request
            val workId = generateWorkId("personalized", System.currentTimeMillis().toString())
            val inputData = createInputData {
                putString("analysis_type", InsightGenerationType.PERSONALIZED_ANALYSIS.name)
                putString("analysis_depth", analysisDepth.name)
                putString("trigger_reason", triggerReason.name)
                putFloat("ml_confidence", shouldGenerate.confidence)
                putLong("optimal_timing", optimalTiming.timestamp)
                putString("work_id", workId)
                putString("custom_parameters", serializeMap(customParameters))
                putString("user_context", serializeUserContext(getCurrentUserContext()))
            }

            val delay = Duration.ofMillis(optimalTiming.timestamp - System.currentTimeMillis())
                .coerceAtLeast(Duration.ZERO)

            val workRequest = OneTimeWorkRequestBuilder<AdvancedInsightGenerationWorker>()
                .setInputData(inputData)
                .setInitialDelay(delay.inWholeMinutes, TimeUnit.MINUTES)
                .setConstraints(createIntelligentConstraints(analysisDepth))
                .setBackoffCriteria(createAdaptiveBackoffCriteria(shouldGenerate.confidence))
                .addTag(WORK_TAG_PERSONALIZED)
                .addTag("depth_${analysisDepth.name}")
                .addTag("trigger_${triggerReason.name}")
                .build()

            workManager.enqueueUniqueWork(
                "personalized_insights_${System.currentTimeMillis()}",
                ExistingWorkPolicy.KEEP,
                workRequest
            )

            // Update ML model with scheduling decision
            personalizationModel.recordSchedulingDecision(
                decision = true,
                context = getCurrentUserContext(),
                timing = optimalTiming,
                confidence = shouldGenerate.confidence
            )

            Result.success(workId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to schedule personalized insights", e)
            Result.failure(e)
        }
    }

    /**
     * Schedule predictive insights based on trend analysis
     */
    suspend fun schedulePredictiveInsights(
        predictionHorizon: PredictionHorizon,
        confidence: Float,
        triggeringTrends: List<TrendAnalysisResult>
    ): Result<String> = withContext(dispatcher) {
        try {
            if (!currentConfig.predictiveInsightsEnabled) {
                return@withContext Result.failure(IllegalStateException("Predictive insights disabled"))
            }

            // Validate prediction confidence
            if (confidence < currentConfig.minimumPredictionConfidence) {
                Log.d(TAG, "Prediction confidence too low: $confidence < ${currentConfig.minimumPredictionConfidence}")
                return@withContext Result.failure(IllegalStateException("Insufficient prediction confidence"))
            }

            // Calculate priority based on trend significance
            val priority = calculatePredictivePriority(triggeringTrends, confidence)

            // Determine optimal delivery time
            val deliveryTime = calculatePredictiveDeliveryTime(
                horizon = predictionHorizon,
                trends = triggeringTrends,
                userPatterns = getUserNotificationPatterns()
            )

            val workId = generateWorkId("predictive", predictionHorizon.name)
            val inputData = createInputData {
                putString("analysis_type", InsightGenerationType.PREDICTIVE_ANALYSIS.name)
                putString("prediction_horizon", predictionHorizon.name)
                putFloat("confidence", confidence)
                putString("triggering_trends", serializeTrends(triggeringTrends))
                putString("work_id", workId)
                putLong("delivery_time", deliveryTime)
                putFloat("calculated_priority", priority)
            }

            val delay = Duration.ofMillis(deliveryTime - System.currentTimeMillis())
                .coerceAtLeast(Duration.ZERO)

            val workRequest = OneTimeWorkRequestBuilder<AdvancedInsightGenerationWorker>()
                .setInputData(inputData)
                .setInitialDelay(delay.inWholeMinutes, TimeUnit.MINUTES)
                .setConstraints(createPredictiveConstraints(priority))
                .addTag(WORK_TAG_PREDICTIVE)
                .addTag("horizon_${predictionHorizon.name}")
                .build()

            workManager.enqueueUniqueWork(
                "predictive_insights_${predictionHorizon.name}",
                ExistingWorkPolicy.REPLACE,
                workRequest
            )

            Result.success(workId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to schedule predictive insights", e)
            Result.failure(e)
        }
    }

    /**
     * Schedule emergency insights for critical situations
     */
    suspend fun scheduleEmergencyInsights(
        emergencyType: EmergencyInsightType,
        severity: EmergencySeverity,
        triggeringData: Any? = null,
        immediateDelivery: Boolean = true
    ): Result<String> = withContext(dispatcher) {
        try {
            Log.w(TAG, "Scheduling emergency insights: type=$emergencyType, severity=$severity")

            val workId = generateWorkId("emergency", emergencyType.name)
            val inputData = createInputData {
                putString("analysis_type", InsightGenerationType.EMERGENCY.name)
                putString("emergency_type", emergencyType.name)
                putString("severity", severity.name)
                putBoolean("immediate_delivery", immediateDelivery)
                putString("work_id", workId)
                putLong("trigger_time", System.currentTimeMillis())
                triggeringData?.let { putString("triggering_data", serializeAny(it)) }
            }

            val delay = if (immediateDelivery) 0L else calculateEmergencyDelay(severity)
            val constraints = createEmergencyConstraints(severity)

            val workRequest = OneTimeWorkRequestBuilder<AdvancedInsightGenerationWorker>()
                .setInputData(inputData)
                .setInitialDelay(delay, TimeUnit.MINUTES)
                .setConstraints(constraints)
                .addTag(WORK_TAG_EMERGENCY)
                .addTag("severity_${severity.name}")
                .addTag("type_${emergencyType.name}")
                .build()

            // Emergency insights get immediate priority
            workManager.enqueueUniqueWork(
                "emergency_insights_${emergencyType.name}",
                ExistingWorkPolicy.REPLACE,
                workRequest
            )

            // Record emergency scheduling
            performanceMonitor.recordEmergencyScheduling(emergencyType, severity)

            Result.success(workId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to schedule emergency insights", e)
            Result.failure(e)
        }
    }

    // ========== ADVANCED WORK MANAGEMENT ==========

    /**
     * Intelligent work cancellation with priority consideration
     */
    suspend fun cancelWork(
        workId: String? = null,
        tag: String? = null,
        type: InsightGenerationType? = null,
        preserveHighPriority: Boolean = true
    ): Result<Int> = withContext(dispatcher) {
        try {
            var cancelledCount = 0

            when {
                workId != null -> {
                    workManager.cancelUniqueWork(workId)
                    insightGenerationQueue.removeWork(workId)
                    activeWorkCache.remove(workId)
                    cancelledCount = 1
                }

                tag != null -> {
                    if (preserveHighPriority) {
                        cancelledCount = cancelWorkByTagWithPriorityFilter(tag)
                    } else {
                        workManager.cancelAllWorkByTag(tag)
                        cancelledCount = activeWorkCache.values.count { it.tags.contains(tag) }
                        activeWorkCache.values.removeIf { it.tags.contains(tag) }
                    }
                }

                type != null -> {
                    val typeTag = getTagForType(type)
                    cancelledCount = cancelWork(tag = typeTag, preserveHighPriority = preserveHighPriority).getOrElse { 0 }
                }

                else -> {
                    return@withContext Result.failure(IllegalArgumentException("Must specify workId, tag, or type"))
                }
            }

            Log.d(TAG, "Cancelled $cancelledCount work items")
            Result.success(cancelledCount)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to cancel work", e)
            Result.failure(e)
        }
    }

    /**
     * Reschedule work with updated parameters
     */
    suspend fun rescheduleWork(
        workId: String,
        newDelay: Duration? = null,
        newPriority: InsightPriority? = null,
        newConstraints: Constraints? = null
    ): Result<String> = withContext(dispatcher) {
        try {
            // Find existing work
            val existingWork = findWorkById(workId)
                ?: return@withContext Result.failure(IllegalArgumentException("Work not found: $workId"))

            // Create updated work request
            val updatedInputData = existingWork.inputData.let { data ->
                val builder = Data.Builder()
                for (key in data.keyValueMap.keys) {
                    when (val value = data.keyValueMap[key]) {
                        is String -> builder.putString(key, value)
                        is Int -> builder.putInt(key, value)
                        is Long -> builder.putLong(key, value)
                        is Float -> builder.putFloat(key, value)
                        is Boolean -> builder.putBoolean(key, value)
                    }
                }

                // Update priority if specified
                newPriority?.let { builder.putString("priority", it.name) }

                builder.build()
            }

            val delay = newDelay ?: Duration.ofMinutes(5) // Default 5 minute delay
            val constraints = newConstraints ?: existingWork.constraints
            val tags = existingWork.tags

            // Cancel existing work
            cancelWork(workId = workId)

            // Create new work request
            val newWorkId = generateWorkId("rescheduled", workId)
            val workRequest = OneTimeWorkRequestBuilder<AdvancedInsightGenerationWorker>()
                .setInputData(updatedInputData)
                .setInitialDelay(delay.inWholeMinutes, TimeUnit.MINUTES)
                .setConstraints(constraints)
                .apply { tags.forEach { addTag(it) } }
                .build()

            workManager.enqueueUniqueWork(
                "rescheduled_$newWorkId",
                ExistingWorkPolicy.REPLACE,
                workRequest
            )

            Log.d(TAG, "Work rescheduled: $workId -> $newWorkId")
            Result.success(newWorkId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to reschedule work", e)
            Result.failure(e)
        }
    }

    /**
     * Batch schedule multiple insights with optimization
     */
    suspend fun batchScheduleInsights(
        requests: List<BatchInsightRequest>
    ): Result<List<String>> = withContext(dispatcher) {
        try {
            if (requests.isEmpty()) {
                return@withContext Result.success(emptyList())
            }

            // Optimize batch scheduling
            val optimizedRequests = optimizeBatchScheduling(requests)
            val workIds = mutableListOf<String>()
            val failures = mutableListOf<Exception>()

            // Execute batch with proper error handling
            optimizedRequests.forEach { request ->
                try {
                    val result = when (request.type) {
                        InsightGenerationType.POST_SESSION ->
                            schedulePostSessionInsights(
                                sessionId = request.sessionId ?: 0L,
                                priority = request.priority
                            )

                        InsightGenerationType.PERSONALIZED_ANALYSIS ->
                            schedulePersonalizedInsights(
                                analysisDepth = request.analysisDepth ?: AnalysisDepth.STANDARD,
                                triggerReason = request.triggerReason ?: InsightTriggerReason.BATCH,
                                customParameters = request.customParameters
                            )

                        InsightGenerationType.PREDICTIVE_ANALYSIS ->
                            schedulePredictiveInsights(
                                predictionHorizon = request.predictionHorizon ?: PredictionHorizon.SHORT_TERM,
                                confidence = request.confidence ?: 0.8f,
                                triggeringTrends = request.triggeringTrends ?: emptyList()
                            )

                        else -> Result.failure(IllegalArgumentException("Unsupported batch type: ${request.type}"))
                    }

                    result.fold(
                        onSuccess = { workId -> workIds.add(workId) },
                        onFailure = { error -> failures.add(Exception("Failed to schedule ${request.type}: ${error.message}", error)) }
                    )

                } catch (e: Exception) {
                    failures.add(e)
                }
            }

            // Return results with partial success handling
            if (failures.isNotEmpty() && workIds.isEmpty()) {
                Result.failure(Exception("All batch requests failed: ${failures.joinToString { it.message ?: "Unknown" }}"))
            } else {
                if (failures.isNotEmpty()) {
                    Log.w(TAG, "Batch scheduling completed with ${failures.size} failures")
                }
                Result.success(workIds)
            }

        } catch (e: Exception) {
            Log.e(TAG, "Batch scheduling failed", e)
            Result.failure(e)
        }
    }

    // ========== HEALTH MONITORING AND PERFORMANCE ==========

    /**
     * Get comprehensive scheduler status
     */
    fun getSchedulerStatus(): SchedulerStatus {
        return SchedulerStatus(
            state = _schedulerState.value,
            health = _schedulerHealth.value,
            isInitialized = isInitialized.get(),
            configuration = currentConfig,
            performanceMetrics = performanceMonitor.getCurrentMetrics(),
            circuitBreakerState = circuitBreaker.getState(),
            activeWorkCount = activeWorkCache.size,
            queuedInsightsCount = insightGenerationQueue.size(),
            totalInsightsGenerated = totalInsightsGenerated.get(),
            errorRate = calculateCurrentErrorRate(),
            lastHealthCheck = lastHealthCheck.get(),
            engagementMetrics = engagementTracker.getCurrentMetrics(),
            batteryOptimizationStatus = batteryOptimizer.getStatus(),
            mlModelStatus = personalizationModel.getStatus()
        )
    }

    /**
     * Force health check and diagnostics
     */
    suspend fun runHealthCheck(): HealthCheckResult = withContext(dispatcher) {
        try {
            Log.d(TAG, "Running comprehensive health check")

            val startTime = System.currentTimeMillis()
            val healthIssues = mutableListOf<HealthIssue>()
            val performanceMetrics = mutableMapOf<String, Any>()

            // Check core components
            healthIssues.addAll(checkCoreComponents())

            // Check performance metrics
            val perfCheck = performanceMonitor.runHealthCheck()
            healthIssues.addAll(perfCheck.issues)
            performanceMetrics.putAll(perfCheck.metrics)

            // Check work manager state
            val workHealthCheck = checkWorkManagerHealth()
            healthIssues.addAll(workHealthCheck.issues)

            // Check ML models
            val mlHealthCheck = checkMLModelsHealth()
            healthIssues.addAll(mlHealthCheck.issues)

            // Check resource usage
            val resourceCheck = checkResourceUsage()
            healthIssues.addAll(resourceCheck.issues)

            // Calculate overall health score
            val healthScore = calculateHealthScore(healthIssues)
            val overallHealth = when {
                healthScore >= 0.9f -> SchedulerHealth.EXCELLENT
                healthScore >= 0.7f -> SchedulerHealth.GOOD
                healthScore >= 0.5f -> SchedulerHealth.WARNING("Some issues detected")
                else -> SchedulerHealth.CRITICAL("Multiple critical issues")
            }

            _schedulerHealth.value = overallHealth
            lastHealthCheck.set(System.currentTimeMillis())

            val result = HealthCheckResult(
                overallHealth = overallHealth,
                healthScore = healthScore,
                issues = healthIssues,
                performanceMetrics = performanceMetrics,
                duration = System.currentTimeMillis() - startTime,
                timestamp = System.currentTimeMillis()
            )

            Log.d(TAG, "Health check completed: score=$healthScore, issues=${healthIssues.size}")
            result

        } catch (e: Exception) {
            Log.e(TAG, "Health check failed", e)
            HealthCheckResult(
                overallHealth = SchedulerHealth.CRITICAL("Health check failed: ${e.message}"),
                healthScore = 0f,
                issues = listOf(HealthIssue.CRITICAL("Health check exception", e.message ?: "Unknown")),
                performanceMetrics = emptyMap(),
                duration = 0L,
                timestamp = System.currentTimeMillis()
            )
        }
    }

    /**
     * Get detailed performance analytics
     */
    suspend fun getPerformanceAnalytics(
        timeRange: TimeRange? = null
    ): SchedulerPerformanceAnalytics = withContext(dispatcher) {
        performanceMonitor.getDetailedAnalytics(timeRange)
    }

    /**
     * Optimize scheduler performance
     */
    suspend fun optimizePerformance(): Result<PerformanceOptimizationResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Starting performance optimization")

            val optimizationActions = mutableListOf<OptimizationAction>()
            var improvementScore = 0f

            // Optimize work queue
            val queueOptimization = insightGenerationQueue.optimize()
            optimizationActions.addAll(queueOptimization.actions)
            improvementScore += queueOptimization.improvementScore

            // Optimize timing algorithms
            val timingOptimization = timingOptimizer.optimize(
                engagementData = engagementTracker.getEngagementHistory(),
                performanceData = performanceMonitor.getTimingData()
            )
            optimizationActions.addAll(timingOptimization.actions)
            improvementScore += timingOptimization.improvementScore

            // Optimize ML models
            val mlOptimization = personalizationModel.optimize()
            optimizationActions.addAll(mlOptimization.actions)
            improvementScore += mlOptimization.improvementScore

            // Optimize battery usage
            val batteryOptimization = batteryOptimizer.optimize(getCurrentBatteryLevel())
            optimizationActions.addAll(batteryOptimization.actions)
            improvementScore += batteryOptimization.improvementScore

            // Apply optimizations
            applyOptimizations(optimizationActions)

            val result = PerformanceOptimizationResult(
                actions = optimizationActions,
                overallImprovementScore = improvementScore / 4f, // Average of all optimizations
                timestamp = System.currentTimeMillis(),
                estimatedImpact = calculateEstimatedImpact(optimizationActions)
            )

            Log.d(TAG, "Performance optimization completed: score=${result.overallImprovementScore}")
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Performance optimization failed", e)
            Result.failure(e)
        }
    }

    // ========== MACHINE LEARNING AND INTELLIGENCE ==========

    /**
     * Update ML models with user feedback
     */
    suspend fun updateMLModelsWithFeedback(
        insightId: String,
        feedback: InsightFeedback
    ): Result<Unit> = withContext(dispatcher) {
        try {
            // Update engagement tracker
            engagementTracker.recordInsightFeedback(insightId, feedback)

            // Update timing optimizer
            timingOptimizer.recordFeedback(insightId, feedback)

            // Update personalization model
            personalizationModel.recordFeedback(insightId, feedback)

            // Update priority calculator
            priorityCalculator.recordFeedback(insightId, feedback)

            Log.d(TAG, "ML models updated with feedback for insight: $insightId")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update ML models with feedback", e)
            Result.failure(e)
        }
    }

    /**
     * Train ML models with historical data
     */
    suspend fun trainMLModels(
        trainingData: MLTrainingData? = null,
        forceRetrain: Boolean = false
    ): Result<MLTrainingResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Starting ML model training")

            val data = trainingData ?: collectTrainingData()
            val trainingResults = mutableListOf<ModelTrainingResult>()

            // Train engagement model
            val engagementResult = engagementTracker.train(data.engagementData, forceRetrain)
            trainingResults.add(engagementResult)

            // Train timing optimization model
            val timingResult = timingOptimizer.train(data.timingData, forceRetrain)
            trainingResults.add(timingResult)

            // Train personalization model
            val personalizationResult = personalizationModel.train(data.personalizationData, forceRetrain)
            trainingResults.add(personalizationResult)

            // Train priority calculation model
            val priorityResult = priorityCalculator.train(data.priorityData, forceRetrain)
            trainingResults.add(priorityResult)

            val overallResult = MLTrainingResult(
                models = trainingResults,
                overallAccuracy = trainingResults.map { it.accuracy }.average().toFloat(),
                trainingDuration = trainingResults.sumOf { it.trainingDuration },
                timestamp = System.currentTimeMillis(),
                dataSize = data.totalSamples
            )

            Log.d(TAG, "ML model training completed: accuracy=${overallResult.overallAccuracy}")
            Result.success(overallResult)

        } catch (e: Exception) {
            Log.e(TAG, "ML model training failed", e)
            Result.failure(e)
        }
    }

    // ========== PRIVATE IMPLEMENTATION ==========

    private suspend fun loadSchedulerState() {
        try {
            // Load configuration
            currentConfig = loadConfiguration()

            // Load A/B test configuration
            abTestConfig = loadABTestConfiguration()

            // Load personalization model
            personalizationModel = loadPersonalizationModel()

            // Load performance metrics
            performanceMonitor.loadHistoricalData()

            // Load engagement data
            engagementTracker.loadHistoricalData()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load scheduler state", e)
        }
    }

    private suspend fun initializeComponents() {
        // Initialize engagement tracker
        engagementTracker.initialize(preferences)

        // Initialize timing optimizer
        timingOptimizer.initialize(preferences, repository)

        // Initialize priority calculator
        priorityCalculator.initialize(preferences, repository)

        // Initialize circuit breaker
        circuitBreaker.initialize(preferences)

        // Initialize performance monitor
        performanceMonitor.initialize(preferences)

        // Initialize battery optimizer
        batteryOptimizer.initialize(context, preferences)
    }

    private fun startHealthMonitoring() {
        scope.launch {
            while (isActive) {
                try {
                    val healthResult = runHealthCheck()
                    _performanceMetrics.postValue(performanceMonitor.getCurrentMetrics())

                    // Auto-recovery if issues detected
                    if (healthResult.healthScore < 0.5f) {
                        attemptAutoRecovery(healthResult.issues)
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Health monitoring error", e)
                }

                delay(TimeUnit.MINUTES.toMillis(HEALTH_CHECK_INTERVAL_MINUTES))
            }
        }
    }

    private fun startPerformanceMonitoring() {
        scope.launch {
            performanceMonitor.startContinuousMonitoring(scope)
        }
    }

    private fun startEngagementTracking() {
        scope.launch {
            engagementTracker.startTracking(scope)
        }
    }

    private suspend fun initializeMLModels() {
        // Check if models need training
        val shouldTrain = personalizationModel.needsTraining() ||
                timingOptimizer.needsTraining() ||
                engagementTracker.needsTraining() ||
                priorityCalculator.needsTraining()

        if (shouldTrain) {
            Log.d(TAG, "ML models need training, starting background training")
            scope.launch {
                trainMLModels()
            }
        }
    }

    // Additional helper methods would be implemented here...
    // Due to length constraints, showing the core structure and key methods

    /**
     * Cleanup resources
     */
    fun cleanup() {
        scope.cancel()
        performanceMonitor.cleanup()
        engagementTracker.cleanup()
        timingOptimizer.cleanup()
        priorityCalculator.cleanup()
        batteryOptimizer.cleanup()
        Log.d(TAG, "InsightsScheduler cleanup completed")
    }
}

// ========== SUPPORTING CLASSES AND DATA STRUCTURES ==========

/**
 * Advanced scheduler configuration
 */
data class AdvancedSchedulerConfig(
    // Basic settings
    val insightsEnabled: Boolean = true,
    val postSessionInsightsEnabled: Boolean = true,
    val dailyInsightsEnabled: Boolean = true,
    val weeklyInsightsEnabled: Boolean = true,
    val personalizedInsightsEnabled: Boolean = true,
    val predictiveInsightsEnabled: Boolean = true,

    // Advanced settings
    val maxDailyInsights: Int = MAX_DAILY_INSIGHTS,
    val maxWeeklyInsights: Int = MAX_WEEKLY_INSIGHTS,
    val minimumPredictionConfidence: Float = 0.7f,
    val insightCooldownPeriodHours: Long = INSIGHT_COOLDOWN_PERIOD_HOURS,
    val enableMLOptimization: Boolean = true,
    val enableBatteryOptimization: Boolean = true,
    val enableAdaptiveScheduling: Boolean = true,

    // Timing preferences
    val preferredDeliveryWindow: TimeWindow = TimeWindow(8, 22), // 8 AM to 10 PM
    val quietHours: TimeWindow = TimeWindow(22, 7), // 10 PM to 7 AM
    val workdayScheduling: Boolean = true,
    val weekendScheduling: Boolean = true,

    // Quality settings
    val minimumInsightQuality: Float = 0.6f,
    val prioritizeHighQualityInsights: Boolean = true,
    val enableInsightFiltering: Boolean = true,
    val maxRetryAttempts: Int = 3,

    // Machine learning settings
    val enableUserPatternLearning: Boolean = true,
    val enableTimingOptimization: Boolean = true,
    val enableEngagementTracking: Boolean = true,
    val mlModelUpdateFrequency: MLUpdateFrequency = MLUpdateFrequency.WEEKLY,

    // A/B testing
    val enableABTesting: Boolean = false,
    val abTestGroup: String = "control"
)

/**
 * Comprehensive scheduler status
 */
data class SchedulerStatus(
    val state: SchedulerState,
    val health: SchedulerHealth,
    val isInitialized: Boolean,
    val configuration: AdvancedSchedulerConfig,
    val performanceMetrics: SchedulerPerformanceMetrics,
    val circuitBreakerState: CircuitBreakerState,
    val activeWorkCount: Int,
    val queuedInsightsCount: Int,
    val totalInsightsGenerated: Long,
    val errorRate: Float,
    val lastHealthCheck: Long,
    val engagementMetrics: UserEngagementMetrics,
    val batteryOptimizationStatus: BatteryOptimizationStatus,
    val mlModelStatus: MLModelStatus
)

/**
 * Health check result
 */
data class HealthCheckResult(
    val overallHealth: SchedulerHealth,
    val healthScore: Float,
    val issues: List<HealthIssue>,
    val performanceMetrics: Map<String, Any>,
    val duration: Long,
    val timestamp: Long
)

// ========== ENUMS AND SEALED CLASSES ==========

sealed class SchedulerState {
    object INITIALIZING : SchedulerState()
    object ACTIVE : SchedulerState()
    object PAUSED : SchedulerState()
    data class ERROR(val message: String) : SchedulerState()
    object SHUTTING_DOWN : SchedulerState()
}

sealed class SchedulerHealth {
    object UNKNOWN : SchedulerHealth()
    object EXCELLENT : SchedulerHealth()
    object GOOD : SchedulerHealth()
    data class WARNING(val message: String) : SchedulerHealth()
    data class CRITICAL(val message: String) : SchedulerHealth()
}

enum class InsightPriority(val value: Int) {
    EMERGENCY(4),
    HIGH(3),
    NORMAL(2),
    LOW(1),
    BACKGROUND(0)
}

enum class PredictionHorizon {
    SHORT_TERM,   // Next 1-3 days
    MEDIUM_TERM,  // Next 1-2 weeks
    LONG_TERM     // Next 1-3 months
}

enum class EmergencyInsightType {
    SLEEP_QUALITY_CRISIS,
    HEALTH_RISK_DETECTED,
    PATTERN_ANOMALY,
    SYSTEM_MALFUNCTION
}

enum class EmergencySeverity {
    LOW, MEDIUM, HIGH, CRITICAL
}

enum class InsightTriggerReason {
    SCHEDULED,
    USER_REQUESTED,
    PATTERN_DETECTED,
    EMERGENCY,
    BATCH,
    ML_RECOMMENDED
}

enum class AnalysisDepth {
    BASIC,
    STANDARD,
    COMPREHENSIVE,
    DEEP_ANALYSIS
}

enum class MLUpdateFrequency {
    DAILY, WEEKLY, MONTHLY, ON_DEMAND
}

// Additional supporting classes and implementations would continue...
// This shows the comprehensive structure and advanced features of the scheduler