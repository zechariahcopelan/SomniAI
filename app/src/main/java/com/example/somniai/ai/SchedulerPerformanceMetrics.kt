package com.example.somniai.ai

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import kotlinx.serialization.Serializable
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.*

/**
 * Comprehensive AI Scheduling and Timing Performance Metrics
 *
 * Tracks all aspects of the InsightsScheduler performance including:
 * - Scheduling accuracy and timing optimization
 * - ML model performance for scheduling decisions
 * - User engagement and timing preferences learning
 * - Resource utilization and battery optimization
 * - Work queue management and prioritization
 * - Circuit breaker states and error handling
 * - Health diagnostics and system optimization
 */

// ========== CORE SCHEDULING METRICS ==========

/**
 * Primary scheduler performance metrics
 */
@Serializable
@Parcelize
data class SchedulerPerformanceMetrics(
    // Core scheduling performance
    val totalScheduledInsights: Long = 0L,
    val successfullyScheduled: Long = 0L,
    val schedulingSuccessRate: Float = 0f,
    val averageSchedulingLatency: Long = 0L,
    val currentActiveWork: Int = 0,
    val queuedWorkItems: Int = 0,

    // Timing optimization performance
    val timingAccuracyScore: Float = 0f,
    val averageDeliveryDelayMinutes: Long = 0L,
    val optimalTimingHitRate: Float = 0f,
    val userEngagementScore: Float = 0f,

    // ML model performance
    val mlModelAccuracy: Float = 0f,
    val mlPredictionConfidence: Float = 0f,
    val mlModelTrainingStatus: MLModelStatus = MLModelStatus.UNKNOWN,
    val personalizedRecommendationAccuracy: Float = 0f,

    // Resource utilization
    val memoryUsageBytes: Long = 0L,
    val batteryOptimizationScore: Float = 0f,
    val networkUsageOptimization: Float = 0f,
    val cpuUtilization: Float = 0f,

    // Error tracking and circuit breaker
    val errorRate: Float = 0f,
    val circuitBreakerState: CircuitBreakerState = CircuitBreakerState.CLOSED,
    val retryAttempts: Long = 0L,
    val failureRecoveryTime: Long = 0L,

    // Health and diagnostics
    val overallHealthScore: Float = 0f,
    val componentHealthScores: Map<String, Float> = emptyMap(),
    val lastHealthCheckTimestamp: Long = 0L,
    val systemPerformanceGrade: PerformanceGrade = PerformanceGrade.UNKNOWN,

    // Timestamps and tracking
    val metricsTimestamp: Long = System.currentTimeMillis(),
    val collectionPeriodMinutes: Int = 15,
    val shouldTriggerOptimization: Boolean = false
) : Parcelable

/**
 * Detailed timing optimization metrics
 */
@Serializable
@Parcelize
data class TimingOptimizationMetrics(
    // Timing accuracy tracking
    val scheduledVsActualDeliveryAccuracy: Float = 0f,
    val optimalDeliveryWindowHitRate: Float = 0f,
    val userPreferredTimeAlignmentScore: Float = 0f,
    val timeZoneOptimizationAccuracy: Float = 0f,

    // User pattern learning
    val userPatternConfidence: Float = 0f,
    val engagementPatternRecognition: Float = 0f,
    val seasonalAdjustmentAccuracy: Float = 0f,
    val weekdayWeekendOptimization: Float = 0f,

    // Delivery timing statistics
    val averageOptimalDeliveryTime: Long = 0L, // Time of day in minutes since midnight
    val deliveryTimeVariance: Float = 0f,
    val quickDeliverySuccessRate: Float = 0f,
    val scheduledDeliverySuccessRate: Float = 0f,

    // Context-aware timing
    val sleepScheduleAlignmentScore: Float = 0f,
    val activityBasedTimingScore: Float = 0f,
    val environmentalFactorConsiderationScore: Float = 0f,
    val batteryLevelTimingOptimization: Float = 0f,

    // Timing prediction models
    val nextOptimalDeliveryPrediction: Long = 0L,
    val timingPredictionConfidence: Float = 0f,
    val adaptiveSchedulingEffectiveness: Float = 0f,

    val lastUpdated: Long = System.currentTimeMillis()
) : Parcelable

/**
 * ML model performance metrics for scheduling decisions
 */
@Serializable
@Parcelize
data class MLSchedulingMetrics(
    // Model accuracy and performance
    val overallModelAccuracy: Float = 0f,
    val predictionConfidenceAverage: Float = 0f,
    val modelTrainingAccuracy: Float = 0f,
    val crossValidationScore: Float = 0f,

    // Individual model performance
    val engagementPredictionAccuracy: Float = 0f,
    val timingOptimizationAccuracy: Float = 0f,
    val priorityCalculationAccuracy: Float = 0f,
    val personalizationEffectiveness: Float = 0f,

    // Training and optimization
    val totalTrainingSamples: Long = 0L,
    val lastTrainingTimestamp: Long = 0L,
    val trainingDataQualityScore: Float = 0f,
    val modelDriftDetectionScore: Float = 0f,

    // Feature importance and analysis
    val topFeatureImportances: Map<String, Float> = emptyMap(),
    val featureCorrelationStrength: Float = 0f,
    val dimensionalityReductionEfficiency: Float = 0f,

    // Real-time adaptation
    val onlineLearningEffectiveness: Float = 0f,
    val adaptationSpeed: Float = 0f,
    val conceptDriftAdaptation: Float = 0f,
    val reinforcementLearningPerformance: Float = 0f,

    // Model deployment metrics
    val inferenceLatency: Long = 0L,
    val modelLoadTime: Long = 0L,
    val memoryFootprint: Long = 0L,
    val modelVersions: Map<String, String> = emptyMap(),

    val timestamp: Long = System.currentTimeMillis()
) : Parcelable

/**
 * User engagement and feedback metrics
 */
@Serializable
@Parcelize
data class UserEngagementMetrics(
    // Engagement tracking
    val totalInsightsDelivered: Long = 0L,
    val insightsViewed: Long = 0L,
    val insightsActedUpon: Long = 0L,
    val insightsDismissed: Long = 0L,
    val engagementRate: Float = 0f,

    // Timing effectiveness
    val deliveredAtOptimalTime: Long = 0L,
    val deliveredAtSuboptimalTime: Long = 0L,
    val userTimingFeedbackScore: Float = 0f,
    val timingAdjustmentRequests: Long = 0L,

    // Content engagement
    val averageViewingDuration: Long = 0L,
    val insightQualityRating: Float = 0f,
    val userSatisfactionScore: Float = 0f,
    val repeatEngagementRate: Float = 0f,

    // Behavioral patterns
    val preferredDeliveryHours: List<Int> = emptyList(),
    val engagementByDayOfWeek: Map<String, Float> = emptyMap(),
    val seasonalEngagementPatterns: Map<String, Float> = emptyMap(),
    val deviceUsagePatterns: Map<String, Float> = emptyMap(),

    // Feedback and learning
    val positiveFeedbackRate: Float = 0f,
    val negativeFeedbackRate: Float = 0f,
    val feedbackResponseTime: Long = 0L,
    val improvementSuggestions: Long = 0L,

    // Personalization effectiveness
    val personalizationAccuracy: Float = 0f,
    val adaptationSpeed: Float = 0f,
    val userPreferenceLearningRate: Float = 0f,

    val lastUpdated: Long = System.currentTimeMillis()
) : Parcelable

/**
 * Work queue management and prioritization metrics
 */
@Serializable
@Parcelize
data class WorkQueueMetrics(
    // Queue performance
    val totalWorkItemsProcessed: Long = 0L,
    val averageQueueWaitTime: Long = 0L,
    val queueThroughputPerHour: Float = 0f,
    val queueBacklogSize: Int = 0,

    // Priority management
    val highPriorityItemsProcessed: Long = 0L,
    val mediumPriorityItemsProcessed: Long = 0L,
    val lowPriorityItemsProcessed: Long = 0L,
    val emergencyItemsProcessed: Long = 0L,
    val priorityAccuracyScore: Float = 0f,

    // Work distribution by type
    val postSessionInsights: Long = 0L,
    val personalizedInsights: Long = 0L,
    val predictiveInsights: Long = 0L,
    val emergencyInsights: Long = 0L,
    val maintenanceWork: Long = 0L,

    // Scheduling optimization
    val batchingEfficiency: Float = 0f,
    val loadBalancingScore: Float = 0f,
    val resourceUtilizationOptimization: Float = 0f,
    val parallelProcessingEfficiency: Float = 0f,

    // Error handling
    val failedWorkItems: Long = 0L,
    val retriedWorkItems: Long = 0L,
    val averageRetryAttempts: Float = 0f,
    val deadLetterQueueSize: Int = 0,

    // Performance optimization
    val workCompletionRate: Float = 0f,
    val averageExecutionTime: Long = 0L,
    val resourceContentionEvents: Long = 0L,
    val schedulingConflicts: Long = 0L,

    val timestamp: Long = System.currentTimeMillis()
) : Parcelable

/**
 * Battery and resource optimization metrics
 */
@Serializable
@Parcelize
data class BatteryOptimizationMetrics(
    // Battery tracking
    val currentBatteryLevel: Int = 100,
    val batteryOptimizedScheduling: Long = 0L,
    val batteryConstrainedOperations: Long = 0L,
    val powerEfficientModeActivations: Long = 0L,

    // Resource optimization
    val memoryOptimizationScore: Float = 0f,
    val cpuUtilizationOptimization: Float = 0f,
    val networkUsageOptimization: Float = 0f,
    val storageOptimizationScore: Float = 0f,

    // Doze mode and background processing
    val dozeOptimizedOperations: Long = 0L,
    val backgroundProcessingEfficiency: Float = 0f,
    val wakeupMinimizationScore: Float = 0f,
    val batteryWhitelistOptimization: Float = 0f,

    // Charging state optimization
    val chargingStateScheduling: Long = 0L,
    val lowPowerModeAdaptations: Long = 0L,
    val thermalThrottlingHandled: Long = 0L,
    val powerManagementScore: Float = 0f,

    val lastUpdated: Long = System.currentTimeMillis()
) : Parcelable

/**
 * Circuit breaker and error handling metrics
 */
@Serializable
@Parcelize
data class CircuitBreakerMetrics(
    // Circuit breaker state tracking
    val currentState: CircuitBreakerState = CircuitBreakerState.CLOSED,
    val stateChanges: Long = 0L,
    val openToHalfOpenTransitions: Long = 0L,
    val halfOpenToClosedTransitions: Long = 0L,
    val halfOpenToOpenTransitions: Long = 0L,

    // Failure tracking
    val totalFailures: Long = 0L,
    val consecutiveFailures: Int = 0,
    val failureRate: Float = 0f,
    val averageFailureRecoveryTime: Long = 0L,

    // Success tracking in half-open state
    val halfOpenSuccesses: Long = 0L,
    val halfOpenFailures: Long = 0L,
    val successRate: Float = 0f,

    // Timeout and threshold metrics
    val openStateTimeouts: Long = 0L,
    val thresholdViolations: Long = 0L,
    val responseTimeThresholdBreaches: Long = 0L,
    val errorRateThresholdBreaches: Long = 0L,

    // Recovery metrics
    val automaticRecoveries: Long = 0L,
    val manualRecoveries: Long = 0L,
    val partialRecoveries: Long = 0L,
    val fullRecoveries: Long = 0L,

    val lastStateChange: Long = 0L,
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable

/**
 * Health monitoring and diagnostics metrics
 */
@Serializable
@Parcelize
data class HealthDiagnosticsMetrics(
    // Overall health
    val overallHealthScore: Float = 0f,
    val healthGrade: PerformanceGrade = PerformanceGrade.UNKNOWN,
    val healthTrendDirection: TrendDirection = TrendDirection.STABLE,
    val lastHealthCheckTimestamp: Long = 0L,

    // Component health scores
    val schedulerCoreHealth: Float = 0f,
    val mlModelsHealth: Float = 0f,
    val workQueueHealth: Float = 0f,
    val timingOptimizerHealth: Float = 0f,
    val resourceManagerHealth: Float = 0f,
    val circuitBreakerHealth: Float = 0f,

    // Performance indicators
    val latencyHealth: Float = 0f,
    val throughputHealth: Float = 0f,
    val errorRateHealth: Float = 0f,
    val resourceUtilizationHealth: Float = 0f,

    // System diagnostics
    val memoryLeakDetection: Float = 0f,
    val performanceDegradationDetection: Float = 0f,
    val anomalyDetectionScore: Float = 0f,
    val predictiveHealthScore: Float = 0f,

    // Health trends
    val healthScoreHistory: List<Float> = emptyList(),
    val performanceOptimizationOpportunities: List<String> = emptyList(),
    val healthAlerts: List<String> = emptyList(),
    val recommendedActions: List<String> = emptyList(),

    val timestamp: Long = System.currentTimeMillis()
) : Parcelable

// ========== COMPREHENSIVE ANALYTICS CONTAINERS ==========

/**
 * Complete scheduler performance analytics aggregation
 */
@Serializable
@Parcelize
data class SchedulerPerformanceAnalytics(
    val coreMetrics: SchedulerPerformanceMetrics,
    val timingOptimization: TimingOptimizationMetrics,
    val mlPerformance: MLSchedulingMetrics,
    val userEngagement: UserEngagementMetrics,
    val workQueue: WorkQueueMetrics,
    val batteryOptimization: BatteryOptimizationMetrics,
    val circuitBreaker: CircuitBreakerMetrics,
    val healthDiagnostics: HealthDiagnosticsMetrics,

    // Aggregated insights
    val overallEfficiencyScore: Float = 0f,
    val optimizationRecommendations: List<OptimizationRecommendation> = emptyList(),
    val performanceTrends: Map<String, TrendData> = emptyMap(),
    val benchmarkComparisons: Map<String, BenchmarkResult> = emptyMap(),

    // Time range for analytics
    val analyticsTimeRange: TimeRange,
    val generatedAt: Long = System.currentTimeMillis()
) : Parcelable

/**
 * Historical performance tracking
 */
@Serializable
@Parcelize
data class PerformanceHistoryEntry(
    val timestamp: Long,
    val metrics: SchedulerPerformanceMetrics,
    val eventContext: String? = null,
    val optimizationActions: List<String> = emptyList()
) : Parcelable

/**
 * Performance optimization recommendations
 */
@Serializable
@Parcelize
data class OptimizationRecommendation(
    val category: OptimizationCategory,
    val description: String,
    val expectedImprovement: Float,
    val implementationEffort: ImplementationEffort,
    val priority: OptimizationPriority,
    val affectedComponents: List<String> = emptyList(),
    val estimatedImpact: Map<String, Float> = emptyMap()
) : Parcelable

/**
 * Trend analysis data
 */
@Serializable
@Parcelize
data class TrendData(
    val direction: TrendDirection,
    val magnitude: Float,
    val confidence: Float,
    val timeframe: Long,
    val dataPoints: Int,
    val statisticalSignificance: Float = 0f
) : Parcelable

/**
 * Benchmark comparison results
 */
@Serializable
@Parcelize
data class BenchmarkResult(
    val metric: String,
    val currentValue: Float,
    val benchmarkValue: Float,
    val percentageDifference: Float,
    val performanceCategory: BenchmarkCategory
) : Parcelable

/**
 * Time range specification
 */
@Serializable
@Parcelize
data class TimeRange(
    val startTime: Long,
    val endTime: Long,
    val description: String = ""
) : Parcelable {
    val durationMillis: Long get() = endTime - startTime
    val durationMinutes: Long get() = durationMillis / (60 * 1000)
    val durationHours: Long get() = durationMinutes / 60
    val durationDays: Long get() = durationHours / 24
}

// ========== ENUMS AND SUPPORTING TYPES ==========

enum class MLModelStatus {
    UNKNOWN,
    TRAINING,
    TRAINED,
    DEPLOYED,
    UPDATING,
    FAILED,
    DEPRECATED
}

enum class CircuitBreakerState {
    CLOSED,
    OPEN,
    HALF_OPEN
}

enum class PerformanceGrade {
    UNKNOWN,
    EXCELLENT,  // A+ (95-100%)
    GOOD,       // A-B (80-94%)
    FAIR,       // C (65-79%)
    POOR,       // D (50-64%)
    CRITICAL    // F (0-49%)
}

enum class TrendDirection {
    IMPROVING,
    STABLE,
    DECLINING,
    VOLATILE
}

enum class OptimizationCategory {
    TIMING_ACCURACY,
    ML_PERFORMANCE,
    RESOURCE_UTILIZATION,
    USER_ENGAGEMENT,
    ERROR_HANDLING,
    QUEUE_MANAGEMENT,
    BATTERY_EFFICIENCY,
    SYSTEM_HEALTH
}

enum class ImplementationEffort {
    MINIMAL,    // < 1 day
    LOW,        // 1-3 days
    MEDIUM,     // 1-2 weeks
    HIGH,       // 2-4 weeks
    EXTENSIVE   // > 1 month
}

enum class OptimizationPriority {
    CRITICAL,
    HIGH,
    MEDIUM,
    LOW,
    OPTIONAL
}

enum class BenchmarkCategory {
    EXCELLENT,
    ABOVE_AVERAGE,
    AVERAGE,
    BELOW_AVERAGE,
    POOR
}

// ========== METRICS COLLECTION AND AGGREGATION ==========

/**
 * Real-time metrics collector for scheduler performance
 */
class SchedulerMetricsCollector {
    private val metricsHistory = mutableListOf<PerformanceHistoryEntry>()
    private val currentMetrics = AtomicReference(SchedulerPerformanceMetrics())

    fun updateMetrics(update: (SchedulerPerformanceMetrics) -> SchedulerPerformanceMetrics) {
        val updated = update(currentMetrics.get())
        currentMetrics.set(updated)

        // Add to history
        metricsHistory.add(
            PerformanceHistoryEntry(
                timestamp = System.currentTimeMillis(),
                metrics = updated
            )
        )

        // Maintain history size
        if (metricsHistory.size > 1000) {
            metricsHistory.removeAt(0)
        }
    }

    fun getCurrentMetrics(): SchedulerPerformanceMetrics = currentMetrics.get()

    fun getMetricsHistory(timeRange: TimeRange? = null): List<PerformanceHistoryEntry> {
        return if (timeRange != null) {
            metricsHistory.filter {
                it.timestamp >= timeRange.startTime && it.timestamp <= timeRange.endTime
            }
        } else {
            metricsHistory.toList()
        }
    }

    fun calculateTrends(metric: String, timeRange: TimeRange): TrendData? {
        val history = getMetricsHistory(timeRange)
        if (history.size < 2) return null

        // Simple trend calculation - could be enhanced with more sophisticated algorithms
        val values = history.map { getMetricValue(it.metrics, metric) }
        val firstHalf = values.take(values.size / 2).average()
        val secondHalf = values.drop(values.size / 2).average()

        val percentChange = if (firstHalf != 0.0) {
            ((secondHalf - firstHalf) / firstHalf) * 100
        } else 0.0

        val direction = when {
            abs(percentChange) < 5 -> TrendDirection.IMPROVING
            percentChange > 0 -> TrendDirection.IMPROVING
            else -> TrendDirection.DECLINING
        }

        return TrendData(
            direction = direction,
            magnitude = percentChange.toFloat(),
            confidence = min(1.0f, values.size / 20.0f), // Higher confidence with more data points
            timeframe = timeRange.durationMillis,
            dataPoints = values.size
        )
    }

    private fun getMetricValue(metrics: SchedulerPerformanceMetrics, metricName: String): Double {
        return when (metricName) {
            "schedulingSuccessRate" -> metrics.schedulingSuccessRate.toDouble()
            "timingAccuracyScore" -> metrics.timingAccuracyScore.toDouble()
            "userEngagementScore" -> metrics.userEngagementScore.toDouble()
            "errorRate" -> metrics.errorRate.toDouble()
            "overallHealthScore" -> metrics.overallHealthScore.toDouble()
            else -> 0.0
        }
    }
}

/**
 * Performance analytics aggregator
 */
object SchedulerAnalyticsAggregator {

    fun generateComprehensiveAnalytics(
        collector: SchedulerMetricsCollector,
        timeRange: TimeRange
    ): SchedulerPerformanceAnalytics {
        val currentMetrics = collector.getCurrentMetrics()
        val history = collector.getMetricsHistory(timeRange)

        // Calculate trends for key metrics
        val trends = mapOf(
            "schedulingSuccessRate" to collector.calculateTrends("schedulingSuccessRate", timeRange),
            "timingAccuracy" to collector.calculateTrends("timingAccuracyScore", timeRange),
            "userEngagement" to collector.calculateTrends("userEngagementScore", timeRange),
            "errorRate" to collector.calculateTrends("errorRate", timeRange)
        ).mapNotNull { (key, value) -> value?.let { key to it } }.toMap()

        // Generate optimization recommendations
        val recommendations = generateOptimizationRecommendations(currentMetrics, trends)

        // Calculate overall efficiency score
        val efficiencyScore = calculateOverallEfficiencyScore(currentMetrics)

        return SchedulerPerformanceAnalytics(
            coreMetrics = currentMetrics,
            timingOptimization = TimingOptimizationMetrics(), // Would be populated from actual data
            mlPerformance = MLSchedulingMetrics(),
            userEngagement = UserEngagementMetrics(),
            workQueue = WorkQueueMetrics(),
            batteryOptimization = BatteryOptimizationMetrics(),
            circuitBreaker = CircuitBreakerMetrics(),
            healthDiagnostics = HealthDiagnosticsMetrics(),
            overallEfficiencyScore = efficiencyScore,
            optimizationRecommendations = recommendations,
            performanceTrends = trends,
            analyticsTimeRange = timeRange
        )
    }

    private fun generateOptimizationRecommendations(
        metrics: SchedulerPerformanceMetrics,
        trends: Map<String, TrendData>
    ): List<OptimizationRecommendation> {
        val recommendations = mutableListOf<OptimizationRecommendation>()

        // Check scheduling success rate
        if (metrics.schedulingSuccessRate < 0.9f) {
            recommendations.add(
                OptimizationRecommendation(
                    category = OptimizationCategory.ERROR_HANDLING,
                    description = "Improve scheduling success rate through enhanced error handling",
                    expectedImprovement = 0.15f,
                    implementationEffort = ImplementationEffort.MEDIUM,
                    priority = OptimizationPriority.HIGH
                )
            )
        }

        // Check timing accuracy
        if (metrics.timingAccuracyScore < 0.8f) {
            recommendations.add(
                OptimizationRecommendation(
                    category = OptimizationCategory.TIMING_ACCURACY,
                    description = "Enhance ML timing optimization models",
                    expectedImprovement = 0.2f,
                    implementationEffort = ImplementationEffort.HIGH,
                    priority = OptimizationPriority.MEDIUM
                )
            )
        }

        // Check user engagement
        if (metrics.userEngagementScore < 0.7f) {
            recommendations.add(
                OptimizationRecommendation(
                    category = OptimizationCategory.USER_ENGAGEMENT,
                    description = "Improve personalization algorithms for better engagement",
                    expectedImprovement = 0.25f,
                    implementationEffort = ImplementationEffort.HIGH,
                    priority = OptimizationPriority.HIGH
                )
            )
        }

        return recommendations
    }

    private fun calculateOverallEfficiencyScore(metrics: SchedulerPerformanceMetrics): Float {
        val weights = mapOf(
            "scheduling" to 0.25f,
            "timing" to 0.20f,
            "engagement" to 0.20f,
            "health" to 0.15f,
            "errors" to 0.10f,
            "resources" to 0.10f
        )

        return (metrics.schedulingSuccessRate * weights["scheduling"]!! +
                metrics.timingAccuracyScore * weights["timing"]!! +
                metrics.userEngagementScore * weights["engagement"]!! +
                metrics.overallHealthScore * weights["health"]!! +
                (1f - metrics.errorRate) * weights["errors"]!! +
                metrics.batteryOptimizationScore * weights["resources"]!!).coerceIn(0f, 1f)
    }
}