package com.example.somniai.ai

import com.example.somniai.ai.AIConstants.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*

/**
 * Advanced Model Configuration Management
 *
 * Companion to AIConfiguration.kt that provides dynamic model management,
 * performance monitoring, and intelligent switching strategies.
 * Works alongside the static configuration to provide runtime optimization.
 */
object ModelConfigManager {

    // ========== PERFORMANCE TRACKING ==========

    private val modelPerformanceStats = ConcurrentHashMap<AIModel, ModelPerformanceStats>()
    private val modelHealthStatus = ConcurrentHashMap<AIModel, ModelHealthStatus>()
    private val switchingHistory = mutableListOf<ModelSwitchEvent>()
    private val performanceWindow = 24 * 60 * 60 * 1000L // 24 hours

    private val _activeModel = MutableStateFlow<AIModel?>(null)
    val activeModel: StateFlow<AIModel?> = _activeModel.asStateFlow()

    private val _modelHealthUpdates = MutableSharedFlow<ModelHealthUpdate>()
    val modelHealthUpdates: SharedFlow<ModelHealthUpdate> = _modelHealthUpdates.asSharedFlow()

    // ========== INITIALIZATION ==========

    init {
        initializePerformanceTracking()
        startHealthMonitoring()
    }

    private fun initializePerformanceTracking() {
        // Initialize performance stats for all available models
        AIModel.values().forEach { model ->
            modelPerformanceStats[model] = ModelPerformanceStats(
                model = model,
                totalRequests = AtomicLong(0),
                successfulRequests = AtomicLong(0),
                failedRequests = AtomicLong(0),
                totalResponseTimeMs = AtomicLong(0),
                totalTokensUsed = AtomicLong(0),
                totalCost = 0.0,
                averageConfidence = 0.0f,
                lastUsed = 0L
            )

            modelHealthStatus[model] = ModelHealthStatus(
                model = model,
                isHealthy = true,
                lastHealthCheck = System.currentTimeMillis(),
                consecutiveFailures = 0,
                responseTimePercentile95 = 0L,
                errorRate = 0.0f,
                status = HealthStatusType.UNKNOWN
            )
        }
    }

    private fun startHealthMonitoring() {
        // Start background health monitoring
        CoroutineScope(Dispatchers.IO).launch {
            while (true) {
                delay(60000L) // Check every minute
                performHealthChecks()
            }
        }
    }

    // ========== INTELLIGENT MODEL SELECTION ==========

    /**
     * Select optimal model based on current performance metrics and requirements
     */
    suspend fun selectOptimalModel(
        requirements: ModelRequirements,
        excludeModels: Set<AIModel> = emptySet()
    ): ModelSelectionResult {

        val availableModels = AIConfiguration.getAvailableModels().keys
            .filter { it !in excludeModels }
            .filter { meetsRequirements(it, requirements) }

        if (availableModels.isEmpty()) {
            return ModelSelectionResult(
                selectedModel = null,
                reason = "No models available that meet requirements",
                confidence = 0.0f,
                fallbackOptions = emptyList()
            )
        }

        // Score each model based on multiple factors
        val modelScores = availableModels.map { model ->
            val score = calculateModelScore(model, requirements)
            model to score
        }.sortedByDescending { it.second }

        val bestModel = modelScores.first().first
        val bestScore = modelScores.first().second

        val fallbacks = modelScores.drop(1).take(3).map { it.first }

        return ModelSelectionResult(
            selectedModel = bestModel,
            reason = "Selected based on performance score: ${bestScore.toInt()}",
            confidence = (bestScore / 100f).coerceIn(0f, 1f),
            fallbackOptions = fallbacks,
            performanceStats = modelPerformanceStats[bestModel]
        )
    }

    /**
     * Calculate comprehensive score for model selection
     */
    private fun calculateModelScore(model: AIModel, requirements: ModelRequirements): Float {
        val stats = modelPerformanceStats[model] ?: return 0f
        val health = modelHealthStatus[model] ?: return 0f
        val config = AIConfiguration.getModelConfig(model) ?: return 0f

        var score = 0f

        // Performance factors (40% weight)
        score += calculatePerformanceScore(stats) * 0.4f

        // Health factors (25% weight)
        score += calculateHealthScore(health) * 0.25f

        // Cost efficiency (15% weight)
        score += calculateCostScore(config, requirements) * 0.15f

        // Capability match (20% weight)
        score += calculateCapabilityScore(config, requirements) * 0.2f

        return score.coerceIn(0f, 100f)
    }

    private fun calculatePerformanceScore(stats: ModelPerformanceStats): Float {
        val totalRequests = stats.totalRequests.get()
        if (totalRequests == 0L) return 50f // Default score for unused models

        val successRate = stats.successfulRequests.get().toFloat() / totalRequests
        val avgResponseTime = stats.totalResponseTimeMs.get().toFloat() / totalRequests
        val avgConfidence = stats.averageConfidence

        var score = 0f
        score += successRate * 40f // Success rate (0-40 points)
        score += maxOf(0f, 30f - (avgResponseTime / 1000f)) // Response time (0-30 points)
        score += avgConfidence * 30f // Confidence (0-30 points)

        return score.coerceIn(0f, 100f)
    }

    private fun calculateHealthScore(health: ModelHealthStatus): Float {
        var score = 100f

        if (!health.isHealthy) score -= 50f
        score -= health.consecutiveFailures * 10f
        score -= health.errorRate * 50f

        // Penalize models that haven't been checked recently
        val timeSinceCheck = System.currentTimeMillis() - health.lastHealthCheck
        if (timeSinceCheck > 60 * 60 * 1000L) score -= 20f // 1 hour

        return score.coerceIn(0f, 100f)
    }

    private fun calculateCostScore(config: ModelConfig, requirements: ModelRequirements): Float {
        val estimatedCost = config.costPerToken * requirements.estimatedTokens
        val budgetRatio = if (requirements.maxCostPerRequest > 0) {
            estimatedCost / requirements.maxCostPerRequest
        } else 1f

        return maxOf(0f, 100f - (budgetRatio * 100f)).coerceIn(0f, 100f)
    }

    private fun calculateCapabilityScore(config: ModelConfig, requirements: ModelRequirements): Float {
        val requiredCapabilities = requirements.requiredCapabilities
        val modelCapabilities = config.capabilities

        val matchedCapabilities = requiredCapabilities.intersect(modelCapabilities).size
        val totalRequired = requiredCapabilities.size

        return if (totalRequired > 0) {
            (matchedCapabilities.toFloat() / totalRequired) * 100f
        } else 100f
    }

    private fun meetsRequirements(model: AIModel, requirements: ModelRequirements): Boolean {
        val config = AIConfiguration.getModelConfig(model) ?: return false
        val health = modelHealthStatus[model] ?: return false

        // Check basic requirements
        if (!config.isEnabled) return false
        if (!health.isHealthy && requirements.requireHealthy) return false
        if (config.maxTokens < requirements.minTokens) return false

        // Check required capabilities
        if (!config.capabilities.containsAll(requirements.requiredCapabilities)) return false

        // Check performance requirements
        val stats = modelPerformanceStats[model]
        if (stats != null && requirements.minSuccessRate > 0) {
            val successRate = if (stats.totalRequests.get() > 0) {
                stats.successfulRequests.get().toFloat() / stats.totalRequests.get()
            } else 1f
            if (successRate < requirements.minSuccessRate) return false
        }

        return true
    }

    // ========== DYNAMIC MODEL SWITCHING ==========

    /**
     * Attempt intelligent model switch based on current conditions
     */
    suspend fun attemptIntelligentSwitch(
        currentModel: AIModel,
        reason: SwitchReason,
        requirements: ModelRequirements = ModelRequirements.default()
    ): SwitchResult {

        val selectionResult = selectOptimalModel(
            requirements = requirements,
            excludeModels = setOf(currentModel)
        )

        val newModel = selectionResult.selectedModel
        if (newModel == null) {
            return SwitchResult(
                success = false,
                newModel = currentModel,
                reason = "No alternative models available",
                switchEvent = null
            )
        }

        // Perform the switch
        val switchEvent = ModelSwitchEvent(
            fromModel = currentModel,
            toModel = newModel,
            reason = reason,
            timestamp = System.currentTimeMillis(),
            triggerConditions = mapOf(
                "currentModelHealth" to modelHealthStatus[currentModel]?.isHealthy.toString(),
                "newModelScore" to selectionResult.confidence.toString()
            ),
            success = true
        )

        switchingHistory.add(switchEvent)
        _activeModel.value = newModel

        // Update AIConfiguration
        AIConfiguration.setPrimaryModel(newModel)

        return SwitchResult(
            success = true,
            newModel = newModel,
            reason = selectionResult.reason,
            switchEvent = switchEvent
        )
    }

    /**
     * Get next fallback model using intelligent selection
     */
    suspend fun getIntelligentFallback(
        failedModel: AIModel,
        requirements: ModelRequirements = ModelRequirements.default()
    ): AIModel? {

        val selectionResult = selectOptimalModel(
            requirements = requirements,
            excludeModels = setOf(failedModel)
        )

        // Record the failure for future selection
        recordModelFailure(failedModel)

        return selectionResult.selectedModel
    }

    // ========== PERFORMANCE MONITORING ==========

    /**
     * Record successful model request
     */
    fun recordSuccess(
        model: AIModel,
        responseTimeMs: Long,
        tokensUsed: Int,
        cost: Double,
        confidence: Float
    ) {
        val stats = modelPerformanceStats[model] ?: return

        stats.totalRequests.incrementAndGet()
        stats.successfulRequests.incrementAndGet()
        stats.totalResponseTimeMs.addAndGet(responseTimeMs)
        stats.totalTokensUsed.addAndGet(tokensUsed.toLong())
        stats.totalCost += cost
        stats.lastUsed = System.currentTimeMillis()

        // Update rolling average confidence
        val totalSuccesses = stats.successfulRequests.get()
        stats.averageConfidence = ((stats.averageConfidence * (totalSuccesses - 1)) + confidence) / totalSuccesses

        // Update health status
        updateModelHealth(model, success = true, responseTimeMs)
    }

    /**
     * Record failed model request
     */
    fun recordFailure(model: AIModel, error: String, responseTimeMs: Long? = null) {
        val stats = modelPerformanceStats[model] ?: return

        stats.totalRequests.incrementAndGet()
        stats.failedRequests.incrementAndGet()
        responseTimeMs?.let { stats.totalResponseTimeMs.addAndGet(it) }
        stats.lastUsed = System.currentTimeMillis()

        updateModelHealth(model, success = false, responseTimeMs)
    }

    private fun recordModelFailure(model: AIModel) {
        recordFailure(model, "Model selection failure")
    }

    private fun updateModelHealth(model: AIModel, success: Boolean, responseTimeMs: Long?) {
        val health = modelHealthStatus[model] ?: return

        if (success) {
            health.consecutiveFailures = 0
            health.status = HealthStatusType.HEALTHY
        } else {
            health.consecutiveFailures++
            health.status = when {
                health.consecutiveFailures >= 5 -> HealthStatusType.UNHEALTHY
                health.consecutiveFailures >= 3 -> HealthStatusType.DEGRADED
                else -> HealthStatusType.WARNING
            }
        }

        health.isHealthy = health.consecutiveFailures < 3
        health.lastHealthCheck = System.currentTimeMillis()

        // Calculate error rate over recent window
        val stats = modelPerformanceStats[model]
        if (stats != null && stats.totalRequests.get() > 0) {
            health.errorRate = stats.failedRequests.get().toFloat() / stats.totalRequests.get()
        }

        // Update response time percentile
        responseTimeMs?.let {
            // Simplified percentile calculation - in production would use proper percentile tracking
            health.responseTimePercentile95 = maxOf(health.responseTimePercentile95, it)
        }

        // Emit health update
        CoroutineScope(Dispatchers.IO).launch {
            _modelHealthUpdates.emit(
                ModelHealthUpdate(
                    model = model,
                    oldStatus = health.status,
                    newStatus = health.status,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private suspend fun performHealthChecks() {
        modelHealthStatus.forEach { (model, health) ->
            // Simple health check - in production would ping actual endpoints
            val timeSinceLastUse = System.currentTimeMillis() - (modelPerformanceStats[model]?.lastUsed ?: 0L)

            // Mark models as stale if not used recently
            if (timeSinceLastUse > 2 * 60 * 60 * 1000L) { // 2 hours
                health.status = HealthStatusType.STALE
            }
        }
    }

    // ========== ANALYTICS AND REPORTING ==========

    /**
     * Get comprehensive performance report
     */
    fun getPerformanceReport(): ModelPerformanceReport {
        val modelReports = modelPerformanceStats.map { (model, stats) ->
            val health = modelHealthStatus[model]!!
            val config = AIConfiguration.getModelConfig(model)

            IndividualModelReport(
                model = model,
                stats = stats,
                health = health,
                efficiency = calculateEfficiency(stats),
                costEfficiency = calculateCostEfficiency(stats, config),
                recommendedUse = getRecommendedUse(model, stats, health)
            )
        }

        return ModelPerformanceReport(
            reportTimestamp = System.currentTimeMillis(),
            modelReports = modelReports,
            switchingHistory = switchingHistory.takeLast(50),
            recommendations = generateRecommendations(modelReports)
        )
    }

    private fun calculateEfficiency(stats: ModelPerformanceStats): Float {
        val totalRequests = stats.totalRequests.get()
        if (totalRequests == 0L) return 0f

        val successRate = stats.successfulRequests.get().toFloat() / totalRequests
        val avgResponseTime = stats.totalResponseTimeMs.get().toFloat() / totalRequests
        val responseEfficiency = maxOf(0f, 1f - (avgResponseTime / 10000f)) // Normalize to 10 seconds

        return (successRate + responseEfficiency) / 2f
    }

    private fun calculateCostEfficiency(stats: ModelPerformanceStats, config: ModelConfig?): Float {
        if (config == null || stats.totalTokensUsed.get() == 0L) return 0f

        val avgTokensPerRequest = stats.totalTokensUsed.get().toFloat() / stats.totalRequests.get()
        val avgCostPerRequest = stats.totalCost / stats.totalRequests.get()

        // Compare to baseline efficient model cost
        val baselineCost = 0.002f // $0.002 per request baseline
        return maxOf(0f, 1f - (avgCostPerRequest.toFloat() / baselineCost))
    }

    private fun getRecommendedUse(model: AIModel, stats: ModelPerformanceStats, health: ModelHealthStatus): String {
        return when {
            !health.isHealthy -> "Not recommended - health issues"
            stats.averageConfidence > 0.8f && calculateEfficiency(stats) > 0.8f -> "Highly recommended for all tasks"
            stats.averageConfidence > 0.6f -> "Good for standard tasks"
            calculateEfficiency(stats) > 0.7f -> "Good for time-sensitive tasks"
            else -> "Use as fallback only"
        }
    }

    private fun generateRecommendations(reports: List<IndividualModelReport>): List<String> {
        val recommendations = mutableListOf<String>()

        val bestModel = reports.maxByOrNull { it.efficiency }
        val worstModel = reports.minByOrNull { it.efficiency }

        bestModel?.let {
            recommendations.add("Consider using ${it.model} as primary model (efficiency: ${(it.efficiency * 100).toInt()}%)")
        }

        worstModel?.let {
            if (it.efficiency < 0.3f) {
                recommendations.add("Consider disabling ${it.model} due to poor performance")
            }
        }

        val unhealthyModels = reports.filter { !it.health.isHealthy }
        if (unhealthyModels.isNotEmpty()) {
            recommendations.add("Health check needed for: ${unhealthyModels.map { it.model }.joinToString()}")
        }

        return recommendations
    }

    /**
     * Reset all performance statistics
     */
    fun resetPerformanceStats() {
        modelPerformanceStats.clear()
        switchingHistory.clear()
        initializePerformanceTracking()
    }

    /**
     * Export performance data for analysis
     */
    fun exportPerformanceData(): String {
        return getPerformanceReport().toString()
    }
}

// ========== DATA CLASSES ==========

data class ModelRequirements(
    val requiredCapabilities: Set<ModelCapability> = emptySet(),
    val minTokens: Int = 100,
    val estimatedTokens: Int = 1000,
    val maxCostPerRequest: Double = 0.05, // $0.05
    val requireHealthy: Boolean = true,
    val minSuccessRate: Float = 0.8f,
    val maxResponseTimeMs: Long = 10000L,
    val prioritizeSpeed: Boolean = false,
    val prioritizeCost: Boolean = false,
    val prioritizeQuality: Boolean = true
) {
    companion object {
        fun default() = ModelRequirements()

        fun sleepAnalysis() = ModelRequirements(
            requiredCapabilities = setOf(ModelCapability.SLEEP_ANALYSIS),
            estimatedTokens = 800,
            maxCostPerRequest = 0.01,
            prioritizeQuality = true
        )

        fun realTime() = ModelRequirements(
            estimatedTokens = 300,
            maxResponseTimeMs = 3000L,
            prioritizeSpeed = true
        )
    }
}

data class ModelSelectionResult(
    val selectedModel: AIModel?,
    val reason: String,
    val confidence: Float,
    val fallbackOptions: List<AIModel>,
    val performanceStats: ModelPerformanceStats? = null
)

data class SwitchResult(
    val success: Boolean,
    val newModel: AIModel,
    val reason: String,
    val switchEvent: ModelSwitchEvent?
)

data class ModelPerformanceStats(
    val model: AIModel,
    val totalRequests: AtomicLong,
    val successfulRequests: AtomicLong,
    val failedRequests: AtomicLong,
    val totalResponseTimeMs: AtomicLong,
    val totalTokensUsed: AtomicLong,
    var totalCost: Double,
    var averageConfidence: Float,
    var lastUsed: Long
)

data class ModelHealthStatus(
    val model: AIModel,
    var isHealthy: Boolean,
    var lastHealthCheck: Long,
    var consecutiveFailures: Int,
    var responseTimePercentile95: Long,
    var errorRate: Float,
    var status: HealthStatusType
)

data class ModelSwitchEvent(
    val fromModel: AIModel,
    val toModel: AIModel,
    val reason: SwitchReason,
    val timestamp: Long,
    val triggerConditions: Map<String, String>,
    val success: Boolean
)

data class ModelHealthUpdate(
    val model: AIModel,
    val oldStatus: HealthStatusType,
    val newStatus: HealthStatusType,
    val timestamp: Long
)

data class ModelPerformanceReport(
    val reportTimestamp: Long,
    val modelReports: List<IndividualModelReport>,
    val switchingHistory: List<ModelSwitchEvent>,
    val recommendations: List<String>
)

data class IndividualModelReport(
    val model: AIModel,
    val stats: ModelPerformanceStats,
    val health: ModelHealthStatus,
    val efficiency: Float,
    val costEfficiency: Float,
    val recommendedUse: String
)

// ========== ENUMS ==========

enum class SwitchReason {
    PERFORMANCE_DEGRADATION,
    HEALTH_CHECK_FAILURE,
    COST_OPTIMIZATION,
    USER_REQUEST,
    AUTOMATIC_OPTIMIZATION,
    FALLBACK_TRIGGERED,
    CAPABILITY_REQUIREMENT
}

enum class HealthStatusType {
    HEALTHY,
    WARNING,
    DEGRADED,
    UNHEALTHY,
    STALE,
    UNKNOWN
}