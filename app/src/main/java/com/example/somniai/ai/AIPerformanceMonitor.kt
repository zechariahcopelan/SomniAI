package com.example.somniai.ai

import com.example.somniai.ai.AIConstants.*
import com.example.somniai.utils.TimeUtils
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.*

/**
 * AI Performance Monitor for SomniAI
 *
 * Comprehensive performance monitoring system that tracks AI processing times,
 * token usage, costs, and overall system performance. Provides real-time metrics,
 * historical analysis, and optimization recommendations.
 */
object AIPerformanceMonitor {

    // ========== PERFORMANCE STATE ==========

    private val performanceMetrics = ConcurrentHashMap<String, PerformanceMetric>()
    private val sessionMetrics = ConcurrentHashMap<String, AISessionMetrics>()
    private val modelPerformance = ConcurrentHashMap<AIModel, ModelPerformanceStats>()
    private val costTracking = AtomicReference(CostTrackingData())

    private val _performanceUpdates = MutableSharedFlow<PerformanceUpdate>()
    val performanceUpdates: SharedFlow<PerformanceUpdate> = _performanceUpdates.asSharedFlow()

    private val _alertsFlow = MutableSharedFlow<PerformanceAlert>()
    val alertsFlow: SharedFlow<PerformanceAlert> = _alertsFlow.asSharedFlow()

    // Performance history (circular buffer)
    private val performanceHistory = mutableListOf<PerformanceSnapshot>()
    private val maxHistorySize = 1000

    // Monitoring state
    private var isMonitoringActive = true
    private var monitoringStartTime = System.currentTimeMillis()

    // ========== PERFORMANCE TRACKING ==========

    /**
     * Start tracking an AI operation
     */
    fun startOperation(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        sessionId: String? = null,
        metadata: Map<String, String> = emptyMap()
    ): PerformanceTracker {
        if (!isMonitoringActive) return NoOpTracker()

        val tracker = PerformanceTracker(
            operationId = operationId,
            operationType = operationType,
            model = model,
            sessionId = sessionId,
            startTime = System.currentTimeMillis(),
            metadata = metadata
        )

        // Store session tracking if provided
        sessionId?.let { id ->
            sessionMetrics[id] = sessionMetrics.getOrDefault(id, AISessionMetrics(id))
                .copy(activeOperations = sessionMetrics[id]?.activeOperations?.plus(operationId) ?: listOf(operationId))
        }

        return tracker
    }

    /**
     * Record completed AI operation performance
     */
    fun recordOperation(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        processingTimeMs: Long,
        tokenUsage: TokenUsage,
        success: Boolean,
        confidenceScore: Float? = null,
        errorMessage: String? = null,
        sessionId: String? = null,
        metadata: Map<String, String> = emptyMap()
    ) {
        if (!isMonitoringActive) return

        val timestamp = System.currentTimeMillis()
        val cost = calculateCost(tokenUsage, model)

        // Update operation metrics
        val metricKey = "${operationType}_${model}"
        val metric = performanceMetrics.getOrPut(metricKey) {
            PerformanceMetric(
                key = metricKey,
                operationType = operationType,
                model = model,
                totalOperations = AtomicLong(0),
                successfulOperations = AtomicLong(0),
                totalProcessingTime = AtomicLong(0),
                totalTokensUsed = AtomicLong(0),
                totalCost = 0.0,
                averageProcessingTime = 0L,
                averageTokensUsed = 0,
                averageConfidence = 0f,
                successRate = 0f,
                lastUpdated = timestamp
            )
        }

        // Update metric counters
        metric.totalOperations.incrementAndGet()
        if (success) metric.successfulOperations.incrementAndGet()
        metric.totalProcessingTime.addAndGet(processingTimeMs)
        metric.totalTokensUsed.addAndGet(tokenUsage.totalTokens.toLong())
        metric.totalCost += cost

        // Calculate running averages
        val totalOps = metric.totalOperations.get()
        metric.averageProcessingTime = metric.totalProcessingTime.get() / totalOps
        metric.averageTokensUsed = (metric.totalTokensUsed.get() / totalOps).toInt()
        metric.successRate = metric.successfulOperations.get().toFloat() / totalOps

        // Update confidence (running average)
        confidenceScore?.let { confidence ->
            val currentConfidence = metric.averageConfidence
            metric.averageConfidence = ((currentConfidence * (totalOps - 1)) + confidence) / totalOps
        }

        metric.lastUpdated = timestamp

        // Update model-specific performance
        updateModelPerformance(model, processingTimeMs, tokenUsage, success, confidenceScore, cost)

        // Update session metrics
        sessionId?.let { id ->
            updateSessionMetrics(id, operationId, operationType, processingTimeMs, tokenUsage, success, cost)
        }

        // Update cost tracking
        updateCostTracking(cost, tokenUsage)

        // Record performance snapshot
        recordPerformanceSnapshot(operationType, model, processingTimeMs, tokenUsage, success, timestamp)

        // Check for performance alerts
        checkPerformanceAlerts(operationType, model, processingTimeMs, success, cost)

        // Emit performance update
        CoroutineScope(Dispatchers.IO).launch {
            _performanceUpdates.emit(
                PerformanceUpdate(
                    operationId = operationId,
                    operationType = operationType,
                    model = model,
                    processingTime = processingTimeMs,
                    tokenUsage = tokenUsage,
                    success = success,
                    cost = cost,
                    timestamp = timestamp
                )
            )
        }
    }

    /**
     * Record batch operation performance
     */
    fun recordBatchOperation(
        batchId: String,
        operationType: AIOperationType,
        model: AIModel,
        operations: List<BatchOperationResult>,
        totalProcessingTimeMs: Long,
        metadata: Map<String, String> = emptyMap()
    ) {
        if (!isMonitoringActive || operations.isEmpty()) return

        val totalTokens = operations.sumOf { it.tokenUsage.totalTokens }
        val successfulOps = operations.count { it.success }
        val avgConfidence = operations.mapNotNull { it.confidenceScore }.average().toFloat()
        val totalCost = operations.sumOf { it.cost }

        val aggregatedTokenUsage = TokenUsage(
            promptTokens = operations.sumOf { it.tokenUsage.promptTokens },
            completionTokens = operations.sumOf { it.tokenUsage.completionTokens },
            totalTokens = totalTokens
        )

        // Record as single operation with batch metadata
        recordOperation(
            operationId = batchId,
            operationType = operationType,
            model = model,
            processingTimeMs = totalProcessingTimeMs,
            tokenUsage = aggregatedTokenUsage,
            success = successfulOps == operations.size,
            confidenceScore = if (avgConfidence.isNaN()) null else avgConfidence,
            metadata = metadata + mapOf(
                "batch_size" to operations.size.toString(),
                "batch_success_rate" to (successfulOps.toFloat() / operations.size).toString()
            )
        )
    }

    // ========== PERFORMANCE ANALYSIS ==========

    /**
     * Get current performance metrics
     */
    fun getCurrentMetrics(): Map<String, PerformanceMetric> {
        return performanceMetrics.toMap()
    }

    /**
     * Get model-specific performance statistics
     */
    fun getModelPerformance(): Map<AIModel, ModelPerformanceStats> {
        return modelPerformance.toMap()
    }

    /**
     * Get session-specific metrics
     */
    fun getSessionMetrics(sessionId: String): AISessionMetrics? {
        return sessionMetrics[sessionId]
    }

    /**
     * Get comprehensive performance report
     */
    fun generatePerformanceReport(
        timeRangeMs: Long = 24 * 60 * 60 * 1000L, // Default: last 24 hours
        includeDetails: Boolean = true
    ): PerformanceReport {
        val endTime = System.currentTimeMillis()
        val startTime = endTime - timeRangeMs

        val relevantSnapshots = performanceHistory.filter { it.timestamp >= startTime }
        val relevantMetrics = performanceMetrics.values.filter { it.lastUpdated >= startTime }

        return PerformanceReport(
            reportId = "PERF-${endTime}",
            timeRange = TimeRange(startTime, endTime),
            generatedAt = endTime,

            // Overall statistics
            totalOperations = relevantSnapshots.size,
            successfulOperations = relevantSnapshots.count { it.success },
            totalProcessingTime = relevantSnapshots.sumOf { it.processingTimeMs },
            totalTokensUsed = relevantSnapshots.sumOf { it.tokenUsage.totalTokens.toLong() },
            totalCost = relevantSnapshots.sumOf { it.cost },

            // Performance metrics
            averageProcessingTime = if (relevantSnapshots.isNotEmpty()) {
                relevantSnapshots.map { it.processingTimeMs }.average().toLong()
            } else 0L,

            averageTokensPerOperation = if (relevantSnapshots.isNotEmpty()) {
                relevantSnapshots.map { it.tokenUsage.totalTokens }.average().toInt()
            } else 0,

            overallSuccessRate = if (relevantSnapshots.isNotEmpty()) {
                relevantSnapshots.count { it.success }.toFloat() / relevantSnapshots.size
            } else 0f,

            // Model breakdown
            modelMetrics = if (includeDetails) modelPerformance.toMap() else emptyMap(),
            operationMetrics = if (includeDetails) relevantMetrics.associateBy { it.key } else emptyMap(),

            // Trends and insights
            performanceTrends = analyzePerformanceTrends(relevantSnapshots),
            bottlenecks = identifyBottlenecks(relevantMetrics),
            optimizationOpportunities = identifyOptimizationOpportunities(relevantMetrics, relevantSnapshots),
            costAnalysis = analyzeCostEfficiency(relevantSnapshots),

            // Quality metrics
            averageConfidence = relevantSnapshots.mapNotNull { it.confidenceScore }.average().toFloat(),
            qualityDistribution = analyzeQualityDistribution(relevantSnapshots),

            // Recommendations
            recommendations = generatePerformanceRecommendations(relevantMetrics, relevantSnapshots)
        )
    }

    /**
     * Get real-time performance dashboard data
     */
    fun getDashboardMetrics(): PerformanceDashboard {
        val recentWindow = 5 * 60 * 1000L // Last 5 minutes
        val currentTime = System.currentTimeMillis()

        val recentSnapshots = performanceHistory.filter {
            it.timestamp >= currentTime - recentWindow
        }

        val activeOperations = sessionMetrics.values.flatMap { it.activeOperations }.size
        val currentCost = costTracking.get()

        return PerformanceDashboard(
            timestamp = currentTime,
            activeOperations = activeOperations,
            operationsPerMinute = if (recentSnapshots.isNotEmpty()) {
                (recentSnapshots.size / 5.0).toFloat()
            } else 0f,

            averageResponseTime = if (recentSnapshots.isNotEmpty()) {
                recentSnapshots.map { it.processingTimeMs }.average().toLong()
            } else 0L,

            successRate = if (recentSnapshots.isNotEmpty()) {
                recentSnapshots.count { it.success }.toFloat() / recentSnapshots.size
            } else 1f,

            tokenUsageRate = if (recentSnapshots.isNotEmpty()) {
                recentSnapshots.sumOf { it.tokenUsage.totalTokens } / 5.0 // Per minute
            } else 0.0,

            currentCostPerHour = currentCost.costPerHour,
            todaysCost = currentCost.todaysCost,

            topModels = modelPerformance.entries
                .sortedByDescending { it.value.totalOperations.get() }
                .take(3)
                .associate { it.key to it.value },

            alerts = getActiveAlerts()
        )
    }

    // ========== COST TRACKING ==========

    /**
     * Get current cost tracking data
     */
    fun getCostTracking(): CostTrackingData {
        return costTracking.get()
    }

    /**
     * Set cost budget and limits
     */
    fun setCostLimits(
        dailyLimit: Double,
        monthlyLimit: Double,
        operationLimit: Double
    ) {
        val current = costTracking.get()
        costTracking.set(
            current.copy(
                dailyCostLimit = dailyLimit,
                monthlyCostLimit = monthlyLimit,
                operationCostLimit = operationLimit
            )
        )
    }

    /**
     * Check if operation would exceed cost limits
     */
    fun checkCostLimits(estimatedCost: Double): CostLimitCheck {
        val current = costTracking.get()
        val wouldExceedDaily = current.todaysCost + estimatedCost > current.dailyCostLimit
        val wouldExceedMonthly = current.monthCost + estimatedCost > current.monthlyCostLimit
        val wouldExceedOperation = estimatedCost > current.operationCostLimit

        return CostLimitCheck(
            withinLimits = !wouldExceedDaily && !wouldExceedMonthly && !wouldExceedOperation,
            wouldExceedDaily = wouldExceedDaily,
            wouldExceedMonthly = wouldExceedMonthly,
            wouldExceedOperation = wouldExceedOperation,
            estimatedCost = estimatedCost,
            remainingDailyBudget = maxOf(0.0, current.dailyCostLimit - current.todaysCost),
            remainingMonthlyBudget = maxOf(0.0, current.monthlyCostLimit - current.monthCost)
        )
    }

    // ========== ALERTS AND MONITORING ==========

    /**
     * Configure performance alert thresholds
     */
    fun configureAlerts(config: AlertConfiguration) {
        // Store alert configuration and start monitoring
        CoroutineScope(Dispatchers.IO).launch {
            monitorPerformanceAlerts(config)
        }
    }

    private suspend fun monitorPerformanceAlerts(config: AlertConfiguration) {
        while (isMonitoringActive) {
            delay(config.checkIntervalMs)

            // Check various alert conditions
            checkProcessingTimeAlerts(config)
            checkSuccessRateAlerts(config)
            checkCostAlerts(config)
            checkTokenUsageAlerts(config)
        }
    }

    private suspend fun checkProcessingTimeAlerts(config: AlertConfiguration) {
        performanceMetrics.values.forEach { metric ->
            if (metric.averageProcessingTime > config.maxProcessingTimeMs) {
                _alertsFlow.emit(
                    PerformanceAlert(
                        type = AlertType.SLOW_PROCESSING,
                        severity = AlertSeverity.WARNING,
                        message = "Average processing time (${metric.averageProcessingTime}ms) exceeds threshold",
                        metric = metric.key,
                        value = metric.averageProcessingTime.toDouble(),
                        threshold = config.maxProcessingTimeMs.toDouble(),
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }
    }

    private suspend fun checkSuccessRateAlerts(config: AlertConfiguration) {
        performanceMetrics.values.forEach { metric ->
            if (metric.successRate < config.minSuccessRate && metric.totalOperations.get() >= 10) {
                _alertsFlow.emit(
                    PerformanceAlert(
                        type = AlertType.LOW_SUCCESS_RATE,
                        severity = AlertSeverity.CRITICAL,
                        message = "Success rate (${(metric.successRate * 100).toInt()}%) below threshold",
                        metric = metric.key,
                        value = metric.successRate.toDouble(),
                        threshold = config.minSuccessRate.toDouble(),
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }
    }

    private suspend fun checkCostAlerts(config: AlertConfiguration) {
        val current = costTracking.get()

        if (current.todaysCost > current.dailyCostLimit * 0.8) {
            _alertsFlow.emit(
                PerformanceAlert(
                    type = AlertType.HIGH_COST,
                    severity = if (current.todaysCost > current.dailyCostLimit) AlertSeverity.CRITICAL else AlertSeverity.WARNING,
                    message = "Daily cost approaching limit: $${String.format("%.2f", current.todaysCost)}",
                    value = current.todaysCost,
                    threshold = current.dailyCostLimit,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private suspend fun checkTokenUsageAlerts(config: AlertConfiguration) {
        val recentUsage = performanceHistory.takeLast(10).sumOf { it.tokenUsage.totalTokens }

        if (recentUsage > config.maxTokensPerMinute * 10) { // 10-operation window
            _alertsFlow.emit(
                PerformanceAlert(
                    type = AlertType.HIGH_TOKEN_USAGE,
                    severity = AlertSeverity.WARNING,
                    message = "High token usage detected: $recentUsage tokens in recent operations",
                    value = recentUsage.toDouble(),
                    threshold = (config.maxTokensPerMinute * 10).toDouble(),
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    // ========== UTILITY METHODS ==========

    /**
     * Reset all performance metrics
     */
    fun resetMetrics() {
        performanceMetrics.clear()
        sessionMetrics.clear()
        modelPerformance.clear()
        performanceHistory.clear()
        costTracking.set(CostTrackingData())
        monitoringStartTime = System.currentTimeMillis()
    }

    /**
     * Enable/disable monitoring
     */
    fun setMonitoringEnabled(enabled: Boolean) {
        isMonitoringActive = enabled
        if (enabled && monitoringStartTime == 0L) {
            monitoringStartTime = System.currentTimeMillis()
        }
    }

    /**
     * Export performance data for analysis
     */
    fun exportPerformanceData(): String {
        val report = generatePerformanceReport(includeDetails = true)
        return "Performance Data Export\n" +
                "======================\n" +
                "Generated: ${TimeUtils.formatDateTime(System.currentTimeMillis())}\n" +
                "Total Operations: ${report.totalOperations}\n" +
                "Success Rate: ${(report.overallSuccessRate * 100).toInt()}%\n" +
                "Total Cost: $${String.format("%.2f", report.totalCost)}\n" +
                "Average Processing Time: ${report.averageProcessingTime}ms\n" +
                "\nDetailed metrics available via API"
    }

    // ========== PRIVATE HELPER METHODS ==========

    private fun calculateCost(tokenUsage: TokenUsage, model: AIModel): Double {
        val modelConfig = AIConfiguration.getModelConfig(model)
        return if (modelConfig != null) {
            tokenUsage.totalTokens * modelConfig.costPerToken
        } else {
            tokenUsage.totalTokens * ESTIMATED_COST_PER_TOKEN
        }
    }

    private fun updateModelPerformance(
        model: AIModel,
        processingTimeMs: Long,
        tokenUsage: TokenUsage,
        success: Boolean,
        confidenceScore: Float?,
        cost: Double
    ) {
        val stats = modelPerformance.getOrPut(model) {
            ModelPerformanceStats(
                model = model,
                totalOperations = AtomicLong(0),
                successfulOperations = AtomicLong(0),
                totalResponseTimeMs = AtomicLong(0),
                totalTokensUsed = AtomicLong(0),
                totalCost = 0.0,
                averageConfidence = 0.0f,
                lastUsed = System.currentTimeMillis()
            )
        }

        stats.totalOperations.incrementAndGet()
        if (success) stats.successfulOperations.incrementAndGet()
        stats.totalResponseTimeMs.addAndGet(processingTimeMs)
        stats.totalTokensUsed.addAndGet(tokenUsage.totalTokens.toLong())
        stats.totalCost += cost
        stats.lastUsed = System.currentTimeMillis()

        // Update running average confidence
        confidenceScore?.let { confidence ->
            val totalOps = stats.totalOperations.get()
            stats.averageConfidence = ((stats.averageConfidence * (totalOps - 1)) + confidence) / totalOps
        }
    }

    private fun updateSessionMetrics(
        sessionId: String,
        operationId: String,
        operationType: AIOperationType,
        processingTimeMs: Long,
        tokenUsage: TokenUsage,
        success: Boolean,
        cost: Double
    ) {
        val metrics = sessionMetrics.getOrPut(sessionId) { AISessionMetrics(sessionId) }

        // Remove from active operations
        val updatedActiveOps = metrics.activeOperations.filter { it != operationId }

        sessionMetrics[sessionId] = metrics.copy(
            activeOperations = updatedActiveOps,
            totalOperations = metrics.totalOperations + 1,
            successfulOperations = if (success) metrics.successfulOperations + 1 else metrics.successfulOperations,
            totalProcessingTime = metrics.totalProcessingTime + processingTimeMs,
            totalTokensUsed = metrics.totalTokensUsed + tokenUsage.totalTokens,
            totalCost = metrics.totalCost + cost,
            lastActivity = System.currentTimeMillis()
        )
    }

    private fun updateCostTracking(cost: Double, tokenUsage: TokenUsage) {
        val current = costTracking.get()
        val now = System.currentTimeMillis()

        // Calculate hourly rate based on recent activity
        val hourlyRate = if (now - current.lastUpdate > 0) {
            val hoursElapsed = (now - current.lastUpdate) / (1000.0 * 60 * 60)
            if (hoursElapsed > 0) cost / hoursElapsed else 0.0
        } else 0.0

        costTracking.set(
            current.copy(
                todaysCost = current.todaysCost + cost,
                monthCost = current.monthCost + cost,
                totalCost = current.totalCost + cost,
                costPerHour = ((current.costPerHour * 0.9) + (hourlyRate * 0.1)), // Exponential smoothing
                totalTokensUsed = current.totalTokensUsed + tokenUsage.totalTokens,
                lastUpdate = now
            )
        )
    }

    private fun recordPerformanceSnapshot(
        operationType: AIOperationType,
        model: AIModel,
        processingTimeMs: Long,
        tokenUsage: TokenUsage,
        success: Boolean,
        timestamp: Long
    ) {
        val snapshot = PerformanceSnapshot(
            timestamp = timestamp,
            operationType = operationType,
            model = model,
            processingTimeMs = processingTimeMs,
            tokenUsage = tokenUsage,
            success = success,
            cost = calculateCost(tokenUsage, model),
            confidenceScore = null // Will be updated separately if available
        )

        synchronized(performanceHistory) {
            performanceHistory.add(snapshot)
            if (performanceHistory.size > maxHistorySize) {
                performanceHistory.removeAt(0)
            }
        }
    }

    private fun checkPerformanceAlerts(
        operationType: AIOperationType,
        model: AIModel,
        processingTimeMs: Long,
        success: Boolean,
        cost: Double
    ) {
        // Immediate alert checks
        if (processingTimeMs > SLOW_PROCESSING_THRESHOLD_MS) {
            CoroutineScope(Dispatchers.IO).launch {
                _alertsFlow.emit(
                    PerformanceAlert(
                        type = AlertType.SLOW_PROCESSING,
                        severity = AlertSeverity.WARNING,
                        message = "Operation took ${processingTimeMs}ms (above threshold)",
                        metric = "${operationType}_${model}",
                        value = processingTimeMs.toDouble(),
                        threshold = SLOW_PROCESSING_THRESHOLD_MS.toDouble(),
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        if (!success) {
            CoroutineScope(Dispatchers.IO).launch {
                _alertsFlow.emit(
                    PerformanceAlert(
                        type = AlertType.OPERATION_FAILURE,
                        severity = AlertSeverity.ERROR,
                        message = "Operation failed for $operationType using $model",
                        metric = "${operationType}_${model}",
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }
    }

    private fun analyzePerformanceTrends(snapshots: List<PerformanceSnapshot>): Map<String, TrendData> {
        if (snapshots.size < 2) return emptyMap()

        val trends = mutableMapOf<String, TrendData>()

        // Processing time trend
        val processingTimes = snapshots.map { it.processingTimeMs.toDouble() }
        trends["processing_time"] = calculateTrend(processingTimes)

        // Cost trend
        val costs = snapshots.map { it.cost }
        trends["cost"] = calculateTrend(costs)

        // Success rate trend (sliding window)
        val windowSize = minOf(20, snapshots.size / 4)
        val successRates = snapshots.windowed(windowSize, step = windowSize).map { window ->
            window.count { it.success }.toDouble() / window.size
        }
        if (successRates.size >= 2) {
            trends["success_rate"] = calculateTrend(successRates)
        }

        return trends
    }

    private fun calculateTrend(values: List<Double>): TrendData {
        if (values.size < 2) return TrendData(TrendDirection.STABLE, 0.0, 0f)

        val firstHalf = values.take(values.size / 2).average()
        val secondHalf = values.drop(values.size / 2).average()

        val percentChange = if (firstHalf != 0.0) {
            ((secondHalf - firstHalf) / firstHalf) * 100
        } else 0.0

        val direction = when {
            abs(percentChange) < 5 -> TrendDirection.STABLE
            percentChange > 0 -> TrendDirection.IMPROVING
            else -> TrendDirection.DECLINING
        }

        val correlation = calculateCorrelation(values.indices.map { it.toDouble() }, values)

        return TrendData(direction, percentChange, correlation)
    }

    private fun calculateCorrelation(x: List<Double>, y: List<Double>): Float {
        if (x.size != y.size || x.size < 2) return 0f

        val n = x.size
        val sumX = x.sum()
        val sumY = y.sum()
        val sumXY = x.zip(y) { xi, yi -> xi * yi }.sum()
        val sumX2 = x.sumOf { it * it }
        val sumY2 = y.sumOf { it * it }

        val numerator = n * sumXY - sumX * sumY
        val denominator = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))

        return if (denominator != 0.0) (numerator / denominator).toFloat() else 0f
    }

    private fun identifyBottlenecks(metrics: List<PerformanceMetric>): List<PerformanceBottleneck> {
        return metrics.filter { metric ->
            metric.averageProcessingTime > ACCEPTABLE_PROCESSING_THRESHOLD_MS ||
                    metric.successRate < 0.9f ||
                    metric.averageTokensUsed > 3000
        }.map { metric ->
            val issues = mutableListOf<String>()
            if (metric.averageProcessingTime > ACCEPTABLE_PROCESSING_THRESHOLD_MS) {
                issues.add("High processing time")
            }
            if (metric.successRate < 0.9f) {
                issues.add("Low success rate")
            }
            if (metric.averageTokensUsed > 3000) {
                issues.add("High token usage")
            }

            PerformanceBottleneck(
                metric = metric.key,
                issues = issues,
                severity = if (metric.successRate < 0.8f) BottleneckSeverity.HIGH else BottleneckSeverity.MEDIUM,
                impact = calculateBottleneckImpact(metric),
                recommendations = generateBottleneckRecommendations(metric, issues)
            )
        }
    }

    private fun identifyOptimizationOpportunities(
        metrics: List<PerformanceMetric>,
        snapshots: List<PerformanceSnapshot>
    ): List<OptimizationOpportunity> {
        val opportunities = mutableListOf<OptimizationOpportunity>()

        // High-cost operations
        val highCostModels = snapshots.groupBy { it.model }
            .filter { (_, ops) -> ops.map { it.cost }.average() > 0.01 }
            .keys

        if (highCostModels.isNotEmpty()) {
            opportunities.add(
                OptimizationOpportunity(
                    type = OptimizationType.COST_OPTIMIZATION,
                    description = "Consider switching to lower-cost models for routine operations",
                    potentialSavings = "20-50% cost reduction",
                    implementationEffort = ImplementationEffort.MEDIUM,
                    affectedModels = highCostModels.toList()
                )
            )
        }

        // Slow operations
        val slowOperations = metrics.filter { it.averageProcessingTime > FAST_PROCESSING_THRESHOLD_MS * 2 }
        if (slowOperations.isNotEmpty()) {
            opportunities.add(
                OptimizationOpportunity(
                    type = OptimizationType.PERFORMANCE_OPTIMIZATION,
                    description = "Optimize slow operations through caching or model switching",
                    potentialSavings = "30-60% faster response times",
                    implementationEffort = ImplementationEffort.HIGH,
                    affectedOperations = slowOperations.map { it.operationType }
                )
            )
        }

        return opportunities
    }

    private fun analyzeCostEfficiency(snapshots: List<PerformanceSnapshot>): CostAnalysis {
        if (snapshots.isEmpty()) return CostAnalysis()

        val totalCost = snapshots.sumOf { it.cost }
        val successfulOps = snapshots.count { it.success }
        val costPerSuccess = if (successfulOps > 0) totalCost / successfulOps else 0.0

        val modelCosts = snapshots.groupBy { it.model }
            .mapValues { (_, ops) -> ops.sumOf { it.cost } }

        val mostExpensiveModel = modelCosts.maxByOrNull { it.value }?.key
        val mostEfficientModel = snapshots.groupBy { it.model }
            .mapValues { (_, ops) ->
                val cost = ops.sumOf { it.cost }
                val successes = ops.count { it.success }
                if (successes > 0) cost / successes else Double.MAX_VALUE
            }
            .minByOrNull { it.value }?.key

        return CostAnalysis(
            totalCost = totalCost,
            costPerOperation = totalCost / snapshots.size,
            costPerSuccess = costPerSuccess,
            costByModel = modelCosts,
            mostExpensiveModel = mostExpensiveModel,
            mostEfficientModel = mostEfficientModel,
            efficiency = if (totalCost > 0) successfulOps / totalCost else 0.0
        )
    }

    private fun analyzeQualityDistribution(snapshots: List<PerformanceSnapshot>): Map<String, Int> {
        val qualitySnapshots = snapshots.filter { it.confidenceScore != null }
        if (qualitySnapshots.isEmpty()) return emptyMap()

        return qualitySnapshots.groupBy { snapshot ->
            when {
                snapshot.confidenceScore!! >= 0.9f -> "Excellent"
                snapshot.confidenceScore!! >= 0.8f -> "Good"
                snapshot.confidenceScore!! >= 0.7f -> "Fair"
                snapshot.confidenceScore!! >= 0.6f -> "Poor"
                else -> "Very Poor"
            }
        }.mapValues { it.value.size }
    }

    private fun generatePerformanceRecommendations(
        metrics: List<PerformanceMetric>,
        snapshots: List<PerformanceSnapshot>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        val avgProcessingTime = metrics.map { it.averageProcessingTime }.average()
        if (avgProcessingTime > ACCEPTABLE_PROCESSING_THRESHOLD_MS) {
            recommendations.add("Consider optimizing processing pipeline - average response time is ${avgProcessingTime.toLong()}ms")
        }

        val overallSuccessRate = metrics.map { it.successRate }.average()
        if (overallSuccessRate < 0.9) {
            recommendations.add("Investigate error patterns - overall success rate is ${(overallSuccessRate * 100).toInt()}%")
        }

        val totalCost = snapshots.sumOf { it.cost }
        if (totalCost > 10.0) { // Arbitrary threshold
            recommendations.add("Monitor costs closely - total spending is $${String.format("%.2f", totalCost)}")
        }

        val tokenUsage = metrics.map { it.averageTokensUsed }.average()
        if (tokenUsage > 2000) {
            recommendations.add("Consider prompt optimization to reduce token usage (avg: ${tokenUsage.toInt()} tokens)")
        }

        return recommendations
    }

    private fun calculateBottleneckImpact(metric: PerformanceMetric): Float {
        val timeImpact = if (metric.averageProcessingTime > ACCEPTABLE_PROCESSING_THRESHOLD_MS) {
            (metric.averageProcessingTime.toFloat() / ACCEPTABLE_PROCESSING_THRESHOLD_MS) * 0.4f
        } else 0f

        val successImpact = (1f - metric.successRate) * 0.6f

        return (timeImpact + successImpact).coerceIn(0f, 1f)
    }

    private fun generateBottleneckRecommendations(metric: PerformanceMetric, issues: List<String>): List<String> {
        val recommendations = mutableListOf<String>()

        if (issues.contains("High processing time")) {
            recommendations.add("Consider switching to faster model or implementing caching")
        }
        if (issues.contains("Low success rate")) {
            recommendations.add("Review error logs and implement retry logic")
        }
        if (issues.contains("High token usage")) {
            recommendations.add("Optimize prompts to reduce token consumption")
        }

        return recommendations
    }

    private fun getActiveAlerts(): List<PerformanceAlert> {
        // In a real implementation, this would maintain a list of active alerts
        // For now, return empty list
        return emptyList()
    }
}

// ========== PERFORMANCE TRACKER ==========

interface PerformanceTracker {
    fun recordToken(tokens: Int)
    fun recordError(error: String)
    fun recordSuccess(confidenceScore: Float? = null)
    fun complete(tokenUsage: TokenUsage, success: Boolean, confidenceScore: Float? = null)
}

class PerformanceTracker(
    private val operationId: String,
    private val operationType: AIOperationType,
    private val model: AIModel,
    private val sessionId: String?,
    private val startTime: Long,
    private val metadata: Map<String, String>
) : PerformanceTracker {

    private var tokens = 0
    private var errors = mutableListOf<String>()
    private var completed = false

    override fun recordToken(tokens: Int) {
        this.tokens += tokens
    }

    override fun recordError(error: String) {
        errors.add(error)
    }

    override fun recordSuccess(confidenceScore: Float?) {
        if (!completed) {
            complete(
                TokenUsage(0, 0, tokens),
                success = true,
                confidenceScore = confidenceScore
            )
        }
    }

    override fun complete(tokenUsage: TokenUsage, success: Boolean, confidenceScore: Float?) {
        if (completed) return
        completed = true

        val processingTime = System.currentTimeMillis() - startTime
        AIPerformanceMonitor.recordOperation(
            operationId = operationId,
            operationType = operationType,
            model = model,
            processingTimeMs = processingTime,
            tokenUsage = tokenUsage,
            success = success,
            confidenceScore = confidenceScore,
            errorMessage = errors.firstOrNull(),
            sessionId = sessionId,
            metadata = metadata
        )
    }
}

class NoOpTracker : PerformanceTracker {
    override fun recordToken(tokens: Int) {}
    override fun recordError(error: String) {}
    override fun recordSuccess(confidenceScore: Float?) {}
    override fun complete(tokenUsage: TokenUsage, success: Boolean, confidenceScore: Float?) {}
}

// ========== DATA CLASSES ==========

data class TokenUsage(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int
)

data class PerformanceMetric(
    val key: String,
    val operationType: AIOperationType,
    val model: AIModel,
    val totalOperations: AtomicLong,
    val successfulOperations: AtomicLong,
    val totalProcessingTime: AtomicLong,
    val totalTokensUsed: AtomicLong,
    var totalCost: Double,
    var averageProcessingTime: Long,
    var averageTokensUsed: Int,
    var averageConfidence: Float,
    var successRate: Float,
    var lastUpdated: Long
)

data class AISessionMetrics(
    val sessionId: String,
    val activeOperations: List<String> = emptyList(),
    val totalOperations: Int = 0,
    val successfulOperations: Int = 0,
    val totalProcessingTime: Long = 0L,
    val totalTokensUsed: Int = 0,
    val totalCost: Double = 0.0,
    val startTime: Long = System.currentTimeMillis(),
    val lastActivity: Long = System.currentTimeMillis()
)

data class CostTrackingData(
    val todaysCost: Double = 0.0,
    val monthCost: Double = 0.0,
    val totalCost: Double = 0.0,
    val costPerHour: Double = 0.0,
    val totalTokensUsed: Int = 0,
    val dailyCostLimit: Double = 10.0,
    val monthlyCostLimit: Double = 100.0,
    val operationCostLimit: Double = 1.0,
    val lastUpdate: Long = System.currentTimeMillis()
)

data class PerformanceSnapshot(
    val timestamp: Long,
    val operationType: AIOperationType,
    val model: AIModel,
    val processingTimeMs: Long,
    val tokenUsage: TokenUsage,
    val success: Boolean,
    val cost: Double,
    val confidenceScore: Float?
)

data class PerformanceUpdate(
    val operationId: String,
    val operationType: AIOperationType,
    val model: AIModel,
    val processingTime: Long,
    val tokenUsage: TokenUsage,
    val success: Boolean,
    val cost: Double,
    val timestamp: Long
)

data class PerformanceAlert(
    val type: AlertType,
    val severity: AlertSeverity,
    val message: String,
    val metric: String = "",
    val value: Double? = null,
    val threshold: Double? = null,
    val timestamp: Long
)

data class PerformanceReport(
    val reportId: String,
    val timeRange: TimeRange,
    val generatedAt: Long,
    val totalOperations: Int,
    val successfulOperations: Int,
    val totalProcessingTime: Long,
    val totalTokensUsed: Long,
    val totalCost: Double,
    val averageProcessingTime: Long,
    val averageTokensPerOperation: Int,
    val overallSuccessRate: Float,
    val modelMetrics: Map<AIModel, ModelPerformanceStats>,
    val operationMetrics: Map<String, PerformanceMetric>,
    val performanceTrends: Map<String, TrendData>,
    val bottlenecks: List<PerformanceBottleneck>,
    val optimizationOpportunities: List<OptimizationOpportunity>,
    val costAnalysis: CostAnalysis,
    val averageConfidence: Float,
    val qualityDistribution: Map<String, Int>,
    val recommendations: List<String>
)

data class PerformanceDashboard(
    val timestamp: Long,
    val activeOperations: Int,
    val operationsPerMinute: Float,
    val averageResponseTime: Long,
    val successRate: Float,
    val tokenUsageRate: Double,
    val currentCostPerHour: Double,
    val todaysCost: Double,
    val topModels: Map<AIModel, ModelPerformanceStats>,
    val alerts: List<PerformanceAlert>
)

data class BatchOperationResult(
    val operationId: String,
    val tokenUsage: TokenUsage,
    val success: Boolean,
    val confidenceScore: Float?,
    val cost: Double,
    val processingTimeMs: Long
)

data class CostLimitCheck(
    val withinLimits: Boolean,
    val wouldExceedDaily: Boolean,
    val wouldExceedMonthly: Boolean,
    val wouldExceedOperation: Boolean,
    val estimatedCost: Double,
    val remainingDailyBudget: Double,
    val remainingMonthlyBudget: Double
)

data class AlertConfiguration(
    val checkIntervalMs: Long = 60000L, // 1 minute
    val maxProcessingTimeMs: Long = ACCEPTABLE_PROCESSING_THRESHOLD_MS,
    val minSuccessRate: Float = 0.9f,
    val maxCostPerOperation: Double = 0.5,
    val maxTokensPerMinute: Int = 10000
)

data class TrendData(
    val direction: TrendDirection,
    val percentChange: Double,
    val correlation: Float
)

data class PerformanceBottleneck(
    val metric: String,
    val issues: List<String>,
    val severity: BottleneckSeverity,
    val impact: Float,
    val recommendations: List<String>
)

data class OptimizationOpportunity(
    val type: OptimizationType,
    val description: String,
    val potentialSavings: String,
    val implementationEffort: ImplementationEffort,
    val affectedModels: List<AIModel> = emptyList(),
    val affectedOperations: List<AIOperationType> = emptyList()
)

data class CostAnalysis(
    val totalCost: Double = 0.0,
    val costPerOperation: Double = 0.0,
    val costPerSuccess: Double = 0.0,
    val costByModel: Map<AIModel, Double> = emptyMap(),
    val mostExpensiveModel: AIModel? = null,
    val mostEfficientModel: AIModel? = null,
    val efficiency: Double = 0.0
)

data class TimeRange(
    val startTime: Long,
    val endTime: Long,
    val description: String = ""
)

// ========== ENUMS ==========

enum class AIOperationType {
    SLEEP_ANALYSIS,
    INSIGHT_GENERATION,
    TREND_ANALYSIS,
    PATTERN_RECOGNITION,
    RECOMMENDATION_GENERATION,
    DATA_VALIDATION,
    BATCH_PROCESSING,
    REPORT_GENERATION
}

enum class AlertType {
    SLOW_PROCESSING,
    LOW_SUCCESS_RATE,
    HIGH_COST,
    HIGH_TOKEN_USAGE,
    OPERATION_FAILURE,
    SYSTEM_ERROR
}

enum class AlertSeverity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
}

enum class BottleneckSeverity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class OptimizationType {
    COST_OPTIMIZATION,
    PERFORMANCE_OPTIMIZATION,
    QUALITY_OPTIMIZATION,
    RESOURCE_OPTIMIZATION
}

enum class ImplementationEffort {
    LOW,
    MEDIUM,
    HIGH
}

enum class TrendDirection {
    IMPROVING,
    STABLE,
    DECLINING,
    VOLATILE
}