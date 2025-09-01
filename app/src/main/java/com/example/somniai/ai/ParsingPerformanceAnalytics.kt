package com.example.somniai.ai

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.*

/**
 * Comprehensive AI Response Parsing Performance Analytics for SomniAI
 *
 * Advanced parsing performance monitoring system that tracks:
 * - AI response parsing speed and accuracy
 * - Response structure validation and compliance
 * - Content quality assessment and scoring
 * - Parsing bottleneck identification and optimization
 * - Model-specific parsing performance metrics
 * - Real-time parsing health monitoring
 * - Parsing error analysis and pattern recognition
 * - Response validation performance tracking
 * - Content extraction efficiency metrics
 * - Schema compliance and validation analytics
 *
 * Integrates seamlessly with existing AIPerformanceMonitor and AIInsightsEngine
 */
object ParsingPerformanceAnalytics {

    companion object {
        private const val TAG = "ParsingPerformanceAnalytics"

        // Performance thresholds
        private const val FAST_PARSING_THRESHOLD_MS = 50L
        private const val ACCEPTABLE_PARSING_THRESHOLD_MS = 200L
        private const val SLOW_PARSING_THRESHOLD_MS = 500L

        // Quality thresholds
        private const val MIN_CONTENT_QUALITY_SCORE = 0.7f
        private const val EXCELLENT_QUALITY_THRESHOLD = 0.9f
        private const val POOR_QUALITY_THRESHOLD = 0.5f

        // Error rate thresholds
        private const val ACCEPTABLE_ERROR_RATE = 0.05f
        private const val HIGH_ERROR_RATE_THRESHOLD = 0.15f
        private const val CRITICAL_ERROR_RATE = 0.3f

        // Validation thresholds
        private const val MIN_SCHEMA_COMPLIANCE = 0.8f
        private const val EXCELLENT_COMPLIANCE_THRESHOLD = 0.95f

        // Performance monitoring constants
        private const val METRICS_RETENTION_HOURS = 72L
        private const val PERFORMANCE_SNAPSHOT_INTERVAL_MS = 300000L // 5 minutes
        private const val MAX_ERROR_SAMPLES = 100
        private const val MAX_PERFORMANCE_HISTORY = 1000
    }

    // ========== CORE STATE MANAGEMENT ==========

    private val parsingMetrics = ConcurrentHashMap<String, ParsingMetric>()
    private val modelParsingPerformance = ConcurrentHashMap<AIModel, ModelParsingStats>()
    private val responseTypeMetrics = ConcurrentHashMap<ResponseType, ResponseTypeStats>()
    private val validationMetrics = ConcurrentHashMap<String, ValidationMetrics>()

    // Performance tracking
    private val totalParsingOperations = AtomicLong(0)
    private val successfulParsing = AtomicLong(0)
    private val failedParsing = AtomicLong(0)
    private val totalParsingTime = AtomicLong(0)
    private val averageParsingTime = AtomicReference(0L)

    // Real-time monitoring
    private val _parsingUpdates = MutableSharedFlow<ParsingPerformanceUpdate>()
    val parsingUpdates: SharedFlow<ParsingPerformanceUpdate> = _parsingUpdates.asSharedFlow()

    private val _parsingAlerts = MutableSharedFlow<ParsingAlert>()
    val parsingAlerts: SharedFlow<ParsingAlert> = _parsingAlerts.asSharedFlow()

    // Performance history and analytics
    private val parsingHistory = mutableListOf<ParsingSnapshot>()
    private val errorSamples = mutableListOf<ParsingError>()
    private val qualityHistory = mutableListOf<QualitySnapshot>()

    // Monitoring state
    private var isMonitoringActive = true
    private var monitoringScope: CoroutineScope? = null

    // ========== INITIALIZATION ==========

    /**
     * Initialize parsing performance analytics
     */
    fun initialize(): Result<Unit> {
        return try {
            Log.d(TAG, "Initializing Parsing Performance Analytics")

            // Start background monitoring
            startPerformanceMonitoring()

            // Initialize baseline metrics
            initializeBaselineMetrics()

            Log.d(TAG, "Parsing Performance Analytics initialized successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize parsing analytics", e)
            Result.failure(e)
        }
    }

    // ========== CORE PARSING TRACKING ==========

    /**
     * Start tracking a parsing operation
     */
    fun startParsingOperation(
        operationId: String,
        model: AIModel,
        responseType: ResponseType,
        contentLength: Int,
        expectedSchema: ResponseSchema? = null
    ): ParsingTracker {
        if (!isMonitoringActive) return NoOpParsingTracker()

        val tracker = ParsingTracker(
            operationId = operationId,
            model = model,
            responseType = responseType,
            contentLength = contentLength,
            expectedSchema = expectedSchema,
            startTime = System.currentTimeMillis()
        )

        totalParsingOperations.incrementAndGet()

        return tracker
    }

    /**
     * Record completed parsing operation with comprehensive metrics
     */
    fun recordParsingOperation(
        operationId: String,
        model: AIModel,
        responseType: ResponseType,
        parsingTimeMs: Long,
        success: Boolean,
        contentLength: Int,
        parsedContentLength: Int,
        qualityScore: Float,
        validationResults: ValidationResults,
        errorDetails: ParsingErrorDetails? = null,
        extractedData: ExtractedDataMetrics? = null
    ) {
        if (!isMonitoringActive) return

        val timestamp = System.currentTimeMillis()

        // Update global counters
        if (success) {
            successfulParsing.incrementAndGet()
        } else {
            failedParsing.incrementAndGet()
            recordParsingError(operationId, model, errorDetails, timestamp)
        }

        totalParsingTime.addAndGet(parsingTimeMs)
        updateAverageParsingTime()

        // Update model-specific metrics
        updateModelParsingMetrics(model, parsingTimeMs, success, qualityScore, validationResults)

        // Update response type metrics
        updateResponseTypeMetrics(responseType, parsingTimeMs, success, qualityScore, contentLength)

        // Update validation metrics
        updateValidationMetrics(validationResults, model, responseType)

        // Record parsing snapshot
        recordParsingSnapshot(
            model, responseType, parsingTimeMs, success, qualityScore,
            validationResults, contentLength, parsedContentLength, timestamp
        )

        // Check for performance alerts
        checkParsingPerformanceAlerts(model, responseType, parsingTimeMs, success, qualityScore)

        // Emit performance update
        emitParsingUpdate(operationId, model, responseType, parsingTimeMs, success, qualityScore)

        // Update extraction metrics if available
        extractedData?.let { data ->
            updateExtractionMetrics(model, responseType, data)
        }
    }

    /**
     * Record batch parsing operation
     */
    fun recordBatchParsingOperation(
        batchId: String,
        model: AIModel,
        operations: List<BatchParsingResult>,
        totalParsingTimeMs: Long
    ) {
        if (!isMonitoringActive || operations.isEmpty()) return

        val successfulOps = operations.count { it.success }
        val avgQuality = operations.map { it.qualityScore }.average().toFloat()
        val totalContentLength = operations.sumOf { it.contentLength }
        val avgValidationScore = operations.map { it.validationResults.overallScore }.average().toFloat()

        // Record as aggregated operation
        recordParsingOperation(
            operationId = batchId,
            model = model,
            responseType = ResponseType.BATCH,
            parsingTimeMs = totalParsingTimeMs,
            success = successfulOps == operations.size,
            contentLength = totalContentLength,
            parsedContentLength = operations.sumOf { it.parsedContentLength },
            qualityScore = avgQuality,
            validationResults = ValidationResults(
                overallScore = avgValidationScore,
                schemaCompliance = operations.map { it.validationResults.schemaCompliance }.average().toFloat(),
                contentValidation = operations.map { it.validationResults.contentValidation }.average().toFloat(),
                structureValidation = operations.map { it.validationResults.structureValidation }.average().toFloat(),
                details = operations.flatMap { it.validationResults.details }
            )
        )
    }

    // ========== PARSING QUALITY ANALYSIS ==========

    /**
     * Analyze parsing quality for a specific response
     */
    fun analyzeParsingQuality(
        rawResponse: String,
        parsedContent: Any?,
        expectedSchema: ResponseSchema?,
        model: AIModel
    ): ParsingQualityAnalysis {
        val startTime = System.currentTimeMillis()

        try {
            val contentQuality = assessContentQuality(rawResponse, parsedContent)
            val structuralQuality = assessStructuralQuality(parsedContent, expectedSchema)
            val extractionEfficiency = calculateExtractionEfficiency(rawResponse, parsedContent)
            val schemaCompliance = calculateSchemaCompliance(parsedContent, expectedSchema)

            val overallQuality = (contentQuality * 0.4f) + (structuralQuality * 0.3f) +
                    (extractionEfficiency * 0.2f) + (schemaCompliance * 0.1f)

            val analysisTime = System.currentTimeMillis() - startTime

            return ParsingQualityAnalysis(
                overallQuality = overallQuality,
                contentQuality = contentQuality,
                structuralQuality = structuralQuality,
                extractionEfficiency = extractionEfficiency,
                schemaCompliance = schemaCompliance,
                analysisTimeMs = analysisTime,
                recommendations = generateQualityRecommendations(
                    contentQuality, structuralQuality, extractionEfficiency, schemaCompliance, model
                ),
                timestamp = System.currentTimeMillis()
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing parsing quality", e)
            return ParsingQualityAnalysis.createFailedAnalysis(e)
        }
    }

    /**
     * Validate response structure and content
     */
    fun validateResponse(
        response: String,
        expectedSchema: ResponseSchema?,
        responseType: ResponseType
    ): ValidationResults {
        val startTime = System.currentTimeMillis()
        val validationDetails = mutableListOf<ValidationDetail>()

        try {
            // Schema validation
            val schemaScore = validateSchema(response, expectedSchema, validationDetails)

            // Content validation
            val contentScore = validateContent(response, responseType, validationDetails)

            // Structure validation
            val structureScore = validateStructure(response, responseType, validationDetails)

            // Format validation
            val formatScore = validateFormat(response, responseType, validationDetails)

            val overallScore = (schemaScore * 0.3f) + (contentScore * 0.3f) +
                    (structureScore * 0.2f) + (formatScore * 0.2f)

            val validationTime = System.currentTimeMillis() - startTime

            return ValidationResults(
                overallScore = overallScore,
                schemaCompliance = schemaScore,
                contentValidation = contentScore,
                structureValidation = structureScore,
                formatValidation = formatScore,
                validationTimeMs = validationTime,
                details = validationDetails,
                timestamp = System.currentTimeMillis()
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error validating response", e)
            return ValidationResults.createFailedValidation(e)
        }
    }

    // ========== PERFORMANCE ANALYTICS ==========

    /**
     * Get comprehensive parsing performance report
     */
    fun getParsingPerformanceReport(
        timeRangeMs: Long = 24 * 60 * 60 * 1000L, // 24 hours
        includeDetails: Boolean = true
    ): ParsingPerformanceReport {
        val endTime = System.currentTimeMillis()
        val startTime = endTime - timeRangeMs

        val relevantSnapshots = parsingHistory.filter { it.timestamp >= startTime }
        val relevantErrors = errorSamples.filter { it.timestamp >= startTime }

        return ParsingPerformanceReport(
            reportId = "parsing_${endTime}",
            timeRange = PerformanceTimeRange(startTime, endTime),
            generatedAt = endTime,

            // Overall metrics
            totalOperations = relevantSnapshots.size,
            successfulOperations = relevantSnapshots.count { it.success },
            totalParsingTime = relevantSnapshots.sumOf { it.parsingTimeMs },
            averageParsingTime = if (relevantSnapshots.isNotEmpty()) {
                relevantSnapshots.map { it.parsingTimeMs }.average().toLong()
            } else 0L,

            // Quality metrics
            averageQualityScore = if (relevantSnapshots.isNotEmpty()) {
                relevantSnapshots.map { it.qualityScore }.average().toFloat()
            } else 0f,

            averageValidationScore = if (relevantSnapshots.isNotEmpty()) {
                relevantSnapshots.map { it.validationResults.overallScore }.average().toFloat()
            } else 0f,

            // Performance breakdown
            modelPerformance = if (includeDetails) modelParsingPerformance.toMap() else emptyMap(),
            responseTypePerformance = if (includeDetails) responseTypeMetrics.toMap() else emptyMap(),

            // Error analysis
            errorRate = if (relevantSnapshots.isNotEmpty()) {
                relevantErrors.size.toFloat() / relevantSnapshots.size
            } else 0f,
            errorAnalysis = analyzeParsingErrors(relevantErrors),

            // Performance trends
            performanceTrends = analyzeParsingTrends(relevantSnapshots),
            qualityTrends = analyzeQualityTrends(relevantSnapshots),

            // Optimization opportunities
            bottlenecks = identifyParsingBottlenecks(relevantSnapshots),
            optimizations = identifyOptimizationOpportunities(relevantSnapshots),

            // Recommendations
            recommendations = generateParsingRecommendations(relevantSnapshots, relevantErrors)
        )
    }

    /**
     * Get real-time parsing dashboard metrics
     */
    fun getParsingDashboard(): ParsingDashboard {
        val currentTime = System.currentTimeMillis()
        val recentWindow = 5 * 60 * 1000L // Last 5 minutes

        val recentSnapshots = parsingHistory.filter {
            it.timestamp >= currentTime - recentWindow
        }

        return ParsingDashboard(
            timestamp = currentTime,
            activeOperations = getCurrentActiveOperations(),
            operationsPerMinute = calculateOperationsPerMinute(recentSnapshots),
            averageParsingTime = averageParsingTime.get(),
            successRate = calculateCurrentSuccessRate(recentSnapshots),
            averageQualityScore = calculateAverageQuality(recentSnapshots),
            errorRate = calculateCurrentErrorRate(recentSnapshots),
            topPerformingModels = getTopPerformingModels(3),
            currentBottlenecks = getCurrentBottlenecks(),
            recentAlerts = getRecentAlerts(10)
        )
    }

    // ========== ERROR ANALYSIS ==========

    /**
     * Analyze parsing error patterns
     */
    fun analyzeParsingErrorPatterns(
        timeRangeMs: Long = 24 * 60 * 60 * 1000L
    ): ParsingErrorAnalysis {
        val cutoffTime = System.currentTimeMillis() - timeRangeMs
        val relevantErrors = errorSamples.filter { it.timestamp >= cutoffTime }

        if (relevantErrors.isEmpty()) {
            return ParsingErrorAnalysis.empty()
        }

        val errorsByType = relevantErrors.groupBy { it.errorType }
        val errorsByModel = relevantErrors.groupBy { it.model }
        val errorsByResponseType = relevantErrors.groupBy { it.responseType }

        val commonPatterns = identifyCommonErrorPatterns(relevantErrors)
        val errorFrequency = calculateErrorFrequency(relevantErrors)
        val recoveryStrategies = generateRecoveryStrategies(errorsByType)

        return ParsingErrorAnalysis(
            totalErrors = relevantErrors.size,
            errorsByType = errorsByType.mapValues { it.value.size },
            errorsByModel = errorsByModel.mapValues { it.value.size },
            errorsByResponseType = errorsByResponseType.mapValues { it.value.size },
            commonPatterns = commonPatterns,
            errorFrequency = errorFrequency,
            recoveryStrategies = recoveryStrategies,
            recommendations = generateErrorReductionRecommendations(relevantErrors),
            timestamp = System.currentTimeMillis()
        )
    }

    // ========== OPTIMIZATION RECOMMENDATIONS ==========

    /**
     * Generate parsing optimization recommendations
     */
    fun generateOptimizationRecommendations(): List<ParsingOptimizationRecommendation> {
        val recommendations = mutableListOf<ParsingOptimizationRecommendation>()

        // Analyze model performance
        val modelPerf = modelParsingPerformance.values
        val slowModels = modelPerf.filter { it.averageParsingTime > ACCEPTABLE_PARSING_THRESHOLD_MS }

        if (slowModels.isNotEmpty()) {
            recommendations.add(
                ParsingOptimizationRecommendation(
                    type = OptimizationType.PERFORMANCE,
                    priority = RecommendationPriority.HIGH,
                    title = "Optimize Slow Parsing Models",
                    description = "Models ${slowModels.map { it.model }.joinToString()} show slow parsing performance",
                    expectedImprovement = "30-50% faster parsing times",
                    implementationEffort = ImplementationEffort.MEDIUM,
                    steps = listOf(
                        "Review response preprocessing for slow models",
                        "Implement response caching for repeated patterns",
                        "Consider switching to faster parsing libraries",
                        "Optimize regex patterns and JSON parsing"
                    )
                )
            )
        }

        // Analyze error patterns
        val highErrorRateModels = modelPerf.filter { it.errorRate > ACCEPTABLE_ERROR_RATE }

        if (highErrorRateModels.isNotEmpty()) {
            recommendations.add(
                ParsingOptimizationRecommendation(
                    type = OptimizationType.QUALITY,
                    priority = RecommendationPriority.CRITICAL,
                    title = "Reduce Parsing Error Rates",
                    description = "High error rates detected in parsing operations",
                    expectedImprovement = "50-70% reduction in parsing errors",
                    implementationEffort = ImplementationEffort.HIGH,
                    steps = listOf(
                        "Implement robust error handling",
                        "Add response preprocessing validation",
                        "Improve schema validation logic",
                        "Add fallback parsing strategies"
                    )
                )
            )
        }

        // Quality optimization
        val lowQualityModels = modelPerf.filter { it.averageQuality < MIN_CONTENT_QUALITY_SCORE }

        if (lowQualityModels.isNotEmpty()) {
            recommendations.add(
                ParsingOptimizationRecommendation(
                    type = OptimizationType.QUALITY,
                    priority = RecommendationPriority.MEDIUM,
                    title = "Improve Content Quality Assessment",
                    description = "Content quality scores below threshold for some models",
                    expectedImprovement = "20-30% better quality scores",
                    implementationEffort = ImplementationEffort.MEDIUM,
                    steps = listOf(
                        "Enhance content quality metrics",
                        "Implement advanced validation rules",
                        "Add semantic content analysis",
                        "Improve response filtering"
                    )
                )
            )
        }

        return recommendations.sortedBy { it.priority.ordinal }
    }

    // ========== PRIVATE HELPER METHODS ==========

    private fun startPerformanceMonitoring() {
        monitoringScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

        monitoringScope?.launch {
            while (isMonitoringActive) {
                delay(PERFORMANCE_SNAPSHOT_INTERVAL_MS)

                try {
                    performMaintenanceTasks()
                    checkSystemHealth()

                } catch (e: Exception) {
                    Log.e(TAG, "Error in performance monitoring", e)
                }
            }
        }
    }

    private fun initializeBaselineMetrics() {
        // Initialize with default metrics for each model
        AIModel.values().forEach { model ->
            modelParsingPerformance[model] = ModelParsingStats(
                model = model,
                totalOperations = AtomicLong(0),
                successfulOperations = AtomicLong(0),
                totalParsingTime = AtomicLong(0),
                averageParsingTime = 0L,
                averageQuality = 0f,
                errorRate = 0f,
                lastUpdated = System.currentTimeMillis()
            )
        }

        // Initialize response type metrics
        ResponseType.values().forEach { responseType ->
            responseTypeMetrics[responseType] = ResponseTypeStats(
                responseType = responseType,
                totalOperations = AtomicLong(0),
                averageParsingTime = 0L,
                averageQuality = 0f,
                averageContentLength = 0,
                successRate = 0f
            )
        }
    }

    private fun updateModelParsingMetrics(
        model: AIModel,
        parsingTimeMs: Long,
        success: Boolean,
        qualityScore: Float,
        validationResults: ValidationResults
    ) {
        val stats = modelParsingPerformance.getOrPut(model) {
            ModelParsingStats(
                model = model,
                totalOperations = AtomicLong(0),
                successfulOperations = AtomicLong(0),
                totalParsingTime = AtomicLong(0),
                averageParsingTime = 0L,
                averageQuality = 0f,
                errorRate = 0f,
                lastUpdated = System.currentTimeMillis()
            )
        }

        stats.totalOperations.incrementAndGet()
        if (success) stats.successfulOperations.incrementAndGet()
        stats.totalParsingTime.addAndGet(parsingTimeMs)

        // Update averages
        val totalOps = stats.totalOperations.get()
        stats.averageParsingTime = stats.totalParsingTime.get() / totalOps
        stats.errorRate = 1f - (stats.successfulOperations.get().toFloat() / totalOps)

        // Update quality (running average)
        stats.averageQuality = ((stats.averageQuality * (totalOps - 1)) + qualityScore) / totalOps
        stats.lastUpdated = System.currentTimeMillis()
    }

    private fun updateResponseTypeMetrics(
        responseType: ResponseType,
        parsingTimeMs: Long,
        success: Boolean,
        qualityScore: Float,
        contentLength: Int
    ) {
        val stats = responseTypeMetrics.getOrPut(responseType) {
            ResponseTypeStats(
                responseType = responseType,
                totalOperations = AtomicLong(0),
                averageParsingTime = 0L,
                averageQuality = 0f,
                averageContentLength = 0,
                successRate = 0f
            )
        }

        val currentOps = stats.totalOperations.incrementAndGet()

        // Update running averages
        stats.averageParsingTime = ((stats.averageParsingTime * (currentOps - 1)) + parsingTimeMs) / currentOps
        stats.averageQuality = ((stats.averageQuality * (currentOps - 1)) + qualityScore) / currentOps
        stats.averageContentLength = ((stats.averageContentLength * (currentOps - 1)) + contentLength) / currentOps.toInt()

        if (success) {
            stats.successRate = ((stats.successRate * (currentOps - 1)) + 1f) / currentOps
        } else {
            stats.successRate = (stats.successRate * (currentOps - 1)) / currentOps
        }
    }

    private fun updateValidationMetrics(
        validationResults: ValidationResults,
        model: AIModel,
        responseType: ResponseType
    ) {
        val key = "${model}_${responseType}"
        val metrics = validationMetrics.getOrPut(key) {
            ValidationMetrics(
                model = model,
                responseType = responseType,
                totalValidations = AtomicLong(0),
                averageScore = 0f,
                schemaComplianceRate = 0f,
                contentValidationRate = 0f,
                structureValidationRate = 0f
            )
        }

        val currentValidations = metrics.totalValidations.incrementAndGet()

        // Update running averages
        metrics.averageScore = ((metrics.averageScore * (currentValidations - 1)) + validationResults.overallScore) / currentValidations
        metrics.schemaComplianceRate = ((metrics.schemaComplianceRate * (currentValidations - 1)) + validationResults.schemaCompliance) / currentValidations
        metrics.contentValidationRate = ((metrics.contentValidationRate * (currentValidations - 1)) + validationResults.contentValidation) / currentValidations
        metrics.structureValidationRate = ((metrics.structureValidationRate * (currentValidations - 1)) + validationResults.structureValidation) / currentValidations
    }

    private fun recordParsingSnapshot(
        model: AIModel,
        responseType: ResponseType,
        parsingTimeMs: Long,
        success: Boolean,
        qualityScore: Float,
        validationResults: ValidationResults,
        contentLength: Int,
        parsedContentLength: Int,
        timestamp: Long
    ) {
        val snapshot = ParsingSnapshot(
            model = model,
            responseType = responseType,
            parsingTimeMs = parsingTimeMs,
            success = success,
            qualityScore = qualityScore,
            validationResults = validationResults,
            contentLength = contentLength,
            parsedContentLength = parsedContentLength,
            timestamp = timestamp
        )

        synchronized(parsingHistory) {
            parsingHistory.add(snapshot)
            if (parsingHistory.size > MAX_PERFORMANCE_HISTORY) {
                parsingHistory.removeAt(0)
            }
        }
    }

    private fun recordParsingError(
        operationId: String,
        model: AIModel,
        errorDetails: ParsingErrorDetails?,
        timestamp: Long
    ) {
        if (errorDetails == null) return

        val error = ParsingError(
            operationId = operationId,
            model = model,
            responseType = errorDetails.responseType,
            errorType = errorDetails.errorType,
            errorMessage = errorDetails.errorMessage,
            stackTrace = errorDetails.stackTrace,
            contentSample = errorDetails.contentSample,
            timestamp = timestamp
        )

        synchronized(errorSamples) {
            errorSamples.add(error)
            if (errorSamples.size > MAX_ERROR_SAMPLES) {
                errorSamples.removeAt(0)
            }
        }
    }

    private fun updateAverageParsingTime() {
        val totalOps = totalParsingOperations.get()
        if (totalOps > 0) {
            averageParsingTime.set(totalParsingTime.get() / totalOps)
        }
    }

    private fun checkParsingPerformanceAlerts(
        model: AIModel,
        responseType: ResponseType,
        parsingTimeMs: Long,
        success: Boolean,
        qualityScore: Float
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            // Slow parsing alert
            if (parsingTimeMs > SLOW_PARSING_THRESHOLD_MS) {
                _parsingAlerts.emit(
                    ParsingAlert(
                        type = ParsingAlertType.SLOW_PARSING,
                        severity = AlertSeverity.WARNING,
                        message = "Slow parsing detected: ${parsingTimeMs}ms for $model on $responseType",
                        model = model,
                        responseType = responseType,
                        value = parsingTimeMs.toFloat(),
                        threshold = SLOW_PARSING_THRESHOLD_MS.toFloat(),
                        timestamp = System.currentTimeMillis()
                    )
                )
            }

            // Quality alert
            if (qualityScore < POOR_QUALITY_THRESHOLD) {
                _parsingAlerts.emit(
                    ParsingAlert(
                        type = ParsingAlertType.LOW_QUALITY,
                        severity = AlertSeverity.WARNING,
                        message = "Low parsing quality: ${qualityScore} for $model",
                        model = model,
                        responseType = responseType,
                        value = qualityScore,
                        threshold = POOR_QUALITY_THRESHOLD,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }

            // Failure alert
            if (!success) {
                _parsingAlerts.emit(
                    ParsingAlert(
                        type = ParsingAlertType.PARSING_FAILURE,
                        severity = AlertSeverity.ERROR,
                        message = "Parsing failed for $model on $responseType",
                        model = model,
                        responseType = responseType,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }
    }

    private fun emitParsingUpdate(
        operationId: String,
        model: AIModel,
        responseType: ResponseType,
        parsingTimeMs: Long,
        success: Boolean,
        qualityScore: Float
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            _parsingUpdates.emit(
                ParsingPerformanceUpdate(
                    operationId = operationId,
                    model = model,
                    responseType = responseType,
                    parsingTime = parsingTimeMs,
                    success = success,
                    qualityScore = qualityScore,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    // Content analysis methods
    private fun assessContentQuality(rawResponse: String, parsedContent: Any?): Float {
        if (parsedContent == null) return 0f

        var score = 0f

        // Completeness score
        val completeness = if (rawResponse.isNotEmpty()) {
            minOf(parsedContent.toString().length.toFloat() / rawResponse.length, 1f)
        } else 0f
        score += completeness * 0.4f

        // Structure score
        val hasStructure = when (parsedContent) {
            is JSONObject -> parsedContent.length() > 0
            is JSONArray -> parsedContent.length() > 0
            is Map<*, *> -> parsedContent.isNotEmpty()
            is List<*> -> parsedContent.isNotEmpty()
            else -> parsedContent.toString().isNotBlank()
        }
        score += if (hasStructure) 0.3f else 0f

        // Content richness score
        val contentRichness = calculateContentRichness(parsedContent)
        score += contentRichness * 0.3f

        return score.coerceIn(0f, 1f)
    }

    private fun assessStructuralQuality(parsedContent: Any?, expectedSchema: ResponseSchema?): Float {
        if (parsedContent == null) return 0f
        if (expectedSchema == null) return 0.8f // Default score if no schema expected

        // Implement schema validation logic
        return validateAgainstSchema(parsedContent, expectedSchema)
    }

    private fun calculateExtractionEfficiency(rawResponse: String, parsedContent: Any?): Float {
        if (rawResponse.isEmpty() || parsedContent == null) return 0f

        val rawLength = rawResponse.length
        val parsedLength = parsedContent.toString().length

        // Efficiency is based on how much useful content was extracted
        return if (rawLength > 0) {
            minOf(parsedLength.toFloat() / rawLength, 1f)
        } else 0f
    }

    private fun calculateSchemaCompliance(parsedContent: Any?, expectedSchema: ResponseSchema?): Float {
        if (expectedSchema == null) return 1f // No schema to comply with
        if (parsedContent == null) return 0f

        return validateAgainstSchema(parsedContent, expectedSchema)
    }

    private fun calculateContentRichness(content: Any?): Float {
        return when (content) {
            is JSONObject -> minOf(content.length().toFloat() / 10f, 1f)
            is JSONArray -> minOf(content.length().toFloat() / 5f, 1f)
            is Map<*, *> -> minOf(content.size.toFloat() / 10f, 1f)
            is List<*> -> minOf(content.size.toFloat() / 5f, 1f)
            is String -> minOf(content.length.toFloat() / 1000f, 1f)
            else -> 0.5f
        }
    }

    private fun validateAgainstSchema(content: Any?, schema: ResponseSchema): Float {
        // Simplified schema validation - in real implementation would be more comprehensive
        return try {
            when {
                content is JSONObject && schema.type == "object" -> 0.9f
                content is JSONArray && schema.type == "array" -> 0.9f
                content is String && schema.type == "string" -> 0.9f
                content is Number && schema.type == "number" -> 0.9f
                else -> 0.5f
            }
        } catch (e: Exception) {
            0.1f
        }
    }

    // Validation helper methods
    private fun validateSchema(
        response: String,
        expectedSchema: ResponseSchema?,
        validationDetails: MutableList<ValidationDetail>
    ): Float {
        if (expectedSchema == null) return 1f

        return try {
            when (expectedSchema.type) {
                "object" -> validateJsonObject(response, expectedSchema, validationDetails)
                "array" -> validateJsonArray(response, expectedSchema, validationDetails)
                else -> 0.8f // Default for unknown schema types
            }
        } catch (e: Exception) {
            validationDetails.add(
                ValidationDetail(
                    field = "schema",
                    issue = "Schema validation failed: ${e.message}",
                    severity = ValidationSeverity.ERROR
                )
            )
            0f
        }
    }

    private fun validateJsonObject(
        response: String,
        schema: ResponseSchema,
        validationDetails: MutableList<ValidationDetail>
    ): Float {
        return try {
            val json = JSONObject(response)
            var score = 1f

            // Check required fields
            schema.required.forEach { requiredField ->
                if (!json.has(requiredField)) {
                    validationDetails.add(
                        ValidationDetail(
                            field = requiredField,
                            issue = "Required field missing",
                            severity = ValidationSeverity.ERROR
                        )
                    )
                    score -= 0.2f
                }
            }

            score.coerceIn(0f, 1f)
        } catch (e: JSONException) {
            validationDetails.add(
                ValidationDetail(
                    field = "json",
                    issue = "Invalid JSON format: ${e.message}",
                    severity = ValidationSeverity.ERROR
                )
            )
            0f
        }
    }

    private fun validateJsonArray(
        response: String,
        schema: ResponseSchema,
        validationDetails: MutableList<ValidationDetail>
    ): Float {
        return try {
            val jsonArray = JSONArray(response)
            if (jsonArray.length() > 0) 0.9f else 0.5f
        } catch (e: JSONException) {
            validationDetails.add(
                ValidationDetail(
                    field = "array",
                    issue = "Invalid JSON array format: ${e.message}",
                    severity = ValidationSeverity.ERROR
                )
            )
            0f
        }
    }

    private fun validateContent(
        response: String,
        responseType: ResponseType,
        validationDetails: MutableList<ValidationDetail>
    ): Float {
        var score = 1f

        // Check content length
        if (response.isEmpty()) {
            validationDetails.add(
                ValidationDetail(
                    field = "content",
                    issue = "Empty response content",
                    severity = ValidationSeverity.ERROR
                )
            )
            return 0f
        }

        // Response type specific validation
        when (responseType) {
            ResponseType.INSIGHTS -> {
                if (!response.contains("title") || !response.contains("description")) {
                    score -= 0.3f
                    validationDetails.add(
                        ValidationDetail(
                            field = "content",
                            issue = "Missing expected insight fields",
                            severity = ValidationSeverity.WARNING
                        )
                    )
                }
            }
            ResponseType.JSON -> {
                try {
                    JSONObject(response)
                } catch (e: JSONException) {
                    try {
                        JSONArray(response)
                    } catch (e2: JSONException) {
                        score -= 0.5f
                        validationDetails.add(
                            ValidationDetail(
                                field = "json",
                                issue = "Invalid JSON format",
                                severity = ValidationSeverity.ERROR
                            )
                        )
                    }
                }
            }
            else -> {
                // Generic content validation
                if (response.length < 10) {
                    score -= 0.2f
                    validationDetails.add(
                        ValidationDetail(
                            field = "content",
                            issue = "Content too short",
                            severity = ValidationSeverity.WARNING
                        )
                    )
                }
            }
        }

        return score.coerceIn(0f, 1f)
    }

    private fun validateStructure(
        response: String,
        responseType: ResponseType,
        validationDetails: MutableList<ValidationDetail>
    ): Float {
        return when (responseType) {
            ResponseType.JSON -> {
                try {
                    JSONObject(response)
                    1f
                } catch (e: JSONException) {
                    try {
                        JSONArray(response)
                        1f
                    } catch (e2: JSONException) {
                        validationDetails.add(
                            ValidationDetail(
                                field = "structure",
                                issue = "Invalid JSON structure",
                                severity = ValidationSeverity.ERROR
                            )
                        )
                        0f
                    }
                }
            }
            ResponseType.TEXT -> {
                // For text responses, check for basic structure
                if (response.contains("\n") || response.length > 50) 0.8f else 0.6f
            }
            else -> 0.7f // Default structure score
        }
    }

    private fun validateFormat(
        response: String,
        responseType: ResponseType,
        validationDetails: MutableList<ValidationDetail>
    ): Float {
        // Format validation logic
        return when (responseType) {
            ResponseType.JSON -> {
                if (response.trim().startsWith("{") || response.trim().startsWith("[")) 1f else 0.5f
            }
            ResponseType.TEXT -> {
                // Check for proper text formatting
                val hasProperSentences = response.contains(".") || response.contains("!")
                if (hasProperSentences) 0.9f else 0.6f
            }
            else -> 0.8f
        }
    }

    // Analysis helper methods
    private fun analyzeParsingTrends(snapshots: List<ParsingSnapshot>): Map<String, TrendData> {
        if (snapshots.size < 10) return emptyMap()

        val trends = mutableMapOf<String, TrendData>()

        // Parsing time trend
        val parsingTimes = snapshots.map { it.parsingTimeMs.toDouble() }
        trends["parsing_time"] = calculateTrendData(parsingTimes)

        // Quality trend
        val qualityScores = snapshots.map { it.qualityScore.toDouble() }
        trends["quality"] = calculateTrendData(qualityScores)

        // Success rate trend
        val windowSize = maxOf(5, snapshots.size / 10)
        val successRates = snapshots.windowed(windowSize, step = windowSize).map { window ->
            window.count { it.success }.toDouble() / window.size
        }
        if (successRates.size >= 2) {
            trends["success_rate"] = calculateTrendData(successRates)
        }

        return trends
    }

    private fun analyzeQualityTrends(snapshots: List<ParsingSnapshot>): QualityTrendData {
        if (snapshots.isEmpty()) return QualityTrendData.empty()

        val recentSnapshots = snapshots.takeLast(minOf(100, snapshots.size))
        val previousSnapshots = if (snapshots.size > 100) {
            snapshots.drop(snapshots.size - 200).take(100)
        } else emptyList()

        val recentAvgQuality = recentSnapshots.map { it.qualityScore }.average().toFloat()
        val previousAvgQuality = if (previousSnapshots.isNotEmpty()) {
            previousSnapshots.map { it.qualityScore }.average().toFloat()
        } else recentAvgQuality

        val qualityChange = recentAvgQuality - previousAvgQuality
        val trend = when {
            qualityChange > 0.1f -> TrendDirection.IMPROVING
            qualityChange < -0.1f -> TrendDirection.DECLINING
            else -> TrendDirection.STABLE
        }

        return QualityTrendData(
            currentQuality = recentAvgQuality,
            previousQuality = previousAvgQuality,
            qualityChange = qualityChange,
            trend = trend,
            confidence = if (snapshots.size >= 50) 0.8f else 0.5f
        )
    }

    private fun calculateTrendData(values: List<Double>): TrendData {
        if (values.size < 2) return TrendData.empty()

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

        return TrendData(
            direction = direction,
            percentChange = percentChange,
            confidence = if (values.size >= 20) 0.8f else 0.5f
        )
    }

    private fun identifyParsingBottlenecks(snapshots: List<ParsingSnapshot>): List<ParsingBottleneck> {
        val bottlenecks = mutableListOf<ParsingBottleneck>()

        // Model-specific bottlenecks
        val modelPerformance = snapshots.groupBy { it.model }
        modelPerformance.forEach { (model, modelSnapshots) ->
            val avgTime = modelSnapshots.map { it.parsingTimeMs }.average()
            if (avgTime > ACCEPTABLE_PARSING_THRESHOLD_MS) {
                bottlenecks.add(
                    ParsingBottleneck(
                        type = BottleneckType.MODEL_PERFORMANCE,
                        description = "Model $model shows slow parsing (${avgTime.toLong()}ms avg)",
                        severity = if (avgTime > SLOW_PARSING_THRESHOLD_MS) BottleneckSeverity.HIGH else BottleneckSeverity.MEDIUM,
                        affectedComponent = model.toString(),
                        impact = calculateImpactScore(avgTime, modelSnapshots.size),
                        recommendations = listOf(
                            "Optimize response preprocessing for $model",
                            "Consider response caching strategies",
                            "Review parsing algorithms for $model responses"
                        )
                    )
                )
            }
        }

        // Response type bottlenecks
        val responseTypePerformance = snapshots.groupBy { it.responseType }
        responseTypePerformance.forEach { (responseType, typeSnapshots) ->
            val errorRate = typeSnapshots.count { !it.success }.toFloat() / typeSnapshots.size
            if (errorRate > ACCEPTABLE_ERROR_RATE) {
                bottlenecks.add(
                    ParsingBottleneck(
                        type = BottleneckType.ERROR_RATE,
                        description = "High error rate for $responseType responses (${(errorRate * 100).toInt()}%)",
                        severity = if (errorRate > HIGH_ERROR_RATE_THRESHOLD) BottleneckSeverity.CRITICAL else BottleneckSeverity.HIGH,
                        affectedComponent = responseType.toString(),
                        impact = errorRate,
                        recommendations = listOf(
                            "Implement better error handling for $responseType",
                            "Add validation for $responseType responses",
                            "Review response format requirements"
                        )
                    )
                )
            }
        }

        return bottlenecks.sortedByDescending { it.impact }
    }

    private fun identifyOptimizationOpportunities(snapshots: List<ParsingSnapshot>): List<ParsingOptimizationOpportunity> {
        val opportunities = mutableListOf<ParsingOptimizationOpportunity>()

        // Caching opportunities
        val duplicateResponses = snapshots.groupBy { it.contentLength }
            .filter { it.value.size > 5 }

        if (duplicateResponses.isNotEmpty()) {
            opportunities.add(
                ParsingOptimizationOpportunity(
                    type = OptimizationType.CACHING,
                    description = "Implement response caching for frequently parsed content lengths",
                    potentialImprovement = "20-40% faster parsing for cached responses",
                    effort = OptimizationEffort.MEDIUM,
                    priority = OptimizationPriority.MEDIUM
                )
            )
        }

        // Batch processing opportunities
        val highVolumeModels = snapshots.groupBy { it.model }
            .filter { it.value.size > 50 }
            .keys

        if (highVolumeModels.isNotEmpty()) {
            opportunities.add(
                ParsingOptimizationOpportunity(
                    type = OptimizationType.BATCH_PROCESSING,
                    description = "Implement batch parsing for high-volume models",
                    potentialImprovement = "30-50% better throughput",
                    effort = OptimizationEffort.HIGH,
                    priority = OptimizationPriority.HIGH
                )
            )
        }

        return opportunities.sortedByDescending { it.priority.ordinal }
    }

    private fun generateParsingRecommendations(
        snapshots: List<ParsingSnapshot>,
        errors: List<ParsingError>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (snapshots.isEmpty()) {
            recommendations.add("Start collecting parsing performance data")
            return recommendations
        }

        val avgParsingTime = snapshots.map { it.parsingTimeMs }.average()
        if (avgParsingTime > ACCEPTABLE_PARSING_THRESHOLD_MS) {
            recommendations.add("Optimize parsing algorithms - average time is ${avgParsingTime.toLong()}ms")
        }

        val errorRate = errors.size.toFloat() / snapshots.size
        if (errorRate > ACCEPTABLE_ERROR_RATE) {
            recommendations.add("Improve error handling - current error rate is ${(errorRate * 100).toInt()}%")
        }

        val avgQuality = snapshots.map { it.qualityScore }.average()
        if (avgQuality < MIN_CONTENT_QUALITY_SCORE) {
            recommendations.add("Enhance content quality assessment - average score is ${String.format("%.2f", avgQuality)}")
        }

        return recommendations
    }

    private fun generateQualityRecommendations(
        contentQuality: Float,
        structuralQuality: Float,
        extractionEfficiency: Float,
        schemaCompliance: Float,
        model: AIModel
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (contentQuality < 0.7f) {
            recommendations.add("Improve content extraction quality for $model")
        }

        if (structuralQuality < 0.7f) {
            recommendations.add("Enhance structural validation for $model responses")
        }

        if (extractionEfficiency < 0.6f) {
            recommendations.add("Optimize content extraction efficiency for $model")
        }

        if (schemaCompliance < 0.8f) {
            recommendations.add("Improve schema compliance validation for $model")
        }

        return recommendations
    }

    private fun identifyCommonErrorPatterns(errors: List<ParsingError>): List<ErrorPattern> {
        val patterns = mutableListOf<ErrorPattern>()

        // Group by error message patterns
        val messagePatterns = errors.groupBy { error ->
            // Extract pattern from error message (simplified)
            error.errorMessage.split(":").first()
        }

        messagePatterns.filter { it.value.size >= 3 }.forEach { (pattern, errorList) ->
            patterns.add(
                ErrorPattern(
                    pattern = pattern,
                    frequency = errorList.size,
                    affectedModels = errorList.map { it.model }.distinct(),
                    description = "Common error pattern: $pattern",
                    suggestedFix = "Implement specific handling for $pattern errors"
                )
            )
        }

        return patterns.sortedByDescending { it.frequency }
    }

    private fun calculateErrorFrequency(errors: List<ParsingError>): Map<String, Int> {
        return errors.groupBy { "${it.model}_${it.errorType}" }
            .mapValues { it.value.size }
    }

    private fun generateRecoveryStrategies(errorsByType: Map<ParsingErrorType, List<ParsingError>>): Map<ParsingErrorType, List<String>> {
        return errorsByType.mapValues { (errorType, _) ->
            when (errorType) {
                ParsingErrorType.JSON_PARSE_ERROR -> listOf(
                    "Implement JSON preprocessing",
                    "Add fallback JSON parsers",
                    "Validate JSON structure before parsing"
                )
                ParsingErrorType.SCHEMA_VALIDATION_ERROR -> listOf(
                    "Relax schema validation rules",
                    "Implement partial schema matching",
                    "Add schema version compatibility"
                )
                ParsingErrorType.CONTENT_EXTRACTION_ERROR -> listOf(
                    "Implement robust content extraction",
                    "Add content format detection",
                    "Use multiple extraction strategies"
                )
                ParsingErrorType.TIMEOUT_ERROR -> listOf(
                    "Implement parsing timeouts",
                    "Use streaming parsing for large responses",
                    "Add asynchronous parsing support"
                )
                else -> listOf("Implement general error recovery strategies")
            }
        }
    }

    private fun generateErrorReductionRecommendations(errors: List<ParsingError>): List<String> {
        val recommendations = mutableListOf<String>()

        val errorsByType = errors.groupBy { it.errorType }
        val mostCommonError = errorsByType.maxByOrNull { it.value.size }?.key

        if (mostCommonError != null) {
            recommendations.add("Focus on reducing ${mostCommonError} errors - they account for ${errorsByType[mostCommonError]?.size} of ${errors.size} total errors")
        }

        val errorsByModel = errors.groupBy { it.model }
        val mostProblematicModel = errorsByModel.maxByOrNull { it.value.size }?.key

        if (mostProblematicModel != null) {
            recommendations.add("Improve parsing for $mostProblematicModel - highest error count")
        }

        return recommendations
    }

    // Utility methods
    private fun performMaintenanceTasks() {
        val cutoffTime = System.currentTimeMillis() - (METRICS_RETENTION_HOURS * 60 * 60 * 1000)

        // Clean old history
        synchronized(parsingHistory) {
            parsingHistory.removeAll { it.timestamp < cutoffTime }
        }

        synchronized(errorSamples) {
            errorSamples.removeAll { it.timestamp < cutoffTime }
        }

        synchronized(qualityHistory) {
            qualityHistory.removeAll { it.timestamp < cutoffTime }
        }
    }

    private fun checkSystemHealth() {
        val recentWindow = 10 * 60 * 1000L // 10 minutes
        val currentTime = System.currentTimeMillis()
        val recentSnapshots = parsingHistory.filter { it.timestamp >= currentTime - recentWindow }

        if (recentSnapshots.isNotEmpty()) {
            val recentErrorRate = recentSnapshots.count { !it.success }.toFloat() / recentSnapshots.size
            val recentAvgTime = recentSnapshots.map { it.parsingTimeMs }.average()

            // System health alerts
            if (recentErrorRate > HIGH_ERROR_RATE_THRESHOLD) {
                CoroutineScope(Dispatchers.IO).launch {
                    _parsingAlerts.emit(
                        ParsingAlert(
                            type = ParsingAlertType.SYSTEM_HEALTH,
                            severity = AlertSeverity.CRITICAL,
                            message = "High system error rate: ${(recentErrorRate * 100).toInt()}%",
                            value = recentErrorRate,
                            threshold = HIGH_ERROR_RATE_THRESHOLD,
                            timestamp = currentTime
                        )
                    )
                }
            }

            if (recentAvgTime > SLOW_PARSING_THRESHOLD_MS) {
                CoroutineScope(Dispatchers.IO).launch {
                    _parsingAlerts.emit(
                        ParsingAlert(
                            type = ParsingAlertType.SYSTEM_HEALTH,
                            severity = AlertSeverity.WARNING,
                            message = "System parsing performance degraded: ${recentAvgTime.toLong()}ms average",
                            value = recentAvgTime.toFloat(),
                            threshold = SLOW_PARSING_THRESHOLD_MS.toFloat(),
                            timestamp = currentTime
                        )
                    )
                }
            }
        }
    }

    private fun calculateImpactScore(avgTime: Double, sampleSize: Int): Float {
        val timeImpact = if (avgTime > SLOW_PARSING_THRESHOLD_MS) {
            (avgTime / SLOW_PARSING_THRESHOLD_MS).toFloat().coerceAtMost(3f)
        } else 1f

        val volumeImpact = (sampleSize / 100f).coerceAtMost(2f)

        return ((timeImpact + volumeImpact) / 2f).coerceIn(0f, 1f)
    }

    private fun getCurrentActiveOperations(): Int {
        // In a real implementation, this would track active parsing operations
        return 0
    }

    private fun calculateOperationsPerMinute(snapshots: List<ParsingSnapshot>): Float {
        if (snapshots.isEmpty()) return 0f
        val windowMinutes = 5f
        return snapshots.size / windowMinutes
    }

    private fun calculateCurrentSuccessRate(snapshots: List<ParsingSnapshot>): Float {
        if (snapshots.isEmpty()) return 1f
        return snapshots.count { it.success }.toFloat() / snapshots.size
    }

    private fun calculateAverageQuality(snapshots: List<ParsingSnapshot>): Float {
        if (snapshots.isEmpty()) return 0f
        return snapshots.map { it.qualityScore }.average().toFloat()
    }

    private fun calculateCurrentErrorRate(snapshots: List<ParsingSnapshot>): Float {
        if (snapshots.isEmpty()) return 0f
        return snapshots.count { !it.success }.toFloat() / snapshots.size
    }

    private fun getTopPerformingModels(count: Int): Map<AIModel, ModelPerformanceMetrics> {
        return modelParsingPerformance.entries
            .sortedByDescending { it.value.getPerformanceScore() }
            .take(count)
            .associate { it.key to it.value.toMetrics() }
    }

    private fun getCurrentBottlenecks(): List<String> {
        val recentSnapshots = parsingHistory.takeLast(100)
        val bottlenecks = identifyParsingBottlenecks(recentSnapshots)
        return bottlenecks.take(3).map { it.description }
    }

    private fun getRecentAlerts(count: Int): List<ParsingAlert> {
        // In a real implementation, would maintain a list of recent alerts
        return emptyList()
    }

    private fun updateExtractionMetrics(model: AIModel, responseType: ResponseType, data: ExtractedDataMetrics) {
        // Update extraction-specific metrics
        // Implementation would track data extraction efficiency, accuracy, etc.
    }

    /**
     * Cleanup resources and shutdown monitoring
     */
    fun shutdown() {
        isMonitoringActive = false
        monitoringScope?.cancel()

        parsingMetrics.clear()
        modelParsingPerformance.clear()
        responseTypeMetrics.clear()
        validationMetrics.clear()
        parsingHistory.clear()
        errorSamples.clear()
        qualityHistory.clear()

        Log.d(TAG, "Parsing Performance Analytics shutdown completed")
    }
}

// ========== PARSING TRACKER ==========

interface ParsingTracker {
    fun recordValidation(validationResults: ValidationResults)
    fun recordError(errorType: ParsingErrorType, message: String, stackTrace: String? = null)
    fun complete(
        success: Boolean,
        parsedContentLength: Int,
        qualityScore: Float,
        extractedData: ExtractedDataMetrics? = null
    )
}

class ParsingTracker(
    private val operationId: String,
    private val model: AIModel,
    private val responseType: ResponseType,
    private val contentLength: Int,
    private val expectedSchema: ResponseSchema?,
    private val startTime: Long
) : ParsingTracker {

    private var validationResults: ValidationResults? = null
    private var errorDetails: ParsingErrorDetails? = null
    private var completed = false

    override fun recordValidation(validationResults: ValidationResults) {
        this.validationResults = validationResults
    }

    override fun recordError(errorType: ParsingErrorType, message: String, stackTrace: String?) {
        this.errorDetails = ParsingErrorDetails(
            responseType = responseType,
            errorType = errorType,
            errorMessage = message,
            stackTrace = stackTrace,
            contentSample = "Sample content..." // Would include actual sample
        )
    }

    override fun complete(
        success: Boolean,
        parsedContentLength: Int,
        qualityScore: Float,
        extractedData: ExtractedDataMetrics?
    ) {
        if (completed) return
        completed = true

        val parsingTime = System.currentTimeMillis() - startTime
        val validation = validationResults ?: ValidationResults.createDefaultValidation()

        ParsingPerformanceAnalytics.recordParsingOperation(
            operationId = operationId,
            model = model,
            responseType = responseType,
            parsingTimeMs = parsingTime,
            success = success,
            contentLength = contentLength,
            parsedContentLength = parsedContentLength,
            qualityScore = qualityScore,
            validationResults = validation,
            errorDetails = errorDetails,
            extractedData = extractedData
        )
    }
}

class NoOpParsingTracker : ParsingTracker {
    override fun recordValidation(validationResults: ValidationResults) {}
    override fun recordError(errorType: ParsingErrorType, message: String, stackTrace: String?) {}
    override fun complete(success: Boolean, parsedContentLength: Int, qualityScore: Float, extractedData: ExtractedDataMetrics?) {}
}

// ========== DATA CLASSES ==========

data class ModelParsingStats(
    val model: AIModel,
    val totalOperations: AtomicLong,
    val successfulOperations: AtomicLong,
    val totalParsingTime: AtomicLong,
    var averageParsingTime: Long,
    var averageQuality: Float,
    var errorRate: Float,
    var lastUpdated: Long
) {
    fun getPerformanceScore(): Float {
        val timeScore = (ACCEPTABLE_PARSING_THRESHOLD_MS.toFloat() - averageParsingTime) / ACCEPTABLE_PARSING_THRESHOLD_MS
        val qualityScore = averageQuality
        val reliabilityScore = 1f - errorRate

        return ((timeScore * 0.4f) + (qualityScore * 0.4f) + (reliabilityScore * 0.2f))
            .coerceIn(0f, 1f)
    }

    fun toMetrics(): ModelPerformanceMetrics {
        return ModelPerformanceMetrics(
            model = model,
            totalRequests = totalOperations.get(),
            successfulRequests = successfulOperations.get(),
            averageLatency = averageParsingTime,
            averageQuality = averageQuality,
            errorRate = errorRate,
            lastUpdated = lastUpdated
        )
    }
}

data class ResponseTypeStats(
    val responseType: ResponseType,
    val totalOperations: AtomicLong,
    var averageParsingTime: Long,
    var averageQuality: Float,
    var averageContentLength: Int,
    var successRate: Float
)

data class ValidationMetrics(
    val model: AIModel,
    val responseType: ResponseType,
    val totalValidations: AtomicLong,
    var averageScore: Float,
    var schemaComplianceRate: Float,
    var contentValidationRate: Float,
    var structureValidationRate: Float
)

data class ParsingSnapshot(
    val model: AIModel,
    val responseType: ResponseType,
    val parsingTimeMs: Long,
    val success: Boolean,
    val qualityScore: Float,
    val validationResults: ValidationResults,
    val contentLength: Int,
    val parsedContentLength: Int,
    val timestamp: Long
)

data class ParsingError(
    val operationId: String,
    val model: AIModel,
    val responseType: ResponseType,
    val errorType: ParsingErrorType,
    val errorMessage: String,
    val stackTrace: String?,
    val contentSample: String?,
    val timestamp: Long
)

data class QualitySnapshot(
    val timestamp: Long,
    val qualityScore: Float,
    val model: AIModel,
    val responseType: ResponseType
)

data class ParsingPerformanceUpdate(
    val operationId: String,
    val model: AIModel,
    val responseType: ResponseType,
    val parsingTime: Long,
    val success: Boolean,
    val qualityScore: Float,
    val timestamp: Long
)

data class ParsingAlert(
    val type: ParsingAlertType,
    val severity: AlertSeverity,
    val message: String,
    val model: AIModel? = null,
    val responseType: ResponseType? = null,
    val value: Float? = null,
    val threshold: Float? = null,
    val timestamp: Long
)

data class ValidationResults(
    val overallScore: Float,
    val schemaCompliance: Float,
    val contentValidation: Float,
    val structureValidation: Float,
    val formatValidation: Float = 0f,
    val validationTimeMs: Long = 0L,
    val details: List<ValidationDetail> = emptyList(),
    val timestamp: Long = System.currentTimeMillis()
) {
    companion object {
        fun createDefaultValidation(): ValidationResults {
            return ValidationResults(
                overallScore = 0.5f,
                schemaCompliance = 0.5f,
                contentValidation = 0.5f,
                structureValidation = 0.5f
            )
        }

        fun createFailedValidation(error: Exception): ValidationResults {
            return ValidationResults(
                overallScore = 0f,
                schemaCompliance = 0f,
                contentValidation = 0f,
                structureValidation = 0f,
                details = listOf(
                    ValidationDetail(
                        field = "validation",
                        issue = "Validation failed: ${error.message}",
                        severity = ValidationSeverity.ERROR
                    )
                )
            )
        }
    }
}

data class ValidationDetail(
    val field: String,
    val issue: String,
    val severity: ValidationSeverity,
    val suggestedFix: String? = null
)

data class ParsingErrorDetails(
    val responseType: ResponseType,
    val errorType: ParsingErrorType,
    val errorMessage: String,
    val stackTrace: String?,
    val contentSample: String?
)

data class ExtractedDataMetrics(
    val fieldsExtracted: Int,
    val totalExpectedFields: Int,
    val extractionAccuracy: Float,
    val extractionTime: Long
)

data class ParsingQualityAnalysis(
    val overallQuality: Float,
    val contentQuality: Float,
    val structuralQuality: Float,
    val extractionEfficiency: Float,
    val schemaCompliance: Float,
    val analysisTimeMs: Long,
    val recommendations: List<String>,
    val timestamp: Long
) {
    companion object {
        fun createFailedAnalysis(error: Exception): ParsingQualityAnalysis {
            return ParsingQualityAnalysis(
                overallQuality = 0f,
                contentQuality = 0f,
                structuralQuality = 0f,
                extractionEfficiency = 0f,
                schemaCompliance = 0f,
                analysisTimeMs = 0L,
                recommendations = listOf("Quality analysis failed: ${error.message}"),
                timestamp = System.currentTimeMillis()
            )
        }
    }
}

data class ParsingPerformanceReport(
    val reportId: String,
    val timeRange: PerformanceTimeRange,
    val generatedAt: Long,
    val totalOperations: Int,
    val successfulOperations: Int,
    val totalParsingTime: Long,
    val averageParsingTime: Long,
    val averageQualityScore: Float,
    val averageValidationScore: Float,
    val modelPerformance: Map<AIModel, ModelParsingStats>,
    val responseTypePerformance: Map<ResponseType, ResponseTypeStats>,
    val errorRate: Float,
    val errorAnalysis: ParsingErrorAnalysis,
    val performanceTrends: Map<String, TrendData>,
    val qualityTrends: QualityTrendData,
    val bottlenecks: List<ParsingBottleneck>,
    val optimizations: List<ParsingOptimizationOpportunity>,
    val recommendations: List<String>
)

data class ParsingDashboard(
    val timestamp: Long,
    val activeOperations: Int,
    val operationsPerMinute: Float,
    val averageParsingTime: Long,
    val successRate: Float,
    val averageQualityScore: Float,
    val errorRate: Float,
    val topPerformingModels: Map<AIModel, ModelPerformanceMetrics>,
    val currentBottlenecks: List<String>,
    val recentAlerts: List<ParsingAlert>
)

data class ParsingErrorAnalysis(
    val totalErrors: Int,
    val errorsByType: Map<ParsingErrorType, Int>,
    val errorsByModel: Map<AIModel, Int>,
    val errorsByResponseType: Map<ResponseType, Int>,
    val commonPatterns: List<ErrorPattern>,
    val errorFrequency: Map<String, Int>,
    val recoveryStrategies: Map<ParsingErrorType, List<String>>,
    val recommendations: List<String>,
    val timestamp: Long
) {
    companion object {
        fun empty(): ParsingErrorAnalysis {
            return ParsingErrorAnalysis(
                totalErrors = 0,
                errorsByType = emptyMap(),
                errorsByModel = emptyMap(),
                errorsByResponseType = emptyMap(),
                commonPatterns = emptyList(),
                errorFrequency = emptyMap(),
                recoveryStrategies = emptyMap(),
                recommendations = emptyList(),
                timestamp = System.currentTimeMillis()
            )
        }
    }
}

data class BatchParsingResult(
    val operationId: String,
    val success: Boolean,
    val qualityScore: Float,
    val validationResults: ValidationResults,
    val contentLength: Int,
    val parsedContentLength: Int,
    val parsingTimeMs: Long
)

data class PerformanceTimeRange(
    val startTime: Long,
    val endTime: Long
)

data class TrendData(
    val direction: TrendDirection,
    val percentChange: Double,
    val confidence: Float
) {
    companion object {
        fun empty(): TrendData {
            return TrendData(
                direction = TrendDirection.STABLE,
                percentChange = 0.0,
                confidence = 0f
            )
        }
    }
}

data class QualityTrendData(
    val currentQuality: Float,
    val previousQuality: Float,
    val qualityChange: Float,
    val trend: TrendDirection,
    val confidence: Float
) {
    companion object {
        fun empty(): QualityTrendData {
            return QualityTrendData(
                currentQuality = 0f,
                previousQuality = 0f,
                qualityChange = 0f,
                trend = TrendDirection.STABLE,
                confidence = 0f
            )
        }
    }
}

data class ParsingBottleneck(
    val type: BottleneckType,
    val description: String,
    val severity: BottleneckSeverity,
    val affectedComponent: String,
    val impact: Float,
    val recommendations: List<String>
)

data class ParsingOptimizationOpportunity(
    val type: OptimizationType,
    val description: String,
    val potentialImprovement: String,
    val effort: OptimizationEffort,
    val priority: OptimizationPriority
)

data class ParsingOptimizationRecommendation(
    val type: OptimizationType,
    val priority: RecommendationPriority,
    val title: String,
    val description: String,
    val expectedImprovement: String,
    val implementationEffort: ImplementationEffort,
    val steps: List<String>
)

data class ErrorPattern(
    val pattern: String,
    val frequency: Int,
    val affectedModels: List<AIModel>,
    val description: String,
    val suggestedFix: String
)

data class ModelPerformanceMetrics(
    val model: AIModel,
    val totalRequests: Long,
    val successfulRequests: Long,
    val averageLatency: Long,
    val averageQuality: Float,
    val errorRate: Float,
    val lastUpdated: Long
)

// ========== ENUMS ==========

enum class ResponseType {
    INSIGHTS,
    JSON,
    TEXT,
    STRUCTURED,
    BATCH,
    ANALYSIS,
    RECOMMENDATION,
    SUMMARY
}

enum class ParsingErrorType {
    JSON_PARSE_ERROR,
    SCHEMA_VALIDATION_ERROR,
    CONTENT_EXTRACTION_ERROR,
    TIMEOUT_ERROR,
    MEMORY_ERROR,
    FORMAT_ERROR,
    UNKNOWN_ERROR
}

enum class ParsingAlertType {
    SLOW_PARSING,
    LOW_QUALITY,
    HIGH_ERROR_RATE,
    PARSING_FAILURE,
    SYSTEM_HEALTH,
    VALIDATION_FAILURE
}

enum class ValidationSeverity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
}

enum class BottleneckType {
    MODEL_PERFORMANCE,
    ERROR_RATE,
    VALIDATION_ISSUES,
    RESOURCE_USAGE,
    SYSTEM_LOAD
}

enum class BottleneckSeverity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class OptimizationType {
    PERFORMANCE,
    QUALITY,
    CACHING,
    BATCH_PROCESSING,
    ERROR_REDUCTION
}

enum class OptimizationEffort {
    LOW,
    MEDIUM,
    HIGH
}

enum class OptimizationPriority {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class RecommendationPriority {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class ImplementationEffort {
    MINIMAL,
    LOW,
    MEDIUM,
    HIGH,
    EXTENSIVE
}

enum class AlertSeverity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
}