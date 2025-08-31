package com.example.somniai.extensions

import com.example.somniai.data.*
import com.example.somniai.ai.*
import com.example.somniai.utils.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.json.JSONObject
import org.json.JSONArray
import kotlin.math.*

/**
 * AI Extensions for SomniAI
 *
 * Comprehensive extension functions that enhance AI data processing capabilities,
 * provide convenient sleep data analysis methods, and simplify AI model interactions.
 * These extensions integrate seamlessly with the existing SomniAI architecture.
 */

// ========== SLEEP SESSION AI EXTENSIONS ==========

/**
 * Generate AI analysis prompt from sleep session
 */
fun SleepSession.toAIPrompt(
    analysisType: AIAnalysisType = AIAnalysisType.COMPREHENSIVE,
    includeRecommendations: Boolean = true
): String {
    return when (analysisType) {
        AIAnalysisType.QUICK -> generateQuickAnalysisPrompt(includeRecommendations)
        AIAnalysisType.COMPREHENSIVE -> generateComprehensiveAnalysisPrompt(includeRecommendations)
        AIAnalysisType.COMPARATIVE -> generateComparativeAnalysisPrompt(includeRecommendations)
        AIAnalysisType.PATTERN_FOCUSED -> generatePatternAnalysisPrompt(includeRecommendations)
    }
}

private fun SleepSession.generateQuickAnalysisPrompt(includeRecommendations: Boolean): String {
    return buildString {
        appendLine("=== QUICK SLEEP ANALYSIS ===")
        appendLine("Quality: ${sleepQualityScore?.let { "${it}/10" } ?: "Not calculated"}")
        appendLine("Duration: ${TimeUtils.formatDuration(duration)}")
        appendLine("Efficiency: ${sleepEfficiency}%")
        appendLine("Sleep Latency: ${TimeUtils.formatDuration(sleepLatency)}")
        appendLine()
        if (includeRecommendations) {
            appendLine("Provide a brief quality assessment and 2-3 key recommendations.")
        } else {
            appendLine("Provide a brief quality assessment only.")
        }
    }
}

private fun SleepSession.generateComprehensiveAnalysisPrompt(includeRecommendations: Boolean): String {
    return DataConverter.sleepSessionToAIPrompt(this, includeRecommendations)
}

private fun SleepSession.generateComparativeAnalysisPrompt(includeRecommendations: Boolean): String {
    return buildString {
        appendLine("=== COMPARATIVE SLEEP ANALYSIS ===")
        appendLine("Current Session: ${TimeUtils.formatDate(startTime)}")
        append(generateQuickAnalysisPrompt(false))
        appendLine()
        appendLine("Compare this session to optimal sleep patterns and provide:")
        appendLine("1. Performance vs. sleep science benchmarks")
        appendLine("2. Areas of strength and improvement")
        if (includeRecommendations) {
            appendLine("3. Specific recommendations for optimization")
        }
    }
}

private fun SleepSession.generatePatternAnalysisPrompt(includeRecommendations: Boolean): String {
    return buildString {
        appendLine("=== SLEEP PATTERN ANALYSIS ===")
        append(generateQuickAnalysisPrompt(false))
        appendLine()
        appendLine("Focus on identifying patterns in:")
        appendLine("- Sleep architecture (${getPhaseBreakdownText()})")
        appendLine("- Movement patterns (${movementEvents.size} events)")
        appendLine("- Environmental factors (avg noise: ${averageNoiseLevel} dB)")
        appendLine()
        if (includeRecommendations) {
            appendLine("Identify concerning patterns and suggest specific interventions.")
        } else {
            appendLine("Identify notable patterns and their significance.")
        }
    }
}

private fun SleepSession.getPhaseBreakdownText(): String {
    val total = lightSleepDuration + deepSleepDuration + remSleepDuration + awakeDuration
    if (total == 0L) return "No phase data"

    return listOf(
        "Light: ${(lightSleepDuration * 100 / total)}%",
        "Deep: ${(deepSleepDuration * 100 / total)}%",
        "REM: ${(remSleepDuration * 100 / total)}%",
        "Awake: ${(awakeDuration * 100 / total)}%"
    ).joinToString(", ")
}

/**
 * Extract key metrics for AI analysis
 */
fun SleepSession.extractAIMetrics(): AIMetricsBundle {
    return AIMetricsBundle(
        sessionId = id,
        qualityScore = sleepQualityScore,
        efficiencyScore = sleepEfficiency,
        durationHours = duration / (1000f * 60 * 60),
        sleepLatencyMinutes = sleepLatency / (1000 * 60),
        sleepArchitecture = SleepArchitectureMetrics(
            lightSleepPercent = calculatePhasePercentage(lightSleepDuration),
            deepSleepPercent = calculatePhasePercentage(deepSleepDuration),
            remSleepPercent = calculatePhasePercentage(remSleepDuration),
            awakePercent = calculatePhasePercentage(awakeDuration)
        ),
        environmentalMetrics = EnvironmentalMetrics(
            avgMovementIntensity = averageMovementIntensity,
            movementEventCount = movementEvents.size,
            avgNoiseLevel = averageNoiseLevel,
            noiseEventCount = noiseEvents.size,
            maxNoiseLevel = noiseEvents.maxOfOrNull { it.decibelLevel }
        ),
        timingMetrics = TimingMetrics(
            bedtime = startTime,
            wakeTime = endTime,
            phaseTransitionCount = phaseTransitions.size
        ),
        confidenceScore = confidence
    )
}

private fun SleepSession.calculatePhasePercentage(phaseDuration: Long): Float {
    val total = lightSleepDuration + deepSleepDuration + remSleepDuration + awakeDuration
    return if (total > 0) (phaseDuration.toFloat() / total) * 100f else 0f
}

/**
 * Check if session is suitable for AI analysis
 */
fun SleepSession.isAIAnalysisReady(): Boolean {
    return duration >= 2 * 60 * 60 * 1000L && // At least 2 hours
            confidence >= 0.5f &&
            sleepEfficiency > 0f &&
            (lightSleepDuration + deepSleepDuration + remSleepDuration) > 0L
}

/**
 * Get AI analysis priority level
 */
fun SleepSession.getAIAnalysisPriority(): AIAnalysisPriority {
    return when {
        sleepQualityScore?.let { it < 4f } == true -> AIAnalysisPriority.HIGH
        sleepEfficiency < 60f -> AIAnalysisPriority.HIGH
        duration > 12 * 60 * 60 * 1000L -> AIAnalysisPriority.MEDIUM // Very long sleep
        duration < 4 * 60 * 60 * 1000L -> AIAnalysisPriority.MEDIUM // Very short sleep
        phaseTransitions.size > 50 -> AIAnalysisPriority.MEDIUM // Very restless
        else -> AIAnalysisPriority.LOW
    }
}

// ========== SLEEP SESSIONS COLLECTION EXTENSIONS ==========

/**
 * Generate trend analysis prompt for multiple sessions
 */
fun List<SleepSession>.toTrendAnalysisPrompt(
    timeframe: String = "recent",
    focusAreas: List<TrendFocusArea> = listOf(TrendFocusArea.QUALITY, TrendFocusArea.PATTERNS)
): String {
    return DataConverter.sleepSessionsToTrendPrompt(this, "trend").let { basePrompt ->
        if (focusAreas.isNotEmpty()) {
            basePrompt + "\n\nFocus specifically on: ${focusAreas.joinToString { it.description }}"
        } else basePrompt
    }
}

/**
 * Extract comparative metrics across sessions
 */
fun List<SleepSession>.extractComparativeMetrics(): ComparativeMetricsBundle {
    if (isEmpty()) return ComparativeMetricsBundle.empty()

    val qualityScores = mapNotNull { it.sleepQualityScore }
    val efficiencies = map { it.sleepEfficiency }
    val durations = map { it.duration }

    return ComparativeMetricsBundle(
        sessionCount = size,
        timeRange = TimeRange(
            minOf { it.startTime },
            maxOf { it.startTime },
            "Analysis period"
        ),
        qualityMetrics = if (qualityScores.isNotEmpty()) {
            MetricStatistics(
                average = qualityScores.average().toFloat(),
                min = qualityScores.minOrNull() ?: 0f,
                max = qualityScores.maxOrNull() ?: 0f,
                standardDeviation = qualityScores.standardDeviation(),
                trend = qualityScores.calculateTrendDirection()
            )
        } else null,
        efficiencyMetrics = MetricStatistics(
            average = efficiencies.average().toFloat(),
            min = efficiencies.minOrNull() ?: 0f,
            max = efficiencies.maxOrNull() ?: 0f,
            standardDeviation = efficiencies.map { it.toDouble() }.standardDeviation().toFloat(),
            trend = efficiencies.calculateTrendDirection()
        ),
        durationMetrics = MetricStatistics(
            average = durations.average().toFloat(),
            min = durations.minOrNull()?.toFloat() ?: 0f,
            max = durations.maxOrNull()?.toFloat() ?: 0f,
            standardDeviation = durations.map { it.toDouble() }.standardDeviation().toFloat(),
            trend = durations.map { it.toFloat() }.calculateTrendDirection()
        )
    )
}

/**
 * Filter sessions suitable for AI analysis
 */
fun List<SleepSession>.filterForAIAnalysis(): List<SleepSession> {
    return filter { it.isAIAnalysisReady() }
}

/**
 * Group sessions by analysis priority
 */
fun List<SleepSession>.groupByAIAnalysisPriority(): Map<AIAnalysisPriority, List<SleepSession>> {
    return groupBy { it.getAIAnalysisPriority() }
}

/**
 * Find sessions with anomalies for AI analysis
 */
fun List<SleepSession>.findAnomaliesForAI(): List<SleepSession> {
    if (size < 3) return emptyList()

    val qualityScores = mapNotNull { it.sleepQualityScore }
    val efficiencies = map { it.sleepEfficiency }
    val durations = map { it.duration.toFloat() }

    val qualityOutliers = qualityScores.findOutliers()
    val efficiencyOutliers = efficiencies.findOutliers()
    val durationOutliers = durations.findOutliers()

    return filter { session ->
        session.sleepQualityScore in qualityOutliers ||
                session.sleepEfficiency in efficiencyOutliers ||
                session.duration.toFloat() in durationOutliers
    }
}

// ========== AI RESPONSE EXTENSIONS ==========

/**
 * Parse AI response into structured insights
 */
fun String.parseToInsights(
    sessionId: Long,
    aiModel: String = "AI Generated",
    confidenceThreshold: Float = MIN_CONFIDENCE_SCORE
): List<ProcessedInsight> {
    return DataConverter.parseAIResponseToInsights(this, sessionId, aiModel)
        .filter { it.confidence >= confidenceThreshold }
}

/**
 * Extract recommendations from AI response
 */
fun String.extractRecommendations(maxCount: Int = 5): List<String> {
    return FormatUtils.extractRecommendations(this).take(maxCount)
}

/**
 * Validate AI response quality
 */
fun String.validateAIResponseQuality(): AIResponseQuality {
    val hasStructure = contains(":") && lines().count { it.trim().isNotEmpty() } >= 3
    val hasRecommendations = contains("recommend", ignoreCase = true) || contains("suggest", ignoreCase = true)
    val hasSleepTerms = listOf("sleep", "quality", "efficiency", "duration").any {
        contains(it, ignoreCase = true)
    }
    val confidence = FormatUtils.extractConfidenceScore(this)

    val qualityScore = listOf(
        if (hasStructure) 0.3f else 0f,
        if (hasRecommendations) 0.3f else 0f,
        if (hasSleepTerms) 0.2f else 0f,
        if (length > 100) 0.2f else 0f
    ).sum()

    return AIResponseQuality(
        overallScore = qualityScore,
        hasStructure = hasStructure,
        hasRecommendations = hasRecommendations,
        hasSleepRelevantContent = hasSleepTerms,
        confidence = confidence,
        wordCount = split("\\s+".toRegex()).size,
        isAcceptable = qualityScore >= 0.6f
    )
}

/**
 * Extract confidence score with fallback
 */
fun String.extractConfidenceWithFallback(fallback: Float = 0.7f): Float {
    return FormatUtils.extractConfidenceScore(this).takeIf { it > 0f } ?: fallback
}

// ========== AI MODEL EXTENSIONS ==========

/**
 * Get optimal configuration for sleep analysis
 */
fun AIModel.getOptimalSleepConfig(): ModelConfig? {
    return AIConfiguration.getModelConfig(this)?.let { config ->
        when (this) {
            AIModel.GPT3_5_TURBO -> config.copy(
                temperature = 0.3f, // More consistent for analysis
                maxTokens = 1000,   // Sufficient for sleep insights
            )
            AIModel.GPT4 -> config.copy(
                temperature = 0.4f, // Slightly higher for more nuanced analysis
                maxTokens = 2000,   // More detailed analysis
            )
            AIModel.CLAUDE_3_SONNET -> config.copy(
                temperature = 0.3f,
                maxTokens = 1500,
            )
            AIModel.GEMINI_PRO -> config.copy(
                temperature = 0.2f, // Very consistent
                maxTokens = 1200,
            )
            else -> config
        }
    }
}

/**
 * Check if model is suitable for operation type
 */
fun AIModel.isSuitableFor(operationType: AIOperationType): Boolean {
    val config = AIConfiguration.getModelConfig(this) ?: return false

    return when (operationType) {
        AIOperationType.SLEEP_ANALYSIS -> ModelCapability.SLEEP_ANALYSIS in config.capabilities
        AIOperationType.PATTERN_RECOGNITION -> ModelCapability.ADVANCED_REASONING in config.capabilities
        AIOperationType.REPORT_GENERATION -> config.maxTokens >= 2000
        AIOperationType.TREND_ANALYSIS -> ModelCapability.COMPLEX_ANALYSIS in config.capabilities
        else -> true
    }
}

/**
 * Estimate cost for operation
 */
fun AIModel.estimateCost(estimatedTokens: Int): Double {
    return AIConfiguration.getModelConfig(this)?.let { config ->
        estimatedTokens * config.costPerToken
    } ?: (estimatedTokens * ESTIMATED_COST_PER_TOKEN)
}

/**
 * Get recommended models for operation type
 */
fun AIOperationType.getRecommendedModels(): List<AIModel> {
    return AIConfiguration.getModelsWithCapability(
        when (this) {
            AIOperationType.SLEEP_ANALYSIS -> ModelCapability.SLEEP_ANALYSIS
            AIOperationType.PATTERN_RECOGNITION -> ModelCapability.ADVANCED_REASONING
            AIOperationType.TREND_ANALYSIS -> ModelCapability.COMPLEX_ANALYSIS
            AIOperationType.REPORT_GENERATION -> ModelCapability.TEXT_GENERATION
            else -> ModelCapability.TEXT_GENERATION
        }
    ).sortedBy { model ->
        // Sort by cost efficiency
        model.estimateCost(1000)
    }
}

// ========== PERFORMANCE MONITORING EXTENSIONS ==========

/**
 * Track AI operation with automatic error handling
 */
suspend fun <T> AIModel.executeTrackedOperation(
    operationId: String,
    operationType: AIOperationType,
    sessionId: String? = null,
    operation: suspend () -> T
): T {
    val tracker = AIPerformanceMonitor.startOperation(
        operationId = operationId,
        operationType = operationType,
        model = this,
        sessionId = sessionId
    )

    return try {
        val result = operation()
        tracker.recordSuccess()
        result
    } catch (error: Throwable) {
        tracker.recordError(error.message ?: "Unknown error")

        // Use error handler for recovery
        val errorResult = AIErrorHandler.handleError(
            operationId = operationId,
            operationType = operationType,
            model = this,
            error = error,
            operation = operation
        )

        if (errorResult.isSuccess) {
            errorResult.value!!
        } else {
            throw errorResult.error ?: error
        }
    }
}

/**
 * Execute with automatic fallback and performance tracking
 */
suspend fun <T> AIOperationType.executeWithFallback(
    operationId: String,
    primaryModel: AIModel = AIModel.GPT3_5_TURBO,
    operation: suspend (AIModel) -> T
): T {
    val recommendedModels = getRecommendedModels()
    val modelsToTry = listOfNotNull(primaryModel) + recommendedModels.filter { it != primaryModel }

    return AIErrorHandler.executeWithFallback(
        operationId = operationId,
        operationType = this,
        primaryModel = primaryModel,
        fallbackModels = modelsToTry.drop(1),
        operation = operation
    ).let { result ->
        if (result.isSuccess) {
            result.value!!
        } else {
            throw result.error ?: RuntimeException("All fallback attempts failed")
        }
    }
}

// ========== VALIDATION EXTENSIONS ==========

/**
 * Validate and sanitize for AI processing
 */
fun SleepSession.validateForAI(): ValidationResult {
    return ValidationUtils.validateSleepSession(this)
}

/**
 * Batch validate sessions
 */
fun List<SleepSession>.batchValidateForAI(): BatchValidationResult {
    return ValidationUtils.validateBatchData(this)
}

/**
 * Get AI-ready sessions with validation
 */
fun List<SleepSession>.getAIReadySessions(
    minConfidence: Float = 0.7f
): List<SleepSession> {
    return filter { session ->
        val validation = session.validateForAI()
        validation.isValid && validation.validationScore >= minConfidence
    }
}

// ========== FORMATTING EXTENSIONS ==========

/**
 * Format for AI prompt with specific focus
 */
fun SleepSession.formatForAI(focus: AIFocus): String {
    return when (focus) {
        AIFocus.QUALITY_ASSESSMENT -> FormatUtils.formatSleepMetricsForAI(
            quality = sleepQualityScore,
            efficiency = sleepEfficiency,
            duration = duration,
            latency = sleepLatency
        )

        AIFocus.SLEEP_ARCHITECTURE -> FormatUtils.formatSleepArchitectureForAI(
            lightSleep = lightSleepDuration,
            deepSleep = deepSleepDuration,
            remSleep = remSleepDuration,
            awakeTime = awakeDuration
        )

        AIFocus.ENVIRONMENTAL_FACTORS -> FormatUtils.formatEnvironmentalDataForAI(
            avgMovement = averageMovementIntensity,
            movementEvents = movementEvents.size,
            avgNoise = averageNoiseLevel,
            noiseEvents = noiseEvents.size,
            maxNoise = noiseEvents.maxOfOrNull { it.decibelLevel }
        )

        AIFocus.COMPREHENSIVE -> toAIPrompt(AIAnalysisType.COMPREHENSIVE)
    }
}

/**
 * Format insights for display
 */
fun List<ProcessedInsight>.formatForDisplay(maxLength: Int = 100): List<String> {
    return map { insight ->
        val text = "${insight.title}: ${insight.description}"
        if (text.length > maxLength) {
            text.take(maxLength - 3) + "..."
        } else text
    }
}

// ========== ASYNC PROCESSING EXTENSIONS ==========

/**
 * Process sessions concurrently with AI
 */
suspend fun List<SleepSession>.processWithAIConcurrently(
    operation: suspend (SleepSession) -> ProcessedInsight,
    maxConcurrency: Int = 3
): List<ProcessedInsight> = coroutineScope {

    val semaphore = Semaphore(maxConcurrency)

    map { session ->
        async {
            semaphore.withPermit {
                try {
                    operation(session)
                } catch (e: Exception) {
                    // Return empty insight on error
                    ProcessedInsight.createEmpty(session.id, e.message ?: "Processing failed")
                }
            }
        }
    }.awaitAll()
}

/**
 * Generate insights flow for real-time processing
 */
fun List<SleepSession>.generateInsightsFlow(
    batchSize: Int = 5,
    delayBetweenBatches: Long = 1000L
): Flow<List<ProcessedInsight>> = flow {

    chunked(batchSize).forEach { batch ->
        val insights = batch.mapNotNull { session ->
            if (session.isAIAnalysisReady()) {
                try {
                    // Simulate AI processing
                    session.generateBasicInsight()
                } catch (e: Exception) {
                    null
                }
            } else null
        }

        emit(insights)
        delay(delayBetweenBatches)
    }
}

// ========== UTILITY EXTENSIONS ==========

/**
 * Generate basic insight from session (fallback method)
 */
fun SleepSession.generateBasicInsight(): ProcessedInsight {
    val qualityAssessment = sleepQualityScore?.let { score ->
        when {
            score >= 8f -> "Excellent sleep quality achieved"
            score >= 6f -> "Good sleep quality with room for improvement"
            score >= 4f -> "Average sleep quality - focus on optimization"
            else -> "Poor sleep quality - requires attention"
        }
    } ?: "Sleep quality assessment unavailable"

    val recommendation = when {
        sleepEfficiency < 70f -> "Focus on improving sleep efficiency through better sleep hygiene"
        duration < 6 * 60 * 60 * 1000L -> "Consider extending sleep duration for better recovery"
        sleepLatency > 30 * 60 * 1000L -> "Work on reducing time to fall asleep"
        else -> "Maintain current sleep patterns and monitor consistency"
    }

    return ProcessedInsight(
        id = "basic_${id}_${System.currentTimeMillis()}",
        originalInsight = SleepInsight(
            id = "basic_${id}",
            sessionId = id,
            category = InsightCategory.QUALITY,
            title = "Sleep Quality Analysis",
            description = qualityAssessment,
            recommendation = recommendation,
            priority = 2,
            timestamp = System.currentTimeMillis(),
            isAiGenerated = false,
            confidence = 0.6f
        ),
        category = InsightCategory.QUALITY,
        priority = 2,
        title = "Sleep Quality Analysis",
        description = qualityAssessment,
        recommendation = recommendation,
        evidence = emptyList(),
        dataPoints = emptyList(),
        confidence = 0.6f,
        qualityScore = 0.6f,
        relevanceScore = 0.8f,
        actionabilityScore = 0.7f,
        noveltyScore = 0.3f,
        personalizationScore = 0.4f,
        validationResults = ValidationResult(
            isValid = true,
            validationScore = 0.6f,
            validationChecks = mapOf("basic_generation" to true),
            validationMessages = emptyList()
        ),
        aiGenerated = false,
        aiModelUsed = "Basic Rule Engine",
        processingMetadata = ProcessingMetadata(
            processingVersion = "1.0",
            processingTimeMs = 1L,
            algorithmUsed = "Basic Analysis",
            parametersUsed = emptyMap(),
            dataSourcesCount = 1,
            validationSteps = listOf("basic_validation")
        ),
        implementationDifficulty = ImplementationDifficulty.LOW,
        expectedImpact = ExpectedImpact.MEDIUM,
        timeToImpact = TimeToImpact.SHORT_TERM,
        timestamp = System.currentTimeMillis()
    )
}

// ========== STATISTICAL EXTENSIONS ==========

/**
 * Calculate standard deviation for lists
 */
fun List<Double>.standardDeviation(): Double {
    if (size <= 1) return 0.0
    val mean = average()
    val variance = map { (it - mean).pow(2) }.average()
    return sqrt(variance)
}

fun List<Float>.standardDeviation(): Float {
    return map { it.toDouble() }.standardDeviation().toFloat()
}

/**
 * Find statistical outliers using IQR method
 */
fun <T : Number> List<T>.findOutliers(): List<T> {
    if (size < 4) return emptyList()

    val sorted = sortedBy { it.toDouble() }
    val q1Index = size / 4
    val q3Index = 3 * size / 4

    val q1 = sorted[q1Index].toDouble()
    val q3 = sorted[q3Index].toDouble()
    val iqr = q3 - q1

    val lowerBound = q1 - 1.5 * iqr
    val upperBound = q3 + 1.5 * iqr

    return filter {
        val value = it.toDouble()
        value < lowerBound || value > upperBound
    }
}

/**
 * Calculate trend direction for numeric lists
 */
fun List<Float>.calculateTrendDirection(): TrendDirection {
    if (size < 2) return TrendDirection.INSUFFICIENT_DATA

    val firstHalf = take(size / 2).average()
    val secondHalf = drop(size / 2).average()

    val change = ((secondHalf - firstHalf) / firstHalf * 100).toFloat()

    return when {
        abs(change) < 5f -> TrendDirection.STABLE
        change > 10f -> TrendDirection.IMPROVING
        change > 5f -> TrendDirection.IMPROVING
        change < -10f -> TrendDirection.DECLINING
        change < -5f -> TrendDirection.DECLINING
        else -> TrendDirection.STABLE
    }
}

// ========== FACTORY EXTENSIONS ==========

/**
 * Create empty insight for error cases
 */
fun ProcessedInsight.Companion.createEmpty(sessionId: Long, errorMessage: String): ProcessedInsight {
    return ProcessedInsight(
        id = "empty_${sessionId}_${System.currentTimeMillis()}",
        originalInsight = SleepInsight(
            id = "empty_${sessionId}",
            sessionId = sessionId,
            category = InsightCategory.QUALITY,
            title = "Analysis Unavailable",
            description = "Unable to generate insight: $errorMessage",
            recommendation = "Try again later or check data quality",
            priority = 3,
            timestamp = System.currentTimeMillis(),
            isAiGenerated = false,
            confidence = 0.1f
        ),
        category = InsightCategory.QUALITY,
        priority = 3,
        title = "Analysis Unavailable",
        description = "Unable to generate insight: $errorMessage",
        recommendation = "Try again later or check data quality",
        evidence = emptyList(),
        dataPoints = emptyList(),
        confidence = 0.1f,
        qualityScore = 0.1f,
        relevanceScore = 0.1f,
        actionabilityScore = 0.1f,
        noveltyScore = 0.1f,
        personalizationScore = 0.1f,
        validationResults = ValidationResult(
            isValid = false,
            validationScore = 0.1f,
            validationChecks = mapOf("empty_generation" to false),
            validationMessages = listOf(errorMessage)
        ),
        aiGenerated = false,
        aiModelUsed = "Error Handler",
        processingMetadata = ProcessingMetadata(
            processingVersion = "1.0",
            processingTimeMs = 1L,
            algorithmUsed = "Error Generation",
            parametersUsed = emptyMap(),
            dataSourcesCount = 0,
            validationSteps = emptyList()
        ),
        implementationDifficulty = ImplementationDifficulty.LOW,
        expectedImpact = ExpectedImpact.LOW,
        timeToImpact = TimeToImpact.IMMEDIATE,
        timestamp = System.currentTimeMillis()
    )
}

// ========== DATA CLASSES FOR EXTENSIONS ==========

data class AIMetricsBundle(
    val sessionId: Long,
    val qualityScore: Float?,
    val efficiencyScore: Float,
    val durationHours: Float,
    val sleepLatencyMinutes: Long,
    val sleepArchitecture: SleepArchitectureMetrics,
    val environmentalMetrics: EnvironmentalMetrics,
    val timingMetrics: TimingMetrics,
    val confidenceScore: Float
)

data class SleepArchitectureMetrics(
    val lightSleepPercent: Float,
    val deepSleepPercent: Float,
    val remSleepPercent: Float,
    val awakePercent: Float
) {
    val isBalanced: Boolean
        get() = lightSleepPercent in 45f..55f &&
                deepSleepPercent in 15f..25f &&
                remSleepPercent in 20f..25f &&
                awakePercent <= 10f
}

data class EnvironmentalMetrics(
    val avgMovementIntensity: Float,
    val movementEventCount: Int,
    val avgNoiseLevel: Float,
    val noiseEventCount: Int,
    val maxNoiseLevel: Float?
) {
    val movementLevel: String
        get() = when {
            avgMovementIntensity < 2f -> "Very Low"
            avgMovementIntensity < 4f -> "Low"
            avgMovementIntensity < 6f -> "Moderate"
            avgMovementIntensity < 8f -> "High"
            else -> "Very High"
        }

    val noiseLevel: String
        get() = when {
            avgNoiseLevel < 30f -> "Very Quiet"
            avgNoiseLevel < 40f -> "Quiet"
            avgNoiseLevel < 50f -> "Moderate"
            avgNoiseLevel < 60f -> "Noisy"
            else -> "Very Noisy"
        }
}

data class TimingMetrics(
    val bedtime: Long,
    val wakeTime: Long?,
    val phaseTransitionCount: Int
)

data class ComparativeMetricsBundle(
    val sessionCount: Int,
    val timeRange: TimeRange,
    val qualityMetrics: MetricStatistics?,
    val efficiencyMetrics: MetricStatistics,
    val durationMetrics: MetricStatistics
) {
    companion object {
        fun empty() = ComparativeMetricsBundle(
            sessionCount = 0,
            timeRange = TimeRange(0L, 0L, "Empty"),
            qualityMetrics = null,
            efficiencyMetrics = MetricStatistics.empty(),
            durationMetrics = MetricStatistics.empty()
        )
    }
}

data class MetricStatistics(
    val average: Float,
    val min: Float,
    val max: Float,
    val standardDeviation: Float,
    val trend: TrendDirection
) {
    companion object {
        fun empty() = MetricStatistics(
            average = 0f,
            min = 0f,
            max = 0f,
            standardDeviation = 0f,
            trend = TrendDirection.STABLE
        )
    }
}

data class AIResponseQuality(
    val overallScore: Float,
    val hasStructure: Boolean,
    val hasRecommendations: Boolean,
    val hasSleepRelevantContent: Boolean,
    val confidence: Float,
    val wordCount: Int,
    val isAcceptable: Boolean
)

// ========== ENUMS ==========

enum class AIAnalysisType {
    QUICK,
    COMPREHENSIVE,
    COMPARATIVE,
    PATTERN_FOCUSED
}

enum class AIAnalysisPriority {
    LOW,
    MEDIUM,
    HIGH
}

enum class TrendFocusArea(val description: String) {
    QUALITY("sleep quality trends"),
    DURATION("sleep duration patterns"),
    EFFICIENCY("sleep efficiency changes"),
    TIMING("bedtime and wake time consistency"),
    PATTERNS("recurring sleep patterns"),
    ENVIRONMENTAL("environmental factor impacts")
}

enum class AIFocus {
    QUALITY_ASSESSMENT,
    SLEEP_ARCHITECTURE,
    ENVIRONMENTAL_FACTORS,
    COMPREHENSIVE
}