package com.example.somniai.ai

import android.os.Parcelable
import com.example.somniai.data.*
import kotlinx.parcelize.Parcelize
import java.util.*
import kotlin.math.roundToInt

/**
 * Comprehensive AI Insights Data Models for SomniAI
 *
 * Enterprise-grade data models supporting:
 * - Multi-service AI integration (OpenAI, Claude, Gemini)
 * - Advanced insight generation and processing
 * - AI performance monitoring and analytics
 * - Machine learning personalization
 * - AI error analysis and pattern recognition
 * - Prompt engineering and response optimization
 * - AI service orchestration and fallback strategies
 * - Comprehensive AI analytics and metrics
 * - User context and personalization models
 * - AI configuration and model management
 */

// ========== AI SERVICE INTEGRATION MODELS ==========

/**
 * Supported AI models with capabilities and configurations
 */
enum class AIModel(
    val serviceName: String,
    val modelId: String,
    val displayName: String,
    val capabilities: Set<AICapability>,
    val maxTokens: Int,
    val costPerToken: Double,
    val averageLatency: Long,
    val reliability: Float
) {
    // OpenAI Models
    GPT4(
        serviceName = "openai",
        modelId = "gpt-4-1106-preview",
        displayName = "GPT-4 Turbo",
        capabilities = setOf(
            AICapability.TEXT_GENERATION,
            AICapability.REASONING,
            AICapability.STRUCTURED_OUTPUT,
            AICapability.FUNCTION_CALLING,
            AICapability.CONTEXT_UNDERSTANDING
        ),
        maxTokens = 128000,
        costPerToken = 0.00001,
        averageLatency = 3500L,
        reliability = 0.98f
    ),

    GPT3_5_TURBO(
        serviceName = "openai",
        modelId = "gpt-3.5-turbo-1106",
        displayName = "GPT-3.5 Turbo",
        capabilities = setOf(
            AICapability.TEXT_GENERATION,
            AICapability.FUNCTION_CALLING,
            AICapability.STRUCTURED_OUTPUT
        ),
        maxTokens = 16385,
        costPerToken = 0.000001,
        averageLatency = 1500L,
        reliability = 0.95f
    ),

    // Anthropic Claude Models
    CLAUDE_3_OPUS(
        serviceName = "anthropic",
        modelId = "claude-3-opus-20240229",
        displayName = "Claude 3 Opus",
        capabilities = setOf(
            AICapability.TEXT_GENERATION,
            AICapability.REASONING,
            AICapability.ANALYSIS,
            AICapability.CONTEXT_UNDERSTANDING,
            AICapability.SAFETY_FILTERING
        ),
        maxTokens = 200000,
        costPerToken = 0.000015,
        averageLatency = 4000L,
        reliability = 0.97f
    ),

    CLAUDE_3_SONNET(
        serviceName = "anthropic",
        modelId = "claude-3-sonnet-20240229",
        displayName = "Claude 3 Sonnet",
        capabilities = setOf(
            AICapability.TEXT_GENERATION,
            AICapability.REASONING,
            AICapability.ANALYSIS,
            AICapability.CONTEXT_UNDERSTANDING
        ),
        maxTokens = 200000,
        costPerToken = 0.000003,
        averageLatency = 2000L,
        reliability = 0.96f
    ),

    // Google Gemini Models
    GEMINI_PRO(
        serviceName = "google",
        modelId = "gemini-pro",
        displayName = "Gemini Pro",
        capabilities = setOf(
            AICapability.TEXT_GENERATION,
            AICapability.REASONING,
            AICapability.MULTIMODAL,
            AICapability.FUNCTION_CALLING
        ),
        maxTokens = 32768,
        costPerToken = 0.0000005,
        averageLatency = 2500L,
        reliability = 0.94f
    ),

    GEMINI_PRO_VISION(
        serviceName = "google",
        modelId = "gemini-pro-vision",
        displayName = "Gemini Pro Vision",
        capabilities = setOf(
            AICapability.TEXT_GENERATION,
            AICapability.MULTIMODAL,
            AICapability.IMAGE_ANALYSIS,
            AICapability.REASONING
        ),
        maxTokens = 16384,
        costPerToken = 0.00000025,
        averageLatency = 3000L,
        reliability = 0.92f
    );

    /**
     * Check if model supports specific capability
     */
    fun supports(capability: AICapability): Boolean = capability in capabilities

    /**
     * Get estimated cost for token count
     */
    fun estimateCost(tokens: Int): Double = tokens * costPerToken

    /**
     * Get performance score based on latency, cost, and reliability
     */
    fun getPerformanceScore(): Float {
        val latencyScore = (5000f - averageLatency) / 5000f // Lower latency = higher score
        val costScore = (0.00001 - costPerToken).toFloat() * 1000000f // Lower cost = higher score
        val reliabilityScore = reliability

        return ((latencyScore * 0.3f) + (costScore * 0.2f) + (reliabilityScore * 0.5f))
            .coerceIn(0f, 1f)
    }

    /**
     * Check if model is suitable for real-time operations
     */
    fun isRealTimeCapable(): Boolean = averageLatency <= 2000L && reliability >= 0.95f

    companion object {
        /**
         * Get models by service
         */
        fun byService(serviceName: String): List<AIModel> =
            values().filter { it.serviceName == serviceName }

        /**
         * Get models supporting specific capability
         */
        fun withCapability(capability: AICapability): List<AIModel> =
            values().filter { it.supports(capability) }

        /**
         * Get best model for specific requirements
         */
        fun getBestModel(
            capability: AICapability,
            maxLatency: Long = Long.MAX_VALUE,
            maxCost: Double = Double.MAX_VALUE,
            minReliability: Float = 0.9f
        ): AIModel? {
            return values()
                .filter {
                    it.supports(capability) &&
                            it.averageLatency <= maxLatency &&
                            it.costPerToken <= maxCost &&
                            it.reliability >= minReliability
                }
                .maxByOrNull { it.getPerformanceScore() }
        }
    }
}

/**
 * AI capabilities for model selection and validation
 */
enum class AICapability {
    TEXT_GENERATION,
    REASONING,
    ANALYSIS,
    STRUCTURED_OUTPUT,
    FUNCTION_CALLING,
    CONTEXT_UNDERSTANDING,
    MULTIMODAL,
    IMAGE_ANALYSIS,
    SAFETY_FILTERING,
    STREAMING,
    FINE_TUNING,
    EMBEDDINGS
}

/**
 * AI service configuration for each provider
 */
@Parcelize
data class AIServiceConfig(
    val serviceName: String,
    val isEnabled: Boolean,
    val apiKey: String? = null,
    val baseUrl: String,
    val timeout: Long = 30000L,
    val retryAttempts: Int = 3,
    val rateLimitRequests: Int = 20,
    val rateLimitWindow: Long = 60000L,
    val priority: Int = 1,
    val fallbackEnabled: Boolean = true,
    val customHeaders: Map<String, String> = emptyMap(),
    val models: List<AIModel> = emptyList()
) : Parcelable {

    val isConfigured: Boolean
        get() = isEnabled && !apiKey.isNullOrBlank()

    companion object {
        fun openAI(apiKey: String) = AIServiceConfig(
            serviceName = "openai",
            isEnabled = true,
            apiKey = apiKey,
            baseUrl = "https://api.openai.com/v1/",
            models = listOf(AIModel.GPT4, AIModel.GPT3_5_TURBO)
        )

        fun anthropic(apiKey: String) = AIServiceConfig(
            serviceName = "anthropic",
            isEnabled = true,
            apiKey = apiKey,
            baseUrl = "https://api.anthropic.com/v1/",
            customHeaders = mapOf("anthropic-version" to "2023-06-01"),
            models = listOf(AIModel.CLAUDE_3_OPUS, AIModel.CLAUDE_3_SONNET)
        )

        fun google(apiKey: String) = AIServiceConfig(
            serviceName = "google",
            isEnabled = true,
            apiKey = apiKey,
            baseUrl = "https://generativelanguage.googleapis.com/v1beta/",
            models = listOf(AIModel.GEMINI_PRO, AIModel.GEMINI_PRO_VISION)
        )
    }
}

// ========== AI INSIGHT GENERATION MODELS ==========

/**
 * Types of insights that can be generated
 */
enum class InsightGenerationType {
    POST_SESSION,
    DAILY_ANALYSIS,
    WEEKLY_SUMMARY,
    TREND_ANALYSIS,
    PERSONALIZED_ANALYSIS,
    COMPARATIVE_ANALYSIS,
    PREDICTIVE_ANALYSIS,
    EMERGENCY_ANALYSIS,
    PATTERN_RECOGNITION,
    GOAL_TRACKING,
    HEALTH_ASSESSMENT,
    RECOMMENDATION_ENGINE
}

/**
 * Context for AI insight generation
 */
@Parcelize
data class InsightGenerationContext(
    val generationType: InsightGenerationType,
    val sessionData: SleepSession? = null,
    val sessionsData: List<SleepSession> = emptyList(),
    val qualityAnalysis: SessionQualityAnalysis? = null,
    val sessionSummary: SessionSummary? = null,
    val trendAnalysis: TrendAnalysis? = null,
    val patternAnalysis: SleepPatternAnalysis? = null,
    val personalBaseline: PersonalBaseline? = null,
    val habitAnalysis: HabitAnalysis? = null,
    val goalAnalysis: GoalAnalysis? = null,
    val onsetAnalysis: SleepOnsetAnalysis? = null,
    val userProfile: UserProfileContext? = null,
    val environmentContext: EnvironmentContext? = null,
    val healthContext: HealthContext? = null,
    val preferences: InsightPreferences = InsightPreferences(),
    val priority: InsightPriority = InsightPriority.NORMAL,
    val requestId: String = generateRequestId(),
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable {

    /**
     * Get data completeness score for quality assessment
     */
    fun getDataCompletenessScore(): Float {
        var score = 0f
        var maxScore = 0f

        // Session data
        if (sessionData != null) score += 3f
        maxScore += 3f

        // Historical data
        if (sessionsData.isNotEmpty()) score += 2f
        maxScore += 2f

        // Analysis data
        if (qualityAnalysis != null) score += 2f
        if (trendAnalysis != null) score += 2f
        if (patternAnalysis != null) score += 1f
        maxScore += 5f

        // Context data
        if (userProfile != null) score += 1f
        if (environmentContext != null) score += 1f
        if (healthContext != null) score += 1f
        maxScore += 3f

        return if (maxScore > 0) score / maxScore else 0f
    }

    /**
     * Check if sufficient data is available for generation
     */
    fun hasSufficientData(): Boolean {
        return when (generationType) {
            InsightGenerationType.POST_SESSION -> sessionData != null
            InsightGenerationType.DAILY_ANALYSIS -> sessionsData.size >= 3
            InsightGenerationType.TREND_ANALYSIS -> sessionsData.size >= 7
            InsightGenerationType.PERSONALIZED_ANALYSIS ->
                userProfile != null && sessionsData.size >= 5
            else -> getDataCompletenessScore() >= 0.3f
        }
    }

    companion object {
        private fun generateRequestId(): String = "ig_${System.currentTimeMillis()}_${Random().nextInt(1000)}"
    }
}

/**
 * Request for AI insight generation
 */
@Parcelize
data class AIInsightGenerationRequest(
    val generationType: InsightGenerationType,
    val context: InsightGenerationContext,
    val targetModel: AIModel? = null,
    val modelPreferences: ModelSelectionPreferences = ModelSelectionPreferences(),
    val outputPreferences: OutputPreferences = OutputPreferences(),
    val personalizationLevel: PersonalizationLevel = PersonalizationLevel.ADAPTIVE,
    val priority: RequestPriority = RequestPriority.NORMAL,
    val timeout: Long = 45000L,
    val retryOnFailure: Boolean = true,
    val allowFallback: Boolean = true,
    val sessionId: Long? = null,
    val userId: String? = null,
    val userContext: UserContext? = null,
    val deviceCapabilities: DeviceCapabilities? = null,
    val privacySettings: PrivacySettings? = null,
    val requestMetadata: Map<String, Any> = emptyMap()
) : Parcelable {

    /**
     * Get estimated complexity score for model selection
     */
    fun getComplexityScore(): Float {
        var complexity = 0f

        // Base complexity by type
        complexity += when (generationType) {
            InsightGenerationType.POST_SESSION -> 0.3f
            InsightGenerationType.DAILY_ANALYSIS -> 0.5f
            InsightGenerationType.TREND_ANALYSIS -> 0.7f
            InsightGenerationType.PERSONALIZED_ANALYSIS -> 0.8f
            InsightGenerationType.PREDICTIVE_ANALYSIS -> 0.9f
            else -> 0.6f
        }

        // Data volume complexity
        val sessionCount = context.sessionsData.size
        complexity += when {
            sessionCount > 30 -> 0.3f
            sessionCount > 10 -> 0.2f
            sessionCount > 3 -> 0.1f
            else -> 0f
        }

        // Personalization complexity
        complexity += when (personalizationLevel) {
            PersonalizationLevel.NONE -> 0f
            PersonalizationLevel.BASIC -> 0.1f
            PersonalizationLevel.ADAPTIVE -> 0.2f
            PersonalizationLevel.ADVANCED -> 0.3f
        }

        return complexity.coerceIn(0f, 1f)
    }

    /**
     * Get recommended models for this request
     */
    fun getRecommendedModels(): List<AIModel> {
        val complexity = getComplexityScore()
        val capability = AICapability.TEXT_GENERATION

        return when {
            complexity >= 0.8f -> listOf(AIModel.GPT4, AIModel.CLAUDE_3_OPUS)
            complexity >= 0.5f -> listOf(AIModel.CLAUDE_3_SONNET, AIModel.GPT4)
            else -> listOf(AIModel.GPT3_5_TURBO, AIModel.GEMINI_PRO, AIModel.CLAUDE_3_SONNET)
        }
    }
}

/**
 * Response from AI insight generation
 */
@Parcelize
data class AIInsightGenerationResponse(
    val jobId: String,
    val requestId: String,
    val status: GenerationStatus,
    val insights: List<GeneratedInsight> = emptyList(),
    val modelUsed: AIModel? = null,
    val processingTime: Long = 0L,
    val tokenUsage: TokenUsage? = null,
    val qualityScore: Float = 0f,
    val confidence: Float = 0f,
    val fallbackUsed: Boolean = false,
    val error: String? = null,
    val warnings: List<String> = emptyList(),
    val metadata: Map<String, Any> = emptyMap(),
    val generatedAt: Long = System.currentTimeMillis()
) : Parcelable {

    val isSuccessful: Boolean
        get() = status == GenerationStatus.COMPLETED && insights.isNotEmpty()

    val hasHighQuality: Boolean
        get() = qualityScore >= 0.8f && confidence >= 0.7f

    fun getAverageInsightQuality(): Float {
        return if (insights.isNotEmpty()) {
            insights.map { it.qualityScore }.average().toFloat()
        } else 0f
    }

    fun getInsightsByPriority(): Map<InsightPriority, List<GeneratedInsight>> {
        return insights.groupBy { it.priority }
    }

    fun getTopInsights(count: Int = 5): List<GeneratedInsight> {
        return insights
            .sortedWith(
                compareByDescending<GeneratedInsight> { it.priority.value }
                    .thenByDescending { it.qualityScore }
                    .thenByDescending { it.confidence }
            )
            .take(count)
    }
}

/**
 * Generated insight from AI with metadata
 */
@Parcelize
data class GeneratedInsight(
    val id: String = generateInsightId(),
    val category: InsightCategory,
    val type: InsightType,
    val priority: InsightPriority,
    val title: String,
    val description: String,
    val recommendation: String,
    val evidence: List<String> = emptyList(),
    val dataPoints: List<DataPoint> = emptyList(),
    val confidence: Float,
    val qualityScore: Float,
    val relevanceScore: Float,
    val actionability: Float,
    val personalizedFor: String? = null,
    val validityPeriod: Long = 0L,
    val tags: List<String> = emptyList(),
    val relatedInsights: List<String> = emptyList(),
    val generatedBy: AIModel,
    val generationContext: Map<String, Any> = emptyMap(),
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable {

    val isHighQuality: Boolean
        get() = qualityScore >= 0.8f && confidence >= 0.7f && relevanceScore >= 0.7f

    val isActionable: Boolean
        get() = actionability >= 0.6f && recommendation.isNotBlank()

    val isExpired: Boolean
        get() = validityPeriod > 0 && System.currentTimeMillis() > timestamp + validityPeriod

    fun toSleepInsight(): SleepInsight {
        return SleepInsight(
            sessionId = 0L, // Will be set externally
            category = category,
            title = title,
            description = description,
            recommendation = recommendation,
            priority = priority.value,
            isAiGenerated = true,
            timestamp = timestamp
        )
    }

    companion object {
        private fun generateInsightId(): String = "ai_insight_${System.currentTimeMillis()}_${Random().nextInt(10000)}"
    }
}

// ========== AI PERFORMANCE AND MONITORING MODELS ==========

/**
 * Performance metrics for AI models
 */
@Parcelize
data class ModelPerformanceMetrics(
    val model: AIModel,
    val timeWindow: Long = 3600000L, // 1 hour
    val totalRequests: Long = 0L,
    val successfulRequests: Long = 0L,
    val failedRequests: Long = 0L,
    val averageLatency: Long = 0L,
    val medianLatency: Long = 0L,
    val p95Latency: Long = 0L,
    val averageTokenUsage: Int = 0,
    val totalCost: Double = 0.0,
    val averageQuality: Float = 0f,
    val averageConfidence: Float = 0f,
    val errorRate: Float = 0f,
    val timeoutRate: Float = 0f,
    val qualityDistribution: Map<String, Int> = emptyMap(),
    val latencyDistribution: Map<String, Int> = emptyMap(),
    val lastUpdated: Long = System.currentTimeMillis()
) : Parcelable {

    val successRate: Float
        get() = if (totalRequests > 0) successfulRequests.toFloat() / totalRequests else 0f

    val costPerRequest: Double
        get() = if (totalRequests > 0) totalCost / totalRequests else 0.0

    val performanceScore: Float
        get() {
            val latencyScore = (5000f - averageLatency) / 5000f // Lower latency = higher score
            val qualityScore = averageQuality
            val reliabilityScore = successRate
            val costEfficiency = (0.01 - costPerRequest).toFloat() * 100f

            return ((latencyScore * 0.25f) + (qualityScore * 0.35f) +
                    (reliabilityScore * 0.25f) + (costEfficiency * 0.15f))
                .coerceIn(0f, 1f)
        }

    val isHighPerforming: Boolean
        get() = performanceScore >= 0.8f && successRate >= 0.95f && averageLatency <= 3000L

    fun getLatencyPercentile(percentile: Int): Long {
        return when (percentile) {
            50 -> medianLatency
            95 -> p95Latency
            else -> averageLatency
        }
    }
}

/**
 * Token usage tracking for cost management
 */
@Parcelize
data class TokenUsage(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int = promptTokens + completionTokens,
    val estimatedCost: Double = 0.0,
    val model: AIModel? = null
) : Parcelable {

    val costPerToken: Double
        get() = if (totalTokens > 0) estimatedCost / totalTokens else 0.0

    val efficiency: Float
        get() = if (promptTokens > 0) completionTokens.toFloat() / promptTokens else 0f

    fun isWithinBudget(maxCost: Double): Boolean = estimatedCost <= maxCost

    fun isEfficient(): Boolean = efficiency >= 0.5f && efficiency <= 2.0f
}

/**
 * AI service health monitoring
 */
@Parcelize
data class AIServiceHealth(
    val serviceName: String,
    val isHealthy: Boolean,
    val status: ServiceStatus,
    val lastHealthCheck: Long,
    val responseTime: Long,
    val errorRate: Float,
    val availabilityPercentage: Float,
    val quotaUsage: QuotaUsage? = null,
    val recentErrors: List<String> = emptyList(),
    val performanceMetrics: ServicePerformanceMetrics? = null,
    val recommendations: List<String> = emptyList()
) : Parcelable {

    val isOperational: Boolean
        get() = isHealthy && status == ServiceStatus.OPERATIONAL && errorRate < 0.1f

    val needsAttention: Boolean
        get() = !isHealthy || errorRate > 0.2f || availabilityPercentage < 95f

    fun getHealthGrade(): String {
        return when {
            availabilityPercentage >= 99f && errorRate < 0.01f -> "A+"
            availabilityPercentage >= 98f && errorRate < 0.05f -> "A"
            availabilityPercentage >= 95f && errorRate < 0.1f -> "B"
            availabilityPercentage >= 90f && errorRate < 0.2f -> "C"
            else -> "D"
        }
    }
}

enum class ServiceStatus {
    OPERATIONAL,
    DEGRADED,
    PARTIAL_OUTAGE,
    MAJOR_OUTAGE,
    MAINTENANCE,
    UNKNOWN
}

/**
 * Quota usage monitoring for rate limiting
 */
@Parcelize
data class QuotaUsage(
    val requestsUsed: Long,
    val requestsLimit: Long,
    val tokensUsed: Long,
    val tokensLimit: Long,
    val costUsed: Double,
    val costLimit: Double,
    val resetTime: Long,
    val timeWindow: Long
) : Parcelable {

    val requestsPercentage: Float
        get() = if (requestsLimit > 0) (requestsUsed.toFloat() / requestsLimit) * 100f else 0f

    val tokensPercentage: Float
        get() = if (tokensLimit > 0) (tokensUsed.toFloat() / tokensLimit) * 100f else 0f

    val costPercentage: Float
        get() = if (costLimit > 0) ((costUsed / costLimit) * 100f).toFloat() else 0f

    val isNearLimit: Boolean
        get() = requestsPercentage >= 80f || tokensPercentage >= 80f || costPercentage >= 80f

    val isOverLimit: Boolean
        get() = requestsPercentage >= 100f || tokensPercentage >= 100f || costPercentage >= 100f

    fun getTimeUntilReset(): Long = resetTime - System.currentTimeMillis()
}

// ========== AI PERSONALIZATION MODELS ==========

/**
 * User context for AI personalization
 */
@Parcelize
data class UserProfileContext(
    val userId: String,
    val age: Int? = null,
    val gender: String? = null,
    val sleepGoals: List<String> = emptyList(),
    val sleepChallenges: List<String> = emptyList(),
    val lifestyle: LifestyleContext? = null,
    val preferences: UserPreferences = UserPreferences(),
    val historicalPatterns: SleepPatterns? = null,
    val healthConditions: List<String> = emptyList(),
    val medications: List<String> = emptyList(),
    val timezone: String = TimeZone.getDefault().id,
    val language: String = "en",
    val expertiseLevel: ExpertiseLevel = ExpertiseLevel.BEGINNER
) : Parcelable {

    fun getPersonalizationFactors(): Map<String, Any> {
        return mapOf(
            "age_group" to getAgeGroup(),
            "sleep_goals" to sleepGoals,
            "challenges" to sleepChallenges,
            "expertise" to expertiseLevel.name,
            "language" to language,
            "timezone" to timezone
        )
    }

    private fun getAgeGroup(): String {
        return when (age) {
            null -> "unknown"
            in 18..25 -> "young_adult"
            in 26..35 -> "adult"
            in 36..50 -> "middle_aged"
            in 51..65 -> "older_adult"
            else -> "senior"
        }
    }
}

/**
 * Lifestyle context for personalized insights
 */
@Parcelize
data class LifestyleContext(
    val workSchedule: WorkSchedule? = null,
    val exerciseHabits: ExerciseHabits? = null,
    val dietaryHabits: DietaryHabits? = null,
    val stressLevel: Int = 5, // 1-10 scale
    val socialContext: SocialContext? = null,
    val environmentalFactors: List<String> = emptyList()
) : Parcelable

/**
 * Environment context for insights
 */
@Parcelize
data class EnvironmentContext(
    val location: LocationContext? = null,
    val weather: WeatherContext? = null,
    val airQuality: AirQualityContext? = null,
    val noiseLevel: Float = 0f,
    val lightLevel: Float = 0f,
    val temperature: Float = 0f,
    val humidity: Float = 0f
) : Parcelable

/**
 * Health context for medical insights
 */
@Parcelize
data class HealthContext(
    val sleepDisorders: List<String> = emptyList(),
    val medications: List<MedicationInfo> = emptyList(),
    val symptoms: List<String> = emptyList(),
    val medicalHistory: List<String> = emptyList(),
    val vitals: VitalsContext? = null,
    val lastDoctorVisit: Long? = null
) : Parcelable

// ========== AI PROMPT ENGINEERING MODELS ==========

/**
 * Structured prompt for AI models
 */
@Parcelize
data class StructuredPrompt(
    val systemPrompt: String,
    val userPrompt: String,
    val context: Map<String, Any> = emptyMap(),
    val examples: List<PromptExample> = emptyList(),
    val constraints: List<String> = emptyList(),
    val outputFormat: OutputFormat = OutputFormat.TEXT,
    val maxTokens: Int = 2000,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val responseSchema: ResponseSchema? = null,
    val validationRules: List<ValidationRule> = emptyList()
) : Parcelable {

    fun getTokenEstimate(): Int {
        // Rough estimate: 1 token ≈ 4 characters
        val promptLength = systemPrompt.length + userPrompt.length +
                context.values.joinToString(" ").length
        return (promptLength / 4) + 200 // Add buffer for examples and formatting
    }

    fun isValid(): Boolean {
        return systemPrompt.isNotBlank() &&
                userPrompt.isNotBlank() &&
                maxTokens > 0 &&
                temperature in 0f..2f &&
                topP in 0f..1f
    }
}

/**
 * Example for few-shot learning
 */
@Parcelize
data class PromptExample(
    val input: String,
    val output: String,
    val explanation: String = ""
) : Parcelable

/**
 * Output format specifications
 */
enum class OutputFormat {
    TEXT,
    JSON,
    STRUCTURED,
    MARKDOWN,
    BULLET_POINTS,
    NUMBERED_LIST
}

/**
 * Response schema for structured outputs
 */
@Parcelize
data class ResponseSchema(
    val type: String,
    val properties: Map<String, PropertySchema> = emptyMap(),
    val required: List<String> = emptyList(),
    val examples: List<String> = emptyList()
) : Parcelable {

    companion object {
        val INSIGHTS_ARRAY = ResponseSchema(
            type = "array",
            properties = mapOf(
                "insights" to PropertySchema(
                    type = "array",
                    description = "List of generated insights"
                )
            ),
            required = listOf("insights")
        )

        val SINGLE_INSIGHT = ResponseSchema(
            type = "object",
            properties = mapOf(
                "title" to PropertySchema("string", "Insight title"),
                "description" to PropertySchema("string", "Detailed description"),
                "recommendation" to PropertySchema("string", "Actionable recommendation"),
                "confidence" to PropertySchema("number", "Confidence score 0-1")
            ),
            required = listOf("title", "description", "recommendation")
        )
    }
}

@Parcelize
data class PropertySchema(
    val type: String,
    val description: String = "",
    val enum: List<String> = emptyList(),
    val minimum: Double? = null,
    val maximum: Double? = null
) : Parcelable

/**
 * Validation rules for AI responses
 */
enum class ValidationRule {
    NON_EMPTY,
    VALID_JSON,
    SCHEMA_COMPLIANT,
    APPROPRIATE_LENGTH,
    NO_HARMFUL_CONTENT,
    FACTUALLY_CONSISTENT,
    ACTIONABLE_RECOMMENDATIONS
}

// ========== AI ANALYTICS AND METRICS MODELS ==========

/**
 * Comprehensive AI analytics
 */
@Parcelize
data class AIAnalyticsReport(
    val timeRange: AnalyticsTimeRange,
    val overallMetrics: OverallAIMetrics,
    val modelPerformance: Map<AIModel, ModelPerformanceMetrics>,
    val serviceHealth: Map<String, AIServiceHealth>,
    val insightQuality: InsightQualityMetrics,
    val userEngagement: UserEngagementMetrics,
    val costAnalysis: CostAnalysisMetrics,
    val recommendations: List<AIRecommendation>,
    val trends: List<AITrend>,
    val generatedAt: Long = System.currentTimeMillis()
) : Parcelable {

    fun getBestPerformingModel(): AIModel? {
        return modelPerformance.maxByOrNull { it.value.performanceScore }?.key
    }

    fun getTotalCost(): Double {
        return modelPerformance.values.sumOf { it.totalCost }
    }

    fun getOverallHealthScore(): Float {
        val healthScores = serviceHealth.values.map { it.availabilityPercentage / 100f }
        return if (healthScores.isNotEmpty()) healthScores.average().toFloat() else 0f
    }
}

/**
 * Overall AI system metrics
 */
@Parcelize
data class OverallAIMetrics(
    val totalRequests: Long,
    val successfulRequests: Long,
    val failedRequests: Long,
    val averageLatency: Long,
    val totalCost: Double,
    val averageQuality: Float,
    val userSatisfaction: Float,
    val systemUptime: Float
) : Parcelable {

    val successRate: Float
        get() = if (totalRequests > 0) successfulRequests.toFloat() / totalRequests else 0f

    val failureRate: Float
        get() = if (totalRequests > 0) failedRequests.toFloat() / totalRequests else 0f

    val costPerRequest: Double
        get() = if (totalRequests > 0) totalCost / totalRequests else 0.0

    val efficiency: Float
        get() = (successRate * userSatisfaction * (averageQuality / 10f)) / 3f
}

/**
 * Insight quality metrics
 */
@Parcelize
data class InsightQualityMetrics(
    val totalInsights: Long,
    val averageQuality: Float,
    val averageConfidence: Float,
    val averageRelevance: Float,
    val averageActionability: Float,
    val userFeedbackScore: Float,
    val qualityDistribution: Map<String, Int>,
    val categoryPerformance: Map<String, Float>
) : Parcelable {

    val overallQualityScore: Float
        get() = (averageQuality + averageConfidence + averageRelevance + averageActionability) / 4f

    fun getQualityGrade(): String {
        return when {
            overallQualityScore >= 0.9f -> "A+"
            overallQualityScore >= 0.8f -> "A"
            overallQualityScore >= 0.7f -> "B+"
            overallQualityScore >= 0.6f -> "B"
            overallQualityScore >= 0.5f -> "C"
            else -> "D"
        }
    }
}

/**
 * User engagement with AI insights
 */
@Parcelize
data class UserEngagementMetrics(
    val insightsViewed: Long,
    val insightsActedUpon: Long,
    val averageTimeSpent: Long,
    val feedbackSubmitted: Long,
    val positiveFeedbackRate: Float,
    val insightDismissalRate: Float,
    val repeatEngagementRate: Float
) : Parcelable {

    val actionRate: Float
        get() = if (insightsViewed > 0) insightsActedUpon.toFloat() / insightsViewed else 0f

    val engagementScore: Float
        get() = (actionRate + positiveFeedbackRate + (1f - insightDismissalRate)) / 3f
}

/**
 * Cost analysis for AI operations
 */
@Parcelize
data class CostAnalysisMetrics(
    val totalCost: Double,
    val costByModel: Map<AIModel, Double>,
    val costByService: Map<String, Double>,
    val costByInsightType: Map<InsightGenerationType, Double>,
    val averageCostPerInsight: Double,
    val costTrend: Float, // Percentage change
    val budgetUtilization: Float, // Percentage of budget used
    val costEfficiency: Float // Quality per dollar
) : Parcelable {

    val mostExpensiveModel: AIModel?
        get() = costByModel.maxByOrNull { it.value }?.key

    val mostCostEffectiveModel: AIModel?
        get() = costByModel.minByOrNull { it.value }?.key

    fun isOverBudget(budget: Double): Boolean = totalCost > budget

    fun getProjectedMonthlyCost(): Double = totalCost * 30 // Assuming daily data
}

// ========== SUPPORTING ENUMS AND DATA CLASSES ==========

enum class GenerationStatus {
    PENDING,
    IN_PROGRESS,
    COMPLETED,
    FAILED,
    CANCELLED,
    TIMEOUT,
    ERROR
}

enum class InsightPriority(val value: Int) {
    LOW(3),
    NORMAL(2),
    HIGH(1),
    URGENT(0)
}

enum class InsightType {
    INFORMATIONAL,
    ACTIONABLE,
    WARNING,
    ACHIEVEMENT,
    RECOMMENDATION,
    PREDICTION,
    COMPARISON,
    TREND
}

enum class PersonalizationLevel {
    NONE,
    BASIC,
    ADAPTIVE,
    ADVANCED
}

enum class RequestPriority {
    LOW,
    NORMAL,
    HIGH,
    URGENT
}

enum class ExpertiseLevel {
    BEGINNER,
    INTERMEDIATE,
    ADVANCED,
    EXPERT
}

enum class AnalyticsTimeRange {
    LAST_HOUR,
    LAST_24_HOURS,
    LAST_7_DAYS,
    LAST_30_DAYS,
    LAST_90_DAYS,
    ALL_TIME
}

/**
 * Model selection preferences
 */
@Parcelize
data class ModelSelectionPreferences(
    val preferredService: String? = null,
    val maxLatency: Long = 5000L,
    val maxCost: Double = 0.01,
    val minReliability: Float = 0.9f,
    val requireCapabilities: Set<AICapability> = emptySet(),
    val avoidModels: Set<AIModel> = emptySet(),
    val fallbackStrategy: FallbackStrategy = FallbackStrategy.BEST_AVAILABLE
) : Parcelable

enum class FallbackStrategy {
    BEST_AVAILABLE,
    CHEAPEST,
    FASTEST,
    MOST_RELIABLE,
    NO_FALLBACK
}

/**
 * Output preferences for AI generation
 */
@Parcelize
data class OutputPreferences(
    val maxInsights: Int = 10,
    val minConfidence: Float = 0.6f,
    val preferredCategories: Set<InsightCategory> = emptySet(),
    val excludeCategories: Set<InsightCategory> = emptySet(),
    val outputFormat: OutputFormat = OutputFormat.STRUCTURED,
    val includeEvidence: Boolean = true,
    val includeDataPoints: Boolean = true,
    val language: String = "en",
    val tonality: Tonality = Tonality.SUPPORTIVE
) : Parcelable

enum class Tonality {
    PROFESSIONAL,
    FRIENDLY,
    SUPPORTIVE,
    DIRECT,
    ENCOURAGING
}

/**
 * AI recommendation for system optimization
 */
@Parcelize
data class AIRecommendation(
    val type: RecommendationType,
    val title: String,
    val description: String,
    val impact: ImpactLevel,
    val effort: EffortLevel,
    val priority: Int,
    val implementation: String,
    val expectedOutcome: String,
    val timeframe: String
) : Parcelable

enum class RecommendationType {
    MODEL_OPTIMIZATION,
    COST_REDUCTION,
    QUALITY_IMPROVEMENT,
    PERFORMANCE_ENHANCEMENT,
    USER_EXPERIENCE,
    SYSTEM_RELIABILITY
}

enum class ImpactLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class EffortLevel {
    MINIMAL,
    LOW,
    MEDIUM,
    HIGH,
    EXTENSIVE
}

/**
 * AI trend analysis
 */
@Parcelize
data class AITrend(
    val metric: String,
    val direction: TrendDirection,
    val magnitude: Float,
    val confidence: Float,
    val description: String,
    val prediction: String
) : Parcelable

enum class TrendDirection {
    IMPROVING,
    STABLE,
    DECLINING,
    VOLATILE,
    UNKNOWN
}

// ========== ADDITIONAL SUPPORTING CLASSES ==========

// Placeholder data classes for complex contexts
@Parcelize
data class SleepPatterns(
    val bedtimePattern: String = "",
    val durationPattern: String = "",
    val qualityPattern: String = ""
) : Parcelable

@Parcelize
data class UserPreferences(
    val language: String = "en",
    val timezone: String = TimeZone.getDefault().id,
    val units: String = "metric"
) : Parcelable

@Parcelize
data class WorkSchedule(
    val type: String = "regular",
    val startTime: String = "09:00",
    val endTime: String = "17:00"
) : Parcelable

@Parcelize
data class ExerciseHabits(
    val frequency: Int = 3,
    val intensity: String = "moderate",
    val preferredTime: String = "evening"
) : Parcelable

@Parcelize
data class DietaryHabits(
    val lastMealTime: String = "19:00",
    val caffeineIntake: String = "moderate",
    val alcoholConsumption: String = "occasional"
) : Parcelable

@Parcelize
data class SocialContext(
    val livingArrangement: String = "alone",
    val socialSupport: String = "good",
    val workStress: String = "moderate"
) : Parcelable

@Parcelize
data class LocationContext(
    val latitude: Double = 0.0,
    val longitude: Double = 0.0,
    val city: String = "",
    val timezone: String = TimeZone.getDefault().id
) : Parcelable

@Parcelize
data class WeatherContext(
    val temperature: Float = 0f,
    val humidity: Float = 0f,
    val pressure: Float = 0f,
    val conditions: String = ""
) : Parcelable

@Parcelize
data class AirQualityContext(
    val aqi: Int = 0,
    val pm25: Float = 0f,
    val pm10: Float = 0f,
    val quality: String = ""
) : Parcelable

@Parcelize
data class MedicationInfo(
    val name: String,
    val dosage: String,
    val timing: String,
    val effects: List<String> = emptyList()
) : Parcelable

@Parcelize
data class VitalsContext(
    val heartRate: Int = 0,
    val bloodPressure: String = "",
    val temperature: Float = 0f,
    val weight: Float = 0f
) : Parcelable

@Parcelize
data class InsightPreferences(
    val maxInsights: Int = 10,
    val categories: Set<InsightCategory> = emptySet(),
    val minPriority: InsightPriority = InsightPriority.LOW,
    val includePredictive: Boolean = true
) : Parcelable

@Parcelize
data class ServicePerformanceMetrics(
    val averageLatency: Long = 0L,
    val throughput: Float = 0f,
    val errorRate: Float = 0f,
    val uptime: Float = 100f
) : Parcelable

// Extension functions for better usability
fun AIModel.isRecommendedFor(generationType: InsightGenerationType): Boolean {
    return when (generationType) {
        InsightGenerationType.POST_SESSION -> this in listOf(GPT3_5_TURBO, GEMINI_PRO)
        InsightGenerationType.PERSONALIZED_ANALYSIS -> this in listOf(GPT4, CLAUDE_3_OPUS)
        InsightGenerationType.PREDICTIVE_ANALYSIS -> this in listOf(GPT4, CLAUDE_3_OPUS)
        else -> true
    }
}

fun GeneratedInsight.hasStrongEvidence(): Boolean {
    return evidence.size >= 3 && confidence >= 0.8f
}