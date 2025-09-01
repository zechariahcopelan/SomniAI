package com.example.somniai.ai

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import kotlinx.serialization.Serializable
import java.util.concurrent.TimeUnit

/**
 * AI Model capabilities for different sleep analysis tasks
 */
object AIModelCapabilities {
    const val TEXT_GENERATION = "text_generation"
    const val ANALYSIS = "analysis"
    const val INSIGHTS = "insights"
    const val PATTERN_RECOGNITION = "pattern_recognition"
    const val CONVERSATION = "conversation"
    const val CLASSIFICATION = "classification"
    const val SUMMARIZATION = "summarization"
    const val RECOMMENDATION = "recommendation"
    const val PREDICTION = "prediction"
    const val REASONING = "reasoning"
    const val TREND_ANALYSIS = "trend_analysis"
    const val BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    const val HEALTH_ASSESSMENT = "health_assessment"
}

/**
 * Communication styles for AI-generated insights
 */
@Serializable
enum class CommunicationStyle {
    PROFESSIONAL,    // Clinical and formal
    CONVERSATIONAL,  // Friendly and casual
    EDUCATIONAL,     // Teaching and informative
    CLINICAL,        // Medical and precise
    MOTIVATIONAL,    // Encouraging and positive
    EMPATHETIC,      // Understanding and supportive
    SCIENTIFIC,      // Data-driven and analytical
    SIMPLE,          // Easy to understand
    SUPPORTIVE       // Caring and helpful (matches your existing code)
}

/**
 * AI Model reliability and performance tiers
 */
@Serializable
enum class ModelTier {
    PREMIUM,         // Highest accuracy, slower, most expensive
    STANDARD,        // Balanced performance and cost
    FAST,           // Quick responses, lower accuracy
    EXPERIMENTAL    // New models, unpredictable performance
}

/**
 * Expertise levels for insight generation
 */
@Serializable
enum class ExpertiseLevel {
    BEGINNER,       // Simple explanations
    GENERAL,        // Standard user level
    INTERMEDIATE,   // Some sleep knowledge
    ADVANCED,       // Health-conscious users
    EXPERT         // Medical/research level
}

/**
 * Cultural contexts for personalization
 */
@Serializable
enum class CulturalContext {
    WESTERN,        // US/EU cultural norms
    EASTERN,        // Asian cultural perspectives
    LATIN,          // Latin American context
    MIDDLE_EASTERN, // Middle Eastern considerations
    AFRICAN,        // African cultural context
    UNIVERSAL       // Cross-cultural approach
}

/**
 * Response formats for different use cases
 */
@Serializable
enum class ResponseFormat {
    INSIGHTS,       // Structured insight objects
    CONVERSATION,   // Conversational responses
    SUMMARY,        // Brief summaries
    DETAILED,       // Comprehensive analysis
    BULLET_POINTS,  // Quick actionable items
    NARRATIVE       // Story-like format
}

/**
 * Personalization levels for AI responses
 */
@Serializable
enum class PersonalizationLevel {
    NONE,           // Generic responses
    BASIC,          // Simple user data integration
    ADAPTIVE,       // Learning from user behavior
    COMPREHENSIVE,  // Full personalization
    PREDICTIVE      // Anticipatory responses
}

/**
 * Analysis depth for insight generation
 */
@Serializable
enum class AnalysisDepth {
    SURFACE,        // Basic analysis
    STANDARD,       // Normal depth
    COMPREHENSIVE,  // Deep analysis
    EXHAUSTIVE      // Maximum analysis
}

/**
 * Comprehensive AI Model definitions with enterprise-grade configurations
 */
@Serializable
enum class AIModel(
    val serviceName: String,
    val modelId: String,
    val displayName: String,
    val capabilities: List<String>,
    val contextWindow: Int,
    val maxTokens: Int,
    val costPerToken: Double, // USD per token
    val averageLatency: Long, // milliseconds
    val reliability: Float, // 0.0 to 1.0
    val tier: ModelTier,
    val supportedStyles: List<CommunicationStyle>,
    val isMultimodal: Boolean = false,
    val maxRetries: Int = 3,
    val timeout: Long = TimeUnit.SECONDS.toMillis(30),
    val isDeprecated: Boolean = false,
    val apiVersion: String = "v1",
    val rateLimit: Int = 20, // requests per minute
    val priority: Int = 1 // 1 = highest priority
) {
    // ========== OPENAI MODELS ==========
    GPT4(
        serviceName = "openai",
        modelId = "gpt-4",
        displayName = "GPT-4",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.PATTERN_RECOGNITION,
            AIModelCapabilities.REASONING,
            AIModelCapabilities.RECOMMENDATION,
            AIModelCapabilities.HEALTH_ASSESSMENT
        ),
        contextWindow = 8192,
        maxTokens = 4096,
        costPerToken = 0.00003,
        averageLatency = 3000,
        reliability = 0.95f,
        tier = ModelTier.PREMIUM,
        supportedStyles = CommunicationStyle.values().toList(),
        timeout = TimeUnit.SECONDS.toMillis(45),
        priority = 1
    ),

    GPT4_TURBO(
        serviceName = "openai",
        modelId = "gpt-4-turbo-preview",
        displayName = "GPT-4 Turbo",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.PATTERN_RECOGNITION,
            AIModelCapabilities.REASONING,
            AIModelCapabilities.RECOMMENDATION,
            AIModelCapabilities.TREND_ANALYSIS
        ),
        contextWindow = 128000,
        maxTokens = 4096,
        costPerToken = 0.00001,
        averageLatency = 2000,
        reliability = 0.93f,
        tier = ModelTier.PREMIUM,
        supportedStyles = CommunicationStyle.values().toList(),
        timeout = TimeUnit.SECONDS.toMillis(40),
        priority = 2
    ),

    GPT3_5_TURBO(
        serviceName = "openai",
        modelId = "gpt-3.5-turbo",
        displayName = "GPT-3.5 Turbo",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.RECOMMENDATION,
            AIModelCapabilities.SUMMARIZATION
        ),
        contextWindow = 16384,
        maxTokens = 4096,
        costPerToken = 0.000001,
        averageLatency = 1500,
        reliability = 0.88f,
        tier = ModelTier.FAST,
        supportedStyles = listOf(
            CommunicationStyle.CONVERSATIONAL,
            CommunicationStyle.SIMPLE,
            CommunicationStyle.SUPPORTIVE,
            CommunicationStyle.EDUCATIONAL
        ),
        timeout = TimeUnit.SECONDS.toMillis(30),
        rateLimit = 60,
        priority = 4
    ),

    // ========== ANTHROPIC MODELS ==========
    CLAUDE_3_OPUS(
        serviceName = "anthropic",
        modelId = "claude-3-opus-20240229",
        displayName = "Claude 3 Opus",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.PATTERN_RECOGNITION,
            AIModelCapabilities.REASONING,
            AIModelCapabilities.HEALTH_ASSESSMENT,
            AIModelCapabilities.BEHAVIORAL_ANALYSIS
        ),
        contextWindow = 200000,
        maxTokens = 4096,
        costPerToken = 0.000015,
        averageLatency = 2800,
        reliability = 0.96f,
        tier = ModelTier.PREMIUM,
        supportedStyles = listOf(
            CommunicationStyle.PROFESSIONAL,
            CommunicationStyle.CLINICAL,
            CommunicationStyle.SCIENTIFIC,
            CommunicationStyle.EMPATHETIC,
            CommunicationStyle.EDUCATIONAL
        ),
        isMultimodal = true,
        timeout = TimeUnit.SECONDS.toMillis(50),
        apiVersion = "2023-06-01",
        priority = 1
    ),

    CLAUDE_3_SONNET(
        serviceName = "anthropic",
        modelId = "claude-3-sonnet-20240229",
        displayName = "Claude 3 Sonnet",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.RECOMMENDATION,
            AIModelCapabilities.PATTERN_RECOGNITION,
            AIModelCapabilities.HEALTH_ASSESSMENT
        ),
        contextWindow = 200000,
        maxTokens = 4096,
        costPerToken = 0.000003,
        averageLatency = 2000,
        reliability = 0.92f,
        tier = ModelTier.STANDARD,
        supportedStyles = listOf(
            CommunicationStyle.CONVERSATIONAL,
            CommunicationStyle.SUPPORTIVE,
            CommunicationStyle.EDUCATIONAL,
            CommunicationStyle.EMPATHETIC,
            CommunicationStyle.PROFESSIONAL
        ),
        isMultimodal = true,
        timeout = TimeUnit.SECONDS.toMillis(40),
        apiVersion = "2023-06-01",
        priority = 3
    ),

    CLAUDE_3_HAIKU(
        serviceName = "anthropic",
        modelId = "claude-3-haiku-20240307",
        displayName = "Claude 3 Haiku",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.RECOMMENDATION,
            AIModelCapabilities.SUMMARIZATION
        ),
        contextWindow = 200000,
        maxTokens = 4096,
        costPerToken = 0.00000025,
        averageLatency = 1000,
        reliability = 0.85f,
        tier = ModelTier.FAST,
        supportedStyles = listOf(
            CommunicationStyle.SIMPLE,
            CommunicationStyle.CONVERSATIONAL,
            CommunicationStyle.SUPPORTIVE,
            CommunicationStyle.MOTIVATIONAL
        ),
        isMultimodal = true,
        timeout = TimeUnit.SECONDS.toMillis(25),
        apiVersion = "2023-06-01",
        rateLimit = 100,
        priority = 5
    ),

    // ========== LEGACY CLAUDE ==========
    CLAUDE(
        serviceName = "anthropic",
        modelId = "claude-2.1",
        displayName = "Claude 2.1",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.RECOMMENDATION
        ),
        contextWindow = 200000,
        maxTokens = 4096,
        costPerToken = 0.000008,
        averageLatency = 3500,
        reliability = 0.87f,
        tier = ModelTier.STANDARD,
        supportedStyles = listOf(
            CommunicationStyle.PROFESSIONAL,
            CommunicationStyle.EDUCATIONAL,
            CommunicationStyle.CLINICAL
        ),
        timeout = TimeUnit.SECONDS.toMillis(45),
        apiVersion = "2023-06-01",
        isDeprecated = true,
        priority = 8
    ),

    // ========== GOOGLE MODELS ==========
    GEMINI_PRO(
        serviceName = "google",
        modelId = "gemini-pro",
        displayName = "Gemini Pro",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.PATTERN_RECOGNITION,
            AIModelCapabilities.REASONING,
            AIModelCapabilities.TREND_ANALYSIS
        ),
        contextWindow = 32768,
        maxTokens = 8192,
        costPerToken = 0.0000005,
        averageLatency = 1800,
        reliability = 0.89f,
        tier = ModelTier.STANDARD,
        supportedStyles = listOf(
            CommunicationStyle.SCIENTIFIC,
            CommunicationStyle.PROFESSIONAL,
            CommunicationStyle.EDUCATIONAL,
            CommunicationStyle.CONVERSATIONAL
        ),
        timeout = TimeUnit.SECONDS.toMillis(35),
        apiVersion = "v1beta",
        priority = 6
    ),

    GEMINI_PRO_VISION(
        serviceName = "google",
        modelId = "gemini-pro-vision",
        displayName = "Gemini Pro Vision",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.CLASSIFICATION,
            AIModelCapabilities.PATTERN_RECOGNITION
        ),
        contextWindow = 16384,
        maxTokens = 4096,
        costPerToken = 0.00000025,
        averageLatency = 2200,
        reliability = 0.83f,
        tier = ModelTier.EXPERIMENTAL,
        supportedStyles = listOf(
            CommunicationStyle.SCIENTIFIC,
            CommunicationStyle.SIMPLE,
            CommunicationStyle.EDUCATIONAL
        ),
        isMultimodal = true,
        timeout = TimeUnit.SECONDS.toMillis(40),
        apiVersion = "v1beta",
        priority = 7
    ),

    // ========== FALLBACK MODELS ==========
    GEMINI(
        serviceName = "google",
        modelId = "gemini-pro",
        displayName = "Gemini",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.RECOMMENDATION
        ),
        contextWindow = 32768,
        maxTokens = 8192,
        costPerToken = 0.0000005,
        averageLatency = 1800,
        reliability = 0.89f,
        tier = ModelTier.STANDARD,
        supportedStyles = listOf(
            CommunicationStyle.CONVERSATIONAL,
            CommunicationStyle.SIMPLE
        ),
        timeout = TimeUnit.SECONDS.toMillis(35),
        apiVersion = "v1beta",
        priority = 9
    ),

    GPT3_5(
        serviceName = "openai",
        modelId = "gpt-3.5-turbo",
        displayName = "GPT-3.5",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.RECOMMENDATION
        ),
        contextWindow = 16384,
        maxTokens = 4096,
        costPerToken = 0.000001,
        averageLatency = 1500,
        reliability = 0.88f,
        tier = ModelTier.FAST,
        supportedStyles = listOf(
            CommunicationStyle.SIMPLE,
            CommunicationStyle.CONVERSATIONAL
        ),
        timeout = TimeUnit.SECONDS.toMillis(30),
        rateLimit = 60,
        priority = 10
    );

    // ========== COMPUTED PROPERTIES ==========

    /**
     * Check if model supports a specific capability
     */
    fun hasCapability(capability: String): Boolean {
        return capabilities.contains(capability)
    }

    /**
     * Check if model supports a communication style
     */
    fun supportsStyle(style: CommunicationStyle): Boolean {
        return supportedStyles.contains(style)
    }

    /**
     * Get estimated cost for a given number of tokens
     */
    fun calculateCost(tokens: Int): Double {
        return tokens * costPerToken
    }

    /**
     * Get quality score based on reliability and tier
     */
    fun getQualityScore(): Float {
        return when (tier) {
            ModelTier.PREMIUM -> reliability
            ModelTier.STANDARD -> reliability * 0.9f
            ModelTier.FAST -> reliability * 0.8f
            ModelTier.EXPERIMENTAL -> reliability * 0.7f
        }
    }

    /**
     * Check if model is suitable for sleep analysis tasks
     */
    val isHealthCapable: Boolean
        get() = hasCapability(AIModelCapabilities.HEALTH_ASSESSMENT) ||
                hasCapability(AIModelCapabilities.BEHAVIORAL_ANALYSIS) ||
                hasCapability(AIModelCapabilities.PATTERN_RECOGNITION)

    /**
     * Check if model is fast enough for real-time insights
     */
    val isRealtimeCapable: Boolean
        get() = averageLatency < 2000L && tier != ModelTier.PREMIUM

    /**
     * Get priority group for fallback selection
     */
    val priorityGroup: Int
        get() = when (priority) {
            in 1..2 -> 1 // Primary models
            in 3..5 -> 2 // Secondary models
            in 6..8 -> 3 // Tertiary models
            else -> 4    // Fallback models
        }

    companion object {
        /**
         * Get all available models, excluding deprecated ones
         */
        fun getActiveModels(): List<AIModel> {
            return values().filter { !it.isDeprecated }
        }

        /**
         * Get models by service provider
         */
        fun getModelsByService(serviceName: String): List<AIModel> {
            return values().filter { it.serviceName == serviceName && !it.isDeprecated }
        }

        /**
         * Get models by capability
         */
        fun getModelsByCapability(capability: String): List<AIModel> {
            return values().filter { it.hasCapability(capability) && !it.isDeprecated }
        }

        /**
         * Get models by tier
         */
        fun getModelsByTier(tier: ModelTier): List<AIModel> {
            return values().filter { it.tier == tier && !it.isDeprecated }
        }

        /**
         * Get fastest models for real-time use
         */
        fun getFastModels(): List<AIModel> {
            return values()
                .filter { it.isRealtimeCapable && !it.isDeprecated }
                .sortedBy { it.averageLatency }
        }

        /**
         * Get most reliable models
         */
        fun getReliableModels(): List<AIModel> {
            return values()
                .filter { it.reliability >= 0.9f && !it.isDeprecated }
                .sortedByDescending { it.reliability }
        }

        /**
         * Get models suitable for health/sleep analysis
         */
        fun getHealthCapableModels(): List<AIModel> {
            return values()
                .filter { it.isHealthCapable && !it.isDeprecated }
                .sortedBy { it.priority }
        }

        /**
         * Get primary models for production use
         */
        fun getPrimaryModels(): List<AIModel> {
            return values()
                .filter { it.priorityGroup == 1 && !it.isDeprecated }
                .sortedBy { it.priority }
        }

        /**
         * Get fallback models for when primary models fail
         */
        fun getFallbackModels(): List<AIModel> {
            return values()
                .filter { it.priorityGroup >= 3 && !it.isDeprecated }
                .sortedBy { it.priority }
        }

        /**
         * Select optimal model for a specific task
         */
        fun selectOptimalModel(
            requiredCapability: String,
            preferredStyle: CommunicationStyle? = null,
            maxLatency: Long = 5000L,
            minReliability: Float = 0.8f
        ): AIModel? {
            return values()
                .filter { model ->
                    !model.isDeprecated &&
                            model.hasCapability(requiredCapability) &&
                            model.averageLatency <= maxLatency &&
                            model.reliability >= minReliability &&
                            (preferredStyle == null || model.supportsStyle(preferredStyle))
                }
                .minByOrNull { it.priority }
        }
    }
}

/**
 * Configuration for AI model usage in sleep analysis
 */
@Parcelize
@Serializable
data class AIModelConfiguration(
    val model: AIModel,
    val enabled: Boolean = true,
    val priority: Int = model.priority,
    val weight: Float = 1.0f,
    val customTimeout: Long? = null,
    val customRateLimit: Int? = null,
    val fallbackModels: List<AIModel> = emptyList(),
    val specializedFor: List<String> = emptyList(), // Specific use cases
    val restrictions: List<String> = emptyList() // Usage restrictions
) : Parcelable {

    /**
     * Get effective timeout considering custom override
     */
    val effectiveTimeout: Long
        get() = customTimeout ?: model.timeout

    /**
     * Get effective rate limit considering custom override
     */
    val effectiveRateLimit: Int
        get() = customRateLimit ?: model.rateLimit

    /**
     * Check if model is specialized for a specific use case
     */
    fun isSpecializedFor(useCase: String): Boolean {
        return specializedFor.contains(useCase)
    }

    /**
     * Check if model has any restrictions
     */
    fun hasRestrictions(): Boolean {
        return restrictions.isNotEmpty()
    }

    companion object {
        /**
         * Create default configuration for a model
         */
        fun forModel(model: AIModel): AIModelConfiguration {
            return AIModelConfiguration(
                model = model,
                enabled = !model.isDeprecated,
                priority = model.priority,
                fallbackModels = when (model.priorityGroup) {
                    1 -> AIModel.getPrimaryModels().drop(1)
                    else -> AIModel.getFallbackModels()
                }
            )
        }

        /**
         * Create configuration optimized for sleep insights
         */
        fun forSleepInsights(): List<AIModelConfiguration> {
            return AIModel.getHealthCapableModels().map { model ->
                AIModelConfiguration(
                    model = model,
                    enabled = true,
                    priority = model.priority,
                    specializedFor = listOf("sleep_analysis", "health_insights"),
                    fallbackModels = AIModel.getFallbackModels()
                )
            }
        }

        /**
         * Create configuration optimized for real-time analysis
         */
        fun forRealtimeAnalysis(): List<AIModelConfiguration> {
            return AIModel.getFastModels().map { model ->
                AIModelConfiguration(
                    model = model,
                    enabled = true,
                    priority = model.priority,
                    specializedFor = listOf("real_time", "quick_insights"),
                    customTimeout = TimeUnit.SECONDS.toMillis(15),
                    fallbackModels = listOf(AIModel.GPT3_5_TURBO, AIModel.CLAUDE_3_HAIKU)
                )
            }
        }
    }
}

/**
 * Model performance tracking for optimization
 */
@Serializable
data class ModelPerformanceMetrics(
    val modelId: String,
    val totalRequests: Long = 0,
    val successfulRequests: Long = 0,
    val failedRequests: Long = 0,
    val averageLatency: Long = 0,
    val averageQuality: Float = 0f,
    val averageCost: Double = 0.0,
    val lastUsed: Long = System.currentTimeMillis(),
    val circuitBreakerTripped: Boolean = false,
    val lastError: String? = null
) {
    /**
     * Calculate success rate
     */
    val successRate: Float
        get() = if (totalRequests > 0) successfulRequests.toFloat() / totalRequests else 0f

    /**
     * Check if performance is acceptable
     */
    val isPerformanceAcceptable: Boolean
        get() = successRate >= 0.8f && averageQuality >= 0.7f

    /**
     * Get performance score for model selection
     */
    val performanceScore: Float
        get() = (successRate * 0.4f) + (averageQuality * 0.4f) +
                ((5000f - averageLatency.coerceAtMost(5000)) / 5000f * 0.2f)
}

/**
 * Constants for AI model management
 */
object AIModelConstants {
    const val DEFAULT_TIMEOUT_MS = 30000L
    const val DEFAULT_RATE_LIMIT = 20
    const val MIN_RELIABILITY_THRESHOLD = 0.7f
    const val MAX_LATENCY_REALTIME_MS = 2000L
    const val CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    const val PERFORMANCE_MONITORING_WINDOW_HOURS = 24L

    // Model selection weights
    const val WEIGHT_RELIABILITY = 0.4f
    const val WEIGHT_LATENCY = 0.3f
    const val WEIGHT_COST = 0.2f
    const val WEIGHT_CAPABILITY = 0.1f
}