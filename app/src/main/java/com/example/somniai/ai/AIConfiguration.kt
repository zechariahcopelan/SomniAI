package com.example.somniai.ai

import com.example.somniai.ai.AIConstants.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ConcurrentHashMap

/**
 * AI Configuration Management for SomniAI
 *
 * Centralized configuration for all AI models, API endpoints, and model-specific settings.
 * Supports dynamic configuration switching, fallback mechanisms, and performance optimization.
 */
object AIConfiguration {

    // ========== CONFIGURATION STATE ==========

    private val _currentConfiguration = MutableStateFlow(getDefaultConfiguration())
    val currentConfiguration: StateFlow<AIConfig> = _currentConfiguration.asStateFlow()

    private val modelConfigurations = ConcurrentHashMap<AIModel, ModelConfig>()
    private val apiEndpoints = ConcurrentHashMap<AIProvider, EndpointConfig>()
    private var fallbackChain: List<AIModel> = listOf()

    // ========== INITIALIZATION ==========

    init {
        initializeDefaultConfigurations()
        setupFallbackChain()
    }

    private fun initializeDefaultConfigurations() {
        // OpenAI Models
        modelConfigurations[AIModel.GPT3_5_TURBO] = ModelConfig(
            model = AIModel.GPT3_5_TURBO,
            provider = AIProvider.OPENAI,
            maxTokens = OpenAI.GPT35_MAX_TOKENS,
            temperature = OpenAI.TEMPERATURE,
            topP = OpenAI.TOP_P,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.0015f,
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS,
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.SLEEP_ANALYSIS
            )
        )

        modelConfigurations[AIModel.GPT4] = ModelConfig(
            model = AIModel.GPT4,
            provider = AIProvider.OPENAI,
            maxTokens = OpenAI.GPT4_MAX_TOKENS,
            temperature = OpenAI.TEMPERATURE,
            topP = OpenAI.TOP_P,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.03f,
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS * 2, // GPT-4 is slower
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.ADVANCED_REASONING,
                ModelCapability.COMPLEX_ANALYSIS,
                ModelCapability.SLEEP_ANALYSIS
            )
        )

        // Anthropic Models
        modelConfigurations[AIModel.CLAUDE_3_OPUS] = ModelConfig(
            model = AIModel.CLAUDE_3_OPUS,
            provider = AIProvider.ANTHROPIC,
            maxTokens = Anthropic.CLAUDE_MAX_TOKENS,
            temperature = Anthropic.TEMPERATURE,
            topP = 1.0f,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.015f,
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS,
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.ADVANCED_REASONING,
                ModelCapability.SLEEP_ANALYSIS,
                ModelCapability.LONG_CONTEXT
            )
        )

        modelConfigurations[AIModel.CLAUDE_3_SONNET] = ModelConfig(
            model = AIModel.CLAUDE_3_SONNET,
            provider = AIProvider.ANTHROPIC,
            maxTokens = Anthropic.CLAUDE_MAX_TOKENS,
            temperature = Anthropic.TEMPERATURE,
            topP = 1.0f,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.003f,
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS,
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.SLEEP_ANALYSIS
            )
        )

        // Google Models
        modelConfigurations[AIModel.GEMINI_PRO] = ModelConfig(
            model = AIModel.GEMINI_PRO,
            provider = AIProvider.GOOGLE,
            maxTokens = Google.GEMINI_MAX_TOKENS,
            temperature = Google.TEMPERATURE,
            topP = 1.0f,
            topK = Google.TOP_K,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.00025f,
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS,
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.SLEEP_ANALYSIS
            )
        )

        modelConfigurations[AIModel.GEMINI_PRO_VISION] = ModelConfig(
            model = AIModel.GEMINI_PRO_VISION,
            provider = AIProvider.GOOGLE,
            maxTokens = Google.GEMINI_MAX_TOKENS,
            temperature = Google.TEMPERATURE,
            topP = 1.0f,
            topK = Google.TOP_K,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.00025f,
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS,
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.REASONING,
                ModelCapability.VISION,
                ModelCapability.SLEEP_ANALYSIS
            )
        )

        // Local/Custom Model
        modelConfigurations[AIModel.LOCAL_MODEL] = ModelConfig(
            model = AIModel.LOCAL_MODEL,
            provider = AIProvider.LOCAL,
            maxTokens = 4096,
            temperature = 0.7f,
            topP = 0.9f,
            frequencyPenalty = 0.0f,
            presencePenalty = 0.0f,
            costPerToken = 0.0f, // No cost for local models
            requestTimeoutMs = AI_REQUEST_TIMEOUT_MS / 2, // Should be faster locally
            maxRetries = MAX_RETRY_ATTEMPTS,
            isEnabled = true,
            capabilities = setOf(
                ModelCapability.TEXT_GENERATION,
                ModelCapability.BASIC_ANALYSIS
            )
        )

        // Initialize API endpoints
        initializeApiEndpoints()
    }

    private fun initializeApiEndpoints() {
        apiEndpoints[AIProvider.OPENAI] = EndpointConfig(
            provider = AIProvider.OPENAI,
            baseUrl = "https://api.openai.com/v1",
            chatCompletionEndpoint = "/chat/completions",
            embeddingsEndpoint = "/embeddings",
            modelsEndpoint = "/models",
            apiKeyHeader = "Authorization",
            apiKeyPrefix = "Bearer ",
            requestHeaders = mapOf(
                "Content-Type" to "application/json",
                "User-Agent" to "SomniAI/1.0"
            ),
            rateLimitRequests = 3500, // Per minute
            rateLimitTokens = 90000, // Per minute
            requiresApiKey = true
        )

        apiEndpoints[AIProvider.ANTHROPIC] = EndpointConfig(
            provider = AIProvider.ANTHROPIC,
            baseUrl = "https://api.anthropic.com/v1",
            chatCompletionEndpoint = "/messages",
            embeddingsEndpoint = null, // Anthropic doesn't provide embeddings
            modelsEndpoint = "/models",
            apiKeyHeader = "x-api-key",
            apiKeyPrefix = "",
            requestHeaders = mapOf(
                "Content-Type" to "application/json",
                "anthropic-version" to "2023-06-01",
                "User-Agent" to "SomniAI/1.0"
            ),
            rateLimitRequests = 1000, // Per minute
            rateLimitTokens = 25000, // Per minute
            requiresApiKey = true
        )

        apiEndpoints[AIProvider.GOOGLE] = EndpointConfig(
            provider = AIProvider.GOOGLE,
            baseUrl = "https://generativelanguage.googleapis.com/v1beta",
            chatCompletionEndpoint = "/models/gemini-pro:generateContent",
            embeddingsEndpoint = "/models/embedding-001:embedContent",
            modelsEndpoint = "/models",
            apiKeyHeader = "x-goog-api-key",
            apiKeyPrefix = "",
            requestHeaders = mapOf(
                "Content-Type" to "application/json",
                "User-Agent" to "SomniAI/1.0"
            ),
            rateLimitRequests = 1000, // Per minute
            rateLimitTokens = 32000, // Per minute
            requiresApiKey = true
        )

        apiEndpoints[AIProvider.LOCAL] = EndpointConfig(
            provider = AIProvider.LOCAL,
            baseUrl = "http://localhost:8080/v1",
            chatCompletionEndpoint = "/chat/completions",
            embeddingsEndpoint = "/embeddings",
            modelsEndpoint = "/models",
            apiKeyHeader = null,
            apiKeyPrefix = null,
            requestHeaders = mapOf(
                "Content-Type" to "application/json",
                "User-Agent" to "SomniAI/1.0"
            ),
            rateLimitRequests = 1000, // Per minute
            rateLimitTokens = 10000, // Per minute
            requiresApiKey = false
        )
    }

    private fun setupFallbackChain() {
        fallbackChain = listOf(
            AIModel.GPT3_5_TURBO,
            AIModel.CLAUDE_3_SONNET,
            AIModel.GEMINI_PRO,
            AIModel.LOCAL_MODEL
        )
    }

    // ========== CONFIGURATION MANAGEMENT ==========

    /**
     * Get current AI configuration
     */
    fun getCurrentConfig(): AIConfig = _currentConfiguration.value

    /**
     * Update current configuration
     */
    fun updateConfiguration(config: AIConfig) {
        _currentConfiguration.value = config
    }

    /**
     * Get configuration for specific model
     */
    fun getModelConfig(model: AIModel): ModelConfig? {
        return modelConfigurations[model]
    }

    /**
     * Update configuration for specific model
     */
    fun updateModelConfig(model: AIModel, config: ModelConfig) {
        modelConfigurations[model] = config

        // Update current configuration if this is the active model
        val current = getCurrentConfig()
        if (current.primaryModel == model) {
            updateConfiguration(current.copy(primaryModelConfig = config))
        }
    }

    /**
     * Get API endpoint configuration for provider
     */
    fun getEndpointConfig(provider: AIProvider): EndpointConfig? {
        return apiEndpoints[provider]
    }

    /**
     * Set primary AI model
     */
    fun setPrimaryModel(model: AIModel) {
        val modelConfig = getModelConfig(model)
        val endpointConfig = getEndpointConfig(modelConfig?.provider ?: AIProvider.OPENAI)

        if (modelConfig != null && endpointConfig != null) {
            val newConfig = getCurrentConfig().copy(
                primaryModel = model,
                primaryModelConfig = modelConfig,
                primaryEndpointConfig = endpointConfig
            )
            updateConfiguration(newConfig)
        }
    }

    /**
     * Get next fallback model in chain
     */
    fun getNextFallbackModel(currentModel: AIModel): AIModel? {
        val currentIndex = fallbackChain.indexOf(currentModel)
        return if (currentIndex >= 0 && currentIndex < fallbackChain.size - 1) {
            fallbackChain[currentIndex + 1]
        } else null
    }

    /**
     * Get all available models with their capabilities
     */
    fun getAvailableModels(): Map<AIModel, ModelConfig> {
        return modelConfigurations.filter { it.value.isEnabled }
    }

    /**
     * Get models with specific capability
     */
    fun getModelsWithCapability(capability: ModelCapability): List<AIModel> {
        return modelConfigurations.filter {
            it.value.isEnabled && capability in it.value.capabilities
        }.keys.toList()
    }

    // ========== CONFIGURATION PROFILES ==========

    /**
     * Create optimized configuration for sleep analysis
     */
    fun createSleepAnalysisProfile(): AIConfig {
        return AIConfig(
            profileName = "Sleep Analysis Optimized",
            primaryModel = AIModel.GPT3_5_TURBO,
            primaryModelConfig = getModelConfig(AIModel.GPT3_5_TURBO)!!.copy(
                temperature = 0.3f, // Lower temperature for more consistent analysis
                maxTokens = 1000 // Shorter responses for faster processing
            ),
            primaryEndpointConfig = getEndpointConfig(AIProvider.OPENAI)!!,
            fallbackModels = listOf(AIModel.CLAUDE_3_SONNET, AIModel.GEMINI_PRO),
            enableFallback = true,
            maxProcessingTimeMs = ACCEPTABLE_PROCESSING_THRESHOLD_MS,
            confidenceThreshold = HIGH_CONFIDENCE_THRESHOLD,
            enableCaching = true,
            enableBatching = false, // Real-time analysis
            customSettings = mapOf(
                "focus" to "sleep_quality",
                "analysis_depth" to "standard",
                "include_recommendations" to "true"
            )
        )
    }

    /**
     * Create configuration optimized for detailed reporting
     */
    fun createDetailedReportProfile(): AIConfig {
        return AIConfig(
            profileName = "Detailed Report Generation",
            primaryModel = AIModel.GPT4,
            primaryModelConfig = getModelConfig(AIModel.GPT4)!!.copy(
                temperature = 0.4f,
                maxTokens = 4000 // Longer responses for detailed reports
            ),
            primaryEndpointConfig = getEndpointConfig(AIProvider.OPENAI)!!,
            fallbackModels = listOf(AIModel.CLAUDE_3_OPUS, AIModel.GPT3_5_TURBO),
            enableFallback = true,
            maxProcessingTimeMs = MAX_PROCESSING_TIME_MS,
            confidenceThreshold = MEDIUM_CONFIDENCE_THRESHOLD,
            enableCaching = true,
            enableBatching = true,
            customSettings = mapOf(
                "focus" to "comprehensive_analysis",
                "analysis_depth" to "detailed",
                "include_visualizations" to "true"
            )
        )
    }

    /**
     * Create configuration for fast, real-time insights
     */
    fun createRealTimeProfile(): AIConfig {
        return AIConfig(
            profileName = "Real-Time Insights",
            primaryModel = AIModel.GEMINI_PRO,
            primaryModelConfig = getModelConfig(AIModel.GEMINI_PRO)!!.copy(
                temperature = 0.2f,
                maxTokens = 500 // Very short responses
            ),
            primaryEndpointConfig = getEndpointConfig(AIProvider.GOOGLE)!!,
            fallbackModels = listOf(AIModel.GPT3_5_TURBO, AIModel.LOCAL_MODEL),
            enableFallback = true,
            maxProcessingTimeMs = FAST_PROCESSING_THRESHOLD_MS,
            confidenceThreshold = LOW_CONFIDENCE_THRESHOLD,
            enableCaching = true,
            enableBatching = false,
            customSettings = mapOf(
                "focus" to "immediate_insights",
                "analysis_depth" to "quick",
                "prioritize_speed" to "true"
            )
        )
    }

    // ========== UTILITY METHODS ==========

    /**
     * Validate current configuration
     */
    fun validateConfiguration(): ConfigurationValidationResult {
        val config = getCurrentConfig()
        val issues = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        // Check if primary model is available
        val modelConfig = getModelConfig(config.primaryModel)
        if (modelConfig == null || !modelConfig.isEnabled) {
            issues.add("Primary model ${config.primaryModel} is not available or disabled")
        }

        // Check if endpoint configuration exists
        val endpointConfig = getEndpointConfig(config.primaryModelConfig.provider)
        if (endpointConfig == null) {
            issues.add("Endpoint configuration missing for provider ${config.primaryModelConfig.provider}")
        }

        // Check API key requirements
        if (endpointConfig?.requiresApiKey == true && !hasValidApiKey(config.primaryModelConfig.provider)) {
            issues.add("API key required but not configured for ${config.primaryModelConfig.provider}")
        }

        // Performance warnings
        if (config.maxProcessingTimeMs > SLOW_PROCESSING_THRESHOLD_MS) {
            warnings.add("Processing timeout is set quite high, may impact user experience")
        }

        if (config.primaryModelConfig.maxTokens > 2000 && config.profileName.contains("Real-Time")) {
            warnings.add("High token limit may slow down real-time responses")
        }

        return ConfigurationValidationResult(
            isValid = issues.isEmpty(),
            issues = issues,
            warnings = warnings
        )
    }

    /**
     * Get default configuration
     */
    fun getDefaultConfiguration(): AIConfig {
        val primaryModel = AIModel.GPT3_5_TURBO
        return AIConfig(
            profileName = "Default",
            primaryModel = primaryModel,
            primaryModelConfig = getModelConfig(primaryModel) ?: ModelConfig.default(),
            primaryEndpointConfig = getEndpointConfig(AIProvider.OPENAI) ?: EndpointConfig.default(),
            fallbackModels = fallbackChain,
            enableFallback = true,
            maxProcessingTimeMs = ACCEPTABLE_PROCESSING_THRESHOLD_MS,
            confidenceThreshold = MEDIUM_CONFIDENCE_THRESHOLD,
            enableCaching = true,
            enableBatching = false
        )
    }

    /**
     * Reset to default configuration
     */
    fun resetToDefault() {
        updateConfiguration(getDefaultConfiguration())
    }

    /**
     * Check if API key is configured for provider
     */
    private fun hasValidApiKey(provider: AIProvider): Boolean {
        // In a real implementation, this would check secure storage
        // For now, we'll assume keys are configured
        return when (provider) {
            AIProvider.LOCAL -> true // No key needed
            else -> true // Assume configured (would check actual storage)
        }
    }

    /**
     * Export current configuration for backup
     */
    fun exportConfiguration(): String {
        // In a real implementation, this would serialize the configuration
        // excluding sensitive information like API keys
        return getCurrentConfig().toString()
    }
}

// ========== CONFIGURATION DATA CLASSES ==========

data class AIConfig(
    val profileName: String,
    val primaryModel: AIModel,
    val primaryModelConfig: ModelConfig,
    val primaryEndpointConfig: EndpointConfig,
    val fallbackModels: List<AIModel> = emptyList(),
    val enableFallback: Boolean = true,
    val maxProcessingTimeMs: Long = ACCEPTABLE_PROCESSING_THRESHOLD_MS,
    val confidenceThreshold: Float = MEDIUM_CONFIDENCE_THRESHOLD,
    val enableCaching: Boolean = true,
    val enableBatching: Boolean = false,
    val customSettings: Map<String, String> = emptyMap()
)

data class ModelConfig(
    val model: AIModel,
    val provider: AIProvider,
    val maxTokens: Int,
    val temperature: Float,
    val topP: Float,
    val topK: Int? = null,
    val frequencyPenalty: Float = 0.0f,
    val presencePenalty: Float = 0.0f,
    val costPerToken: Float,
    val requestTimeoutMs: Long,
    val maxRetries: Int,
    val isEnabled: Boolean,
    val capabilities: Set<ModelCapability>
) {
    companion object {
        fun default() = ModelConfig(
            model = AIModel.GPT3_5_TURBO,
            provider = AIProvider.OPENAI,
            maxTokens = 1000,
            temperature = 0.7f,
            topP = 0.9f,
            costPerToken = 0.0015f,
            requestTimeoutMs = 15000L,
            maxRetries = 3,
            isEnabled = true,
            capabilities = setOf(ModelCapability.TEXT_GENERATION)
        )
    }
}

data class EndpointConfig(
    val provider: AIProvider,
    val baseUrl: String,
    val chatCompletionEndpoint: String,
    val embeddingsEndpoint: String?,
    val modelsEndpoint: String,
    val apiKeyHeader: String?,
    val apiKeyPrefix: String?,
    val requestHeaders: Map<String, String>,
    val rateLimitRequests: Int,
    val rateLimitTokens: Int,
    val requiresApiKey: Boolean
) {
    companion object {
        fun default() = EndpointConfig(
            provider = AIProvider.OPENAI,
            baseUrl = "https://api.openai.com/v1",
            chatCompletionEndpoint = "/chat/completions",
            embeddingsEndpoint = "/embeddings",
            modelsEndpoint = "/models",
            apiKeyHeader = "Authorization",
            apiKeyPrefix = "Bearer ",
            requestHeaders = mapOf("Content-Type" to "application/json"),
            rateLimitRequests = 3500,
            rateLimitTokens = 90000,
            requiresApiKey = true
        )
    }
}

data class ConfigurationValidationResult(
    val isValid: Boolean,
    val issues: List<String>,
    val warnings: List<String>
)

// ========== ENUMS ==========

enum class AIModel(val displayName: String, val provider: String) {
    GPT3_5_TURBO("GPT-3.5 Turbo", "OpenAI"),
    GPT4("GPT-4", "OpenAI"),
    CLAUDE_3_OPUS("Claude 3 Opus", "Anthropic"),
    CLAUDE_3_SONNET("Claude 3 Sonnet", "Anthropic"),
    GEMINI_PRO("Gemini Pro", "Google"),
    GEMINI_PRO_VISION("Gemini Pro Vision", "Google"),
    LOCAL_MODEL("Local Model", "Local"),
    CUSTOM("Custom Model", "Custom")
}

enum class AIProvider {
    OPENAI,
    ANTHROPIC,
    GOOGLE,
    LOCAL,
    CUSTOM
}

enum class ModelCapability {
    TEXT_GENERATION,
    REASONING,
    ADVANCED_REASONING,
    COMPLEX_ANALYSIS,
    VISION,
    SLEEP_ANALYSIS,
    LONG_CONTEXT,
    BASIC_ANALYSIS,
    REAL_TIME_PROCESSING
}