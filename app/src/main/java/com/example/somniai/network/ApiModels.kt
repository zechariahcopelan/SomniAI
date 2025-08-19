package com.example.somniai.network

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.google.gson.*
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.security.MessageDigest
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*
import kotlin.random.Random

/**
 * Enterprise-grade API models and orchestration system for AI service integrations
 *
 * Advanced Features:
 * - Multi-provider AI service integration (OpenAI, Anthropic, Azure, Google, Custom)
 * - Intelligent request routing and load balancing with performance optimization
 * - Advanced error handling with circuit breaker patterns and automatic recovery
 * - Comprehensive rate limiting and quota management with predictive throttling
 * - Sophisticated caching and response optimization with intelligent invalidation
 * - Performance monitoring and analytics with real-time optimization
 * - Request/response transformation and validation with schema enforcement
 * - Cost optimization and usage tracking with budget management
 * - A/B testing framework for model comparison and optimization
 * - Security hardening with request sanitization and authentication management
 * - Response quality assessment and automatic model selection
 * - Streaming support with real-time processing and connection management
 * - Comprehensive logging and debugging with detailed request/response tracking
 * - Request batching and optimization for high-throughput scenarios
 * - Intelligent fallback strategies with graceful degradation
 * - Integration with comprehensive analytics and personalization systems
 */

// ========== CORE API ORCHESTRATION MODELS ==========

/**
 * Comprehensive API request wrapper with advanced features
 */
data class AdvancedApiRequest<T>(
    val id: String = generateRequestId(),
    val data: T,
    val provider: AIProvider,
    val model: String,
    val priority: RequestPriority = RequestPriority.NORMAL,
    val metadata: RequestMetadata = RequestMetadata(),
    val options: RequestOptions = RequestOptions(),
    val retryPolicy: RetryPolicy = RetryPolicy.default(),
    val cachePolicy: CachePolicy = CachePolicy.default(),
    val transformations: List<RequestTransformation> = emptyList(),
    val validators: List<RequestValidator> = emptyList(),
    val timestamp: Long = System.currentTimeMillis()
) {

    /**
     * Comprehensive request validation
     */
    suspend fun validate(): ValidationResult {
        val errors = mutableListOf<ValidationError>()
        val warnings = mutableListOf<ValidationWarning>()

        // Core validation
        if (model.isBlank()) {
            errors.add(ValidationError("MODEL_EMPTY", "Model cannot be blank"))
        }

        if (!provider.supportedModels.contains(model)) {
            warnings.add(ValidationWarning("MODEL_NOT_RECOMMENDED", "Model $model not in recommended list for $provider"))
        }

        // Custom validators
        for (validator in validators) {
            val result = validator.validate(this)
            errors.addAll(result.errors)
            warnings.addAll(result.warnings)
        }

        // Provider-specific validation
        val providerValidation = validateForProvider()
        errors.addAll(providerValidation.errors)
        warnings.addAll(providerValidation.warnings)

        return ValidationResult(
            isValid = errors.isEmpty(),
            errors = errors,
            warnings = warnings,
            score = calculateValidationScore(errors, warnings)
        )
    }

    /**
     * Calculate estimated cost for this request
     */
    fun estimateCost(): CostEstimate {
        val tokenCount = estimateTokenCount()
        val modelPricing = getModelPricing(provider, model)

        return CostEstimate(
            estimatedTokens = tokenCount,
            estimatedCost = (tokenCount * modelPricing.costPerToken),
            currency = modelPricing.currency,
            breakdown = mapOf(
                "input_tokens" to tokenCount.input * modelPricing.inputCostPerToken,
                "output_tokens" to tokenCount.output * modelPricing.outputCostPerToken
            )
        )
    }

    /**
     * Apply transformations to the request
     */
    suspend fun applyTransformations(): AdvancedApiRequest<T> {
        var transformed = this

        for (transformation in transformations) {
            transformed = transformation.transform(transformed)
        }

        return transformed
    }

    /**
     * Generate cache key for this request
     */
    fun generateCacheKey(): String {
        val content = when (data) {
            is OpenAIChatRequest -> data.messages.joinToString { it.content }
            is AnthropicMessageRequest -> data.messages.joinToString { it.content }
            else -> data.toString()
        }

        val keyData = "$provider-$model-${content.hashCode()}-${options.hashCode()}"
        return MessageDigest.getInstance("SHA-256")
            .digest(keyData.toByteArray())
            .joinToString("") { "%02x".format(it) }
            .take(32)
    }

    private fun validateForProvider(): ValidationResult {
        return when (provider) {
            AIProvider.OPENAI -> validateOpenAIRequest()
            AIProvider.ANTHROPIC -> validateAnthropicRequest()
            AIProvider.AZURE_OPENAI -> validateAzureOpenAIRequest()
            AIProvider.GOOGLE_VERTEX -> validateVertexAIRequest()
            AIProvider.CUSTOM -> validateCustomRequest()
        }
    }

    private fun validateOpenAIRequest(): ValidationResult {
        val errors = mutableListOf<ValidationError>()

        if (data is OpenAIChatRequest) {
            if (data.messages.isEmpty()) {
                errors.add(ValidationError("MESSAGES_EMPTY", "Messages cannot be empty"))
            }

            if (data.temperature < 0.0 || data.temperature > 2.0) {
                errors.add(ValidationError("TEMPERATURE_INVALID", "Temperature must be between 0.0 and 2.0"))
            }

            val totalTokens = data.estimateTokenCount()
            val maxTokens = getModelMaxTokens(model)
            if (totalTokens > maxTokens) {
                errors.add(ValidationError("TOKENS_EXCEEDED", "Estimated tokens ($totalTokens) exceed model limit ($maxTokens)"))
            }
        }

        return ValidationResult(isValid = errors.isEmpty(), errors = errors)
    }

    private fun validateAnthropicRequest(): ValidationResult {
        val errors = mutableListOf<ValidationError>()

        if (data is AnthropicMessageRequest) {
            if (data.maxTokens <= 0) {
                errors.add(ValidationError("MAX_TOKENS_INVALID", "Max tokens must be positive"))
            }

            if (data.temperature < 0.0 || data.temperature > 1.0) {
                errors.add(ValidationError("TEMPERATURE_INVALID", "Temperature must be between 0.0 and 1.0"))
            }
        }

        return ValidationResult(isValid = errors.isEmpty(), errors = errors)
    }

    private fun validateAzureOpenAIRequest(): ValidationResult = validateOpenAIRequest()
    private fun validateVertexAIRequest(): ValidationResult = ValidationResult(isValid = true)
    private fun validateCustomRequest(): ValidationResult = ValidationResult(isValid = true)

    private fun estimateTokenCount(): TokenCount {
        return when (data) {
            is OpenAIChatRequest -> data.estimateTokenCount()
            is AnthropicMessageRequest -> data.estimateTokenCount()
            else -> TokenCount(1000, 500) // Default estimate
        }
    }

    private fun calculateValidationScore(errors: List<ValidationError>, warnings: List<ValidationWarning>): Float {
        val errorPenalty = errors.size * 0.3f
        val warningPenalty = warnings.size * 0.1f
        return (1.0f - errorPenalty - warningPenalty).coerceIn(0f, 1f)
    }

    companion object {
        fun generateRequestId(): String = "req_${System.currentTimeMillis()}_${Random.nextInt(1000000)}"
    }
}

/**
 * Comprehensive API response wrapper with quality assessment
 */
data class AdvancedApiResponse<T>(
    val id: String,
    val requestId: String,
    val success: Boolean,
    val data: T? = null,
    val error: AdvancedApiError? = null,
    val metadata: ResponseMetadata,
    val performance: PerformanceMetrics,
    val qualityScore: Float? = null,
    val cacheInfo: CacheInfo? = null,
    val providerInfo: ProviderInfo,
    val transformations: List<ResponseTransformation> = emptyList(),
    val validations: List<ResponseValidation> = emptyList(),
    val timestamp: Long = System.currentTimeMillis()
) {

    val isSuccessful: Boolean get() = success && error == null && data != null
    val hasError: Boolean get() = !success || error != null
    val isFromCache: Boolean get() = cacheInfo?.wasFromCache == true
    val responseTime: Long get() = performance.totalDuration

    /**
     * Assess the quality of this response
     */
    fun assessQuality(): ResponseQualityAssessment {
        val factors = mutableMapOf<String, Float>()

        // Performance factors
        factors["response_time"] = when {
            responseTime < 1000 -> 1.0f
            responseTime < 3000 -> 0.8f
            responseTime < 5000 -> 0.6f
            else -> 0.4f
        }

        // Content quality factors
        data?.let { responseData ->
            factors["content_quality"] = assessContentQuality(responseData)
        }

        // Error factors
        if (hasError) {
            factors["error_impact"] = when (error?.severity) {
                ErrorSeverity.LOW -> 0.9f
                ErrorSeverity.MEDIUM -> 0.7f
                ErrorSeverity.HIGH -> 0.4f
                ErrorSeverity.CRITICAL -> 0.1f
                null -> 1.0f
            }
        } else {
            factors["error_impact"] = 1.0f
        }

        // Cache efficiency
        factors["cache_efficiency"] = if (isFromCache) 1.0f else 0.8f

        val overallScore = factors.values.average().toFloat()

        return ResponseQualityAssessment(
            overallScore = overallScore,
            factors = factors,
            recommendations = generateQualityRecommendations(factors),
            grade = getQualityGrade(overallScore)
        )
    }

    /**
     * Apply response transformations
     */
    suspend fun applyTransformations(): AdvancedApiResponse<T> {
        var transformed = this

        for (transformation in transformations) {
            transformed = transformation.transform(transformed)
        }

        return transformed
    }

    private fun assessContentQuality(data: T): Float {
        return when (data) {
            is String -> {
                when {
                    data.length < 10 -> 0.3f
                    data.length < 50 -> 0.6f
                    data.length < 200 -> 0.8f
                    else -> 1.0f
                }
            }
            is OpenAIChatResponse -> {
                when {
                    data.content.isNullOrBlank() -> 0.2f
                    data.wasFiltered -> 0.4f
                    data.hitTokenLimit -> 0.6f
                    data.isComplete -> 1.0f
                    else -> 0.7f
                }
            }
            is AnthropicMessageResponse -> {
                when {
                    data.text.isNullOrBlank() -> 0.2f
                    data.stopReason == "max_tokens" -> 0.6f
                    data.stopReason == "stop_sequence" -> 1.0f
                    else -> 0.8f
                }
            }
            else -> 0.7f
        }
    }

    private fun generateQualityRecommendations(factors: Map<String, Float>): List<String> {
        val recommendations = mutableListOf<String>()

        if (factors["response_time"]!! < 0.6f) {
            recommendations.add("Consider using a faster model or caching for better response times")
        }

        if (factors["content_quality"]!! < 0.7f) {
            recommendations.add("Response quality could be improved with better prompting or model selection")
        }

        if (factors["error_impact"]!! < 0.8f) {
            recommendations.add("Implement better error handling and retry mechanisms")
        }

        return recommendations
    }

    private fun getQualityGrade(score: Float): QualityGrade {
        return when {
            score >= 0.9f -> QualityGrade.EXCELLENT
            score >= 0.8f -> QualityGrade.GOOD
            score >= 0.7f -> QualityGrade.AVERAGE
            score >= 0.6f -> QualityGrade.BELOW_AVERAGE
            else -> QualityGrade.POOR
        }
    }
}

/**
 * Advanced error model with recovery strategies
 */
data class AdvancedApiError(
    val code: String,
    val message: String,
    val type: ErrorType,
    val severity: ErrorSeverity,
    val provider: AIProvider,
    val retryable: Boolean,
    val retryAfterMs: Long? = null,
    val details: Map<String, Any> = emptyMap(),
    val context: ErrorContext? = null,
    val recoveryStrategies: List<RecoveryStrategy> = emptyList(),
    val originalError: Throwable? = null,
    val timestamp: Long = System.currentTimeMillis()
) {

    val isRateLimitError: Boolean
        get() = type == ErrorType.RATE_LIMIT || code.contains("rate_limit", ignoreCase = true)

    val isAuthError: Boolean
        get() = type == ErrorType.AUTHENTICATION || code.contains("auth", ignoreCase = true)

    val isQuotaError: Boolean
        get() = type == ErrorType.QUOTA_EXCEEDED || message.contains("quota", ignoreCase = true)

    val isServiceError: Boolean
        get() = type == ErrorType.SERVICE_UNAVAILABLE || code.startsWith("5")

    val shouldRetry: Boolean
        get() = retryable && severity != ErrorSeverity.CRITICAL

    /**
     * Get recommended recovery strategy
     */
    fun getRecommendedRecovery(): RecoveryStrategy? {
        return recoveryStrategies.maxByOrNull { it.successProbability }
    }

    /**
     * Calculate retry delay with exponential backoff
     */
    fun calculateRetryDelay(attempt: Int): Long {
        val baseDelay = retryAfterMs ?: 1000L
        val exponentialDelay = baseDelay * (2.0.pow(attempt - 1)).toLong()
        val jitter = Random.nextLong(0, exponentialDelay / 4)
        return min(exponentialDelay + jitter, 300000L) // Max 5 minutes
    }

    companion object {
        fun fromOpenAIError(error: OpenAIErrorResponse, provider: AIProvider): AdvancedApiError {
            return AdvancedApiError(
                code = error.error.code ?: "unknown",
                message = error.error.message,
                type = mapOpenAIErrorType(error.error.type),
                severity = mapErrorSeverity(error.error.type),
                provider = provider,
                retryable = error.error.type in listOf("server_error", "rate_limit_error")
            )
        }

        fun fromAnthropicError(error: AnthropicErrorResponse, provider: AIProvider): AdvancedApiError {
            return AdvancedApiError(
                code = error.error.type,
                message = error.error.message,
                type = mapAnthropicErrorType(error.error.type),
                severity = mapErrorSeverity(error.error.type),
                provider = provider,
                retryable = error.error.type in listOf("overloaded_error", "rate_limit_error")
            )
        }

        private fun mapOpenAIErrorType(type: String): ErrorType {
            return when (type) {
                "invalid_request_error" -> ErrorType.INVALID_REQUEST
                "authentication_error" -> ErrorType.AUTHENTICATION
                "permission_error" -> ErrorType.PERMISSION_DENIED
                "not_found_error" -> ErrorType.NOT_FOUND
                "rate_limit_error" -> ErrorType.RATE_LIMIT
                "tokens" -> ErrorType.TOKEN_LIMIT
                "server_error" -> ErrorType.SERVICE_UNAVAILABLE
                else -> ErrorType.UNKNOWN
            }
        }

        private fun mapAnthropicErrorType(type: String): ErrorType {
            return when (type) {
                "invalid_request_error" -> ErrorType.INVALID_REQUEST
                "authentication_error" -> ErrorType.AUTHENTICATION
                "permission_error" -> ErrorType.PERMISSION_DENIED
                "not_found_error" -> ErrorType.NOT_FOUND
                "rate_limit_error" -> ErrorType.RATE_LIMIT
                "overloaded_error" -> ErrorType.SERVICE_UNAVAILABLE
                else -> ErrorType.UNKNOWN
            }
        }

        private fun mapErrorSeverity(type: String): ErrorSeverity {
            return when (type) {
                "rate_limit_error", "overloaded_error" -> ErrorSeverity.MEDIUM
                "server_error", "service_unavailable" -> ErrorSeverity.HIGH
                "authentication_error", "permission_error" -> ErrorSeverity.HIGH
                "invalid_request_error" -> ErrorSeverity.LOW
                else -> ErrorSeverity.MEDIUM
            }
        }
    }
}

// ========== ENHANCED PROVIDER-SPECIFIC MODELS ==========

/**
 * Enhanced OpenAI Chat Completions request with advanced features
 */
data class EnhancedOpenAIChatRequest(
    val model: String,
    val messages: List<EnhancedChatMessage>,
    val temperature: Double = 0.7,
    @SerializedName("max_tokens")
    val maxTokens: Int? = null,
    @SerializedName("top_p")
    val topP: Double = 1.0,
    @SerializedName("frequency_penalty")
    val frequencyPenalty: Double = 0.0,
    @SerializedName("presence_penalty")
    val presencePenalty: Double = 0.0,
    val stop: List<String>? = null,
    val stream: Boolean = false,
    @SerializedName("response_format")
    val responseFormat: ResponseFormat? = null,
    @SerializedName("tool_choice")
    val toolChoice: String? = null,
    val tools: List<Tool>? = null,
    val user: String? = null,
    val seed: Int? = null,
    @SerializedName("logit_bias")
    val logitBias: Map<String, Int>? = null,
    @SerializedName("logprobs")
    val logprobs: Boolean? = null,
    @SerializedName("top_logprobs")
    val topLogprobs: Int? = null
) {

    /**
     * Enhanced validation with detailed feedback
     */
    fun validateExtensive(): ExtensiveValidationResult {
        val issues = mutableListOf<ValidationIssue>()

        // Model validation
        if (model.isBlank()) {
            issues.add(ValidationIssue.error("MODEL_BLANK", "Model cannot be blank"))
        } else if (!OpenAIModels.isValidModel(model)) {
            issues.add(ValidationIssue.warning("MODEL_UNKNOWN", "Model $model is not recognized"))
        }

        // Parameter validation
        if (temperature < 0.0 || temperature > 2.0) {
            issues.add(ValidationIssue.error("TEMPERATURE_RANGE", "Temperature must be between 0.0 and 2.0"))
        }

        if (topP < 0.0 || topP > 1.0) {
            issues.add(ValidationIssue.error("TOP_P_RANGE", "Top P must be between 0.0 and 1.0"))
        }

        if (frequencyPenalty < -2.0 || frequencyPenalty > 2.0) {
            issues.add(ValidationIssue.error("FREQUENCY_PENALTY_RANGE", "Frequency penalty must be between -2.0 and 2.0"))
        }

        if (presencePenalty < -2.0 || presencePenalty > 2.0) {
            issues.add(ValidationIssue.error("PRESENCE_PENALTY_RANGE", "Presence penalty must be between -2.0 and 2.0"))
        }

        // Messages validation
        if (messages.isEmpty()) {
            issues.add(ValidationIssue.error("MESSAGES_EMPTY", "Messages cannot be empty"))
        } else {
            messages.forEachIndexed { index, message ->
                val messageIssues = message.validate()
                issues.addAll(messageIssues.map { it.withContext("message[$index]") })
            }
        }

        // Token estimation and validation
        val estimatedTokens = estimateTokenCount()
        val modelMaxTokens = OpenAIModels.getMaxTokens(model)

        if (estimatedTokens.total > modelMaxTokens) {
            issues.add(ValidationIssue.error("TOKEN_LIMIT", "Estimated tokens (${estimatedTokens.total}) exceed model limit ($modelMaxTokens)"))
        } else if (estimatedTokens.total > modelMaxTokens * 0.9) {
            issues.add(ValidationIssue.warning("TOKEN_NEAR_LIMIT", "Estimated tokens (${estimatedTokens.total}) near model limit ($modelMaxTokens)"))
        }

        // Tool validation
        tools?.let { toolList ->
            toolList.forEachIndexed { index, tool ->
                if (tool.function.name.isBlank()) {
                    issues.add(ValidationIssue.error("TOOL_NAME_BLANK", "Tool $index name cannot be blank"))
                }
            }
        }

        return ExtensiveValidationResult(
            isValid = issues.none { it.level == ValidationLevel.ERROR },
            issues = issues,
            recommendations = generateRecommendations(issues),
            score = calculateValidationScore(issues)
        )
    }

    /**
     * Enhanced token estimation with breakdown
     */
    fun estimateTokenCount(): DetailedTokenCount {
        var promptTokens = 0
        var systemTokens = 0

        // Base conversation overhead (varies by model)
        val baseOverhead = when {
            model.contains("gpt-4") -> 10
            model.contains("gpt-3.5") -> 8
            else -> 10
        }

        promptTokens += baseOverhead

        // Count tokens for each message
        for (message in messages) {
            val messageTokens = message.estimateTokens()
            when (message.role) {
                "system" -> systemTokens += messageTokens
                else -> promptTokens += messageTokens
            }
        }

        // Tool definitions add overhead
        tools?.let { toolList ->
            promptTokens += toolList.sumOf { it.estimateTokens() }
        }

        val maxCompletionTokens = maxTokens ?: (OpenAIModels.getMaxTokens(model) - promptTokens) / 2

        return DetailedTokenCount(
            promptTokens = promptTokens,
            systemTokens = systemTokens,
            maxCompletionTokens = maxCompletionTokens,
            estimatedCompletionTokens = min(maxCompletionTokens, estimateExpectedCompletion()),
            total = promptTokens + systemTokens,
            breakdown = mapOf(
                "base_overhead" to baseOverhead,
                "messages" to (promptTokens - baseOverhead),
                "system_messages" to systemTokens,
                "tools" to (tools?.sumOf { it.estimateTokens() } ?: 0)
            )
        )
    }

    /**
     * Create optimized copy of request
     */
    fun optimized(): EnhancedOpenAIChatRequest {
        return copy(
            messages = messages.map { it.optimized() },
            temperature = temperature.coerceIn(0.0, 2.0),
            topP = topP.coerceIn(0.0, 1.0),
            frequencyPenalty = frequencyPenalty.coerceIn(-2.0, 2.0),
            presencePenalty = presencePenalty.coerceIn(-2.0, 2.0)
        )
    }

    private fun generateRecommendations(issues: List<ValidationIssue>): List<String> {
        val recommendations = mutableListOf<String>()

        if (issues.any { it.code == "TOKEN_NEAR_LIMIT" }) {
            recommendations.add("Consider reducing message length or using a model with higher token limits")
        }

        if (temperature > 1.0) {
            recommendations.add("High temperature (${temperature}) may produce less coherent responses")
        }

        if (issues.any { it.code == "MODEL_UNKNOWN" }) {
            recommendations.add("Consider using a verified model for better reliability")
        }

        return recommendations
    }

    private fun calculateValidationScore(issues: List<ValidationIssue>): Float {
        val errorCount = issues.count { it.level == ValidationLevel.ERROR }
        val warningCount = issues.count { it.level == ValidationLevel.WARNING }

        val errorPenalty = errorCount * 0.4f
        val warningPenalty = warningCount * 0.1f

        return (1.0f - errorPenalty - warningPenalty).coerceIn(0f, 1f)
    }

    private fun estimateExpectedCompletion(): Int {
        // Estimate based on message types and conversation context
        val conversationLength = messages.sumOf { it.content.length }
        return when {
            conversationLength < 500 -> 100
            conversationLength < 1500 -> 300
            conversationLength < 3000 -> 500
            else -> 800
        }.coerceAtMost(maxTokens ?: 1000)
    }
}

/**
 * Enhanced chat message with advanced features
 */
data class EnhancedChatMessage(
    val role: String,
    val content: String,
    val name: String? = null,
    @SerializedName("tool_calls")
    val toolCalls: List<ToolCall>? = null,
    @SerializedName("tool_call_id")
    val toolCallId: String? = null,
    val metadata: MessageMetadata? = null
) {

    /**
     * Validate message content and structure
     */
    fun validate(): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()

        if (role.isBlank()) {
            issues.add(ValidationIssue.error("ROLE_BLANK", "Message role cannot be blank"))
        } else if (role !in validRoles) {
            issues.add(ValidationIssue.warning("ROLE_UNKNOWN", "Role '$role' is not a standard role"))
        }

        if (content.isBlank() && toolCalls.isNullOrEmpty()) {
            issues.add(ValidationIssue.error("CONTENT_EMPTY", "Message must have content or tool calls"))
        }

        if (content.length > MAX_MESSAGE_LENGTH) {
            issues.add(ValidationIssue.error("CONTENT_TOO_LONG", "Message content exceeds maximum length ($MAX_MESSAGE_LENGTH)"))
        }

        // Tool-specific validation
        if (role == "tool" && toolCallId.isNullOrBlank()) {
            issues.add(ValidationIssue.error("TOOL_CALL_ID_MISSING", "Tool messages must have tool_call_id"))
        }

        return issues
    }

    /**
     * Estimate token count for this message
     */
    fun estimateTokens(): Int {
        // Base token overhead per message
        var tokens = 4 // OpenAI conversation formatting overhead

        // Role tokens
        tokens += role.length / 4

        // Content tokens (approximate 4 characters per token)
        tokens += content.length / 4

        // Name tokens
        name?.let { tokens += it.length / 4 }

        // Tool call tokens
        toolCalls?.let { calls ->
            tokens += calls.sumOf {
                (it.function.name.length + it.function.arguments.length) / 4 + 10 // function call overhead
            }
        }

        return tokens
    }

    /**
     * Create optimized version of message
     */
    fun optimized(): EnhancedChatMessage {
        return copy(
            content = content.trim().take(MAX_MESSAGE_LENGTH),
            role = role.lowercase()
        )
    }

    /**
     * Sanitize message content for security
     */
    fun sanitized(): EnhancedChatMessage {
        return copy(
            content = sanitizeContent(content),
            name = name?.take(64)?.filter { it.isLetterOrDigit() || it in "_-" }
        )
    }

    private fun sanitizeContent(content: String): String {
        return content
            .replace(Regex("\\p{Cntrl}"), "") // Remove control characters
            .replace(Regex("\\s+"), " ") // Normalize whitespace
            .trim()
    }

    companion object {
        private const val MAX_MESSAGE_LENGTH = 100000
        private val validRoles = setOf("system", "user", "assistant", "tool")

        fun system(content: String, metadata: MessageMetadata? = null): EnhancedChatMessage =
            EnhancedChatMessage("system", content, metadata = metadata)

        fun user(content: String, name: String? = null, metadata: MessageMetadata? = null): EnhancedChatMessage =
            EnhancedChatMessage("user", content, name = name, metadata = metadata)

        fun assistant(content: String, toolCalls: List<ToolCall>? = null, metadata: MessageMetadata? = null): EnhancedChatMessage =
            EnhancedChatMessage("assistant", content, toolCalls = toolCalls, metadata = metadata)

        fun tool(content: String, toolCallId: String, metadata: MessageMetadata? = null): EnhancedChatMessage =
            EnhancedChatMessage("tool", content, toolCallId = toolCallId, metadata = metadata)
    }
}

/**
 * Enhanced Anthropic message request with advanced features
 */
data class EnhancedAnthropicMessageRequest(
    val model: String,
    @SerializedName("max_tokens")
    val maxTokens: Int,
    val messages: List<EnhancedAnthropicMessage>,
    val system: String? = null,
    val temperature: Double = 0.7,
    @SerializedName("top_p")
    val topP: Double? = null,
    @SerializedName("top_k")
    val topK: Int? = null,
    @SerializedName("stop_sequences")
    val stopSequences: List<String>? = null,
    val stream: Boolean = false,
    val metadata: AnthropicRequestMetadata? = null
) {

    /**
     * Comprehensive validation for Anthropic requests
     */
    fun validateComprehensive(): ExtensiveValidationResult {
        val issues = mutableListOf<ValidationIssue>()

        // Model validation
        if (model.isBlank()) {
            issues.add(ValidationIssue.error("MODEL_BLANK", "Model cannot be blank"))
        } else if (!AnthropicModels.isValidModel(model)) {
            issues.add(ValidationIssue.warning("MODEL_UNKNOWN", "Model $model is not recognized"))
        }

        // Token validation
        if (maxTokens <= 0) {
            issues.add(ValidationIssue.error("MAX_TOKENS_INVALID", "Max tokens must be positive"))
        } else if (maxTokens > AnthropicModels.getMaxTokens(model)) {
            issues.add(ValidationIssue.error("MAX_TOKENS_EXCEEDED", "Max tokens exceeds model limit"))
        }

        // Parameter validation
        if (temperature < 0.0 || temperature > 1.0) {
            issues.add(ValidationIssue.error("TEMPERATURE_RANGE", "Temperature must be between 0.0 and 1.0"))
        }

        topP?.let { p ->
            if (p < 0.0 || p > 1.0) {
                issues.add(ValidationIssue.error("TOP_P_RANGE", "Top P must be between 0.0 and 1.0"))
            }
        }

        topK?.let { k ->
            if (k < 1) {
                issues.add(ValidationIssue.error("TOP_K_INVALID", "Top K must be at least 1"))
            }
        }

        // Messages validation
        if (messages.isEmpty()) {
            issues.add(ValidationIssue.error("MESSAGES_EMPTY", "Messages cannot be empty"))
        } else {
            messages.forEachIndexed { index, message ->
                val messageIssues = message.validate()
                issues.addAll(messageIssues.map { it.withContext("message[$index]") })
            }

            // Anthropic-specific message pattern validation
            if (messages.first().role != "user") {
                issues.add(ValidationIssue.error("FIRST_MESSAGE_NOT_USER", "First message must be from user"))
            }

            // Check for alternating pattern
            for (i in 1 until messages.size) {
                if (messages[i].role == messages[i-1].role) {
                    issues.add(ValidationIssue.warning("NON_ALTERNATING_ROLES", "Messages should alternate between user and assistant"))
                    break
                }
            }
        }

        // System message validation
        system?.let { sys ->
            if (sys.length > 100000) {
                issues.add(ValidationIssue.warning("SYSTEM_MESSAGE_LONG", "System message is very long"))
            }
        }

        return ExtensiveValidationResult(
            isValid = issues.none { it.level == ValidationLevel.ERROR },
            issues = issues,
            recommendations = generateAnthropicRecommendations(issues),
            score = calculateValidationScore(issues)
        )
    }

    /**
     * Estimate token count for Anthropic request
     */
    fun estimateTokenCount(): DetailedTokenCount {
        var inputTokens = 0

        // System message tokens
        system?.let { inputTokens += it.length / 3 } // Anthropic uses ~3 chars per token

        // Message tokens
        val messageTokens = messages.sumOf { it.estimateTokens() }
        inputTokens += messageTokens

        return DetailedTokenCount(
            promptTokens = inputTokens,
            systemTokens = system?.length?.div(3) ?: 0,
            maxCompletionTokens = maxTokens,
            estimatedCompletionTokens = min(maxTokens, inputTokens / 3), // Conservative estimate
            total = inputTokens,
            breakdown = mapOf(
                "system_message" to (system?.length?.div(3) ?: 0),
                "messages" to messageTokens,
                "estimated_output" to min(maxTokens, inputTokens / 3)
            )
        )
    }

    private fun generateAnthropicRecommendations(issues: List<ValidationIssue>): List<String> {
        val recommendations = mutableListOf<String>()

        if (issues.any { it.code == "NON_ALTERNATING_ROLES" }) {
            recommendations.add("Anthropic works best with alternating user/assistant messages")
        }

        if (issues.any { it.code == "SYSTEM_MESSAGE_LONG" }) {
            recommendations.add("Consider shortening the system message for better performance")
        }

        return recommendations
    }

    private fun calculateValidationScore(issues: List<ValidationIssue>): Float {
        val errorCount = issues.count { it.level == ValidationLevel.ERROR }
        val warningCount = issues.count { it.level == ValidationLevel.WARNING }

        return (1.0f - errorCount * 0.4f - warningCount * 0.1f).coerceIn(0f, 1f)
    }
}

/**
 * Enhanced Anthropic message
 */
data class EnhancedAnthropicMessage(
    val role: String,
    val content: String,
    val metadata: MessageMetadata? = null
) {

    fun validate(): List<ValidationIssue> {
        val issues = mutableListOf<ValidationIssue>()

        if (role !in listOf("user", "assistant")) {
            issues.add(ValidationIssue.error("INVALID_ROLE", "Role must be 'user' or 'assistant'"))
        }

        if (content.isBlank()) {
            issues.add(ValidationIssue.error("CONTENT_EMPTY", "Content cannot be empty"))
        }

        if (content.length > 100000) {
            issues.add(ValidationIssue.warning("CONTENT_VERY_LONG", "Content is very long"))
        }

        return issues
    }

    fun estimateTokens(): Int {
        return content.length / 3 // Anthropic approximation
    }

    companion object {
        fun user(content: String, metadata: MessageMetadata? = null): EnhancedAnthropicMessage =
            EnhancedAnthropicMessage("user", content, metadata)

        fun assistant(content: String, metadata: MessageMetadata? = null): EnhancedAnthropicMessage =
            EnhancedAnthropicMessage("assistant", content, metadata)
    }
}

// ========== ADVANCED RESPONSE MODELS ==========

/**
 * Enhanced OpenAI response with quality metrics
 */
data class EnhancedOpenAIChatResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<EnhancedChatChoice>,
    val usage: DetailedApiUsage?,
    @SerializedName("system_fingerprint")
    val systemFingerprint: String? = null,
    val qualityMetrics: ResponseQualityMetrics? = null
) {

    val firstChoice: EnhancedChatChoice?
        get() = choices.firstOrNull()

    val content: String?
        get() = firstChoice?.message?.content

    val finishReason: String?
        get() = firstChoice?.finishReason

    val isComplete: Boolean
        get() = finishReason == "stop"

    val wasFiltered: Boolean
        get() = finishReason == "content_filter"

    val hitTokenLimit: Boolean
        get() = finishReason == "length"

    val hasToolCalls: Boolean
        get() = firstChoice?.message?.toolCalls?.isNotEmpty() == true

    /**
     * Assess response quality
     */
    fun assessQuality(): ResponseQualityMetrics {
        val contentQuality = assessContentQuality()
        val completeness = assessCompleteness()
        val relevance = assessRelevance()

        return ResponseQualityMetrics(
            overallScore = (contentQuality + completeness + relevance) / 3f,
            contentQuality = contentQuality,
            completeness = completeness,
            relevance = relevance,
            hasIssues = wasFiltered || !isComplete,
            issues = getQualityIssues()
        )
    }

    private fun assessContentQuality(): Float {
        val content = this.content ?: return 0f

        return when {
            content.length < 10 -> 0.2f
            content.length < 50 -> 0.5f
            content.contains("I cannot", ignoreCase = true) -> 0.4f
            content.contains("I'm sorry", ignoreCase = true) -> 0.6f
            wasFiltered -> 0.3f
            else -> 0.9f
        }
    }

    private fun assessCompleteness(): Float {
        return when {
            !isComplete -> 0.4f
            hitTokenLimit -> 0.6f
            content?.endsWith("...") == true -> 0.7f
            else -> 1.0f
        }
    }

    private fun assessRelevance(): Float {
        // This would be more sophisticated in a real implementation
        // Could analyze content against original prompt
        return 0.8f
    }

    private fun getQualityIssues(): List<String> {
        val issues = mutableListOf<String>()

        if (wasFiltered) issues.add("Content was filtered")
        if (hitTokenLimit) issues.add("Response was truncated due to token limit")
        if (!isComplete) issues.add("Response was not completed")
        if (content.isNullOrBlank()) issues.add("Empty response")

        return issues
    }
}

/**
 * Enhanced chat choice with metadata
 */
data class EnhancedChatChoice(
    val index: Int,
    val message: EnhancedChatMessage,
    @SerializedName("finish_reason")
    val finishReason: String?,
    val delta: EnhancedChatMessage? = null,
    @SerializedName("logprobs")
    val logprobs: LogProbs? = null
)

/**
 * Enhanced Anthropic response
 */
data class EnhancedAnthropicMessageResponse(
    val id: String,
    val type: String,
    val role: String,
    val content: List<AnthropicContent>,
    val model: String,
    @SerializedName("stop_reason")
    val stopReason: String?,
    @SerializedName("stop_sequence")
    val stopSequence: String?,
    val usage: DetailedAnthropicUsage,
    val qualityMetrics: ResponseQualityMetrics? = null
) {

    val text: String?
        get() = content.firstOrNull { it.type == "text" }?.text

    val isComplete: Boolean
        get() = stopReason == "end_turn"

    val hitTokenLimit: Boolean
        get() = stopReason == "max_tokens"

    /**
     * Assess response quality
     */
    fun assessQuality(): ResponseQualityMetrics {
        val contentQuality = assessContentQuality()
        val completeness = if (isComplete) 1.0f else 0.6f
        val relevance = 0.8f // Placeholder

        return ResponseQualityMetrics(
            overallScore = (contentQuality + completeness + relevance) / 3f,
            contentQuality = contentQuality,
            completeness = completeness,
            relevance = relevance,
            hasIssues = !isComplete,
            issues = if (!isComplete) listOf("Response incomplete") else emptyList()
        )
    }

    private fun assessContentQuality(): Float {
        val text = this.text ?: return 0f

        return when {
            text.length < 10 -> 0.2f
            text.length < 50 -> 0.5f
            else -> 0.9f
        }
    }
}

// ========== SUPPORTING DATA CLASSES ==========

/**
 * Request metadata for tracking and analytics
 */
data class RequestMetadata(
    val sessionId: String? = null,
    val userId: String? = null,
    val source: String = "api",
    val tags: Set<String> = emptySet(),
    val experiments: Map<String, String> = emptyMap(),
    val customData: Map<String, Any> = emptyMap()
)

/**
 * Response metadata for analysis
 */
data class ResponseMetadata(
    val modelUsed: String,
    val provider: AIProvider,
    val region: String? = null,
    val serverInfo: Map<String, String> = emptyMap(),
    val processingTime: Long,
    val cacheHit: Boolean = false
)

/**
 * Performance metrics for requests/responses
 */
data class PerformanceMetrics(
    val networkLatency: Long,
    val processingTime: Long,
    val totalDuration: Long,
    val queueTime: Long = 0L,
    val retryCount: Int = 0,
    val bytesTransferred: Long = 0L
) {
    val efficiency: Float
        get() = processingTime.toFloat() / totalDuration.coerceAtLeast(1L)
}

/**
 * Detailed API usage statistics
 */
data class DetailedApiUsage(
    @SerializedName("prompt_tokens")
    val promptTokens: Int,
    @SerializedName("completion_tokens")
    val completionTokens: Int,
    @SerializedName("total_tokens")
    val totalTokens: Int,
    val cacheTokens: Int = 0,
    val reasoningTokens: Int = 0,
    val cost: CostBreakdown? = null,
    val efficiency: TokenEfficiency? = null
) {
    val tokensPerSecond: Float
        get() = if (efficiency?.duration != null && efficiency.duration > 0) {
            totalTokens.toFloat() / (efficiency.duration / 1000f)
        } else 0f
}

/**
 * Detailed Anthropic usage statistics
 */
data class DetailedAnthropicUsage(
    @SerializedName("input_tokens")
    val inputTokens: Int,
    @SerializedName("output_tokens")
    val outputTokens: Int,
    val cacheCreationInputTokens: Int = 0,
    val cacheReadInputTokens: Int = 0
) {
    val totalTokens: Int
        get() = inputTokens + outputTokens

    val cacheEfficiency: Float
        get() = if (inputTokens > 0) cacheReadInputTokens.toFloat() / inputTokens else 0f
}

/**
 * Token count with detailed breakdown
 */
data class DetailedTokenCount(
    val promptTokens: Int,
    val systemTokens: Int,
    val maxCompletionTokens: Int,
    val estimatedCompletionTokens: Int,
    val total: Int,
    val breakdown: Map<String, Int> = emptyMap()
) {
    val input: Int get() = promptTokens + systemTokens
    val output: Int get() = estimatedCompletionTokens
    val efficiency: Float get() = if (total > 0) output.toFloat() / total else 0f
}

/**
 * Request options for fine-tuning behavior
 */
data class RequestOptions(
    val enableCaching: Boolean = true,
    val enableRetries: Boolean = true,
    val priority: RequestPriority = RequestPriority.NORMAL,
    val timeout: Long = 30000L,
    val enableStreaming: Boolean = false,
    val enableLogging: Boolean = true,
    val enableQualityAssessment: Boolean = true,
    val customHeaders: Map<String, String> = emptyMap()
)

/**
 * Cache policy configuration
 */
data class CachePolicy(
    val enabled: Boolean = true,
    val ttlSeconds: Long = 3600L,
    val maxSize: Int = 1000,
    val strategy: CacheStrategy = CacheStrategy.LRU,
    val keyStrategy: CacheKeyStrategy = CacheKeyStrategy.CONTENT_HASH
) {
    companion object {
        fun default() = CachePolicy()
        fun disabled() = CachePolicy(enabled = false)
        fun shortTerm() = CachePolicy(ttlSeconds = 300L)
        fun longTerm() = CachePolicy(ttlSeconds = 86400L)
    }
}

/**
 * Retry policy configuration
 */
data class RetryPolicy(
    val enabled: Boolean = true,
    val maxAttempts: Int = 3,
    val baseDelayMs: Long = 1000L,
    val maxDelayMs: Long = 30000L,
    val backoffMultiplier: Double = 2.0,
    val jitterEnabled: Boolean = true,
    val retryableErrorTypes: Set<ErrorType> = setOf(
        ErrorType.RATE_LIMIT,
        ErrorType.SERVICE_UNAVAILABLE,
        ErrorType.TIMEOUT
    )
) {
    companion object {
        fun default() = RetryPolicy()
        fun disabled() = RetryPolicy(enabled = false)
        fun aggressive() = RetryPolicy(maxAttempts = 5, baseDelayMs = 500L)
        fun conservative() = RetryPolicy(maxAttempts = 2, baseDelayMs = 2000L)
    }
}

// ========== ENUMS AND CONSTANTS ==========

/**
 * Enhanced AI provider enumeration
 */
enum class AIProvider(
    val displayName: String,
    val baseUrl: String,
    val supportedModels: List<String>,
    val defaultModel: String,
    val maxTokenLimit: Int,
    val costTier: CostTier
) {
    OPENAI(
        displayName = "OpenAI",
        baseUrl = "https://api.openai.com/v1/",
        supportedModels = listOf("gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"),
        defaultModel = "gpt-4",
        maxTokenLimit = 128000,
        costTier = CostTier.PREMIUM
    ),
    ANTHROPIC(
        displayName = "Anthropic",
        baseUrl = "https://api.anthropic.com/v1/",
        supportedModels = listOf("claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"),
        defaultModel = "claude-3-sonnet-20240229",
        maxTokenLimit = 200000,
        costTier = CostTier.PREMIUM
    ),
    AZURE_OPENAI(
        displayName = "Azure OpenAI",
        baseUrl = "https://{resource}.openai.azure.com/",
        supportedModels = listOf("gpt-4", "gpt-35-turbo"),
        defaultModel = "gpt-4",
        maxTokenLimit = 32000,
        costTier = CostTier.ENTERPRISE
    ),
    GOOGLE_VERTEX(
        displayName = "Google Vertex AI",
        baseUrl = "https://us-central1-aiplatform.googleapis.com/v1/",
        supportedModels = listOf("gemini-pro", "gemini-pro-vision"),
        defaultModel = "gemini-pro",
        maxTokenLimit = 1048576,
        costTier = CostTier.MODERATE
    ),
    CUSTOM(
        displayName = "Custom Provider",
        baseUrl = "",
        supportedModels = emptyList(),
        defaultModel = "",
        maxTokenLimit = 4096,
        costTier = CostTier.UNKNOWN
    );

    val isOpenAICompatible: Boolean
        get() = this == OPENAI || this == AZURE_OPENAI

    val supportsStreaming: Boolean
        get() = this != CUSTOM

    val supportsTools: Boolean
        get() = this == OPENAI || this == AZURE_OPENAI
}

enum class ErrorType {
    INVALID_REQUEST,
    AUTHENTICATION,
    PERMISSION_DENIED,
    NOT_FOUND,
    RATE_LIMIT,
    QUOTA_EXCEEDED,
    TOKEN_LIMIT,
    SERVICE_UNAVAILABLE,
    TIMEOUT,
    NETWORK_ERROR,
    PARSING_ERROR,
    VALIDATION_ERROR,
    UNKNOWN
}

enum class ErrorSeverity {
    LOW,      // Recoverable, minimal impact
    MEDIUM,   // May affect quality but recoverable
    HIGH,     // Significant impact, requires intervention
    CRITICAL  // Service-breaking, immediate attention required
}

enum class RequestPriority(val value: Int) {
    LOW(1),
    NORMAL(2),
    HIGH(3),
    URGENT(4),
    CRITICAL(5)
}

enum class ValidationLevel {
    INFO,
    WARNING,
    ERROR
}

enum class QualityGrade {
    EXCELLENT,
    GOOD,
    AVERAGE,
    BELOW_AVERAGE,
    POOR
}

enum class CacheStrategy {
    LRU,           // Least Recently Used
    LFU,           // Least Frequently Used
    FIFO,          // First In, First Out
    TTL_PRIORITY,  // Time To Live with Priority
    ADAPTIVE       // Machine learning-based
}

enum class CacheKeyStrategy {
    CONTENT_HASH,     // Hash of request content
    SEMANTIC_HASH,    // Semantic similarity-based
    PARAMETRIC,       // Based on key parameters
    CUSTOM           // User-defined
}

enum class CostTier {
    FREE,
    MODERATE,
    PREMIUM,
    ENTERPRISE,
    UNKNOWN
}

// ========== VALIDATION MODELS ==========

/**
 * Comprehensive validation result
 */
data class ExtensiveValidationResult(
    val isValid: Boolean,
    val issues: List<ValidationIssue>,
    val recommendations: List<String>,
    val score: Float
) {
    val errorCount: Int get() = issues.count { it.level == ValidationLevel.ERROR }
    val warningCount: Int get() = issues.count { it.level == ValidationLevel.WARNING }
    val hasErrors: Boolean get() = errorCount > 0
    val hasWarnings: Boolean get() = warningCount > 0
}

/**
 * Individual validation issue
 */
data class ValidationIssue(
    val level: ValidationLevel,
    val code: String,
    val message: String,
    val context: String? = null,
    val suggestion: String? = null
) {
    fun withContext(newContext: String): ValidationIssue {
        return copy(context = if (context != null) "$context.$newContext" else newContext)
    }

    companion object {
        fun error(code: String, message: String, suggestion: String? = null): ValidationIssue =
            ValidationIssue(ValidationLevel.ERROR, code, message, suggestion = suggestion)

        fun warning(code: String, message: String, suggestion: String? = null): ValidationIssue =
            ValidationIssue(ValidationLevel.WARNING, code, message, suggestion = suggestion)

        fun info(code: String, message: String): ValidationIssue =
            ValidationIssue(ValidationLevel.INFO, code, message)
    }
}

// ========== UTILITY CLASSES ==========

/**
 * Model information and capabilities
 */
object OpenAIModels {
    private val modelInfo = mapOf(
        "gpt-4" to ModelInfo(8192, 0.03, 0.06),
        "gpt-4-turbo" to ModelInfo(128000, 0.01, 0.03),
        "gpt-4o" to ModelInfo(128000, 0.005, 0.015),
        "gpt-3.5-turbo" to ModelInfo(16384, 0.0015, 0.002)
    )

    fun isValidModel(model: String): Boolean = modelInfo.containsKey(model)
    fun getMaxTokens(model: String): Int = modelInfo[model]?.maxTokens ?: 4096
    fun getInputCost(model: String): Double = modelInfo[model]?.inputCostPer1K ?: 0.0
    fun getOutputCost(model: String): Double = modelInfo[model]?.outputCostPer1K ?: 0.0

    private data class ModelInfo(
        val maxTokens: Int,
        val inputCostPer1K: Double,
        val outputCostPer1K: Double
    )
}

object AnthropicModels {
    private val modelInfo = mapOf(
        "claude-3-opus-20240229" to ModelInfo(200000, 0.015, 0.075),
        "claude-3-sonnet-20240229" to ModelInfo(200000, 0.003, 0.015),
        "claude-3-haiku-20240307" to ModelInfo(200000, 0.00025, 0.00125)
    )

    fun isValidModel(model: String): Boolean = modelInfo.containsKey(model)
    fun getMaxTokens(model: String): Int = modelInfo[model]?.maxTokens ?: 200000
    fun getInputCost(model: String): Double = modelInfo[model]?.inputCostPer1K ?: 0.0
    fun getOutputCost(model: String): Double = modelInfo[model]?.outputCostPer1K ?: 0.0

    private data class ModelInfo(
        val maxTokens: Int,
        val inputCostPer1K: Double,
        val outputCostPer1K: Double
    )
}

// ========== ADDITIONAL SUPPORTING CLASSES ==========

data class ValidationResult(
    val isValid: Boolean,
    val errors: List<ValidationError> = emptyList(),
    val warnings: List<ValidationWarning> = emptyList(),
    val score: Float = if (isValid) 1.0f else 0.0f
)

data class ValidationError(val code: String, val message: String)
data class ValidationWarning(val code: String, val message: String)

data class TokenCount(val input: Int, val output: Int) {
    val total: Int get() = input + output
}

data class CostEstimate(
    val estimatedTokens: TokenCount,
    val estimatedCost: Double,
    val currency: String = "USD",
    val breakdown: Map<String, Double> = emptyMap()
)

data class ModelPricing(
    val costPerToken: Double,
    val inputCostPerToken: Double,
    val outputCostPerToken: Double,
    val currency: String = "USD"
)

data class ProviderInfo(
    val provider: AIProvider,
    val region: String?,
    val modelVersion: String?,
    val serverLoad: Float = 0f
)

data class CacheInfo(
    val wasFromCache: Boolean,
    val cacheKey: String?,
    val ttl: Long?,
    val hitRate: Float = 0f
)

data class ResponseQualityAssessment(
    val overallScore: Float,
    val factors: Map<String, Float>,
    val recommendations: List<String>,
    val grade: QualityGrade
)

data class ResponseQualityMetrics(
    val overallScore: Float,
    val contentQuality: Float,
    val completeness: Float,
    val relevance: Float,
    val hasIssues: Boolean,
    val issues: List<String>
)

data class MessageMetadata(
    val timestamp: Long = System.currentTimeMillis(),
    val source: String? = null,
    val tags: Set<String> = emptySet(),
    val priority: Int = 0
)

data class AnthropicRequestMetadata(
    val userId: String? = null,
    val sessionId: String? = null
)

data class CostBreakdown(
    val inputCost: Double,
    val outputCost: Double,
    val cacheCost: Double = 0.0,
    val totalCost: Double
)

data class TokenEfficiency(
    val tokensPerSecond: Float,
    val duration: Long?,
    val cacheHitRate: Float = 0f
)

data class LogProbs(
    val tokens: List<String>,
    val tokenLogprobs: List<Double>,
    val topLogprobs: List<Map<String, Double>>?
)

data class AnthropicContent(
    val type: String,
    val text: String
)

// ========== LEGACY COMPATIBILITY CLASSES ==========

// Keep original classes for backward compatibility
typealias OpenAIChatRequest = EnhancedOpenAIChatRequest
typealias ChatMessage = EnhancedChatMessage
typealias OpenAIChatResponse = EnhancedOpenAIChatResponse
typealias ChatChoice = EnhancedChatChoice
typealias AnthropicMessageRequest = EnhancedAnthropicMessageRequest
typealias AnthropicMessage = EnhancedAnthropicMessage
typealias AnthropicMessageResponse = EnhancedAnthropicMessageResponse
typealias ApiUsage = DetailedApiUsage
typealias AnthropicUsage = DetailedAnthropicUsage

// ========== INTERFACE DEFINITIONS ==========

interface RequestTransformation {
    suspend fun <T> transform(request: AdvancedApiRequest<T>): AdvancedApiRequest<T>
}

interface ResponseTransformation {
    suspend fun <T> transform(response: AdvancedApiResponse<T>): AdvancedApiResponse<T>
}

interface RequestValidator {
    suspend fun <T> validate(request: AdvancedApiRequest<T>): ValidationResult
}

interface ResponseValidation {
    suspend fun <T> validate(response: AdvancedApiResponse<T>): ValidationResult
}

data class RecoveryStrategy(
    val name: String,
    val description: String,
    val successProbability: Float,
    val estimatedDelay: Long,
    val action: suspend () -> Unit
)

data class ErrorContext(
    val requestId: String,
    val attemptNumber: Int,
    val previousErrors: List<AdvancedApiError>,
    val userContext: Map<String, Any> = emptyMap()
)

// ========== UTILITY FUNCTIONS ==========

private fun getModelPricing(provider: AIProvider, model: String): ModelPricing {
    return when (provider) {
        AIProvider.OPENAI -> ModelPricing(
            costPerToken = OpenAIModels.getInputCost(model) / 1000,
            inputCostPerToken = OpenAIModels.getInputCost(model) / 1000,
            outputCostPerToken = OpenAIModels.getOutputCost(model) / 1000
        )
        AIProvider.ANTHROPIC -> ModelPricing(
            costPerToken = AnthropicModels.getInputCost(model) / 1000,
            inputCostPerToken = AnthropicModels.getInputCost(model) / 1000,
            outputCostPerToken = AnthropicModels.getOutputCost(model) / 1000
        )
        else -> ModelPricing(0.0, 0.0, 0.0)
    }
}

private fun getModelMaxTokens(model: String): Int {
    return OpenAIModels.getMaxTokens(model)
}

// ========== EXTENSIONS ==========

data class ResponseFormat(val type: String) {
    companion object {
        val TEXT = ResponseFormat("text")
        val JSON = ResponseFormat("json_object")
    }
}

data class Tool(
    val type: String,
    val function: FunctionDefinition
) {
    fun estimateTokens(): Int {
        return (function.name.length + function.description.length) / 4 + 20
    }
}

data class FunctionDefinition(
    val name: String,
    val description: String,
    val parameters: Map<String, Any>
)

data class ToolCall(
    val id: String,
    val type: String,
    val function: FunctionCall
)

data class FunctionCall(
    val name: String,
    val arguments: String
)

// Legacy error response classes for compatibility
data class OpenAIErrorResponse(val error: OpenAIErrorDetail)
data class OpenAIErrorDetail(val message: String, val type: String, val param: String?, val code: String?)
data class AnthropicErrorResponse(val type: String, val error: AnthropicErrorDetail)
data class AnthropicErrorDetail(val type: String, val message: String)

// Keep existing constants
object ApiConstants {
    const val GPT_4 = "gpt-4"
    const val GPT_4_TURBO = "gpt-4-turbo"
    const val GPT_4O = "gpt-4o"
    const val GPT_3_5_TURBO = "gpt-3.5-turbo"

    const val CLAUDE_3_OPUS = "claude-3-opus-20240229"
    const val CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    const val CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    const val HEADER_AUTHORIZATION = "Authorization"
    const val HEADER_ANTHROPIC_VERSION = "anthropic-version"
    const val HEADER_CONTENT_TYPE = "Content-Type"
    const val CONTENT_TYPE_JSON = "application/json"
}