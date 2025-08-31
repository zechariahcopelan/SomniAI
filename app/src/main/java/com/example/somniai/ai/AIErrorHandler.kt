package com.example.somniai.ai


import com.example.somniai.ai.AIConstants.*
import com.example.somniai.utils.TimeUtils
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*
import kotlin.random.Random

/**
 * AI Error Handler for SomniAI
 *
 * Comprehensive error handling system for AI operations with intelligent retry logic,
 * error classification, recovery strategies, and pattern analysis. Ensures robust
 * AI pipeline operation even under adverse conditions.
 */
object AIErrorHandler {

    // ========== ERROR TRACKING STATE ==========

    private val errorHistory = mutableListOf<AIError>()
    private val errorPatterns = ConcurrentHashMap<String, ErrorPattern>()
    private val modelErrorStats = ConcurrentHashMap<AIModel, ModelErrorStats>()
    private val retryAttempts = ConcurrentHashMap<String, RetryTracker>()

    private val _errorEvents = MutableSharedFlow<AIErrorEvent>()
    val errorEvents: SharedFlow<AIErrorEvent> = _errorEvents.asSharedFlow()

    private val _recoveryEvents = MutableSharedFlow<RecoveryEvent>()
    val recoveryEvents: SharedFlow<RecoveryEvent> = _recoveryEvents.asSharedFlow()

    // Configuration
    private var globalRetryConfig = RetryConfiguration()
    private var circuitBreakerConfig = CircuitBreakerConfiguration()
    private var isErrorHandlingEnabled = true

    // Circuit breaker state
    private val circuitBreakers = ConcurrentHashMap<String, CircuitBreaker>()

    // ========== MAIN ERROR HANDLING INTERFACE ==========

    /**
     * Handle AI operation error with automatic classification and recovery
     */
    suspend fun <T> handleError(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        error: Throwable,
        context: ErrorContext = ErrorContext(),
        operation: suspend () -> T
    ): ErrorHandlingResult<T> {
        if (!isErrorHandlingEnabled) {
            return ErrorHandlingResult.failure(error, "Error handling disabled")
        }

        // Classify the error
        val errorClassification = classifyError(error, operationType, model, context)

        // Record error for analysis
        recordError(operationId, operationType, model, errorClassification, context)

        // Check circuit breaker
        val circuitBreakerKey = "${model}_${operationType}"
        val circuitBreaker = circuitBreakers.getOrPut(circuitBreakerKey) {
            CircuitBreaker(circuitBreakerKey, circuitBreakerConfig)
        }

        if (circuitBreaker.isOpen()) {
            return ErrorHandlingResult.failure(
                CircuitBreakerOpenException("Circuit breaker open for $circuitBreakerKey"),
                "Circuit breaker protection activated"
            )
        }

        // Determine if retry is appropriate
        val retryStrategy = determineRetryStrategy(errorClassification, operationId, context)

        return when (retryStrategy.shouldRetry) {
            true -> executeWithRetry(
                operationId = operationId,
                operationType = operationType,
                model = model,
                retryStrategy = retryStrategy,
                circuitBreaker = circuitBreaker,
                operation = operation
            )
            false -> handleNonRetryableError(errorClassification, context, operation)
        }
    }

    /**
     * Execute operation with retry logic and fallback strategies
     */
    suspend fun <T> executeWithRetry(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        maxRetries: Int = globalRetryConfig.maxRetries,
        retryDelay: Long = globalRetryConfig.baseDelayMs,
        backoffStrategy: BackoffStrategy = globalRetryConfig.backoffStrategy,
        operation: suspend () -> T
    ): ErrorHandlingResult<T> {

        val retryTracker = retryAttempts.getOrPut(operationId) {
            RetryTracker(operationId, maxRetries)
        }

        var lastError: Throwable? = null
        var attempt = 0

        while (attempt <= maxRetries) {
            try {
                val result = operation()

                // Success - clean up retry tracking
                retryAttempts.remove(operationId)
                recordSuccessfulRecovery(operationId, operationType, model, attempt)

                return ErrorHandlingResult.success(result, attempt)

            } catch (error: Throwable) {
                lastError = error
                attempt++
                retryTracker.recordAttempt(error)

                // Classify error to determine if we should continue retrying
                val classification = classifyError(error, operationType, model, ErrorContext())

                if (attempt > maxRetries || !classification.isRetryable) {
                    break
                }

                // Calculate delay for next attempt
                val delayMs = calculateRetryDelay(attempt, retryDelay, backoffStrategy)

                // Emit retry event
                _errorEvents.emit(
                    AIErrorEvent(
                        operationId = operationId,
                        error = classification,
                        action = ErrorAction.RETRY,
                        attempt = attempt,
                        nextDelayMs = delayMs,
                        timestamp = System.currentTimeMillis()
                    )
                )

                delay(delayMs)
            }
        }

        // All retries exhausted
        retryAttempts.remove(operationId)
        recordFailedRecovery(operationId, operationType, model, attempt, lastError)

        return ErrorHandlingResult.failure(
            lastError ?: RuntimeException("Unknown error after $attempt attempts"),
            "Max retries exceeded"
        )
    }

    /**
     * Execute operation with fallback to alternative model
     */
    suspend fun <T> executeWithFallback(
        operationId: String,
        operationType: AIOperationType,
        primaryModel: AIModel,
        fallbackModels: List<AIModel> = AIConfiguration.getCurrentConfig().fallbackModels,
        operation: suspend (AIModel) -> T
    ): ErrorHandlingResult<T> {

        val modelsToTry = listOf(primaryModel) + fallbackModels
        var lastError: Throwable? = null

        for ((index, model) in modelsToTry.withIndex()) {
            try {
                val result = operation(model)

                if (index > 0) {
                    // Record successful fallback
                    recordSuccessfulFallback(operationId, operationType, primaryModel, model)
                }

                return ErrorHandlingResult.success(result, index)

            } catch (error: Throwable) {
                lastError = error
                recordModelError(model, operationType, error)

                val classification = classifyError(error, operationType, model, ErrorContext())

                // If this is a non-recoverable error type, don't try other models
                if (classification.category == ErrorCategory.INPUT_VALIDATION ||
                    classification.category == ErrorCategory.AUTHENTICATION) {
                    break
                }

                // Emit fallback attempt event
                _errorEvents.emit(
                    AIErrorEvent(
                        operationId = operationId,
                        error = classification,
                        action = ErrorAction.FALLBACK,
                        attempt = index + 1,
                        fallbackModel = if (index + 1 < modelsToTry.size) modelsToTry[index + 1] else null,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        // All fallbacks failed
        recordFailedFallback(operationId, operationType, primaryModel, modelsToTry.size)

        return ErrorHandlingResult.failure(
            lastError ?: RuntimeException("All fallback models failed"),
            "All fallback options exhausted"
        )
    }

    // ========== ERROR CLASSIFICATION ==========

    /**
     * Classify error by type, severity, and recoverability
     */
    fun classifyError(
        error: Throwable,
        operationType: AIOperationType,
        model: AIModel,
        context: ErrorContext
    ): ErrorClassification {

        val category = categorizeError(error)
        val severity = determineSeverity(error, operationType, context)
        val isRetryable = determineRetryability(error, category)
        val recoverability = determineRecoverability(error, category, operationType)

        val pattern = detectErrorPattern(error, operationType, model)

        return ErrorClassification(
            errorId = generateErrorId(),
            category = category,
            severity = severity,
            isRetryable = isRetryable,
            recoverability = recoverability,
            originalException = error,
            errorMessage = error.message ?: "Unknown error",
            stackTrace = error.stackTraceToString(),
            operationType = operationType,
            model = model,
            context = context,
            pattern = pattern,
            timestamp = System.currentTimeMillis(),
            suggestedActions = suggestRecoveryActions(category, severity, recoverability)
        )
    }

    private fun categorizeError(error: Throwable): ErrorCategory {
        return when {
            // Network and connectivity errors
            error is UnknownHostException -> ErrorCategory.NETWORK
            error is SocketTimeoutException -> ErrorCategory.TIMEOUT
            error.message?.contains("timeout", ignoreCase = true) == true -> ErrorCategory.TIMEOUT
            error.message?.contains("connection", ignoreCase = true) == true -> ErrorCategory.NETWORK

            // API and service errors
            error.message?.contains("401") == true -> ErrorCategory.AUTHENTICATION
            error.message?.contains("403") == true -> ErrorCategory.AUTHORIZATION
            error.message?.contains("429") == true -> ErrorCategory.RATE_LIMIT
            error.message?.contains("500") == true -> ErrorCategory.SERVER_ERROR
            error.message?.contains("502") == true -> ErrorCategory.SERVER_ERROR
            error.message?.contains("503") == true -> ErrorCategory.SERVICE_UNAVAILABLE

            // AI-specific errors
            error.message?.contains("token", ignoreCase = true) == true -> ErrorCategory.TOKEN_LIMIT
            error.message?.contains("model", ignoreCase = true) == true -> ErrorCategory.MODEL_ERROR
            error.message?.contains("prompt", ignoreCase = true) == true -> ErrorCategory.INPUT_VALIDATION
            error.message?.contains("content policy", ignoreCase = true) == true -> ErrorCategory.CONTENT_POLICY

            // Parsing and format errors
            error is org.json.JSONException -> ErrorCategory.PARSING_ERROR
            error.message?.contains("json", ignoreCase = true) == true -> ErrorCategory.PARSING_ERROR
            error.message?.contains("format", ignoreCase = true) == true -> ErrorCategory.PARSING_ERROR

            // Resource and system errors
            error is OutOfMemoryError -> ErrorCategory.RESOURCE_EXHAUSTION
            error.message?.contains("memory", ignoreCase = true) == true -> ErrorCategory.RESOURCE_EXHAUSTION

            // Default classification
            else -> ErrorCategory.UNKNOWN
        }
    }

    private fun determineSeverity(
        error: Throwable,
        operationType: AIOperationType,
        context: ErrorContext
    ): ErrorSeverity {
        return when {
            // Critical operations or system-breaking errors
            operationType == AIOperationType.SLEEP_ANALYSIS && context.isUserFacing -> ErrorSeverity.HIGH
            error is OutOfMemoryError -> ErrorSeverity.CRITICAL
            error.message?.contains("500") == true -> ErrorSeverity.HIGH

            // Important but recoverable
            error.message?.contains("timeout") == true -> ErrorSeverity.MEDIUM
            error.message?.contains("429") == true -> ErrorSeverity.MEDIUM
            error.message?.contains("503") == true -> ErrorSeverity.MEDIUM

            // Minor issues
            error is org.json.JSONException -> ErrorSeverity.LOW
            error.message?.contains("format") == true -> ErrorSeverity.LOW

            else -> ErrorSeverity.MEDIUM
        }
    }

    private fun determineRetryability(error: Throwable, category: ErrorCategory): Boolean {
        return when (category) {
            // Retryable errors
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.SERVER_ERROR,
            ErrorCategory.SERVICE_UNAVAILABLE,
            ErrorCategory.RESOURCE_EXHAUSTION -> true

            // Non-retryable errors
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.INPUT_VALIDATION,
            ErrorCategory.CONTENT_POLICY,
            ErrorCategory.TOKEN_LIMIT -> false

            // Context-dependent
            ErrorCategory.MODEL_ERROR,
            ErrorCategory.PARSING_ERROR,
            ErrorCategory.UNKNOWN -> error.message?.let { msg ->
                !msg.contains("invalid", ignoreCase = true) &&
                        !msg.contains("malformed", ignoreCase = true)
            } ?: true
        }
    }

    private fun determineRecoverability(
        error: Throwable,
        category: ErrorCategory,
        operationType: AIOperationType
    ): RecoverabilityLevel {
        return when (category) {
            // Fully recoverable with retry/fallback
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.SERVICE_UNAVAILABLE,
            ErrorCategory.SERVER_ERROR -> RecoverabilityLevel.FULL

            // Recoverable with alternative approach
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.MODEL_ERROR,
            ErrorCategory.TOKEN_LIMIT -> RecoverabilityLevel.PARTIAL

            // Requires user intervention
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.AUTHORIZATION,
            ErrorCategory.INPUT_VALIDATION -> RecoverabilityLevel.MANUAL

            // Not recoverable
            ErrorCategory.CONTENT_POLICY,
            ErrorCategory.RESOURCE_EXHAUSTION -> RecoverabilityLevel.NONE

            else -> RecoverabilityLevel.PARTIAL
        }
    }

    // ========== RETRY STRATEGIES ==========

    private fun determineRetryStrategy(
        classification: ErrorClassification,
        operationId: String,
        context: ErrorContext
    ): RetryStrategy {

        val baseStrategy = when (classification.category) {
            ErrorCategory.RATE_LIMIT -> RetryStrategy(
                shouldRetry = true,
                maxRetries = 5,
                baseDelayMs = 2000L,
                backoffStrategy = BackoffStrategy.LINEAR,
                reason = "Rate limit - linear backoff"
            )

            ErrorCategory.TIMEOUT -> RetryStrategy(
                shouldRetry = true,
                maxRetries = 3,
                baseDelayMs = 1000L,
                backoffStrategy = BackoffStrategy.EXPONENTIAL,
                reason = "Timeout - exponential backoff"
            )

            ErrorCategory.NETWORK -> RetryStrategy(
                shouldRetry = true,
                maxRetries = 4,
                baseDelayMs = 500L,
                backoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER,
                reason = "Network error - exponential with jitter"
            )

            ErrorCategory.SERVER_ERROR -> RetryStrategy(
                shouldRetry = true,
                maxRetries = 3,
                baseDelayMs = 2000L,
                backoffStrategy = BackoffStrategy.EXPONENTIAL,
                reason = "Server error - exponential backoff"
            )

            else -> RetryStrategy(
                shouldRetry = classification.isRetryable,
                maxRetries = if (classification.isRetryable) 2 else 0,
                baseDelayMs = 1000L,
                backoffStrategy = BackoffStrategy.EXPONENTIAL,
                reason = "Default strategy based on classification"
            )
        }

        // Adjust based on context and history
        return adjustRetryStrategy(baseStrategy, classification, operationId, context)
    }

    private fun adjustRetryStrategy(
        baseStrategy: RetryStrategy,
        classification: ErrorClassification,
        operationId: String,
        context: ErrorContext
    ): RetryStrategy {

        val existingTracker = retryAttempts[operationId]
        val currentAttempts = existingTracker?.currentAttempt ?: 0

        // Reduce retries if we've been failing consistently
        val adjustedMaxRetries = if (currentAttempts > baseStrategy.maxRetries / 2) {
            maxOf(1, baseStrategy.maxRetries - 1)
        } else {
            baseStrategy.maxRetries
        }

        // Increase delay for high-priority operations
        val adjustedDelay = if (context.isUserFacing) {
            minOf(baseStrategy.baseDelayMs * 2, 10000L)
        } else {
            baseStrategy.baseDelayMs
        }

        return baseStrategy.copy(
            maxRetries = adjustedMaxRetries,
            baseDelayMs = adjustedDelay
        )
    }

    private fun calculateRetryDelay(
        attempt: Int,
        baseDelay: Long,
        strategy: BackoffStrategy
    ): Long {
        return when (strategy) {
            BackoffStrategy.FIXED -> baseDelay

            BackoffStrategy.LINEAR -> baseDelay * attempt

            BackoffStrategy.EXPONENTIAL -> baseDelay * (2.0.pow(attempt - 1).toLong())

            BackoffStrategy.EXPONENTIAL_JITTER -> {
                val exponentialDelay = baseDelay * (2.0.pow(attempt - 1).toLong())
                val jitter = Random.nextLong(0, exponentialDelay / 4)
                exponentialDelay + jitter
            }

            BackoffStrategy.FIBONACCI -> {
                val fib = fibonacci(attempt)
                baseDelay * fib
            }
        }.coerceAtMost(30000L) // Max 30 seconds
    }

    private fun fibonacci(n: Int): Long {
        return if (n <= 1) 1L else {
            var a = 1L
            var b = 1L
            repeat(n - 1) {
                val temp = a + b
                a = b
                b = temp
            }
            b
        }
    }

    // ========== ERROR RECOVERY STRATEGIES ==========

    private suspend fun <T> handleNonRetryableError(
        classification: ErrorClassification,
        context: ErrorContext,
        operation: suspend () -> T
    ): ErrorHandlingResult<T> {

        return when (classification.recoverability) {
            RecoverabilityLevel.FULL -> {
                // Try alternative recovery methods
                tryAlternativeRecovery(classification, operation)
            }

            RecoverabilityLevel.PARTIAL -> {
                // Try degraded operation mode
                tryDegradedMode(classification, operation)
            }

            RecoverabilityLevel.MANUAL -> {
                // Return with user action required
                ErrorHandlingResult.requiresManualIntervention(
                    classification.originalException,
                    generateUserActionMessage(classification),
                    classification.suggestedActions
                )
            }

            RecoverabilityLevel.NONE -> {
                // Unrecoverable error
                ErrorHandlingResult.failure(
                    classification.originalException,
                    "Unrecoverable error: ${classification.errorMessage}"
                )
            }
        }
    }

    private suspend fun <T> tryAlternativeRecovery(
        classification: ErrorClassification,
        operation: suspend () -> T
    ): ErrorHandlingResult<T> {

        return when (classification.category) {
            ErrorCategory.TOKEN_LIMIT -> {
                // Try with reduced token usage
                ErrorHandlingResult.failure(
                    classification.originalException,
                    "Token limit exceeded - consider reducing prompt size"
                )
            }

            ErrorCategory.MODEL_ERROR -> {
                // Model might be temporarily unavailable
                delay(5000L) // Wait 5 seconds
                try {
                    val result = operation()
                    ErrorHandlingResult.success(result, 0)
                } catch (error: Throwable) {
                    ErrorHandlingResult.failure(error, "Alternative recovery failed")
                }
            }

            else -> ErrorHandlingResult.failure(
                classification.originalException,
                "No alternative recovery available"
            )
        }
    }

    private suspend fun <T> tryDegradedMode(
        classification: ErrorClassification,
        operation: suspend () -> T
    ): ErrorHandlingResult<T> {

        // Implement degraded mode operation
        // This could involve using cached responses, simplified operations, etc.
        return ErrorHandlingResult.failure(
            classification.originalException,
            "Degraded mode not implemented for this operation type"
        )
    }

    // ========== ERROR PATTERN DETECTION ==========

    private fun detectErrorPattern(
        error: Throwable,
        operationType: AIOperationType,
        model: AIModel
    ): ErrorPattern? {

        val patternKey = "${operationType}_${model}_${error::class.simpleName}"
        val pattern = errorPatterns.getOrPut(patternKey) {
            ErrorPattern(
                key = patternKey,
                operationType = operationType,
                model = model,
                errorType = error::class.simpleName ?: "Unknown",
                count = AtomicInteger(0),
                firstOccurrence = System.currentTimeMillis(),
                lastOccurrence = System.currentTimeMillis(),
                frequency = 0.0
            )
        }

        pattern.count.incrementAndGet()
        pattern.lastOccurrence = System.currentTimeMillis()

        // Calculate frequency (errors per hour)
        val timeSpanMs = pattern.lastOccurrence - pattern.firstOccurrence
        pattern.frequency = if (timeSpanMs > 0) {
            (pattern.count.get().toDouble() / timeSpanMs) * (60 * 60 * 1000) // Per hour
        } else 0.0

        return pattern
    }

    /**
     * Analyze error patterns and generate insights
     */
    fun analyzeErrorPatterns(timeRangeMs: Long = 24 * 60 * 60 * 1000L): ErrorAnalysisResult {
        val currentTime = System.currentTimeMillis()
        val cutoffTime = currentTime - timeRangeMs

        val recentErrors = errorHistory.filter { it.timestamp >= cutoffTime }
        val activePatterns = errorPatterns.values.filter {
            it.lastOccurrence >= cutoffTime && it.count.get() >= 3
        }

        return ErrorAnalysisResult(
            timeRange = TimeRange(cutoffTime, currentTime),
            totalErrors = recentErrors.size,
            errorsByCategory = recentErrors.groupBy { it.classification.category }
                .mapValues { it.value.size },
            errorsByModel = recentErrors.groupBy { it.model }
                .mapValues { it.value.size },
            errorsByOperation = recentErrors.groupBy { it.operationType }
                .mapValues { it.value.size },
            patterns = activePatterns.sortedByDescending { it.frequency },
            insights = generateErrorInsights(recentErrors, activePatterns),
            recommendations = generateErrorRecommendations(recentErrors, activePatterns)
        )
    }

    private fun generateErrorInsights(
        errors: List<AIError>,
        patterns: List<ErrorPattern>
    ): List<String> {
        val insights = mutableListOf<String>()

        // Most common error category
        val commonCategory = errors.groupBy { it.classification.category }
            .maxByOrNull { it.value.size }
        commonCategory?.let { (category, categoryErrors) ->
            insights.add("Most common error type: $category (${categoryErrors.size} occurrences)")
        }

        // Problematic models
        val modelErrors = errors.groupBy { it.model }
        val problematicModel = modelErrors.filter { it.value.size >= 5 }
            .maxByOrNull { it.value.size }
        problematicModel?.let { (model, modelErrorList) ->
            insights.add("Model with most errors: $model (${modelErrorList.size} errors)")
        }

        // High-frequency patterns
        val highFrequencyPattern = patterns.filter { it.frequency > 1.0 }.maxByOrNull { it.frequency }
        highFrequencyPattern?.let { pattern ->
            insights.add("High-frequency error pattern: ${pattern.errorType} occurring ${String.format("%.1f", pattern.frequency)} times per hour")
        }

        // Error rate trends
        if (errors.size >= 10) {
            val timeSegments = 4
            val segmentSize = errors.size / timeSegments
            val segments = errors.chunked(segmentSize)

            if (segments.size >= 2) {
                val trend = if (segments.last().size > segments.first().size) "increasing" else "decreasing"
                insights.add("Error rate is $trend over the analysis period")
            }
        }

        return insights
    }

    private fun generateErrorRecommendations(
        errors: List<AIError>,
        patterns: List<ErrorPattern>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        // Rate limit issues
        val rateLimitErrors = errors.filter { it.classification.category == ErrorCategory.RATE_LIMIT }
        if (rateLimitErrors.size >= 5) {
            recommendations.add("Consider implementing request throttling or upgrading API limits")
        }

        // Network issues
        val networkErrors = errors.filter { it.classification.category == ErrorCategory.NETWORK }
        if (networkErrors.size >= 3) {
            recommendations.add("Review network connectivity and consider implementing connection pooling")
        }

        // Model-specific issues
        val modelWithMostErrors = errors.groupBy { it.model }.maxByOrNull { it.value.size }
        if (modelWithMostErrors != null && modelWithMostErrors.value.size >= 5) {
            recommendations.add("Consider switching primary model from ${modelWithMostErrors.key} due to high error rate")
        }

        // High-frequency patterns
        patterns.filter { it.frequency > 2.0 }.forEach { pattern ->
            recommendations.add("Investigate root cause of frequent ${pattern.errorType} errors in ${pattern.operationType}")
        }

        return recommendations
    }

    // ========== CIRCUIT BREAKER IMPLEMENTATION ==========

    private class CircuitBreaker(
        private val key: String,
        private val config: CircuitBreakerConfiguration
    ) {
        private var state = CircuitBreakerState.CLOSED
        private var failureCount = 0
        private var lastFailureTime = 0L
        private var successCount = 0

        fun isOpen(): Boolean = state == CircuitBreakerState.OPEN

        fun recordSuccess() {
            if (state == CircuitBreakerState.HALF_OPEN) {
                successCount++
                if (successCount >= config.successThreshold) {
                    state = CircuitBreakerState.CLOSED
                    failureCount = 0
                    successCount = 0
                }
            } else if (state == CircuitBreakerState.CLOSED) {
                // Reset failure count on success
                failureCount = maxOf(0, failureCount - 1)
            }
        }

        fun recordFailure() {
            failureCount++
            lastFailureTime = System.currentTimeMillis()

            when (state) {
                CircuitBreakerState.CLOSED -> {
                    if (failureCount >= config.failureThreshold) {
                        state = CircuitBreakerState.OPEN
                    }
                }
                CircuitBreakerState.HALF_OPEN -> {
                    state = CircuitBreakerState.OPEN
                    successCount = 0
                }
                CircuitBreakerState.OPEN -> {
                    // Already open, check if we should transition to half-open
                    if (System.currentTimeMillis() - lastFailureTime >= config.timeoutMs) {
                        state = CircuitBreakerState.HALF_OPEN
                        successCount = 0
                    }
                }
            }
        }

        fun canAttempt(): Boolean {
            return when (state) {
                CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN -> true
                CircuitBreakerState.OPEN -> {
                    if (System.currentTimeMillis() - lastFailureTime >= config.timeoutMs) {
                        state = CircuitBreakerState.HALF_OPEN
                        successCount = 0
                        true
                    } else false
                }
            }
        }
    }

    // ========== ERROR RECORDING AND TRACKING ==========

    private fun recordError(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        classification: ErrorClassification,
        context: ErrorContext
    ) {
        val aiError = AIError(
            operationId = operationId,
            operationType = operationType,
            model = model,
            classification = classification,
            context = context,
            timestamp = System.currentTimeMillis()
        )

        synchronized(errorHistory) {
            errorHistory.add(aiError)
            if (errorHistory.size > 1000) {
                errorHistory.removeAt(0)
            }
        }

        // Update model error statistics
        updateModelErrorStats(model, classification)

        // Emit error event
        CoroutineScope(Dispatchers.IO).launch {
            _errorEvents.emit(
                AIErrorEvent(
                    operationId = operationId,
                    error = classification,
                    action = ErrorAction.RECORDED,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private fun recordSuccessfulRecovery(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        attempts: Int
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            _recoveryEvents.emit(
                RecoveryEvent(
                    operationId = operationId,
                    operationType = operationType,
                    model = model,
                    recoveryType = RecoveryType.RETRY_SUCCESS,
                    attempts = attempts,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private fun recordFailedRecovery(
        operationId: String,
        operationType: AIOperationType,
        model: AIModel,
        attempts: Int,
        finalError: Throwable?
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            _recoveryEvents.emit(
                RecoveryEvent(
                    operationId = operationId,
                    operationType = operationType,
                    model = model,
                    recoveryType = RecoveryType.RETRY_FAILED,
                    attempts = attempts,
                    finalError = finalError?.message,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private fun recordSuccessfulFallback(
        operationId: String,
        operationType: AIOperationType,
        primaryModel: AIModel,
        fallbackModel: AIModel
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            _recoveryEvents.emit(
                RecoveryEvent(
                    operationId = operationId,
                    operationType = operationType,
                    model = primaryModel,
                    recoveryType = RecoveryType.FALLBACK_SUCCESS,
                    fallbackModel = fallbackModel,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private fun recordFailedFallback(
        operationId: String,
        operationType: AIOperationType,
        primaryModel: AIModel,
        attemptedModels: Int
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            _recoveryEvents.emit(
                RecoveryEvent(
                    operationId = operationId,
                    operationType = operationType,
                    model = primaryModel,
                    recoveryType = RecoveryType.FALLBACK_FAILED,
                    attempts = attemptedModels,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private fun recordModelError(model: AIModel, operationType: AIOperationType, error: Throwable) {
        val stats = modelErrorStats.getOrPut(model) {
            ModelErrorStats(model)
        }

        stats.totalErrors.incrementAndGet()
        stats.lastErrorTime.set(System.currentTimeMillis())

        val operationKey = operationType.toString()
        stats.errorsByOperation[operationKey] =
            stats.errorsByOperation.getOrDefault(operationKey, 0) + 1
    }

    private fun updateModelErrorStats(model: AIModel, classification: ErrorClassification) {
        val stats = modelErrorStats.getOrPut(model) {
            ModelErrorStats(model)
        }

        stats.totalErrors.incrementAndGet()
        stats.lastErrorTime.set(classification.timestamp)

        val categoryKey = classification.category.toString()
        stats.errorsByCategory[categoryKey] =
            stats.errorsByCategory.getOrDefault(categoryKey, 0) + 1
    }

    // ========== CONFIGURATION AND UTILITIES ==========

    /**
     * Configure global retry behavior
     */
    fun configureRetry(config: RetryConfiguration) {
        globalRetryConfig = config
    }

    /**
     * Configure circuit breaker behavior
     */
    fun configureCircuitBreaker(config: CircuitBreakerConfiguration) {
        circuitBreakerConfig = config
    }

    /**
     * Enable or disable error handling
     */
    fun setErrorHandlingEnabled(enabled: Boolean) {
        isErrorHandlingEnabled = enabled
    }

    /**
     * Get current error statistics
     */
    fun getErrorStatistics(): ErrorStatistics {
        val recentErrors = errorHistory.filter {
            it.timestamp >= System.currentTimeMillis() - (24 * 60 * 60 * 1000L)
        }

        return ErrorStatistics(
            totalErrors = errorHistory.size,
            recentErrors = recentErrors.size,
            errorsByCategory = recentErrors.groupBy { it.classification.category }
                .mapValues { it.value.size },
            errorsByModel = recentErrors.groupBy { it.model }
                .mapValues { it.value.size },
            mostCommonError = recentErrors.groupBy { it.classification.category }
                .maxByOrNull { it.value.size }?.key,
            averageRecoveryTime = calculateAverageRecoveryTime(recentErrors),
            circuitBreakerStates = circuitBreakers.mapValues { it.value.state }
        )
    }

    /**
     * Reset error history and statistics
     */
    fun resetErrorHistory() {
        errorHistory.clear()
        errorPatterns.clear()
        modelErrorStats.clear()
        retryAttempts.clear()
        circuitBreakers.clear()
    }

    // ========== HELPER METHODS ==========

    private fun suggestRecoveryActions(
        category: ErrorCategory,
        severity: ErrorSeverity,
        recoverability: RecoverabilityLevel
    ): List<RecoveryAction> {
        val actions = mutableListOf<RecoveryAction>()

        when (category) {
            ErrorCategory.RATE_LIMIT -> {
                actions.add(RecoveryAction.IMPLEMENT_BACKOFF)
                actions.add(RecoveryAction.REDUCE_REQUEST_RATE)
            }
            ErrorCategory.AUTHENTICATION -> {
                actions.add(RecoveryAction.REFRESH_CREDENTIALS)
                actions.add(RecoveryAction.CHECK_API_KEY)
            }
            ErrorCategory.NETWORK -> {
                actions.add(RecoveryAction.RETRY_OPERATION)
                actions.add(RecoveryAction.CHECK_CONNECTIVITY)
            }
            ErrorCategory.MODEL_ERROR -> {
                actions.add(RecoveryAction.SWITCH_MODEL)
                actions.add(RecoveryAction.VALIDATE_INPUT)
            }
            else -> {
                if (recoverability != RecoverabilityLevel.NONE) {
                    actions.add(RecoveryAction.RETRY_OPERATION)
                }
            }
        }

        return actions
    }

    private fun generateUserActionMessage(classification: ErrorClassification): String {
        return when (classification.category) {
            ErrorCategory.AUTHENTICATION -> "Please check your API credentials and try again"
            ErrorCategory.AUTHORIZATION -> "Access denied. Please verify your permissions"
            ErrorCategory.INPUT_VALIDATION -> "Please review and correct your input data"
            ErrorCategory.CONTENT_POLICY -> "Content violates policy. Please modify your request"
            else -> "Manual intervention required: ${classification.errorMessage}"
        }
    }

    private fun generateErrorId(): String {
        return "ERR-${System.currentTimeMillis()}-${Random.nextInt(1000, 9999)}"
    }

    private fun calculateAverageRecoveryTime(errors: List<AIError>): Long {
        // This would need to be implemented based on tracking recovery events
        return 0L
    }
}

// ========== DATA CLASSES ==========

data class ErrorHandlingResult<T>(
    val isSuccess: Boolean,
    val value: T? = null,
    val error: Throwable? = null,
    val errorMessage: String? = null,
    val attemptsUsed: Int = 0,
    val requiresManualIntervention: Boolean = false,
    val suggestedActions: List<RecoveryAction> = emptyList()
) {
    companion object {
        fun <T> success(value: T, attempts: Int = 0) = ErrorHandlingResult(
            isSuccess = true,
            value = value,
            attemptsUsed = attempts
        )

        fun <T> failure(error: Throwable, message: String) = ErrorHandlingResult<T>(
            isSuccess = false,
            error = error,
            errorMessage = message
        )

        fun <T> requiresManualIntervention(
            error: Throwable,
            message: String,
            actions: List<RecoveryAction>
        ) = ErrorHandlingResult<T>(
            isSuccess = false,
            error = error,
            errorMessage = message,
            requiresManualIntervention = true,
            suggestedActions = actions
        )
    }
}

data class ErrorClassification(
    val errorId: String,
    val category: ErrorCategory,
    val severity: ErrorSeverity,
    val isRetryable: Boolean,
    val recoverability: RecoverabilityLevel,
    val originalException: Throwable,
    val errorMessage: String,
    val stackTrace: String,
    val operationType: AIOperationType,
    val model: AIModel,
    val context: ErrorContext,
    val pattern: ErrorPattern?,
    val timestamp: Long,
    val suggestedActions: List<RecoveryAction>
)

data class ErrorContext(
    val isUserFacing: Boolean = false,
    val timeoutMs: Long? = null,
    val requestSize: Int? = null,
    val metadata: Map<String, String> = emptyMap()
)

data class RetryStrategy(
    val shouldRetry: Boolean,
    val maxRetries: Int,
    val baseDelayMs: Long,
    val backoffStrategy: BackoffStrategy,
    val reason: String
)

data class RetryConfiguration(
    val maxRetries: Int = MAX_RETRY_ATTEMPTS,
    val baseDelayMs: Long = ERROR_RETRY_DELAY_MS,
    val backoffStrategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    val enableCircuitBreaker: Boolean = true
)

data class CircuitBreakerConfiguration(
    val failureThreshold: Int = 5,
    val successThreshold: Int = 3,
    val timeoutMs: Long = 60000L // 1 minute
)

data class RetryTracker(
    val operationId: String,
    val maxRetries: Int,
    var currentAttempt: Int = 0,
    val errors: MutableList<Throwable> = mutableListOf()
) {
    fun recordAttempt(error: Throwable) {
        currentAttempt++
        errors.add(error)
    }
}

data class ErrorPattern(
    val key: String,
    val operationType: AIOperationType,
    val model: AIModel,
    val errorType: String,
    val count: AtomicInteger,
    val firstOccurrence: Long,
    var lastOccurrence: Long,
    var frequency: Double
)

data class ModelErrorStats(
    val model: AIModel,
    val totalErrors: AtomicInteger = AtomicInteger(0),
    val lastErrorTime: AtomicLong = AtomicLong(0),
    val errorsByCategory: MutableMap<String, Int> = mutableMapOf(),
    val errorsByOperation: MutableMap<String, Int> = mutableMapOf()
)

data class AIError(
    val operationId: String,
    val operationType: AIOperationType,
    val model: AIModel,
    val classification: ErrorClassification,
    val context: ErrorContext,
    val timestamp: Long
)

data class AIErrorEvent(
    val operationId: String,
    val error: ErrorClassification,
    val action: ErrorAction,
    val attempt: Int = 0,
    val nextDelayMs: Long? = null,
    val fallbackModel: AIModel? = null,
    val timestamp: Long
)

data class RecoveryEvent(
    val operationId: String,
    val operationType: AIOperationType,
    val model: AIModel,
    val recoveryType: RecoveryType,
    val attempts: Int = 0,
    val fallbackModel: AIModel? = null,
    val finalError: String? = null,
    val timestamp: Long
)

data class ErrorAnalysisResult(
    val timeRange: TimeRange,
    val totalErrors: Int,
    val errorsByCategory: Map<ErrorCategory, Int>,
    val errorsByModel: Map<AIModel, Int>,
    val errorsByOperation: Map<AIOperationType, Int>,
    val patterns: List<ErrorPattern>,
    val insights: List<String>,
    val recommendations: List<String>
)

data class ErrorStatistics(
    val totalErrors: Int,
    val recentErrors: Int,
    val errorsByCategory: Map<ErrorCategory, Int>,
    val errorsByModel: Map<AIModel, Int>,
    val mostCommonError: ErrorCategory?,
    val averageRecoveryTime: Long,
    val circuitBreakerStates: Map<String, CircuitBreakerState>
)

data class TimeRange(
    val startTime: Long,
    val endTime: Long
)

// ========== ENUMS ==========

enum class ErrorCategory {
    NETWORK,
    TIMEOUT,
    AUTHENTICATION,
    AUTHORIZATION,
    RATE_LIMIT,
    SERVER_ERROR,
    SERVICE_UNAVAILABLE,
    TOKEN_LIMIT,
    MODEL_ERROR,
    INPUT_VALIDATION,
    CONTENT_POLICY,
    PARSING_ERROR,
    RESOURCE_EXHAUSTION,
    UNKNOWN
}

enum class ErrorSeverity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class RecoverabilityLevel {
    NONE,
    MANUAL,
    PARTIAL,
    FULL
}

enum class BackoffStrategy {
    FIXED,
    LINEAR,
    EXPONENTIAL,
    EXPONENTIAL_JITTER,
    FIBONACCI
}

enum class CircuitBreakerState {
    CLOSED,
    OPEN,
    HALF_OPEN
}

enum class ErrorAction {
    RECORDED,
    RETRY,
    FALLBACK,
    MANUAL_INTERVENTION,
    IGNORE
}

enum class RecoveryType {
    RETRY_SUCCESS,
    RETRY_FAILED,
    FALLBACK_SUCCESS,
    FALLBACK_FAILED,
    MANUAL_RECOVERY
}

enum class RecoveryAction {
    RETRY_OPERATION,
    SWITCH_MODEL,
    IMPLEMENT_BACKOFF,
    REDUCE_REQUEST_RATE,
    REFRESH_CREDENTIALS,
    CHECK_API_KEY,
    CHECK_CONNECTIVITY,
    VALIDATE_INPUT,
    CONTACT_SUPPORT,
    MANUAL_INTERVENTION
}

// ========== EXCEPTIONS ==========

class CircuitBreakerOpenException(message: String) : Exception(message)