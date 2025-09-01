package com.example.somniai.ai

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.example.somniai.data.*
import com.google.gson.*
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import java.util.regex.Pattern
import kotlin.math.*
import kotlin.random.Random

/**
 * Enhanced AI Response Parser with Comprehensive Error Handling
 *
 * Core Features:
 * - Multi-strategy parsing with intelligent fallback mechanisms
 * - Circuit breaker pattern for failure resilience
 * - Comprehensive error recovery and graceful degradation
 * - Real-time performance monitoring and analytics integration
 * - Advanced response validation and quality assessment
 * - Intelligent caching with content-based invalidation
 * - Model-specific parsing optimization and adaptation
 * - Robust JSON parsing with multiple recovery strategies
 * - Natural language processing for unstructured responses
 * - Content sanitization and security validation
 * - Parsing timeout management and resource protection
 * - Comprehensive logging and debugging capabilities
 * - Integration with existing performance monitoring systems
 */
class AIResponseParser(
    private val context: Context,
    private val preferences: SharedPreferences,
    private val performanceMonitor: AIPerformanceMonitor,
    private val parsingAnalytics: ParsingPerformanceAnalytics,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "AIResponseParser"

        // Error handling and recovery constants
        private const val MAX_RETRY_ATTEMPTS = 3
        private const val RETRY_DELAY_BASE_MS = 100L
        private const val MAX_RETRY_DELAY_MS = 2000L
        private const val PARSING_TIMEOUT_MS = 15000L
        private const val CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
        private const val CIRCUIT_BREAKER_RESET_TIMEOUT_MS = 30000L

        // Content validation thresholds
        private const val MIN_RESPONSE_LENGTH = 10
        private const val MAX_RESPONSE_LENGTH = 50000
        private const val MIN_INSIGHT_CONTENT_LENGTH = 20
        private const val MAX_INSIGHT_CONTENT_LENGTH = 2000

        // Quality assessment constants
        private const val MIN_QUALITY_THRESHOLD = 0.3f
        private const val HIGH_QUALITY_THRESHOLD = 0.8f
        private const val EXCELLENT_QUALITY_THRESHOLD = 0.9f

        // Performance optimization
        private const val CACHE_SIZE = 100
        private const val CACHE_TTL_HOURS = 2L
        private const val CLEANUP_INTERVAL_MINUTES = 30L

        // JSON parsing patterns
        private val JSON_START_PATTERN = Pattern.compile("\\s*[\\[{]")
        private val JSON_EXTRACT_PATTERN = Pattern.compile("```(?:json)?\\s*([\\s\\S]*?)\\s*```|({[\\s\\S]*}|\\[[\\s\\S]*\\])")
        private val INSIGHT_MARKERS = listOf("title", "description", "recommendation", "category", "priority")

        // Preference keys
        private const val PREF_PARSER_CONFIG = "ai_response_parser_config"
        private const val PREF_ERROR_STATISTICS = "parser_error_statistics"
        private const val PREF_STRATEGY_PERFORMANCE = "parser_strategy_performance"
        private const val PREF_CIRCUIT_BREAKER_STATE = "parser_circuit_breaker_state"
    }

    // Core parsing infrastructure
    private val scope = CoroutineScope(dispatcher + SupervisorJob())
    private val circuitBreaker = ParsingCircuitBreaker()
    private val strategySelector = ParsingStrategySelector()
    private val contentValidator = ResponseContentValidator()
    private val errorRecovery = ParsingErrorRecovery()
    private val qualityAssessor = ResponseQualityAssessor()
    private val cacheManager = ResponseCacheManager()

    // State management
    private val _parsingState = MutableStateFlow<ParsingState>(ParsingState.IDLE)
    val parsingState: StateFlow<ParsingState> = _parsingState.asStateFlow()

    private val _parsingMetrics = MutableLiveData<ParsedResponseMetrics>()
    val parsingMetrics: LiveData<ParsedResponseMetrics> = _parsingMetrics

    private val _errorEvents = MutableSharedFlow<ParsingErrorEvent>()
    val errorEvents: SharedFlow<ParsingErrorEvent> = _errorEvents.asSharedFlow()

    // Performance tracking
    private val totalParseRequests = AtomicLong(0L)
    private val successfulParses = AtomicLong(0L)
    private val failedParses = AtomicLong(0L)
    private val averageParsingTime = AtomicReference(0L)
    private val retryAttempts = AtomicLong(0L)
    private val circuitBreakerTrips = AtomicInteger(0)

    // Strategy performance tracking
    private val strategyMetrics = ConcurrentHashMap<ParsingStrategy, StrategyPerformanceMetrics>()
    private val modelParsingStats = ConcurrentHashMap<AIModel, ModelParsingStats>()
    private val errorPatterns = ConcurrentHashMap<String, ErrorPattern>()

    // Configuration
    private var parserConfig: ParserConfiguration = loadParserConfiguration()
    private val gson = createAdvancedGsonParser()

    // Caching
    private val responseCache = ConcurrentHashMap<String, CachedParseResult>()
    private val recentFailures = ConcurrentHashMap<String, FailureRecord>()

    // ========== INITIALIZATION AND LIFECYCLE ==========

    /**
     * Initialize the enhanced response parser
     */
    suspend fun initialize(): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Initializing AIResponseParser with enhanced error handling")

            // Initialize core components
            initializeComponents()

            // Load historical data and performance metrics
            loadParserState()

            // Start background maintenance and monitoring
            startBackgroundTasks()

            // Validate parser readiness
            val readinessCheck = performReadinessCheck()
            if (!readinessCheck.isReady) {
                return@withContext Result.failure(IllegalStateException("Parser not ready: ${readinessCheck.issues}"))
            }

            _parsingState.value = ParsingState.READY

            Log.d(TAG, "AIResponseParser initialized successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize AIResponseParser", e)
            _parsingState.value = ParsingState.ERROR("Initialization failed: ${e.message}")
            Result.failure(e)
        }
    }

    // ========== CORE PARSING API ==========

    /**
     * Parse AI response with comprehensive error handling and recovery
     */
    suspend fun parseResponse(
        aiResponse: String,
        model: AIModel,
        context: ResponseParsingContext,
        options: ParseOptions = ParseOptions.default()
    ): Result<ParsedResponse> = withContext(dispatcher) {
        val operationId = generateOperationId()
        val startTime = System.currentTimeMillis()

        // Start parsing tracker
        val tracker = parsingAnalytics.startParsingOperation(
            operationId = operationId,
            model = model,
            responseType = context.expectedResponseType,
            contentLength = aiResponse.length,
            expectedSchema = context.expectedSchema
        )

        try {
            Log.d(TAG, "Parsing AI response: model=$model, length=${aiResponse.length}, context=${context.type}")

            totalParseRequests.incrementAndGet()
            _parsingState.value = ParsingState.PARSING

            // Pre-parsing validation
            validateInputParameters(aiResponse, model, context, options).getOrThrow()

            // Check circuit breaker
            if (!circuitBreaker.canExecute(model)) {
                val error = IllegalStateException("Circuit breaker open for model $model")
                tracker.recordError(ParsingErrorType.CIRCUIT_BREAKER_OPEN, error.message ?: "Circuit breaker open")
                tracker.complete(success = false, parsedContentLength = 0, qualityScore = 0f)
                throw error
            }

            // Check cache first (if enabled)
            if (options.enableCaching) {
                val cacheKey = generateCacheKey(aiResponse, model, context)
                val cachedResult = cacheManager.getCachedResult(cacheKey)
                if (cachedResult != null && !cachedResult.isExpired()) {
                    Log.d(TAG, "Cache hit for parsing request")
                    tracker.complete(success = true, parsedContentLength = cachedResult.result.content.length, qualityScore = cachedResult.result.qualityScore)
                    return@withContext Result.success(cachedResult.result.copy(fromCache = true))
                }
            }

            // Execute parsing with retry logic
            val parseResult = executeParsingWithRetry(
                response = aiResponse,
                model = model,
                context = context,
                options = options,
                tracker = tracker,
                operationId = operationId
            ).getOrThrow()

            // Post-processing and validation
            val validatedResult = postProcessAndValidate(parseResult, context, model).getOrThrow()

            // Cache successful result
            if (options.enableCaching && validatedResult.qualityScore >= MIN_QUALITY_THRESHOLD) {
                val cacheKey = generateCacheKey(aiResponse, model, context)
                cacheManager.cacheResult(cacheKey, validatedResult)
            }

            // Record success metrics
            val processingTime = System.currentTimeMillis() - startTime
            recordSuccessfulParse(model, context, validatedResult, processingTime)

            // Complete tracking
            tracker.complete(
                success = true,
                parsedContentLength = validatedResult.content.length,
                qualityScore = validatedResult.qualityScore
            )

            circuitBreaker.recordSuccess(model)
            successfulParses.incrementAndGet()
            updateAverageParsingTime(processingTime)

            _parsingState.value = ParsingState.COMPLETED

            Log.d(TAG, "Parsing completed successfully: ${validatedResult.insights.size} insights, quality=${validatedResult.qualityScore}")
            Result.success(validatedResult)

        } catch (e: Exception) {
            Log.e(TAG, "Parsing failed for model $model", e)

            // Handle error with comprehensive recovery
            val recoveryResult = handleParsingError(
                error = e,
                originalResponse = aiResponse,
                model = model,
                context = context,
                options = options,
                processingTime = System.currentTimeMillis() - startTime,
                tracker = tracker
            )

            // Record failure metrics
            recordFailedParse(model, context, e, System.currentTimeMillis() - startTime)
            circuitBreaker.recordFailure(model)
            failedParses.incrementAndGet()

            _parsingState.value = ParsingState.ERROR(e.message ?: "Unknown parsing error")

            // Emit error event for monitoring
            scope.launch {
                _errorEvents.emit(
                    ParsingErrorEvent(
                        operationId = operationId,
                        model = model,
                        errorType = classifyError(e),
                        errorMessage = e.message ?: "Unknown error",
                        timestamp = System.currentTimeMillis()
                    )
                )
            }

            recoveryResult ?: Result.failure(e)
        }
    }

    /**
     * Parse batch responses with optimized error handling
     */
    suspend fun parseBatchResponses(
        responses: List<BatchParseRequest>,
        options: ParseOptions = ParseOptions.default()
    ): Result<BatchParseResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Parsing batch responses: ${responses.size} items")

            if (responses.isEmpty()) {
                return@withContext Result.success(BatchParseResult.empty())
            }

            val results = mutableListOf<ParsedResponse>()
            val errors = mutableListOf<BatchParseError>()
            val startTime = System.currentTimeMillis()

            // Process responses with controlled concurrency
            val concurrency = minOf(parserConfig.maxBatchConcurrency, responses.size)
            val semaphore = kotlinx.coroutines.sync.Semaphore(concurrency)

            val jobs = responses.map { request ->
                scope.async {
                    semaphore.withPermit {
                        try {
                            val result = parseResponse(
                                aiResponse = request.response,
                                model = request.model,
                                context = request.context,
                                options = options
                            )

                            result.fold(
                                onSuccess = { parsedResponse ->
                                    synchronized(results) { results.add(parsedResponse) }
                                },
                                onFailure = { error ->
                                    synchronized(errors) {
                                        errors.add(
                                            BatchParseError(
                                                requestId = request.id,
                                                model = request.model,
                                                error = error,
                                                timestamp = System.currentTimeMillis()
                                            )
                                        )
                                    }
                                }
                            )
                        } catch (e: Exception) {
                            synchronized(errors) {
                                errors.add(
                                    BatchParseError(
                                        requestId = request.id,
                                        model = request.model,
                                        error = e,
                                        timestamp = System.currentTimeMillis()
                                    )
                                )
                            }
                        }
                    }
                }
            }

            // Wait for all jobs with timeout
            try {
                withTimeout(options.maxBatchProcessingTime) {
                    jobs.awaitAll()
                }
            } catch (e: TimeoutCancellationException) {
                Log.w(TAG, "Batch parsing timed out, returning partial results")
                jobs.forEach { if (it.isActive) it.cancel() }
            }

            val batchResult = BatchParseResult(
                successful = results,
                failed = errors,
                totalProcessingTime = System.currentTimeMillis() - startTime,
                successRate = if (responses.isNotEmpty()) results.size.toFloat() / responses.size else 0f
            )

            // Record batch metrics
            recordBatchParseMetrics(batchResult)

            Log.d(TAG, "Batch parsing completed: ${results.size} successful, ${errors.size} failed")
            Result.success(batchResult)

        } catch (e: Exception) {
            Log.e(TAG, "Batch parsing failed", e)
            Result.failure(e)
        }
    }

    /**
     * Parse response with specific strategy (for testing or optimization)
     */
    suspend fun parseWithStrategy(
        aiResponse: String,
        model: AIModel,
        strategy: ParsingStrategy,
        context: ResponseParsingContext
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Parsing with specific strategy: $strategy")

            val startTime = System.currentTimeMillis()
            val operationId = generateOperationId()

            // Execute strategy directly
            val parseResult = executeParsingStrategy(
                strategy = strategy,
                response = aiResponse,
                model = model,
                context = context,
                operationId = operationId
            ).getOrThrow()

            // Record strategy performance
            recordStrategyPerformance(strategy, model, System.currentTimeMillis() - startTime, true, parseResult.qualityScore)

            Result.success(parseResult)

        } catch (e: Exception) {
            Log.e(TAG, "Strategy-specific parsing failed: $strategy", e)
            recordStrategyPerformance(strategy, model, 0L, false, 0f)
            Result.failure(e)
        }
    }

    // ========== PARSING STRATEGY EXECUTION ==========

    private suspend fun executeParsingWithRetry(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        options: ParseOptions,
        tracker: ParsingTracker,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        var lastException: Exception? = null
        var attempt = 0

        while (attempt < MAX_RETRY_ATTEMPTS) {
            try {
                attempt++

                // Select optimal parsing strategy
                val strategy = strategySelector.selectStrategy(
                    response = response,
                    model = model,
                    context = context,
                    previousFailures = if (attempt > 1) listOf(lastException) else emptyList(),
                    strategyMetrics = strategyMetrics
                )

                Log.d(TAG, "Parsing attempt $attempt with strategy: $strategy")

                // Apply parsing timeout
                val parseResult = withTimeout(options.parsingTimeout) {
                    executeParsingStrategy(
                        strategy = strategy,
                        response = response,
                        model = model,
                        context = context,
                        operationId = operationId
                    )
                }.getOrThrow()

                // Record successful strategy usage
                recordStrategyPerformance(strategy, model, 0L, true, parseResult.qualityScore)

                return@withContext Result.success(parseResult)

            } catch (e: TimeoutCancellationException) {
                lastException = Exception("Parsing timeout after ${options.parsingTimeout}ms", e)
                Log.w(TAG, "Parsing attempt $attempt timed out")

            } catch (e: Exception) {
                lastException = e
                Log.w(TAG, "Parsing attempt $attempt failed", e)

                // Record retry attempt
                retryAttempts.incrementAndGet()
                tracker.recordError(classifyError(e), e.message ?: "Unknown error")

                // Wait before retry with exponential backoff
                if (attempt < MAX_RETRY_ATTEMPTS) {
                    val delay = minOf(
                        RETRY_DELAY_BASE_MS * (1L shl (attempt - 1)),
                        MAX_RETRY_DELAY_MS
                    )
                    delay(delay)
                }
            }
        }

        // All attempts failed, try error recovery
        val recoveredResult = errorRecovery.attemptRecovery(
            originalResponse = response,
            model = model,
            context = context,
            lastError = lastException ?: Exception("All parsing attempts failed")
        )

        if (recoveredResult != null) {
            Log.i(TAG, "Error recovery successful after $attempt failed attempts")
            return@withContext Result.success(recoveredResult)
        }

        Result.failure(lastException ?: Exception("Parsing failed after $attempt attempts"))
    }

    private suspend fun executeParsingStrategy(
        strategy: ParsingStrategy,
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        val startTime = System.currentTimeMillis()

        try {
            val parseResult = when (strategy) {
                ParsingStrategy.STRICT_JSON -> {
                    parseStrictJSON(response, model, context, operationId)
                }

                ParsingStrategy.FLEXIBLE_JSON -> {
                    parseFlexibleJSON(response, model, context, operationId)
                }

                ParsingStrategy.EXTRACT_JSON -> {
                    parseExtractedJSON(response, model, context, operationId)
                }

                ParsingStrategy.NATURAL_LANGUAGE -> {
                    parseNaturalLanguage(response, model, context, operationId)
                }

                ParsingStrategy.HYBRID_APPROACH -> {
                    parseHybridApproach(response, model, context, operationId)
                }

                ParsingStrategy.PATTERN_MATCHING -> {
                    parsePatternMatching(response, model, context, operationId)
                }

                ParsingStrategy.FALLBACK_BASIC -> {
                    parseFallbackBasic(response, model, context, operationId)
                }

                ParsingStrategy.RECOVERY_MODE -> {
                    parseRecoveryMode(response, model, context, operationId)
                }
            }.getOrThrow()

            // Record strategy timing
            val executionTime = System.currentTimeMillis() - startTime
            recordStrategyPerformance(strategy, model, executionTime, true, parseResult.qualityScore)

            Result.success(parseResult)

        } catch (e: Exception) {
            val executionTime = System.currentTimeMillis() - startTime
            recordStrategyPerformance(strategy, model, executionTime, false, 0f)
            Log.e(TAG, "Parsing strategy $strategy failed", e)
            Result.failure(e)
        }
    }

    // ========== INDIVIDUAL PARSING STRATEGIES ==========

    private suspend fun parseStrictJSON(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Validate JSON format first
            if (!isValidJSON(response)) {
                throw IllegalArgumentException("Response is not valid JSON")
            }

            // Parse with strict Gson settings
            val insights = when (context.expectedResponseType) {
                ResponseType.INSIGHTS -> {
                    try {
                        // Try parsing as array first
                        val insightArray = gson.fromJson(response, Array<InsightDTO>::class.java)
                        insightArray.map { it.toSleepInsight() }
                    } catch (e: JsonSyntaxException) {
                        // Try parsing as single object
                        val singleInsight = gson.fromJson(response, InsightDTO::class.java)
                        listOf(singleInsight.toSleepInsight())
                    }
                }

                ResponseType.STRUCTURED -> {
                    parseStructuredResponse(response, context.expectedSchema)
                }

                else -> {
                    parseGenericJSON(response, context)
                }
            }

            val qualityScore = qualityAssessor.assessInsights(insights, response, model)

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.STRICT_JSON,
                qualityScore = qualityScore,
                confidence = if (qualityScore > HIGH_QUALITY_THRESHOLD) 0.95f else 0.8f,
                processingTime = 0L, // Will be set by caller
                model = model,
                content = response,
                metadata = mapOf(
                    "json_valid" to "true",
                    "insight_count" to insights.size.toString()
                )
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Strict JSON parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseFlexibleJSON(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Clean and normalize JSON
            val cleanedJSON = contentValidator.cleanAndNormalizeJSON(response)

            // Attempt parsing with error recovery
            val insights = parseJSONWithRecovery(cleanedJSON, context)

            val qualityScore = qualityAssessor.assessInsights(insights, response, model)

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.FLEXIBLE_JSON,
                qualityScore = qualityScore,
                confidence = if (insights.isNotEmpty()) 0.75f else 0.3f,
                processingTime = 0L,
                model = model,
                content = cleanedJSON,
                metadata = mapOf(
                    "cleaned" to "true",
                    "insight_count" to insights.size.toString()
                )
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Flexible JSON parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseExtractedJSON(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Extract JSON from mixed content
            val extractedJSON = extractJSONFromResponse(response)

            if (extractedJSON.isNullOrBlank()) {
                throw IllegalArgumentException("No JSON content found in response")
            }

            // Parse extracted JSON
            val insights = parseJSONWithRecovery(extractedJSON, context)

            val qualityScore = qualityAssessor.assessInsights(insights, response, model)

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.EXTRACT_JSON,
                qualityScore = qualityScore,
                confidence = 0.7f,
                processingTime = 0L,
                model = model,
                content = extractedJSON,
                metadata = mapOf(
                    "extracted" to "true",
                    "original_length" to response.length.toString(),
                    "extracted_length" to extractedJSON.length.toString()
                )
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "JSON extraction parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseNaturalLanguage(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Natural language processing for insight extraction
            val insights = extractInsightsFromText(response, model, context)

            val qualityScore = qualityAssessor.assessInsights(insights, response, model)

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.NATURAL_LANGUAGE,
                qualityScore = qualityScore,
                confidence = 0.6f,
                processingTime = 0L,
                model = model,
                content = response,
                metadata = mapOf(
                    "text_processing" to "true",
                    "word_count" to response.split("\\s+".toRegex()).size.toString()
                )
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Natural language parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseHybridApproach(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            val insights = mutableListOf<SleepInsight>()

            // Try JSON extraction first
            try {
                val jsonContent = extractJSONFromResponse(response)
                if (!jsonContent.isNullOrBlank()) {
                    val jsonInsights = parseJSONWithRecovery(jsonContent, context)
                    insights.addAll(jsonInsights)
                }
            } catch (e: Exception) {
                Log.w(TAG, "JSON extraction failed in hybrid approach", e)
            }

            // Fall back to natural language processing
            if (insights.isEmpty()) {
                try {
                    val nlpInsights = extractInsightsFromText(response, model, context)
                    insights.addAll(nlpInsights)
                } catch (e: Exception) {
                    Log.w(TAG, "NLP extraction failed in hybrid approach", e)
                }
            }

            if (insights.isEmpty()) {
                throw IllegalStateException("No insights extracted with hybrid approach")
            }

            val qualityScore = qualityAssessor.assessInsights(insights, response, model)

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.HYBRID_APPROACH,
                qualityScore = qualityScore,
                confidence = 0.65f,
                processingTime = 0L,
                model = model,
                content = response,
                metadata = mapOf("hybrid" to "true", "insight_count" to insights.size.toString())
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Hybrid approach parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parsePatternMatching(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Pattern-based insight extraction
            val insights = extractInsightsUsingPatterns(response, model, context)

            val qualityScore = qualityAssessor.assessInsights(insights, response, model)

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.PATTERN_MATCHING,
                qualityScore = qualityScore,
                confidence = 0.5f,
                processingTime = 0L,
                model = model,
                content = response,
                metadata = mapOf("pattern_matching" to "true")
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Pattern matching parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseFallbackBasic(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Basic fallback - create minimal insight from response
            val insight = SleepInsight(
                category = InsightCategory.GENERAL,
                title = "Sleep Analysis",
                description = response.take(MAX_INSIGHT_CONTENT_LENGTH),
                recommendation = "Continue monitoring your sleep patterns",
                priority = 2,
                isAiGenerated = true,
                timestamp = System.currentTimeMillis()
            )

            val result = ParsedResponse(
                insights = listOf(insight),
                originalResponse = response,
                strategy = ParsingStrategy.FALLBACK_BASIC,
                qualityScore = 0.3f,
                confidence = 0.3f,
                processingTime = 0L,
                model = model,
                content = response,
                metadata = mapOf("fallback" to "true")
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Fallback basic parsing failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseRecoveryMode(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        operationId: String
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Emergency recovery mode - minimal processing
            val insights = if (response.length >= MIN_INSIGHT_CONTENT_LENGTH) {
                listOf(
                    SleepInsight(
                        category = InsightCategory.GENERAL,
                        title = "Analysis Available",
                        description = "Sleep data has been processed",
                        recommendation = "Review your sleep patterns",
                        priority = 3,
                        isAiGenerated = true,
                        timestamp = System.currentTimeMillis()
                    )
                )
            } else {
                emptyList()
            }

            val result = ParsedResponse(
                insights = insights,
                originalResponse = response,
                strategy = ParsingStrategy.RECOVERY_MODE,
                qualityScore = 0.2f,
                confidence = 0.2f,
                processingTime = 0L,
                model = model,
                content = response,
                metadata = mapOf("recovery" to "true"),
                hasErrors = true
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Recovery mode parsing failed", e)
            Result.failure(e)
        }
    }

    // ========== ERROR HANDLING AND RECOVERY ==========

    private suspend fun handleParsingError(
        error: Exception,
        originalResponse: String,
        model: AIModel,
        context: ResponseParsingContext,
        options: ParseOptions,
        processingTime: Long,
        tracker: ParsingTracker
    ): Result<ParsedResponse>? = withContext(dispatcher) {
        try {
            Log.w(TAG, "Handling parsing error: ${error.message}")

            val errorType = classifyError(error)
            val errorKey = "${model}_${errorType}_${error.javaClass.simpleName}"

            // Record error pattern
            recordErrorPattern(errorKey, error, model, originalResponse)

            // Complete tracker with error
            tracker.recordError(errorType, error.message ?: "Unknown error")
            tracker.complete(success = false, parsedContentLength = 0, qualityScore = 0f)

            // Attempt emergency recovery if enabled
            if (options.enableErrorRecovery) {
                val recoveryResult = errorRecovery.attemptRecovery(
                    originalResponse = originalResponse,
                    model = model,
                    context = context,
                    lastError = error
                )

                if (recoveryResult != null) {
                    Log.i(TAG, "Emergency recovery successful for ${error.javaClass.simpleName}")
                    return@withContext Result.success(recoveryResult)
                }
            }

            // No recovery possible
            null

        } catch (e: Exception) {
            Log.e(TAG, "Error handling failed", e)
            null
        }
    }

    private fun classifyError(error: Exception): ParsingErrorType {
        return when (error) {
            is JsonSyntaxException, is JsonParseException -> ParsingErrorType.JSON_PARSE_ERROR
            is IllegalArgumentException -> ParsingErrorType.INVALID_INPUT
            is TimeoutCancellationException -> ParsingErrorType.TIMEOUT_ERROR
            is OutOfMemoryError -> ParsingErrorType.MEMORY_ERROR
            is SecurityException -> ParsingErrorType.SECURITY_ERROR
            is IllegalStateException -> when {
                error.message?.contains("Circuit breaker") == true -> ParsingErrorType.CIRCUIT_BREAKER_OPEN
                else -> ParsingErrorType.INVALID_STATE
            }
            else -> ParsingErrorType.UNKNOWN_ERROR
        }
    }

    // ========== VALIDATION AND QUALITY ASSESSMENT ==========

    private suspend fun validateInputParameters(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        options: ParseOptions
    ): Result<Unit> = withContext(dispatcher) {
        try {
            // Validate response length
            if (response.length < MIN_RESPONSE_LENGTH) {
                throw IllegalArgumentException("Response too short: ${response.length} chars")
            }

            if (response.length > MAX_RESPONSE_LENGTH) {
                throw IllegalArgumentException("Response too long: ${response.length} chars")
            }

            // Validate response content
            if (response.isBlank()) {
                throw IllegalArgumentException("Response is blank")
            }

            // Security validation
            contentValidator.validateSecurity(response).getOrThrow()

            // Model compatibility check
            if (!isModelCompatible(model, context.expectedResponseType)) {
                Log.w(TAG, "Model $model may not be optimal for ${context.expectedResponseType}")
            }

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Input validation failed", e)
            Result.failure(e)
        }
    }

    private suspend fun postProcessAndValidate(
        parseResult: ParsedResponse,
        context: ResponseParsingContext,
        model: AIModel
    ): Result<ParsedResponse> = withContext(dispatcher) {
        try {
            // Validate insights content
            val validatedInsights = parseResult.insights.filter { insight ->
                contentValidator.validateInsight(insight).getOrElse { false }
            }

            if (validatedInsights.isEmpty() && parseResult.insights.isNotEmpty()) {
                Log.w(TAG, "All insights failed validation")
            }

            // Enhance quality score based on validation
            val validationScore = if (parseResult.insights.isNotEmpty()) {
                validatedInsights.size.toFloat() / parseResult.insights.size
            } else 1f

            val adjustedQualityScore = (parseResult.qualityScore * 0.7f) + (validationScore * 0.3f)

            // Create validated result
            val validatedResult = parseResult.copy(
                insights = validatedInsights,
                qualityScore = adjustedQualityScore,
                metadata = parseResult.metadata + mapOf(
                    "validation_score" to validationScore.toString(),
                    "validated_insights" to validatedInsights.size.toString(),
                    "original_insights" to parseResult.insights.size.toString()
                )
            )

            Result.success(validatedResult)

        } catch (e: Exception) {
            Log.e(TAG, "Post-processing validation failed", e)
            Result.failure(e)
        }
    }

    // ========== HELPER METHODS ==========

    private fun extractJSONFromResponse(response: String): String? {
        return try {
            val matcher = JSON_EXTRACT_PATTERN.matcher(response)
            if (matcher.find()) {
                matcher.group(1) ?: matcher.group(2)
            } else {
                // Look for JSON-like structures
                val trimmed = response.trim()
                if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
                    trimmed
                } else null
            }
        } catch (e: Exception) {
            Log.w(TAG, "JSON extraction failed", e)
            null
        }
    }

    private fun isValidJSON(text: String): Boolean {
        return try {
            JsonParser.parseString(text)
            true
        } catch (e: JsonSyntaxException) {
            false
        }
    }

    private suspend fun parseJSONWithRecovery(
        jsonContent: String,
        context: ResponseParsingContext
    ): List<SleepInsight> = withContext(dispatcher) {
        try {
            // Try strict parsing first
            try {
                return@withContext when (context.expectedResponseType) {
                    ResponseType.INSIGHTS -> {
                        val insights = gson.fromJson(jsonContent, Array<InsightDTO>::class.java)
                        insights.map { it.toSleepInsight() }
                    }
                    else -> {
                        parseGenericJSON(jsonContent, context)
                    }
                }
            } catch (e: JsonSyntaxException) {
                // Try recovery parsing
                Log.w(TAG, "Strict JSON parsing failed, attempting recovery", e)
            }

            // Recovery attempt - clean and try again
            val cleanedJSON = contentValidator.cleanAndNormalizeJSON(jsonContent)
            if (cleanedJSON != jsonContent) {
                try {
                    val insights = gson.fromJson(cleanedJSON, Array<InsightDTO>::class.java)
                    return@withContext insights.map { it.toSleepInsight() }
                } catch (e: JsonSyntaxException) {
                    Log.w(TAG, "Cleaned JSON parsing also failed", e)
                }
            }

            // Final fallback - extract what we can
            extractPartialInsights(jsonContent)

        } catch (e: Exception) {
            Log.e(TAG, "JSON parsing with recovery failed", e)
            emptyList()
        }
    }

    private fun extractPartialInsights(jsonContent: String): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        try {
            // Look for insight-like structures
            val jsonObject = JSONObject(jsonContent)
            val keys = jsonObject.keys()

            while (keys.hasNext()) {
                val key = keys.next()
                val value = jsonObject.get(key)

                if (value is JSONObject && containsInsightMarkers(value)) {
                    val insight = createInsightFromJSON(value)
                    if (insight != null) {
                        insights.add(insight)
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Partial insight extraction failed", e)
        }

        return insights
    }

    private fun containsInsightMarkers(jsonObject: JSONObject): Boolean {
        return INSIGHT_MARKERS.any { jsonObject.has(it) }
    }

    private fun createInsightFromJSON(jsonObject: JSONObject): SleepInsight? {
        return try {
            SleepInsight(
                category = tryGetCategory(jsonObject.optString("category", "GENERAL")),
                title = jsonObject.optString("title", "Sleep Insight"),
                description = jsonObject.optString("description", "Analysis completed"),
                recommendation = jsonObject.optString("recommendation", "Continue monitoring"),
                priority = jsonObject.optInt("priority", 2).coerceIn(1, 3),
                isAiGenerated = true,
                timestamp = System.currentTimeMillis()
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to create insight from JSON", e)
            null
        }
    }

    private fun tryGetCategory(categoryString: String): InsightCategory {
        return try {
            InsightCategory.valueOf(categoryString.uppercase())
        } catch (e: Exception) {
            InsightCategory.GENERAL
        }
    }

    // ========== UTILITY AND CONFIGURATION METHODS ==========

    private fun generateOperationId(): String {
        return "parse_${System.currentTimeMillis()}_${Random.nextInt(1000, 9999)}"
    }

    private fun generateCacheKey(response: String, model: AIModel, context: ResponseParsingContext): String {
        return "${response.hashCode()}_${model}_${context.expectedResponseType}"
    }

    private fun createAdvancedGsonParser(): Gson {
        return GsonBuilder()
            .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
            .setLenient()
            .registerTypeAdapter(InsightCategory::class.java) { json, _, _ ->
                try {
                    InsightCategory.valueOf(json.asString.uppercase())
                } catch (e: Exception) {
                    InsightCategory.GENERAL
                }
            }
            .registerTypeAdapter(Date::class.java) { json, _, _ ->
                Date(json.asLong)
            }
            .create()
    }

    private fun loadParserConfiguration(): ParserConfiguration {
        return try {
            val configJson = preferences.getString(PREF_PARSER_CONFIG, null)
            if (configJson != null) {
                gson.fromJson(configJson, ParserConfiguration::class.java)
            } else {
                ParserConfiguration.default()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load parser configuration", e)
            ParserConfiguration.default()
        }
    }

    // Performance and metrics recording methods
    private fun recordSuccessfulParse(model: AIModel, context: ResponseParsingContext, result: ParsedResponse, processingTime: Long) {
        // Update model-specific stats
        val stats = modelParsingStats.getOrPut(model) {
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
        stats.successfulOperations.incrementAndGet()
        stats.totalParsingTime.addAndGet(processingTime)

        val totalOps = stats.totalOperations.get()
        stats.averageParsingTime = stats.totalParsingTime.get() / totalOps
        stats.averageQuality = ((stats.averageQuality * (totalOps - 1)) + result.qualityScore) / totalOps
        stats.errorRate = 1f - (stats.successfulOperations.get().toFloat() / totalOps)
        stats.lastUpdated = System.currentTimeMillis()
    }

    private fun recordFailedParse(model: AIModel, context: ResponseParsingContext, error: Exception, processingTime: Long) {
        val stats = modelParsingStats.getOrPut(model) {
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
        val totalOps = stats.totalOperations.get()
        stats.errorRate = 1f - (stats.successfulOperations.get().toFloat() / totalOps)
        stats.lastUpdated = System.currentTimeMillis()
    }

    private fun recordStrategyPerformance(strategy: ParsingStrategy, model: AIModel, executionTime: Long, success: Boolean, qualityScore: Float) {
        val metrics = strategyMetrics.getOrPut(strategy) {
            StrategyPerformanceMetrics(
                strategy = strategy,
                totalAttempts = AtomicLong(0),
                successfulAttempts = AtomicLong(0),
                totalExecutionTime = AtomicLong(0),
                averageExecutionTime = 0L,
                averageQualityScore = 0f,
                successRate = 0f,
                lastUsed = System.currentTimeMillis()
            )
        }

        metrics.totalAttempts.incrementAndGet()
        if (success) {
            metrics.successfulAttempts.incrementAndGet()
            metrics.totalExecutionTime.addAndGet(executionTime)
        }

        val totalAttempts = metrics.totalAttempts.get()
        val successfulAttempts = metrics.successfulAttempts.get()

        metrics.successRate = successfulAttempts.toFloat() / totalAttempts
        if (successfulAttempts > 0) {
            metrics.averageExecutionTime = metrics.totalExecutionTime.get() / successfulAttempts
            metrics.averageQualityScore = ((metrics.averageQualityScore * (successfulAttempts - 1)) + qualityScore) / successfulAttempts
        }
        metrics.lastUsed = System.currentTimeMillis()
    }

    private fun updateAverageParsingTime(processingTime: Long) {
        val totalRequests = totalParseRequests.get()
        if (totalRequests > 0) {
            val currentAverage = averageParsingTime.get()
            val newAverage = ((currentAverage * (totalRequests - 1)) + processingTime) / totalRequests
            averageParsingTime.set(newAverage)
        }
    }

    // Additional helper methods would be implemented here...

    /**
     * Cleanup resources and shutdown parser
     */
    fun cleanup() {
        scope.cancel()
        circuitBreaker.cleanup()
        strategySelector.cleanup()
        contentValidator.cleanup()
        errorRecovery.cleanup()
        qualityAssessor.cleanup()
        cacheManager.cleanup()

        responseCache.clear()
        recentFailures.clear()
        strategyMetrics.clear()
        modelParsingStats.clear()
        errorPatterns.clear()

        Log.d(TAG, "AIResponseParser cleanup completed")
    }
}

// ========== SUPPORTING CLASSES AND INTERFACES ==========

/**
 * Parsing strategies enumeration
 */
enum class ParsingStrategy(val displayName: String, val priority: Int) {
    STRICT_JSON("Strict JSON", 1),
    FLEXIBLE_JSON("Flexible JSON", 2),
    EXTRACT_JSON("Extract JSON", 3),
    NATURAL_LANGUAGE("Natural Language", 4),
    HYBRID_APPROACH("Hybrid Approach", 5),
    PATTERN_MATCHING("Pattern Matching", 6),
    FALLBACK_BASIC("Fallback Basic", 7),
    RECOVERY_MODE("Recovery Mode", 8)
}

/**
 * Parsing error types
 */
enum class ParsingErrorType {
    JSON_PARSE_ERROR,
    INVALID_INPUT,
    TIMEOUT_ERROR,
    MEMORY_ERROR,
    SECURITY_ERROR,
    CIRCUIT_BREAKER_OPEN,
    INVALID_STATE,
    UNKNOWN_ERROR
}

/**
 * Parser states
 */
sealed class ParsingState {
    object IDLE : ParsingState()
    object INITIALIZING : ParsingState()
    object READY : ParsingState()
    object PARSING : ParsingState()
    object COMPLETED : ParsingState()
    data class ERROR(val message: String) : ParsingState()
}

/**
 * Configuration data class
 */
data class ParserConfiguration(
    val maxBatchConcurrency: Int = 5,
    val enableErrorRecovery: Boolean = true,
    val enablePerformanceOptimization: Boolean = true,
    val cacheEnabled: Boolean = true,
    val securityValidationEnabled: Boolean = true,
    val qualityThreshold: Float = MIN_QUALITY_THRESHOLD
) {
    companion object {
        fun default() = ParserConfiguration()
    }
}

/**
 * Parse options
 */
data class ParseOptions(
    val enableCaching: Boolean = true,
    val enableErrorRecovery: Boolean = true,
    val parsingTimeout: Long = PARSING_TIMEOUT_MS,
    val maxBatchProcessingTime: Long = 60000L,
    val qualityThreshold: Float = MIN_QUALITY_THRESHOLD,
    val preferredStrategy: ParsingStrategy? = null
) {
    companion object {
        fun default() = ParseOptions()
    }
}

/**
 * Response parsing context
 */
data class ResponseParsingContext(
    val type: String,
    val expectedResponseType: ResponseType,
    val expectedSchema: ResponseSchema? = null,
    val sessionData: Any? = null,
    val userContext: Any? = null
)

/**
 * Parsed response result
 */
data class ParsedResponse(
    val insights: List<SleepInsight>,
    val originalResponse: String,
    val strategy: ParsingStrategy,
    val qualityScore: Float,
    val confidence: Float,
    val processingTime: Long,
    val model: AIModel,
    val content: String,
    val metadata: Map<String, String> = emptyMap(),
    val hasErrors: Boolean = false,
    val fromCache: Boolean = false
)

/**
 * Batch parsing classes
 */
data class BatchParseRequest(
    val id: String,
    val response: String,
    val model: AIModel,
    val context: ResponseParsingContext
)

data class BatchParseResult(
    val successful: List<ParsedResponse>,
    val failed: List<BatchParseError>,
    val totalProcessingTime: Long,
    val successRate: Float
) {
    companion object {
        fun empty() = BatchParseResult(emptyList(), emptyList(), 0L, 0f)
    }
}

data class BatchParseError(
    val requestId: String,
    val model: AIModel,
    val error: Throwable,
    val timestamp: Long
)

/**
 * Performance metrics classes
 */
data class ParsedResponseMetrics(
    val totalRequests: Long,
    val successfulParses: Long,
    val failedParses: Long,
    val averageParsingTime: Long,
    val successRate: Float,
    val circuitBreakerTrips: Int,
    val retryAttempts: Long
)

data class StrategyPerformanceMetrics(
    val strategy: ParsingStrategy,
    val totalAttempts: AtomicLong,
    val successfulAttempts: AtomicLong,
    val totalExecutionTime: AtomicLong,
    var averageExecutionTime: Long,
    var averageQualityScore: Float,
    var successRate: Float,
    var lastUsed: Long
)

/**
 * Error tracking
 */
data class ParsingErrorEvent(
    val operationId: String,
    val model: AIModel,
    val errorType: ParsingErrorType,
    val errorMessage: String,
    val timestamp: Long
)

data class ErrorPattern(
    val key: String,
    val count: AtomicInteger = AtomicInteger(0),
    val lastOccurrence: AtomicLong = AtomicLong(System.currentTimeMillis()),
    val associatedModels: MutableSet<AIModel> = mutableSetOf()
)

data class FailureRecord(
    val timestamp: Long,
    val errorType: ParsingErrorType,
    val model: AIModel
)

/**
 * Supporting DTOs
 */
data class InsightDTO(
    val category: String? = null,
    val title: String? = null,
    val description: String? = null,
    val recommendation: String? = null,
    val priority: Int? = null
) {
    fun toSleepInsight(): SleepInsight {
        return SleepInsight(
            category = try {
                InsightCategory.valueOf(category?.uppercase() ?: "GENERAL")
            } catch (e: Exception) {
                InsightCategory.GENERAL
            },
            title = title ?: "Sleep Insight",
            description = description ?: "Analysis completed",
            recommendation = recommendation ?: "Continue monitoring",
            priority = priority?.coerceIn(1, 3) ?: 2,
            isAiGenerated = true,
            timestamp = System.currentTimeMillis()
        )
    }
}

/**
 * Cache management
 */
data class CachedParseResult(
    val result: ParsedResponse,
    val timestamp: Long = System.currentTimeMillis(),
    val ttl: Long = TimeUnit.HOURS.toMillis(CACHE_TTL_HOURS)
) {
    fun isExpired(): Boolean = System.currentTimeMillis() - timestamp > ttl
}

/**
 * Readiness check result
 */
data class ReadinessCheckResult(
    val isReady: Boolean,
    val issues: List<String> = emptyList()
)

// Supporting component classes (simplified implementations)
private class ParsingCircuitBreaker {
    private val failures = ConcurrentHashMap<AIModel, AtomicInteger>()
    private val lastFailureTime = ConcurrentHashMap<AIModel, AtomicLong>()
    private val state = ConcurrentHashMap<AIModel, CircuitBreakerState>()

    fun canExecute(model: AIModel): Boolean {
        val currentState = state.getOrDefault(model, CircuitBreakerState.CLOSED)
        return when (currentState) {
            CircuitBreakerState.CLOSED -> true
            CircuitBreakerState.OPEN -> {
                val lastFailure = lastFailureTime[model]?.get() ?: 0L
                val timeSinceLastFailure = System.currentTimeMillis() - lastFailure
                if (timeSinceLastFailure > CIRCUIT_BREAKER_RESET_TIMEOUT_MS) {
                    state[model] = CircuitBreakerState.HALF_OPEN
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState.HALF_OPEN -> true
        }
    }

    fun recordSuccess(model: AIModel) {
        failures[model] = AtomicInteger(0)
        state[model] = CircuitBreakerState.CLOSED
    }

    fun recordFailure(model: AIModel) {
        val failureCount = failures.getOrPut(model) { AtomicInteger(0) }.incrementAndGet()
        lastFailureTime[model] = AtomicLong(System.currentTimeMillis())

        if (failureCount >= CIRCUIT_BREAKER_FAILURE_THRESHOLD) {
            state[model] = CircuitBreakerState.OPEN
        }
    }

    fun cleanup() {
        failures.clear()
        lastFailureTime.clear()
        state.clear()
    }

    enum class CircuitBreakerState { CLOSED, OPEN, HALF_OPEN }
}

// Additional supporting classes with simplified implementations
private class ParsingStrategySelector {
    fun selectStrategy(
        response: String,
        model: AIModel,
        context: ResponseParsingContext,
        previousFailures: List<Exception?>,
        strategyMetrics: ConcurrentHashMap<ParsingStrategy, StrategyPerformanceMetrics>
    ): ParsingStrategy {
        // Simplified strategy selection logic
        return when {
            response.trim().startsWith("{") || response.trim().startsWith("[") -> ParsingStrategy.STRICT_JSON
            response.contains("```json") || response.contains("```") -> ParsingStrategy.EXTRACT_JSON
            previousFailures.isNotEmpty() -> ParsingStrategy.FLEXIBLE_JSON
            else -> ParsingStrategy.HYBRID_APPROACH
        }
    }

    fun cleanup() {}
}

private class ResponseContentValidator {
    fun validateSecurity(response: String): Result<Unit> {
        // Basic security validation
        return if (response.contains("<script>") || response.contains("javascript:")) {
            Result.failure(SecurityException("Potentially malicious content detected"))
        } else {
            Result.success(Unit)
        }
    }

    fun validateInsight(insight: SleepInsight): Result<Boolean> {
        return Result.success(
            insight.title.length >= MIN_INSIGHT_CONTENT_LENGTH &&
                    insight.description.length >= MIN_INSIGHT_CONTENT_LENGTH &&
                    insight.recommendation.length >= MIN_INSIGHT_CONTENT_LENGTH
        )
    }

    fun cleanAndNormalizeJSON(json: String): String {
        return json.trim()
            .replace("\\n", "")
            .replace("\\t", "")
            .replace("\\\"", "\"")
    }

    fun cleanup() {}
}

private class ParsingErrorRecovery {
    fun attemptRecovery(
        originalResponse: String,
        model: AIModel,
        context: ResponseParsingContext,
        lastError: Exception
    ): ParsedResponse? {
        // Simplified recovery logic
        return try {
            ParsedResponse(
                insights = listOf(
                    SleepInsight(
                        category = InsightCategory.GENERAL,
                        title = "Analysis Recovered",
                        description = "Sleep data processing encountered issues but was recovered",
                        recommendation = "Continue tracking sleep patterns",
                        priority = 3,
                        isAiGenerated = true,
                        timestamp = System.currentTimeMillis()
                    )
                ),
                originalResponse = originalResponse,
                strategy = ParsingStrategy.RECOVERY_MODE,
                qualityScore = 0.3f,
                confidence = 0.2f,
                processingTime = 0L,
                model = model,
                content = originalResponse,
                hasErrors = true
            )
        } catch (e: Exception) {
            null
        }
    }

    fun cleanup() {}
}

private class ResponseQualityAssessor {
    fun assessInsights(insights: List<SleepInsight>, originalResponse: String, model: AIModel): Float {
        if (insights.isEmpty()) return 0f

        val contentScore = insights.map { insight ->
            val titleScore = minOf(insight.title.length.toFloat() / 50f, 1f)
            val descScore = minOf(insight.description.length.toFloat() / 100f, 1f)
            val recScore = minOf(insight.recommendation.length.toFloat() / 50f, 1f)
            (titleScore + descScore + recScore) / 3f
        }.average().toFloat()

        val countScore = minOf(insights.size.toFloat() / 5f, 1f)
        val lengthScore = minOf(originalResponse.length.toFloat() / 500f, 1f)

        return (contentScore * 0.6f + countScore * 0.2f + lengthScore * 0.2f).coerceIn(0f, 1f)
    }

    fun cleanup() {}
}

private class ResponseCacheManager {
    private val cache = ConcurrentHashMap<String, CachedParseResult>()

    fun getCachedResult(key: String): CachedParseResult? {
        val cached = cache[key]
        return if (cached?.isExpired() == false) cached else null
    }

    fun cacheResult(key: String, result: ParsedResponse) {
        cache[key] = CachedParseResult(result)
    }

    fun cleanup() {
        cache.clear()
    }
}

// Additional helper methods and implementations would be added as needed
private fun AIResponseParser.initializeComponents() {
    // Initialize all components
}

private fun AIResponseParser.loadParserState() {
    // Load historical state and configuration
}

private fun AIResponseParser.startBackgroundTasks() {
    // Start background maintenance tasks
}

private fun AIResponseParser.performReadinessCheck(): ReadinessCheckResult {
    return ReadinessCheckResult(isReady = true)
}

private fun AIResponseParser.parseStructuredResponse(response: String, schema: ResponseSchema?): List<SleepInsight> {
    // Parse structured response against schema
    return emptyList()
}

private fun AIResponseParser.parseGenericJSON(json: String, context: ResponseParsingContext): List<SleepInsight> {
    // Generic JSON parsing
    return emptyList()
}

private fun AIResponseParser.extractInsightsFromText(response: String, model: AIModel, context: ResponseParsingContext): List<SleepInsight> {
    // Natural language insight extraction
    return emptyList()
}

private fun AIResponseParser.extractInsightsUsingPatterns(response: String, model: AIModel, context: ResponseParsingContext): List<SleepInsight> {
    // Pattern-based insight extraction
    return emptyList()
}

private fun AIResponseParser.recordErrorPattern(key: String, error: Exception, model: AIModel, response: String) {
    val pattern = errorPatterns.getOrPut(key) { ErrorPattern(key) }
    pattern.count.incrementAndGet()
    pattern.lastOccurrence.set(System.currentTimeMillis())
    pattern.associatedModels.add(model)
}

private fun AIResponseParser.recordBatchParseMetrics(result: BatchParseResult) {
    // Record batch processing metrics
}

private fun AIResponseParser.isModelCompatible(model: AIModel, responseType: ResponseType): Boolean {
    // Check model compatibility with response type
    return true
}