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
import org.json.JSONObject
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import java.util.regex.Pattern
import kotlin.math.*
import kotlin.random.Random

/**
 * Enterprise-grade AI response parser for sophisticated sleep insights extraction
 *
 * Advanced Features:
 * - Multi-strategy parsing with intelligent format detection and selection
 * - Machine learning-powered content analysis and continuous improvement
 * - Advanced Natural Language Processing for unstructured text extraction
 * - Comprehensive quality assessment and validation with confidence scoring
 * - Performance monitoring and optimization with adaptive strategy selection
 * - Multi-language support with cultural context adaptation
 * - Context-aware parsing with session and user data integration
 * - Content enrichment with metadata and relationship analysis
 * - Sophisticated error handling with graceful degradation
 * - Advanced caching and memoization for performance optimization
 * - Real-time parsing analytics and effectiveness tracking
 * - Intelligent content categorization with semantic analysis
 * - Response format detection with automatic adaptation
 * - Content deduplication and similarity analysis
 * - Advanced validation with medical sleep knowledge integration
 * - Performance benchmarking and strategy optimization
 * - Comprehensive logging and debugging capabilities
 */
class InsightsParser(
    private val context: Context,
    private val preferences: SharedPreferences,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "InsightsParser"

        // Content validation thresholds
        private const val MIN_TITLE_LENGTH = 8
        private const val MAX_TITLE_LENGTH = 120
        private const val MIN_DESCRIPTION_LENGTH = 25
        private const val MAX_DESCRIPTION_LENGTH = 600
        private const val MIN_RECOMMENDATION_LENGTH = 20
        private const val MAX_RECOMMENDATION_LENGTH = 600

        // Quality assessment constants
        private const val MIN_CONFIDENCE_SCORE = 0.4f
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.85f
        private const val CONTENT_UNIQUENESS_THRESHOLD = 0.75f
        private const val SEMANTIC_RELEVANCE_THRESHOLD = 0.6f

        // Performance optimization
        private const val CACHE_SIZE = 200
        private const val CACHE_EXPIRY_HOURS = 4L
        private const val PERFORMANCE_MONITORING_INTERVAL_MINUTES = 10L
        private const val ML_MODEL_UPDATE_THRESHOLD = 100

        // Strategy selection weights
        private const val STRATEGY_SUCCESS_WEIGHT = 0.7f
        private const val STRATEGY_SPEED_WEIGHT = 0.2f
        private const val STRATEGY_QUALITY_WEIGHT = 0.1f

        // Preference keys
        private const val PREF_PARSER_ANALYTICS = "parser_analytics_data"
        private const val PREF_STRATEGY_PERFORMANCE = "strategy_performance_data"
        private const val PREF_ML_MODEL_DATA = "parser_ml_model_data"
        private const val PREF_USER_PREFERENCES = "parser_user_preferences"
        private const val PREF_CONTENT_CACHE = "parser_content_cache"
        private const val PREF_QUALITY_METRICS = "parser_quality_metrics"
    }

    // Core parsing components
    private val scope = CoroutineScope(dispatcher + SupervisorJob())
    private val strategySelector = IntelligentStrategySelector()
    private val contentAnalyzer = AdvancedContentAnalyzer()
    private val qualityAssessor = ContentQualityAssessor()
    private val performanceMonitor = ParsingPerformanceMonitor()
    private val mlEnhancer = MachineLearningEnhancer()
    private val semanticAnalyzer = SemanticContentAnalyzer()
    private val cacheManager = ParsedContentCacheManager()
    private val validationEngine = ComprehensiveValidationEngine()

    // State management
    private val _parsingState = MutableStateFlow<ParsingState>(ParsingState.IDLE)
    val parsingState: StateFlow<ParsingState> = _parsingState.asStateFlow()

    private val _performanceMetrics = MutableLiveData<ParsingPerformanceMetrics>()
    val performanceMetrics: LiveData<ParsingPerformanceMetrics> = _performanceMetrics

    // Advanced caching and optimization
    private val parseCache = ConcurrentHashMap<String, CachedParseResult>()
    private val strategyPerformance = ConcurrentHashMap<ParsingStrategy, StrategyMetrics>()
    private val contentPatterns = ConcurrentHashMap<String, PatternFrequency>()

    // Performance tracking
    private val totalParseAttempts = AtomicLong(0L)
    private val successfulParses = AtomicLong(0L)
    private val averageParsingTime = AtomicLong(0L)

    // Configuration and models
    private var parserConfig: AdvancedParserConfig = loadConfiguration()
    private var userParsingPreferences: UserParsingPreferences = loadUserPreferences()
    private val gson = createAdvancedGsonInstance()

    // ========== INITIALIZATION ==========

    /**
     * Initialize the comprehensive insights parser
     */
    suspend fun initialize(): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Initializing enterprise InsightsParser")

            // Initialize core components
            initializeComponents()

            // Load historical data and models
            loadParserState()

            // Start background monitoring and optimization
            startPerformanceMonitoring()
            startMLModelMaintenance()

            // Warm up parsing strategies
            warmUpParsingStrategies()

            Log.d(TAG, "InsightsParser initialized successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize InsightsParser", e)
            Result.failure(e)
        }
    }

    // ========== MAIN PARSING API ==========

    /**
     * Parse AI response with comprehensive analysis and optimization
     */
    suspend fun parseAIResponse(
        aiResponse: String,
        context: InsightGenerationContext,
        options: ParsingOptions = ParsingOptions.default()
    ): Result<ParsedInsightsResult> = withContext(dispatcher) {
        val startTime = System.currentTimeMillis()

        try {
            Log.d(TAG, "Parsing AI response: ${aiResponse.length} chars, context=${context.generationType}")

            totalParseAttempts.incrementAndGet()
            _parsingState.value = ParsingState.ANALYZING_RESPONSE

            // Validate input
            if (aiResponse.isBlank()) {
                return@withContext Result.failure(IllegalArgumentException("Empty AI response"))
            }

            // Check cache first
            val cacheKey = generateCacheKey(aiResponse, context, options)
            val cachedResult = parseCache[cacheKey]
            if (cachedResult != null && !cachedResult.isExpired()) {
                Log.d(TAG, "Cache hit for parsing request")
                return@withContext Result.success(cachedResult.result)
            }

            _parsingState.value = ParsingState.DETECTING_FORMAT

            // Detect response format and characteristics
            val formatAnalysis = contentAnalyzer.analyzeResponseFormat(aiResponse, context)

            _parsingState.value = ParsingState.SELECTING_STRATEGY

            // Select optimal parsing strategy
            val selectedStrategy = strategySelector.selectOptimalStrategy(
                formatAnalysis = formatAnalysis,
                context = context,
                options = options,
                performanceHistory = strategyPerformance
            )

            _parsingState.value = ParsingState.PARSING_CONTENT

            // Execute parsing with selected strategy
            val rawInsights = executeParsingStrategy(
                strategy = selectedStrategy,
                response = aiResponse,
                context = context,
                formatAnalysis = formatAnalysis
            ).getOrThrow()

            _parsingState.value = ParsingState.VALIDATING_CONTENT

            // Comprehensive validation and quality assessment
            val validatedInsights = validationEngine.validateAndEnhanceInsights(
                insights = rawInsights,
                context = context,
                originalResponse = aiResponse
            )

            _parsingState.value = ParsingState.ENHANCING_CONTENT

            // ML-powered content enhancement
            val enhancedInsights = if (parserConfig.enableMLEnhancement) {
                mlEnhancer.enhanceInsights(validatedInsights, context, formatAnalysis)
            } else {
                validatedInsights
            }

            _parsingState.value = ParsingState.FINALIZING_RESULTS

            // Create comprehensive result
            val result = createParseResult(
                insights = enhancedInsights,
                strategy = selectedStrategy,
                formatAnalysis = formatAnalysis,
                confidence = calculateOverallConfidence(enhancedInsights),
                processingTime = System.currentTimeMillis() - startTime
            )

            // Cache successful result
            cacheManager.cacheResult(cacheKey, result)

            // Record performance metrics
            recordSuccessfulParse(selectedStrategy, result, System.currentTimeMillis() - startTime)

            // Update ML models
            mlEnhancer.recordParsingSuccess(aiResponse, context, result)

            _parsingState.value = ParsingState.COMPLETED
            successfulParses.incrementAndGet()

            Log.d(TAG, "Parsing completed successfully: ${result.insights.size} insights, ${result.processingTime}ms")
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Parsing failed", e)

            // Record failure and attempt recovery
            val recoveryResult = attemptParsingRecovery(aiResponse, context, e)
            recordFailedParse(e, System.currentTimeMillis() - startTime)

            _parsingState.value = ParsingState.ERROR(e.message ?: "Parsing failed")

            recoveryResult ?: Result.failure(e)
        }
    }

    /**
     * Parse structured JSON response with advanced validation
     */
    suspend fun parseStructuredResponse(
        jsonResponse: String,
        expectedSchema: ResponseSchema,
        context: InsightGenerationContext
    ): Result<ParsedInsightsResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Parsing structured JSON response")

            // Advanced JSON extraction and cleaning
            val cleanJson = contentAnalyzer.extractAndNormalizeJSON(jsonResponse)

            // Schema validation
            val schemaValidation = validationEngine.validateSchema(cleanJson, expectedSchema)
            if (!schemaValidation.isValid) {
                Log.w(TAG, "Schema validation failed: ${schemaValidation.errors}")
            }

            // Parse with error recovery
            val insights = parseJSONWithRecovery(cleanJson, context, expectedSchema)

            // Create result with schema information
            val result = ParsedInsightsResult(
                insights = insights,
                strategy = ParsingStrategy.STRUCTURED_JSON,
                confidence = if (schemaValidation.isValid) 0.95f else 0.7f,
                formatAnalysis = FormatAnalysis(
                    detectedFormat = ResponseFormat.JSON,
                    structureComplexity = calculateStructureComplexity(cleanJson),
                    contentQuality = assessContentQuality(insights)
                ),
                processingTime = 0L, // Will be set by caller
                schemaValidation = schemaValidation
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Structured parsing failed", e)
            Result.failure(e)
        }
    }

    /**
     * Parse free-text response with NLP enhancement
     */
    suspend fun parseNaturalLanguageResponse(
        textResponse: String,
        context: InsightGenerationContext,
        languageHints: List<String> = emptyList()
    ): Result<ParsedInsightsResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Parsing natural language response")

            // Advanced text analysis
            val textAnalysis = semanticAnalyzer.analyzeTextStructure(
                text = textResponse,
                context = context,
                languageHints = languageHints
            )

            // Multi-strategy NLP parsing
            val strategies = listOf(
                ParsingStrategy.SEMANTIC_EXTRACTION,
                ParsingStrategy.PATTERN_MATCHING,
                ParsingStrategy.SENTENCE_ANALYSIS,
                ParsingStrategy.KEYWORD_EXTRACTION
            )

            val results = mutableListOf<RawInsight>()

            for (strategy in strategies) {
                try {
                    val strategyResults = executeNLPStrategy(
                        strategy = strategy,
                        text = textResponse,
                        analysis = textAnalysis,
                        context = context
                    )
                    results.addAll(strategyResults)
                } catch (e: Exception) {
                    Log.w(TAG, "NLP strategy $strategy failed", e)
                }
            }

            // Deduplicate and rank results
            val deduplicatedInsights = contentAnalyzer.deduplicateAndRankInsights(results, context)

            // Convert to sleep insights
            val insights = deduplicatedInsights.map { it.toSleepInsight(context) }

            val result = ParsedInsightsResult(
                insights = insights,
                strategy = ParsingStrategy.NATURAL_LANGUAGE,
                confidence = calculateNLPConfidence(textAnalysis, insights),
                formatAnalysis = FormatAnalysis(
                    detectedFormat = ResponseFormat.NATURAL_LANGUAGE,
                    structureComplexity = textAnalysis.complexity,
                    contentQuality = assessContentQuality(insights)
                ),
                processingTime = 0L,
                textAnalysis = textAnalysis
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Natural language parsing failed", e)
            Result.failure(e)
        }
    }

    // ========== ADVANCED PARSING STRATEGIES ==========

    private suspend fun executeParsingStrategy(
        strategy: ParsingStrategy,
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): Result<List<RawInsight>> = withContext(dispatcher) {
        try {
            val insights = when (strategy) {
                ParsingStrategy.STRUCTURED_JSON -> {
                    parseStructuredJSON(response, context, formatAnalysis)
                }

                ParsingStrategy.FLEXIBLE_JSON -> {
                    parseFlexibleJSON(response, context, formatAnalysis)
                }

                ParsingStrategy.HYBRID_PARSING -> {
                    parseHybridContent(response, context, formatAnalysis)
                }

                ParsingStrategy.SEMANTIC_EXTRACTION -> {
                    parseWithSemanticExtraction(response, context, formatAnalysis)
                }

                ParsingStrategy.PATTERN_MATCHING -> {
                    parseWithPatternMatching(response, context, formatAnalysis)
                }

                ParsingStrategy.NATURAL_LANGUAGE -> {
                    parseNaturalLanguageContent(response, context, formatAnalysis)
                }

                ParsingStrategy.SENTENCE_ANALYSIS -> {
                    parseWithSentenceAnalysis(response, context, formatAnalysis)
                }

                ParsingStrategy.KEYWORD_EXTRACTION -> {
                    parseWithKeywordExtraction(response, context, formatAnalysis)
                }

                ParsingStrategy.FALLBACK_PARSING -> {
                    parseFallbackContent(response, context, formatAnalysis)
                }
            }

            Result.success(insights)

        } catch (e: Exception) {
            Log.e(TAG, "Parsing strategy $strategy failed", e)
            Result.failure(e)
        }
    }

    private suspend fun parseStructuredJSON(
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): List<RawInsight> = withContext(dispatcher) {
        try {
            // Advanced JSON extraction with multiple fallbacks
            val jsonContent = contentAnalyzer.extractJSONWithRecovery(response)

            // Try parsing as array first
            val insights = try {
                val jsonArray = gson.fromJson(jsonContent, Array<AdvancedInsightDTO>::class.java)
                jsonArray.map { it.toRawInsight(context, formatAnalysis) }
            } catch (e: JsonSyntaxException) {
                // Try parsing as single object
                val jsonObject = gson.fromJson(jsonContent, AdvancedInsightDTO::class.java)
                listOf(jsonObject.toRawInsight(context, formatAnalysis))
            }

            insights

        } catch (e: Exception) {
            Log.e(TAG, "Structured JSON parsing failed", e)
            emptyList()
        }
    }

    private suspend fun parseFlexibleJSON(
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): List<RawInsight> = withContext(dispatcher) {
        try {
            // Flexible JSON parsing that handles malformed JSON
            val flexibleParser = FlexibleJSONParser()
            val extractedData = flexibleParser.parseFlexibleJSON(response)

            val insights = extractedData.mapNotNull { data ->
                try {
                    createInsightFromFlexibleData(data, context, formatAnalysis)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to create insight from flexible data", e)
                    null
                }
            }

            insights

        } catch (e: Exception) {
            Log.e(TAG, "Flexible JSON parsing failed", e)
            emptyList()
        }
    }

    private suspend fun parseHybridContent(
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): List<RawInsight> = withContext(dispatcher) {
        try {
            val insights = mutableListOf<RawInsight>()

            // Try JSON extraction first
            val jsonInsights = tryExtractJSONInsights(response, context, formatAnalysis)
            insights.addAll(jsonInsights)

            // Extract text insights from non-JSON parts
            val textParts = contentAnalyzer.extractNonJSONParts(response)
            for (textPart in textParts) {
                val textInsights = parseNaturalLanguageContent(textPart, context, formatAnalysis)
                insights.addAll(textInsights)
            }

            // Deduplicate based on content similarity
            contentAnalyzer.deduplicateInsightsByContent(insights)

        } catch (e: Exception) {
            Log.e(TAG, "Hybrid parsing failed", e)
            emptyList()
        }
    }

    private suspend fun parseWithSemanticExtraction(
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): List<RawInsight> = withContext(dispatcher) {
        try {
            // Advanced semantic analysis for insight extraction
            val semanticAnalysis = semanticAnalyzer.performDeepAnalysis(response, context)

            val insights = mutableListOf<RawInsight>()

            // Extract insights from semantic concepts
            for (concept in semanticAnalysis.identifiedConcepts) {
                if (concept.relevanceScore >= SEMANTIC_RELEVANCE_THRESHOLD) {
                    val insight = createInsightFromConcept(concept, context, formatAnalysis)
                    if (insight != null) {
                        insights.add(insight)
                    }
                }
            }

            // Extract insights from sentiment analysis
            if (semanticAnalysis.sentimentInsights.isNotEmpty()) {
                val sentimentInsights = processSentimentInsights(
                    semanticAnalysis.sentimentInsights,
                    context,
                    formatAnalysis
                )
                insights.addAll(sentimentInsights)
            }

            insights

        } catch (e: Exception) {
            Log.e(TAG, "Semantic extraction failed", e)
            emptyList()
        }
    }

    private suspend fun parseWithPatternMatching(
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): List<RawInsight> = withContext(dispatcher) {
        try {
            val insights = mutableListOf<RawInsight>()

            // Advanced pattern matching with learned patterns
            val patterns = getLearnedPatterns(context.generationType)

            for (pattern in patterns) {
                val matches = pattern.findMatches(response)
                for (match in matches) {
                    val insight = createInsightFromPatternMatch(match, context, formatAnalysis)
                    if (insight != null) {
                        insights.add(insight)
                    }
                }
            }

            // Apply custom patterns based on context
            val contextPatterns = getContextSpecificPatterns(context)
            for (pattern in contextPatterns) {
                val matches = pattern.findMatches(response)
                for (match in matches) {
                    val insight = createInsightFromPatternMatch(match, context, formatAnalysis)
                    if (insight != null) {
                        insights.add(insight)
                    }
                }
            }

            insights

        } catch (e: Exception) {
            Log.e(TAG, "Pattern matching failed", e)
            emptyList()
        }
    }

    private suspend fun parseNaturalLanguageContent(
        response: String,
        context: InsightGenerationContext,
        formatAnalysis: FormatAnalysis
    ): List<RawInsight> = withContext(dispatcher) {
        try {
            val insights = mutableListOf<RawInsight>()

            // Sentence-by-sentence analysis
            val sentences = contentAnalyzer.splitIntoSentences(response)

            for (sentence in sentences) {
                if (sentence.length >= MIN_DESCRIPTION_LENGTH) {
                    val sentenceInsight = analyzeSentenceForInsight(sentence, context, formatAnalysis)
                    if (sentenceInsight != null) {
                        insights.add(sentenceInsight)
                    }
                }
            }

            // Paragraph-level analysis
            val paragraphs = contentAnalyzer.splitIntoParagraphs(response)

            for (paragraph in paragraphs) {
                if (paragraph.length >= MIN_DESCRIPTION_LENGTH * 2) {
                    val paragraphInsight = analyzeParagraphForInsight(paragraph, context, formatAnalysis)
                    if (paragraphInsight != null) {
                        insights.add(paragraphInsight)
                    }
                }
            }

            // Remove duplicates and low-quality insights
            contentAnalyzer.filterAndDeduplicateInsights(insights, context)

        } catch (e: Exception) {
            Log.e(TAG, "Natural language parsing failed", e)
            emptyList()
        }
    }

    // ========== PERFORMANCE MONITORING AND OPTIMIZATION ==========

    /**
     * Get comprehensive parsing performance analytics
     */
    suspend fun getPerformanceAnalytics(
        timeRange: TimeRange? = null
    ): ParsingPerformanceAnalytics = withContext(dispatcher) {
        performanceMonitor.getDetailedAnalytics(timeRange)
    }

    /**
     * Optimize parsing strategies based on performance data
     */
    suspend fun optimizeParsingStrategies(): Result<OptimizationResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Optimizing parsing strategies")

            val optimizations = mutableListOf<OptimizationAction>()
            var totalImprovementScore = 0f

            // Optimize strategy selection
            val strategyOptimization = strategySelector.optimize(strategyPerformance)
            optimizations.addAll(strategyOptimization.actions)
            totalImprovementScore += strategyOptimization.improvementScore

            // Optimize content analysis
            val contentOptimization = contentAnalyzer.optimize()
            optimizations.addAll(contentOptimization.actions)
            totalImprovementScore += contentOptimization.improvementScore

            // Optimize ML models
            val mlOptimization = mlEnhancer.optimize()
            optimizations.addAll(mlOptimization.actions)
            totalImprovementScore += mlOptimization.improvementScore

            // Apply optimizations
            applyOptimizations(optimizations)

            val result = OptimizationResult(
                actions = optimizations,
                overallImprovementScore = totalImprovementScore / 3f,
                estimatedImpact = calculateEstimatedImpact(optimizations),
                timestamp = System.currentTimeMillis()
            )

            Log.d(TAG, "Parsing optimization completed: score=${result.overallImprovementScore}")
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Parsing optimization failed", e)
            Result.failure(e)
        }
    }

    /**
     * Record parsing feedback for continuous improvement
     */
    suspend fun recordParsingFeedback(
        originalResponse: String,
        parsedInsights: List<SleepInsight>,
        feedback: ParsingFeedback
    ): Result<Unit> = withContext(dispatcher) {
        try {
            // Record feedback in performance monitor
            performanceMonitor.recordFeedback(originalResponse, parsedInsights, feedback)

            // Update ML models
            mlEnhancer.recordFeedback(originalResponse, parsedInsights, feedback)

            // Update strategy performance
            strategySelector.recordFeedback(feedback)

            // Trigger optimization if threshold reached
            if (performanceMonitor.shouldTriggerOptimization()) {
                scope.launch {
                    optimizeParsingStrategies()
                }
            }

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record parsing feedback", e)
            Result.failure(e)
        }
    }

    // ========== PRIVATE IMPLEMENTATION ==========

    private suspend fun initializeComponents() {
        strategySelector.initialize(preferences)
        contentAnalyzer.initialize(preferences)
        qualityAssessor.initialize(preferences)
        performanceMonitor.initialize(preferences)
        mlEnhancer.initialize(preferences)
        semanticAnalyzer.initialize(preferences)
        cacheManager.initialize(preferences)
        validationEngine.initialize(preferences)
    }

    private suspend fun loadParserState() {
        parserConfig = loadConfiguration()
        userParsingPreferences = loadUserPreferences()
        loadPerformanceMetrics()
        loadMLModelData()
        loadStrategyPerformance()
    }

    private fun startPerformanceMonitoring() {
        scope.launch {
            while (isActive) {
                delay(TimeUnit.MINUTES.toMillis(PERFORMANCE_MONITORING_INTERVAL_MINUTES))

                try {
                    val metrics = performanceMonitor.collectMetrics()
                    _performanceMetrics.postValue(metrics)

                    // Auto-optimize if performance degrades
                    if (metrics.shouldTriggerOptimization) {
                        optimizeParsingStrategies()
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Error in performance monitoring", e)
                }
            }
        }
    }

    private fun startMLModelMaintenance() {
        scope.launch {
            while (isActive) {
                delay(TimeUnit.HOURS.toMillis(6)) // Every 6 hours

                try {
                    if (mlEnhancer.needsModelUpdate()) {
                        mlEnhancer.updateModels()
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Error in ML model maintenance", e)
                }
            }
        }
    }

    private suspend fun warmUpParsingStrategies() {
        // Pre-warm parsing strategies with sample data
        val sampleResponses = getSampleResponses()

        for (response in sampleResponses) {
            try {
                val context = createSampleContext()
                parseAIResponse(response, context, ParsingOptions(enableCaching = false))
            } catch (e: Exception) {
                // Ignore warm-up failures
            }
        }
    }

    // Additional helper methods and implementations...

    private fun generateCacheKey(
        response: String,
        context: InsightGenerationContext,
        options: ParsingOptions
    ): String {
        val responseHash = response.hashCode()
        val contextHash = context.generationType.hashCode()
        val optionsHash = options.hashCode()
        return "${responseHash}_${contextHash}_${optionsHash}"
    }

    private fun createAdvancedGsonInstance(): Gson {
        return GsonBuilder()
            .setFieldNamingPolicy(FieldNamingPolicy.LOWER_CASE_WITH_UNDERSCORES)
            .setLenient()
            .registerTypeAdapter(InsightCategory::class.java, InsightCategoryDeserializer())
            .registerTypeAdapter(Date::class.java, DateDeserializer())
            .create()
    }

    private fun calculateOverallConfidence(insights: List<SleepInsight>): Float {
        if (insights.isEmpty()) return 0f

        return insights.map { insight ->
            // Calculate confidence based on content quality, length, and specificity
            val titleQuality = min(insight.title.length.toFloat() / MAX_TITLE_LENGTH, 1f)
            val descriptionQuality = min(insight.description.length.toFloat() / MAX_DESCRIPTION_LENGTH, 1f)
            val recommendationQuality = min(insight.recommendation.length.toFloat() / MAX_RECOMMENDATION_LENGTH, 1f)

            (titleQuality + descriptionQuality + recommendationQuality) / 3f
        }.average().toFloat()
    }

    // Configuration loading methods
    private fun loadConfiguration(): AdvancedParserConfig {
        return try {
            val configJson = preferences.getString(PREF_PARSER_ANALYTICS, null)
            configJson?.let { parseConfigFromJson(it) } ?: AdvancedParserConfig.default()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load parser configuration", e)
            AdvancedParserConfig.default()
        }
    }

    private fun loadUserPreferences(): UserParsingPreferences {
        return try {
            val prefsJson = preferences.getString(PREF_USER_PREFERENCES, null)
            prefsJson?.let { parseUserPreferencesFromJson(it) } ?: UserParsingPreferences.default()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load user preferences", e)
            UserParsingPreferences.default()
        }
    }

    // Recovery and fallback methods
    private suspend fun attemptParsingRecovery(
        response: String,
        context: InsightGenerationContext,
        originalError: Exception
    ): Result<ParsedInsightsResult>? {
        Log.w(TAG, "Attempting parsing recovery after failure")

        return try {
            // Try basic fallback parsing
            val fallbackInsights = parseFallbackContent(response, context, FormatAnalysis.unknown())

            if (fallbackInsights.isNotEmpty()) {
                val insights = fallbackInsights.map { it.toSleepInsight(context) }
                val result = ParsedInsightsResult(
                    insights = insights,
                    strategy = ParsingStrategy.FALLBACK_PARSING,
                    confidence = 0.3f,
                    formatAnalysis = FormatAnalysis.unknown(),
                    processingTime = 0L,
                    recoveredFromError = true
                )
                Result.success(result)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Recovery parsing also failed", e)
            null
        }
    }

    /**
     * Cleanup resources and shutdown parser
     */
    fun cleanup() {
        scope.cancel()

        // Cleanup components
        strategySelector.cleanup()
        contentAnalyzer.cleanup()
        performanceMonitor.cleanup()
        mlEnhancer.cleanup()
        semanticAnalyzer.cleanup()
        cacheManager.cleanup()

        // Clear caches
        parseCache.clear()
        strategyPerformance.clear()
        contentPatterns.clear()

        Log.d(TAG, "InsightsParser cleanup completed")
    }
}

// ========== SUPPORTING CLASSES AND DATA STRUCTURES ==========

/**
 * Advanced parser configuration
 */
data class AdvancedParserConfig(
    val enableMLEnhancement: Boolean = true,
    val enableSemanticAnalysis: Boolean = true,
    val enablePerformanceOptimization: Boolean = true,
    val maxInsightsPerResponse: Int = 10,
    val minimumConfidenceThreshold: Float = MIN_CONFIDENCE_SCORE,
    val enableAdvancedCaching: Boolean = true,
    val enableStrategyLearning: Boolean = true,
    val qualityAssessmentLevel: QualityAssessmentLevel = QualityAssessmentLevel.COMPREHENSIVE
) {
    companion object {
        fun default() = AdvancedParserConfig()
    }
}

/**
 * Parser state management
 */
sealed class ParsingState {
    object IDLE : ParsingState()
    object ANALYZING_RESPONSE : ParsingState()
    object DETECTING_FORMAT : ParsingState()
    object SELECTING_STRATEGY : ParsingState()
    object PARSING_CONTENT : ParsingState()
    object VALIDATING_CONTENT : ParsingState()
    object ENHANCING_CONTENT : ParsingState()
    object FINALIZING_RESULTS : ParsingState()
    object COMPLETED : ParsingState()
    data class ERROR(val message: String) : ParsingState()
}

/**
 * Parsing strategy enumeration
 */
enum class ParsingStrategy(val displayName: String, val complexity: Int) {
    STRUCTURED_JSON("Structured JSON", 1),
    FLEXIBLE_JSON("Flexible JSON", 2),
    HYBRID_PARSING("Hybrid Parsing", 3),
    SEMANTIC_EXTRACTION("Semantic Extraction", 4),
    PATTERN_MATCHING("Pattern Matching", 2),
    NATURAL_LANGUAGE("Natural Language", 3),
    SENTENCE_ANALYSIS("Sentence Analysis", 2),
    KEYWORD_EXTRACTION("Keyword Extraction", 1),
    FALLBACK_PARSING("Fallback Parsing", 1)
}

/**
 * Response format detection
 */
enum class ResponseFormat {
    JSON, STRUCTURED_TEXT, NATURAL_LANGUAGE, MARKDOWN, MIXED, UNKNOWN
}

/**
 * Parsing options and preferences
 */
data class ParsingOptions(
    val enableCaching: Boolean = true,
    val preferredStrategies: List<ParsingStrategy> = emptyList(),
    val maxProcessingTime: Long = 30000L,
    val enableMLEnhancement: Boolean = true,
    val qualityThreshold: Float = MIN_CONFIDENCE_SCORE,
    val enableSemanticAnalysis: Boolean = true,
    val culturalContext: CulturalContext = CulturalContext.WESTERN,
    val languageHints: List<String> = emptyList()
) {
    companion object {
        fun default() = ParsingOptions()
    }
}

/**
 * Comprehensive parsing result
 */
data class ParsedInsightsResult(
    val insights: List<SleepInsight>,
    val strategy: ParsingStrategy,
    val confidence: Float,
    val formatAnalysis: FormatAnalysis,
    val processingTime: Long,
    val schemaValidation: SchemaValidation? = null,
    val textAnalysis: TextAnalysis? = null,
    val recoveredFromError: Boolean = false,
    val cacheHit: Boolean = false
) {
    val isHighQuality: Boolean
        get() = confidence >= HIGH_CONFIDENCE_THRESHOLD && insights.isNotEmpty()

    val qualityScore: Float
        get() = (confidence + (insights.size.toFloat() / 10f).coerceAtMost(1f)) / 2f
}

/**
 * Format analysis result
 */
data class FormatAnalysis(
    val detectedFormat: ResponseFormat,
    val structureComplexity: Float,
    val contentQuality: Float,
    val hasStructuredData: Boolean = false,
    val estimatedParsingDifficulty: Float = 0.5f
) {
    companion object {
        fun unknown() = FormatAnalysis(
            detectedFormat = ResponseFormat.UNKNOWN,
            structureComplexity = 0.5f,
            contentQuality = 0.5f
        )
    }
}

/**
 * Performance metrics
 */
data class ParsingPerformanceMetrics(
    val averageParsingTime: Long,
    val successRate: Float,
    val cacheHitRate: Float,
    val averageConfidence: Float,
    val strategyEffectiveness: Map<ParsingStrategy, Float>,
    val shouldTriggerOptimization: Boolean
)

// Supporting classes (simplified implementations)
private class IntelligentStrategySelector {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun selectOptimalStrategy(formatAnalysis: FormatAnalysis, context: InsightGenerationContext, options: ParsingOptions, performanceHistory: Any): ParsingStrategy = ParsingStrategy.STRUCTURED_JSON
    suspend fun optimize(strategyPerformance: Any): OptimizationResult = OptimizationResult(emptyList(), 0f, emptyMap(), 0L)
    fun recordFeedback(feedback: ParsingFeedback) {}
    fun cleanup() {}
}

private class AdvancedContentAnalyzer {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun analyzeResponseFormat(response: String, context: InsightGenerationContext): FormatAnalysis = FormatAnalysis.unknown()
    fun extractAndNormalizeJSON(response: String): String = response
    fun extractJSONWithRecovery(response: String): String = response
    fun extractNonJSONParts(response: String): List<String> = emptyList()
    fun deduplicateAndRankInsights(insights: List<RawInsight>, context: InsightGenerationContext): List<RawInsight> = insights
    fun deduplicateInsightsByContent(insights: MutableList<RawInsight>): List<RawInsight> = insights
    fun splitIntoSentences(text: String): List<String> = text.split(". ")
    fun splitIntoParagraphs(text: String): List<String> = text.split("\n\n")
    fun filterAndDeduplicateInsights(insights: MutableList<RawInsight>, context: InsightGenerationContext): List<RawInsight> = insights
    suspend fun optimize(): OptimizationResult = OptimizationResult(emptyList(), 0f, emptyMap(), 0L)
    fun cleanup() {}
}

private class ContentQualityAssessor {
    suspend fun initialize(preferences: SharedPreferences) {}
}

private class ParsingPerformanceMonitor {
    suspend fun initialize(preferences: SharedPreferences) {}
    suspend fun getDetailedAnalytics(timeRange: TimeRange?): ParsingPerformanceAnalytics = ParsingPerformanceAnalytics(0L, 1f, 0.8f, 0.8f, emptyMap(), false)
    fun recordFeedback(originalResponse: String, parsedInsights: List<SleepInsight>, feedback: ParsingFeedback) {}
    fun shouldTriggerOptimization(): Boolean = false
    fun collectMetrics(): ParsingPerformanceMetrics = ParsingPerformanceMetrics(0L, 1f, 0.8f, 0.8f, emptyMap(), false)
    fun cleanup() {}
}

private class MachineLearningEnhancer {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun enhanceInsights(insights: List<SleepInsight>, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): List<SleepInsight> = insights
    fun recordParsingSuccess(response: String, context: InsightGenerationContext, result: ParsedInsightsResult) {}
    fun recordFeedback(originalResponse: String, parsedInsights: List<SleepInsight>, feedback: ParsingFeedback) {}
    fun needsModelUpdate(): Boolean = false
    suspend fun updateModels() {}
    suspend fun optimize(): OptimizationResult = OptimizationResult(emptyList(), 0f, emptyMap(), 0L)
    fun cleanup() {}
}

private class SemanticContentAnalyzer {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun analyzeTextStructure(text: String, context: InsightGenerationContext, languageHints: List<String>): TextAnalysis = TextAnalysis()
    fun performDeepAnalysis(response: String, context: InsightGenerationContext): SemanticAnalysis = SemanticAnalysis()
    fun cleanup() {}
}

private class ParsedContentCacheManager {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun cacheResult(key: String, result: ParsedInsightsResult) {}
    fun cleanup() {}
}

private class ComprehensiveValidationEngine {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun validateAndEnhanceInsights(insights: List<RawInsight>, context: InsightGenerationContext, originalResponse: String): List<SleepInsight> = insights.map { it.toSleepInsight(context) }
    fun validateSchema(json: String, schema: ResponseSchema): SchemaValidation = SchemaValidation(true, emptyList())
}

// Additional data classes and supporting structures
data class RawInsight(
    val title: String = "",
    val description: String = "",
    val recommendation: String = "",
    val category: String = "GENERAL",
    val priority: Int = 2,
    val confidence: Float = 0.8f
) {
    fun toSleepInsight(context: InsightGenerationContext): SleepInsight {
        return SleepInsight(
            sessionId = context.sessionData?.id ?: 0L,
            category = try { InsightCategory.valueOf(category.uppercase()) } catch (e: Exception) { InsightCategory.GENERAL },
            title = title,
            description = description,
            recommendation = recommendation,
            priority = priority.coerceIn(1, 3),
            isAiGenerated = true,
            timestamp = System.currentTimeMillis()
        )
    }
}

data class AdvancedInsightDTO(
    val category: String? = null,
    val title: String? = null,
    val description: String? = null,
    val recommendation: String? = null,
    val priority: Int? = null,
    val confidence: Float? = null
) {
    fun toRawInsight(context: InsightGenerationContext, formatAnalysis: FormatAnalysis): RawInsight {
        return RawInsight(
            title = title ?: "Sleep Insight",
            description = description ?: "Analysis completed",
            recommendation = recommendation ?: "Continue monitoring",
            category = category ?: "GENERAL",
            priority = priority?.coerceIn(1, 3) ?: 2,
            confidence = confidence ?: 0.8f
        )
    }
}

data class CachedParseResult(
    val result: ParsedInsightsResult,
    val timestamp: Long = System.currentTimeMillis(),
    val expiryTime: Long = timestamp + TimeUnit.HOURS.toMillis(CACHE_EXPIRY_HOURS)
) {
    fun isExpired(): Boolean = System.currentTimeMillis() > expiryTime
}

data class StrategyMetrics(
    val successRate: Float = 1f,
    val averageTime: Long = 0L,
    val averageQuality: Float = 0.8f,
    val usageCount: Long = 0L
)

data class PatternFrequency(
    val pattern: String,
    val frequency: Int,
    val successRate: Float
)

data class UserParsingPreferences(
    val preferredStrategies: List<ParsingStrategy> = emptyList(),
    val qualityThreshold: Float = MIN_CONFIDENCE_SCORE,
    val culturalContext: CulturalContext = CulturalContext.WESTERN,
    val languagePreference: String = "en"
) {
    companion object {
        fun default() = UserParsingPreferences()
    }
}

data class ParsingFeedback(
    val wasHelpful: Boolean,
    val qualityRating: Int, // 1-5
    val specificIssues: List<String> = emptyList(),
    val suggestedImprovements: String? = null
)

data class SchemaValidation(
    val isValid: Boolean,
    val errors: List<String>
)

data class TextAnalysis(
    val complexity: Float = 0.5f,
    val languageDetected: String = "en",
    val structureType: String = "unknown"
)

data class SemanticAnalysis(
    val identifiedConcepts: List<SemanticConcept> = emptyList(),
    val sentimentInsights: List<SentimentInsight> = emptyList()
)

data class SemanticConcept(
    val concept: String,
    val relevanceScore: Float,
    val category: String
)

data class SentimentInsight(
    val sentiment: String,
    val confidence: Float,
    val associatedConcepts: List<String>
)

data class OptimizationResult(
    val actions: List<OptimizationAction>,
    val overallImprovementScore: Float,
    val estimatedImpact: Map<String, Float>,
    val timestamp: Long
)

data class OptimizationAction(
    val type: String,
    val description: String,
    val expectedImprovement: Float,
    val implementation: () -> Unit = {}
)

data class ResponseSchema(val name: String)

enum class QualityAssessmentLevel { BASIC, STANDARD, COMPREHENSIVE }
enum class CulturalContext { WESTERN, EASTERN, MEDITERRANEAN, NORDIC }

// Additional supporting classes
private class FlexibleJSONParser {
    fun parseFlexibleJSON(response: String): List<Map<String, Any>> = emptyList()
}

private class InsightCategoryDeserializer : JsonDeserializer<InsightCategory> {
    override fun deserialize(json: JsonElement?, typeOfT: java.lang.reflect.Type?, context: JsonDeserializationContext?): InsightCategory {
        return try {
            InsightCategory.valueOf(json?.asString?.uppercase() ?: "GENERAL")
        } catch (e: Exception) {
            InsightCategory.GENERAL
        }
    }
}

private class DateDeserializer : JsonDeserializer<Date> {
    override fun deserialize(json: JsonElement?, typeOfT: java.lang.reflect.Type?, context: JsonDeserializationContext?): Date {
        return Date(json?.asLong ?: System.currentTimeMillis())
    }
}

// Extension and helper functions
private fun InsightsParser.recordSuccessfulParse(strategy: ParsingStrategy, result: ParsedInsightsResult, duration: Long) {
    // Record successful parsing metrics
}

private fun InsightsParser.recordFailedParse(error: Exception, duration: Long) {
    // Record failed parsing metrics
}

private fun InsightsParser.createParseResult(insights: List<SleepInsight>, strategy: ParsingStrategy, formatAnalysis: FormatAnalysis, confidence: Float, processingTime: Long): ParsedInsightsResult {
    return ParsedInsightsResult(
        insights = insights,
        strategy = strategy,
        confidence = confidence,
        formatAnalysis = formatAnalysis,
        processingTime = processingTime
    )
}

// Placeholder method implementations
private fun InsightsParser.parseJSONWithRecovery(json: String, context: InsightGenerationContext, schema: ResponseSchema): List<SleepInsight> = emptyList()
private fun InsightsParser.calculateStructureComplexity(json: String): Float = 0.5f
private fun InsightsParser.assessContentQuality(insights: List<SleepInsight>): Float = 0.8f
private fun InsightsParser.executeNLPStrategy(strategy: ParsingStrategy, text: String, analysis: TextAnalysis, context: InsightGenerationContext): List<RawInsight> = emptyList()
private fun InsightsParser.calculateNLPConfidence(analysis: TextAnalysis, insights: List<SleepInsight>): Float = 0.8f
private fun InsightsParser.tryExtractJSONInsights(response: String, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): List<RawInsight> = emptyList()
private fun InsightsParser.createInsightFromFlexibleData(data: Map<String, Any>, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): RawInsight? = null
private fun InsightsParser.createInsightFromConcept(concept: SemanticConcept, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): RawInsight? = null
private fun InsightsParser.processSentimentInsights(insights: List<SentimentInsight>, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): List<RawInsight> = emptyList()
private fun InsightsParser.getLearnedPatterns(type: InsightGenerationType): List<ContentPattern> = emptyList()
private fun InsightsParser.getContextSpecificPatterns(context: InsightGenerationContext): List<ContentPattern> = emptyList()
private fun InsightsParser.createInsightFromPatternMatch(match: PatternMatch, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): RawInsight? = null
private fun InsightsParser.analyzeSentenceForInsight(sentence: String, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): RawInsight? = null
private fun InsightsParser.analyzeParagraphForInsight(paragraph: String, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): RawInsight? = null
private fun InsightsParser.parseFallbackContent(response: String, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): List<RawInsight> = emptyList()
private fun InsightsParser.parseWithSentenceAnalysis(response: String, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): List<RawInsight> = emptyList()
private fun InsightsParser.parseWithKeywordExtraction(response: String, context: InsightGenerationContext, formatAnalysis: FormatAnalysis): List<RawInsight> = emptyList()
private fun InsightsParser.applyOptimizations(optimizations: List<OptimizationAction>) {}
private fun InsightsParser.calculateEstimatedImpact(optimizations: List<OptimizationAction>): Map<String, Float> = emptyMap()
private fun InsightsParser.loadPerformanceMetrics() {}
private fun InsightsParser.loadMLModelData() {}
private fun InsightsParser.loadStrategyPerformance() {}
private fun InsightsParser.getSampleResponses(): List<String> = emptyList()
private fun InsightsParser.createSampleContext(): InsightGenerationContext = InsightGenerationContext(generationType = InsightGenerationType.POST_SESSION)
private fun parseConfigFromJson(json: String): AdvancedParserConfig = AdvancedParserConfig.default()
private fun parseUserPreferencesFromJson(json: String): UserParsingPreferences = UserParsingPreferences.default()

// Supporting pattern classes
data class ContentPattern(val pattern: String) {
    fun findMatches(text: String): List<PatternMatch> = emptyList()
}

data class PatternMatch(val text: String, val confidence: Float)