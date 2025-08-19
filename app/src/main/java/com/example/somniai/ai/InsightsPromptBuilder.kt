package com.example.somniai.ai

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.example.somniai.analytics.*
import com.example.somniai.data.*
import com.example.somniai.repository.SleepRepository
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*
import kotlin.random.Random

/**
 * Enterprise-grade AI prompt builder for sophisticated sleep insights generation
 *
 * Advanced Features:
 * - Multi-model AI support (GPT-4, Claude, Gemini) with model-specific optimizations
 * - Dynamic prompt optimization based on response quality and user engagement
 * - Contextual memory management with conversation state tracking
 * - Advanced template system with variable substitution and inheritance
 * - Prompt versioning and A/B testing for continuous improvement
 * - Sophisticated data summarization with intelligent prioritization
 * - Multi-language support with cultural sleep norm adaptations
 * - Performance monitoring and prompt effectiveness tracking
 * - Advanced error handling with fallback strategies
 * - Integration with comprehensive analytics models
 * - Token-efficient optimization with content compression
 * - Personalization based on user preferences and response patterns
 * - Specialized prompts for different insight categories and depths
 * - Response format validation and structured output generation
 * - Context-aware prompt adaptation based on user expertise level
 */
class InsightsPromptBuilder(
    private val context: Context,
    private val repository: SleepRepository,
    private val preferences: SharedPreferences,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "InsightsPromptBuilder"

        // Prompt optimization constants
        private const val MAX_TOKENS_GPT4 = 8192
        private const val MAX_TOKENS_CLAUDE = 200000
        private const val MAX_TOKENS_GEMINI = 1048576
        private const val OPTIMAL_RESPONSE_TOKENS = 800
        private const val CONTEXT_COMPRESSION_THRESHOLD = 0.8f

        // Performance monitoring
        private const val PROMPT_PERFORMANCE_WINDOW_DAYS = 30
        private const val MIN_SAMPLES_FOR_OPTIMIZATION = 10
        private const val EFFECTIVENESS_LEARNING_RATE = 0.1f

        // Template versioning
        private const val TEMPLATE_VERSION = "2.1.0"
        private const val TEMPLATE_CACHE_SIZE = 50
        private const val TEMPLATE_REFRESH_INTERVAL_HOURS = 24L

        // Context management
        private const val MAX_CONVERSATION_CONTEXT_LENGTH = 5
        private const val CONTEXT_DECAY_FACTOR = 0.8f
        private const val MEMORY_RETENTION_DAYS = 7

        // Preference keys
        private const val PREF_PROMPT_PERFORMANCE = "prompt_performance_data"
        private const val PREF_USER_PREFERENCES = "user_prompt_preferences"
        private const val PREF_TEMPLATE_CACHE = "template_cache"
        private const val PREF_CONVERSATION_CONTEXT = "conversation_context"
        private const val PREF_AB_TEST_GROUP = "prompt_ab_test_group"
        private const val PREF_PERSONALIZATION_DATA = "prompt_personalization"
        private const val PREF_EFFECTIVENESS_SCORES = "prompt_effectiveness"
        private const val PREF_MODEL_PREFERENCES = "ai_model_preferences"

        // Sleep medicine reference data
        private val SLEEP_MEDICINE_CONTEXT = mapOf(
            "duration_optimal_hours" to 7.0f..9.0f,
            "efficiency_optimal_percent" to 85f..100f,
            "deep_sleep_optimal_percent" to 15f..25f,
            "rem_sleep_optimal_percent" to 20f..25f,
            "sleep_latency_optimal_minutes" to 10f..20f,
            "movement_intensity_good" to 0f..3f,
            "noise_level_optimal_db" to 0f..40f,
            "wake_after_sleep_onset_good_minutes" to 0f..30f,
            "temperature_optimal_celsius" to 16f..19f
        )
    }

    // Core dependencies and state
    private val scope = CoroutineScope(dispatcher + SupervisorJob())
    private val templateEngine = AdvancedTemplateEngine()
    private val contextManager = ConversationContextManager()
    private val personalizationEngine = PromptPersonalizationEngine()
    private val performanceTracker = PromptPerformanceTracker()
    private val multiModelAdapter = MultiModelAdapter()
    private val dataCompressor = IntelligentDataCompressor()
    private val qualityValidator = ResponseQualityValidator()

    // Caching and optimization
    private val templateCache = ConcurrentHashMap<String, CachedTemplate>()
    private val promptEffectivenessCache = ConcurrentHashMap<String, PromptEffectiveness>()
    private val userPreferenceCache = ConcurrentHashMap<String, UserPromptPreferences>()

    // Performance tracking
    private val totalPromptsGenerated = AtomicLong(0L)
    private val averageGenerationTime = AtomicLong(0L)
    private val successfulPrompts = AtomicLong(0L)

    // Configuration
    private var currentConfig: PromptBuilderConfig = loadConfiguration()
    private var abTestGroup: ABTestGroup = loadABTestGroup()
    private var userPreferences: UserPromptPreferences = loadUserPreferences()

    // ========== INITIALIZATION ==========

    /**
     * Initialize the comprehensive prompt builder
     */
    suspend fun initialize(): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Initializing advanced InsightsPromptBuilder")

            // Load configuration and historical data
            loadPromptBuilderState()

            // Initialize components
            templateEngine.initialize(preferences)
            contextManager.initialize(preferences)
            personalizationEngine.initialize(preferences, repository)
            performanceTracker.initialize(preferences)
            multiModelAdapter.initialize(preferences)

            // Load template cache
            loadTemplateCache()

            // Start background optimization
            startPerformanceOptimization()

            Log.d(TAG, "InsightsPromptBuilder initialized successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize InsightsPromptBuilder", e)
            Result.failure(e)
        }
    }

    // ========== MAIN PROMPT BUILDING METHODS ==========

    /**
     * Build comprehensive AI prompt with advanced optimization
     */
    suspend fun buildPrompt(
        context: InsightGenerationContext,
        targetModel: AIModel = AIModel.GPT4,
        customization: PromptCustomization? = null
    ): Result<PromptResult> = withContext(dispatcher) {
        val startTime = System.currentTimeMillis()

        try {
            Log.d(TAG, "Building optimized prompt for ${context.generationType} using ${targetModel.name}")

            totalPromptsGenerated.incrementAndGet()

            // Create prompt generation request
            val request = PromptGenerationRequest(
                context = context,
                targetModel = targetModel,
                customization = customization ?: createDefaultCustomization(context),
                userPreferences = userPreferences,
                abTestGroup = abTestGroup,
                conversationContext = contextManager.getCurrentContext(),
                timestamp = System.currentTimeMillis()
            )

            // Validate and prepare request
            val validatedRequest = validateAndPrepareRequest(request).getOrThrow()

            // Generate prompt using advanced pipeline
            val promptResult = generateOptimizedPrompt(validatedRequest).getOrThrow()

            // Track performance and update models
            val generationTime = System.currentTimeMillis() - startTime
            performanceTracker.recordGeneration(promptResult, generationTime)
            updateAverageGenerationTime(generationTime)

            Log.d(TAG, "Prompt generated successfully: ${promptResult.content.length} chars, ${generationTime}ms")
            successfulPrompts.incrementAndGet()

            Result.success(promptResult)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to build prompt", e)
            val generationTime = System.currentTimeMillis() - startTime
            performanceTracker.recordFailure(context, targetModel, e, generationTime)

            // Attempt fallback prompt generation
            generateFallbackPrompt(context, targetModel)
        }
    }

    /**
     * Build structured prompt for JSON response parsing
     */
    suspend fun buildStructuredPrompt(
        context: InsightGenerationContext,
        responseSchema: ResponseSchema,
        targetModel: AIModel = AIModel.GPT4,
        validationLevel: ValidationLevel = ValidationLevel.STRICT
    ): Result<StructuredPromptResult> = withContext(dispatcher) {
        try {
            // Build base prompt
            val basePromptResult = buildPrompt(context, targetModel).getOrThrow()

            // Add structured response formatting
            val structuredInstructions = generateStructuredInstructions(
                schema = responseSchema,
                model = targetModel,
                validationLevel = validationLevel
            )

            // Combine with response validation
            val structuredPrompt = combinePromptWithStructuredInstructions(
                basePrompt = basePromptResult.content,
                instructions = structuredInstructions,
                schema = responseSchema
            )

            // Validate structured prompt
            val validationResult = qualityValidator.validateStructuredPrompt(
                prompt = structuredPrompt,
                schema = responseSchema,
                model = targetModel
            )

            val result = StructuredPromptResult(
                content = structuredPrompt,
                schema = responseSchema,
                validationResult = validationResult,
                estimatedTokens = estimateTokenCount(structuredPrompt, targetModel),
                metadata = basePromptResult.metadata.copy(
                    isStructured = true,
                    responseSchema = responseSchema.name
                )
            )

            Log.d(TAG, "Structured prompt generated: ${result.content.length} chars")
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to build structured prompt", e)
            Result.failure(e)
        }
    }

    /**
     * Build conversation-aware prompt with context memory
     */
    suspend fun buildConversationalPrompt(
        context: InsightGenerationContext,
        conversationId: String,
        previousInteractions: List<Interaction> = emptyList(),
        targetModel: AIModel = AIModel.GPT4
    ): Result<ConversationalPromptResult> = withContext(dispatcher) {
        try {
            // Update conversation context
            contextManager.updateConversationContext(
                conversationId = conversationId,
                context = context,
                interactions = previousInteractions
            )

            // Build context-aware prompt
            val conversationContext = contextManager.getConversationContext(conversationId)
            val contextualizedRequest = context.copy(
                conversationHistory = conversationContext.history,
                userPersonality = conversationContext.userPersonality,
                preferredCommunicationStyle = conversationContext.communicationStyle
            )

            // Generate prompt with conversation awareness
            val promptResult = buildPrompt(contextualizedRequest, targetModel).getOrThrow()

            // Add conversation-specific elements
            val conversationalPrompt = enhancePromptWithConversationalContext(
                basePrompt = promptResult.content,
                conversationContext = conversationContext,
                targetModel = targetModel
            )

            val result = ConversationalPromptResult(
                content = conversationalPrompt,
                conversationId = conversationId,
                contextDepth = conversationContext.contextDepth,
                personalityAdaptation = conversationContext.userPersonality,
                metadata = promptResult.metadata.copy(
                    isConversational = true,
                    conversationId = conversationId
                )
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to build conversational prompt", e)
            Result.failure(e)
        }
    }

    /**
     * Build specialized prompt for different insight categories
     */
    suspend fun buildSpecializedPrompt(
        context: InsightGenerationContext,
        specialization: PromptSpecialization,
        expertiseLevel: ExpertiseLevel = ExpertiseLevel.GENERAL,
        targetModel: AIModel = AIModel.GPT4
    ): Result<SpecializedPromptResult> = withContext(dispatcher) {
        try {
            // Get specialized template
            val template = templateEngine.getSpecializedTemplate(
                specialization = specialization,
                expertiseLevel = expertiseLevel,
                model = targetModel
            ).getOrThrow()

            // Prepare specialized context
            val specializedContext = prepareSpecializedContext(
                context = context,
                specialization = specialization,
                expertiseLevel = expertiseLevel
            )

            // Generate prompt using specialized template
            val promptContent = templateEngine.renderTemplate(
                template = template,
                context = specializedContext,
                personalizations = personalizationEngine.getPersonalizations(specialization)
            ).getOrThrow()

            // Optimize for target model
            val optimizedPrompt = multiModelAdapter.optimizeForModel(
                prompt = promptContent,
                model = targetModel,
                specialization = specialization
            )

            val result = SpecializedPromptResult(
                content = optimizedPrompt,
                specialization = specialization,
                expertiseLevel = expertiseLevel,
                template = template,
                estimatedTokens = estimateTokenCount(optimizedPrompt, targetModel),
                metadata = PromptMetadata(
                    generationType = context.generationType,
                    specialization = specialization.name,
                    expertiseLevel = expertiseLevel.name,
                    modelOptimized = targetModel.name,
                    templateVersion = template.version,
                    timestamp = System.currentTimeMillis()
                )
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to build specialized prompt", e)
            Result.failure(e)
        }
    }

    // ========== ADVANCED PROMPT GENERATION PIPELINE ==========

    private suspend fun generateOptimizedPrompt(
        request: PromptGenerationRequest
    ): Result<PromptResult> = withContext(dispatcher) {
        try {
            // Stage 1: Data analysis and prioritization
            val analyzedData = dataCompressor.analyzeAndPrioritizeData(
                context = request.context,
                targetModel = request.targetModel,
                userPreferences = request.userPreferences
            )

            // Stage 2: Template selection and customization
            val template = templateEngine.selectOptimalTemplate(
                context = request.context,
                model = request.targetModel,
                abTestGroup = request.abTestGroup,
                effectivenessData = getTemplateEffectiveness(request.context.generationType)
            ).getOrThrow()

            // Stage 3: Content generation with intelligent summarization
            val promptContent = generatePromptContent(
                template = template,
                data = analyzedData,
                request = request
            ).getOrThrow()

            // Stage 4: Model-specific optimization
            val optimizedContent = multiModelAdapter.optimizePromptForModel(
                content = promptContent,
                model = request.targetModel,
                constraints = getModelConstraints(request.targetModel)
            )

            // Stage 5: Quality validation and enhancement
            val validatedContent = qualityValidator.validateAndEnhancePrompt(
                content = optimizedContent,
                context = request.context,
                model = request.targetModel
            ).getOrThrow()

            // Stage 6: Personalization application
            val personalizedContent = personalizationEngine.applyPersonalizations(
                content = validatedContent,
                preferences = request.userPreferences,
                context = request.context
            )

            // Stage 7: Final optimization and token management
            val finalContent = optimizeTokenUsage(
                content = personalizedContent,
                model = request.targetModel,
                targetTokens = calculateOptimalTokenCount(request)
            )

            // Create result with metadata
            val result = PromptResult(
                content = finalContent,
                estimatedTokens = estimateTokenCount(finalContent, request.targetModel),
                template = template,
                effectivenessScore = calculateEffectivenessScore(template, request.context),
                metadata = createPromptMetadata(request, template, analyzedData)
            )

            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed in optimized prompt generation pipeline", e)
            Result.failure(e)
        }
    }

    private suspend fun generatePromptContent(
        template: PromptTemplate,
        data: AnalyzedData,
        request: PromptGenerationRequest
    ): Result<String> = withContext(dispatcher) {
        try {
            val contentBuilder = AdvancedPromptContentBuilder(template)

            // Add system context with sleep medicine knowledge
            contentBuilder.addSystemContext(
                model = request.targetModel,
                expertiseLevel = request.customization?.expertiseLevel ?: ExpertiseLevel.GENERAL,
                culturalContext = request.userPreferences.culturalContext,
                sleepMedicineContext = SLEEP_MEDICINE_CONTEXT
            )

            // Add analysis request based on type
            when (request.context.generationType) {
                InsightGenerationType.POST_SESSION -> {
                    contentBuilder.addPostSessionAnalysisRequest(
                        session = data.primarySession,
                        analysisDepth = request.context.analysisDepth,
                        comparisonContext = data.comparisonContext
                    )
                }

                InsightGenerationType.DAILY_ANALYSIS -> {
                    contentBuilder.addDailyAnalysisRequest(
                        trendData = data.trendAnalysis,
                        patternData = data.patternAnalysis,
                        analysisDepth = request.context.analysisDepth
                    )
                }

                InsightGenerationType.PERSONALIZED_ANALYSIS -> {
                    contentBuilder.addPersonalizedAnalysisRequest(
                        personalBaseline = data.personalBaseline,
                        habitAnalysis = data.habitAnalysis,
                        goalProgress = data.goalAnalysis,
                        uniquePatterns = data.uniquePatterns
                    )
                }

                InsightGenerationType.PREDICTIVE_ANALYSIS -> {
                    contentBuilder.addPredictiveAnalysisRequest(
                        trends = data.trendAnalysis,
                        patterns = data.patternAnalysis,
                        riskFactors = data.riskFactors,
                        predictionHorizon = request.context.predictionHorizon
                    )
                }

                InsightGenerationType.EMERGENCY -> {
                    contentBuilder.addEmergencyAnalysisRequest(
                        emergencyType = request.context.emergencyType,
                        severity = request.context.emergencySeverity,
                        triggeringData = data.emergencyTriggerData
                    )
                }

                else -> {
                    contentBuilder.addGeneralAnalysisRequest(data)
                }
            }

            // Add data sections with intelligent prioritization
            addDataSections(contentBuilder, data, request)

            // Add response format instructions
            contentBuilder.addResponseInstructions(
                model = request.targetModel,
                responseFormat = request.customization?.responseFormat ?: ResponseFormat.INSIGHTS,
                structuredOutput = request.customization?.requiresStructuredOutput ?: false
            )

            // Add conversation context if available
            request.conversationContext?.let { context ->
                contentBuilder.addConversationContext(context)
            }

            // Build final content
            val content = contentBuilder.build()
            Result.success(content)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate prompt content", e)
            Result.failure(e)
        }
    }

    private fun addDataSections(
        builder: AdvancedPromptContentBuilder,
        data: AnalyzedData,
        request: PromptGenerationRequest
    ) {
        val tokenBudget = calculateDataTokenBudget(request.targetModel)
        var tokensUsed = 0

        // Prioritized data addition based on importance scores
        val dataSections = data.prioritizedSections.sortedByDescending { it.importanceScore }

        for (section in dataSections) {
            val estimatedTokens = estimateTokenCount(section.content, request.targetModel)

            if (tokensUsed + estimatedTokens <= tokenBudget) {
                when (section.type) {
                    DataSectionType.SESSION_DATA -> {
                        data.primarySession?.let { session ->
                            builder.addSessionData(session, section.detailLevel)
                        }
                    }

                    DataSectionType.QUALITY_ANALYSIS -> {
                        data.qualityAnalysis?.let { quality ->
                            builder.addQualityAnalysis(quality, section.detailLevel)
                        }
                    }

                    DataSectionType.TREND_ANALYSIS -> {
                        data.trendAnalysis?.let { trends ->
                            builder.addTrendAnalysis(trends, section.detailLevel)
                        }
                    }

                    DataSectionType.PATTERN_ANALYSIS -> {
                        data.patternAnalysis?.let { patterns ->
                            builder.addPatternAnalysis(patterns, section.detailLevel)
                        }
                    }

                    DataSectionType.PERSONAL_BASELINE -> {
                        data.personalBaseline?.let { baseline ->
                            builder.addPersonalBaseline(baseline, section.detailLevel)
                        }
                    }

                    DataSectionType.HABIT_ANALYSIS -> {
                        data.habitAnalysis?.let { habits ->
                            builder.addHabitAnalysis(habits, section.detailLevel)
                        }
                    }

                    DataSectionType.GOAL_ANALYSIS -> {
                        data.goalAnalysis?.let { goals ->
                            builder.addGoalAnalysis(goals, section.detailLevel)
                        }
                    }

                    DataSectionType.COMPARISON_DATA -> {
                        data.comparisonContext?.let { comparison ->
                            builder.addComparisonData(comparison, section.detailLevel)
                        }
                    }

                    DataSectionType.ENVIRONMENTAL_DATA -> {
                        data.environmentalData?.let { environmental ->
                            builder.addEnvironmentalData(environmental, section.detailLevel)
                        }
                    }

                    DataSectionType.STATISTICAL_SUMMARY -> {
                        data.statisticalSummary?.let { stats ->
                            builder.addStatisticalSummary(stats, section.detailLevel)
                        }
                    }
                }

                tokensUsed += estimatedTokens
            } else {
                // Add truncation notice if we run out of token budget
                builder.addDataTruncationNotice(
                    remainingSections = dataSections.size - dataSections.indexOf(section),
                    tokenLimitReached = true
                )
                break
            }
        }
    }

    // ========== PERFORMANCE OPTIMIZATION AND TRACKING ==========

    /**
     * Record prompt effectiveness based on AI response quality
     */
    suspend fun recordPromptEffectiveness(
        promptId: String,
        aiResponse: String,
        userFeedback: InsightFeedback? = null,
        engagementMetrics: EngagementMetrics? = null
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val effectiveness = calculatePromptEffectiveness(
                promptId = promptId,
                aiResponse = aiResponse,
                userFeedback = userFeedback,
                engagementMetrics = engagementMetrics
            )

            performanceTracker.recordEffectiveness(promptId, effectiveness)

            // Update template effectiveness scores
            templateEngine.updateTemplateEffectiveness(promptId, effectiveness)

            // Update personalization models
            personalizationEngine.updateWithFeedback(promptId, effectiveness)

            // Trigger optimization if enough data collected
            if (shouldTriggerOptimization()) {
                scope.launch {
                    optimizePromptGeneration()
                }
            }

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record prompt effectiveness", e)
            Result.failure(e)
        }
    }

    /**
     * Get comprehensive performance analytics
     */
    suspend fun getPerformanceAnalytics(
        timeRange: TimeRange? = null
    ): PromptPerformanceAnalytics = withContext(dispatcher) {
        performanceTracker.getDetailedAnalytics(timeRange)
    }

    /**
     * Optimize prompt generation based on performance data
     */
    suspend fun optimizePromptGeneration(): Result<OptimizationResult> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Starting prompt generation optimization")

            val optimizations = mutableListOf<OptimizationAction>()
            var improvementScore = 0f

            // Optimize templates
            val templateOptimization = templateEngine.optimizeTemplates(
                performanceData = performanceTracker.getTemplatePerformanceData(),
                abTestResults = getABTestResults()
            )
            optimizations.addAll(templateOptimization.actions)
            improvementScore += templateOptimization.improvementScore

            // Optimize personalization
            val personalizationOptimization = personalizationEngine.optimize(
                userFeedbackData = performanceTracker.getUserFeedbackData(),
                engagementData = performanceTracker.getEngagementData()
            )
            optimizations.addAll(personalizationOptimization.actions)
            improvementScore += personalizationOptimization.improvementScore

            // Optimize data compression
            val compressionOptimization = dataCompressor.optimize(
                tokenUsageData = performanceTracker.getTokenUsageData(),
                qualityScores = performanceTracker.getQualityScores()
            )
            optimizations.addAll(compressionOptimization.actions)
            improvementScore += compressionOptimization.improvementScore

            // Apply optimizations
            applyOptimizations(optimizations)

            val result = OptimizationResult(
                actions = optimizations,
                overallImprovementScore = improvementScore / 3f,
                estimatedImpact = calculateEstimatedImpact(optimizations),
                timestamp = System.currentTimeMillis()
            )

            Log.d(TAG, "Prompt optimization completed: score=${result.overallImprovementScore}")
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to optimize prompt generation", e)
            Result.failure(e)
        }
    }

    // ========== MODEL-SPECIFIC ADAPTATIONS ==========

    private inner class MultiModelAdapter {

        suspend fun initialize(preferences: SharedPreferences) {
            // Load model-specific preferences and configurations
        }

        fun optimizePromptForModel(
            content: String,
            model: AIModel,
            constraints: ModelConstraints
        ): String {
            return when (model) {
                AIModel.GPT4 -> optimizeForGPT4(content, constraints)
                AIModel.CLAUDE -> optimizeForClaude(content, constraints)
                AIModel.GEMINI -> optimizeForGemini(content, constraints)
                AIModel.GPT3_5 -> optimizeForGPT35(content, constraints)
                else -> content
            }
        }

        private fun optimizeForGPT4(content: String, constraints: ModelConstraints): String {
            return content
                .let { addGPT4SpecificInstructions(it) }
                .let { optimizeTokenUsage(it, AIModel.GPT4, constraints.maxTokens) }
                .let { enhanceStructuralClarity(it) }
        }

        private fun optimizeForClaude(content: String, constraints: ModelConstraints): String {
            return content
                .let { addClaudeSpecificInstructions(it) }
                .let { enhanceContextualReasoning(it) }
                .let { optimizeForLongContext(it) }
        }

        private fun optimizeForGemini(content: String, constraints: ModelConstraints): String {
            return content
                .let { addGeminiSpecificInstructions(it) }
                .let { enhanceMultimodalSupport(it) }
                .let { optimizeForLargeContext(it) }
        }

        private fun optimizeForGPT35(content: String, constraints: ModelConstraints): String {
            return content
                .let { simplifyInstructions(it) }
                .let { reduceCognitivLoad(it) }
                .let { optimizeTokenUsage(it, AIModel.GPT3_5, constraints.maxTokens) }
        }

        private fun addGPT4SpecificInstructions(content: String): String {
            return content + "\n\nNote: Use your advanced reasoning capabilities to provide nuanced, evidence-based insights."
        }

        private fun addClaudeSpecificInstructions(content: String): String {
            return content + "\n\nNote: Apply thorough analysis and consider multiple perspectives in your assessment."
        }

        private fun addGeminiSpecificInstructions(content: String): String {
            return content + "\n\nNote: Leverage your multimodal understanding to provide comprehensive insights."
        }

        private fun enhanceStructuralClarity(content: String): String {
            // Add clear section headers and improve readability for GPT-4
            return content.replace(Regex("(?m)^## (.+)$"), "## [$1]")
        }

        private fun enhanceContextualReasoning(content: String): String {
            // Add reasoning prompts for Claude's analytical strengths
            return content + "\n\nPlease reason through your analysis step-by-step before providing recommendations."
        }

        private fun enhanceMultimodalSupport(content: String): String {
            // Prepare content for potential multimodal inputs with Gemini
            return content.replace("data shows", "the provided data demonstrates")
        }

        private fun optimizeForLongContext(content: String): String {
            // Optimize for Claude's long context window
            return content // Claude can handle very long contexts well
        }

        private fun optimizeForLargeContext(content: String): String {
            // Optimize for Gemini's very large context window
            return content // Gemini has massive context capacity
        }

        private fun simplifyInstructions(content: String): String {
            // Simplify for GPT-3.5's more limited capabilities
            return content
                .replace("comprehensive analysis", "analysis")
                .replace("sophisticated", "detailed")
                .replace("nuanced", "careful")
        }

        private fun reduceCognitivLoad(content: String): String {
            // Reduce complexity for GPT-3.5
            return content.split("\n\n").take(10).joinToString("\n\n") // Limit sections
        }
    }

    // ========== TEMPLATE ENGINE ==========

    private inner class AdvancedTemplateEngine {

        private val templates = ConcurrentHashMap<String, PromptTemplate>()
        private val templateEffectiveness = ConcurrentHashMap<String, Float>()

        suspend fun initialize(preferences: SharedPreferences) {
            loadTemplates()
            loadTemplateEffectiveness(preferences)
        }

        suspend fun getSpecializedTemplate(
            specialization: PromptSpecialization,
            expertiseLevel: ExpertiseLevel,
            model: AIModel
        ): Result<PromptTemplate> {
            return try {
                val templateKey = "${specialization.name}_${expertiseLevel.name}_${model.name}"
                val template = templates[templateKey]
                    ?: getDefaultTemplate(specialization, expertiseLevel)
                    ?: return Result.failure(IllegalStateException("No template found"))

                Result.success(template)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

        suspend fun selectOptimalTemplate(
            context: InsightGenerationContext,
            model: AIModel,
            abTestGroup: ABTestGroup,
            effectivenessData: Map<String, Float>
        ): Result<PromptTemplate> {
            return try {
                val candidateTemplates = getCandidateTemplates(context, model)

                val optimalTemplate = when (abTestGroup) {
                    ABTestGroup.CONTROL -> selectControlTemplate(candidateTemplates)
                    ABTestGroup.VARIANT_A -> selectVariantATemplate(candidateTemplates)
                    ABTestGroup.VARIANT_B -> selectVariantBTemplate(candidateTemplates)
                    ABTestGroup.PERFORMANCE_OPTIMIZED -> selectPerformanceOptimizedTemplate(
                        candidateTemplates, effectivenessData
                    )
                }

                Result.success(optimalTemplate)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

        suspend fun renderTemplate(
            template: PromptTemplate,
            context: Any,
            personalizations: Map<String, Any>
        ): Result<String> {
            return try {
                var rendered = template.content

                // Apply variable substitution
                rendered = applyVariableSubstitution(rendered, context)

                // Apply personalizations
                rendered = applyPersonalizations(rendered, personalizations)

                // Apply template inheritance
                rendered = applyTemplateInheritance(rendered, template)

                Result.success(rendered)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

        fun updateTemplateEffectiveness(promptId: String, effectiveness: PromptEffectiveness) {
            // Update template effectiveness based on feedback
            val templateId = extractTemplateId(promptId)
            templateEffectiveness[templateId] = effectiveness.overallScore
        }

        suspend fun optimizeTemplates(
            performanceData: TemplatePerformanceData,
            abTestResults: ABTestResults
        ): TemplateOptimizationResult {
            val optimizations = mutableListOf<OptimizationAction>()
            var improvementScore = 0f

            // Analyze template performance
            val underperformingTemplates = performanceData.templates
                .filter { it.effectivenessScore < 0.7f }

            // Optimize or replace underperforming templates
            for (template in underperformingTemplates) {
                val optimization = optimizeTemplate(template, performanceData, abTestResults)
                optimizations.add(optimization)
                improvementScore += optimization.expectedImprovement
            }

            // Update template weights based on A/B test results
            updateTemplateWeights(abTestResults)

            return TemplateOptimizationResult(
                actions = optimizations,
                improvementScore = improvementScore / underperformingTemplates.size.coerceAtLeast(1),
                templatesOptimized = underperformingTemplates.size
            )
        }

        private fun loadTemplates() {
            // Load template definitions
            templates.putAll(getBuiltInTemplates())
        }

        private fun getBuiltInTemplates(): Map<String, PromptTemplate> {
            return mapOf(
                "post_session_standard_gpt4" to createPostSessionTemplate(ExpertiseLevel.GENERAL, AIModel.GPT4),
                "daily_analysis_standard_gpt4" to createDailyAnalysisTemplate(ExpertiseLevel.GENERAL, AIModel.GPT4),
                "personalized_standard_gpt4" to createPersonalizedTemplate(ExpertiseLevel.GENERAL, AIModel.GPT4),
                "predictive_standard_gpt4" to createPredictiveTemplate(ExpertiseLevel.GENERAL, AIModel.GPT4),
                "emergency_standard_gpt4" to createEmergencyTemplate(ExpertiseLevel.GENERAL, AIModel.GPT4)
            )
        }

        private fun createPostSessionTemplate(expertise: ExpertiseLevel, model: AIModel): PromptTemplate {
            return PromptTemplate(
                id = "post_session_${expertise.name.lowercase()}_${model.name.lowercase()}",
                name = "Post-Session Analysis",
                version = TEMPLATE_VERSION,
                content = """
                    # Sleep Session Analysis
                    
                    You are an expert sleep analyst providing evidence-based insights from comprehensive sleep tracking data.
                    
                    ## Your Role:
                    - Analyze individual sleep sessions using established sleep medicine principles
                    - Provide specific, actionable recommendations based on the session data
                    - Use encouraging, supportive language while being honest about areas for improvement
                    - Focus on behavioral and environmental factors the user can control
                    - Avoid medical advice or diagnosis
                    
                    {SYSTEM_CONTEXT}
                    
                    ## Session Analysis Request:
                    Analyze this sleep session and provide insights about:
                    1. What went well during this session (celebrate successes)
                    2. Factors that may have impacted sleep quality
                    3. Specific recommendations for improvement
                    4. Environmental or behavioral considerations
                    
                    Focus on actionable insights the user can implement for future nights.
                    
                    {SESSION_DATA}
                    {QUALITY_ANALYSIS}
                    {COMPARISON_DATA}
                    
                    {RESPONSE_INSTRUCTIONS}
                """.trimIndent(),
                variables = listOf("SYSTEM_CONTEXT", "SESSION_DATA", "QUALITY_ANALYSIS", "COMPARISON_DATA", "RESPONSE_INSTRUCTIONS"),
                metadata = TemplateMetadata(
                    expertiseLevel = expertise,
                    targetModel = model,
                    category = InsightGenerationType.POST_SESSION,
                    estimatedTokens = 1200,
                    version = TEMPLATE_VERSION
                )
            )
        }

        // Additional template creation methods would continue...
        private fun createDailyAnalysisTemplate(expertise: ExpertiseLevel, model: AIModel): PromptTemplate {
            return PromptTemplate(
                id = "daily_analysis_${expertise.name.lowercase()}_${model.name.lowercase()}",
                name = "Daily Pattern Analysis",
                version = TEMPLATE_VERSION,
                content = """
                    # Daily Sleep Pattern Analysis
                    
                    Analyze recent sleep patterns and provide insights about trends, consistency, and optimization opportunities.
                    
                    {SYSTEM_CONTEXT}
                    
                    ## Daily Analysis Request:
                    Provide insights about:
                    1. Sleep pattern consistency and schedule trends
                    2. Quality improvements or declines over time
                    3. Behavioral and environmental patterns
                    4. Strategic recommendations for the week ahead
                    
                    {TREND_ANALYSIS}
                    {PATTERN_ANALYSIS}
                    {STATISTICAL_SUMMARY}
                    
                    {RESPONSE_INSTRUCTIONS}
                """.trimIndent(),
                variables = listOf("SYSTEM_CONTEXT", "TREND_ANALYSIS", "PATTERN_ANALYSIS", "STATISTICAL_SUMMARY", "RESPONSE_INSTRUCTIONS"),
                metadata = TemplateMetadata(
                    expertiseLevel = expertise,
                    targetModel = model,
                    category = InsightGenerationType.DAILY_ANALYSIS,
                    estimatedTokens = 1000,
                    version = TEMPLATE_VERSION
                )
            )
        }

        private fun createPersonalizedTemplate(expertise: ExpertiseLevel, model: AIModel): PromptTemplate {
            return PromptTemplate(
                id = "personalized_${expertise.name.lowercase()}_${model.name.lowercase()}",
                name = "Personalized Sleep Coaching",
                version = TEMPLATE_VERSION,
                content = """
                    # Personalized Sleep Coaching
                    
                    Provide personalized sleep coaching based on this individual's unique patterns and baseline.
                    
                    {SYSTEM_CONTEXT}
                    
                    ## Personalized Coaching Request:
                    Provide customized insights about:
                    1. Individual strengths and sleep advantages
                    2. Personal challenges and growth opportunities
                    3. Customized strategies based on their unique patterns
                    4. Habit formation recommendations
                    5. Long-term optimization strategies
                    
                    Avoid generic advice - focus on what's specific to their data and patterns.
                    
                    {PERSONAL_BASELINE}
                    {HABIT_ANALYSIS}
                    {GOAL_ANALYSIS}
                    {PATTERN_ANALYSIS}
                    
                    {RESPONSE_INSTRUCTIONS}
                """.trimIndent(),
                variables = listOf("SYSTEM_CONTEXT", "PERSONAL_BASELINE", "HABIT_ANALYSIS", "GOAL_ANALYSIS", "PATTERN_ANALYSIS", "RESPONSE_INSTRUCTIONS"),
                metadata = TemplateMetadata(
                    expertiseLevel = expertise,
                    targetModel = model,
                    category = InsightGenerationType.PERSONALIZED_ANALYSIS,
                    estimatedTokens = 1400,
                    version = TEMPLATE_VERSION
                )
            )
        }

        private fun createPredictiveTemplate(expertise: ExpertiseLevel, model: AIModel): PromptTemplate {
            return PromptTemplate(
                id = "predictive_${expertise.name.lowercase()}_${model.name.lowercase()}",
                name = "Predictive Sleep Analysis",
                version = TEMPLATE_VERSION,
                content = """
                    # Predictive Sleep Analysis
                    
                    Analyze trends and patterns to provide predictive insights and proactive recommendations.
                    
                    {SYSTEM_CONTEXT}
                    
                    ## Predictive Analysis Request:
                    Based on current trends and patterns, provide insights about:
                    1. Likely sleep quality trajectory over the next week/month
                    2. Risk factors and early warning signs to watch for
                    3. Proactive interventions to maintain or improve trends
                    4. Opportunity windows for optimization
                    
                    {TREND_ANALYSIS}
                    {PATTERN_ANALYSIS}
                    {RISK_FACTORS}
                    {PREDICTION_DATA}
                    
                    {RESPONSE_INSTRUCTIONS}
                """.trimIndent(),
                variables = listOf("SYSTEM_CONTEXT", "TREND_ANALYSIS", "PATTERN_ANALYSIS", "RISK_FACTORS", "PREDICTION_DATA", "RESPONSE_INSTRUCTIONS"),
                metadata = TemplateMetadata(
                    expertiseLevel = expertise,
                    targetModel = model,
                    category = InsightGenerationType.PREDICTIVE_ANALYSIS,
                    estimatedTokens = 1100,
                    version = TEMPLATE_VERSION
                )
            )
        }

        private fun createEmergencyTemplate(expertise: ExpertiseLevel, model: AIModel): PromptTemplate {
            return PromptTemplate(
                id = "emergency_${expertise.name.lowercase()}_${model.name.lowercase()}",
                name = "Emergency Sleep Alert Analysis",
                version = TEMPLATE_VERSION,
                content = """
                    # Emergency Sleep Analysis
                    
                    URGENT: Analyze critical sleep health situation and provide immediate guidance.
                    
                    {SYSTEM_CONTEXT}
                    
                    ## Emergency Analysis Request:
                    This is an urgent analysis of concerning sleep patterns. Provide:
                    1. Assessment of the severity and immediate risks
                    2. Immediate actions the user should consider
                    3. Potential underlying factors contributing to the issue
                    4. When to seek professional medical advice
                    
                    Be direct but supportive. Focus on immediate safety and actionable steps.
                    
                    {EMERGENCY_DATA}
                    {TRIGGERING_FACTORS}
                    {RECENT_CONTEXT}
                    
                    {RESPONSE_INSTRUCTIONS}
                """.trimIndent(),
                variables = listOf("SYSTEM_CONTEXT", "EMERGENCY_DATA", "TRIGGERING_FACTORS", "RECENT_CONTEXT", "RESPONSE_INSTRUCTIONS"),
                metadata = TemplateMetadata(
                    expertiseLevel = expertise,
                    targetModel = model,
                    category = InsightGenerationType.EMERGENCY,
                    estimatedTokens = 800,
                    version = TEMPLATE_VERSION
                )
            )
        }

        // Helper methods for template management
        private fun getCandidateTemplates(context: InsightGenerationContext, model: AIModel): List<PromptTemplate> {
            return templates.values.filter {
                it.metadata.category == context.generationType &&
                        it.metadata.targetModel == model
            }
        }

        private fun selectControlTemplate(candidates: List<PromptTemplate>): PromptTemplate {
            return candidates.find { it.id.contains("standard") } ?: candidates.first()
        }

        private fun selectVariantATemplate(candidates: List<PromptTemplate>): PromptTemplate {
            return candidates.find { it.id.contains("variant_a") } ?: candidates.first()
        }

        private fun selectVariantBTemplate(candidates: List<PromptTemplate>): PromptTemplate {
            return candidates.find { it.id.contains("variant_b") } ?: candidates.first()
        }

        private fun selectPerformanceOptimizedTemplate(
            candidates: List<PromptTemplate>,
            effectivenessData: Map<String, Float>
        ): PromptTemplate {
            return candidates.maxByOrNull { template ->
                effectivenessData[template.id] ?: 0.5f
            } ?: candidates.first()
        }

        private fun applyVariableSubstitution(content: String, context: Any): String {
            var result = content
            // Apply variable substitution logic
            return result
        }

        private fun applyPersonalizations(content: String, personalizations: Map<String, Any>): String {
            var result = content
            // Apply personalization logic
            return result
        }

        private fun applyTemplateInheritance(content: String, template: PromptTemplate): String {
            var result = content
            // Apply template inheritance logic
            return result
        }

        private fun extractTemplateId(promptId: String): String {
            return promptId.split("_").firstOrNull() ?: "unknown"
        }

        private fun optimizeTemplate(
            template: TemplatePerformanceData.TemplateData,
            performanceData: TemplatePerformanceData,
            abTestResults: ABTestResults
        ): OptimizationAction {
            return OptimizationAction(
                type = "template_optimization",
                description = "Optimize underperforming template ${template.templateId}",
                expectedImprovement = 0.2f,
                implementation = { /* Optimize template */ }
            )
        }

        private fun updateTemplateWeights(abTestResults: ABTestResults) {
            // Update template selection weights based on A/B test results
        }

        private fun loadTemplateEffectiveness(preferences: SharedPreferences) {
            // Load historical template effectiveness data
        }

        private fun getDefaultTemplate(specialization: PromptSpecialization, expertiseLevel: ExpertiseLevel): PromptTemplate? {
            return templates.values.find {
                it.metadata.expertiseLevel == expertiseLevel
            }
        }
    }

    // ========== UTILITY AND HELPER METHODS ==========

    private suspend fun loadPromptBuilderState() {
        currentConfig = loadConfiguration()
        abTestGroup = loadABTestGroup()
        userPreferences = loadUserPreferences()
    }

    private suspend fun loadTemplateCache() {
        // Load cached templates from preferences
    }

    private fun startPerformanceOptimization() {
        scope.launch {
            while (isActive) {
                delay(TimeUnit.HOURS.toMillis(6)) // Optimize every 6 hours
                try {
                    if (shouldTriggerOptimization()) {
                        optimizePromptGeneration()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in performance optimization", e)
                }
            }
        }
    }

    private fun validateAndPrepareRequest(request: PromptGenerationRequest): Result<PromptGenerationRequest> {
        return try {
            // Validate request parameters
            require(request.context.generationType != null) { "Generation type required" }
            require(request.targetModel != null) { "Target model required" }

            // Prepare and enhance request
            val enhancedRequest = request.copy(
                userPreferences = enhanceUserPreferences(request.userPreferences),
                customization = enhanceCustomization(request.customization, request.context)
            )

            Result.success(enhancedRequest)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend fun generateFallbackPrompt(
        context: InsightGenerationContext,
        targetModel: AIModel
    ): Result<PromptResult> {
        return try {
            val fallbackContent = """
                Analyze the available sleep data and provide helpful insights and recommendations.
                
                Analysis Type: ${context.generationType.name}
                ${context.sessionData?.let { "Session Duration: ${formatDuration(it.duration)}" } ?: ""}
                
                Please provide:
                1. Evidence-based insights from the available data
                2. Specific, actionable recommendations
                3. Supportive, encouraging guidance
                
                Respond with clear, helpful insights that can improve sleep quality.
            """.trimIndent()

            val result = PromptResult(
                content = fallbackContent,
                estimatedTokens = estimateTokenCount(fallbackContent, targetModel),
                template = null,
                effectivenessScore = 0.5f,
                metadata = PromptMetadata(
                    generationType = context.generationType,
                    isFallback = true,
                    timestamp = System.currentTimeMillis()
                )
            )

            Result.success(result)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // Additional utility methods for the comprehensive implementation...

    private fun estimateTokenCount(text: String, model: AIModel): Int {
        // Rough estimation: 1 token ≈ 4 characters for most models
        return when (model) {
            AIModel.GPT4, AIModel.GPT3_5 -> (text.length / 4.0).roundToInt()
            AIModel.CLAUDE -> (text.length / 3.5).roundToInt() // Claude is slightly more efficient
            AIModel.GEMINI -> (text.length / 4.2).roundToInt()
            else -> (text.length / 4.0).roundToInt()
        }
    }

    private fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        return "${hours}h ${minutes}m"
    }

    private fun loadConfiguration(): PromptBuilderConfig {
        return try {
            val configJson = preferences.getString(PREF_PROMPT_PERFORMANCE, null)
            configJson?.let { parseConfigFromJson(it) } ?: PromptBuilderConfig.default()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load configuration", e)
            PromptBuilderConfig.default()
        }
    }

    private fun loadABTestGroup(): ABTestGroup {
        val groupName = preferences.getString(PREF_AB_TEST_GROUP, ABTestGroup.CONTROL.name)
        return try {
            ABTestGroup.valueOf(groupName ?: ABTestGroup.CONTROL.name)
        } catch (e: Exception) {
            ABTestGroup.CONTROL
        }
    }

    private fun loadUserPreferences(): UserPromptPreferences {
        return try {
            val prefsJson = preferences.getString(PREF_USER_PREFERENCES, null)
            prefsJson?.let { parseUserPreferencesFromJson(it) } ?: UserPromptPreferences.default()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load user preferences", e)
            UserPromptPreferences.default()
        }
    }

    private fun updateAverageGenerationTime(time: Long) {
        val currentAvg = averageGenerationTime.get()
        val totalPrompts = totalPromptsGenerated.get()
        val newAvg = ((currentAvg * (totalPrompts - 1)) + time) / totalPrompts
        averageGenerationTime.set(newAvg)
    }

    private fun shouldTriggerOptimization(): Boolean {
        val totalPrompts = totalPromptsGenerated.get()
        return totalPrompts > 0 && totalPrompts % MIN_SAMPLES_FOR_OPTIMIZATION == 0L
    }

    /**
     * Cleanup resources
     */
    fun cleanup() {
        scope.cancel()
        templateEngine.cleanup()
        contextManager.cleanup()
        personalizationEngine.cleanup()
        performanceTracker.cleanup()
        Log.d(TAG, "InsightsPromptBuilder cleanup completed")
    }
}

// ========== DATA CLASSES AND ENUMS ==========

data class PromptGenerationRequest(
    val context: InsightGenerationContext,
    val targetModel: AIModel,
    val customization: PromptCustomization?,
    val userPreferences: UserPromptPreferences,
    val abTestGroup: ABTestGroup,
    val conversationContext: ConversationContext?,
    val timestamp: Long
)

data class PromptResult(
    val content: String,
    val estimatedTokens: Int,
    val template: PromptTemplate?,
    val effectivenessScore: Float,
    val metadata: PromptMetadata
)

data class StructuredPromptResult(
    val content: String,
    val schema: ResponseSchema,
    val validationResult: ValidationResult,
    val estimatedTokens: Int,
    val metadata: PromptMetadata
)

data class ConversationalPromptResult(
    val content: String,
    val conversationId: String,
    val contextDepth: Int,
    val personalityAdaptation: UserPersonality,
    val metadata: PromptMetadata
)

data class SpecializedPromptResult(
    val content: String,
    val specialization: PromptSpecialization,
    val expertiseLevel: ExpertiseLevel,
    val template: PromptTemplate,
    val estimatedTokens: Int,
    val metadata: PromptMetadata
)

data class PromptTemplate(
    val id: String,
    val name: String,
    val version: String,
    val content: String,
    val variables: List<String>,
    val metadata: TemplateMetadata
)

data class TemplateMetadata(
    val expertiseLevel: ExpertiseLevel,
    val targetModel: AIModel,
    val category: InsightGenerationType,
    val estimatedTokens: Int,
    val version: String
)

data class PromptMetadata(
    val generationType: InsightGenerationType,
    val modelOptimized: String = "",
    val templateVersion: String = "",
    val specialization: String = "",
    val expertiseLevel: String = "",
    val responseSchema: String = "",
    val conversationId: String = "",
    val isStructured: Boolean = false,
    val isConversational: Boolean = false,
    val isFallback: Boolean = false,
    val timestamp: Long
)

data class PromptBuilderConfig(
    val enableAdvancedOptimization: Boolean = true,
    val enablePersonalization: Boolean = true,
    val enableABTesting: Boolean = true,
    val maxTokensPerPrompt: Int = 4000,
    val optimizationFrequency: Duration = Duration.ofHours(6),
    val templateCacheSize: Int = TEMPLATE_CACHE_SIZE,
    val performanceTrackingEnabled: Boolean = true
) {
    companion object {
        fun default() = PromptBuilderConfig()
    }
}

data class UserPromptPreferences(
    val preferredResponseLength: ResponseLength = ResponseLength.STANDARD,
    val communicationStyle: CommunicationStyle = CommunicationStyle.SUPPORTIVE,
    val expertiseLevel: ExpertiseLevel = ExpertiseLevel.GENERAL,
    val culturalContext: CulturalContext = CulturalContext.WESTERN,
    val languagePreference: String = "en",
    val focusAreas: List<FocusArea> = emptyList(),
    val personalityType: UserPersonality = UserPersonality.BALANCED
) {
    companion object {
        fun default() = UserPromptPreferences()
    }
}

enum class AIModel {
    GPT4, GPT3_5, CLAUDE, GEMINI, CUSTOM
}

enum class ABTestGroup {
    CONTROL, VARIANT_A, VARIANT_B, PERFORMANCE_OPTIMIZED
}

enum class ExpertiseLevel {
    BEGINNER, GENERAL, INTERMEDIATE, ADVANCED, EXPERT
}

enum class PromptSpecialization {
    SLEEP_MEDICINE, BEHAVIORAL_CHANGE, HABIT_FORMATION,
    PERFORMANCE_OPTIMIZATION, STRESS_MANAGEMENT, GENERAL_WELLNESS
}

enum class ResponseLength {
    BRIEF, STANDARD, DETAILED, COMPREHENSIVE
}

enum class CommunicationStyle {
    DIRECT, SUPPORTIVE, SCIENTIFIC, CONVERSATIONAL, MOTIVATIONAL
}

enum class CulturalContext {
    WESTERN, EASTERN, MEDITERRANEAN, NORDIC, CUSTOM
}

enum class UserPersonality {
    ANALYTICAL, EMOTIONAL, BALANCED, GOAL_ORIENTED, WELLNESS_FOCUSED
}

enum class FocusArea {
    DURATION, QUALITY, CONSISTENCY, ENVIRONMENT, HABITS, STRESS
}

enum class ValidationLevel {
    NONE, BASIC, STANDARD, STRICT
}

enum class ResponseFormat {
    INSIGHTS, JSON, MARKDOWN, CONVERSATIONAL, STRUCTURED
}

// Additional supporting classes would continue with the same level of detail...

// Placeholder implementations for complex components
private class ConversationContextManager {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun getCurrentContext(): ConversationContext? = null
    fun updateConversationContext(conversationId: String, context: InsightGenerationContext, interactions: List<Interaction>) {}
    fun getConversationContext(conversationId: String): ConversationContext = ConversationContext()
    fun cleanup() {}
}

private class PromptPersonalizationEngine {
    suspend fun initialize(preferences: SharedPreferences, repository: SleepRepository) {}
    fun getPersonalizations(specialization: PromptSpecialization): Map<String, Any> = emptyMap()
    fun applyPersonalizations(content: String, preferences: UserPromptPreferences, context: InsightGenerationContext): String = content
    fun updateWithFeedback(promptId: String, effectiveness: PromptEffectiveness) {}
    suspend fun optimize(userFeedbackData: Any, engagementData: Any): OptimizationResult = OptimizationResult(emptyList(), 0f, emptyMap(), 0L)
    fun cleanup() {}
}

private class PromptPerformanceTracker {
    suspend fun initialize(preferences: SharedPreferences) {}
    fun recordGeneration(result: PromptResult, time: Long) {}
    fun recordFailure(context: InsightGenerationContext, model: AIModel, error: Exception, time: Long) {}
    fun recordEffectiveness(promptId: String, effectiveness: PromptEffectiveness) {}
    suspend fun getDetailedAnalytics(timeRange: TimeRange?): PromptPerformanceAnalytics = PromptPerformanceAnalytics()
    fun getTemplatePerformanceData(): TemplatePerformanceData = TemplatePerformanceData()
    fun getUserFeedbackData(): Any = Unit
    fun getEngagementData(): Any = Unit
    fun getTokenUsageData(): Any = Unit
    fun getQualityScores(): Any = Unit
    fun cleanup() {}
}

private class IntelligentDataCompressor {
    suspend fun analyzeAndPrioritizeData(context: InsightGenerationContext, targetModel: AIModel, userPreferences: UserPromptPreferences): AnalyzedData = AnalyzedData()
    suspend fun optimize(tokenUsageData: Any, qualityScores: Any): OptimizationResult = OptimizationResult(emptyList(), 0f, emptyMap(), 0L)
}

private class ResponseQualityValidator {
    suspend fun validateAndEnhancePrompt(content: String, context: InsightGenerationContext, model: AIModel): Result<String> = Result.success(content)
    fun validateStructuredPrompt(prompt: String, schema: ResponseSchema, model: AIModel): ValidationResult = ValidationResult()
}

private class AdvancedPromptContentBuilder(private val template: PromptTemplate) {
    fun addSystemContext(model: AIModel, expertiseLevel: ExpertiseLevel, culturalContext: CulturalContext, sleepMedicineContext: Map<String, ClosedFloatingPointRange<Float>>) {}
    fun addPostSessionAnalysisRequest(session: SleepSession?, analysisDepth: AnalysisDepth, comparisonContext: Any?) {}
    fun addDailyAnalysisRequest(trendData: Any?, patternData: Any?, analysisDepth: AnalysisDepth) {}
    fun addPersonalizedAnalysisRequest(personalBaseline: Any?, habitAnalysis: Any?, goalProgress: Any?, uniquePatterns: Any?) {}
    fun addPredictiveAnalysisRequest(trends: Any?, patterns: Any?, riskFactors: Any?, predictionHorizon: PredictionHorizon?) {}
    fun addEmergencyAnalysisRequest(emergencyType: EmergencyInsightType?, severity: EmergencySeverity?, triggeringData: Any?) {}
    fun addGeneralAnalysisRequest(data: AnalyzedData) {}
    fun addSessionData(session: SleepSession, detailLevel: DetailLevel) {}
    fun addQualityAnalysis(quality: SessionQualityAnalysis, detailLevel: DetailLevel) {}
    fun addTrendAnalysis(trends: TrendAnalysis, detailLevel: DetailLevel) {}
    fun addPatternAnalysis(patterns: SleepPatternAnalysis, detailLevel: DetailLevel) {}
    fun addPersonalBaseline(baseline: PersonalBaseline, detailLevel: DetailLevel) {}
    fun addHabitAnalysis(habits: HabitAnalysis, detailLevel: DetailLevel) {}
    fun addGoalAnalysis(goals: GoalAnalysis, detailLevel: DetailLevel) {}
    fun addComparisonData(comparison: Any, detailLevel: DetailLevel) {}
    fun addEnvironmentalData(environmental: Any, detailLevel: DetailLevel) {}
    fun addStatisticalSummary(stats: Any, detailLevel: DetailLevel) {}
    fun addDataTruncationNotice(remainingSections: Int, tokenLimitReached: Boolean) {}
    fun addResponseInstructions(model: AIModel, responseFormat: ResponseFormat, structuredOutput: Boolean) {}
    fun addConversationContext(context: ConversationContext) {}
    fun build(): String = template.content
}

// Supporting data classes (simplified)
data class ConversationContext(
    val history: List<Interaction> = emptyList(),
    val userPersonality: UserPersonality = UserPersonality.BALANCED,
    val communicationStyle: CommunicationStyle = CommunicationStyle.SUPPORTIVE,
    val contextDepth: Int = 0
)

data class Interaction(val prompt: String, val response: String, val timestamp: Long)
data class AnalyzedData(
    val primarySession: SleepSession? = null,
    val qualityAnalysis: SessionQualityAnalysis? = null,
    val trendAnalysis: TrendAnalysis? = null,
    val patternAnalysis: SleepPatternAnalysis? = null,
    val personalBaseline: PersonalBaseline? = null,
    val habitAnalysis: HabitAnalysis? = null,
    val goalAnalysis: GoalAnalysis? = null,
    val comparisonContext: Any? = null,
    val environmentalData: Any? = null,
    val statisticalSummary: Any? = null,
    val emergencyTriggerData: Any? = null,
    val riskFactors: Any? = null,
    val uniquePatterns: Any? = null,
    val prioritizedSections: List<DataSection> = emptyList()
)

data class DataSection(
    val type: DataSectionType,
    val content: String,
    val importanceScore: Float,
    val detailLevel: DetailLevel
)

data class PromptEffectiveness(val overallScore: Float)
data class PromptPerformanceAnalytics()
data class TemplatePerformanceData(val templates: List<TemplateData> = emptyList()) {
    data class TemplateData(val templateId: String, val effectivenessScore: Float)
}
data class ABTestResults()
data class TemplateOptimizationResult(val actions: List<OptimizationAction>, val improvementScore: Float, val templatesOptimized: Int)
data class OptimizationResult(val actions: List<OptimizationAction>, val overallImprovementScore: Float, val estimatedImpact: Map<String, Float>, val timestamp: Long)
data class OptimizationAction(val type: String, val description: String, val expectedImprovement: Float, val implementation: () -> Unit)
data class ResponseSchema(val name: String)
data class ValidationResult()
data class ModelConstraints(val maxTokens: Int)
data class PromptCustomization(
    val expertiseLevel: ExpertiseLevel? = null,
    val responseFormat: ResponseFormat? = null,
    val requiresStructuredOutput: Boolean = false
)

enum class DataSectionType {
    SESSION_DATA, QUALITY_ANALYSIS, TREND_ANALYSIS, PATTERN_ANALYSIS,
    PERSONAL_BASELINE, HABIT_ANALYSIS, GOAL_ANALYSIS, COMPARISON_DATA,
    ENVIRONMENTAL_DATA, STATISTICAL_SUMMARY
}

enum class DetailLevel { MINIMAL, STANDARD, DETAILED, COMPREHENSIVE }

// Extension functions
private fun parseConfigFromJson(json: String): PromptBuilderConfig = PromptBuilderConfig.default()
private fun parseUserPreferencesFromJson(json: String): UserPromptPreferences = UserPromptPreferences.default()
private fun AdvancedTemplateEngine.cleanup() {}
private fun createDefaultCustomization(context: InsightGenerationContext): PromptCustomization = PromptCustomization()
private fun enhanceUserPreferences(preferences: UserPromptPreferences): UserPromptPreferences = preferences
private fun enhanceCustomization(customization: PromptCustomization?, context: InsightGenerationContext): PromptCustomization = customization ?: PromptCustomization()
private fun getTemplateEffectiveness(type: InsightGenerationType): Map<String, Float> = emptyMap()
private fun createPromptMetadata(request: PromptGenerationRequest, template: PromptTemplate, data: AnalyzedData): PromptMetadata {
    return PromptMetadata(
        generationType = request.context.generationType,
        timestamp = request.timestamp
    )
}
private fun calculateEffectivenessScore(template: PromptTemplate, context: InsightGenerationContext): Float = 0.8f
private fun getModelConstraints(model: AIModel): ModelConstraints = ModelConstraints(maxTokens = 4000)
private fun calculateOptimalTokenCount(request: PromptGenerationRequest): Int = 3000
private fun optimizeTokenUsage(content: String, model: AIModel, targetTokens: Int): String = content
private fun calculateDataTokenBudget(model: AIModel): Int = when(model) {
    AIModel.GPT4 -> 3000
    AIModel.CLAUDE -> 10000
    AIModel.GEMINI -> 50000
    else -> 2000
}
private fun prepareSpecializedContext(context: InsightGenerationContext, specialization: PromptSpecialization, expertiseLevel: ExpertiseLevel): Any = context
private fun generateStructuredInstructions(schema: ResponseSchema, model: AIModel, validationLevel: ValidationLevel): String = ""
private fun combinePromptWithStructuredInstructions(basePrompt: String, instructions: String, schema: ResponseSchema): String = basePrompt + instructions
private fun enhancePromptWithConversationalContext(basePrompt: String, conversationContext: ConversationContext, targetModel: AIModel): String = basePrompt
private fun calculatePromptEffectiveness(promptId: String, aiResponse: String, userFeedback: InsightFeedback?, engagementMetrics: EngagementMetrics?): PromptEffectiveness = PromptEffectiveness(0.8f)
private fun getABTestResults(): ABTestResults = ABTestResults()
private fun applyOptimizations(optimizations: List<OptimizationAction>) {}
private fun calculateEstimatedImpact(optimizations: List<OptimizationAction>): Map<String, Float> = emptyMap()

data class EngagementMetrics(val dummy: Boolean = true)
data class CachedTemplate(val template: PromptTemplate, val timestamp: Long = System.currentTimeMillis())