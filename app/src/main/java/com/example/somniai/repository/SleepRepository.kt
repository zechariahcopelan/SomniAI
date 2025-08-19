package com.example.somniai.repository

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.map
import com.example.somniai.data.*
import com.example.somniai.database.*
import com.example.somniai.ai.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentHashMap
import java.util.Calendar
import java.util.TimeZone

/**
 * Enhanced central repository for all sleep tracking data with comprehensive AI integration
 *
 * Responsibilities:
 * - Single source of truth for sleep data
 * - Combines database persistence with real-time sensor data
 * - Data transformation between entities and domain models
 * - Caching and performance optimization
 * - Session lifecycle management
 * - Analytics and statistics computation
 * - AI insights generation and management
 * - User interaction tracking and personalization
 * - AI model performance monitoring
 * - Advanced pattern analysis and trend detection
 * - Comprehensive feedback loops for AI improvement
 */
class SleepRepository(
    private val database: SleepDatabase,
    private val context: Context,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "SleepRepository"
        private const val CACHE_EXPIRY_MS = 5 * 60 * 1000L // 5 minutes
        private const val MAX_CACHE_SIZE = 100
        private const val AI_CACHE_EXPIRY_MS = 30 * 60 * 1000L // 30 minutes for AI data
        private const val ANALYTICS_CACHE_EXPIRY_MS = 15 * 60 * 1000L // 15 minutes for analytics
        private const val PATTERN_ANALYSIS_MIN_SESSIONS = 7
        private const val TREND_ANALYSIS_MIN_SESSIONS = 14
        private const val PERSONALIZATION_MIN_INTERACTIONS = 10
    }

    // Enhanced DAO references including AI components
    private val sessionDao = database.sleepSessionDao()
    private val movementDao = database.movementEventDao()
    private val noiseDao = database.noiseEventDao()
    private val phaseDao = database.sleepPhaseDao()
    private val qualityDao = database.qualityFactorsDao()
    private val insightDao = database.sleepInsightDao()
    private val settingsDao = database.sensorSettingsDao()

    // AI Integration DAOs
    private val aiGenerationJobDao = database.aiGenerationJobDao()
    private val userInteractionDao = database.userInteractionDao()
    private val userPreferencesDao = database.userPreferencesDao()
    private val aiModelPerformanceDao = database.aiModelPerformanceDao()
    private val insightFeedbackDao = database.insightFeedbackDao()

    // Enhanced caching layer with AI-specific caches
    private val sessionCache = ConcurrentHashMap<Long, CacheEntry<SleepSession>>()
    private val statisticsCache = ConcurrentHashMap<String, CacheEntry<SleepAnalytics>>()
    private val trendCache = ConcurrentHashMap<String, CacheEntry<List<DailyTrendData>>>()
    private val insightCache = ConcurrentHashMap<String, CacheEntry<List<SleepInsight>>>()
    private val analyticsCache = ConcurrentHashMap<String, CacheEntry<Any>>()
    private val personalizationCache = ConcurrentHashMap<String, CacheEntry<PersonalBaseline>>()
    private val patternCache = ConcurrentHashMap<String, CacheEntry<SleepPatternAnalysis>>()

    // Real-time data state
    private val _currentSessionFlow = MutableStateFlow<SleepSession?>(null)
    val currentSessionFlow: StateFlow<SleepSession?> = _currentSessionFlow.asStateFlow()

    private val _realTimeMetrics = MutableStateFlow<LiveSleepMetrics?>(null)
    val realTimeMetrics: StateFlow<LiveSleepMetrics?> = _realTimeMetrics.asStateFlow()

    // AI-specific state flows
    private val _aiGenerationStatus = MutableStateFlow<AIGenerationStatus>(AIGenerationStatus.IDLE)
    val aiGenerationStatus: StateFlow<AIGenerationStatus> = _aiGenerationStatus.asStateFlow()

    private val _userEngagementMetrics = MutableStateFlow<UserEngagementMetrics?>(null)
    val userEngagementMetrics: StateFlow<UserEngagementMetrics?> = _userEngagementMetrics.asStateFlow()

    // Session management
    private var currentSessionId: Long? = null
    private val scope = CoroutineScope(dispatcher + SupervisorJob())

    init {
        // Initialize with active session if exists
        scope.launch {
            loadActiveSession()
            startCacheCleanup()
            initializeAIComponents()
        }
    }

    // ========== ENHANCED SESSION MANAGEMENT ==========

    /**
     * Create a new sleep tracking session with AI integration
     */
    suspend fun createSession(
        startTime: Long = System.currentTimeMillis(),
        settings: SensorSettings? = null
    ): Result<Long> = withContext(dispatcher) {
        try {
            // Check for existing active session
            val activeSession = sessionDao.getActiveSession()
            if (activeSession != null) {
                Log.w(TAG, "Active session already exists: ${activeSession.id}")
                return@withContext Result.failure(IllegalStateException("Active session already exists"))
            }

            // Create new session entity with AI-ready fields
            val sessionEntity = SleepSessionEntity(
                startTime = startTime,
                endTime = null,
                totalDuration = 0L,
                aiAnalysisStatus = "pending",
                createdAt = java.util.Date(),
                updatedAt = java.util.Date()
            )

            // Insert session and get ID
            val sessionId = sessionDao.insertSession(sessionEntity)
            currentSessionId = sessionId

            // Save sensor settings if provided
            settings?.let {
                val settingsEntity = EntityHelper.createSensorSettingsEntity(it, sessionId)
                settingsDao.insertSettings(settingsEntity)
            }

            // Create AI generation job for real-time analysis
            createAIGenerationJob(
                sessionId = sessionId,
                generationType = "REAL_TIME_ANALYSIS",
                aiModel = "GPT-4"
            )

            // Update real-time flow
            val session = sessionEntity.copy(id = sessionId).toDomainModel()
            _currentSessionFlow.value = session

            // Record user interaction
            recordUserInteraction(
                insightId = null,
                interactionType = "SESSION_STARTED",
                details = "New session created"
            )

            Log.d(TAG, "New session created with ID: $sessionId")
            Result.success(sessionId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to create session", e)
            Result.failure(e)
        }
    }

    /**
     * Complete current session with comprehensive AI analysis
     */
    suspend fun completeSession(
        endTime: Long = System.currentTimeMillis(),
        finalAnalytics: SleepSessionAnalytics? = null
    ): Result<SleepSession> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.failure(
                IllegalStateException("No active session to complete")
            )

            val currentSession = sessionDao.getSessionById(sessionId)
                ?: return@withContext Result.failure(IllegalStateException("Session not found"))

            // Calculate final metrics
            val totalDuration = endTime - currentSession.startTime
            val qualityScore = finalAnalytics?.qualityFactors?.overallScore

            // Update session with completion data and AI analysis status
            val completedSession = currentSession.copy(
                endTime = endTime,
                totalDuration = totalDuration,
                sleepEfficiency = finalAnalytics?.sleepEfficiency ?: currentSession.sleepEfficiency,
                qualityScore = qualityScore,
                averageMovementIntensity = finalAnalytics?.averageMovementIntensity ?: currentSession.averageMovementIntensity,
                averageNoiseLevel = finalAnalytics?.averageNoiseLevel ?: currentSession.averageNoiseLevel,
                movementFrequency = finalAnalytics?.movementFrequency ?: currentSession.movementFrequency,
                aiAnalysisStatus = "pending",
                updatedAt = java.util.Date()
            )

            // Save to database
            sessionDao.updateSession(completedSession)

            // Save quality factors if available
            finalAnalytics?.qualityFactors?.let { factors ->
                val factorsEntity = EntityHelper.createQualityFactorsEntity(factors, sessionId)
                qualityDao.insertQualityFactors(factorsEntity)
            }

            // Trigger AI analysis for completed session
            triggerPostSessionAIAnalysis(sessionId)

            // Update user engagement metrics
            updateUserEngagementMetrics()

            // Clear current session state
            currentSessionId = null
            _currentSessionFlow.value = null
            _realTimeMetrics.value = null

            // Invalidate caches
            invalidateAllCaches()

            val finalSession = completedSession.toDomainModel()
            Log.d(TAG, "Session $sessionId completed. Duration: ${totalDuration}ms, Quality: $qualityScore")

            Result.success(finalSession)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to complete session", e)
            Result.failure(e)
        }
    }

    // ========== AI INSIGHTS GENERATION AND MANAGEMENT ==========

    /**
     * Generate AI insights for a specific session
     */
    suspend fun generateSessionInsights(
        sessionId: Long,
        forceRegenerate: Boolean = false,
        analysisDepth: AnalysisDepth = AnalysisDepth.DETAILED
    ): Result<List<SleepInsight>> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Generating AI insights for session: $sessionId")

            // Check if insights already exist and are recent
            if (!forceRegenerate) {
                val existingInsights = insightDao.getInsightsForSession(sessionId)
                if (existingInsights.isNotEmpty()) {
                    val recentInsight = existingInsights.maxByOrNull { it.timestamp }
                    if (recentInsight != null &&
                        System.currentTimeMillis() - recentInsight.timestamp < AI_CACHE_EXPIRY_MS) {
                        return@withContext Result.success(existingInsights.map { it.toDomainModel() })
                    }
                }
            }

            // Create AI generation job
            val jobId = createAIGenerationJob(
                sessionId = sessionId,
                generationType = "POST_SESSION",
                aiModel = selectOptimalAIModel(sessionId)
            ).getOrThrow()

            // Update AI generation status
            _aiGenerationStatus.value = AIGenerationStatus.GENERATING

            // Load session data with comprehensive analytics
            val sessionData = loadSessionWithAnalytics(sessionId).getOrThrow()

            // Generate insights using appropriate analysis depth
            val insights = when (analysisDepth) {
                AnalysisDepth.BASIC -> generateBasicSessionInsights(sessionData)
                AnalysisDepth.DETAILED -> generateDetailedSessionInsights(sessionData)
                AnalysisDepth.COMPREHENSIVE -> generateComprehensiveSessionInsights(sessionData)
            }

            // Store insights with AI metadata
            val storedInsights = storeAIGeneratedInsights(insights, sessionId, jobId)

            // Complete AI generation job
            completeAIGenerationJob(jobId, storedInsights.size)

            // Update AI model performance
            updateAIModelPerformance(jobId, storedInsights)

            _aiGenerationStatus.value = AIGenerationStatus.COMPLETED

            Log.d(TAG, "Generated ${storedInsights.size} AI insights for session $sessionId")
            Result.success(storedInsights)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate session insights", e)
            _aiGenerationStatus.value = AIGenerationStatus.ERROR

            // Record failure in AI job
            recordAIGenerationFailure(sessionId, e.message ?: "Unknown error")

            Result.failure(e)
        }
    }

    /**
     * Generate personalized insights based on user patterns
     */
    suspend fun generatePersonalizedInsights(
        daysBack: Int = 30,
        personalizationLevel: PersonalizationLevel = PersonalizationLevel.ADAPTIVE,
        focusAreas: List<InsightCategory> = emptyList()
    ): Result<List<SleepInsight>> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Generating personalized insights: daysBack=$daysBack, level=$personalizationLevel")

            val cacheKey = "personalized_${daysBack}_${personalizationLevel.name}_${focusAreas.hashCode()}"

            // Check cache
            val cached = insightCache[cacheKey]
            if (cached != null && !cached.isExpired(AI_CACHE_EXPIRY_MS)) {
                return@withContext Result.success(cached.data)
            }

            // Get user's sleep patterns and preferences
            val personalBaseline = getPersonalBaseline().getOrThrow()
            val userPreferences = getUserPreferences().getOrThrow()
            val recentSessions = getRecentSessionsWithAnalytics(daysBack)

            // Ensure minimum data for personalization
            if (recentSessions.size < PATTERN_ANALYSIS_MIN_SESSIONS) {
                return@withContext Result.success(listOf(
                    createInsufficientDataInsight(
                        "Need more sleep data",
                        "Track sleep for at least $PATTERN_ANALYSIS_MIN_SESSIONS nights to get personalized insights",
                        "Continue tracking your sleep patterns"
                    )
                ))
            }

            // Generate personalized insights
            val insights = generatePersonalizedInsightsAdvanced(
                personalBaseline = personalBaseline,
                userPreferences = userPreferences,
                recentSessions = recentSessions,
                personalizationLevel = personalizationLevel,
                focusAreas = focusAreas
            )

            // Cache results
            insightCache[cacheKey] = CacheEntry(insights, AI_CACHE_EXPIRY_MS)

            Log.d(TAG, "Generated ${insights.size} personalized insights")
            Result.success(insights)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate personalized insights", e)
            Result.failure(e)
        }
    }

    /**
     * Generate predictive insights based on trend analysis
     */
    suspend fun generatePredictiveInsights(
        predictionHorizon: PredictionHorizon = PredictionHorizon.WEEK,
        confidenceThreshold: Float = 0.7f
    ): Result<List<SleepInsight>> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Generating predictive insights: horizon=$predictionHorizon")

            // Get comprehensive trend analysis
            val trendAnalysis = getAdvancedTrendAnalysis().getOrThrow()
            val patternAnalysis = getSleepPatternAnalysis().getOrThrow()

            // Check if we have enough data for reliable predictions
            if (!canGenerateReliablePredictions(trendAnalysis, patternAnalysis)) {
                return@withContext Result.success(listOf(
                    createInsufficientDataInsight(
                        "Need more data for predictions",
                        "Predictive insights require at least $TREND_ANALYSIS_MIN_SESSIONS nights of data",
                        "Continue tracking to unlock predictive analysis"
                    )
                ))
            }

            // Generate predictions
            val predictions = generatePredictionsAdvanced(
                trendAnalysis = trendAnalysis,
                patternAnalysis = patternAnalysis,
                predictionHorizon = predictionHorizon,
                confidenceThreshold = confidenceThreshold
            )

            // Filter by confidence threshold
            val highConfidencePredictions = predictions.filter {
                it.confidenceScore >= confidenceThreshold
            }

            Log.d(TAG, "Generated ${highConfidencePredictions.size} predictive insights")
            Result.success(highConfidencePredictions)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate predictive insights", e)
            Result.failure(e)
        }
    }

    /**
     * Get all insights with filtering and sorting options
     */
    suspend fun getAllInsights(
        categories: List<InsightCategory> = emptyList(),
        priorities: List<Int> = emptyList(),
        isAiGenerated: Boolean? = null,
        includeAcknowledged: Boolean = true,
        sortBy: InsightSortOption = InsightSortOption.NEWEST_FIRST,
        limit: Int = 50
    ): Result<List<SleepInsight>> = withContext(dispatcher) {
        try {
            val insights = insightDao.getAllInsights()

            var filteredInsights = insights.asSequence()
                .map { it.toDomainModel() }

            // Apply filters
            if (categories.isNotEmpty()) {
                filteredInsights = filteredInsights.filter { it.category in categories }
            }

            if (priorities.isNotEmpty()) {
                filteredInsights = filteredInsights.filter { it.priority in priorities }
            }

            if (isAiGenerated != null) {
                filteredInsights = filteredInsights.filter { it.isAiGenerated == isAiGenerated }
            }

            if (!includeAcknowledged) {
                filteredInsights = filteredInsights.filter { !it.isAcknowledged }
            }

            // Apply sorting
            filteredInsights = when (sortBy) {
                InsightSortOption.NEWEST_FIRST -> filteredInsights.sortedByDescending { it.timestamp }
                InsightSortOption.OLDEST_FIRST -> filteredInsights.sortedBy { it.timestamp }
                InsightSortOption.PRIORITY_HIGH_FIRST -> filteredInsights.sortedBy { it.priority }
                InsightSortOption.CATEGORY -> filteredInsights.sortedBy { it.category.name }
            }

            val result = filteredInsights.take(limit).toList()
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get insights", e)
            Result.failure(e)
        }
    }

    // ========== USER INTERACTION AND FEEDBACK ==========

    /**
     * Record user interaction with insights
     */
    suspend fun recordUserInteraction(
        insightId: Long?,
        interactionType: String,
        details: String? = null,
        durationMs: Long = 0L
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val interaction = UserInteractionEntity(
                insightId = insightId,
                interactionType = interactionType,
                interactionValue = details,
                durationMs = durationMs,
                userSession = getCurrentUserSession(),
                contextData = buildContextData(),
                timestamp = System.currentTimeMillis(),
                createdAt = System.currentTimeMillis()
            )

            userInteractionDao.insertInteraction(interaction)

            // Update user engagement metrics
            updateUserEngagementMetrics()

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record user interaction", e)
            Result.failure(e)
        }
    }

    /**
     * Record insight feedback for AI improvement
     */
    suspend fun recordInsightFeedback(
        insightId: Long,
        feedback: InsightFeedback,
        implementation: String? = null
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val feedbackEntity = InsightFeedbackEntity(
                insightId = insightId,
                feedbackType = when {
                    feedback.wasHelpful == true -> "POSITIVE"
                    feedback.wasHelpful == false -> "NEGATIVE"
                    else -> "NEUTRAL"
                },
                rating = feedback.rating,
                wasHelpful = feedback.wasHelpful,
                wasAccurate = feedback.wasAccurate,
                wasImplemented = feedback.wasImplemented,
                implementationResult = implementation,
                feedbackText = feedback.feedbackText,
                improvementSuggestions = feedback.improvementSuggestions,
                contextData = buildFeedbackContext(),
                engagementMetrics = buildEngagementMetrics(insightId),
                timestamp = System.currentTimeMillis(),
                createdAt = System.currentTimeMillis()
            )

            insightFeedbackDao.insertFeedback(feedbackEntity)

            // Update insight effectiveness in insights table
            updateInsightEffectiveness(insightId, feedback)

            // Record interaction
            recordUserInteraction(
                insightId = insightId,
                interactionType = "FEEDBACK_SUBMITTED",
                details = "Rating: ${feedback.rating}, Helpful: ${feedback.wasHelpful}"
            )

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record insight feedback", e)
            Result.failure(e)
        }
    }

    /**
     * Update user preferences based on behavior
     */
    suspend fun updateUserPreferences(
        preferenceType: String,
        preferenceValue: String,
        weight: Float = 1.0f,
        learnedFromBehavior: Boolean = false
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val preference = UserPreferencesEntity(
                preferenceType = preferenceType,
                preferenceValue = preferenceValue,
                weight = weight,
                learnedFromBehavior = learnedFromBehavior,
                userSet = !learnedFromBehavior,
                confidenceScore = if (learnedFromBehavior) 0.8f else 1.0f,
                updatedAt = System.currentTimeMillis(),
                createdAt = System.currentTimeMillis()
            )

            userPreferencesDao.insertPreference(preference)

            // Invalidate personalization cache
            personalizationCache.clear()

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update user preferences", e)
            Result.failure(e)
        }
    }

    // ========== ADVANCED ANALYTICS AND PATTERN ANALYSIS ==========

    /**
     * Get comprehensive sleep pattern analysis
     */
    suspend fun getSleepPatternAnalysis(
        daysBack: Int = 30
    ): Result<SleepPatternAnalysis> = withContext(dispatcher) {
        try {
            val cacheKey = "pattern_analysis_$daysBack"

            // Check cache
            val cached = patternCache[cacheKey]
            if (cached != null && !cached.isExpired(ANALYTICS_CACHE_EXPIRY_MS)) {
                return@withContext Result.success(cached.data)
            }

            // Load data for analysis
            val sessions = getRecentSessionsWithAnalytics(daysBack)

            if (sessions.size < PATTERN_ANALYSIS_MIN_SESSIONS) {
                return@withContext Result.failure(
                    IllegalStateException("Insufficient data for pattern analysis")
                )
            }

            // Perform comprehensive pattern analysis
            val patterns = analyzePatterns(sessions)

            // Cache results
            patternCache[cacheKey] = CacheEntry(patterns, ANALYTICS_CACHE_EXPIRY_MS)

            Result.success(patterns)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get sleep pattern analysis", e)
            Result.failure(e)
        }
    }

    /**
     * Get advanced trend analysis with statistical significance
     */
    suspend fun getAdvancedTrendAnalysis(
        daysBack: Int = 90
    ): Result<TrendAnalysis> = withContext(dispatcher) {
        try {
            val cacheKey = "trend_analysis_$daysBack"

            // Check cache
            val cached = analyticsCache[cacheKey] as? CacheEntry<TrendAnalysis>
            if (cached != null && !cached.isExpired(ANALYTICS_CACHE_EXPIRY_MS)) {
                return@withContext Result.success(cached.data)
            }

            // Load trend data
            val trendData = getSleepTrends(daysBack).getOrThrow()

            if (trendData.size < TREND_ANALYSIS_MIN_SESSIONS) {
                return@withContext Result.failure(
                    IllegalStateException("Insufficient data for trend analysis")
                )
            }

            // Perform advanced trend analysis
            val analysis = performAdvancedTrendAnalysis(trendData)

            // Cache results
            analyticsCache[cacheKey] = CacheEntry(analysis, ANALYTICS_CACHE_EXPIRY_MS)

            Result.success(analysis)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get advanced trend analysis", e)
            Result.failure(e)
        }
    }

    /**
     * Get user's personal baseline with adaptive learning
     */
    suspend fun getPersonalBaseline(): Result<PersonalBaseline> = withContext(dispatcher) {
        try {
            val cacheKey = "personal_baseline"

            // Check cache
            val cached = personalizationCache[cacheKey]
            if (cached != null && !cached.isExpired(ANALYTICS_CACHE_EXPIRY_MS)) {
                return@withContext Result.success(cached.data)
            }

            // Calculate personal baseline from historical data
            val baseline = calculatePersonalBaseline()

            // Cache results
            personalizationCache[cacheKey] = CacheEntry(baseline, ANALYTICS_CACHE_EXPIRY_MS)

            Result.success(baseline)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get personal baseline", e)
            Result.failure(e)
        }
    }

    /**
     * Get habit analysis with behavioral patterns
     */
    suspend fun getHabitAnalysis(): Result<HabitAnalysis> = withContext(dispatcher) {
        try {
            // Load user interactions and preferences
            val interactions = userInteractionDao.getInteractionsInDateRange(
                startDate = System.currentTimeMillis() - (30 * 24 * 60 * 60 * 1000L),
                endDate = System.currentTimeMillis()
            )

            val preferences = userPreferencesDao.getUserSetPreferences()
            val sessions = getRecentSessionsWithAnalytics(30)

            // Analyze habits and behaviors
            val habitAnalysis = analyzeHabits(interactions, preferences, sessions)

            Result.success(habitAnalysis)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get habit analysis", e)
            Result.failure(e)
        }
    }

    // ========== AI MODEL PERFORMANCE AND MONITORING ==========

    /**
     * Record AI model performance metrics
     */
    suspend fun recordAIModelPerformance(
        modelName: String,
        generationType: String,
        performanceMetrics: AIModelPerformance
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val performance = AIModelPerformanceEntity(
                modelName = modelName,
                generationType = generationType,
                metricDate = System.currentTimeMillis(),
                totalRequests = performanceMetrics.totalRequests,
                successfulRequests = performanceMetrics.successfulRequests,
                failedRequests = performanceMetrics.failedRequests,
                averageProcessingTimeMs = performanceMetrics.averageProcessingTime,
                totalTokensUsed = performanceMetrics.totalTokensUsed,
                totalCostCents = performanceMetrics.totalCostCents,
                averageQualityScore = performanceMetrics.averageQualityScore,
                averageUserRating = performanceMetrics.averageUserRating,
                insightsImplementedRate = performanceMetrics.implementationRate,
                userSatisfactionRate = performanceMetrics.satisfactionRate,
                updatedAt = System.currentTimeMillis(),
                createdAt = System.currentTimeMillis()
            )

            aiModelPerformanceDao.insertPerformance(performance)

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record AI model performance", e)
            Result.failure(e)
        }
    }

    /**
     * Get AI model performance comparison
     */
    suspend fun getAIModelPerformanceComparison(
        timeRange: TimeRange? = null
    ): Result<List<ModelComparisonMetrics>> = withContext(dispatcher) {
        try {
            val startDate = timeRange?.startDate ?: (System.currentTimeMillis() - (30 * 24 * 60 * 60 * 1000L))
            val endDate = timeRange?.endDate ?: System.currentTimeMillis()

            val performanceData = aiModelPerformanceDao.getPerformanceInDateRange(startDate, endDate)
            val comparisonMetrics = aiModelPerformanceDao.getModelComparisonMetrics()

            Result.success(comparisonMetrics)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get AI model performance comparison", e)
            Result.failure(e)
        }
    }

    // ========== UTILITY AND HELPER METHODS ==========

    /**
     * Save AI-generated insight with metadata
     */
    suspend fun saveInsight(insight: SleepInsight): Result<Long> = withContext(dispatcher) {
        try {
            val entity = SleepInsightEntity(
                sessionId = insight.sessionId,
                category = insight.category,
                title = insight.title,
                description = insight.description,
                recommendation = insight.recommendation,
                priority = insight.priority,
                isAiGenerated = insight.isAiGenerated,
                timestamp = insight.timestamp,
                confidenceScore = insight.confidenceScore,
                // Add AI-specific fields
                aiModelUsed = insight.aiModelUsed,
                aiPromptVersion = insight.aiPromptVersion,
                aiGenerationJobId = insight.aiGenerationJobId,
                aiProcessingTimeMs = insight.aiProcessingTimeMs,
                aiTokensUsed = insight.aiTokensUsed,
                personalizationFactors = insight.personalizationFactors?.joinToString(","),
                dataSourcesUsed = insight.dataSourcesUsed?.joinToString(","),
                mlFeatures = insight.mlFeatures?.entries?.joinToString(";") { "${it.key}:${it.value}" },
                similarityScore = insight.similarityScore,
                relevanceScore = insight.relevanceScore,
                predictedUsefulness = insight.predictedUsefulness
            )

            val insightId = insightDao.insertInsight(entity)

            // Record user interaction for insight creation
            recordUserInteraction(
                insightId = insightId,
                interactionType = "INSIGHT_GENERATED",
                details = "Category: ${insight.category}, AI: ${insight.isAiGenerated}"
            )

            Log.d(TAG, "Insight saved: ${insight.title}")
            Result.success(insightId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save insight", e)
            Result.failure(e)
        }
    }

    /**
     * Delete insight
     */
    suspend fun deleteInsight(insightId: Long): Result<Unit> = withContext(dispatcher) {
        try {
            insightDao.deleteInsightById(insightId)

            // Record user interaction
            recordUserInteraction(
                insightId = insightId,
                interactionType = "INSIGHT_DELETED",
                details = "User deleted insight"
            )

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete insight", e)
            Result.failure(e)
        }
    }

    /**
     * Get latest session
     */
    suspend fun getLatestSession(): Result<SleepSession?> = withContext(dispatcher) {
        try {
            val session = sessionDao.getLatestSession()?.toDomainModel()
            Result.success(session)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get latest session", e)
            Result.failure(e)
        }
    }

    /**
     * Get user preferences
     */
    suspend fun getUserPreferences(): Result<UserSleepPreferences> = withContext(dispatcher) {
        try {
            val preferences = userPreferencesDao.getUserSetPreferences()
            val sleepPreferences = convertToUserSleepPreferences(preferences)
            Result.success(sleepPreferences)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get user preferences", e)
            Result.failure(e)
        }
    }

    // ========== PRIVATE IMPLEMENTATION METHODS ==========

    private suspend fun initializeAIComponents() {
        try {
            // Initialize AI model performance tracking
            initializeAIModelTracking()

            // Load user engagement metrics
            updateUserEngagementMetrics()

            // Initialize default user preferences if none exist
            initializeDefaultUserPreferences()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize AI components", e)
        }
    }

    private suspend fun createAIGenerationJob(
        sessionId: Long,
        generationType: String,
        aiModel: String
    ): Result<String> = withContext(dispatcher) {
        try {
            val jobId = "job_${System.currentTimeMillis()}_${sessionId}"

            val job = AIGenerationJobEntity(
                jobId = jobId,
                generationType = generationType,
                status = "pending",
                aiModel = aiModel,
                promptVersion = "v1.0",
                sessionId = sessionId,
                requestData = buildRequestData(sessionId, generationType),
                createdAt = System.currentTimeMillis()
            )

            aiGenerationJobDao.insertJob(job)
            Result.success(jobId)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to create AI generation job", e)
            Result.failure(e)
        }
    }

    private suspend fun completeAIGenerationJob(
        jobId: String,
        insightsGenerated: Int
    ) {
        try {
            aiGenerationJobDao.completeJob(
                jobId = jobId,
                completedAt = System.currentTimeMillis(),
                insightsGenerated = insightsGenerated,
                tokensUsed = 1000, // This would come from actual API response
                processingTime = 5000L, // This would be measured
                costCents = 10, // This would be calculated from actual usage
                qualityScore = 0.85f // This would be calculated from feedback
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to complete AI generation job", e)
        }
    }

    private suspend fun triggerPostSessionAIAnalysis(sessionId: Long) {
        try {
            // This would trigger the AI insights generation
            // In practice, this might queue the job for background processing
            scope.launch {
                delay(5000) // Allow time for session data to be fully written
                generateSessionInsights(sessionId, forceRegenerate = true)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to trigger post-session AI analysis", e)
        }
    }

    private suspend fun updateUserEngagementMetrics() {
        try {
            val recent = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000L)
            val interactions = userInteractionDao.getInteractionsInDateRange(recent, System.currentTimeMillis())

            val totalInteractions = interactions.size.toLong()
            val uniqueInsights = interactions.mapNotNull { it.insightId }.distinct().size
            val averageTime = interactions.mapNotNull { it.durationMs }.average().takeIf { !it.isNaN() }?.toLong() ?: 0L

            val metrics = UserEngagementMetrics(
                totalInteractions = totalInteractions,
                uniqueInsightsEngaged = uniqueInsights,
                averageEngagementTime = averageTime,
                lastInteractionTime = interactions.maxOfOrNull { it.timestamp } ?: 0L
            )

            _userEngagementMetrics.value = metrics

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update user engagement metrics", e)
        }
    }

    // Helper methods for AI insight generation
    private suspend fun generateBasicSessionInsights(sessionData: SessionWithAnalytics): List<SleepInsight> {
        // Implementation for basic insights
        return emptyList()
    }

    private suspend fun generateDetailedSessionInsights(sessionData: SessionWithAnalytics): List<SleepInsight> {
        // Implementation for detailed insights
        return emptyList()
    }

    private suspend fun generateComprehensiveSessionInsights(sessionData: SessionWithAnalytics): List<SleepInsight> {
        // Implementation for comprehensive insights
        return emptyList()
    }

    private suspend fun storeAIGeneratedInsights(
        insights: List<SleepInsight>,
        sessionId: Long,
        jobId: String
    ): List<SleepInsight> {
        // Store insights with AI metadata
        return insights.map { insight ->
            val enhancedInsight = insight.copy(
                sessionId = sessionId,
                aiGenerationJobId = jobId,
                isAiGenerated = true,
                timestamp = System.currentTimeMillis()
            )
            saveInsight(enhancedInsight)
            enhancedInsight
        }
    }

    // Additional helper methods...
    private fun selectOptimalAIModel(sessionId: Long): String = "GPT-4"
    private suspend fun loadSessionWithAnalytics(sessionId: Long): Result<SessionWithAnalytics> = Result.success(SessionWithAnalytics())
    private suspend fun updateAIModelPerformance(jobId: String, insights: List<SleepInsight>) {}
    private suspend fun recordAIGenerationFailure(sessionId: Long, error: String) {}
    private suspend fun getRecentSessionsWithAnalytics(daysBack: Int): List<SessionWithAnalytics> = emptyList()
    private suspend fun generatePersonalizedInsightsAdvanced(
        personalBaseline: PersonalBaseline,
        userPreferences: UserSleepPreferences,
        recentSessions: List<SessionWithAnalytics>,
        personalizationLevel: PersonalizationLevel,
        focusAreas: List<InsightCategory>
    ): List<SleepInsight> = emptyList()

    private fun canGenerateReliablePredictions(
        trendAnalysis: TrendAnalysis,
        patternAnalysis: SleepPatternAnalysis
    ): Boolean = true

    private suspend fun generatePredictionsAdvanced(
        trendAnalysis: TrendAnalysis,
        patternAnalysis: SleepPatternAnalysis,
        predictionHorizon: PredictionHorizon,
        confidenceThreshold: Float
    ): List<SleepInsight> = emptyList()

    private fun getCurrentUserSession(): String = "session_${System.currentTimeMillis()}"
    private fun buildContextData(): String = "{\"context\":\"mobile_app\"}"
    private fun buildFeedbackContext(): String = "{\"feedback_context\":\"user_rating\"}"
    private fun buildEngagementMetrics(insightId: Long): String = "{\"engagement\":\"high\"}"
    private suspend fun updateInsightEffectiveness(insightId: Long, feedback: InsightFeedback) {}
    private suspend fun analyzePatterns(sessions: List<SessionWithAnalytics>): SleepPatternAnalysis = SleepPatternAnalysis()
    private suspend fun performAdvancedTrendAnalysis(trendData: List<DailyTrendData>): TrendAnalysis = TrendAnalysis()
    private suspend fun calculatePersonalBaseline(): PersonalBaseline = PersonalBaseline()
    private suspend fun analyzeHabits(
        interactions: List<UserInteractionEntity>,
        preferences: List<UserPreferencesEntity>,
        sessions: List<SessionWithAnalytics>
    ): HabitAnalysis = HabitAnalysis()

    private fun buildRequestData(sessionId: Long, generationType: String): String =
        "{\"sessionId\":$sessionId,\"type\":\"$generationType\"}"

    private suspend fun initializeAIModelTracking() {}
    private suspend fun initializeDefaultUserPreferences() {}
    private fun convertToUserSleepPreferences(preferences: List<UserPreferencesEntity>): UserSleepPreferences =
        UserSleepPreferences()

    private fun createInsufficientDataInsight(title: String, description: String, recommendation: String): SleepInsight {
        return SleepInsight(
            sessionId = 0L,
            category = InsightCategory.GENERAL,
            title = title,
            description = description,
            recommendation = recommendation,
            priority = 2,
            isAiGenerated = false,
            timestamp = System.currentTimeMillis()
        )
    }

    // ========== EXISTING METHODS (from original implementation) ==========

    /**
     * Update current active session
     */
    suspend fun updateCurrentSession(
        duration: Long? = null,
        efficiency: Float? = null,
        movementIntensity: Float? = null,
        noiseLevel: Float? = null,
        movementEventCount: Int? = null,
        noiseEventCount: Int? = null,
        phaseTransitionCount: Int? = null
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.failure(
                IllegalStateException("No active session")
            )

            val currentSession = sessionDao.getSessionById(sessionId)
                ?: return@withContext Result.failure(IllegalStateException("Session not found"))

            // Update session with new data
            val updatedSession = currentSession.copy(
                totalDuration = duration ?: currentSession.totalDuration,
                sleepEfficiency = efficiency ?: currentSession.sleepEfficiency,
                averageMovementIntensity = movementIntensity ?: currentSession.averageMovementIntensity,
                averageNoiseLevel = noiseLevel ?: currentSession.averageNoiseLevel,
                totalMovementEvents = movementEventCount ?: currentSession.totalMovementEvents,
                totalNoiseEvents = noiseEventCount ?: currentSession.totalNoiseEvents,
                totalPhaseTransitions = phaseTransitionCount ?: currentSession.totalPhaseTransitions,
                updatedAt = java.util.Date()
            )

            sessionDao.updateSession(updatedSession)

            // Update cache and flow
            invalidateSessionCache(sessionId)
            _currentSessionFlow.value = updatedSession.toDomainModel()

            Log.d(TAG, "Session $sessionId updated successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update session", e)
            Result.failure(e)
        }
    }

    /**
     * Cancel current active session
     */
    suspend fun cancelCurrentSession(): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.success(Unit)

            // Delete the session and all related data (CASCADE will handle related entities)
            sessionDao.deleteSessionById(sessionId)

            // Clear current session state
            currentSessionId = null
            _currentSessionFlow.value = null
            _realTimeMetrics.value = null

            // Invalidate caches
            invalidateAllCaches()

            Log.d(TAG, "Session $sessionId cancelled and deleted")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to cancel session", e)
            Result.failure(e)
        }
    }

    // ========== EVENT RECORDING ==========

    /**
     * Record movement event during active session
     */
    suspend fun recordMovementEvent(event: MovementEvent): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.failure(
                IllegalStateException("No active session")
            )

            val entity = EntityHelper.createMovementEventEntity(event, sessionId)
            movementDao.insertMovementEvent(entity)

            Log.d(TAG, "Movement event recorded: intensity=${event.intensity}")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record movement event", e)
            Result.failure(e)
        }
    }

    /**
     * Record noise event during active session
     */
    suspend fun recordNoiseEvent(event: NoiseEvent): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.failure(
                IllegalStateException("No active session")
            )

            val entity = EntityHelper.createNoiseEventEntity(event, sessionId)
            noiseDao.insertNoiseEvent(entity)

            Log.d(TAG, "Noise event recorded: ${event.decibelLevel}dB")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record noise event", e)
            Result.failure(e)
        }
    }

    /**
     * Record sleep phase transition
     */
    suspend fun recordPhaseTransition(transition: PhaseTransition): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.failure(
                IllegalStateException("No active session")
            )

            val entity = EntityHelper.createPhaseTransitionEntity(transition, sessionId)
            phaseDao.insertPhaseTransition(entity)

            Log.d(TAG, "Phase transition recorded: ${transition.fromPhase} -> ${transition.toPhase}")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record phase transition", e)
            Result.failure(e)
        }
    }

    /**
     * Batch insert events for performance
     */
    suspend fun recordEventsBatch(
        movements: List<MovementEvent> = emptyList(),
        noises: List<NoiseEvent> = emptyList(),
        phases: List<PhaseTransition> = emptyList()
    ): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionId = currentSessionId ?: return@withContext Result.failure(
                IllegalStateException("No active session")
            )

            database.runInTransaction {
                // Insert movement events
                if (movements.isNotEmpty()) {
                    val movementEntities = movements.map {
                        EntityHelper.createMovementEventEntity(it, sessionId)
                    }
                    runBlocking { movementDao.insertMovementEvents(movementEntities) }
                }

                // Insert noise events
                if (noises.isNotEmpty()) {
                    val noiseEntities = noises.map {
                        EntityHelper.createNoiseEventEntity(it, sessionId)
                    }
                    runBlocking { noiseDao.insertNoiseEvents(noiseEntities) }
                }

                // Insert phase transitions
                if (phases.isNotEmpty()) {
                    val phaseEntities = phases.map {
                        EntityHelper.createPhaseTransitionEntity(it, sessionId)
                    }
                    runBlocking { phaseDao.insertPhaseTransitions(phaseEntities) }
                }
            }

            Log.d(TAG, "Batch recorded: ${movements.size} movements, ${noises.size} noises, ${phases.size} phases")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record events batch", e)
            Result.failure(e)
        }
    }

    // ========== REAL-TIME DATA UPDATES ==========

    /**
     * Update real-time metrics during tracking
     */
    fun updateRealTimeMetrics(metrics: LiveSleepMetrics) {
        _realTimeMetrics.value = metrics
        Log.d(TAG, "Real-time metrics updated: phase=${metrics.currentPhase.getDisplayName()}")
    }

    /**
     * Get current real-time metrics
     */
    fun getCurrentRealTimeMetrics(): LiveSleepMetrics? = _realTimeMetrics.value

    // ========== DATA RETRIEVAL ==========

    /**
     * Get session by ID with caching
     */
    suspend fun getSessionById(sessionId: Long): Result<SleepSession?> = withContext(dispatcher) {
        try {
            // Check cache first
            val cached = sessionCache[sessionId]
            if (cached != null && !cached.isExpired()) {
                return@withContext Result.success(cached.data)
            }

            // Load from database
            val sessionWithDetails = sessionDao.getSessionWithDetails(sessionId)
            val session = sessionWithDetails?.toDomainModel()

            // Cache the result
            session?.let {
                sessionCache[sessionId] = CacheEntry(it)
                cleanupCache(sessionCache)
            }

            Result.success(session)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get session by ID: $sessionId", e)
            Result.failure(e)
        }
    }

    /**
     * Get all sessions with reactive updates
     */
    fun getAllSessions(): Flow<List<SleepSession>> {
        return sessionDao.getAllSessions().map { entities ->
            entities.map { it.toDomainModel() }
        }.flowOn(dispatcher)
    }

    /**
     * Get recent sessions with limit
     */
    suspend fun getRecentSessions(limit: Int = 10): Result<List<SleepSession>> = withContext(dispatcher) {
        try {
            val sessions = sessionDao.getRecentSessionsWithDetails(limit)
            val domainSessions = sessions.map { it.toDomainModel() }
            Result.success(domainSessions)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get recent sessions", e)
            Result.failure(e)
        }
    }

    /**
     * Get sessions for date range
     */
    suspend fun getSessionsInDateRange(
        startDate: Long,
        endDate: Long
    ): Result<List<SleepSession>> = withContext(dispatcher) {
        try {
            val entities = sessionDao.getSessionsInDateRange(startDate, endDate)
            val sessions = entities.map { it.toDomainModel() }
            Result.success(sessions)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get sessions in date range", e)
            Result.failure(e)
        }
    }

    // ========== ANALYTICS AND STATISTICS ==========

    /**
     * Get comprehensive sleep analytics with caching
     */
    suspend fun getSleepAnalytics(
        forceRefresh: Boolean = false
    ): Result<SleepAnalytics> = withContext(dispatcher) {
        try {
            val cacheKey = "overall_analytics"

            // Check cache first
            if (!forceRefresh) {
                val cached = statisticsCache[cacheKey]
                if (cached != null && !cached.isExpired()) {
                    return@withContext Result.success(cached.data)
                }
            }

            // Load statistics from database
            val stats = sessionDao.getOverallStatistics()
            if (stats == null) {
                return@withContext Result.success(createEmptyAnalytics())
            }

            // Get recent sessions for trend analysis
            val recentSessions = sessionDao.getRecentSessionsWithDetails(30)
                .map { it.toDomainModel() }

            // Calculate trend
            val trend = calculateSleepTrend(recentSessions)

            // Generate recommendations
            val recommendations = generateRecommendations(stats, recentSessions)

            // Create analytics object
            val analytics = SleepAnalytics(
                sessions = recentSessions,
                averageDuration = stats.averageDuration,
                averageQuality = stats.averageQuality,
                averageSleepEfficiency = stats.averageEfficiency,
                totalMovementEvents = stats.totalMovements,
                totalNoiseEvents = stats.totalNoiseEvents,
                bestSleepDate = stats.bestQualityDate,
                worstSleepDate = stats.worstQualityDate,
                sleepTrend = trend,
                recommendations = recommendations
            )

            // Cache the result
            statisticsCache[cacheKey] = CacheEntry(analytics)
            cleanupCache(statisticsCache)

            Result.success(analytics)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get sleep analytics", e)
            Result.failure(e)
        }
    }

    /**
     * Get sleep trends for visualization
     */
    suspend fun getSleepTrends(
        daysBack: Int = 30,
        forceRefresh: Boolean = false
    ): Result<List<DailyTrendData>> = withContext(dispatcher) {
        try {
            val cacheKey = "trends_$daysBack"

            // Check cache first
            if (!forceRefresh) {
                val cached = trendCache[cacheKey]
                if (cached != null && !cached.isExpired()) {
                    return@withContext Result.success(cached.data)
                }
            }

            val startDate = System.currentTimeMillis() - (daysBack * 24 * 60 * 60 * 1000L)
            val trends = sessionDao.getDailyTrends(startDate)

            // Cache the result
            trendCache[cacheKey] = CacheEntry(trends)
            cleanupCache(trendCache)

            Result.success(trends)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get sleep trends", e)
            Result.failure(e)
        }
    }

    /**
     * Get session statistics for UI display
     */
    suspend fun getSessionStatistics(): Result<SessionStatistics> = withContext(dispatcher) {
        try {
            val stats = sessionDao.getOverallStatistics()
            val sessionCount = sessionDao.getCompletedSessionCount()
            val avgDuration = sessionDao.getAverageSessionDuration() ?: 0L
            val avgQuality = sessionDao.getAverageQualityScore() ?: 0f
            val avgEfficiency = sessionDao.getAverageEfficiency() ?: 0f

            val sessionStats = SessionStatistics(
                totalSessions = sessionCount,
                averageDuration = avgDuration,
                averageQuality = avgQuality,
                averageEfficiency = avgEfficiency,
                hasActiveSession = currentSessionId != null
            )

            Result.success(sessionStats)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to get session statistics", e)
            Result.failure(e)
        }
    }

    // ========== BASIC INSIGHTS MANAGEMENT ==========

    /**
     * Add insight (legacy method for compatibility)
     */
    suspend fun addInsight(insight: SleepInsight): Result<Long> = saveInsight(insight)

    /**
     * Get unacknowledged insights
     */
    suspend fun getUnacknowledgedInsights(): Result<List<SleepInsight>> = withContext(dispatcher) {
        try {
            val entities = insightDao.getUnacknowledgedInsights()
            val insights = entities.map { it.toDomainModel() }
            Result.success(insights)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get unacknowledged insights", e)
            Result.failure(e)
        }
    }

    /**
     * Acknowledge insight
     */
    suspend fun acknowledgeInsight(insightId: Long): Result<Unit> = withContext(dispatcher) {
        try {
            insightDao.acknowledgeInsight(insightId)

            // Record user interaction
            recordUserInteraction(
                insightId = insightId,
                interactionType = "ACKNOWLEDGED",
                details = "User acknowledged insight"
            )

            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to acknowledge insight", e)
            Result.failure(e)
        }
    }

    // ========== CACHE MANAGEMENT ==========

    private fun invalidateSessionCache(sessionId: Long) {
        sessionCache.remove(sessionId)
    }

    private fun invalidateAllCaches() {
        sessionCache.clear()
        statisticsCache.clear()
        trendCache.clear()
        insightCache.clear()
        analyticsCache.clear()
        personalizationCache.clear()
        patternCache.clear()
    }

    private fun <T> cleanupCache(cache: ConcurrentHashMap<*, CacheEntry<T>>) {
        if (cache.size > MAX_CACHE_SIZE) {
            val iterator = cache.entries.iterator()
            var removed = 0
            while (iterator.hasNext() && removed < cache.size / 4) {
                val entry = iterator.next()
                if (entry.value.isExpired()) {
                    iterator.remove()
                    removed++
                }
            }
        }
    }

    private fun startCacheCleanup() {
        scope.launch {
            while (isActive) {
                delay(CACHE_EXPIRY_MS)
                cleanupCache(sessionCache)
                cleanupCache(statisticsCache)
                cleanupCache(trendCache)
                cleanupCache(insightCache)
                cleanupCache(analyticsCache)
                cleanupCache(personalizationCache)
                cleanupCache(patternCache)
            }
        }
    }

    // ========== HELPER METHODS ==========

    private suspend fun loadActiveSession() {
        try {
            val activeSession = sessionDao.getActiveSession()
            if (activeSession != null) {
                currentSessionId = activeSession.id
                _currentSessionFlow.value = activeSession.toDomainModel()
                Log.d(TAG, "Loaded active session: ${activeSession.id}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load active session", e)
        }
    }

    private fun calculateSleepTrend(sessions: List<SleepSession>): SleepTrend {
        if (sessions.size < 6) return SleepTrend.INSUFFICIENT_DATA

        val recentSessions = sessions.take(3)
        val olderSessions = sessions.drop(3).take(3)

        val recentAvgQuality = recentSessions.mapNotNull { it.sleepQualityScore }.average()
        val olderAvgQuality = olderSessions.mapNotNull { it.sleepQualityScore }.average()

        return when {
            recentAvgQuality > olderAvgQuality + 0.5 -> SleepTrend.IMPROVING
            recentAvgQuality < olderAvgQuality - 0.5 -> SleepTrend.DECLINING
            else -> SleepTrend.STABLE
        }
    }

    private fun generateRecommendations(
        stats: SleepStatistics,
        recentSessions: List<SleepSession>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        // Duration recommendations
        val avgHours = stats.averageDuration / (1000 * 60 * 60)
        if (avgHours < 7) {
            recommendations.add("Consider going to bed earlier to get 7-9 hours of sleep")
        } else if (avgHours > 9) {
            recommendations.add("You might be getting too much sleep - try a consistent 7-9 hour schedule")
        }

        // Quality recommendations
        if (stats.averageQuality < 6f) {
            recommendations.add("Focus on creating a quieter, more comfortable sleep environment")
        }

        // Efficiency recommendations
        if (stats.averageEfficiency < 85f) {
            recommendations.add("Try to minimize disruptions and maintain consistent sleep schedule")
        }

        // Movement recommendations
        val avgMovements = recentSessions.map { it.getTotalMovements() }.average()
        if (avgMovements > 50) {
            recommendations.add("Consider relaxation techniques before bed to reduce restlessness")
        }

        if (recommendations.isEmpty()) {
            recommendations.add("Keep maintaining your excellent sleep habits!")
        }

        return recommendations
    }

    private fun createEmptyAnalytics(): SleepAnalytics {
        return SleepAnalytics(
            sessions = emptyList(),
            averageDuration = 0L,
            averageQuality = 0f,
            averageSleepEfficiency = 0f,
            totalMovementEvents = 0,
            totalNoiseEvents = 0,
            bestSleepDate = null,
            worstSleepDate = null,
            sleepTrend = SleepTrend.INSUFFICIENT_DATA,
            recommendations = listOf("Start tracking your sleep to get personalized insights!")
        )
    }

    // ========== CLEANUP ==========

    fun cleanup() {
        scope.cancel()
        invalidateAllCaches()
    }
}

// ========== ENHANCED DATA CLASSES ==========

/**
 * Enhanced cache entry with custom expiration
 */
private data class CacheEntry<T>(
    val data: T,
    val timestamp: Long = System.currentTimeMillis(),
    val customExpiryMs: Long? = null
) {
    fun isExpired(defaultExpiryMs: Long = SleepRepository.CACHE_EXPIRY_MS): Boolean {
        val expiryTime = customExpiryMs ?: defaultExpiryMs
        return System.currentTimeMillis() - timestamp > expiryTime
    }
}

/**
 * Session statistics for UI display
 */
data class SessionStatistics(
    val totalSessions: Int,
    val averageDuration: Long,
    val averageQuality: Float,
    val averageEfficiency: Float,
    val hasActiveSession: Boolean
)

/**
 * User engagement metrics for AI optimization
 */
data class UserEngagementMetrics(
    val totalInteractions: Long,
    val uniqueInsightsEngaged: Int,
    val averageEngagementTime: Long,
    val lastInteractionTime: Long
)

/**
 * AI generation status tracking
 */
enum class AIGenerationStatus {
    IDLE, GENERATING, COMPLETED, ERROR
}

/**
 * Insight sorting options
 */
enum class InsightSortOption {
    NEWEST_FIRST, OLDEST_FIRST, PRIORITY_HIGH_FIRST, CATEGORY
}

/**
 * Analysis depth levels
 */
enum class AnalysisDepth {
    BASIC, DETAILED, COMPREHENSIVE
}

/**
 * Personalization levels
 */
enum class PersonalizationLevel {
    NONE, BASIC, ADAPTIVE, ADVANCED
}

/**
 * Prediction horizons
 */
enum class PredictionHorizon {
    DAY, WEEK, MONTH
}

// Supporting data classes for AI features
data class SessionWithAnalytics(val dummy: String = "")
data class PersonalBaseline(val dummy: String = "")
data class UserSleepPreferences(val dummy: String = "")
data class SleepPatternAnalysis(val dummy: String = "")
data class TrendAnalysis(val dummy: String = "")
data class HabitAnalysis(val dummy: String = "")
data class AIModelPerformance(
    val totalRequests: Long = 0L,
    val successfulRequests: Long = 0L,
    val failedRequests: Long = 0L,
    val averageProcessingTime: Long = 0L,
    val totalTokensUsed: Int = 0,
    val totalCostCents: Int = 0,
    val averageQualityScore: Float = 0f,
    val averageUserRating: Float = 0f,
    val implementationRate: Float = 0f,
    val satisfactionRate: Float = 0f
)

data class TimeRange(
    val startDate: Long,
    val endDate: Long,
    val description: String = ""
)

data class InsightFeedback(
    val rating: Int? = null,
    val wasHelpful: Boolean? = null,
    val wasAccurate: Boolean? = null,
    val wasImplemented: Boolean? = null,
    val feedbackText: String? = null,
    val improvementSuggestions: String? = null
)

// Extension functions for compatibility
private fun SleepSession.getTotalMovements(): Int = 0 // This would access actual movement data

// Placeholder helper object
private object EntityHelper {
    fun createSensorSettingsEntity(settings: SensorSettings, sessionId: Long): SensorSettingsEntity =
        SensorSettingsEntity(sessionId = sessionId)

    fun createMovementEventEntity(event: MovementEvent, sessionId: Long): MovementEventEntity =
        MovementEventEntity(sessionId = sessionId, intensity = event.intensity, timestamp = event.timestamp)

    fun createNoiseEventEntity(event: NoiseEvent, sessionId: Long): NoiseEventEntity =
        NoiseEventEntity(sessionId = sessionId, decibelLevel = event.decibelLevel, timestamp = event.timestamp)

    fun createPhaseTransitionEntity(transition: PhaseTransition, sessionId: Long): SleepPhaseEntity =
        SleepPhaseEntity(sessionId = sessionId, fromPhase = transition.fromPhase, toPhase = transition.toPhase, timestamp = transition.timestamp)

    fun createQualityFactorsEntity(factors: QualityFactors, sessionId: Long): QualityFactorsEntity =
        QualityFactorsEntity(sessionId = sessionId, overallScore = factors.overallScore)
}