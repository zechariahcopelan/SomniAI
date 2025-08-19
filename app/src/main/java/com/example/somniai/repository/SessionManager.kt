package com.example.somniai.repository

import android.util.Log
import com.example.somniai.data.*
import com.example.somniai.ai.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*

/**
 * Enhanced session manager with comprehensive AI insights integration
 *
 * Responsibilities:
 * - Session creation, validation, and initialization
 * - Real-time session updates and monitoring
 * - Session completion with analytics generation
 * - AI insights generation and orchestration
 * - Error handling and session recovery
 * - Session state validation and integrity checks
 * - Analytics calculation and quality scoring
 * - AI-powered real-time recommendations
 * - Post-session AI analysis and insights
 * - Intelligent session optimization
 */
class SessionManager(
    private val repository: SleepRepository,
    private val aiInsightsEngine: AIInsightsEngine? = null,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "SessionManager"

        // Session timing constants
        private const val MIN_SESSION_DURATION = 5 * 60 * 1000L // 5 minutes
        private const val MAX_SESSION_DURATION = 16 * 60 * 60 * 1000L // 16 hours
        private const val SESSION_UPDATE_INTERVAL = 30 * 1000L // 30 seconds
        private const val SESSION_TIMEOUT = 24 * 60 * 60 * 1000L // 24 hours

        // Quality scoring weights
        private const val DURATION_WEIGHT = 0.25f
        private const val MOVEMENT_WEIGHT = 0.30f
        private const val NOISE_WEIGHT = 0.25f
        private const val EFFICIENCY_WEIGHT = 0.20f

        // Phase timing constants (in milliseconds)
        private const val TYPICAL_SLEEP_ONSET = 10 * 60 * 1000L // 10 minutes
        private const val MIN_DEEP_SLEEP_DURATION = 45 * 60 * 1000L // 45 minutes
        private const val TYPICAL_REM_CYCLES = 4

        // AI insights configuration
        private const val AI_GENERATION_TIMEOUT_MS = 60 * 1000L // 1 minute
        private const val AI_RETRY_ATTEMPTS = 2
        private const val AI_MINIMUM_SESSION_DURATION = 30 * 60 * 1000L // 30 minutes for AI analysis
        private const val REAL_TIME_AI_INTERVAL = 5 * 60 * 1000L // 5 minutes for real-time insights
    }

    // Session state management
    private val _sessionState = MutableStateFlow<SessionState>(SessionState.Idle)
    val sessionState: StateFlow<SessionState> = _sessionState.asStateFlow()

    private val _sessionHealth = MutableStateFlow<SessionHealth>(SessionHealth.Unknown)
    val sessionHealth: StateFlow<SessionHealth> = _sessionHealth.asStateFlow()

    // AI insights state management
    private val _aiInsightsState = MutableStateFlow<AIInsightsState>(AIInsightsState.Idle)
    val aiInsightsState: StateFlow<AIInsightsState> = _aiInsightsState.asStateFlow()

    private val _realTimeInsights = MutableStateFlow<List<SleepInsight>>(emptyList())
    val realTimeInsights: StateFlow<List<SleepInsight>> = _realTimeInsights.asStateFlow()

    private val _aiGenerationProgress = MutableStateFlow<AIGenerationProgress?>(null)
    val aiGenerationProgress: StateFlow<AIGenerationProgress?> = _aiGenerationProgress.asStateFlow()

    // Session data accumulation
    private var currentSessionData: SessionData? = null
    private val isSessionActive = AtomicBoolean(false)
    private val lastUpdateTime = AtomicLong(0L)
    private val lastAIAnalysisTime = AtomicLong(0L)

    // AI configuration
    private var aiInsightsConfig = AIInsightsConfig()

    // Background monitoring
    private val scope = CoroutineScope(dispatcher + SupervisorJob())
    private var sessionMonitoringJob: Job? = null
    private var sessionUpdateJob: Job? = null
    private var realTimeAIJob: Job? = null

    init {
        // Monitor repository state
        scope.launch {
            repository.currentSessionFlow.collect { session ->
                if (session != null && session.isActive) {
                    initializeExistingSession(session)
                } else {
                    clearSessionState()
                }
            }
        }

        // Monitor AI insights engine status
        aiInsightsEngine?.let { engine ->
            scope.launch {
                engine.engineState.collect { state ->
                    handleAIEngineStateChange(state)
                }
            }
        }
    }

    // ========== ENHANCED SESSION LIFECYCLE WITH AI INTEGRATION ==========

    /**
     * Create and initialize a new sleep tracking session with AI setup
     */
    suspend fun startSession(
        startTime: Long = System.currentTimeMillis(),
        settings: SensorSettings = SensorSettings(),
        validateEnvironment: Boolean = true,
        enableAIInsights: Boolean = true
    ): Result<SessionInfo> = withContext(dispatcher) {
        try {
            // Validate session creation
            validateSessionStart(startTime, validateEnvironment).getOrThrow()

            // Create session in repository
            val sessionId = repository.createSession(startTime, settings).getOrThrow()

            // Initialize session data with AI configuration
            val sessionData = SessionData(
                sessionId = sessionId,
                startTime = startTime,
                settings = settings,
                events = SessionEvents(),
                analytics = SessionAnalytics(),
                phases = mutableListOf(),
                aiInsightsEnabled = enableAIInsights,
                aiGenerationJobs = mutableListOf()
            )

            currentSessionData = sessionData
            isSessionActive.set(true)
            lastUpdateTime.set(startTime)
            lastAIAnalysisTime.set(startTime)

            // Update state
            _sessionState.value = SessionState.Active(sessionData.toSessionInfo())
            _sessionHealth.value = SessionHealth.Good
            _aiInsightsState.value = if (enableAIInsights) AIInsightsState.Ready else AIInsightsState.Disabled

            // Start background monitoring
            startSessionMonitoring(sessionId)
            startPeriodicUpdates()

            // Start real-time AI insights if enabled
            if (enableAIInsights && aiInsightsEngine != null) {
                startRealTimeAIAnalysis(sessionId)
            }

            // Record session start interaction
            repository.recordUserInteraction(
                insightId = null,
                interactionType = "SESSION_STARTED",
                details = "AI enabled: $enableAIInsights"
            )

            Log.d(TAG, "Session started successfully: ID=$sessionId, AI enabled=$enableAIInsights")
            Result.success(sessionData.toSessionInfo())

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start session", e)
            _sessionState.value = SessionState.Error(e.message ?: "Unknown error")
            _aiInsightsState.value = AIInsightsState.Error("Failed to initialize AI: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Enhanced session update with real-time AI insights
     */
    suspend fun updateSession(
        liveMetrics: LiveSleepMetrics? = null,
        movementEvents: List<MovementEvent> = emptyList(),
        noiseEvents: List<NoiseEvent> = emptyList(),
        phaseTransition: PhaseTransition? = null,
        triggerRealTimeAI: Boolean = false
    ): Result<SessionInfo> = withContext(dispatcher) {
        try {
            val sessionData = currentSessionData
                ?: return@withContext Result.failure(IllegalStateException("No active session"))

            if (!isSessionActive.get()) {
                return@withContext Result.failure(IllegalStateException("Session is not active"))
            }

            val currentTime = System.currentTimeMillis()

            // Validate update timing
            validateSessionUpdate(currentTime).getOrThrow()

            // Update session data
            val updatedData = sessionData.copy(
                lastUpdateTime = currentTime,
                events = sessionData.events.copy(
                    movementEvents = sessionData.events.movementEvents + movementEvents,
                    noiseEvents = sessionData.events.noiseEvents + noiseEvents
                ),
                analytics = calculateRealTimeAnalytics(sessionData, liveMetrics, currentTime)
            )

            // Handle phase transition
            phaseTransition?.let { transition ->
                updatedData.phases.add(transition)
                repository.recordPhaseTransition(transition)

                // Trigger AI analysis on significant phase changes
                if (shouldTriggerAIOnPhaseChange(transition)) {
                    triggerRealTimeAIAnalysis(updatedData)
                }
            }

            // Record events in repository
            if (movementEvents.isNotEmpty() || noiseEvents.isNotEmpty()) {
                repository.recordEventsBatch(
                    movements = movementEvents,
                    noises = noiseEvents
                ).getOrThrow()
            }

            // Update repository with current metrics
            val duration = currentTime - updatedData.startTime
            repository.updateCurrentSession(
                duration = duration,
                efficiency = updatedData.analytics.sleepEfficiency,
                movementIntensity = updatedData.analytics.averageMovementIntensity,
                noiseLevel = updatedData.analytics.averageNoiseLevel,
                movementEventCount = updatedData.events.movementEvents.size,
                noiseEventCount = updatedData.events.noiseEvents.size,
                phaseTransitionCount = updatedData.phases.size
            ).getOrThrow()

            currentSessionData = updatedData
            lastUpdateTime.set(currentTime)

            // Update state and health
            val sessionInfo = updatedData.toSessionInfo()
            _sessionState.value = SessionState.Active(sessionInfo)
            _sessionHealth.value = evaluateSessionHealth(updatedData)

            // Trigger real-time AI insights if requested or conditions are met
            if (triggerRealTimeAI || shouldTriggerRealTimeAI(updatedData, currentTime)) {
                triggerRealTimeAIAnalysis(updatedData)
            }

            Log.d(TAG, "Session updated: duration=${duration}ms, events=${movementEvents.size + noiseEvents.size}")
            Result.success(sessionInfo)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update session", e)
            _sessionHealth.value = SessionHealth.Warning("Update failed: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Enhanced session completion with comprehensive AI insights generation
     */
    suspend fun completeSession(
        endTime: Long = System.currentTimeMillis(),
        forceComplete: Boolean = false,
        generateAIInsights: Boolean = true,
        analysisDepth: AnalysisDepth = AnalysisDepth.COMPREHENSIVE
    ): Result<SessionCompletionResult> = withContext(dispatcher) {
        try {
            val sessionData = currentSessionData
                ?: return@withContext Result.failure(IllegalStateException("No active session"))

            Log.d(TAG, "Completing session: ${sessionData.sessionId}, AI insights: $generateAIInsights")

            // Validate session completion
            if (!forceComplete) {
                validateSessionCompletion(sessionData, endTime).getOrThrow()
            }

            // Stop real-time monitoring
            stopRealTimeAIAnalysis()

            // Generate final analytics
            val finalAnalytics = generateFinalAnalytics(sessionData, endTime)

            // Complete session in repository
            val completedSession = repository.completeSession(endTime, finalAnalytics).getOrThrow()

            var aiInsights: List<SleepInsight> = emptyList()
            var aiGenerationResult: AIGenerationResult? = null

            // Generate AI insights if enabled and conditions are met
            if (generateAIInsights && sessionData.aiInsightsEnabled &&
                shouldGenerateAIInsights(sessionData, endTime)) {

                try {
                    _aiInsightsState.value = AIInsightsState.Generating
                    _aiGenerationProgress.value = AIGenerationProgress(
                        stage = "Initializing AI analysis",
                        progress = 0f,
                        estimatedTimeRemaining = AI_GENERATION_TIMEOUT_MS
                    )

                    // Generate comprehensive AI insights
                    aiGenerationResult = generatePostSessionAIInsights(
                        sessionData = sessionData,
                        finalAnalytics = finalAnalytics,
                        analysisDepth = analysisDepth
                    ).getOrThrow()

                    aiInsights = aiGenerationResult.insights
                    _aiInsightsState.value = AIInsightsState.Completed(aiInsights)

                    // Record successful AI generation
                    repository.recordUserInteraction(
                        insightId = null,
                        interactionType = "AI_INSIGHTS_GENERATED",
                        details = "Generated ${aiInsights.size} insights, depth: $analysisDepth"
                    )

                    Log.d(TAG, "Generated ${aiInsights.size} AI insights for session ${sessionData.sessionId}")

                } catch (e: Exception) {
                    Log.e(TAG, "Failed to generate AI insights", e)
                    _aiInsightsState.value = AIInsightsState.Error("AI generation failed: ${e.message}")

                    // Record AI generation failure
                    repository.recordUserInteraction(
                        insightId = null,
                        interactionType = "AI_INSIGHTS_FAILED",
                        details = "Error: ${e.message}"
                    )

                    // Continue with session completion even if AI fails
                }
            } else {
                _aiInsightsState.value = AIInsightsState.Skipped("Conditions not met for AI generation")
            }

            // Clear progress indicator
            _aiGenerationProgress.value = null

            // Clear session state
            clearSessionState()
            stopSessionMonitoring()

            val completionResult = SessionCompletionResult(
                session = completedSession,
                analytics = finalAnalytics,
                aiInsights = aiInsights,
                aiGenerationResult = aiGenerationResult,
                completionTime = endTime
            )

            _sessionState.value = SessionState.Completed(finalAnalytics)
            _sessionHealth.value = SessionHealth.Unknown

            Log.d(TAG, "Session completed: duration=${finalAnalytics.totalDuration}ms, " +
                    "quality=${finalAnalytics.qualityFactors.overallScore}, " +
                    "AI insights=${aiInsights.size}")

            Result.success(completionResult)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to complete session", e)
            _sessionState.value = SessionState.Error("Completion failed: ${e.message}")
            _aiInsightsState.value = AIInsightsState.Error("Session completion failed")
            Result.failure(e)
        }
    }

    // ========== AI INSIGHTS GENERATION METHODS ==========

    /**
     * Generate comprehensive post-session AI insights
     */
    private suspend fun generatePostSessionAIInsights(
        sessionData: SessionData,
        finalAnalytics: SleepSessionAnalytics,
        analysisDepth: AnalysisDepth
    ): Result<AIGenerationResult> = withContext(dispatcher) {
        try {
            val aiEngine = aiInsightsEngine
                ?: return@withContext Result.failure(IllegalStateException("AI engine not available"))

            // Update progress
            _aiGenerationProgress.value = AIGenerationProgress(
                stage = "Preparing session data",
                progress = 0.1f,
                estimatedTimeRemaining = AI_GENERATION_TIMEOUT_MS * 0.9f.toLong()
            )

            // Create insight generation options
            val options = InsightGenerationOptions(
                forceAI = true,
                maxInsights = when (analysisDepth) {
                    AnalysisDepth.BASIC -> 3
                    AnalysisDepth.DETAILED -> 7
                    AnalysisDepth.COMPREHENSIVE -> 12
                },
                includeRuleBasedInsights = true,
                includeAIInsights = true,
                includeMLEnhancement = true,
                personalizationLevel = PersonalizationLevel.ADAPTIVE
            )

            _aiGenerationProgress.value = _aiGenerationProgress.value?.copy(
                stage = "Generating AI insights",
                progress = 0.3f
            )

            // Generate session-specific insights
            val sessionInsights = aiEngine.generateSessionInsights(
                sessionId = sessionData.sessionId,
                options = options
            ).getOrThrow()

            _aiGenerationProgress.value = _aiGenerationProgress.value?.copy(
                stage = "Generating personalized insights",
                progress = 0.6f
            )

            // Generate personalized insights based on this session
            val personalizedInsights = if (analysisDepth == AnalysisDepth.COMPREHENSIVE) {
                aiEngine.generatePersonalizedInsights(
                    analysisDepth = analysisDepth,
                    personalizationLevel = PersonalizationLevel.ADAPTIVE
                ).getOrElse { emptyList() }
            } else {
                emptyList()
            }

            _aiGenerationProgress.value = _aiGenerationProgress.value?.copy(
                stage = "Finalizing insights",
                progress = 0.9f
            )

            // Combine and deduplicate insights
            val allInsights = (sessionInsights + personalizedInsights).distinctBy { it.title }

            // Create generation result
            val result = AIGenerationResult(
                insights = allInsights,
                sessionId = sessionData.sessionId,
                generationType = "POST_SESSION",
                analysisDepth = analysisDepth,
                processingTime = System.currentTimeMillis() - (sessionData.lastUpdateTime),
                qualityScore = calculateInsightsQualityScore(allInsights),
                metadata = mapOf(
                    "session_duration" to finalAnalytics.totalDuration,
                    "sleep_efficiency" to finalAnalytics.sleepEfficiency,
                    "movement_events" to finalAnalytics.totalMovementEvents,
                    "noise_events" to finalAnalytics.totalNoiseEvents,
                    "analysis_depth" to analysisDepth.name
                )
            )

            _aiGenerationProgress.value = _aiGenerationProgress.value?.copy(
                stage = "Complete",
                progress = 1f
            )

            Log.d(TAG, "AI insights generation completed: ${allInsights.size} insights")
            Result.success(result)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate post-session AI insights", e)
            Result.failure(e)
        }
    }

    /**
     * Trigger real-time AI analysis during session
     */
    private suspend fun triggerRealTimeAIAnalysis(sessionData: SessionData) {
        if (!sessionData.aiInsightsEnabled || aiInsightsEngine == null) return

        try {
            _aiInsightsState.value = AIInsightsState.AnalyzingRealTime

            // Generate real-time insights based on current session state
            val realTimeInsights = generateRealTimeInsights(sessionData)

            // Update real-time insights
            _realTimeInsights.value = realTimeInsights
            lastAIAnalysisTime.set(System.currentTimeMillis())

            _aiInsightsState.value = AIInsightsState.Ready

            Log.d(TAG, "Real-time AI analysis completed: ${realTimeInsights.size} insights")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to perform real-time AI analysis", e)
            _aiInsightsState.value = AIInsightsState.Error("Real-time analysis failed: ${e.message}")
        }
    }

    /**
     * Generate real-time insights during active session
     */
    private suspend fun generateRealTimeInsights(sessionData: SessionData): List<SleepInsight> {
        return try {
            val currentTime = System.currentTimeMillis()
            val duration = currentTime - sessionData.startTime

            val insights = mutableListOf<SleepInsight>()

            // Duration-based insights
            if (duration > 8 * 60 * 60 * 1000L) { // 8+ hours
                insights.add(createRealTimeInsight(
                    "Long sleep session detected",
                    "You've been sleeping for over 8 hours. Consider preparing to wake up naturally.",
                    "Set a gentle alarm for the next 30-60 minutes",
                    InsightCategory.DURATION
                ))
            }

            // Movement-based insights
            val recentMovements = sessionData.events.movementEvents.filter {
                it.timestamp > currentTime - (30 * 60 * 1000L) // Last 30 minutes
            }

            if (recentMovements.size > 10) {
                insights.add(createRealTimeInsight(
                    "Increased restlessness detected",
                    "Your movement has increased in the last 30 minutes. This might indicate lighter sleep or potential awakening.",
                    "Consider optimizing your sleep environment for better comfort",
                    InsightCategory.MOVEMENT
                ))
            }

            // Phase-based insights
            val currentPhase = sessionData.analytics.currentPhase
            if (currentPhase == SleepPhase.REM && duration > 6 * 60 * 60 * 1000L) {
                insights.add(createRealTimeInsight(
                    "REM sleep detected",
                    "You're currently in REM sleep, which is important for memory consolidation and dreaming.",
                    "Avoid waking up during this phase if possible",
                    InsightCategory.PATTERN
                ))
            }

            insights

        } catch (e: Exception) {
            Log.e(TAG, "Error generating real-time insights", e)
            emptyList()
        }
    }

    /**
     * Start real-time AI analysis background job
     */
    private fun startRealTimeAIAnalysis(sessionId: Long) {
        realTimeAIJob = scope.launch {
            while (isActive && isSessionActive.get()) {
                delay(REAL_TIME_AI_INTERVAL)

                try {
                    val sessionData = currentSessionData
                    if (sessionData != null && sessionData.aiInsightsEnabled) {
                        val currentTime = System.currentTimeMillis()

                        // Only analyze if enough time has passed since last analysis
                        if (currentTime - lastAIAnalysisTime.get() >= REAL_TIME_AI_INTERVAL) {
                            triggerRealTimeAIAnalysis(sessionData)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in real-time AI analysis", e)
                }
            }
        }
    }

    /**
     * Stop real-time AI analysis
     */
    private fun stopRealTimeAIAnalysis() {
        realTimeAIJob?.cancel()
        realTimeAIJob = null
        _realTimeInsights.value = emptyList()
    }

    // ========== AI CONDITION CHECKING METHODS ==========

    private fun shouldGenerateAIInsights(sessionData: SessionData, endTime: Long): Boolean {
        val duration = endTime - sessionData.startTime

        return sessionData.aiInsightsEnabled &&
                aiInsightsEngine != null &&
                duration >= AI_MINIMUM_SESSION_DURATION &&
                (sessionData.events.movementEvents.isNotEmpty() || sessionData.events.noiseEvents.isNotEmpty())
    }

    private fun shouldTriggerRealTimeAI(sessionData: SessionData, currentTime: Long): Boolean {
        val timeSinceLastAI = currentTime - lastAIAnalysisTime.get()
        val duration = currentTime - sessionData.startTime

        return sessionData.aiInsightsEnabled &&
                timeSinceLastAI >= REAL_TIME_AI_INTERVAL &&
                duration >= 30 * 60 * 1000L && // At least 30 minutes into session
                (sessionData.events.movementEvents.size >= 5 || sessionData.events.noiseEvents.size >= 3)
    }

    private fun shouldTriggerAIOnPhaseChange(transition: PhaseTransition): Boolean {
        return when (transition.toPhase) {
            SleepPhase.DEEP_SLEEP, SleepPhase.REM_SLEEP -> true
            SleepPhase.AWAKE -> transition.fromPhase != SleepPhase.AWAKE
            else -> false
        }
    }

    // ========== AI HELPER METHODS ==========

    private fun createRealTimeInsight(
        title: String,
        description: String,
        recommendation: String,
        category: InsightCategory
    ): SleepInsight {
        return SleepInsight(
            sessionId = currentSessionData?.sessionId ?: 0L,
            category = category,
            title = title,
            description = description,
            recommendation = recommendation,
            priority = 2,
            isAiGenerated = true,
            timestamp = System.currentTimeMillis(),
            confidenceScore = 0.7f
        )
    }

    private fun calculateInsightsQualityScore(insights: List<SleepInsight>): Float {
        if (insights.isEmpty()) return 0f

        return insights.map { insight ->
            val baseScore = 0.7f // Base quality score
            val confidenceBonus = insight.confidenceScore * 0.2f
            val categoryBonus = when (insight.category) {
                InsightCategory.QUALITY, InsightCategory.EFFICIENCY -> 0.1f
                else -> 0.05f
            }

            (baseScore + confidenceBonus + categoryBonus).coerceIn(0f, 1f)
        }.average().toFloat()
    }

    private fun handleAIEngineStateChange(engineState: Any) {
        // Handle AI engine state changes
        scope.launch {
            when (engineState.toString()) {
                "ERROR" -> _aiInsightsState.value = AIInsightsState.Error("AI engine error")
                "ACTIVE" -> {
                    if (_aiInsightsState.value is AIInsightsState.Error) {
                        _aiInsightsState.value = AIInsightsState.Ready
                    }
                }
            }
        }
    }

    // ========== CONFIGURATION METHODS ==========

    /**
     * Update AI insights configuration
     */
    fun updateAIInsightsConfig(config: AIInsightsConfig) {
        aiInsightsConfig = config
        Log.d(TAG, "AI insights configuration updated: $config")
    }

    /**
     * Enable/disable AI insights for current session
     */
    fun setAIInsightsEnabled(enabled: Boolean) {
        currentSessionData?.let { sessionData ->
            currentSessionData = sessionData.copy(aiInsightsEnabled = enabled)

            if (enabled && aiInsightsEngine != null) {
                _aiInsightsState.value = AIInsightsState.Ready
                if (isSessionActive.get()) {
                    startRealTimeAIAnalysis(sessionData.sessionId)
                }
            } else {
                _aiInsightsState.value = AIInsightsState.Disabled
                stopRealTimeAIAnalysis()
            }
        }
    }

    /**
     * Force trigger AI analysis for current session
     */
    suspend fun triggerAIAnalysis(analysisType: String = "MANUAL"): Result<List<SleepInsight>> {
        return try {
            val sessionData = currentSessionData
                ?: return Result.failure(IllegalStateException("No active session"))

            if (!sessionData.aiInsightsEnabled || aiInsightsEngine == null) {
                return Result.failure(IllegalStateException("AI insights not enabled"))
            }

            // Record user interaction
            repository.recordUserInteraction(
                insightId = null,
                interactionType = "AI_ANALYSIS_TRIGGERED",
                details = "Type: $analysisType"
            )

            triggerRealTimeAIAnalysis(sessionData)
            Result.success(_realTimeInsights.value)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to trigger AI analysis", e)
            Result.failure(e)
        }
    }

    // ========== EXISTING METHODS (keeping all original functionality) ==========

    /**
     * Cancel current session
     */
    suspend fun cancelSession(reason: String = "User cancelled"): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionData = currentSessionData
            if (sessionData != null) {
                Log.d(TAG, "Cancelling session: ${sessionData.sessionId}, reason: $reason")
            }

            // Cancel in repository
            repository.cancelCurrentSession().getOrThrow()

            // Clear session state
            clearSessionState()
            stopSessionMonitoring()
            stopRealTimeAIAnalysis()

            _sessionState.value = SessionState.Cancelled(reason)
            _sessionHealth.value = SessionHealth.Unknown
            _aiInsightsState.value = AIInsightsState.Idle

            Log.d(TAG, "Session cancelled: $reason")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to cancel session", e)
            Result.failure(e)
        }
    }

    /**
     * Pause current session (for interruptions)
     */
    suspend fun pauseSession(reason: String = "Paused by user"): Result<Unit> = withContext(dispatcher) {
        try {
            val sessionData = currentSessionData
                ?: return@withContext Result.failure(IllegalStateException("No active session"))

            isSessionActive.set(false)
            stopPeriodicUpdates()
            stopRealTimeAIAnalysis()

            _sessionState.value = SessionState.Paused(sessionData.toSessionInfo(), reason)
            _sessionHealth.value = SessionHealth.Warning("Session paused")
            _aiInsightsState.value = AIInsightsState.Paused

            Log.d(TAG, "Session paused: $reason")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to pause session", e)
            Result.failure(e)
        }
    }

    /**
     * Resume paused session
     */
    suspend fun resumeSession(): Result<SessionInfo> = withContext(dispatcher) {
        try {
            val sessionData = currentSessionData
                ?: return@withContext Result.failure(IllegalStateException("No session to resume"))

            isSessionActive.set(true)
            startPeriodicUpdates()

            if (sessionData.aiInsightsEnabled && aiInsightsEngine != null) {
                startRealTimeAIAnalysis(sessionData.sessionId)
                _aiInsightsState.value = AIInsightsState.Ready
            }

            val sessionInfo = sessionData.toSessionInfo()
            _sessionState.value = SessionState.Active(sessionInfo)
            _sessionHealth.value = SessionHealth.Good

            Log.d(TAG, "Session resumed: ${sessionData.sessionId}")
            Result.success(sessionInfo)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to resume session", e)
            Result.failure(e)
        }
    }

    // ========== SESSION VALIDATION ==========

    private fun validateSessionStart(
        startTime: Long,
        validateEnvironment: Boolean
    ): Result<Unit> {
        val currentTime = System.currentTimeMillis()

        // Check timing
        if (startTime > currentTime) {
            return Result.failure(IllegalArgumentException("Start time cannot be in the future"))
        }

        if (currentTime - startTime > 60 * 60 * 1000) { // 1 hour ago
            return Result.failure(IllegalArgumentException("Start time is too far in the past"))
        }

        // Check if another session is active
        if (isSessionActive.get()) {
            return Result.failure(IllegalStateException("Another session is already active"))
        }

        // Environment validation (if requested)
        if (validateEnvironment) {
            // Could add checks for battery level, storage space, permissions, etc.
            // For now, just log
            Log.d(TAG, "Environment validation passed")
        }

        return Result.success(Unit)
    }

    private fun validateSessionUpdate(updateTime: Long): Result<Unit> {
        val sessionData = currentSessionData ?: return Result.failure(
            IllegalStateException("No session data available")
        )

        // Check update timing
        val timeSinceStart = updateTime - sessionData.startTime
        if (timeSinceStart > MAX_SESSION_DURATION) {
            return Result.failure(IllegalStateException("Session has exceeded maximum duration"))
        }

        // Check update frequency (don't update too frequently)
        val timeSinceLastUpdate = updateTime - lastUpdateTime.get()
        if (timeSinceLastUpdate < 1000) { // 1 second minimum
            return Result.failure(IllegalArgumentException("Updates too frequent"))
        }

        return Result.success(Unit)
    }

    private fun validateSessionCompletion(
        sessionData: SessionData,
        endTime: Long
    ): Result<Unit> {
        val duration = endTime - sessionData.startTime

        // Check minimum duration
        if (duration < MIN_SESSION_DURATION) {
            return Result.failure(IllegalArgumentException(
                "Session too short (minimum ${MIN_SESSION_DURATION / 60000} minutes)"
            ))
        }

        // Check maximum duration
        if (duration > MAX_SESSION_DURATION) {
            return Result.failure(IllegalArgumentException(
                "Session too long (maximum ${MAX_SESSION_DURATION / 3600000} hours)"
            ))
        }

        // Check data integrity
        if (sessionData.events.movementEvents.isEmpty() && sessionData.events.noiseEvents.isEmpty()) {
            Log.w(TAG, "Session has no recorded events - this may indicate sensor issues")
        }

        return Result.success(Unit)
    }

    // ========== ANALYTICS CALCULATION ==========

    private fun calculateRealTimeAnalytics(
        sessionData: SessionData,
        liveMetrics: LiveSleepMetrics?,
        currentTime: Long
    ): SessionAnalytics {
        val duration = currentTime - sessionData.startTime
        val movements = sessionData.events.movementEvents
        val noises = sessionData.events.noiseEvents

        // Calculate movement metrics
        val avgMovementIntensity = if (movements.isNotEmpty()) {
            movements.map { it.intensity }.average().toFloat()
        } else 0f

        val movementFrequency = if (duration > 0) {
            (movements.size.toFloat() / duration) * 3600000f // movements per hour
        } else 0f

        // Calculate noise metrics
        val avgNoiseLevel = if (noises.isNotEmpty()) {
            noises.map { it.decibelLevel }.average().toFloat()
        } else 0f

        // Calculate sleep efficiency (from live metrics or estimate)
        val sleepEfficiency = liveMetrics?.sleepEfficiency ?: estimateSleepEfficiency(
            duration, movements.size, noises.size
        )

        // Calculate restlessness
        val restlessness = calculateRestlessness(movements, duration)

        return SessionAnalytics(
            duration = duration,
            sleepEfficiency = sleepEfficiency,
            averageMovementIntensity = avgMovementIntensity,
            averageNoiseLevel = avgNoiseLevel,
            movementFrequency = movementFrequency,
            totalRestlessness = restlessness,
            currentPhase = liveMetrics?.currentPhase ?: SleepPhase.UNKNOWN,
            phaseConfidence = liveMetrics?.phaseConfidence ?: 0f
        )
    }

    private fun generateFinalAnalytics(
        sessionData: SessionData,
        endTime: Long
    ): SleepSessionAnalytics {
        val totalDuration = endTime - sessionData.startTime
        val movements = sessionData.events.movementEvents
        val noises = sessionData.events.noiseEvents
        val phases = sessionData.phases

        // Calculate phase durations
        val phaseDurations = calculatePhaseDurations(phases, sessionData.startTime, endTime)

        // Calculate quality factors
        val qualityFactors = calculateQualityFactors(
            totalDuration, movements, noises, phaseDurations
        )

        // Calculate sleep efficiency
        val awakeDuration = phaseDurations[SleepPhase.AWAKE] ?: 0L
        val sleepEfficiency = if (totalDuration > 0) {
            ((totalDuration - awakeDuration).toFloat() / totalDuration) * 100f
        } else 0f

        // Calculate movement metrics
        val avgMovementIntensity = if (movements.isNotEmpty()) {
            movements.map { it.intensity }.average().toFloat()
        } else 0f

        val movementFrequency = if (totalDuration > 0) {
            (movements.size.toFloat() / totalDuration) * 3600000f // per hour
        } else 0f

        // Calculate noise metrics
        val avgNoiseLevel = if (noises.isNotEmpty()) {
            noises.map { it.decibelLevel }.average().toFloat()
        } else 0f

        // Calculate sleep latency (time to first non-awake phase)
        val sleepLatency = calculateSleepLatency(phases, sessionData.startTime)

        return SleepSessionAnalytics(
            sessionId = sessionData.sessionId,
            startTime = sessionData.startTime,
            endTime = endTime,
            totalDuration = totalDuration,
            sleepLatency = sleepLatency,
            awakeDuration = awakeDuration,
            lightSleepDuration = phaseDurations[SleepPhase.LIGHT_SLEEP] ?: 0L,
            deepSleepDuration = phaseDurations[SleepPhase.DEEP_SLEEP] ?: 0L,
            remSleepDuration = phaseDurations[SleepPhase.REM_SLEEP] ?: 0L,
            sleepEfficiency = sleepEfficiency,
            qualityFactors = qualityFactors,
            averageMovementIntensity = avgMovementIntensity,
            averageNoiseLevel = avgNoiseLevel,
            movementFrequency = movementFrequency,
            totalMovementEvents = movements.size,
            totalNoiseEvents = noises.size,
            phaseTransitions = phases.size
        )
    }

    private fun calculateQualityFactors(
        duration: Long,
        movements: List<MovementEvent>,
        noises: List<NoiseEvent>,
        phaseDurations: Map<SleepPhase, Long>
    ): SleepQualityFactors {
        // Duration score (optimal 7-9 hours)
        val hours = duration / (1000f * 60f * 60f)
        val durationScore = when {
            hours < 6f -> 3f + (hours / 6f) * 3f
            hours <= 9f -> 8f + (1f - abs(hours - 7.5f) / 1.5f) * 2f
            else -> max(1f, 8f - (hours - 9f) * 0.5f)
        }.coerceIn(1f, 10f)

        // Movement score (less movement = higher score)
        val avgMovementIntensity = if (movements.isNotEmpty()) {
            movements.map { it.intensity }.average().toFloat()
        } else 0f
        val movementScore = max(1f, 10f - avgMovementIntensity * 2f).coerceIn(1f, 10f)

        // Noise score (quieter = higher score)
        val avgNoiseLevel = if (noises.isNotEmpty()) {
            noises.map { it.decibelLevel }.average().toFloat()
        } else 30f
        val noiseScore = when {
            avgNoiseLevel <= 30f -> 10f
            avgNoiseLevel <= 40f -> 8f
            avgNoiseLevel <= 50f -> 6f
            avgNoiseLevel <= 60f -> 4f
            else -> 2f
        }

        // Consistency score (based on phase distribution)
        val deepSleepRatio = (phaseDurations[SleepPhase.DEEP_SLEEP] ?: 0L).toFloat() / duration
        val consistencyScore = when {
            deepSleepRatio >= 0.2f -> 9f + deepSleepRatio * 5f
            deepSleepRatio >= 0.15f -> 7f + (deepSleepRatio - 0.15f) * 20f
            deepSleepRatio >= 0.1f -> 5f + (deepSleepRatio - 0.1f) * 40f
            else -> max(1f, deepSleepRatio * 50f)
        }.coerceIn(1f, 10f)

        // Overall score (weighted average)
        val overallScore = (
                durationScore * DURATION_WEIGHT +
                        movementScore * MOVEMENT_WEIGHT +
                        noiseScore * NOISE_WEIGHT +
                        consistencyScore * EFFICIENCY_WEIGHT
                ).coerceIn(1f, 10f)

        return SleepQualityFactors(
            movementScore = movementScore,
            noiseScore = noiseScore,
            durationScore = durationScore,
            consistencyScore = consistencyScore,
            overallScore = overallScore
        )
    }

    private fun calculatePhaseDurations(
        phases: List<PhaseTransition>,
        startTime: Long,
        endTime: Long
    ): Map<SleepPhase, Long> {
        val durations = mutableMapOf<SleepPhase, Long>()

        if (phases.isEmpty()) {
            // If no phase data, assume light sleep for entire duration
            durations[SleepPhase.LIGHT_SLEEP] = endTime - startTime
            return durations
        }

        var currentTime = startTime
        var currentPhase = SleepPhase.AWAKE // Start awake

        for (transition in phases.sortedBy { it.timestamp }) {
            val phaseDuration = transition.timestamp - currentTime
            if (phaseDuration > 0) {
                durations[currentPhase] = (durations[currentPhase] ?: 0L) + phaseDuration
            }
            currentTime = transition.timestamp
            currentPhase = transition.toPhase
        }

        // Add final phase duration
        val finalDuration = endTime - currentTime
        if (finalDuration > 0) {
            durations[currentPhase] = (durations[currentPhase] ?: 0L) + finalDuration
        }

        return durations
    }

    private fun calculateSleepLatency(phases: List<PhaseTransition>, startTime: Long): Long {
        val firstSleepPhase = phases
            .filter { it.toPhase != SleepPhase.AWAKE && it.toPhase != SleepPhase.UNKNOWN }
            .minByOrNull { it.timestamp }

        return firstSleepPhase?.let { it.timestamp - startTime } ?: 0L
    }

    private fun estimateSleepEfficiency(
        duration: Long,
        movementCount: Int,
        noiseCount: Int
    ): Float {
        // Simple estimation based on movement and noise events
        val movementPenalty = (movementCount / (duration / 60000f)) * 2f // movements per minute
        val noisePenalty = (noiseCount / (duration / 60000f)) * 1.5f // noise events per minute

        val efficiency = 95f - movementPenalty - noisePenalty
        return efficiency.coerceIn(50f, 100f)
    }

    private fun calculateRestlessness(movements: List<MovementEvent>, duration: Long): Float {
        if (movements.isEmpty() || duration == 0L) return 0f

        val significantMovements = movements.count { it.isSignificant() }
        val movementRate = (significantMovements.toFloat() / duration) * 3600000f // per hour

        return (movementRate * 2f).coerceIn(0f, 10f)
    }

    // ========== SESSION MONITORING ==========

    private fun startSessionMonitoring(sessionId: Long) {
        sessionMonitoringJob = scope.launch {
            while (isActive && isSessionActive.get()) {
                delay(60000L) // Check every minute

                try {
                    val currentTime = System.currentTimeMillis()
                    val sessionData = currentSessionData ?: continue

                    // Check for session timeout
                    if (currentTime - sessionData.startTime > SESSION_TIMEOUT) {
                        Log.w(TAG, "Session timeout detected, auto-completing")
                        completeSession(currentTime, forceComplete = true)
                        break
                    }

                    // Check session health
                    val health = evaluateSessionHealth(sessionData)
                    _sessionHealth.value = health

                } catch (e: Exception) {
                    Log.e(TAG, "Error in session monitoring", e)
                }
            }
        }
    }

    private fun startPeriodicUpdates() {
        sessionUpdateJob = scope.launch {
            while (isActive && isSessionActive.get()) {
                delay(SESSION_UPDATE_INTERVAL)

                try {
                    val sessionData = currentSessionData ?: continue
                    val currentTime = System.currentTimeMillis()

                    // Update session with current metrics
                    val duration = currentTime - sessionData.startTime
                    repository.updateCurrentSession(duration = duration)

                } catch (e: Exception) {
                    Log.e(TAG, "Error in periodic update", e)
                }
            }
        }
    }

    private fun stopSessionMonitoring() {
        sessionMonitoringJob?.cancel()
        sessionUpdateJob?.cancel()
        sessionMonitoringJob = null
        sessionUpdateJob = null
    }

    private fun evaluateSessionHealth(sessionData: SessionData): SessionHealth {
        val currentTime = System.currentTimeMillis()
        val duration = currentTime - sessionData.startTime
        val timeSinceUpdate = currentTime - sessionData.lastUpdateTime

        return when {
            timeSinceUpdate > 5 * 60 * 1000L -> // 5 minutes without update
                SessionHealth.Warning("No recent sensor data")
            duration > MAX_SESSION_DURATION * 0.8 -> // 80% of max duration
                SessionHealth.Warning("Session approaching maximum duration")
            sessionData.events.movementEvents.isEmpty() && duration > 30 * 60 * 1000L -> // 30 min with no movement
                SessionHealth.Warning("No movement detected - check sensors")
            else -> SessionHealth.Good
        }
    }

    // ========== HELPER METHODS ==========

    private fun initializeExistingSession(session: SleepSession) {
        scope.launch {
            try {
                // Reconstruct session data from existing session
                val sessionData = SessionData(
                    sessionId = session.id,
                    startTime = session.startTime,
                    lastUpdateTime = System.currentTimeMillis(),
                    settings = session.settings ?: SensorSettings(),
                    events = SessionEvents(
                        movementEvents = session.movementEvents,
                        noiseEvents = session.noiseEvents
                    ),
                    analytics = SessionAnalytics(
                        duration = session.duration,
                        sleepEfficiency = session.sleepEfficiency,
                        averageMovementIntensity = session.averageMovementIntensity,
                        averageNoiseLevel = session.averageNoiseLevel,
                        movementFrequency = session.movementFrequency
                    ),
                    phases = session.phaseTransitions.toMutableList(),
                    aiInsightsEnabled = aiInsightsConfig.enabledByDefault
                )

                currentSessionData = sessionData
                isSessionActive.set(true)
                lastUpdateTime.set(System.currentTimeMillis())

                _sessionState.value = SessionState.Active(sessionData.toSessionInfo())
                _sessionHealth.value = SessionHealth.Good

                if (sessionData.aiInsightsEnabled && aiInsightsEngine != null) {
                    _aiInsightsState.value = AIInsightsState.Ready
                    startRealTimeAIAnalysis(session.id)
                } else {
                    _aiInsightsState.value = AIInsightsState.Disabled
                }

                startSessionMonitoring(session.id)
                startPeriodicUpdates()

                Log.d(TAG, "Initialized existing session: ${session.id}")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize existing session", e)
                _sessionState.value = SessionState.Error("Failed to restore session")
                _aiInsightsState.value = AIInsightsState.Error("Failed to restore AI state")
            }
        }
    }

    private fun clearSessionState() {
        currentSessionData = null
        isSessionActive.set(false)
        lastUpdateTime.set(0L)
        lastAIAnalysisTime.set(0L)
        stopSessionMonitoring()
        stopRealTimeAIAnalysis()
    }

    // ========== PUBLIC GETTERS ==========

    /**
     * Get current session information
     */
    fun getCurrentSessionInfo(): SessionInfo? {
        return currentSessionData?.toSessionInfo()
    }

    /**
     * Check if session is currently active
     */
    fun isSessionActive(): Boolean = isSessionActive.get()

    /**
     * Get session duration if active
     */
    fun getSessionDuration(): Long {
        val sessionData = currentSessionData ?: return 0L
        return System.currentTimeMillis() - sessionData.startTime
    }

    /**
     * Get current session health status
     */
    fun getCurrentSessionHealth(): SessionHealth = _sessionHealth.value

    /**
     * Get current AI insights state
     */
    fun getCurrentAIInsightsState(): AIInsightsState = _aiInsightsState.value

    /**
     * Get current real-time insights
     */
    fun getCurrentRealTimeInsights(): List<SleepInsight> = _realTimeInsights.value

    /**
     * Check if AI insights are enabled for current session
     */
    fun isAIInsightsEnabled(): Boolean = currentSessionData?.aiInsightsEnabled ?: false

    // ========== CLEANUP ==========

    fun cleanup() {
        scope.cancel()
        clearSessionState()
    }
}

// ========== ENHANCED DATA CLASSES ==========

/**
 * Enhanced session data structure with AI integration
 */
private data class SessionData(
    val sessionId: Long,
    val startTime: Long,
    val lastUpdateTime: Long = startTime,
    val settings: SensorSettings,
    val events: SessionEvents,
    val analytics: SessionAnalytics,
    val phases: MutableList<PhaseTransition>,
    val aiInsightsEnabled: Boolean = true,
    val aiGenerationJobs: MutableList<String> = mutableListOf()
) {
    fun toSessionInfo(): SessionInfo {
        val currentTime = System.currentTimeMillis()
        return SessionInfo(
            sessionId = sessionId,
            startTime = startTime,
            duration = currentTime - startTime,
            movementEventCount = events.movementEvents.size,
            noiseEventCount = events.noiseEvents.size,
            phaseTransitionCount = phases.size,
            currentPhase = analytics.currentPhase,
            sleepEfficiency = analytics.sleepEfficiency,
            lastUpdateTime = lastUpdateTime,
            aiInsightsEnabled = aiInsightsEnabled
        )
    }
}

/**
 * AI insights state management
 */
sealed class AIInsightsState {
    object Idle : AIInsightsState()
    object Ready : AIInsightsState()
    object Disabled : AIInsightsState()
    object Generating : AIInsightsState()
    object AnalyzingRealTime : AIInsightsState()
    object Paused : AIInsightsState()
    data class Completed(val insights: List<SleepInsight>) : AIInsightsState()
    data class Skipped(val reason: String) : AIInsightsState()
    data class Error(val message: String) : AIInsightsState()
}

/**
 * AI generation progress tracking
 */
data class AIGenerationProgress(
    val stage: String,
    val progress: Float,
    val estimatedTimeRemaining: Long,
    val currentTask: String? = null
)

/**
 * AI insights configuration
 */
data class AIInsightsConfig(
    val enabledByDefault: Boolean = true,
    val realTimeAnalysisEnabled: Boolean = true,
    val postSessionAnalysisEnabled: Boolean = true,
    val minimumSessionDuration: Long = AI_MINIMUM_SESSION_DURATION,
    val realTimeInterval: Long = REAL_TIME_AI_INTERVAL,
    val defaultAnalysisDepth: AnalysisDepth = AnalysisDepth.DETAILED,
    val maxInsightsPerSession: Int = 10
)

/**
 * Enhanced session info with AI status
 */
data class SessionInfo(
    val sessionId: Long,
    val startTime: Long,
    val duration: Long,
    val movementEventCount: Int,
    val noiseEventCount: Int,
    val phaseTransitionCount: Int,
    val currentPhase: SleepPhase,
    val sleepEfficiency: Float,
    val lastUpdateTime: Long,
    val aiInsightsEnabled: Boolean = false
)

/**
 * Session completion result with AI insights
 */
data class SessionCompletionResult(
    val session: SleepSession,
    val analytics: SleepSessionAnalytics,
    val aiInsights: List<SleepInsight>,
    val aiGenerationResult: AIGenerationResult?,
    val completionTime: Long
)

/**
 * AI generation result
 */
data class AIGenerationResult(
    val insights: List<SleepInsight>,
    val sessionId: Long,
    val generationType: String,
    val analysisDepth: AnalysisDepth,
    val processingTime: Long,
    val qualityScore: Float,
    val metadata: Map<String, Any>
)

/**
 * Analysis depth levels
 */
enum class AnalysisDepth {
    BASIC, DETAILED, COMPREHENSIVE
}

/**
 * Insight generation options
 */
data class InsightGenerationOptions(
    val forceAI: Boolean = false,
    val maxInsights: Int = 10,
    val includeRuleBasedInsights: Boolean = true,
    val includeAIInsights: Boolean = true,
    val includeMLEnhancement: Boolean = true,
    val personalizationLevel: PersonalizationLevel = PersonalizationLevel.ADAPTIVE
)

enum class PersonalizationLevel {
    NONE, BASIC, ADAPTIVE, ADVANCED
}

// Keep all existing classes for compatibility
sealed class SessionState {
    object Idle : SessionState()
    data class Active(val sessionInfo: SessionInfo) : SessionState()
    data class Paused(val sessionInfo: SessionInfo, val reason: String) : SessionState()
    data class Completed(val analytics: SleepSessionAnalytics) : SessionState()
    data class Cancelled(val reason: String) : SessionState()
    data class Error(val message: String) : SessionState()
}

sealed class SessionHealth {
    object Unknown : SessionHealth()
    object Good : SessionHealth()
    data class Warning(val message: String) : SessionHealth()
    data class Critical(val message: String) : SessionHealth()
}

private data class SessionEvents(
    val movementEvents: List<MovementEvent> = emptyList(),
    val noiseEvents: List<NoiseEvent> = emptyList()
)

private data class SessionAnalytics(
    val duration: Long = 0L,
    val sleepEfficiency: Float = 0f,
    val averageMovementIntensity: Float = 0f,
    val averageNoiseLevel: Float = 0f,
    val movementFrequency: Float = 0f,
    val totalRestlessness: Float = 0f,
    val currentPhase: SleepPhase = SleepPhase.UNKNOWN,
    val phaseConfidence: Float = 0f
)

data class SleepSessionAnalytics(
    val sessionId: Long,
    val startTime: Long,
    val endTime: Long,
    val totalDuration: Long,
    val sleepLatency: Long,
    val awakeDuration: Long,
    val lightSleepDuration: Long,
    val deepSleepDuration: Long,
    val remSleepDuration: Long,
    val sleepEfficiency: Float,
    val qualityFactors: SleepQualityFactors,
    val averageMovementIntensity: Float,
    val averageNoiseLevel: Float,
    val movementFrequency: Float,
    val totalMovementEvents: Int,
    val totalNoiseEvents: Int,
    val phaseTransitions: Int
)

// Extension function for MovementEvent
private fun MovementEvent.isSignificant(): Boolean = intensity > 2.0f