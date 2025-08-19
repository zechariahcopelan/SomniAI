package com.example.somniai.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.liveData
import androidx.lifecycle.switchMap
import androidx.lifecycle.viewModelScope
import com.example.somniai.data.*
import com.example.somniai.repository.SleepRepository
import com.example.somniai.repository.SessionManager
import com.example.somniai.analytics.SleepAnalyzer
import com.example.somniai.analytics.SessionAnalytics
import com.example.somniai.ai.AIInsightsEngine
import com.example.somniai.ai.*
import com.example.somniai.database.SleepDatabase
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.util.Date
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.roundToInt

/**
 * Legacy sleep info data class for backward compatibility
 */
data class SleepInfo(
    val duration: Long,
    val quality: Float,
    val endTime: Long
)

/**
 * Enhanced tracking state for UI display
 */
data class TrackingState(
    val isTracking: Boolean = false,
    val isPaused: Boolean = false,
    val sessionId: Long? = null,
    val startTime: Long? = null,
    val duration: Long = 0L,
    val phase: SleepPhase = SleepPhase.UNKNOWN,
    val phaseConfidence: Float = 0f,
    val efficiency: Float = 0f
) {
    val formattedDuration: String
        get() {
            val hours = duration / (1000 * 60 * 60)
            val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)
            val seconds = (duration % (1000 * 60)) / 1000
            return if (hours > 0) {
                String.format("%02d:%02d:%02d", hours, minutes, seconds)
            } else {
                String.format("%02d:%02d", minutes, seconds)
            }
        }
}

/**
 * Enhanced live metrics for real-time display
 */
data class EnhancedLiveMetrics(
    val movementIntensity: Float = 0f,
    val movementLevel: String = "Very Low",
    val movementCount: Int = 0,
    val noiseLevel: Float = 0f,
    val noiseDescription: String = "Very Quiet",
    val noiseCount: Int = 0,
    val restlessness: Float = 0f,
    val restlessnessLevel: String = "Very Calm",
    val heartRate: Int? = null, // Placeholder for future feature
    val lastUpdated: Long = System.currentTimeMillis()
) {
    val movementCountText: String
        get() = "$movementCount events"

    val noiseCountText: String
        get() = "$noiseCount events"

    val restlessnessScoreText: String
        get() = String.format("%.1f/10", restlessness)
}

/**
 * Navigation destinations
 */
enum class NavigationDestination {
    HISTORY, CHARTS, SETTINGS, ABOUT, EXPORT
}

/**
 * AI insight generation status for UI feedback
 */
data class AIInsightStatus(
    val isGenerating: Boolean = false,
    val generationType: InsightGenerationType? = null,
    val progress: Float = 0f,
    val stage: String = "",
    val estimatedTimeRemaining: Long = 0L,
    val error: String? = null
) {
    val isComplete: Boolean
        get() = progress >= 1f && !isGenerating

    val hasError: Boolean
        get() = !error.isNullOrBlank()
}

/**
 * AI insights summary for dashboard display
 */
data class AIInsightsSummary(
    val totalInsights: Int = 0,
    val newInsights: Int = 0,
    val highPriorityInsights: Int = 0,
    val completedRecommendations: Int = 0,
    val lastGenerationTime: Long = 0L,
    val nextGenerationTime: Long = 0L,
    val aiEngineHealth: String = "Unknown"
) {
    val hasNewInsights: Boolean
        get() = newInsights > 0

    val hasHighPriorityInsights: Boolean
        get() = highPriorityInsights > 0

    val formattedLastGeneration: String
        get() = if (lastGenerationTime > 0) {
            val diff = System.currentTimeMillis() - lastGenerationTime
            val minutes = diff / (1000 * 60)
            val hours = minutes / 60
            when {
                minutes < 5 -> "Just now"
                minutes < 60 -> "${minutes}m ago"
                hours < 24 -> "${hours}h ago"
                else -> "${hours / 24}d ago"
            }
        } else "Never"
}

/**
 * Enhanced MainViewModel with comprehensive AI insights integration
 *
 * New AI Features:
 * - AI insights generation and management
 * - Real-time AI status monitoring
 * - Personalized insight preferences
 * - AI feedback collection and learning
 * - Multi-type insight generation (session, personalized, predictive)
 * - AI data preparation and export
 * - Performance monitoring for AI operations
 * - Smart scheduling for AI insight generation
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "MainViewModel"
        private const val ANALYTICS_REFRESH_INTERVAL = 60000L // 1 minute
        private const val STATISTICS_CACHE_DURATION = 300000L // 5 minutes
        private const val REAL_TIME_UPDATE_INTERVAL = 1000L // 1 second
        private const val AI_INSIGHTS_REFRESH_INTERVAL = 300000L // 5 minutes
        private const val AUTO_AI_GENERATION_DELAY = 120000L // 2 minutes after session completion
    }

    // Repository dependencies
    private lateinit var sleepRepository: SleepRepository
    private lateinit var sessionManager: SessionManager
    private lateinit var sleepAnalyzer: SleepAnalyzer
    private lateinit var sessionAnalytics: SessionAnalytics
    private lateinit var aiInsightsEngine: AIInsightsEngine

    // Initialization state
    private val _isInitialized = MutableLiveData<Boolean>()
    val isInitialized: LiveData<Boolean> = _isInitialized

    // Data loading states
    private val _isLoadingStatistics = MutableLiveData<Boolean>()
    val isLoadingStatistics: LiveData<Boolean> = _isLoadingStatistics

    private val _isLoadingSessions = MutableLiveData<Boolean>()
    val isLoadingSessions: LiveData<Boolean> = _isLoadingSessions

    private val _isLoadingAnalytics = MutableLiveData<Boolean>()
    val isLoadingAnalytics: LiveData<Boolean> = _isLoadingAnalytics

    // Enhanced tracking state
    private val _trackingState = MutableLiveData<TrackingState>()
    val trackingState: LiveData<TrackingState> = _trackingState

    private val _isTracking = MutableLiveData<Boolean>()
    val isTracking: LiveData<Boolean> = _isTracking

    private val _isPaused = MutableLiveData<Boolean>()
    val isPaused: LiveData<Boolean> = _isPaused

    private val _trackingStartTime = MutableLiveData<Long?>()
    val trackingStartTime: LiveData<Long?> = _trackingStartTime

    // Current session management
    private val _currentSessionId = MutableLiveData<Long?>()
    val currentSessionId: LiveData<Long?> = _currentSessionId

    private val _currentSession = MutableLiveData<SleepSession?>()
    val currentSession: LiveData<SleepSession?> = _currentSession

    // Sleep statistics from database
    private val _totalSessions = MutableLiveData<Int>()
    val totalSessions: LiveData<Int> = _totalSessions

    private val _averageSleepDuration = MutableLiveData<Long>()
    val averageSleepDuration: LiveData<Long> = _averageSleepDuration

    private val _averageSleepQuality = MutableLiveData<Float>()
    val averageSleepQuality: LiveData<Float> = _averageSleepQuality

    private val _averageSleepEfficiency = MutableLiveData<Float>()
    val averageSleepEfficiency: LiveData<Float> = _averageSleepEfficiency

    // Historical data
    private val _recentSessions = MutableLiveData<List<SleepSession>>()
    val recentSessions: LiveData<List<SleepSession>> = _recentSessions

    private val _lastSleepInfo = MutableLiveData<SleepInfo?>()
    val lastSleepInfo: LiveData<SleepInfo?> = _lastSleepInfo

    private val _sessionStatistics = MutableLiveData<SessionStatistics?>()
    val sessionStatistics: LiveData<SessionStatistics?> = _sessionStatistics

    // Enhanced real-time data
    private val _sensorStatus = MutableLiveData<SensorStatus>()
    val sensorStatus: LiveData<SensorStatus> = _sensorStatus

    private val _liveMetrics = MutableLiveData<LiveSleepMetrics?>()
    val liveMetrics: LiveData<LiveSleepMetrics?> = _liveMetrics

    private val _enhancedLiveMetrics = MutableLiveData<EnhancedLiveMetrics>()
    val enhancedLiveMetrics: LiveData<EnhancedLiveMetrics> = _enhancedLiveMetrics

    // Analytics and insights
    private val _sleepAnalytics = MutableLiveData<SleepAnalytics?>()
    val sleepAnalytics: LiveData<SleepAnalytics?> = _sleepAnalytics

    private val _sleepTrends = MutableLiveData<List<DailyTrendData>>()
    val sleepTrends: LiveData<List<DailyTrendData>> = _sleepTrends

    private val _qualityReport = MutableLiveData<SleepQualityReport?>()
    val qualityReport: LiveData<SleepQualityReport?> = _qualityReport

    private val _insights = MutableLiveData<List<SleepInsight>>()
    val insights: LiveData<List<SleepInsight>> = _insights

    // Comparative analysis
    private val _performanceComparison = MutableLiveData<ComparativeAnalysisResult?>()
    val performanceComparison: LiveData<ComparativeAnalysisResult?> = _performanceComparison

    private val _monthlyStats = MutableLiveData<List<MonthlyStatsData>>()
    val monthlyStats: LiveData<List<MonthlyStatsData>> = _monthlyStats

    // ========== NEW AI INSIGHTS LIVEDATA ==========

    // AI Engine Status and Health
    private val _aiEngineStatus = MutableLiveData<EngineState>()
    val aiEngineStatus: LiveData<EngineState> = _aiEngineStatus

    private val _aiEngineHealth = MutableLiveData<EngineHealth>()
    val aiEngineHealth: LiveData<EngineHealth> = _aiEngineHealth

    private val _aiInsightStatus = MutableLiveData<AIInsightStatus>()
    val aiInsightStatus: LiveData<AIInsightStatus> = _aiInsightStatus

    private val _aiInsightsSummary = MutableLiveData<AIInsightsSummary>()
    val aiInsightsSummary: LiveData<AIInsightsSummary> = _aiInsightsSummary

    // Session-specific AI insights
    private val _sessionInsights = MutableLiveData<List<SleepInsight>>()
    val sessionInsights: LiveData<List<SleepInsight>> = _sessionInsights

    private val _currentSessionInsights = MutableLiveData<List<SleepInsight>>()
    val currentSessionInsights: LiveData<List<SleepInsight>> = _currentSessionInsights

    // Personalized AI insights
    private val _personalizedInsights = MutableLiveData<List<SleepInsight>>()
    val personalizedInsights: LiveData<List<SleepInsight>> = _personalizedInsights

    private val _dailyInsights = MutableLiveData<List<SleepInsight>>()
    val dailyInsights: LiveData<List<SleepInsight>> = _dailyInsights

    private val _weeklyInsights = MutableLiveData<List<SleepInsight>>()
    val weeklyInsights: LiveData<List<SleepInsight>> = _weeklyInsights

    // Predictive AI insights
    private val _predictiveInsights = MutableLiveData<List<SleepInsight>>()
    val predictiveInsights: LiveData<List<SleepInsight>> = _predictiveInsights

    private val _trendPredictions = MutableLiveData<List<SleepInsight>>()
    val trendPredictions: LiveData<List<SleepInsight>> = _trendPredictions

    // AI insights by category
    private val _qualityInsights = MutableLiveData<List<SleepInsight>>()
    val qualityInsights: LiveData<List<SleepInsight>> = _qualityInsights

    private val _durationInsights = MutableLiveData<List<SleepInsight>>()
    val durationInsights: LiveData<List<SleepInsight>> = _durationInsights

    private val _timingInsights = MutableLiveData<List<SleepInsight>>()
    val timingInsights: LiveData<List<SleepInsight>> = _timingInsights

    private val _environmentInsights = MutableLiveData<List<SleepInsight>>()
    val environmentInsights: LiveData<List<SleepInsight>> = _environmentInsights

    // AI performance and analytics
    private val _aiPerformanceMetrics = MutableLiveData<EnginePerformanceMetrics>()
    val aiPerformanceMetrics: LiveData<EnginePerformanceMetrics> = _aiPerformanceMetrics

    private val _aiGenerationStatistics = MutableLiveData<InsightGenerationStatistics>()
    val aiGenerationStatistics: LiveData<InsightGenerationStatistics> = _aiGenerationStatistics

    // AI user preferences and feedback
    private val _aiPreferences = MutableLiveData<UserInsightPreferences>()
    val aiPreferences: LiveData<UserInsightPreferences> = _aiPreferences

    private val _insightFeedback = MutableLiveData<Map<Long, InsightFeedback>>()
    val insightFeedback: LiveData<Map<Long, InsightFeedback>> = _insightFeedback

    // AI data export and sharing
    private val _aiDataSummary = MutableLiveData<AIDataSummary?>()
    val aiDataSummary: LiveData<AIDataSummary?> = _aiDataSummary

    private val _exportedAIData = MutableLiveData<AIAnalysisExport?>()
    val exportedAIData: LiveData<AIAnalysisExport?> = _exportedAIData

    // Navigation and UI state
    private val _navigationEvent = MutableLiveData<NavigationDestination?>()
    val navigationEvent: LiveData<NavigationDestination?> = _navigationEvent

    private val _showConfirmDialog = MutableLiveData<String?>()
    val showConfirmDialog: LiveData<String?> = _showConfirmDialog

    private val _statusMessage = MutableLiveData<String?>()
    val statusMessage: LiveData<String?> = _statusMessage

    // Error handling and status
    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    private val _isServiceConnected = MutableLiveData<Boolean>()
    val isServiceConnected: LiveData<Boolean> = _isServiceConnected

    private val _dataIntegrity = MutableLiveData<DataIntegrityStatus>()
    val dataIntegrity: LiveData<DataIntegrityStatus> = _dataIntegrity

    // Performance monitoring
    private val _performanceMetrics = MutableLiveData<PerformanceMetrics>()
    val performanceMetrics: LiveData<PerformanceMetrics> = _performanceMetrics

    // Internal state
    private var lastStatisticsLoad = 0L
    private var isAnalyticsJobRunning = false
    private var realTimeUpdateJob: kotlinx.coroutines.Job? = null
    private var aiInsightsUpdateJob: kotlinx.coroutines.Job? = null
    private var lastAIInsightsRefresh = 0L

    init {
        // Initialize with default values
        _isInitialized.value = false
        _trackingState.value = TrackingState()
        _isTracking.value = false
        _isPaused.value = false
        _isServiceConnected.value = false
        _isLoadingStatistics.value = false
        _isLoadingSessions.value = false
        _isLoadingAnalytics.value = false
        _enhancedLiveMetrics.value = EnhancedLiveMetrics()

        // Initialize AI-specific state
        _aiEngineStatus.value = EngineState.INITIALIZING
        _aiEngineHealth.value = EngineHealth.UNKNOWN
        _aiInsightStatus.value = AIInsightStatus()
        _aiInsightsSummary.value = AIInsightsSummary()

        // Initialize repository and analytics
        initializeComponents()
    }

    /**
     * Initialize repository and analytics components including AI engine
     */
    private fun initializeComponents() {
        viewModelScope.launch {
            try {
                _statusMessage.value = "Initializing SomniAI..."

                val database = SleepDatabase.getDatabase(getApplication())
                sleepRepository = SleepRepository(database, getApplication())
                sessionManager = SessionManager(sleepRepository)
                sleepAnalyzer = SleepAnalyzer()
                sessionAnalytics = SessionAnalytics()

                // Initialize AI Insights Engine
                initializeAIEngine()

                _isInitialized.value = true
                _statusMessage.value = "SomniAI ready"

                // Load initial data
                loadInitialData()

                // Start real-time updates
                startRealTimeUpdates()
                startAIInsightsMonitoring()

                Log.d(TAG, "ViewModel components initialized successfully")

            } catch (e: Exception) {
                setError("Failed to initialize ViewModel: ${e.message}")
                Log.e(TAG, "Error initializing ViewModel components", e)
            }
        }
    }

    /**
     * Initialize AI Insights Engine with proper configuration
     */
    private suspend fun initializeAIEngine() {
        try {
            _statusMessage.value = "Initializing AI Engine..."

            // TODO: In real implementation, get these from SharedPreferences or config
            val preferences = getApplication<Application>().getSharedPreferences("somniai_ai", 0)

            aiInsightsEngine = AIInsightsEngine(
                context = getApplication(),
                repository = sleepRepository,
                sleepAnalyzer = sleepAnalyzer,
                sessionAnalytics = sessionAnalytics,
                promptBuilder = InsightsPromptBuilder(), // TODO: Initialize properly
                preferences = preferences
            )

            // Initialize the engine
            val initResult = aiInsightsEngine.initialize()
            initResult.onSuccess {
                _aiEngineStatus.value = EngineState.ACTIVE
                _aiEngineHealth.value = EngineHealth.GOOD
                Log.d(TAG, "AI Insights Engine initialized successfully")

                // Start monitoring AI engine status
                monitorAIEngineHealth()

            }.onFailure { exception ->
                _aiEngineStatus.value = EngineState.ERROR(exception.message ?: "Initialization failed")
                _aiEngineHealth.value = EngineHealth.CRITICAL(exception.message ?: "Initialization failed")
                Log.e(TAG, "Failed to initialize AI Insights Engine", exception)
            }

        } catch (e: Exception) {
            _aiEngineStatus.value = EngineState.ERROR(e.message ?: "Unknown error")
            _aiEngineHealth.value = EngineHealth.CRITICAL(e.message ?: "Unknown error")
            Log.e(TAG, "Exception initializing AI Engine", e)
        }
    }

    /**
     * Start monitoring AI engine health and performance
     */
    private fun monitorAIEngineHealth() {
        viewModelScope.launch {
            try {
                // Collect AI engine state
                aiInsightsEngine.engineState.collect { state ->
                    _aiEngineStatus.value = state
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error monitoring AI engine state", e)
            }
        }

        viewModelScope.launch {
            try {
                // Collect AI engine health
                aiInsightsEngine.engineHealth.collect { health ->
                    _aiEngineHealth.value = health
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error monitoring AI engine health", e)
            }
        }

        viewModelScope.launch {
            try {
                // Collect AI performance metrics
                aiInsightsEngine.performanceMetrics.observe(getApplication()) { metrics ->
                    _aiPerformanceMetrics.value = metrics
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error monitoring AI performance metrics", e)
            }
        }
    }

    /**
     * Load initial data from repository including AI insights
     */
    private fun loadInitialData() {
        viewModelScope.launch {
            try {
                // Load basic statistics
                loadSleepStatistics()

                // Load recent sessions
                loadRecentSessions()

                // Load analytics
                loadSleepAnalytics()

                // Load AI insights
                loadAIInsights()

                // Load unacknowledged insights
                loadUnacknowledgedInsights()

                // Check for active session
                checkForActiveSession()

            } catch (e: Exception) {
                setError("Failed to load initial data: ${e.message}")
                Log.e(TAG, "Error loading initial data", e)
            }
        }
    }

    /**
     * Start AI insights monitoring and periodic updates
     */
    private fun startAIInsightsMonitoring() {
        aiInsightsUpdateJob?.cancel()
        aiInsightsUpdateJob = viewModelScope.launch {
            while (true) {
                delay(AI_INSIGHTS_REFRESH_INTERVAL)

                try {
                    // Check if AI engine is healthy
                    if (_aiEngineStatus.value == EngineState.ACTIVE) {
                        // Refresh AI insights summary
                        updateAIInsightsSummary()

                        // Check for scheduled insight generation
                        checkScheduledInsightGeneration()

                        // Update AI performance statistics
                        updateAIPerformanceStatistics()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in AI insights monitoring", e)
                }
            }
        }
    }

    // ========== AI INSIGHTS GENERATION METHODS ==========

    /**
     * Generate AI insights for the current session
     */
    fun generateSessionInsights(
        sessionId: Long? = null,
        options: InsightGenerationOptions = InsightGenerationOptions.default()
    ) {
        val targetSessionId = sessionId ?: _currentSessionId.value
        if (targetSessionId == null) {
            Log.w(TAG, "No session ID available for insight generation")
            return
        }

        viewModelScope.launch {
            try {
                updateAIInsightStatus(
                    isGenerating = true,
                    generationType = InsightGenerationType.POST_SESSION,
                    stage = "Preparing session data..."
                )

                Log.d(TAG, "Generating AI insights for session: $targetSessionId")

                val result = aiInsightsEngine.generateSessionInsights(
                    sessionId = targetSessionId,
                    options = options
                )

                result.onSuccess { insights ->
                    _currentSessionInsights.value = insights
                    _sessionInsights.value = insights

                    // Categorize insights
                    categorizeInsights(insights)

                    // Update summary
                    updateAIInsightsSummary()

                    updateAIInsightStatus(
                        isGenerating = false,
                        progress = 1f,
                        stage = "Insights generated successfully"
                    )

                    Log.d(TAG, "Generated ${insights.size} session insights")

                }.onFailure { exception ->
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Failed to generate session insights: ${exception.message}"
                    )
                    Log.e(TAG, "Failed to generate session insights", exception)
                }

            } catch (e: Exception) {
                updateAIInsightStatus(
                    isGenerating = false,
                    error = "Error generating session insights: ${e.message}"
                )
                Log.e(TAG, "Exception generating session insights", e)
            }
        }
    }

    /**
     * Generate personalized AI insights based on user patterns
     */
    fun generatePersonalizedInsights(
        analysisDepth: AnalysisDepth = AnalysisDepth.COMPREHENSIVE,
        personalizationLevel: PersonalizationLevel = PersonalizationLevel.ADAPTIVE,
        options: InsightGenerationOptions = InsightGenerationOptions.default()
    ) {
        viewModelScope.launch {
            try {
                updateAIInsightStatus(
                    isGenerating = true,
                    generationType = InsightGenerationType.PERSONALIZED_ANALYSIS,
                    stage = "Analyzing sleep patterns..."
                )

                Log.d(TAG, "Generating personalized AI insights")

                val result = aiInsightsEngine.generatePersonalizedInsights(
                    analysisDepth = analysisDepth,
                    personalizationLevel = personalizationLevel,
                    options = options
                )

                result.onSuccess { insights ->
                    _personalizedInsights.value = insights

                    // Categorize insights
                    categorizeInsights(insights)

                    // Update summary
                    updateAIInsightsSummary()

                    updateAIInsightStatus(
                        isGenerating = false,
                        progress = 1f,
                        stage = "Personalized insights generated"
                    )

                    Log.d(TAG, "Generated ${insights.size} personalized insights")

                }.onFailure { exception ->
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Failed to generate personalized insights: ${exception.message}"
                    )
                    Log.e(TAG, "Failed to generate personalized insights", exception)
                }

            } catch (e: Exception) {
                updateAIInsightStatus(
                    isGenerating = false,
                    error = "Error generating personalized insights: ${e.message}"
                )
                Log.e(TAG, "Exception generating personalized insights", e)
            }
        }
    }

    /**
     * Generate daily AI insights with trend analysis
     */
    fun generateDailyInsights(
        daysBack: Int = 7,
        includePredictive: Boolean = true,
        options: InsightGenerationOptions = InsightGenerationOptions.default()
    ) {
        viewModelScope.launch {
            try {
                updateAIInsightStatus(
                    isGenerating = true,
                    generationType = InsightGenerationType.DAILY_ANALYSIS,
                    stage = "Analyzing recent sleep trends..."
                )

                Log.d(TAG, "Generating daily AI insights")

                val result = aiInsightsEngine.generateDailyInsights(
                    daysBack = daysBack,
                    includePredictive = includePredictive,
                    options = options
                )

                result.onSuccess { insights ->
                    _dailyInsights.value = insights

                    // Categorize insights
                    categorizeInsights(insights)

                    // Update summary
                    updateAIInsightsSummary()

                    updateAIInsightStatus(
                        isGenerating = false,
                        progress = 1f,
                        stage = "Daily insights generated"
                    )

                    Log.d(TAG, "Generated ${insights.size} daily insights")

                }.onFailure { exception ->
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Failed to generate daily insights: ${exception.message}"
                    )
                    Log.e(TAG, "Failed to generate daily insights", exception)
                }

            } catch (e: Exception) {
                updateAIInsightStatus(
                    isGenerating = false,
                    error = "Error generating daily insights: ${e.message}"
                )
                Log.e(TAG, "Exception generating daily insights", e)
            }
        }
    }

    /**
     * Generate predictive insights based on trend analysis
     */
    fun generatePredictiveInsights(
        predictionHorizon: PredictionHorizon = PredictionHorizon.WEEK,
        confidence: Float = 0.7f
    ) {
        viewModelScope.launch {
            try {
                updateAIInsightStatus(
                    isGenerating = true,
                    generationType = InsightGenerationType.PREDICTIVE_ANALYSIS,
                    stage = "Analyzing predictive patterns..."
                )

                Log.d(TAG, "Generating predictive AI insights")

                // First get trend analysis from recent sessions
                val recentSessions = _recentSessions.value ?: emptyList()
                if (recentSessions.size < 14) {
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Insufficient data for predictive analysis (need at least 14 sessions)"
                    )
                    return@launch
                }

                val trendAnalysis = sleepAnalyzer.analyzeTrends(recentSessions)
                if (!trendAnalysis.hasSufficientData) {
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Insufficient data quality for reliable predictions"
                    )
                    return@launch
                }

                val result = aiInsightsEngine.generatePredictiveInsights(
                    predictionHorizon = predictionHorizon,
                    confidence = confidence,
                    triggeringTrends = listOf(TrendAnalysisResult(trendAnalysis))
                )

                result.onSuccess { insights ->
                    _predictiveInsights.value = insights
                    _trendPredictions.value = insights.filter {
                        it.category == InsightCategory.PATTERN || it.category == InsightCategory.TIMING
                    }

                    // Update summary
                    updateAIInsightsSummary()

                    updateAIInsightStatus(
                        isGenerating = false,
                        progress = 1f,
                        stage = "Predictive insights generated"
                    )

                    Log.d(TAG, "Generated ${insights.size} predictive insights")

                }.onFailure { exception ->
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Failed to generate predictive insights: ${exception.message}"
                    )
                    Log.e(TAG, "Failed to generate predictive insights", exception)
                }

            } catch (e: Exception) {
                updateAIInsightStatus(
                    isGenerating = false,
                    error = "Error generating predictive insights: ${e.message}"
                )
                Log.e(TAG, "Exception generating predictive insights", e)
            }
        }
    }

    /**
     * Generate comprehensive weekly insights
     */
    fun generateWeeklyInsights() {
        viewModelScope.launch {
            try {
                updateAIInsightStatus(
                    isGenerating = true,
                    generationType = InsightGenerationType.WEEKLY_SUMMARY,
                    stage = "Analyzing weekly patterns..."
                )

                Log.d(TAG, "Generating weekly AI insights")

                val options = InsightGenerationOptions(
                    maxInsights = 7,
                    includeAIInsights = true,
                    includeMLEnhancement = true,
                    personalizationLevel = PersonalizationLevel.ADAPTIVE
                )

                val result = aiInsightsEngine.generateDailyInsights(
                    daysBack = 7,
                    includePredictive = true,
                    options = options
                )

                result.onSuccess { insights ->
                    _weeklyInsights.value = insights

                    // Update summary
                    updateAIInsightsSummary()

                    updateAIInsightStatus(
                        isGenerating = false,
                        progress = 1f,
                        stage = "Weekly insights generated"
                    )

                    Log.d(TAG, "Generated ${insights.size} weekly insights")

                }.onFailure { exception ->
                    updateAIInsightStatus(
                        isGenerating = false,
                        error = "Failed to generate weekly insights: ${exception.message}"
                    )
                    Log.e(TAG, "Failed to generate weekly insights", exception)
                }

            } catch (e: Exception) {
                updateAIInsightStatus(
                    isGenerating = false,
                    error = "Error generating weekly insights: ${e.message}"
                )
                Log.e(TAG, "Exception generating weekly insights", e)
            }
        }
    }

    // ========== AI DATA PREPARATION AND EXPORT METHODS ==========

    /**
     * Generate AI-ready data summary for external use
     */
    fun generateAIDataSummary(
        format: AIDataFormat = AIDataFormat.NARRATIVE,
        includeRecommendations: Boolean = true
    ) {
        viewModelScope.launch {
            try {
                val sessions = _recentSessions.value ?: emptyList()
                if (sessions.isEmpty()) {
                    Log.w(TAG, "No sessions available for AI data summary")
                    return@launch
                }

                Log.d(TAG, "Generating AI data summary")

                val summary = sleepAnalyzer.generateAIDataSummary(
                    sessions = sessions,
                    format = format,
                    includeRecommendations = includeRecommendations
                )

                _aiDataSummary.value = summary

                Log.d(TAG, "Generated AI data summary: ${summary.summary.length} characters")

            } catch (e: Exception) {
                Log.e(TAG, "Error generating AI data summary", e)
                setError("Failed to generate AI data summary: ${e.message}")
            }
        }
    }

    /**
     * Export comprehensive analysis data for AI consumption
     */
    fun exportAnalysisForAI(
        exportFormat: AIExportFormat = AIExportFormat.COMPREHENSIVE_JSON,
        includeRawData: Boolean = false,
        includeAnalytics: Boolean = true,
        includeRecommendations: Boolean = true
    ) {
        viewModelScope.launch {
            try {
                val sessions = _recentSessions.value ?: emptyList()
                if (sessions.isEmpty()) {
                    Log.w(TAG, "No sessions available for AI export")
                    return@launch
                }

                Log.d(TAG, "Exporting analysis for AI")

                val exportData = sleepAnalyzer.exportAnalysisForAI(
                    sessions = sessions,
                    exportFormat = exportFormat,
                    includeRawData = includeRawData,
                    includeAnalytics = includeAnalytics,
                    includeRecommendations = includeRecommendations
                )

                _exportedAIData.value = exportData

                Log.d(TAG, "Exported AI analysis data: ${exportData.dataSize} bytes")

            } catch (e: Exception) {
                Log.e(TAG, "Error exporting analysis for AI", e)
                setError("Failed to export AI analysis: ${e.message}")
            }
        }
    }

    // ========== AI FEEDBACK AND LEARNING METHODS ==========

    /**
     * Record feedback for AI insight effectiveness
     */
    fun recordInsightFeedback(
        insightId: Long,
        feedback: InsightFeedback,
        engagementMetrics: EngagementMetrics? = null
    ) {
        viewModelScope.launch {
            try {
                Log.d(TAG, "Recording insight feedback: $insightId")

                // Record in repository
                val result = sleepRepository.recordInsightFeedback(
                    insightId = insightId,
                    feedback = feedback,
                    implementation = null // Could be extended to track implementation
                )

                result.onSuccess {
                    // Record in AI engine for learning
                    aiInsightsEngine.recordInsightEffectiveness(
                        insightId = insightId,
                        feedback = feedback,
                        engagementMetrics = engagementMetrics
                    )

                    // Update local feedback state
                    val currentFeedback = _insightFeedback.value?.toMutableMap() ?: mutableMapOf()
                    currentFeedback[insightId] = feedback
                    _insightFeedback.value = currentFeedback

                    Log.d(TAG, "Insight feedback recorded successfully")

                }.onFailure { exception ->
                    Log.e(TAG, "Failed to record insight feedback", exception)
                    setError("Failed to record feedback: ${exception.message}")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error recording insight feedback", e)
                setError("Error recording feedback: ${e.message}")
            }
        }
    }

    /**
     * Update user AI preferences
     */
    fun updateAIPreferences(preferences: UserInsightPreferences) {
        viewModelScope.launch {
            try {
                Log.d(TAG, "Updating AI preferences")

                // Update preferences in repository/storage
                // TODO: Implement preference storage

                _aiPreferences.value = preferences

                // Trigger regeneration of personalized insights if needed
                if (preferences.preferredCategories.isNotEmpty()) {
                    generatePersonalizedInsights(
                        options = InsightGenerationOptions(
                            categoryFilter = preferences.preferredCategories.toList(),
                            personalizationLevel = PersonalizationLevel.ADVANCED
                        )
                    )
                }

                Log.d(TAG, "AI preferences updated successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Error updating AI preferences", e)
                setError("Error updating preferences: ${e.message}")
            }
        }
    }

    // ========== AI PERFORMANCE AND MONITORING METHODS ==========

    /**
     * Get AI engine performance analytics
     */
    fun getAIPerformanceAnalytics() {
        viewModelScope.launch {
            try {
                Log.d(TAG, "Getting AI performance analytics")

                val analytics = aiInsightsEngine.getPerformanceAnalytics()
                // TODO: Process and store analytics data

                val statistics = aiInsightsEngine.getGenerationStatistics()
                _aiGenerationStatistics.value = statistics

                Log.d(TAG, "AI performance analytics retrieved")

            } catch (e: Exception) {
                Log.e(TAG, "Error getting AI performance analytics", e)
            }
        }
    }

    /**
     * Run comprehensive AI health check
     */
    fun runAIHealthCheck() {
        viewModelScope.launch {
            try {
                Log.d(TAG, "Running AI health check")

                val healthResult = aiInsightsEngine.runComprehensiveHealthCheck()

                if (healthResult.isHealthy) {
                    _aiEngineHealth.value = EngineHealth.EXCELLENT
                    Log.d(TAG, "AI health check passed: score=${healthResult.healthScore}")
                } else {
                    _aiEngineHealth.value = EngineHealth.WARNING(
                        "Health issues found: ${healthResult.issues.size} issues"
                    )
                    Log.w(TAG, "AI health check found issues: ${healthResult.issues}")
                }

            } catch (e: Exception) {
                _aiEngineHealth.value = EngineHealth.CRITICAL("Health check failed: ${e.message}")
                Log.e(TAG, "AI health check failed", e)
            }
        }
    }

    /**
     * Optimize AI engine performance
     */
    fun optimizeAIPerformance() {
        viewModelScope.launch {
            try {
                Log.d(TAG, "Optimizing AI performance")

                val result = aiInsightsEngine.optimizePerformance()
                result.onSuccess { optimizationResult ->
                    Log.d(TAG, "AI optimization completed: improvement=${optimizationResult.overallImprovementScore}")

                    // Update performance metrics
                    updateAIPerformanceStatistics()

                }.onFailure { exception ->
                    Log.e(TAG, "AI optimization failed", exception)
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error optimizing AI performance", e)
            }
        }
    }

    // ========== AI HELPER METHODS ==========

    /**
     * Update AI insight generation status
     */
    private fun updateAIInsightStatus(
        isGenerating: Boolean = false,
        generationType: InsightGenerationType? = null,
        progress: Float = 0f,
        stage: String = "",
        estimatedTimeRemaining: Long = 0L,
        error: String? = null
    ) {
        _aiInsightStatus.value = AIInsightStatus(
            isGenerating = isGenerating,
            generationType = generationType,
            progress = progress,
            stage = stage,
            estimatedTimeRemaining = estimatedTimeRemaining,
            error = error
        )
    }

    /**
     * Categorize insights by type for organized display
     */
    private fun categorizeInsights(insights: List<SleepInsight>) {
        val qualityInsights = insights.filter { it.category == InsightCategory.QUALITY }
        val durationInsights = insights.filter { it.category == InsightCategory.DURATION }
        val timingInsights = insights.filter { it.category == InsightCategory.TIMING }
        val environmentInsights = insights.filter { it.category == InsightCategory.ENVIRONMENT }

        _qualityInsights.value = qualityInsights
        _durationInsights.value = durationInsights
        _timingInsights.value = timingInsights
        _environmentInsights.value = environmentInsights
    }

    /**
     * Update AI insights summary
     */
    private fun updateAIInsightsSummary() {
        try {
            val allInsights = (_insights.value ?: emptyList()) +
                    (_sessionInsights.value ?: emptyList()) +
                    (_personalizedInsights.value ?: emptyList()) +
                    (_dailyInsights.value ?: emptyList()) +
                    (_predictiveInsights.value ?: emptyList())

            val uniqueInsights = allInsights.distinctBy { it.id }
            val newInsights = uniqueInsights.filter { !it.isAcknowledged }
            val highPriorityInsights = uniqueInsights.filter { it.priority == 1 }

            val summary = AIInsightsSummary(
                totalInsights = uniqueInsights.size,
                newInsights = newInsights.size,
                highPriorityInsights = highPriorityInsights.size,
                completedRecommendations = uniqueInsights.count { insight ->
                    _insightFeedback.value?.get(insight.id)?.wasImplemented == true
                },
                lastGenerationTime = uniqueInsights.maxOfOrNull { it.timestamp } ?: 0L,
                nextGenerationTime = calculateNextGenerationTime(),
                aiEngineHealth = when (_aiEngineHealth.value) {
                    is EngineHealth.EXCELLENT -> "Excellent"
                    is EngineHealth.GOOD -> "Good"
                    is EngineHealth.WARNING -> "Warning"
                    is EngineHealth.CRITICAL -> "Critical"
                    else -> "Unknown"
                }
            )

            _aiInsightsSummary.value = summary

        } catch (e: Exception) {
            Log.e(TAG, "Error updating AI insights summary", e)
        }
    }

    /**
     * Check for scheduled insight generation
     */
    private suspend fun checkScheduledInsightGeneration() {
        try {
            val currentTime = System.currentTimeMillis()
            val lastRefresh = lastAIInsightsRefresh

            // Auto-generate insights after session completion
            _currentSession.value?.let { session ->
                if (session.endTime != null &&
                    currentTime - (session.endTime ?: 0L) > AUTO_AI_GENERATION_DELAY &&
                    _currentSessionInsights.value.isNullOrEmpty()) {

                    Log.d(TAG, "Auto-generating insights for completed session")
                    generateSessionInsights(session.id)
                }
            }

            // Periodic personalized insights generation
            if (currentTime - lastRefresh > AI_INSIGHTS_REFRESH_INTERVAL) {
                val recentSessions = _recentSessions.value ?: emptyList()
                if (recentSessions.size >= 5) {
                    Log.d(TAG, "Auto-generating personalized insights")
                    generatePersonalizedInsights()
                    lastAIInsightsRefresh = currentTime
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error checking scheduled insight generation", e)
        }
    }

    /**
     * Calculate next scheduled generation time
     */
    private fun calculateNextGenerationTime(): Long {
        val currentTime = System.currentTimeMillis()
        return currentTime + AI_INSIGHTS_REFRESH_INTERVAL
    }

    /**
     * Update AI performance statistics
     */
    private fun updateAIPerformanceStatistics() {
        viewModelScope.launch {
            try {
                val statistics = aiInsightsEngine.getGenerationStatistics()
                _aiGenerationStatistics.value = statistics
            } catch (e: Exception) {
                Log.e(TAG, "Error updating AI performance statistics", e)
            }
        }
    }

    /**
     * Load all AI insights from repository
     */
    private fun loadAIInsights() {
        viewModelScope.launch {
            try {
                // Load all insights
                val allInsightsResult = sleepRepository.getAllInsights()
                allInsightsResult.onSuccess { insights ->
                    _insights.value = insights

                    // Categorize insights
                    categorizeInsights(insights)

                    // Update summary
                    updateAIInsightsSummary()

                    Log.d(TAG, "Loaded ${insights.size} AI insights")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error loading AI insights", e)
            }
        }
    }

    // ========== EXISTING METHODS (keeping all original functionality) ==========

    /**
     * Start real-time updates for tracking state
     */
    private fun startRealTimeUpdates() {
        realTimeUpdateJob?.cancel()
        realTimeUpdateJob = viewModelScope.launch {
            while (true) {
                if (_isTracking.value == true) {
                    updateTrackingState()
                }
                delay(REAL_TIME_UPDATE_INTERVAL)
            }
        }
    }

    /**
     * Update tracking state with current duration and data
     */
    private fun updateTrackingState() {
        val startTime = _trackingStartTime.value ?: return
        val currentDuration = System.currentTimeMillis() - startTime
        val liveMetrics = _liveMetrics.value

        val updatedState = _trackingState.value?.copy(
            isTracking = _isTracking.value ?: false,
            isPaused = _isPaused.value ?: false,
            sessionId = _currentSessionId.value,
            startTime = startTime,
            duration = currentDuration,
            phase = liveMetrics?.currentPhase ?: SleepPhase.UNKNOWN,
            phaseConfidence = liveMetrics?.phaseConfidence ?: 0f,
            efficiency = liveMetrics?.sleepEfficiency ?: 0f
        ) ?: TrackingState(
            isTracking = _isTracking.value ?: false,
            duration = currentDuration
        )

        _trackingState.value = updatedState
    }

    // ========== SESSION MANAGEMENT ==========

    /**
     * Start tracking with repository session creation
     */
    fun startTracking(settings: SensorSettings? = null) {
        if (_isTracking.value == true) {
            Log.w(TAG, "Tracking already active")
            return
        }

        viewModelScope.launch {
            try {
                _statusMessage.value = "Starting sleep tracking..."
                val startTime = System.currentTimeMillis()
                val sessionResult = sleepRepository.createSession(startTime, settings)

                sessionResult.onSuccess { sessionId ->
                    _currentSessionId.value = sessionId
                    _trackingStartTime.value = startTime
                    _isTracking.value = true
                    _isPaused.value = false
                    _currentSession.value = null // Will be loaded when available

                    updateTrackingState()
                    _statusMessage.value = "Sleep tracking active"

                    Log.d(TAG, "Tracking started with session ID: $sessionId")
                    clearError()

                }.onFailure { exception ->
                    setError("Failed to start tracking: ${exception.message}")
                    Log.e(TAG, "Error starting tracking", exception)
                }

            } catch (e: Exception) {
                setError("Error starting tracking: ${e.message}")
                Log.e(TAG, "Exception starting tracking", e)
            }
        }
    }

    /**
     * Stop tracking and complete session with AI insight generation
     */
    fun stopTracking() {
        if (_isTracking.value != true) {
            Log.w(TAG, "Tracking not active")
            return
        }

        viewModelScope.launch {
            try {
                _statusMessage.value = "Completing sleep session..."
                val endTime = System.currentTimeMillis()
                val sessionId = _currentSessionId.value

                if (sessionId != null) {
                    // Complete session with analytics
                    val completionResult = sleepRepository.completeSession(endTime, null)

                    completionResult.onSuccess { completedSession ->
                        _isTracking.value = false
                        _isPaused.value = false
                        _currentSessionId.value = null
                        _trackingStartTime.value = null
                        _currentSession.value = completedSession
                        _trackingState.value = TrackingState()

                        // Refresh statistics and analytics
                        loadSleepStatistics()
                        loadRecentSessions()
                        loadSleepAnalytics()

                        // Schedule AI insight generation
                        viewModelScope.launch {
                            delay(5000) // Wait 5 seconds for data to settle
                            generateSessionInsights(completedSession.id)
                        }

                        _statusMessage.value = "Sleep session completed"
                        Log.d(TAG, "Tracking stopped and session completed: ${completedSession.id}")

                    }.onFailure { exception ->
                        setError("Failed to complete session: ${exception.message}")
                        Log.e(TAG, "Error completing session", exception)
                    }
                } else {
                    // Just stop tracking without session completion
                    _isTracking.value = false
                    _isPaused.value = false
                    _trackingStartTime.value = null
                    _trackingState.value = TrackingState()
                    _statusMessage.value = "Ready to track your sleep"
                    Log.w(TAG, "Stopped tracking without session ID")
                }

            } catch (e: Exception) {
                setError("Error stopping tracking: ${e.message}")
                Log.e(TAG, "Exception stopping tracking", e)
            }
        }
    }

    /**
     * Pause tracking (keep session active but pause data collection)
     */
    fun pauseTracking() {
        if (_isTracking.value != true || _isPaused.value == true) {
            Log.w(TAG, "Cannot pause - not tracking or already paused")
            return
        }

        _isPaused.value = true
        _statusMessage.value = "Sleep tracking paused"
        updateTrackingState()
        Log.d(TAG, "Tracking paused")
    }

    /**
     * Resume tracking from paused state
     */
    fun resumeTracking() {
        if (_isTracking.value != true || _isPaused.value != true) {
            Log.w(TAG, "Cannot resume - not tracking or not paused")
            return
        }

        _isPaused.value = false
        _statusMessage.value = "Sleep tracking resumed"
        updateTrackingState()
        Log.d(TAG, "Tracking resumed")
    }

    /**
     * Emergency stop - immediately stop tracking without full completion
     */
    fun emergencyStopTracking() {
        _isTracking.value = false
        _isPaused.value = false
        _currentSessionId.value = null
        _trackingStartTime.value = null
        _trackingState.value = TrackingState()
        _statusMessage.value = "Tracking stopped"
        Log.d(TAG, "Emergency stop executed")
    }

    /**
     * Check for existing active session on startup
     */
    private suspend fun checkForActiveSession() {
        try {
            // Get current session flow from repository
            sleepRepository.currentSessionFlow.collect { session ->
                if (session != null) {
                    _currentSessionId.value = session.id
                    _trackingStartTime.value = session.startTime
                    _isTracking.value = true
                    _currentSession.value = session
                    _statusMessage.value = "Resuming active session"
                    updateTrackingState()
                    Log.d(TAG, "Found active session: ${session.id}")
                } else {
                    _isTracking.value = false
                    _currentSessionId.value = null
                    _trackingStartTime.value = null
                    _currentSession.value = null
                    _trackingState.value = TrackingState()
                    _statusMessage.value = "Ready to track your sleep"
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking for active session", e)
        }
    }

    // ========== NAVIGATION METHODS ==========

    /**
     * Navigate to History activity
     */
    fun navigateToHistory() {
        _navigationEvent.value = NavigationDestination.HISTORY
        Log.d(TAG, "Navigation to History requested")
    }

    /**
     * Navigate to Charts activity
     */
    fun navigateToCharts() {
        _navigationEvent.value = NavigationDestination.CHARTS
        Log.d(TAG, "Navigation to Charts requested")
    }

    /**
     * Navigate to Settings activity
     */
    fun navigateToSettings() {
        _navigationEvent.value = NavigationDestination.SETTINGS
        Log.d(TAG, "Navigation to Settings requested")
    }

    /**
     * Navigate to About activity
     */
    fun navigateToAbout() {
        _navigationEvent.value = NavigationDestination.ABOUT
        Log.d(TAG, "Navigation to About requested")
    }

    /**
     * Navigate to Export activity
     */
    fun navigateToExport() {
        _navigationEvent.value = NavigationDestination.EXPORT
        Log.d(TAG, "Navigation to Export requested")
    }

    /**
     * Clear navigation event after handling
     */
    fun clearNavigationEvent() {
        _navigationEvent.value = null
    }

    // ========== CONFIRMATION DIALOGS ==========

    /**
     * Show confirmation dialog for stopping tracking
     */
    fun showStopTrackingConfirmation() {
        _showConfirmDialog.value = "confirm_stop_tracking"
    }

    /**
     * Show confirmation dialog for emergency stop
     */
    fun showEmergencyStopConfirmation() {
        _showConfirmDialog.value = "confirm_emergency_stop"
    }

    /**
     * Show confirmation dialog for pausing tracking
     */
    fun showPauseTrackingConfirmation() {
        _showConfirmDialog.value = "confirm_pause_tracking"
    }

    /**
     * Clear confirmation dialog
     */
    fun clearConfirmDialog() {
        _showConfirmDialog.value = null
    }

    // ========== DATA LOADING METHODS ==========

    /**
     * Load comprehensive sleep statistics from database
     */
    fun loadSleepStatistics(forceRefresh: Boolean = false) {
        val currentTime = System.currentTimeMillis()
        if (!forceRefresh && currentTime - lastStatisticsLoad < STATISTICS_CACHE_DURATION) {
            return // Use cached data
        }

        viewModelScope.launch {
            _isLoadingStatistics.value = true

            try {
                // Load session statistics
                val statsResult = sleepRepository.getSessionStatistics()
                statsResult.onSuccess { stats ->
                    _totalSessions.value = stats.totalSessions
                    _averageSleepDuration.value = stats.averageDuration
                    _averageSleepQuality.value = stats.averageQuality
                    _averageSleepEfficiency.value = stats.averageEfficiency
                    _sessionStatistics.value = stats

                    // Update last sleep info from most recent session
                    updateLastSleepInfo()

                    lastStatisticsLoad = currentTime
                    Log.d(TAG, "Sleep statistics loaded: ${stats.totalSessions} sessions")

                }.onFailure { exception ->
                    setError("Failed to load statistics: ${exception.message}")
                    Log.e(TAG, "Error loading statistics", exception)
                }

            } catch (e: Exception) {
                setError("Error loading statistics: ${e.message}")
                Log.e(TAG, "Exception loading statistics", e)
            } finally {
                _isLoadingStatistics.value = false
            }
        }
    }

    /**
     * Load recent sessions from database
     */
    fun loadRecentSessions(limit: Int = 10) {
        viewModelScope.launch {
            _isLoadingSessions.value = true

            try {
                val sessionsResult = sleepRepository.getRecentSessions(limit)
                sessionsResult.onSuccess { sessions ->
                    _recentSessions.value = sessions
                    updateLastSleepInfo()
                    Log.d(TAG, "Loaded ${sessions.size} recent sessions")

                }.onFailure { exception ->
                    setError("Failed to load sessions: ${exception.message}")
                    Log.e(TAG, "Error loading recent sessions", exception)
                }

            } catch (e: Exception) {
                setError("Error loading sessions: ${e.message}")
                Log.e(TAG, "Exception loading sessions", e)
            } finally {
                _isLoadingSessions.value = false
            }
        }
    }

    /**
     * Load comprehensive sleep analytics
     */
    fun loadSleepAnalytics(forceRefresh: Boolean = false) {
        if (isAnalyticsJobRunning && !forceRefresh) return

        viewModelScope.launch {
            isAnalyticsJobRunning = true
            _isLoadingAnalytics.value = true

            try {
                // Load comprehensive analytics
                val analyticsResult = sleepRepository.getSleepAnalytics(forceRefresh)
                analyticsResult.onSuccess { analytics ->
                    _sleepAnalytics.value = analytics
                    Log.d(TAG, "Sleep analytics loaded: ${analytics.sessions.size} sessions analyzed")
                }

                // Load trend data
                val trendsResult = sleepRepository.getSleepTrends(30, forceRefresh)
                trendsResult.onSuccess { trends ->
                    _sleepTrends.value = trends
                    Log.d(TAG, "Sleep trends loaded: ${trends.size} data points")
                }

                // Generate quality report
                generateQualityReport()

                // Generate comparative analysis
                generateComparativeAnalysis()

            } catch (e: Exception) {
                setError("Error loading analytics: ${e.message}")
                Log.e(TAG, "Exception loading analytics", e)
            } finally {
                _isLoadingAnalytics.value = false
                isAnalyticsJobRunning = false
            }
        }
    }

    /**
     * Load unacknowledged insights
     */
    private fun loadUnacknowledgedInsights() {
        viewModelScope.launch {
            try {
                val insightsResult = sleepRepository.getUnacknowledgedInsights()
                insightsResult.onSuccess { insights ->
                    _insights.value = insights
                    Log.d(TAG, "Loaded ${insights.size} unacknowledged insights")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading insights", e)
            }
        }
    }

    // ========== REAL-TIME UPDATES ==========

    /**
     * Update real-time sensor status from service broadcasts
     */
    fun updateSensorStatus(status: SensorStatus) {
        _sensorStatus.value = status
        _isServiceConnected.value = status.isFullyActive

        // Update data integrity status
        updateDataIntegrity(status)

        // Update enhanced metrics
        updateEnhancedMetrics(status)

        // Log significant status changes
        if (status.isFullyActive) {
            clearError()
        } else if (!status.isAccelerometerActive || !status.isMicrophoneActive) {
            setError("Sensor connectivity issue: ${status.getOverallStatus()}")
        }
    }

    /**
     * Update live sleep metrics from service
     */
    fun updateLiveMetrics(metrics: LiveSleepMetrics) {
        _liveMetrics.value = metrics

        // Update enhanced metrics
        updateEnhancedMetricsFromLive(metrics)

        // Update current session duration
        if (_isTracking.value == true) {
            updateTrackingState()
        }

        Log.d(TAG, "Live metrics updated: phase=${metrics.currentPhase.getDisplayName()}, " +
                "efficiency=${metrics.sleepEfficiency}%")
    }

    /**
     * Update enhanced metrics from sensor status
     */
    private fun updateEnhancedMetrics(status: SensorStatus) {
        val current = _enhancedLiveMetrics.value ?: EnhancedLiveMetrics()

        val updated = current.copy(
            movementIntensity = status.currentMovementIntensity,
            movementLevel = getMovementLevel(status.currentMovementIntensity),
            movementCount = status.totalMovementEvents,
            noiseLevel = status.currentNoiseLevel,
            noiseDescription = getNoiseLevel(status.currentNoiseLevel),
            noiseCount = status.totalNoiseEvents,
            lastUpdated = System.currentTimeMillis()
        )

        _enhancedLiveMetrics.value = updated
    }

    /**
     * Update enhanced metrics from live metrics
     */
    private fun updateEnhancedMetricsFromLive(metrics: LiveSleepMetrics) {
        val current = _enhancedLiveMetrics.value ?: EnhancedLiveMetrics()

        val updated = current.copy(
            restlessness = metrics.totalRestlessness,
            restlessnessLevel = getRestlessnessLevel(metrics.totalRestlessness),
            lastUpdated = System.currentTimeMillis()
        )

        _enhancedLiveMetrics.value = updated
    }

    /**
     * Get movement level description from intensity
     */
    private fun getMovementLevel(intensity: Float): String {
        return when {
            intensity <= 1.0f -> "Very Low"
            intensity <= 2.0f -> "Low"
            intensity <= 4.0f -> "Medium"
            intensity <= 6.0f -> "High"
            else -> "Very High"
        }
    }

    /**
     * Get noise level description from decibel level
     */
    private fun getNoiseLevel(decibelLevel: Float): String {
        return when {
            decibelLevel <= 30f -> "Very Quiet"
            decibelLevel <= 40f -> "Quiet"
            decibelLevel <= 50f -> "Moderate"
            decibelLevel <= 60f -> "Loud"
            else -> "Very Loud"
        }
    }

    /**
     * Get restlessness level description from score
     */
    private fun getRestlessnessLevel(restlessness: Float): String {
        return when {
            restlessness <= 2f -> "Very Calm"
            restlessness <= 4f -> "Calm"
            restlessness <= 6f -> "Moderate"
            restlessness <= 8f -> "Restless"
            else -> "Very Restless"
        }
    }

    /**
     * Update performance metrics
     */
    fun updatePerformanceMetrics(serviceMetrics: Map<String, Any>) {
        val performanceMetrics = PerformanceMetrics(
            memoryUsage = serviceMetrics["memory_usage_percent"] as? Float ?: 0f,
            dataQueueSize = serviceMetrics["data_queue_size"] as? Int ?: 0,
            averageSaveTime = serviceMetrics["average_save_time"] as? Long ?: 0L,
            sensorHealth = serviceMetrics["sensor_health"] as? Map<String, Any> ?: emptyMap()
        )
        _performanceMetrics.value = performanceMetrics
    }

    // ========== FORMATTING METHODS ==========

    /**
     * Format duration for display
     */
    fun formatDuration(duration: Long): String {
        val hours = duration / (1000 * 60 * 60)
        val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)
        return String.format("%dh %02dm", hours, minutes)
    }

    /**
     * Format quality score for display
     */
    fun formatQualityScore(quality: Float): String {
        return String.format("%.1f/10", quality)
    }

    /**
     * Format efficiency for display
     */
    fun formatEfficiency(efficiency: Float): String {
        return String.format("%.1f%%", efficiency)
    }

    /**
     * Format time ago for display
     */
    fun formatTimeAgo(timestamp: Long): String {
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        val minutes = diff / (1000 * 60)
        val hours = minutes / 60
        val days = hours / 24

        return when {
            minutes < 5 -> "Just now"
            minutes < 60 -> "${minutes}m ago"
            hours < 24 -> "${hours}h ago"
            days < 7 -> "${days}d ago"
            else -> "${days / 7}w ago"
        }
    }

    /**
     * Get formatted last session info
     */
    fun getFormattedLastSessionInfo(): String? {
        val lastSession = _recentSessions.value?.firstOrNull() ?: return null
        val duration = formatDuration(lastSession.duration)
        val quality = lastSession.sleepQualityScore?.let { formatQualityScore(it) } ?: "N/A"
        val timeAgo = lastSession.endTime?.let { formatTimeAgo(it) } ?: ""

        return "Last session: $duration • Quality: $quality • $timeAgo"
    }

    // ========== EXISTING HELPER METHODS (keeping for compatibility) ==========

    /**
     * Generate comprehensive quality report
     */
    private suspend fun generateQualityReport() {
        try {
            val recentSessions = _recentSessions.value ?: return
            if (recentSessions.isEmpty()) return

            val lastSession = recentSessions.firstOrNull() ?: return

            // Use SleepAnalyzer to generate quality analysis
            val qualityAnalysis = sleepAnalyzer.calculateQualityScore(lastSession)
            val efficiencyAnalysis = sleepAnalyzer.calculateSleepEfficiency(lastSession)

            // Create quality report (simplified version)
            val qualityReport = SleepQualityReport(
                reportType = ReportType.WEEKLY,
                timeRange = TimeRange(
                    startDate = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000L),
                    endDate = System.currentTimeMillis(),
                    description = "Last 7 days"
                ),
                overallQualityScore = qualityAnalysis.overallScore,
                qualityGrade = QualityGrade.fromScore(qualityAnalysis.overallScore),
                qualityFactors = QualityFactorAnalysis(
                    movementFactor = createQualityFactor("Movement", qualityAnalysis.movementScore),
                    noiseFactor = createQualityFactor("Noise", qualityAnalysis.noiseScore),
                    durationFactor = createQualityFactor("Duration", qualityAnalysis.durationScore),
                    consistencyFactor = createQualityFactor("Consistency", qualityAnalysis.consistencyScore),
                    efficiencyFactor = createQualityFactor("Efficiency", qualityAnalysis.efficiencyScore),
                    timingFactor = createQualityFactor("Timing", 7.0f),
                    phaseBalanceFactor = createQualityFactor("Phase Balance", qualityAnalysis.phaseDistributionScore),
                    weightedOverallScore = qualityAnalysis.overallScore,
                    factorWeights = mapOf(
                        "Movement" to 0.2f,
                        "Noise" to 0.15f,
                        "Duration" to 0.25f,
                        "Consistency" to 0.15f,
                        "Efficiency" to 0.25f
                    )
                ),
                durationAnalysis = DurationAnalysis(qualityAnalysis.durationScore),
                efficiencyAnalysis = EfficiencyAnalysis(efficiencyAnalysis.combinedEfficiency),
                movementAnalysis = MovementAnalysis(qualityAnalysis.movementScore),
                noiseAnalysis = NoiseAnalysis(qualityAnalysis.noiseScore),
                consistencyAnalysis = ConsistencyAnalysis(qualityAnalysis.consistencyScore),
                personalComparison = PersonalComparisonMetrics(0f),
                keyInsights = qualityAnalysis.improvementAreas.map { area ->
                    QualityInsight(
                        insightId = "insight_${System.currentTimeMillis()}",
                        category = InsightCategory.QUALITY,
                        type = InsightType.INFORMATIONAL,
                        priority = Priority.MEDIUM,
                        confidence = 0.8f,
                        title = "Improvement Area",
                        description = area,
                        evidence = emptyList(),
                        dataPoints = emptyList(),
                        relatedMetrics = emptyList()
                    )
                },
                recommendations = generateQualityRecommendations(qualityAnalysis),
                strengthAreas = listOf(
                    StrengthArea(qualityAnalysis.strongestFactor, qualityAnalysis.overallScore)
                ),
                improvementOpportunities = qualityAnalysis.improvementAreas.map { area ->
                    ImprovementOpportunity(area, Priority.MEDIUM, 5f)
                },
                dataQuality = DataQualityMetrics(1f, 1f, 1f),
                confidenceLevel = ConfidenceLevel.HIGH
            )

            _qualityReport.value = qualityReport

        } catch (e: Exception) {
            Log.e(TAG, "Error generating quality report", e)
        }
    }

    /**
     * Generate comparative analysis
     */
    private suspend fun generateComparativeAnalysis() {
        try {
            val recentSessions = _recentSessions.value ?: return
            val stats = _sessionStatistics.value ?: return

            if (recentSessions.isEmpty()) return

            val currentSession = recentSessions.firstOrNull() ?: return
            val historicalAverage = _averageSleepQuality.value ?: 0f

            val comparison = ComparativeAnalysisResult(
                comparisonType = ComparisonType.PERSONAL_HISTORICAL,
                baselineInfo = BaselineInfo(
                    description = "Personal historical average",
                    date = System.currentTimeMillis()
                ),
                personalComparison = PersonalPerformanceComparison(
                    timeframe = "Last 30 days",
                    currentPeriodMetrics = PeriodMetrics(
                        averageQuality = currentSession.sleepQualityScore ?: 0f,
                        averageDuration = currentSession.duration.toFloat(),
                        averageEfficiency = currentSession.sleepEfficiency,
                        consistency = 7f,
                        sessionCount = 1
                    ),
                    baselinePeriodMetrics = PeriodMetrics(
                        averageQuality = historicalAverage,
                        averageDuration = stats.averageDuration.toFloat(),
                        averageEfficiency = stats.averageEfficiency,
                        consistency = 6f,
                        sessionCount = stats.totalSessions
                    ),
                    qualityImprovement = (currentSession.sleepQualityScore ?: 0f) - historicalAverage,
                    durationImprovement = 0f,
                    efficiencyImprovement = currentSession.sleepEfficiency - stats.averageEfficiency,
                    consistencyImprovement = 0f,
                    overallImprovement = ((currentSession.sleepQualityScore ?: 0f) - historicalAverage) * 10f,
                    qualityPercentile = 75f,
                    durationPercentile = 60f,
                    efficiencyPercentile = 80f,
                    currentStreaks = mapOf("quality" to 3),
                    bestStreaks = mapOf("quality" to 7),
                    streakAnalysis = StreakAnalysis(7, 5, 3, 2, TrendDirection.IMPROVING)
                ),
                temporalComparison = TemporalPerformanceComparison(5f),
                metricComparisons = emptyList(),
                rankingAnalysis = RankingAnalysis(1, 10),
                percentileAnalysis = PercentileAnalysis(mapOf("overall" to 75f)),
                performanceGaps = emptyList(),
                competitiveAdvantages = emptyList(),
                improvementOpportunities = emptyList(),
                comparisonContext = ComparisonContext("Personal historical data"),
                reliabilityMetrics = ComparisonReliabilityMetrics(0.8f)
            )

            _performanceComparison.value = comparison

        } catch (e: Exception) {
            Log.e(TAG, "Error generating comparative analysis", e)
        }
    }

    /**
     * Update last sleep info from recent sessions
     */
    private fun updateLastSleepInfo() {
        val recentSessions = _recentSessions.value
        val lastSession = recentSessions?.firstOrNull()

        lastSession?.let { session ->
            _lastSleepInfo.value = SleepInfo(
                duration = session.duration,
                quality = session.sleepQualityScore ?: 0f,
                endTime = session.endTime ?: System.currentTimeMillis()
            )
        }
    }

    /**
     * Create quality factor for report
     */
    private fun createQualityFactor(name: String, score: Float): QualityFactor {
        return QualityFactor(
            name = name,
            score = score,
            grade = QualityGrade.fromScore(score),
            rawValue = score,
            benchmarkValue = 7f,
            trend = TrendDirection.STABLE,
            impact = FactorImpact.MEDIUM,
            confidence = 0.8f,
            insights = emptyList(),
            recommendations = emptyList()
        )
    }

    /**
     * Generate quality recommendations
     */
    private fun generateQualityRecommendations(analysis: SessionAnalytics.QualityScoreAnalysis): List<QualityRecommendation> {
        return analysis.improvementAreas.map { area ->
            QualityRecommendation(
                recommendationId = "rec_${System.currentTimeMillis()}",
                category = RecommendationCategory.SLEEP_HYGIENE,
                priority = Priority.MEDIUM,
                title = "Improve $area",
                description = "Focus on improving your $area for better sleep quality",
                actionItems = listOf(
                    ActionItem(
                        action = "Monitor and adjust $area",
                        difficulty = ImplementationDifficulty.LOW,
                        timeRequired = "1-2 weeks",
                        expectedOutcome = "Better sleep quality"
                    )
                ),
                expectedImpact = ExpectedImpact.MEDIUM,
                implementationDifficulty = ImplementationDifficulty.LOW,
                timeToImpact = TimeToImpact.SHORT_TERM,
                confidence = 0.7f,
                relatedInsights = emptyList(),
                successMetrics = listOf("Quality score improvement")
            )
        }
    }

    /**
     * Update data integrity status
     */
    private fun updateDataIntegrity(status: SensorStatus) {
        val integrity = DataIntegrityStatus(
            isHealthy = status.isFullyActive,
            lastDataSync = System.currentTimeMillis(),
            pendingEvents = 0, // Would be provided by service
            successRate = 98.5f
        )
        _dataIntegrity.value = integrity
    }

    // ========== PUBLIC API METHODS ==========

    /**
     * Get current tracking state
     */
    fun getCurrentTrackingState(): Boolean = _isTracking.value ?: false

    /**
     * Get current sensor status
     */
    fun getCurrentSensorStatus(): SensorStatus? = _sensorStatus.value

    /**
     * Get current live metrics
     */
    fun getCurrentLiveMetrics(): LiveSleepMetrics? = _liveMetrics.value

    /**
     * Check if sensors are healthy
     */
    fun areSensorsHealthy(): Boolean = _sensorStatus.value?.isFullyActive ?: false

    /**
     * Get session duration if currently tracking
     */
    fun getCurrentSessionDuration(): Long {
        return _trackingStartTime.value?.let { startTime ->
            System.currentTimeMillis() - startTime
        } ?: 0L
    }

    /**
     * Acknowledge insight with AI feedback
     */
    fun acknowledgeInsight(insightId: Long) {
        viewModelScope.launch {
            try {
                sleepRepository.acknowledgeInsight(insightId)

                // Record interaction for AI learning
                sleepRepository.recordUserInteraction(
                    insightId = insightId,
                    interactionType = "ACKNOWLEDGED",
                    details = "User acknowledged insight"
                )

                loadUnacknowledgedInsights() // Refresh insights
                updateAIInsightsSummary()

            } catch (e: Exception) {
                Log.e(TAG, "Error acknowledging insight", e)
            }
        }
    }

    /**
     * Refresh all data including AI insights
     */
    fun refreshAllData() {
        viewModelScope.launch {
            _statusMessage.value = "Refreshing data..."
            loadSleepStatistics(forceRefresh = true)
            loadRecentSessions()
            loadSleepAnalytics(forceRefresh = true)
            loadAIInsights()
            loadUnacknowledgedInsights()
            updateAIInsightsSummary()
            _statusMessage.value = null
        }
    }

    /**
     * Set error message
     */
    fun setError(message: String) {
        _errorMessage.value = message
        Log.w(TAG, "Error set: $message")
    }

    /**
     * Clear error message
     */
    fun clearError() {
        _errorMessage.value = null
    }

    /**
     * Clear status message
     */
    fun clearStatusMessage() {
        _statusMessage.value = null
    }

    /**
     * Get formatted statistics for UI display including AI insights
     */
    fun getFormattedStatistics(): Map<String, String> {
        val avgDuration = _averageSleepDuration.value ?: 0L
        val hours = avgDuration / (1000 * 60 * 60)
        val minutes = (avgDuration % (1000 * 60 * 60)) / (1000 * 60)

        return mapOf(
            "total_sessions" to (_totalSessions.value?.toString() ?: "0"),
            "average_duration" to String.format("%dh %02dm", hours, minutes),
            "average_quality" to String.format("%.1f/10", _averageSleepQuality.value ?: 0f),
            "average_efficiency" to String.format("%.1f%%", _averageSleepEfficiency.value ?: 0f),
            "sensor_status" to (_sensorStatus.value?.getOverallStatus() ?: "Unknown"),
            "is_tracking" to (_isTracking.value?.toString() ?: "false"),
            "insights_count" to (_insights.value?.size?.toString() ?: "0"),
            "ai_engine_status" to (_aiEngineStatus.value?.toString() ?: "Unknown"),
            "ai_insights_total" to (_aiInsightsSummary.value?.totalInsights?.toString() ?: "0"),
            "ai_insights_new" to (_aiInsightsSummary.value?.newInsights?.toString() ?: "0")
        )
    }

    override fun onCleared() {
        super.onCleared()
        realTimeUpdateJob?.cancel()
        aiInsightsUpdateJob?.cancel()

        try {
            sleepRepository.cleanup()
            if (this::aiInsightsEngine.isInitialized) {
                aiInsightsEngine.cleanup()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up repository and AI engine", e)
        }

        Log.d(TAG, "ViewModel cleared")
    }
}

// ========== SUPPORTING DATA CLASSES ==========

/**
 * Data integrity status for monitoring
 */
data class DataIntegrityStatus(
    val isHealthy: Boolean,
    val lastDataSync: Long,
    val pendingEvents: Int,
    val successRate: Float
)

/**
 * Performance metrics for monitoring
 */
data class PerformanceMetrics(
    val memoryUsage: Float,
    val dataQueueSize: Int,
    val averageSaveTime: Long,
    val sensorHealth: Map<String, Any>
)

// ========== PLACEHOLDER CLASSES (to be properly implemented) ==========

// These would be properly implemented based on your AI architecture
private class InsightsPromptBuilder
private data class TrendAnalysisResult(val analysis: com.example.somniai.analytics.TrendAnalysis)