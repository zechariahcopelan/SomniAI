package com.example.somniai.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.somniai.ai.*
import com.example.somniai.data.*
import com.example.somniai.repository.SleepRepository
import com.example.somniai.analytics.SleepAnalyzer
import com.example.somniai.database.SleepDatabase
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*

/**
 * Comprehensive ViewModel for AI-powered sleep insights and recommendations
 *
 * Features:
 * - Integration with sophisticated AI insights engine
 * - Real-time insight generation and management
 * - Advanced filtering, sorting, and categorization
 * - User engagement tracking and personalization
 * - Performance monitoring and optimization
 * - Comprehensive error handling and recovery
 * - Background insight generation and scheduling
 * - Rich analytics and user feedback integration
 * - Multi-format insight display support
 * - Intelligent caching and state management
 */
class InsightsViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "InsightsViewModel"

        // Update intervals
        private const val INSIGHTS_REFRESH_INTERVAL_MINUTES = 30L
        private const val PERFORMANCE_MONITORING_INTERVAL_MINUTES = 5L
        private const val AUTO_GENERATION_CHECK_INTERVAL_HOURS = 2L

        // Configuration constants
        private const val MAX_INSIGHTS_PER_CATEGORY = 20
        private const val INSIGHTS_CACHE_SIZE = 100
        private const val MIN_INSIGHT_QUALITY_SCORE = 0.6f
        private const val MAX_CONCURRENT_GENERATIONS = 3

        // User engagement thresholds
        private const val HIGH_ENGAGEMENT_THRESHOLD = 0.8f
        private const val INSIGHT_STALENESS_HOURS = 24L
        private const val PERSONALIZATION_LEARNING_RATE = 0.1f
    }

    // Core dependencies
    private lateinit var sleepRepository: SleepRepository
    private lateinit var aiInsightsEngine: AIInsightsEngine
    private lateinit var insightsScheduler: InsightsScheduler
    private lateinit var sleepAnalyzer: SleepAnalyzer

    // State management
    private val _isInitialized = MutableLiveData<Boolean>()
    val isInitialized: LiveData<Boolean> = _isInitialized

    private val _insightsState = MutableStateFlow<InsightsState>(InsightsState.Loading)
    val insightsState: StateFlow<InsightsState> = _insightsState.asStateFlow()

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    private val _isRefreshing = MutableLiveData<Boolean>()
    val isRefreshing: LiveData<Boolean> = _isRefreshing

    private val _isGeneratingInsights = MutableLiveData<Boolean>()
    val isGeneratingInsights: LiveData<Boolean> = _isGeneratingInsights

    // Insights data
    private val _allInsights = MutableLiveData<List<SleepInsight>>()
    val allInsights: LiveData<List<SleepInsight>> = _allInsights

    private val _filteredInsights = MutableLiveData<List<SleepInsight>>()
    val filteredInsights: LiveData<List<SleepInsight>> = _filteredInsights

    private val _featuredInsights = MutableLiveData<List<SleepInsight>>()
    val featuredInsights: LiveData<List<SleepInsight>> = _featuredInsights

    private val _unacknowledgedInsights = MutableLiveData<List<SleepInsight>>()
    val unacknowledgedInsights: LiveData<List<SleepInsight>> = _unacknowledgedInsights

    private val _insightsByCategory = MutableLiveData<Map<InsightCategory, List<SleepInsight>>>()
    val insightsByCategory: LiveData<Map<InsightCategory, List<SleepInsight>>> = _insightsByCategory

    private val _recentInsights = MutableLiveData<List<SleepInsight>>()
    val recentInsights: LiveData<List<SleepInsight>> = _recentInsights

    // AI Integration data
    private val _aiGenerationStatus = MutableLiveData<AIGenerationStatus>()
    val aiGenerationStatus: LiveData<AIGenerationStatus> = _aiGenerationStatus

    private val _aiInsightsHistory = MutableLiveData<List<AIInsightGeneration>>()
    val aiInsightsHistory: LiveData<List<AIInsightGeneration>> = _aiInsightsHistory

    private val _insightGenerationProgress = MutableLiveData<InsightGenerationProgress>()
    val insightGenerationProgress: LiveData<InsightGenerationProgress> = _insightGenerationProgress

    // Analytics and performance
    private val _insightsAnalytics = MutableLiveData<InsightsAnalytics>()
    val insightsAnalytics: LiveData<InsightsAnalytics> = _insightsAnalytics

    private val _userEngagementMetrics = MutableLiveData<UserEngagementMetrics>()
    val userEngagementMetrics: LiveData<UserEngagementMetrics> = _userEngagementMetrics

    private val _performanceMetrics = MutableLiveData<InsightsPerformanceMetrics>()
    val performanceMetrics: LiveData<InsightsPerformanceMetrics> = _performanceMetrics

    // Filtering and sorting
    private val _currentFilter = MutableLiveData<InsightsFilter>()
    val currentFilter: LiveData<InsightsFilter> = _currentFilter

    private val _currentSortOption = MutableLiveData<InsightsSortOption>()
    val currentSortOption: LiveData<InsightsSortOption> = _currentSortOption

    private val _searchQuery = MutableLiveData<String>()
    val searchQuery: LiveData<String> = _searchQuery

    private val _selectedCategories = MutableLiveData<Set<InsightCategory>>()
    val selectedCategories: LiveData<Set<InsightCategory>> = _selectedCategories

    // UI state
    private val _displayMode = MutableLiveData<InsightsDisplayMode>()
    val displayMode: LiveData<InsightsDisplayMode> = _displayMode

    private val _expandedInsightIds = MutableLiveData<Set<Long>>()
    val expandedInsightIds: LiveData<Set<Long>> = _expandedInsightIds

    private val _selectedInsightId = MutableLiveData<Long?>()
    val selectedInsightId: LiveData<Long?> = _selectedInsightId

    // Error handling
    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    private val _statusMessage = MutableLiveData<String?>()
    val statusMessage: LiveData<String?> = _statusMessage

    // User preferences and personalization
    private val _userPreferences = MutableLiveData<InsightsUserPreferences>()
    val userPreferences: LiveData<InsightsUserPreferences> = _userPreferences

    private val _personalizationData = MutableLiveData<PersonalizationData>()
    val personalizationData: LiveData<PersonalizationData> = _personalizationData

    // Background jobs and state
    private val isBackgroundJobsRunning = AtomicBoolean(false)
    private val activeGenerationJobs = ConcurrentHashMap<String, Job>()
    private val totalInsightsGenerated = AtomicLong(0L)
    private val totalUserInteractions = AtomicLong(0L)

    // Background monitoring
    private var refreshJob: Job? = null
    private var monitoringJob: Job? = null
    private var autoGenerationJob: Job? = null

    init {
        // Initialize with default values
        _isInitialized.value = false
        _isLoading.value = false
        _isRefreshing.value = false
        _isGeneratingInsights.value = false
        _currentFilter.value = InsightsFilter.DEFAULT
        _currentSortOption.value = InsightsSortOption.NEWEST_FIRST
        _searchQuery.value = ""
        _selectedCategories.value = emptySet()
        _displayMode.value = InsightsDisplayMode.CARDS
        _expandedInsightIds.value = emptySet()
        _aiGenerationStatus.value = AIGenerationStatus.IDLE

        // Initialize components
        initializeComponents()
    }

    // ========== INITIALIZATION ==========

    /**
     * Initialize ViewModel components and dependencies
     */
    private fun initializeComponents() {
        viewModelScope.launch {
            try {
                _statusMessage.value = "Initializing insights system..."

                val database = SleepDatabase.getDatabase(getApplication())
                sleepRepository = SleepRepository(database, getApplication())
                aiInsightsEngine = AIInsightsEngine(getApplication(), sleepRepository)
                insightsScheduler = InsightsScheduler(getApplication(), aiInsightsEngine)
                sleepAnalyzer = SleepAnalyzer()

                // Initialize AI engine
                aiInsightsEngine.initialize().getOrThrow()

                // Load user preferences
                loadUserPreferences()

                // Load existing insights
                loadInitialInsights()

                // Start background monitoring
                startBackgroundJobs()

                _isInitialized.value = true
                _statusMessage.value = null

                Log.d(TAG, "InsightsViewModel initialized successfully")

            } catch (e: Exception) {
                setError("Failed to initialize insights system: ${e.message}")
                Log.e(TAG, "Failed to initialize InsightsViewModel", e)
            }
        }
    }

    /**
     * Load initial insights from database and trigger refresh
     */
    private suspend fun loadInitialInsights() {
        _isLoading.value = true

        try {
            // Load existing insights from database
            val existingInsights = sleepRepository.getAllInsights().getOrElse { emptyList() }
            _allInsights.value = existingInsights

            // Apply current filters
            applyFiltersAndSorting()

            // Update analytics
            updateInsightsAnalytics()

            // Check if we need to generate new insights
            if (existingInsights.isEmpty() || shouldGenerateNewInsights(existingInsights)) {
                generateInitialInsights()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load initial insights", e)
            setError("Failed to load insights: ${e.message}")
        } finally {
            _isLoading.value = false
        }
    }

    /**
     * Start background monitoring and maintenance jobs
     */
    private fun startBackgroundJobs() {
        if (isBackgroundJobsRunning.getAndSet(true)) {
            return // Already running
        }

        // Periodic insights refresh
        refreshJob = viewModelScope.launch {
            while (isActive) {
                delay(TimeUnit.MINUTES.toMillis(INSIGHTS_REFRESH_INTERVAL_MINUTES))
                try {
                    refreshInsights(silent = true)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in background refresh", e)
                }
            }
        }

        // Performance monitoring
        monitoringJob = viewModelScope.launch {
            while (isActive) {
                delay(TimeUnit.MINUTES.toMillis(PERFORMANCE_MONITORING_INTERVAL_MINUTES))
                try {
                    updatePerformanceMetrics()
                } catch (e: Exception) {
                    Log.e(TAG, "Error in performance monitoring", e)
                }
            }
        }

        // Auto-generation check
        autoGenerationJob = viewModelScope.launch {
            while (isActive) {
                delay(TimeUnit.HOURS.toMillis(AUTO_GENERATION_CHECK_INTERVAL_HOURS))
                try {
                    checkAndGenerateInsights()
                } catch (e: Exception) {
                    Log.e(TAG, "Error in auto-generation check", e)
                }
            }
        }
    }

    // ========== INSIGHT GENERATION ==========

    /**
     * Generate new insights using AI engine
     */
    fun generateInsights(
        type: InsightGenerationType = InsightGenerationType.DAILY_ANALYSIS,
        force: Boolean = false
    ) {
        if (_isGeneratingInsights.value == true && !force) {
            Log.w(TAG, "Insight generation already in progress")
            return
        }

        viewModelScope.launch {
            try {
                _isGeneratingInsights.value = true
                _statusMessage.value = "Generating new insights..."

                // Update generation progress
                updateGenerationProgress(0f, "Analyzing sleep data...")

                // Get context for insight generation
                val context = createInsightGenerationContext(type)

                updateGenerationProgress(0.2f, "Preparing AI analysis...")

                // Generate insights using AI engine
                val generationResult = aiInsightsEngine.generateInsights(
                    context = context,
                    options = AIInsightGenerationOptions(
                        enableAdvancedAnalysis = true,
                        includePersonalization = true,
                        maxInsights = 10,
                        qualityThreshold = MIN_INSIGHT_QUALITY_SCORE
                    )
                ).getOrThrow()

                updateGenerationProgress(0.8f, "Processing insights...")

                // Save generated insights
                saveGeneratedInsights(generationResult.insights)

                // Update AI generation history
                recordGenerationSuccess(generationResult)

                updateGenerationProgress(1f, "Insights generated successfully")

                // Refresh UI
                refreshInsights(silent = false)

                totalInsightsGenerated.addAndGet(generationResult.insights.size.toLong())

                _statusMessage.value = "Generated ${generationResult.insights.size} new insights"
                delay(3000)
                _statusMessage.value = null

                Log.d(TAG, "Generated ${generationResult.insights.size} insights successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate insights", e)
                setError("Failed to generate insights: ${e.message}")
                recordGenerationFailure(type, e)
            } finally {
                _isGeneratingInsights.value = false
                _insightGenerationProgress.value = InsightGenerationProgress()
            }
        }
    }

    /**
     * Generate insights for specific session
     */
    fun generateSessionInsights(sessionId: Long) {
        viewModelScope.launch {
            try {
                _statusMessage.value = "Analyzing session..."

                val session = sleepRepository.getSessionById(sessionId).getOrNull()
                if (session == null) {
                    setError("Session not found")
                    return@launch
                }

                val context = InsightGenerationContext(
                    generationType = InsightGenerationType.POST_SESSION,
                    sessionData = session,
                    analysisDepth = AnalysisDepth.COMPREHENSIVE,
                    includeComparisons = true,
                    includeRecommendations = true
                )

                val result = aiInsightsEngine.generateInsights(context).getOrThrow()
                saveGeneratedInsights(result.insights)
                refreshInsights(silent = false)

                _statusMessage.value = "Session analysis complete"
                delay(2000)
                _statusMessage.value = null

            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate session insights", e)
                setError("Failed to analyze session: ${e.message}")
            }
        }
    }

    /**
     * Generate personalized insights based on user patterns
     */
    fun generatePersonalizedInsights() {
        viewModelScope.launch {
            try {
                _statusMessage.value = "Creating personalized insights..."

                val personalData = sleepRepository.getPersonalBaseline().getOrThrow()
                val habitData = sleepRepository.getHabitAnalysis().getOrThrow()

                val context = InsightGenerationContext(
                    generationType = InsightGenerationType.PERSONALIZED_ANALYSIS,
                    personalBaseline = personalData,
                    habitAnalysis = habitData,
                    analysisDepth = AnalysisDepth.COMPREHENSIVE,
                    includeRecommendations = true,
                    customParameters = mapOf(
                        "personalization_level" to "high",
                        "focus_areas" to getUserFocusAreas()
                    )
                )

                val result = aiInsightsEngine.generateInsights(context).getOrThrow()
                saveGeneratedInsights(result.insights)
                refreshInsights(silent = false)

                _statusMessage.value = "Personalized insights ready"
                delay(2000)
                _statusMessage.value = null

            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate personalized insights", e)
                setError("Failed to create personalized insights: ${e.message}")
            }
        }
    }

    /**
     * Generate predictive insights and recommendations
     */
    fun generatePredictiveInsights() {
        viewModelScope.launch {
            try {
                _statusMessage.value = "Generating predictions..."

                val trendData = sleepRepository.getSleepTrends(30, forceRefresh = true).getOrThrow()
                val patternData = sleepRepository.getPatternAnalysis().getOrThrow()

                val context = InsightGenerationContext(
                    generationType = InsightGenerationType.PREDICTIVE_ANALYSIS,
                    trendAnalysis = trendData,
                    patternAnalysis = patternData,
                    predictionHorizon = PredictionHorizon.WEEK,
                    analysisDepth = AnalysisDepth.COMPREHENSIVE,
                    includeRecommendations = true
                )

                val result = aiInsightsEngine.generateInsights(context).getOrThrow()
                saveGeneratedInsights(result.insights)
                refreshInsights(silent = false)

                _statusMessage.value = "Predictions generated"
                delay(2000)
                _statusMessage.value = null

            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate predictive insights", e)
                setError("Failed to generate predictions: ${e.message}")
            }
        }
    }

    // ========== INSIGHT MANAGEMENT ==========

    /**
     * Refresh insights from database and apply filters
     */
    fun refreshInsights(silent: Boolean = false) {
        viewModelScope.launch {
            try {
                if (!silent) {
                    _isRefreshing.value = true
                    _statusMessage.value = "Refreshing insights..."
                }

                // Load from database
                val insights = sleepRepository.getAllInsights().getOrElse { emptyList() }
                _allInsights.value = insights

                // Apply current filters and sorting
                applyFiltersAndSorting()

                // Update categorized insights
                updateCategorizedInsights(insights)

                // Update analytics
                updateInsightsAnalytics()

                // Update user engagement metrics
                updateUserEngagementMetrics()

                if (!silent) {
                    _statusMessage.value = null
                }

                Log.d(TAG, "Refreshed ${insights.size} insights")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh insights", e)
                if (!silent) {
                    setError("Failed to refresh insights: ${e.message}")
                }
            } finally {
                if (!silent) {
                    _isRefreshing.value = false
                }
            }
        }
    }

    /**
     * Acknowledge an insight
     */
    fun acknowledgeInsight(insightId: Long) {
        viewModelScope.launch {
            try {
                sleepRepository.acknowledgeInsight(insightId).getOrThrow()
                recordUserInteraction(insightId, UserInteractionType.ACKNOWLEDGED)
                refreshInsights(silent = true)
                totalUserInteractions.incrementAndGet()

                Log.d(TAG, "Acknowledged insight: $insightId")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to acknowledge insight", e)
                setError("Failed to acknowledge insight: ${e.message}")
            }
        }
    }

    /**
     * Mark insight as helpful
     */
    fun markInsightAsHelpful(insightId: Long, helpful: Boolean) {
        viewModelScope.launch {
            try {
                // Record feedback
                val feedback = InsightFeedback(
                    insightId = insightId,
                    wasHelpful = helpful,
                    timestamp = System.currentTimeMillis(),
                    feedbackType = if (helpful) FeedbackType.POSITIVE else FeedbackType.NEGATIVE
                )

                sleepRepository.recordInsightFeedback(feedback).getOrThrow()

                // Record interaction
                val interactionType = if (helpful) {
                    UserInteractionType.MARKED_HELPFUL
                } else {
                    UserInteractionType.MARKED_UNHELPFUL
                }

                recordUserInteraction(insightId, interactionType)

                // Update AI learning
                aiInsightsEngine.recordFeedback(feedback)

                totalUserInteractions.incrementAndGet()

                Log.d(TAG, "Recorded feedback for insight: $insightId, helpful: $helpful")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to record feedback", e)
                setError("Failed to record feedback: ${e.message}")
            }
        }
    }

    /**
     * Share insight with feedback
     */
    fun shareInsight(insightId: Long, platform: String) {
        viewModelScope.launch {
            try {
                recordUserInteraction(insightId, UserInteractionType.SHARED, platform)
                totalUserInteractions.incrementAndGet()

                Log.d(TAG, "Shared insight: $insightId on $platform")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to record share", e)
            }
        }
    }

    /**
     * Delete insight
     */
    fun deleteInsight(insightId: Long) {
        viewModelScope.launch {
            try {
                sleepRepository.deleteInsight(insightId).getOrThrow()
                recordUserInteraction(insightId, UserInteractionType.DELETED)
                refreshInsights(silent = true)

                _statusMessage.value = "Insight deleted"
                delay(2000)
                _statusMessage.value = null

                Log.d(TAG, "Deleted insight: $insightId")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to delete insight", e)
                setError("Failed to delete insight: ${e.message}")
            }
        }
    }

    // ========== FILTERING AND SORTING ==========

    /**
     * Apply insight filter
     */
    fun applyFilter(filter: InsightsFilter) {
        _currentFilter.value = filter
        applyFiltersAndSorting()
        recordUserInteraction(null, UserInteractionType.FILTERED, filter.name)
    }

    /**
     * Set sort option
     */
    fun setSortOption(sortOption: InsightsSortOption) {
        _currentSortOption.value = sortOption
        applyFiltersAndSorting()
        recordUserInteraction(null, UserInteractionType.SORTED, sortOption.name)
    }

    /**
     * Set search query
     */
    fun setSearchQuery(query: String) {
        _searchQuery.value = query
        applyFiltersAndSorting()
        if (query.isNotBlank()) {
            recordUserInteraction(null, UserInteractionType.SEARCHED, query)
        }
    }

    /**
     * Toggle category selection
     */
    fun toggleCategorySelection(category: InsightCategory) {
        val current = _selectedCategories.value ?: emptySet()
        val updated = if (current.contains(category)) {
            current - category
        } else {
            current + category
        }
        _selectedCategories.value = updated
        applyFiltersAndSorting()
        recordUserInteraction(null, UserInteractionType.CATEGORY_FILTERED, category.name)
    }

    /**
     * Clear all filters
     */
    fun clearFilters() {
        _currentFilter.value = InsightsFilter.DEFAULT
        _searchQuery.value = ""
        _selectedCategories.value = emptySet()
        applyFiltersAndSorting()
        recordUserInteraction(null, UserInteractionType.FILTERS_CLEARED)
    }

    /**
     * Apply current filters and sorting to insights list
     */
    private fun applyFiltersAndSorting() {
        viewModelScope.launch {
            val allInsights = _allInsights.value ?: return@launch
            val filter = _currentFilter.value ?: InsightsFilter.DEFAULT
            val sortOption = _currentSortOption.value ?: InsightsSortOption.NEWEST_FIRST
            val searchQuery = _searchQuery.value ?: ""
            val selectedCategories = _selectedCategories.value ?: emptySet()

            var filteredList = allInsights

            // Apply category filter
            if (selectedCategories.isNotEmpty()) {
                filteredList = filteredList.filter { it.category in selectedCategories }
            }

            // Apply search filter
            if (searchQuery.isNotBlank()) {
                filteredList = filteredList.filter { insight ->
                    insight.title.contains(searchQuery, ignoreCase = true) ||
                            insight.description.contains(searchQuery, ignoreCase = true) ||
                            insight.recommendation.contains(searchQuery, ignoreCase = true)
                }
            }

            // Apply main filter
            filteredList = when (filter) {
                InsightsFilter.ALL -> filteredList
                InsightsFilter.UNACKNOWLEDGED -> filteredList.filter { !it.isAcknowledged }
                InsightsFilter.AI_GENERATED -> filteredList.filter { it.isAiGenerated }
                InsightsFilter.HIGH_PRIORITY -> filteredList.filter { it.priority == 1 }
                InsightsFilter.RECENT -> {
                    val cutoffTime = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(7)
                    filteredList.filter { it.timestamp >= cutoffTime }
                }
                InsightsFilter.ACTIONABLE -> filteredList.filter { it.recommendation.isNotBlank() }
                InsightsFilter.PERSONAL -> filteredList.filter { it.category == InsightCategory.PERSONAL }
                InsightsFilter.DEFAULT -> filteredList.filter { !it.isAcknowledged || it.priority == 1 }
            }

            // Apply sorting
            filteredList = when (sortOption) {
                InsightsSortOption.NEWEST_FIRST -> filteredList.sortedByDescending { it.timestamp }
                InsightsSortOption.OLDEST_FIRST -> filteredList.sortedBy { it.timestamp }
                InsightsSortOption.PRIORITY_HIGH_FIRST -> filteredList.sortedBy { it.priority }
                InsightsSortOption.PRIORITY_LOW_FIRST -> filteredList.sortedByDescending { it.priority }
                InsightsSortOption.CATEGORY -> filteredList.sortedBy { it.category.name }
                InsightsSortOption.RELEVANCE -> {
                    if (searchQuery.isNotBlank()) {
                        filteredList.sortedByDescending { insight ->
                            calculateRelevanceScore(insight, searchQuery)
                        }
                    } else {
                        filteredList.sortedByDescending { it.timestamp }
                    }
                }
            }

            _filteredInsights.value = filteredList

            // Update featured insights
            updateFeaturedInsights(filteredList)

            // Update unacknowledged insights
            _unacknowledgedInsights.value = allInsights.filter { !it.isAcknowledged }

            // Update recent insights
            val recentCutoff = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(3)
            _recentInsights.value = allInsights.filter { it.timestamp >= recentCutoff }
                .sortedByDescending { it.timestamp }
                .take(5)
        }
    }

    // ========== UI STATE MANAGEMENT ==========

    /**
     * Set display mode for insights
     */
    fun setDisplayMode(mode: InsightsDisplayMode) {
        _displayMode.value = mode
        recordUserInteraction(null, UserInteractionType.DISPLAY_MODE_CHANGED, mode.name)
    }

    /**
     * Toggle insight expansion
     */
    fun toggleInsightExpansion(insightId: Long) {
        val current = _expandedInsightIds.value ?: emptySet()
        val updated = if (current.contains(insightId)) {
            current - insightId
        } else {
            current + insightId
        }
        _expandedInsightIds.value = updated

        val interactionType = if (updated.contains(insightId)) {
            UserInteractionType.EXPANDED
        } else {
            UserInteractionType.COLLAPSED
        }
        recordUserInteraction(insightId, interactionType)
    }

    /**
     * Select insight for detailed view
     */
    fun selectInsight(insightId: Long?) {
        _selectedInsightId.value = insightId
        if (insightId != null) {
            recordUserInteraction(insightId, UserInteractionType.SELECTED)
        }
    }

    // ========== HELPER METHODS ==========

    private suspend fun createInsightGenerationContext(type: InsightGenerationType): InsightGenerationContext {
        val recentSession = sleepRepository.getLatestSession().getOrNull()
        val trendData = sleepRepository.getSleepTrends(14).getOrNull()
        val analytics = sleepRepository.getSleepAnalytics().getOrNull()

        return InsightGenerationContext(
            generationType = type,
            sessionData = recentSession,
            trendAnalysis = trendData,
            sleepAnalytics = analytics,
            analysisDepth = AnalysisDepth.DETAILED,
            includeComparisons = true,
            includeRecommendations = true,
            includePersonalization = true
        )
    }

    private suspend fun saveGeneratedInsights(insights: List<SleepInsight>) {
        for (insight in insights) {
            try {
                sleepRepository.saveInsight(insight).getOrThrow()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to save insight: ${insight.title}", e)
            }
        }
    }

    private fun shouldGenerateNewInsights(existingInsights: List<SleepInsight>): Boolean {
        if (existingInsights.isEmpty()) return true

        val lastInsightTime = existingInsights.maxOfOrNull { it.timestamp } ?: 0L
        val hoursSinceLastInsight = (System.currentTimeMillis() - lastInsightTime) / (1000 * 60 * 60)

        return hoursSinceLastInsight >= INSIGHT_STALENESS_HOURS
    }

    private suspend fun generateInitialInsights() {
        try {
            _statusMessage.value = "Generating initial insights..."

            val context = createInsightGenerationContext(InsightGenerationType.DAILY_ANALYSIS)
            val result = aiInsightsEngine.generateInsights(context).getOrThrow()

            saveGeneratedInsights(result.insights)
            refreshInsights(silent = true)

            _statusMessage.value = null

        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate initial insights", e)
        }
    }

    private suspend fun checkAndGenerateInsights() {
        val existingInsights = sleepRepository.getAllInsights().getOrElse { emptyList() }

        if (shouldGenerateNewInsights(existingInsights)) {
            generateInsights(InsightGenerationType.DAILY_ANALYSIS)
        }
    }

    private fun updateCategorizedInsights(insights: List<SleepInsight>) {
        val categorized = insights.groupBy { it.category }
        _insightsByCategory.value = categorized
    }

    private fun updateFeaturedInsights(insights: List<SleepInsight>) {
        // Select featured insights based on priority, recency, and user engagement
        val featured = insights.filter { !it.isAcknowledged && it.priority <= 2 }
            .sortedWith(compareBy<SleepInsight> { it.priority }.thenByDescending { it.timestamp })
            .take(3)

        _featuredInsights.value = featured
    }

    private fun calculateRelevanceScore(insight: SleepInsight, query: String): Float {
        var score = 0f

        if (insight.title.contains(query, ignoreCase = true)) score += 3f
        if (insight.description.contains(query, ignoreCase = true)) score += 2f
        if (insight.recommendation.contains(query, ignoreCase = true)) score += 1f
        if (insight.category.name.contains(query, ignoreCase = true)) score += 1f

        return score
    }

    private suspend fun updateInsightsAnalytics() {
        val allInsights = _allInsights.value ?: return

        val analytics = InsightsAnalytics(
            totalInsights = allInsights.size,
            acknowledgedInsights = allInsights.count { it.isAcknowledged },
            aiGeneratedInsights = allInsights.count { it.isAiGenerated },
            highPriorityInsights = allInsights.count { it.priority == 1 },
            categoryDistribution = allInsights.groupBy { it.category }.mapValues { it.value.size },
            averageAge = calculateAverageInsightAge(allInsights),
            engagementRate = calculateEngagementRate(allInsights),
            generationTrend = calculateGenerationTrend(allInsights)
        )

        _insightsAnalytics.value = analytics
    }

    private fun updateUserEngagementMetrics() {
        val totalInteractions = totalUserInteractions.get()
        val totalInsights = _allInsights.value?.size ?: 1

        val engagementRate = totalInteractions.toFloat() / totalInsights.coerceAtLeast(1)

        val metrics = UserEngagementMetrics(
            totalInteractions = totalInteractions,
            engagementRate = engagementRate,
            isHighEngagement = engagementRate >= HIGH_ENGAGEMENT_THRESHOLD,
            lastInteractionTime = System.currentTimeMillis(),
            preferredCategories = calculatePreferredCategories(),
            averageTimeToAcknowledge = calculateAverageTimeToAcknowledge()
        )

        _userEngagementMetrics.value = metrics
    }

    private suspend fun updatePerformanceMetrics() {
        val generationJobs = activeGenerationJobs.size
        val totalGenerated = totalInsightsGenerated.get()
        val cacheHitRate = aiInsightsEngine.getCacheHitRate()

        val metrics = InsightsPerformanceMetrics(
            activeGenerationJobs = generationJobs,
            totalInsightsGenerated = totalGenerated,
            averageGenerationTime = aiInsightsEngine.getAverageGenerationTime(),
            cacheHitRate = cacheHitRate,
            errorRate = aiInsightsEngine.getErrorRate(),
            systemLoad = calculateSystemLoad()
        )

        _performanceMetrics.value = metrics
    }

    private fun updateGenerationProgress(progress: Float, message: String) {
        _insightGenerationProgress.value = InsightGenerationProgress(
            progress = progress,
            message = message,
            isActive = progress < 1f
        )
    }

    private fun recordGenerationSuccess(result: AIInsightGenerationResult) {
        val generation = AIInsightGeneration(
            timestamp = System.currentTimeMillis(),
            type = result.context.generationType,
            insightsGenerated = result.insights.size,
            success = true,
            processingTime = result.processingTime,
            qualityScore = result.qualityScore
        )

        val currentHistory = _aiInsightsHistory.value ?: emptyList()
        _aiInsightsHistory.value = (listOf(generation) + currentHistory).take(50)

        _aiGenerationStatus.value = AIGenerationStatus.SUCCESS
    }

    private fun recordGenerationFailure(type: InsightGenerationType, error: Exception) {
        val generation = AIInsightGeneration(
            timestamp = System.currentTimeMillis(),
            type = type,
            insightsGenerated = 0,
            success = false,
            error = error.message
        )

        val currentHistory = _aiInsightsHistory.value ?: emptyList()
        _aiInsightsHistory.value = (listOf(generation) + currentHistory).take(50)

        _aiGenerationStatus.value = AIGenerationStatus.ERROR
    }

    private suspend fun recordUserInteraction(
        insightId: Long?,
        type: UserInteractionType,
        details: String? = null
    ) {
        try {
            val interaction = UserInteraction(
                insightId = insightId,
                type = type,
                timestamp = System.currentTimeMillis(),
                details = details
            )

            sleepRepository.recordUserInteraction(interaction)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to record user interaction", e)
        }
    }

    private suspend fun loadUserPreferences() {
        val preferences = InsightsUserPreferences(
            preferredCategories = setOf(InsightCategory.QUALITY, InsightCategory.DURATION),
            autoGenerateInsights = true,
            notificationsEnabled = true,
            displayMode = InsightsDisplayMode.CARDS,
            maxInsightsToShow = 20
        )

        _userPreferences.value = preferences
    }

    private fun getUserFocusAreas(): List<String> {
        val preferences = _userPreferences.value
        return preferences?.preferredCategories?.map { it.name } ?: emptyList()
    }

    // Analytics calculation methods
    private fun calculateAverageInsightAge(insights: List<SleepInsight>): Long {
        if (insights.isEmpty()) return 0L

        val currentTime = System.currentTimeMillis()
        val totalAge = insights.sumOf { currentTime - it.timestamp }
        return totalAge / insights.size
    }

    private fun calculateEngagementRate(insights: List<SleepInsight>): Float {
        if (insights.isEmpty()) return 0f

        val acknowledgedCount = insights.count { it.isAcknowledged }
        return acknowledgedCount.toFloat() / insights.size
    }

    private fun calculateGenerationTrend(insights: List<SleepInsight>): Float {
        // Calculate trend in insight generation over time
        val recentInsights = insights.filter {
            it.timestamp >= System.currentTimeMillis() - TimeUnit.DAYS.toMillis(7)
        }.size

        val previousInsights = insights.filter {
            val weekAgo = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(7)
            val twoWeeksAgo = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(14)
            it.timestamp in twoWeeksAgo..weekAgo
        }.size

        return if (previousInsights > 0) {
            (recentInsights - previousInsights).toFloat() / previousInsights
        } else {
            0f
        }
    }

    private fun calculatePreferredCategories(): List<InsightCategory> {
        val allInsights = _allInsights.value ?: return emptyList()
        val acknowledgedInsights = allInsights.filter { it.isAcknowledged }

        return acknowledgedInsights.groupBy { it.category }
            .mapValues { it.value.size }
            .toList()
            .sortedByDescending { it.second }
            .take(3)
            .map { it.first }
    }

    private fun calculateAverageTimeToAcknowledge(): Long {
        // This would require tracking acknowledgment timestamps
        // For now, return a placeholder
        return TimeUnit.HOURS.toMillis(12)
    }

    private fun calculateSystemLoad(): Float {
        val activeJobs = activeGenerationJobs.size
        val maxJobs = MAX_CONCURRENT_GENERATIONS
        return activeJobs.toFloat() / maxJobs
    }

    // Error handling
    private fun setError(message: String) {
        _errorMessage.value = message
        Log.w(TAG, "Error set: $message")
    }

    fun clearError() {
        _errorMessage.value = null
    }

    fun clearStatusMessage() {
        _statusMessage.value = null
    }

    // ========== PUBLIC API METHODS ==========

    /**
     * Get insight by ID
     */
    fun getInsightById(insightId: Long): SleepInsight? {
        return _allInsights.value?.find { it.id == insightId }
    }

    /**
     * Get insights for category
     */
    fun getInsightsForCategory(category: InsightCategory): List<SleepInsight> {
        return _allInsights.value?.filter { it.category == category } ?: emptyList()
    }

    /**
     * Check if there are unacknowledged insights
     */
    fun hasUnacknowledgedInsights(): Boolean {
        return _unacknowledgedInsights.value?.isNotEmpty() ?: false
    }

    /**
     * Get insights analytics summary
     */
    fun getAnalyticsSummary(): String {
        val analytics = _insightsAnalytics.value ?: return "No analytics available"

        return buildString {
            append("Total: ${analytics.totalInsights}")
            append(" | Acknowledged: ${analytics.acknowledgedInsights}")
            append(" | AI Generated: ${analytics.aiGeneratedInsights}")
            append(" | High Priority: ${analytics.highPriorityInsights}")
            append(" | Engagement: ${(analytics.engagementRate * 100).toInt()}%")
        }
    }

    override fun onCleared() {
        super.onCleared()

        // Cancel background jobs
        refreshJob?.cancel()
        monitoringJob?.cancel()
        autoGenerationJob?.cancel()

        // Cancel active generation jobs
        activeGenerationJobs.values.forEach { it.cancel() }
        activeGenerationJobs.clear()

        // Cleanup components
        if (::aiInsightsEngine.isInitialized) {
            aiInsightsEngine.cleanup()
        }

        if (::insightsScheduler.isInitialized) {
            insightsScheduler.cleanup()
        }

        isBackgroundJobsRunning.set(false)

        Log.d(TAG, "InsightsViewModel cleared")
    }
}

// ========== SUPPORTING DATA CLASSES ==========

/**
 * Insights state management
 */
sealed class InsightsState {
    object Loading : InsightsState()
    object Empty : InsightsState()
    data class Success(val insights: List<SleepInsight>) : InsightsState()
    data class Error(val message: String) : InsightsState()
}

/**
 * AI generation status
 */
enum class AIGenerationStatus {
    IDLE, GENERATING, SUCCESS, ERROR
}

/**
 * Insights filtering options
 */
enum class InsightsFilter {
    ALL, UNACKNOWLEDGED, AI_GENERATED, HIGH_PRIORITY, RECENT, ACTIONABLE, PERSONAL, DEFAULT
}

/**
 * Insights sorting options
 */
enum class InsightsSortOption {
    NEWEST_FIRST, OLDEST_FIRST, PRIORITY_HIGH_FIRST, PRIORITY_LOW_FIRST, CATEGORY, RELEVANCE
}

/**
 * Display mode for insights
 */
enum class InsightsDisplayMode {
    CARDS, LIST, TIMELINE, GRID
}

/**
 * User interaction types for analytics
 */
enum class UserInteractionType {
    VIEWED, ACKNOWLEDGED, EXPANDED, COLLAPSED, SELECTED, SHARED, DELETED,
    MARKED_HELPFUL, MARKED_UNHELPFUL, FILTERED, SORTED, SEARCHED,
    CATEGORY_FILTERED, FILTERS_CLEARED, DISPLAY_MODE_CHANGED
}

/**
 * Insights analytics data
 */
data class InsightsAnalytics(
    val totalInsights: Int,
    val acknowledgedInsights: Int,
    val aiGeneratedInsights: Int,
    val highPriorityInsights: Int,
    val categoryDistribution: Map<InsightCategory, Int>,
    val averageAge: Long,
    val engagementRate: Float,
    val generationTrend: Float
)

/**
 * User engagement metrics
 */
data class UserEngagementMetrics(
    val totalInteractions: Long,
    val engagementRate: Float,
    val isHighEngagement: Boolean,
    val lastInteractionTime: Long,
    val preferredCategories: List<InsightCategory>,
    val averageTimeToAcknowledge: Long
)

/**
 * Performance metrics for insights system
 */
data class InsightsPerformanceMetrics(
    val activeGenerationJobs: Int,
    val totalInsightsGenerated: Long,
    val averageGenerationTime: Long,
    val cacheHitRate: Float,
    val errorRate: Float,
    val systemLoad: Float
)

/**
 * Generation progress tracking
 */
data class InsightGenerationProgress(
    val progress: Float = 0f,
    val message: String = "",
    val isActive: Boolean = false
)

/**
 * AI insight generation record
 */
data class AIInsightGeneration(
    val timestamp: Long,
    val type: InsightGenerationType,
    val insightsGenerated: Int,
    val success: Boolean,
    val processingTime: Long = 0L,
    val qualityScore: Float = 0f,
    val error: String? = null
)

/**
 * User preferences for insights
 */
data class InsightsUserPreferences(
    val preferredCategories: Set<InsightCategory>,
    val autoGenerateInsights: Boolean,
    val notificationsEnabled: Boolean,
    val displayMode: InsightsDisplayMode,
    val maxInsightsToShow: Int
)

/**
 * Personalization data
 */
data class PersonalizationData(
    val focusAreas: List<String> = emptyList(),
    val preferredCommunicationStyle: String = "supportive",
    val expertiseLevel: String = "general",
    val learningPreferences: Map<String, Any> = emptyMap()
)

/**
 * User interaction record
 */
data class UserInteraction(
    val insightId: Long?,
    val type: UserInteractionType,
    val timestamp: Long,
    val details: String? = null
)