package com.example.somniai.viewmodel

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.somniai.data.*
import com.example.somniai.repository.SleepRepository
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.util.*
import kotlin.math.sin
import kotlin.random.Random

/**
 * ViewModel for Charts Activity
 *
 * Manages chart data loading, caching, and UI state for all chart types.
 * Integrates with SleepRepository for comprehensive analytics data.
 */
class ChartsViewModel(
    private val sleepRepository: SleepRepository
) : ViewModel() {

    companion object {
        private const val TAG = "ChartsViewModel"
        private const val CACHE_DURATION_MS = 5 * 60 * 1000L // 5 minutes
    }

    // Loading states
    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    // Chart data
    private val _dailyTrends = MutableLiveData<List<DailyTrendData>?>()
    val dailyTrends: LiveData<List<DailyTrendData>?> = _dailyTrends

    private val _qualityFactors = MutableLiveData<List<QualityFactorBreakdown>?>()
    val qualityFactors: LiveData<List<QualityFactorBreakdown>?> = _qualityFactors

    private val _movementPatterns = MutableLiveData<List<MovementPatternData>?>()
    val movementPatterns: LiveData<List<MovementPatternData>?> = _movementPatterns

    private val _phaseDistribution = MutableLiveData<List<PhaseDistributionData>?>()
    val phaseDistribution: LiveData<List<PhaseDistributionData>?> = _phaseDistribution

    private val _efficiencyTrends = MutableLiveData<List<EfficiencyTrendData>?>()
    val efficiencyTrends: LiveData<List<EfficiencyTrendData>?> = _efficiencyTrends

    private val _weeklyStats = MutableLiveData<List<WeeklyStatsData>?>()
    val weeklyStats: LiveData<List<WeeklyStatsData>?> = _weeklyStats

    // Cache management
    private var lastDataLoad = 0L
    private var cachedDateRange: Pair<Long, Long>? = null

    /**
     * Load duration trends data for the specified date range
     */
    fun loadDurationTrends(startDate: Long, endDate: Long) {
        if (shouldUseCachedData(startDate, endDate)) {
            Log.d(TAG, "Using cached duration trends data")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val trendsResult = sleepRepository.getSleepTrends(
                    daysBack = calculateDaysBack(startDate, endDate),
                    forceRefresh = false
                )

                trendsResult.onSuccess { trends ->
                    _dailyTrends.value = trends
                    updateCacheInfo(startDate, endDate)
                    Log.d(TAG, "Duration trends loaded: ${trends.size} data points")
                }.onFailure { exception ->
                    // If no real data, generate sample data for demo
                    val sampleData = generateSampleDurationTrends(startDate, endDate)
                    _dailyTrends.value = sampleData
                    setError("Using sample data: ${exception.message}")
                    Log.w(TAG, "Using sample duration trends data", exception)
                }

            } catch (e: Exception) {
                // Generate sample data as fallback
                val sampleData = generateSampleDurationTrends(startDate, endDate)
                _dailyTrends.value = sampleData
                setError("Error loading duration trends: ${e.message}")
                Log.e(TAG, "Exception loading duration trends", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Load quality factors data for the specified date range
     */
    fun loadQualityFactors(startDate: Long, endDate: Long) {
        if (shouldUseCachedData(startDate, endDate)) {
            Log.d(TAG, "Using cached quality factors data")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                // Get sessions in date range
                val sessionsResult = sleepRepository.getSessionsInDateRange(startDate, endDate)

                sessionsResult.onSuccess { sessions ->
                    // Generate quality factor breakdowns from sessions
                    val qualityBreakdowns = sessions.mapNotNull { session ->
                        session.qualityFactors?.let { factors ->
                            QualityFactorBreakdown(
                                sessionId = session.id,
                                sessionDate = session.startTime,
                                movementScore = factors.movementScore,
                                noiseScore = factors.noiseScore,
                                durationScore = factors.durationScore,
                                consistencyScore = factors.consistencyScore,
                                efficiencyScore = factors.efficiencyScore,
                                phaseBalanceScore = 7.5f, // Placeholder - would calculate from phase data
                                overallScore = factors.overallScore
                            )
                        }
                    }

                    // If no real data, generate sample data for demo
                    if (qualityBreakdowns.isEmpty()) {
                        val sampleData = generateSampleQualityFactors(startDate, endDate)
                        _qualityFactors.value = sampleData
                    } else {
                        _qualityFactors.value = qualityBreakdowns
                    }

                    updateCacheInfo(startDate, endDate)
                    Log.d(TAG, "Quality factors loaded: ${qualityBreakdowns.size} breakdowns")

                }.onFailure { exception ->
                    // Generate sample data as fallback
                    val sampleData = generateSampleQualityFactors(startDate, endDate)
                    _qualityFactors.value = sampleData
                    setError("Using sample data: ${exception.message}")
                    Log.w(TAG, "Using sample quality factors data", exception)
                }

            } catch (e: Exception) {
                // Generate sample data as fallback
                val sampleData = generateSampleQualityFactors(startDate, endDate)
                _qualityFactors.value = sampleData
                setError("Error loading quality factors: ${e.message}")
                Log.e(TAG, "Exception loading quality factors", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Load movement patterns data for the specified date range
     */
    fun loadMovementPatterns(startDate: Long, endDate: Long) {
        if (shouldUseCachedData(startDate, endDate)) {
            Log.d(TAG, "Using cached movement patterns data")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                // Get sessions and generate movement patterns
                val sessionsResult = sleepRepository.getSessionsInDateRange(startDate, endDate)

                sessionsResult.onSuccess { sessions ->
                    val movementPatterns = sessions.flatMap { session ->
                        // Generate hourly movement patterns for each session
                        generateMovementPatternsForSession(session)
                    }

                    // If no patterns generated, create sample data
                    if (movementPatterns.isEmpty()) {
                        val sampleData = generateSampleMovementPatterns()
                        _movementPatterns.value = sampleData
                    } else {
                        _movementPatterns.value = movementPatterns
                    }

                    updateCacheInfo(startDate, endDate)
                    Log.d(TAG, "Movement patterns loaded: ${movementPatterns.size} patterns")

                }.onFailure { exception ->
                    // Generate sample data as fallback
                    val sampleData = generateSampleMovementPatterns()
                    _movementPatterns.value = sampleData
                    setError("Using sample data: ${exception.message}")
                    Log.w(TAG, "Using sample movement patterns data", exception)
                }

            } catch (e: Exception) {
                // Generate sample data as fallback
                val sampleData = generateSampleMovementPatterns()
                _movementPatterns.value = sampleData
                setError("Error loading movement patterns: ${e.message}")
                Log.e(TAG, "Exception loading movement patterns", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Load sleep phase distribution data for the specified date range
     */
    fun loadPhaseDistribution(startDate: Long, endDate: Long) {
        if (shouldUseCachedData(startDate, endDate)) {
            Log.d(TAG, "Using cached phase distribution data")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val sessionsResult = sleepRepository.getSessionsInDateRange(startDate, endDate)

                sessionsResult.onSuccess { sessions ->
                    val phaseDistributions = sessions.map { session ->
                        PhaseDistributionData(
                            sessionId = session.id,
                            awakeDuration = session.awakeDuration,
                            lightSleepDuration = session.lightSleepDuration,
                            deepSleepDuration = session.deepSleepDuration,
                            remSleepDuration = session.remSleepDuration,
                            totalDuration = session.duration
                        )
                    }

                    // If no phase data, generate sample distribution
                    if (phaseDistributions.isEmpty() || phaseDistributions.all { it.totalDuration == 0L }) {
                        val sampleData = generateSamplePhaseDistribution()
                        _phaseDistribution.value = sampleData
                    } else {
                        _phaseDistribution.value = phaseDistributions
                    }

                    updateCacheInfo(startDate, endDate)
                    Log.d(TAG, "Phase distribution loaded: ${phaseDistributions.size} distributions")

                }.onFailure { exception ->
                    // Generate sample data as fallback
                    val sampleData = generateSamplePhaseDistribution()
                    _phaseDistribution.value = sampleData
                    setError("Using sample data: ${exception.message}")
                    Log.w(TAG, "Using sample phase distribution data", exception)
                }

            } catch (e: Exception) {
                // Generate sample data as fallback
                val sampleData = generateSamplePhaseDistribution()
                _phaseDistribution.value = sampleData
                setError("Error loading phase distribution: ${e.message}")
                Log.e(TAG, "Exception loading phase distribution", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Load efficiency trends data for the specified date range
     */
    fun loadEfficiencyTrends(startDate: Long, endDate: Long) {
        if (shouldUseCachedData(startDate, endDate)) {
            Log.d(TAG, "Using cached efficiency trends data")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                val sessionsResult = sleepRepository.getSessionsInDateRange(startDate, endDate)

                sessionsResult.onSuccess { sessions ->
                    val efficiencyData = sessions.map { session ->
                        EfficiencyTrendData(
                            sessionDate = session.startTime,
                            basicEfficiency = session.sleepEfficiency,
                            adjustedEfficiency = session.sleepEfficiency * 0.95f, // Slightly adjusted
                            qualityWeightedEfficiency = session.sleepEfficiency * 0.9f,
                            sleepLatency = session.sleepLatency,
                            wakeCount = estimateWakeCount(session),
                            movementDisruptions = session.getTotalMovements()
                        )
                    }

                    // If no efficiency data, generate sample trends
                    if (efficiencyData.isEmpty()) {
                        val sampleData = generateSampleEfficiencyTrends(startDate, endDate)
                        _efficiencyTrends.value = sampleData
                    } else {
                        _efficiencyTrends.value = efficiencyData
                    }

                    updateCacheInfo(startDate, endDate)
                    Log.d(TAG, "Efficiency trends loaded: ${efficiencyData.size} data points")

                }.onFailure { exception ->
                    // Generate sample data as fallback
                    val sampleData = generateSampleEfficiencyTrends(startDate, endDate)
                    _efficiencyTrends.value = sampleData
                    setError("Using sample data: ${exception.message}")
                    Log.w(TAG, "Using sample efficiency trends data", exception)
                }

            } catch (e: Exception) {
                // Generate sample data as fallback
                val sampleData = generateSampleEfficiencyTrends(startDate, endDate)
                _efficiencyTrends.value = sampleData
                setError("Error loading efficiency trends: ${e.message}")
                Log.e(TAG, "Exception loading efficiency trends", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Load weekly statistics data for the specified date range
     */
    fun loadWeeklyStats(startDate: Long, endDate: Long) {
        if (shouldUseCachedData(startDate, endDate)) {
            Log.d(TAG, "Using cached weekly stats data")
            return
        }

        viewModelScope.launch {
            _isLoading.value = true
            try {
                // Calculate weeks in range
                val weeksDiff = (endDate - startDate) / (7 * 24 * 60 * 60 * 1000L)
                val weeklyData = mutableListOf<WeeklyStatsData>()

                for (weekOffset in 0 until weeksDiff.toInt()) {
                    val weekStart = startDate + (weekOffset * 7 * 24 * 60 * 60 * 1000L)
                    val weekEnd = weekStart + (7 * 24 * 60 * 60 * 1000L)

                    val weekSessionsResult = sleepRepository.getSessionsInDateRange(weekStart, weekEnd)

                    weekSessionsResult.onSuccess { sessions ->
                        if (sessions.isNotEmpty()) {
                            val weekStats = WeeklyStatsData(
                                weekStart = weekStart,
                                sessionCount = sessions.size,
                                averageDuration = sessions.map { it.duration }.average().toLong(),
                                averageQuality = sessions.mapNotNull { it.sleepQualityScore }.average().toFloat(),
                                averageEfficiency = sessions.map { it.sleepEfficiency }.average().toFloat(),
                                consistencyScore = calculateConsistencyScore(sessions),
                                weekdayAverageDuration = calculateWeekdayAverage(sessions),
                                weekendAverageDuration = calculateWeekendAverage(sessions),
                                bestQualityDate = sessions.maxByOrNull { it.sleepQualityScore ?: 0f }?.startTime,
                                worstQualityDate = sessions.minByOrNull { it.sleepQualityScore ?: 10f }?.startTime
                            )
                            weeklyData.add(weekStats)
                        }
                    }
                }

                // If no weekly data, generate sample data
                if (weeklyData.isEmpty()) {
                    val sampleData = generateSampleWeeklyStats(startDate, endDate)
                    _weeklyStats.value = sampleData
                } else {
                    _weeklyStats.value = weeklyData
                }

                updateCacheInfo(startDate, endDate)
                Log.d(TAG, "Weekly stats loaded: ${weeklyData.size} weeks")

            } catch (e: Exception) {
                // Generate sample data as fallback
                val sampleData = generateSampleWeeklyStats(startDate, endDate)
                _weeklyStats.value = sampleData
                setError("Error loading weekly stats: ${e.message}")
                Log.e(TAG, "Exception loading weekly stats", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    // ========== HELPER METHODS ==========

    private fun shouldUseCachedData(startDate: Long, endDate: Long): Boolean {
        val currentTime = System.currentTimeMillis()
        val cacheValid = currentTime - lastDataLoad < CACHE_DURATION_MS
        val sameRange = cachedDateRange?.let { (cachedStart, cachedEnd) ->
            cachedStart == startDate && cachedEnd == endDate
        } ?: false

        return cacheValid && sameRange
    }

    private fun updateCacheInfo(startDate: Long, endDate: Long) {
        lastDataLoad = System.currentTimeMillis()
        cachedDateRange = Pair(startDate, endDate)
    }

    private fun calculateDaysBack(startDate: Long, endDate: Long): Int {
        return ((endDate - startDate) / (24 * 60 * 60 * 1000L)).toInt()
    }

    private fun setError(message: String) {
        _errorMessage.value = message
        Log.w(TAG, "Error: $message")
    }

    fun clearError() {
        _errorMessage.value = null
    }

    // ========== SAMPLE DATA GENERATORS ==========

    private fun generateSampleDurationTrends(startDate: Long, endDate: Long): List<DailyTrendData> {
        val trends = mutableListOf<DailyTrendData>()
        val daysCount = calculateDaysBack(startDate, endDate)

        repeat(minOf(daysCount, 30)) { index ->
            val sessionDate = startDate + (index * 24 * 60 * 60 * 1000L)
            trends.add(
                DailyTrendData(
                    date = sessionDate,
                    sessionCount = 1,
                    totalDuration = (7.5f + sin(index * 0.2f) * 1.5f + Random.nextFloat() * 0.5f).toLong() * 60 * 60 * 1000L,
                    averageDuration = (7.5f + sin(index * 0.2f) * 1.5f + Random.nextFloat() * 0.5f).toLong() * 60 * 60 * 1000L,
                    averageQuality = 7f + Random.nextFloat() * 2f,
                    averageEfficiency = 80f + Random.nextFloat() * 15f,
                    totalMovements = Random.nextInt(10, 50),
                    totalNoiseEvents = Random.nextInt(2, 15),
                    averageBedtime = sessionDate - (Random.nextLong(2 * 60 * 60 * 1000L)),
                    sleepDebt = 0L
                )
            )
        }
        return trends
    }

    private fun generateSampleQualityFactors(startDate: Long, endDate: Long): List<QualityFactorBreakdown> {
        val sampleData = mutableListOf<QualityFactorBreakdown>()
        val daysCount = calculateDaysBack(startDate, endDate)

        repeat(minOf(daysCount, 10)) { index ->
            val sessionDate = startDate + (index * 24 * 60 * 60 * 1000L)
            sampleData.add(
                QualityFactorBreakdown(
                    sessionId = index.toLong(),
                    sessionDate = sessionDate,
                    movementScore = Random.nextFloat() * 3 + 7, // 7-10 range
                    noiseScore = Random.nextFloat() * 2 + 6, // 6-8 range
                    durationScore = Random.nextFloat() * 2 + 7, // 7-9 range
                    consistencyScore = Random.nextFloat() * 3 + 6, // 6-9 range
                    efficiencyScore = Random.nextFloat() * 2 + 8, // 8-10 range
                    phaseBalanceScore = Random.nextFloat() * 2 + 7, // 7-9 range
                    overallScore = Random.nextFloat() * 2 + 7.5f // 7.5-9.5 range
                )
            )
        }
        return sampleData
    }

    private fun generateSampleMovementPatterns(): List<MovementPatternData> {
        val patterns = mutableListOf<MovementPatternData>()

        // Generate hourly pattern for an 8-hour sleep session
        repeat(8) { hour ->
            val baseIntensity = when (hour) {
                0, 7 -> 3.5f // Higher at sleep start/end
                1, 6 -> 2.0f // Moderate
                2, 3, 4, 5 -> 1.5f // Low during deep sleep
                else -> 2.5f
            }

            patterns.add(
                MovementPatternData(
                    sessionId = 1L,
                    hourOfSession = hour,
                    averageIntensity = baseIntensity + (Random.nextFloat() - 0.5f),
                    movementCount = (baseIntensity * 10 + Random.nextInt(5)).toInt(),
                    significantMovements = (baseIntensity * 2).toInt(),
                    maxIntensity = baseIntensity + Random.nextFloat() * 2,
                    restlessnessScore = baseIntensity + Random.nextFloat()
                )
            )
        }
        return patterns
    }

    private fun generateSamplePhaseDistribution(): List<PhaseDistributionData> {
        // Generate a sample 8-hour sleep distribution
        val totalDuration = 8 * 60 * 60 * 1000L // 8 hours in ms

        return listOf(
            PhaseDistributionData(
                sessionId = 1L,
                awakeDuration = (totalDuration * 0.05).toLong(), // 5% awake
                lightSleepDuration = (totalDuration * 0.50).toLong(), // 50% light sleep
                deepSleepDuration = (totalDuration * 0.25).toLong(), // 25% deep sleep
                remSleepDuration = (totalDuration * 0.20).toLong(), // 20% REM sleep
                totalDuration = totalDuration
            )
        )
    }

    private fun generateSampleEfficiencyTrends(startDate: Long, endDate: Long): List<EfficiencyTrendData> {
        val trends = mutableListOf<EfficiencyTrendData>()
        val daysCount = calculateDaysBack(startDate, endDate)

        repeat(minOf(daysCount, 30)) { index ->
            val sessionDate = startDate + (index * 24 * 60 * 60 * 1000L)
            val baseEfficiency = 85f + sin(index * 0.3) * 10f // Oscillating around 85%

            trends.add(
                EfficiencyTrendData(
                    sessionDate = sessionDate,
                    basicEfficiency = baseEfficiency + Random.nextFloat() * 5,
                    adjustedEfficiency = baseEfficiency + Random.nextFloat() * 3,
                    qualityWeightedEfficiency = baseEfficiency - 5 + Random.nextFloat() * 3,
                    sleepLatency = Random.nextLong(10 * 60 * 1000, 30 * 60 * 1000), // 10-30 minutes
                    wakeCount = Random.nextInt(1, 5),
                    movementDisruptions = Random.nextInt(5, 20)
                )
            )
        }
        return trends
    }

    private fun generateSampleWeeklyStats(startDate: Long, endDate: Long): List<WeeklyStatsData> {
        val weeklyData = mutableListOf<WeeklyStatsData>()
        val weeksDiff = (endDate - startDate) / (7 * 24 * 60 * 60 * 1000L)

        repeat(weeksDiff.toInt()) { weekIndex ->
            val weekStart = startDate + (weekIndex * 7 * 24 * 60 * 60 * 1000L)
            val calendar = Calendar.getInstance().apply { timeInMillis = weekStart }

            weeklyData.add(
                WeeklyStatsData(
                    weekStart = weekStart,
                    sessionCount = Random.nextInt(5, 8), // 5-7 sessions per week
                    averageDuration = (7.5f + Random.nextFloat() * 1.5f).toLong() * 60 * 60 * 1000L, // 7.5-9 hours
                    averageQuality = 7f + Random.nextFloat() * 2f, // 7-9 quality
                    averageEfficiency = 80f + Random.nextFloat() * 15f, // 80-95% efficiency
                    consistencyScore = 6f + Random.nextFloat() * 3f, // 6-9 consistency
                    weekdayAverageDuration = (7f + Random.nextFloat()).toLong() * 60 * 60 * 1000L,
                    weekendAverageDuration = (8f + Random.nextFloat() * 2f).toLong() * 60 * 60 * 1000L,
                    bestQualityDate = weekStart + Random.nextLong(7 * 24 * 60 * 60 * 1000L),
                    worstQualityDate = weekStart + Random.nextLong(7 * 24 * 60 * 60 * 1000L)
                )
            )
        }
        return weeklyData
    }

    // ========== UTILITY METHODS ==========

    private fun generateMovementPatternsForSession(session: SleepSession): List<MovementPatternData> {
        val patterns = mutableListOf<MovementPatternData>()
        val sessionHours = (session.duration / (60 * 60 * 1000L)).toInt()

        repeat(minOf(sessionHours, 12)) { hour ->
            patterns.add(
                MovementPatternData(
                    sessionId = session.id,
                    hourOfSession = hour,
                    averageIntensity = session.averageMovementIntensity + Random.nextFloat() - 0.5f,
                    movementCount = Random.nextInt(5, 25),
                    significantMovements = Random.nextInt(1, 8),
                    maxIntensity = session.averageMovementIntensity + Random.nextFloat() * 2,
                    restlessnessScore = Random.nextFloat() * 5 + 3
                )
            )
        }
        return patterns
    }

    private fun estimateWakeCount(session: SleepSession): Int {
        // Estimate wake count based on movement events and efficiency
        val movementFactor = session.getTotalMovements() / 50f
        val efficiencyFactor = (100f - session.sleepEfficiency) / 20f
        return (movementFactor + efficiencyFactor).toInt().coerceIn(0, 8)
    }

    private fun calculateConsistencyScore(sessions: List<SleepSession>): Float {
        if (sessions.size < 2) return 5f

        val durations = sessions.map { it.duration.toFloat() }
        val mean = durations.average()
        val variance = durations.map { (it - mean) * (it - mean) }.average()
        val coefficient = variance / (mean * mean)

        // Lower coefficient = higher consistency (scale 0-10)
        return (10f - coefficient * 20f).coerceIn(0f, 10f)
    }

    private fun calculateWeekdayAverage(sessions: List<SleepSession>): Long {
        val weekdaySessions = sessions.filter { session ->
            val calendar = Calendar.getInstance().apply { timeInMillis = session.startTime }
            val dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK)
            dayOfWeek in Calendar.MONDAY..Calendar.FRIDAY
        }

        return if (weekdaySessions.isNotEmpty()) {
            weekdaySessions.map { it.duration }.average().toLong()
        } else 0L
    }

    private fun calculateWeekendAverage(sessions: List<SleepSession>): Long {
        val weekendSessions = sessions.filter { session ->
            val calendar = Calendar.getInstance().apply { timeInMillis = session.startTime }
            val dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK)
            dayOfWeek == Calendar.SATURDAY || dayOfWeek == Calendar.SUNDAY
        }

        return if (weekendSessions.isNotEmpty()) {
            weekendSessions.map { it.duration }.average().toLong()
        } else 0L
    }

    fun cleanup() {
        // Clean up any resources if needed
        _errorMessage.value = null
        lastDataLoad = 0L
        cachedDateRange = null
    }

    override fun onCleared() {
        super.onCleared()
        cleanup()
        Log.d(TAG, "ChartsViewModel cleared")
    }
}

/**
 * Factory for creating ChartsViewModel with repository dependency
 */
class ChartsViewModelFactory(
    private val sleepRepository: SleepRepository
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(ChartsViewModel::class.java)) {
            return ChartsViewModel(sleepRepository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}