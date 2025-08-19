package com.example.somniai.sensor

import android.util.Log
import kotlin.math.*

/**
 * Sleep phase indicators based on movement and noise patterns
 */
enum class SleepPhase {
    AWAKE,          // High movement/noise
    LIGHT_SLEEP,    // Moderate movement, some noise sensitivity
    DEEP_SLEEP,     // Minimal movement, low noise sensitivity
    REM_SLEEP,      // Increased movement, stable noise level
    UNKNOWN         // Insufficient data
}

/**
 * Sleep quality factors breakdown
 */
data class SleepQualityFactors(
    val movementScore: Float,      // 0-10 (10 = very still)
    val noiseScore: Float,         // 0-10 (10 = very quiet)
    val durationScore: Float,      // 0-10 (10 = optimal duration)
    val consistencyScore: Float,   // 0-10 (10 = consistent sleep)
    val overallScore: Float        // Weighted average
)

/**
 * Real-time sleep metrics
 */
data class LiveSleepMetrics(
    val currentPhase: SleepPhase,
    val phaseConfidence: Float,    // 0-1 confidence in phase detection
    val movementIntensity: Float,  // Current movement level
    val noiseLevel: Float,         // Current noise level
    val timeInCurrentPhase: Long,  // Time spent in current phase (ms)
    val totalRestlessness: Float,  // Overall restlessness score
    val sleepEfficiency: Float     // Estimated sleep efficiency percentage
)

/**
 * Sleep session analytics
 */
data class SleepSessionAnalytics(
    val totalDuration: Long,
    val sleepLatency: Long,        // Time to fall asleep
    val awakeDuration: Long,       // Time spent awake during session
    val lightSleepDuration: Long,
    val deepSleepDuration: Long,
    val remSleepDuration: Long,
    val sleepEfficiency: Float,    // Percentage of time actually sleeping
    val movementFrequency: Float,  // Movements per hour
    val averageNoiseLevel: Float,
    val qualityFactors: SleepQualityFactors,
    val phaseTransitions: List<PhaseTransition>
)

/**
 * Sleep phase transition event
 */
data class PhaseTransition(
    val timestamp: Long,
    val fromPhase: SleepPhase,
    val toPhase: SleepPhase,
    val confidence: Float
)

/**
 * Processes and analyzes sensor data to extract meaningful sleep metrics
 * Combines accelerometer and microphone data to determine sleep quality and phases
 */
class SensorDataProcessor {

    // Current session state
    private val movementEvents = mutableListOf<MovementEvent>()
    private val noiseEvents = mutableListOf<NoiseEvent>()
    private val phaseTransitions = mutableListOf<PhaseTransition>()

    // Real-time tracking
    private var currentPhase = SleepPhase.UNKNOWN
    private var phaseStartTime = 0L
    private var sessionStartTime = 0L
    private var lastAnalysisTime = 0L

    // Rolling window for real-time analysis
    private val analysisWindowMs = 5 * 60 * 1000L // 5 minutes
    private val phaseDetectionWindowMs = 10 * 60 * 1000L // 10 minutes for phase detection

    /**
     * Start processing for a new sleep session
     */
    fun startSession(startTime: Long = System.currentTimeMillis()) {
        sessionStartTime = startTime
        phaseStartTime = startTime
        lastAnalysisTime = startTime
        currentPhase = SleepPhase.AWAKE

        // Clear previous session data
        movementEvents.clear()
        noiseEvents.clear()
        phaseTransitions.clear()

        Log.d(TAG, "Started new sleep session processing")
    }

    /**
     * Process new movement event and update metrics
     */
    fun processMovementEvent(event: MovementEvent) {
        synchronized(movementEvents) {
            movementEvents.add(event)
        }

        // Trigger real-time analysis if enough time has passed
        if (event.timestamp - lastAnalysisTime > REAL_TIME_UPDATE_INTERVAL) {
            updateRealTimeMetrics(event.timestamp)
        }
    }

    /**
     * Process new noise event and update metrics
     */
    fun processNoiseEvent(event: NoiseEvent) {
        synchronized(noiseEvents) {
            noiseEvents.add(event)
        }

        // Trigger real-time analysis
        if (event.timestamp - lastAnalysisTime > REAL_TIME_UPDATE_INTERVAL) {
            updateRealTimeMetrics(event.timestamp)
        }
    }

    /**
     * Get current live sleep metrics
     */
    fun getLiveMetrics(): LiveSleepMetrics {
        val currentTime = System.currentTimeMillis()

        return LiveSleepMetrics(
            currentPhase = currentPhase,
            phaseConfidence = calculatePhaseConfidence(currentTime),
            movementIntensity = getRecentMovementIntensity(currentTime),
            noiseLevel = getRecentNoiseLevel(currentTime),
            timeInCurrentPhase = currentTime - phaseStartTime,
            totalRestlessness = calculateRestlessness(currentTime),
            sleepEfficiency = calculateCurrentSleepEfficiency(currentTime)
        )
    }

    /**
     * Generate comprehensive session analytics
     */
    fun generateSessionAnalytics(endTime: Long = System.currentTimeMillis()): SleepSessionAnalytics {
        val totalDuration = endTime - sessionStartTime

        return SleepSessionAnalytics(
            totalDuration = totalDuration,
            sleepLatency = calculateSleepLatency(),
            awakeDuration = calculatePhaseDuration(SleepPhase.AWAKE),
            lightSleepDuration = calculatePhaseduration(SleepPhase.LIGHT_SLEEP),
            deepSleepDuration = calculatePhaseduration(SleepPhase.DEEP_SLEEP),
            remSleepDuration = calculatePhaseuration(SleepPhase.REM_SLEEP),
            sleepEfficiency = calculateFinalSleepEfficiency(totalDuration),
            movementFrequency = calculateMovementFrequency(totalDuration),
            averageNoiseLevel = calculateAverageNoiseLevel(),
            qualityFactors = calculateQualityFactors(totalDuration),
            phaseTransitions = phaseTransitions.toList()
        )
    }

    /**
     * Update real-time metrics and detect sleep phase changes
     */
    private fun updateRealTimeMetrics(currentTime: Long) {
        lastAnalysisTime = currentTime

        val newPhase = detectCurrentSleepPhase(currentTime)

        if (newPhase != currentPhase) {
            // Record phase transition
            val transition = PhaseTransition(
                timestamp = currentTime,
                fromPhase = currentPhase,
                toPhase = newPhase,
                confidence = calculatePhaseConfidence(currentTime)
            )

            phaseTransitions.add(transition)
            Log.d(TAG, "Sleep phase transition: ${currentPhase.name} → ${newPhase.name} (confidence: ${transition.confidence})")

            currentPhase = newPhase
            phaseStartTime = currentTime
        }
    }

    /**
     * Detect current sleep phase based on recent sensor data
     */
    private fun detectCurrentSleepPhase(currentTime: Long): SleepPhase {
        val windowStart = currentTime - phaseDetectionWindowMs

        // Get recent events
        val recentMovements = getMovementsInWindow(windowStart, currentTime)
        val recentNoise = getNoiseInWindow(windowStart, currentTime)

        if (recentMovements.isEmpty() && recentNoise.isEmpty()) {
            return SleepPhase.UNKNOWN
        }

        // Calculate movement and noise intensities
        val movementIntensity = recentMovements.map { it.intensity }.average().toFloat()
        val noiseIntensity = recentNoise.map { it.amplitude }.average().toFloat()
        val movementFreq = recentMovements.size.toFloat() / (phaseDetectionWindowMs / 60000f) // per minute

        // Phase detection logic based on research patterns
        return when {
            // High movement and noise = likely awake
            movementIntensity > AWAKE_MOVEMENT_THRESHOLD && movementFreq > AWAKE_FREQUENCY_THRESHOLD ->
                SleepPhase.AWAKE

            // Low movement, low noise = deep sleep
            movementIntensity < DEEP_SLEEP_MOVEMENT_THRESHOLD && movementFreq < DEEP_SLEEP_FREQUENCY_THRESHOLD ->
                SleepPhase.DEEP_SLEEP

            // Moderate movement with low frequency = light sleep
            movementIntensity < LIGHT_SLEEP_MOVEMENT_THRESHOLD && movementFreq < LIGHT_SLEEP_FREQUENCY_THRESHOLD ->
                SleepPhase.LIGHT_SLEEP

            // Higher movement frequency but lower intensity = possible REM
            movementFreq > REM_FREQUENCY_THRESHOLD && movementIntensity < REM_MOVEMENT_THRESHOLD ->
                SleepPhase.REM_SLEEP

            else -> SleepPhase.LIGHT_SLEEP // Default assumption
        }
    }

    /**
     * Calculate confidence in current phase detection
     */
    private fun calculatePhaseConfidence(currentTime: Long): Float {
        val windowStart = currentTime - phaseDetectionWindowMs
        val recentMovements = getMovementsInWindow(windowStart, currentTime)
        val recentNoise = getNoiseInWindow(windowStart, currentTime)

        // Confidence based on data availability and consistency
        val dataPoints = recentMovements.size + recentNoise.size
        val dataConfidence = (dataPoints.toFloat() / 20f).coerceAtMost(1f) // More data = higher confidence

        // Time in current phase increases confidence
        val timeInPhase = currentTime - phaseStartTime
        val timeConfidence = (timeInPhase.toFloat() / (5 * 60 * 1000f)).coerceAtMost(1f) // 5 minutes for full confidence

        return (dataConfidence * 0.6f + timeConfidence * 0.4f).coerceIn(0.1f, 1f)
    }

    /**
     * Calculate recent movement intensity
     */
    private fun getRecentMovementIntensity(currentTime: Long): Float {
        val windowStart = currentTime - analysisWindowMs
        val recentMovements = getMovementsInWindow(windowStart, currentTime)

        return if (recentMovements.isNotEmpty()) {
            recentMovements.map { it.intensity }.average().toFloat()
        } else {
            0f
        }
    }

    /**
     * Calculate recent noise level
     */
    private fun getRecentNoiseLevel(currentTime: Long): Float {
        val windowStart = currentTime - analysisWindowMs
        val recentNoise = getNoiseInWindow(windowStart, currentTime)

        return if (recentNoise.isNotEmpty()) {
            recentNoise.map { it.decibelLevel }.average().toFloat()
        } else {
            0f
        }
    }

    /**
     * Calculate overall restlessness score
     */
    private fun calculateRestlessness(currentTime: Long): Float {
        val sessionDuration = currentTime - sessionStartTime
        if (sessionDuration == 0L) return 0f

        val movementScore = movementEvents.size.toFloat() / (sessionDuration / 3600000f) // movements per hour
        val intensityScore = if (movementEvents.isNotEmpty()) {
            movementEvents.map { it.intensity }.average().toFloat()
        } else {
            0f
        }

        // Combine frequency and intensity (0-10 scale)
        return ((movementScore * 0.6f + intensityScore * 0.4f) * 2f).coerceAtMost(10f)
    }

    /**
     * Calculate current sleep efficiency
     */
    private fun calculateCurrentSleepEfficiency(currentTime: Long): Float {
        val totalTime = currentTime - sessionStartTime
        if (totalTime == 0L) return 0f

        val awakeTime = calculatePhaseuration(SleepPhase.AWAKE)
        val sleepTime = totalTime - awakeTime

        return (sleepTime.toFloat() / totalTime.toFloat() * 100f).coerceIn(0f, 100f)
    }

    /**
     * Calculate time to fall asleep (sleep latency)
     */
    private fun calculateSleepLatency(): Long {
        // Find first transition from AWAKE to any sleep phase
        val firstSleepTransition = phaseTransitions.find {
            it.fromPhase == SleepPhase.AWAKE && it.toPhase != SleepPhase.AWAKE
        }

        return firstSleepTransition?.let {
            it.timestamp - sessionStartTime
        } ?: 0L
    }

    /**
     * Calculate total time spent in specific phase
     */
    private fun calculatePhaseuration(phase: SleepPhase): Long {
        var totalDuration = 0L
        var currentPhaseStart = sessionStartTime
        var currentPhaseType = SleepPhase.AWAKE // Assume starting awake

        for (transition in phaseTransitions) {
            if (currentPhaseType == phase) {
                totalDuration += transition.timestamp - currentPhaseStart
            }
            currentPhaseStart = transition.timestamp
            currentPhaseType = transition.toPhase
        }

        // Add time in current phase if it matches
        if (currentPhaseType == phase) {
            totalDuration += System.currentTimeMillis() - currentPhaseStart
        }

        return totalDuration
    }

    /**
     * Calculate movement frequency (movements per hour)
     */
    private fun calculateMovementFrequency(totalDuration: Long): Float {
        if (totalDuration == 0L) return 0f
        val hours = totalDuration.toFloat() / 3600000f
        return movementEvents.size.toFloat() / hours
    }

    /**
     * Calculate average noise level throughout session
     */
    private fun calculateAverageNoiseLevel(): Float {
        return if (noiseEvents.isNotEmpty()) {
            noiseEvents.map { it.decibelLevel }.average().toFloat()
        } else {
            0f
        }
    }

    /**
     * Calculate comprehensive sleep quality factors
     */
    private fun calculateQualityFactors(totalDuration: Long): SleepQualityFactors {
        val movementScore = calculateMovementScore()
        val noiseScore = calculateNoiseScore()
        val durationScore = calculateDurationScore(totalDuration)
        val consistencyScore = calculateConsistencyScore()

        // Weighted overall score
        val overallScore = (movementScore * 0.3f +
                noiseScore * 0.25f +
                durationScore * 0.25f +
                consistencyScore * 0.2f)

        return SleepQualityFactors(
            movementScore = movementScore,
            noiseScore = noiseScore,
            durationScore = durationScore,
            consistencyScore = consistencyScore,
            overallScore = overallScore
        )
    }

    private fun calculateMovementScore(): Float {
        val restlessness = calculateRestlessness(System.currentTimeMillis())
        return (10f - restlessness).coerceIn(0f, 10f)
    }

    private fun calculateNoiseScore(): Float {
        val avgNoise = calculateAverageNoiseLevel()
        return when {
            avgNoise <= 30f -> 10f // Very quiet
            avgNoise <= 40f -> 8f  // Quiet
            avgNoise <= 50f -> 6f  // Moderate
            avgNoise <= 60f -> 4f  // Noisy
            else -> 2f             // Very noisy
        }
    }

    private fun calculateDurationScore(duration: Long): Float {
        val hours = duration.toFloat() / 3600000f
        return when {
            hours >= 7f && hours <= 9f -> 10f // Optimal
            hours >= 6f && hours < 7f -> 8f   // Good
            hours >= 5f && hours < 6f -> 6f   // Fair
            hours >= 4f && hours < 5f -> 4f   // Poor
            else -> 2f                         // Very poor
        }
    }

    private fun calculateConsistencyScore(): Float {
        // Score based on phase transition frequency (fewer = better)
        val transitionsPerHour = phaseTransitions.size.toFloat() /
                ((System.currentTimeMillis() - sessionStartTime) / 3600000f)

        return when {
            transitionsPerHour <= 2f -> 10f  // Very consistent
            transitionsPerHour <= 4f -> 8f   // Good consistency
            transitionsPerHour <= 6f -> 6f   // Fair consistency
            transitionsPerHour <= 8f -> 4f   // Poor consistency
            else -> 2f                        // Very inconsistent
        }
    }

    private fun calculateFinalSleepEfficiency(totalDuration: Long): Float {
        val awakeTime = calculatePhaseuration(SleepPhase.AWAKE)
        val sleepTime = totalDuration - awakeTime
        return (sleepTime.toFloat() / totalDuration.toFloat() * 100f).coerceIn(0f, 100f)
    }

    // Helper methods for data retrieval
    private fun getMovementsInWindow(startTime: Long, endTime: Long): List<MovementEvent> {
        return synchronized(movementEvents) {
            movementEvents.filter { it.timestamp in startTime..endTime }
        }
    }

    private fun getNoiseInWindow(startTime: Long, endTime: Long): List<NoiseEvent> {
        return synchronized(noiseEvents) {
            noiseEvents.filter { it.timestamp in startTime..endTime }
        }
    }

    companion object {
        private const val TAG = "SensorDataProcessor"

        // Update intervals
        private const val REAL_TIME_UPDATE_INTERVAL = 30_000L // 30 seconds

        // Sleep phase detection thresholds
        private const val AWAKE_MOVEMENT_THRESHOLD = 4.0f
        private const val AWAKE_FREQUENCY_THRESHOLD = 3.0f

        private const val DEEP_SLEEP_MOVEMENT_THRESHOLD = 1.0f
        private const val DEEP_SLEEP_FREQUENCY_THRESHOLD = 0.5f

        private const val LIGHT_SLEEP_MOVEMENT_THRESHOLD = 2.5f
        private const val LIGHT_SLEEP_FREQUENCY_THRESHOLD = 1.5f

        private const val REM_MOVEMENT_THRESHOLD = 2.0f
        private const val REM_FREQUENCY_THRESHOLD = 2.0f
    }
}