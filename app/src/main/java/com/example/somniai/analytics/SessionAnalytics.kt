package com.example.somniai.analytics

import android.util.Log
import com.example.somniai.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.*

/**
 * Individual session analytics engine
 *
 * Provides real-time and post-session analysis for individual sleep sessions:
 * - Real-time sleep phase detection and transitions
 * - Sleep onset and wake detection algorithms
 * - Per-session quality metrics and scoring
 * - Movement and noise pattern analysis within sessions
 * - Session summary generation with detailed insights
 * - Timeline analysis and event correlation
 */
class SessionAnalytics {

    companion object {
        private const val TAG = "SessionAnalytics"

        // Sleep onset detection parameters
        private const val SLEEP_ONSET_MOVEMENT_THRESHOLD = 1.5f
        private const val SLEEP_ONSET_TIME_WINDOW = 15 * 60 * 1000L // 15 minutes
        private const val SLEEP_ONSET_MIN_QUIET_PERIOD = 10 * 60 * 1000L // 10 minutes

        // Wake detection parameters
        private const val WAKE_MOVEMENT_THRESHOLD = 3.0f
        private const val WAKE_SUSTAINED_PERIOD = 5 * 60 * 1000L // 5 minutes
        private const val WAKE_CONFIRMATION_MOVEMENTS = 3

        // Phase transition parameters
        private const val PHASE_TRANSITION_MIN_DURATION = 5 * 60 * 1000L // 5 minutes
        private const val DEEP_SLEEP_MOVEMENT_THRESHOLD = 1.0f
        private const val REM_MOVEMENT_PATTERN_THRESHOLD = 2.5f
        private const val LIGHT_SLEEP_MOVEMENT_RANGE = 1.5f..3.0f

        // Real-time analysis windows
        private const val REAL_TIME_WINDOW = 2 * 60 * 1000L // 2 minutes
        private const val SHORT_TERM_WINDOW = 10 * 60 * 1000L // 10 minutes
        private const val MEDIUM_TERM_WINDOW = 30 * 60 * 1000L // 30 minutes

        // Quality thresholds
        private const val EXCELLENT_EFFICIENCY_THRESHOLD = 90f
        private const val GOOD_EFFICIENCY_THRESHOLD = 80f
        private const val FAIR_EFFICIENCY_THRESHOLD = 70f
    }

    // ========== REAL-TIME SESSION ANALYSIS ==========

    /**
     * Analyze current session state in real-time
     */
    suspend fun analyzeRealTimeSession(
        sessionStart: Long,
        currentTime: Long,
        recentMovements: List<MovementEvent>,
        recentNoises: List<NoiseEvent>,
        currentPhase: SleepPhase,
        phaseHistory: List<PhaseTransition>
    ): RealTimeSessionAnalysis = withContext(Dispatchers.Default) {
        try {
            val sessionDuration = currentTime - sessionStart

            // Current phase analysis
            val phaseAnalysis = analyzeCurrentPhase(
                currentPhase,
                recentMovements,
                recentNoises,
                sessionDuration
            )

            // Movement pattern analysis
            val movementAnalysis = analyzeRecentMovementPattern(
                recentMovements,
                REAL_TIME_WINDOW,
                currentTime
            )

            // Noise environment analysis
            val noiseAnalysis = analyzeNoiseEnvironment(
                recentNoises,
                REAL_TIME_WINDOW,
                currentTime
            )

            // Sleep quality indicators
            val qualityIndicators = calculateRealTimeQualityIndicators(
                sessionDuration,
                recentMovements,
                recentNoises,
                phaseHistory
            )

            // Efficiency tracking
            val currentEfficiency = calculateCurrentEfficiency(
                sessionStart,
                currentTime,
                phaseHistory,
                recentMovements
            )

            // Predictions and recommendations
            val predictions = generateRealTimePredictions(
                sessionDuration,
                phaseHistory,
                movementAnalysis,
                noiseAnalysis
            )

            RealTimeSessionAnalysis(
                sessionDuration = sessionDuration,
                currentPhase = currentPhase,
                phaseAnalysis = phaseAnalysis,
                movementAnalysis = movementAnalysis,
                noiseAnalysis = noiseAnalysis,
                qualityIndicators = qualityIndicators,
                currentEfficiency = currentEfficiency,
                predictions = predictions,
                recommendations = generateRealTimeRecommendations(qualityIndicators, predictions),
                timestamp = currentTime
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error in real-time session analysis", e)
            RealTimeSessionAnalysis(timestamp = currentTime)
        }
    }

    /**
     * Detect sleep onset from movement and noise patterns
     */
    suspend fun detectSleepOnset(
        sessionStart: Long,
        movements: List<MovementEvent>,
        noises: List<NoiseEvent>
    ): SleepOnsetAnalysis = withContext(Dispatchers.Default) {
        try {
            // Sort events by timestamp
            val sortedMovements = movements.sortedBy { it.timestamp }
            val sortedNoises = noises.sortedBy { it.timestamp }

            // Find periods of reduced activity
            val quietPeriods = findQuietPeriods(sortedMovements, sortedNoises, sessionStart)

            // Determine most likely sleep onset time
            val onsetTime = determineOnsetTime(quietPeriods, sessionStart)

            // Calculate sleep latency
            val sleepLatency = if (onsetTime != null) onsetTime - sessionStart else null

            // Analyze pre-sleep activity
            val preOnsetAnalysis = analyzePreOnsetActivity(
                sortedMovements,
                sortedNoises,
                sessionStart,
                onsetTime
            )

            // Confidence scoring
            val confidence = calculateOnsetConfidence(
                quietPeriods,
                preOnsetAnalysis,
                sleepLatency
            )

            SleepOnsetAnalysis(
                onsetTime = onsetTime,
                sleepLatency = sleepLatency,
                confidence = confidence,
                quietPeriods = quietPeriods,
                preOnsetActivity = preOnsetAnalysis,
                onsetPhase = determineOnsetPhase(sortedMovements, onsetTime),
                qualityIndicators = analyzeOnsetQuality(sleepLatency, preOnsetAnalysis)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error detecting sleep onset", e)
            SleepOnsetAnalysis()
        }
    }

    /**
     * Detect wake events and final awakening
     */
    suspend fun detectWakeEvents(
        movements: List<MovementEvent>,
        noises: List<NoiseEvent>,
        sessionEnd: Long?
    ): WakeDetectionAnalysis = withContext(Dispatchers.Default) {
        try {
            val sortedMovements = movements.sortedBy { it.timestamp }

            // Detect micro-awakenings (brief wake periods)
            val microAwakenings = detectMicroAwakenings(sortedMovements)

            // Detect final awakening
            val finalAwakening = sessionEnd?.let {
                detectFinalAwakening(sortedMovements, noises, it)
            }

            // Analyze wake patterns
            val wakePatterns = analyzeWakePatterns(microAwakenings, finalAwakening)

            // Calculate awakening quality
            val awakeningQuality = calculateAwakeningQuality(
                finalAwakening,
                microAwakenings,
                movements
            )

            WakeDetectionAnalysis(
                microAwakenings = microAwakenings,
                finalAwakening = finalAwakening,
                totalWakeEvents = microAwakenings.size,
                wakePatterns = wakePatterns,
                awakeningQuality = awakeningQuality,
                wakeEfficiency = calculateWakeEfficiency(microAwakenings, movements)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error detecting wake events", e)
            WakeDetectionAnalysis()
        }
    }

    // ========== PHASE TRANSITION ANALYSIS ==========

    /**
     * Analyze sleep phase transitions within a session
     */
    suspend fun analyzePhaseTransitions(
        phaseTransitions: List<PhaseTransition>,
        sessionDuration: Long
    ): PhaseTransitionAnalysis = withContext(Dispatchers.Default) {
        try {
            if (phaseTransitions.isEmpty()) {
                return@withContext PhaseTransitionAnalysis()
            }

            val sortedTransitions = phaseTransitions.sortedBy { it.timestamp }

            // Calculate phase durations
            val phaseDurations = calculatePhaseDurations(sortedTransitions, sessionDuration)

            // Analyze transition patterns
            val transitionPatterns = analyzeTransitionPatterns(sortedTransitions)

            // Calculate phase efficiency
            val phaseEfficiency = calculatePhaseEfficiency(phaseDurations, sessionDuration)

            // Detect unusual patterns
            val anomalies = detectPhaseAnomalies(sortedTransitions, phaseDurations)

            // Sleep cycle analysis
            val cycleAnalysis = analyzeSleepCycles(sortedTransitions, sessionDuration)

            PhaseTransitionAnalysis(
                totalTransitions = sortedTransitions.size,
                phaseDurations = phaseDurations,
                transitionPatterns = transitionPatterns,
                phaseEfficiency = phaseEfficiency,
                anomalies = anomalies,
                cycleAnalysis = cycleAnalysis,
                phaseBalance = calculatePhaseBalance(phaseDurations),
                transitionQuality = assessTransitionQuality(sortedTransitions, phaseDurations)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing phase transitions", e)
            PhaseTransitionAnalysis()
        }
    }

    /**
     * Predict next phase transition based on current state
     */
    suspend fun predictNextPhaseTransition(
        currentPhase: SleepPhase,
        timeInCurrentPhase: Long,
        recentMovements: List<MovementEvent>,
        phaseHistory: List<PhaseTransition>
    ): PhaseTransitionPrediction = withContext(Dispatchers.Default) {
        try {
            // Analyze typical phase duration patterns
            val typicalDuration = calculateTypicalPhaseDuration(currentPhase, phaseHistory)

            // Movement-based transition indicators
            val movementIndicators = analyzeMovementTransitionIndicators(
                recentMovements,
                currentPhase
            )

            // Time-based probability
            val timeProbability = calculateTimeBasedTransitionProbability(
                timeInCurrentPhase,
                typicalDuration
            )

            // Next likely phase
            val nextPhase = predictNextPhase(currentPhase, phaseHistory)

            // Confidence calculation
            val confidence = calculateTransitionPredictionConfidence(
                movementIndicators,
                timeProbability,
                timeInCurrentPhase,
                typicalDuration
            )

            // Estimated time to transition
            val estimatedTimeToTransition = estimateTimeToTransition(
                timeInCurrentPhase,
                typicalDuration,
                movementIndicators
            )

            PhaseTransitionPrediction(
                currentPhase = currentPhase,
                predictedNextPhase = nextPhase,
                confidence = confidence,
                estimatedTimeToTransition = estimatedTimeToTransition,
                movementIndicators = movementIndicators,
                timeProbability = timeProbability,
                reasoning = generateTransitionReasoning(
                    currentPhase,
                    nextPhase,
                    movementIndicators,
                    timeProbability
                )
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error predicting phase transition", e)
            PhaseTransitionPrediction(currentPhase = currentPhase)
        }
    }

    // ========== SESSION QUALITY ANALYSIS ==========

    /**
     * Generate comprehensive session quality metrics
     */
    suspend fun analyzeSessionQuality(session: SleepSession): SessionQualityAnalysis = withContext(Dispatchers.Default) {
        try {
            // Basic quality metrics
            val basicMetrics = calculateBasicQualityMetrics(session)

            // Movement quality analysis
            val movementQuality = analyzeMovementQuality(session.movementEvents, session.duration)

            // Noise quality analysis
            val noiseQuality = analyzeNoiseQuality(session.noiseEvents)

            // Phase quality analysis
            val phaseQuality = analyzePhaseQuality(session.phaseTransitions, session.duration)

            // Timing quality analysis
            val timingQuality = analyzeTimingQuality(session)

            // Overall quality score with detailed breakdown
            val qualityBreakdown = calculateDetailedQualityBreakdown(
                basicMetrics,
                movementQuality,
                noiseQuality,
                phaseQuality,
                timingQuality
            )

            // Quality trends within session
            val intraSessionTrends = analyzeIntraSessionTrends(session)

            // Comparative analysis
            val comparativeMetrics = calculateComparativeMetrics(session)

            SessionQualityAnalysis(
                overallScore = qualityBreakdown.overallScore,
                basicMetrics = basicMetrics,
                movementQuality = movementQuality,
                noiseQuality = noiseQuality,
                phaseQuality = phaseQuality,
                timingQuality = timingQuality,
                qualityBreakdown = qualityBreakdown,
                intraSessionTrends = intraSessionTrends,
                comparativeMetrics = comparativeMetrics,
                qualityGrade = determineQualityGrade(qualityBreakdown.overallScore),
                strengthAreas = identifyStrengthAreas(qualityBreakdown),
                improvementAreas = identifyImprovementAreas(qualityBreakdown)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing session quality", e)
            SessionQualityAnalysis()
        }
    }

    // ========== SESSION SUMMARY GENERATION ==========

    /**
     * Generate comprehensive session summary
     */
    suspend fun generateSessionSummary(
        session: SleepSession,
        onsetAnalysis: SleepOnsetAnalysis? = null,
        wakeAnalysis: WakeDetectionAnalysis? = null,
        phaseAnalysis: PhaseTransitionAnalysis? = null,
        qualityAnalysis: SessionQualityAnalysis? = null
    ): SessionSummary = withContext(Dispatchers.Default) {
        try {
            // Core session statistics
            val coreStats = generateCoreStatistics(session)

            // Sleep architecture summary
            val architecture = generateSleepArchitecture(session, phaseAnalysis)

            // Efficiency metrics
            val efficiency = generateEfficiencyMetrics(session, onsetAnalysis, wakeAnalysis)

            // Environmental factors
            val environment = analyzeEnvironmentalFactors(session.movementEvents, session.noiseEvents)

            // Key highlights
            val highlights = generateSessionHighlights(
                session,
                onsetAnalysis,
                wakeAnalysis,
                phaseAnalysis,
                qualityAnalysis
            )

            // Recommendations
            val recommendations = generateSessionRecommendations(
                session,
                qualityAnalysis,
                environment
            )

            // Timeline events
            val timeline = generateSessionTimeline(
                session,
                onsetAnalysis,
                wakeAnalysis,
                phaseAnalysis
            )

            SessionSummary(
                sessionId = session.id,
                startTime = session.startTime,
                endTime = session.endTime,
                coreStatistics = coreStats,
                sleepArchitecture = architecture,
                efficiencyMetrics = efficiency,
                environmentalFactors = environment,
                highlights = highlights,
                recommendations = recommendations,
                timeline = timeline,
                qualityScore = qualityAnalysis?.overallScore ?: session.sleepQualityScore,
                qualityGrade = qualityAnalysis?.qualityGrade ?: "N/A"
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error generating session summary", e)
            SessionSummary(sessionId = session.id)
        }
    }

    // ========== HELPER METHODS - REAL-TIME ANALYSIS ==========

    private fun analyzeCurrentPhase(
        currentPhase: SleepPhase,
        recentMovements: List<MovementEvent>,
        recentNoises: List<NoiseEvent>,
        sessionDuration: Long
    ): CurrentPhaseAnalysis {
        val movementLevel = if (recentMovements.isNotEmpty()) {
            recentMovements.map { it.intensity }.average().toFloat()
        } else 0f

        val noiseLevel = if (recentNoises.isNotEmpty()) {
            recentNoises.map { it.decibelLevel }.average()
        } else 0.0

        val phaseStability = calculatePhaseStability(currentPhase, movementLevel, noiseLevel)
        val confidence = calculatePhaseConfidence(currentPhase, movementLevel, sessionDuration)

        return CurrentPhaseAnalysis(
            phase = currentPhase,
            stability = phaseStability,
            confidence = confidence,
            movementLevel = movementLevel,
            noiseLevel = noiseLevel.toFloat(),
            indicators = generatePhaseIndicators(currentPhase, movementLevel, noiseLevel.toFloat())
        )
    }

    private fun analyzeRecentMovementPattern(
        movements: List<MovementEvent>,
        timeWindow: Long,
        currentTime: Long
    ): MovementPatternAnalysis {
        val recentMovements = movements.filter {
            currentTime - it.timestamp <= timeWindow
        }

        if (recentMovements.isEmpty()) {
            return MovementPatternAnalysis()
        }

        val avgIntensity = recentMovements.map { it.intensity }.average().toFloat()
        val maxIntensity = recentMovements.maxOf { it.intensity }
        val frequency = recentMovements.size.toFloat() / (timeWindow / 60000f) // per minute

        val pattern = when {
            avgIntensity < 1.5f && frequency < 2f -> MovementPattern.VERY_STILL
            avgIntensity < 2.5f && frequency < 5f -> MovementPattern.STILL
            avgIntensity < 4f && frequency < 10f -> MovementPattern.MODERATE
            avgIntensity < 6f && frequency < 20f -> MovementPattern.RESTLESS
            else -> MovementPattern.VERY_RESTLESS
        }

        return MovementPatternAnalysis(
            pattern = pattern,
            averageIntensity = avgIntensity,
            maxIntensity = maxIntensity,
            frequency = frequency,
            totalMovements = recentMovements.size,
            trend = calculateMovementTrend(recentMovements)
        )
    }

    private fun analyzeNoiseEnvironment(
        noises: List<NoiseEvent>,
        timeWindow: Long,
        currentTime: Long
    ): NoiseEnvironmentAnalysis {
        val recentNoises = noises.filter {
            currentTime - it.timestamp <= timeWindow
        }

        if (recentNoises.isEmpty()) {
            return NoiseEnvironmentAnalysis()
        }

        val avgDecibel = recentNoises.map { it.decibelLevel }.average().toFloat()
        val maxDecibel = recentNoises.maxOf { it.decibelLevel }
        val disruptiveEvents = recentNoises.count { it.isDisruptive() }

        val environment = when {
            avgDecibel < 30f -> NoiseEnvironment.VERY_QUIET
            avgDecibel < 40f -> NoiseEnvironment.QUIET
            avgDecibel < 50f -> NoiseEnvironment.MODERATE
            avgDecibel < 60f -> NoiseEnvironment.NOISY
            else -> NoiseEnvironment.VERY_NOISY
        }

        return NoiseEnvironmentAnalysis(
            environment = environment,
            averageDecibel = avgDecibel,
            maxDecibel = maxDecibel,
            disruptiveEvents = disruptiveEvents,
            totalEvents = recentNoises.size,
            qualityImpact = calculateNoiseQualityImpact(avgDecibel, disruptiveEvents)
        )
    }

    private fun calculateRealTimeQualityIndicators(
        sessionDuration: Long,
        recentMovements: List<MovementEvent>,
        recentNoises: List<NoiseEvent>,
        phaseHistory: List<PhaseTransition>
    ): QualityIndicators {
        val durationHours = sessionDuration / (1000f * 60f * 60f)

        // Duration indicator
        val durationIndicator = when {
            durationHours < 4f -> QualityIndicator.POOR
            durationHours < 6f -> QualityIndicator.FAIR
            durationHours < 7f -> QualityIndicator.GOOD
            durationHours <= 9f -> QualityIndicator.EXCELLENT
            durationHours <= 10f -> QualityIndicator.GOOD
            else -> QualityIndicator.FAIR
        }

        // Movement indicator
        val avgMovement = if (recentMovements.isNotEmpty()) {
            recentMovements.map { it.intensity }.average().toFloat()
        } else 0f

        val movementIndicator = when {
            avgMovement < 1.5f -> QualityIndicator.EXCELLENT
            avgMovement < 2.5f -> QualityIndicator.GOOD
            avgMovement < 4f -> QualityIndicator.FAIR
            else -> QualityIndicator.POOR
        }

        // Noise indicator
        val avgNoise = if (recentNoises.isNotEmpty()) {
            recentNoises.map { it.decibelLevel }.average().toFloat()
        } else 0f

        val noiseIndicator = when {
            avgNoise < 35f -> QualityIndicator.EXCELLENT
            avgNoise < 45f -> QualityIndicator.GOOD
            avgNoise < 55f -> QualityIndicator.FAIR
            else -> QualityIndicator.POOR
        }

        // Phase indicator
        val phaseIndicator = if (phaseHistory.size > 2) {
            QualityIndicator.GOOD
        } else {
            QualityIndicator.FAIR
        }

        return QualityIndicators(
            duration = durationIndicator,
            movement = movementIndicator,
            noise = noiseIndicator,
            phases = phaseIndicator
        )
    }

    private fun calculateCurrentEfficiency(
        sessionStart: Long,
        currentTime: Long,
        phaseHistory: List<PhaseTransition>,
        recentMovements: List<MovementEvent>
    ): Float {
        val totalDuration = currentTime - sessionStart
        if (totalDuration <= 0) return 0f

        // Estimate time spent actually sleeping vs awake
        val awakePeriods = estimateAwakePeriods(recentMovements, sessionStart, currentTime)
        val awakeTime = awakePeriods.sumOf { it.second - it.first }

        val sleepTime = totalDuration - awakeTime
        return ((sleepTime.toFloat() / totalDuration) * 100f).coerceIn(0f, 100f)
    }

    // ========== HELPER METHODS - SLEEP ONSET ==========

    private fun findQuietPeriods(
        movements: List<MovementEvent>,
        noises: List<NoiseEvent>,
        sessionStart: Long
    ): List<QuietPeriod> {
        val periods = mutableListOf<QuietPeriod>()
        val timeSlots = mutableMapOf<Long, ActivityLevel>()

        // Analyze activity in 1-minute time slots
        val firstHour = sessionStart + (60 * 60 * 1000L) // First hour only
        var currentSlot = sessionStart

        while (currentSlot < firstHour) {
            val slotEnd = currentSlot + 60000L // 1 minute slots

            val slotMovements = movements.filter {
                it.timestamp >= currentSlot && it.timestamp < slotEnd
            }
            val slotNoises = noises.filter {
                it.timestamp >= currentSlot && it.timestamp < slotEnd
            }

            val activityLevel = calculateActivityLevel(slotMovements, slotNoises)
            timeSlots[currentSlot] = activityLevel

            currentSlot = slotEnd
        }

        // Find continuous quiet periods
        var periodStart: Long? = null
        for ((timestamp, activity) in timeSlots.toSortedMap()) {
            if (activity == ActivityLevel.QUIET || activity == ActivityLevel.VERY_QUIET) {
                if (periodStart == null) {
                    periodStart = timestamp
                }
            } else {
                if (periodStart != null) {
                    val duration = timestamp - periodStart
                    if (duration >= SLEEP_ONSET_MIN_QUIET_PERIOD) {
                        periods.add(QuietPeriod(periodStart, timestamp, duration))
                    }
                    periodStart = null
                }
            }
        }

        return periods
    }

    private fun determineOnsetTime(quietPeriods: List<QuietPeriod>, sessionStart: Long): Long? {
        // Find the first substantial quiet period
        return quietPeriods
            .filter { it.duration >= SLEEP_ONSET_MIN_QUIET_PERIOD }
            .minByOrNull { it.startTime }
            ?.startTime
    }

    private fun calculateOnsetConfidence(
        quietPeriods: List<QuietPeriod>,
        preOnsetActivity: PreOnsetActivity,
        sleepLatency: Long?
    ): Float {
        if (sleepLatency == null) return 0f

        var confidence = 0.5f // Base confidence

        // Longer quiet periods increase confidence
        val longestQuietPeriod = quietPeriods.maxOfOrNull { it.duration } ?: 0L
        confidence += (longestQuietPeriod / (30 * 60 * 1000f)).coerceAtMost(0.3f) // Max 0.3 boost

        // Reasonable sleep latency increases confidence
        when {
            sleepLatency in (5 * 60 * 1000L)..(30 * 60 * 1000L) -> confidence += 0.2f // 5-30 minutes
            sleepLatency < (5 * 60 * 1000L) -> confidence += 0.1f // Very quick
            sleepLatency > (60 * 60 * 1000L) -> confidence -= 0.2f // Over 1 hour
        }

        // Clear activity reduction increases confidence
        if (preOnsetActivity.activityReduction > 0.5f) {
            confidence += 0.2f
        }

        return confidence.coerceIn(0f, 1f)
    }

    // ========== HELPER METHODS - PHASE ANALYSIS ==========

    private fun calculatePhaseDurations(
        transitions: List<PhaseTransition>,
        sessionDuration: Long
    ): Map<SleepPhase, Long> {
        val durations = mutableMapOf<SleepPhase, Long>()

        if (transitions.isEmpty()) return durations

        for (i in transitions.indices) {
            val transition = transitions[i]
            val nextTransition = transitions.getOrNull(i + 1)

            val phaseDuration = if (nextTransition != null) {
                nextTransition.timestamp - transition.timestamp
            } else {
                // Last phase duration until session end
                sessionDuration - (transition.timestamp - transitions.first().timestamp)
            }

            durations[transition.toPhase] = durations.getOrDefault(transition.toPhase, 0L) + phaseDuration
        }

        return durations
    }

    private fun analyzeTransitionPatterns(transitions: List<PhaseTransition>): TransitionPatterns {
        val patterns = mutableMapOf<String, Int>()
        val averageConfidence = transitions.map { it.confidence }.average().toFloat()

        // Count transition types
        for (i in 0 until transitions.size - 1) {
            val from = transitions[i].toPhase
            val to = transitions[i + 1].toPhase
            val pattern = "${from.name}_to_${to.name}"
            patterns[pattern] = patterns.getOrDefault(pattern, 0) + 1
        }

        // Find most common pattern
        val mostCommonPattern = patterns.maxByOrNull { it.value }

        return TransitionPatterns(
            patterns = patterns,
            averageConfidence = averageConfidence,
            mostCommonPattern = mostCommonPattern?.key ?: "Unknown",
            totalTransitions = transitions.size
        )
    }

    private fun detectPhaseAnomalies(
        transitions: List<PhaseTransition>,
        durations: Map<SleepPhase, Long>
    ): List<PhaseAnomaly> {
        val anomalies = mutableListOf<PhaseAnomaly>()

        // Check for unusually short phases
        durations.forEach { (phase, duration) ->
            if (duration < PHASE_TRANSITION_MIN_DURATION) {
                anomalies.add(
                    PhaseAnomaly(
                        type = AnomalyType.SHORT_PHASE,
                        phase = phase,
                        description = "Phase duration (${duration/1000}s) below minimum threshold",
                        severity = AnomalySeverity.MEDIUM
                    )
                )
            }
        }

        // Check for missing deep sleep
        if (!durations.containsKey(SleepPhase.DEEP) || durations[SleepPhase.DEEP]!! < (30 * 60 * 1000L)) {
            anomalies.add(
                PhaseAnomaly(
                    type = AnomalyType.INSUFFICIENT_DEEP_SLEEP,
                    phase = SleepPhase.DEEP,
                    description = "Insufficient or missing deep sleep phase",
                    severity = AnomalySeverity.HIGH
                )
            )
        }

        // Check for excessive transitions
        if (transitions.size > 20) {
            anomalies.add(
                PhaseAnomaly(
                    type = AnomalyType.EXCESSIVE_TRANSITIONS,
                    phase = null,
                    description = "Excessive number of phase transitions (${transitions.size})",
                    severity = AnomalySeverity.MEDIUM
                )
            )
        }

        return anomalies
    }

    private fun analyzeSleepCycles(
        transitions: List<PhaseTransition>,
        sessionDuration: Long
    ): SleepCycleAnalysis {
        // Detect complete sleep cycles (typically 90-120 minutes)
        val cycles = mutableListOf<SleepCycle>()
        var cycleStart: Long? = null
        var hasDeepSleep = false
        var hasREM = false

        for (transition in transitions) {
            when (transition.toPhase) {
                SleepPhase.LIGHT -> {
                    if (cycleStart == null) {
                        cycleStart = transition.timestamp
                    }
                }
                SleepPhase.DEEP -> hasDeepSleep = true
                SleepPhase.REM -> hasREM = true
                SleepPhase.AWAKE -> {
                    // End of cycle if we had deep sleep and/or REM
                    if (cycleStart != null && (hasDeepSleep || hasREM)) {
                        cycles.add(
                            SleepCycle(
                                startTime = cycleStart,
                                endTime = transition.timestamp,
                                duration = transition.timestamp - cycleStart,
                                hadDeepSleep = hasDeepSleep,
                                hadREM = hasREM
                            )
                        )
                    }
                    cycleStart = null
                    hasDeepSleep = false
                    hasREM = false
                }
            }
        }

        val averageCycleDuration = if (cycles.isNotEmpty()) {
            cycles.map { it.duration }.average().toLong()
        } else 0L

        val cycleEfficiency = if (cycles.isNotEmpty()) {
            cycles.count { it.hadDeepSleep && it.hadREM }.toFloat() / cycles.size
        } else 0f

        return SleepCycleAnalysis(
            cycles = cycles,
            cycleCount = cycles.size,
            averageCycleDuration = averageCycleDuration,
            cycleEfficiency = cycleEfficiency,
            completeCycles = cycles.count { it.hadDeepSleep && it.hadREM }
        )
    }

    // ========== DATA CLASSES ==========

    data class RealTimeSessionAnalysis(
        val sessionDuration: Long = 0L,
        val currentPhase: SleepPhase = SleepPhase.AWAKE,
        val phaseAnalysis: CurrentPhaseAnalysis = CurrentPhaseAnalysis(),
        val movementAnalysis: MovementPatternAnalysis = MovementPatternAnalysis(),
        val noiseAnalysis: NoiseEnvironmentAnalysis = NoiseEnvironmentAnalysis(),
        val qualityIndicators: QualityIndicators = QualityIndicators(),
        val currentEfficiency: Float = 0f,
        val predictions: RealTimePredictions = RealTimePredictions(),
        val recommendations: List<String> = emptyList(),
        val timestamp: Long = System.currentTimeMillis()
    )

    data class CurrentPhaseAnalysis(
        val phase: SleepPhase = SleepPhase.AWAKE,
        val stability: Float = 0f,
        val confidence: Float = 0f,
        val movementLevel: Float = 0f,
        val noiseLevel: Float = 0f,
        val indicators: List<String> = emptyList()
    )

    data class MovementPatternAnalysis(
        val pattern: MovementPattern = MovementPattern.STILL,
        val averageIntensity: Float = 0f,
        val maxIntensity: Float = 0f,
        val frequency: Float = 0f,
        val totalMovements: Int = 0,
        val trend: MovementTrend = MovementTrend.STABLE
    )

    data class NoiseEnvironmentAnalysis(
        val environment: NoiseEnvironment = NoiseEnvironment.QUIET,
        val averageDecibel: Float = 0f,
        val maxDecibel: Float = 0f,
        val disruptiveEvents: Int = 0,
        val totalEvents: Int = 0,
        val qualityImpact: Float = 0f
    )

    data class QualityIndicators(
        val duration: QualityIndicator = QualityIndicator.FAIR,
        val movement: QualityIndicator = QualityIndicator.FAIR,
        val noise: QualityIndicator = QualityIndicator.FAIR,
        val phases: QualityIndicator = QualityIndicator.FAIR
    )

    data class RealTimePredictions(
        val expectedTotalDuration: Long = 0L,
        val expectedQualityScore: Float = 0f,
        val nextPhaseTransition: Long = 0L,
        val sleepDebtImpact: Float = 0f
    )

    data class SleepOnsetAnalysis(
        val onsetTime: Long? = null,
        val sleepLatency: Long? = null,
        val confidence: Float = 0f,
        val quietPeriods: List<QuietPeriod> = emptyList(),
        val preOnsetActivity: PreOnsetActivity = PreOnsetActivity(),
        val onsetPhase: SleepPhase = SleepPhase.LIGHT,
        val qualityIndicators: OnsetQualityIndicators = OnsetQualityIndicators()
    )

    data class QuietPeriod(
        val startTime: Long,
        val endTime: Long,
        val duration: Long
    )

    data class PreOnsetActivity(
        val averageMovement: Float = 0f,
        val movementReduction: Float = 0f,
        val activityReduction: Float = 0f,
        val gradualCalming: Boolean = false
    )

    data class OnsetQualityIndicators(
        val latencyQuality: String = "Unknown",
        val transitionQuality: String = "Unknown",
        val environmentQuality: String = "Unknown"
    )

    data class WakeDetectionAnalysis(
        val microAwakenings: List<MicroAwakening> = emptyList(),
        val finalAwakening: FinalAwakening? = null,
        val totalWakeEvents: Int = 0,
        val wakePatterns: WakePatterns = WakePatterns(),
        val awakeningQuality: AwakeningQuality = AwakeningQuality(),
        val wakeEfficiency: Float = 0f
    )

    data class MicroAwakening(
        val startTime: Long,
        val endTime: Long,
        val duration: Long,
        val intensity: Float,
        val cause: String = "Unknown"
    )

    data class FinalAwakening(
        val time: Long,
        val naturalness: Float,
        val alertness: Float,
        val movementPattern: String
    )

    data class WakePatterns(
        val frequency: Float = 0f,
        val distribution: String = "Unknown",
        val clustering: Boolean = false
    )

    data class AwakeningQuality(
        val naturalWake: Boolean = false,
        val alertnessScore: Float = 0f,
        val morningMood: String = "Unknown"
    )

    data class PhaseTransitionAnalysis(
        val totalTransitions: Int = 0,
        val phaseDurations: Map<SleepPhase, Long> = emptyMap(),
        val transitionPatterns: TransitionPatterns = TransitionPatterns(),
        val phaseEfficiency: Float = 0f,
        val anomalies: List<PhaseAnomaly> = emptyList(),
        val cycleAnalysis: SleepCycleAnalysis = SleepCycleAnalysis(),
        val phaseBalance: PhaseBalance = PhaseBalance(),
        val transitionQuality: Float = 0f
    )

    data class TransitionPatterns(
        val patterns: Map<String, Int> = emptyMap(),
        val averageConfidence: Float = 0f,
        val mostCommonPattern: String = "Unknown",
        val totalTransitions: Int = 0
    )

    data class PhaseAnomaly(
        val type: AnomalyType,
        val phase: SleepPhase?,
        val description: String,
        val severity: AnomalySeverity
    )

    data class SleepCycleAnalysis(
        val cycles: List<SleepCycle> = emptyList(),
        val cycleCount: Int = 0,
        val averageCycleDuration: Long = 0L,
        val cycleEfficiency: Float = 0f,
        val completeCycles: Int = 0
    )

    data class SleepCycle(
        val startTime: Long,
        val endTime: Long,
        val duration: Long,
        val hadDeepSleep: Boolean,
        val hadREM: Boolean
    )

    data class PhaseBalance(
        val lightSleepPercentage: Float = 0f,
        val deepSleepPercentage: Float = 0f,
        val remSleepPercentage: Float = 0f,
        val awakePercentage: Float = 0f,
        val isBalanced: Boolean = false
    )

    data class PhaseTransitionPrediction(
        val currentPhase: SleepPhase,
        val predictedNextPhase: SleepPhase = SleepPhase.LIGHT,
        val confidence: Float = 0f,
        val estimatedTimeToTransition: Long = 0L,
        val movementIndicators: MovementTransitionIndicators = MovementTransitionIndicators(),
        val timeProbability: Float = 0f,
        val reasoning: String = ""
    )

    data class MovementTransitionIndicators(
        val increasing: Boolean = false,
        val pattern: String = "Stable",
        val intensity: Float = 0f
    )

    data class SessionQualityAnalysis(
        val overallScore: Float = 0f,
        val basicMetrics: BasicQualityMetrics = BasicQualityMetrics(),
        val movementQuality: MovementQualityMetrics = MovementQualityMetrics(),
        val noiseQuality: NoiseQualityMetrics = NoiseQualityMetrics(),
        val phaseQuality: PhaseQualityMetrics = PhaseQualityMetrics(),
        val timingQuality: TimingQualityMetrics = TimingQualityMetrics(),
        val qualityBreakdown: QualityBreakdown = QualityBreakdown(),
        val intraSessionTrends: IntraSessionTrends = IntraSessionTrends(),
        val comparativeMetrics: ComparativeMetrics = ComparativeMetrics(),
        val qualityGrade: String = "C",
        val strengthAreas: List<String> = emptyList(),
        val improvementAreas: List<String> = emptyList()
    )

    data class SessionSummary(
        val sessionId: Long,
        val startTime: Long = 0L,
        val endTime: Long? = null,
        val coreStatistics: CoreStatistics = CoreStatistics(),
        val sleepArchitecture: SleepArchitecture = SleepArchitecture(),
        val efficiencyMetrics: EfficiencyMetrics = EfficiencyMetrics(),
        val environmentalFactors: EnvironmentalFactors = EnvironmentalFactors(),
        val highlights: List<SessionHighlight> = emptyList(),
        val recommendations: List<String> = emptyList(),
        val timeline: List<TimelineEvent> = emptyList(),
        val qualityScore: Float? = null,
        val qualityGrade: String = "N/A"
    )

    // Supporting data classes would continue here...
    // (truncated for brevity, but would include all the referenced data classes)

    // ========== ENUMS ==========

    enum class MovementPattern {
        VERY_STILL, STILL, MODERATE, RESTLESS, VERY_RESTLESS
    }

    enum class MovementTrend {
        INCREASING, DECREASING, STABLE, FLUCTUATING
    }

    enum class NoiseEnvironment {
        VERY_QUIET, QUIET, MODERATE, NOISY, VERY_NOISY
    }

    enum class QualityIndicator {
        EXCELLENT, GOOD, FAIR, POOR
    }

    enum class ActivityLevel {
        VERY_QUIET, QUIET, MODERATE, ACTIVE, VERY_ACTIVE
    }

    enum class AnomalyType {
        SHORT_PHASE, INSUFFICIENT_DEEP_SLEEP, EXCESSIVE_TRANSITIONS, MISSING_REM
    }

    enum class AnomalySeverity {
        LOW, MEDIUM, HIGH, CRITICAL
    }

    // Placeholder implementations for remaining helper methods
    private fun calculatePhaseStability(phase: SleepPhase, movementLevel: Float, noiseLevel: Double): Float = 5f
    private fun calculatePhaseConfidence(phase: SleepPhase, movementLevel: Float, duration: Long): Float = 0.5f
    private fun generatePhaseIndicators(phase: SleepPhase, movement: Float, noise: Float): List<String> = emptyList()
    private fun calculateMovementTrend(movements: List<MovementEvent>): MovementTrend = MovementTrend.STABLE
    private fun calculateNoiseQualityImpact(avgDecibel: Float, disruptiveEvents: Int): Float = 0f
    private fun estimateAwakePeriods(movements: List<MovementEvent>, start: Long, end: Long): List<Pair<Long, Long>> = emptyList()
    private fun calculateActivityLevel(movements: List<MovementEvent>, noises: List<NoiseEvent>): ActivityLevel = ActivityLevel.QUIET
    private fun analyzePreOnsetActivity(movements: List<MovementEvent>, noises: List<NoiseEvent>, start: Long, onset: Long?): PreOnsetActivity = PreOnsetActivity()
    private fun determineOnsetPhase(movements: List<MovementEvent>, onsetTime: Long?): SleepPhase = SleepPhase.LIGHT
    private fun analyzeOnsetQuality(latency: Long?, activity: PreOnsetActivity): OnsetQualityIndicators = OnsetQualityIndicators()
    private fun detectMicroAwakenings(movements: List<MovementEvent>): List<MicroAwakening> = emptyList()
    private fun detectFinalAwakening(movements: List<MovementEvent>, noises: List<NoiseEvent>, sessionEnd: Long): FinalAwakening? = null
    private fun analyzeWakePatterns(micro: List<MicroAwakening>, final: FinalAwakening?): WakePatterns = WakePatterns()
    private fun calculateAwakeningQuality(final: FinalAwakening?, micro: List<MicroAwakening>, movements: List<MovementEvent>): AwakeningQuality = AwakeningQuality()
    private fun calculateWakeEfficiency(micro: List<MicroAwakening>, movements: List<MovementEvent>): Float = 0f
    private fun calculatePhaseEfficiency(durations: Map<SleepPhase, Long>, sessionDuration: Long): Float = 0f
    private fun calculatePhaseBalance(durations: Map<SleepPhase, Long>): PhaseBalance = PhaseBalance()
    private fun assessTransitionQuality(transitions: List<PhaseTransition>, durations: Map<SleepPhase, Long>): Float = 0f
    private fun generateRealTimePredictions(duration: Long, phases: List<PhaseTransition>, movement: MovementPatternAnalysis, noise: NoiseEnvironmentAnalysis): RealTimePredictions = RealTimePredictions()
    private fun generateRealTimeRecommendations(quality: QualityIndicators, predictions: RealTimePredictions): List<String> = emptyList()
    private fun calculateTypicalPhaseDuration(phase: SleepPhase, history: List<PhaseTransition>): Long = 30 * 60 * 1000L
    private fun analyzeMovementTransitionIndicators(movements: List<MovementEvent>, phase: SleepPhase): MovementTransitionIndicators = MovementTransitionIndicators()
    private fun calculateTimeBasedTransitionProbability(timeInPhase: Long, typical: Long): Float = 0.5f
    private fun predictNextPhase(current: SleepPhase, history: List<PhaseTransition>): SleepPhase = SleepPhase.LIGHT
    private fun calculateTransitionPredictionConfidence(movement: MovementTransitionIndicators, time: Float, current: Long, typical: Long): Float = 0.5f
    private fun estimateTimeToTransition(current: Long, typical: Long, indicators: MovementTransitionIndicators): Long = 15 * 60 * 1000L
    private fun generateTransitionReasoning(current: SleepPhase, next: SleepPhase, movement: MovementTransitionIndicators, time: Float): String = "Analysis in progress"
}

// Additional data classes (simplified for brevity)
data class BasicQualityMetrics(val duration: Float = 0f, val efficiency: Float = 0f)
data class MovementQualityMetrics(val restlessness: Float = 0f, val stability: Float = 0f)
data class NoiseQualityMetrics(val quietness: Float = 0f, val disruption: Float = 0f)
data class PhaseQualityMetrics(val balance: Float = 0f, val transitions: Float = 0f)
data class TimingQualityMetrics(val consistency: Float = 0f, val naturalness: Float = 0f)
data class QualityBreakdown(val overallScore: Float = 0f, val factors: Map<String, Float> = emptyMap())
data class IntraSessionTrends(val improving: Boolean = false, val declining: Boolean = false)
data class ComparativeMetrics(val betterThanAverage: Boolean = false, val percentile: Float = 50f)
data class CoreStatistics(val duration: Long = 0L, val efficiency: Float = 0f, val quality: Float = 0f)
data class SleepArchitecture(val phases: Map<SleepPhase, Long> = emptyMap(), val cycles: Int = 0)
data class EfficiencyMetrics(val overall: Float = 0f, val onset: Long = 0L, val wake: Int = 0)
data class EnvironmentalFactors(val movement: Float = 0f, val noise: Float = 0f, val disruptions: Int = 0)
data class SessionHighlight(val type: String, val description: String, val positive: Boolean = true)
data class TimelineEvent(val timestamp: Long, val event: String, val description: String)