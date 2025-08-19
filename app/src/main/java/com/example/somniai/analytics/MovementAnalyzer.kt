package com.example.somniai.analytics


import android.util.Log
import com.example.somniai.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.*

/**
 * Advanced movement pattern analysis engine for sleep tracking
 *
 * Provides comprehensive movement analytics including:
 * - Real-time movement intensity algorithms with adaptive thresholds
 * - Multi-dimensional restlessness scoring with temporal analysis
 * - Sleep phase detection based on movement patterns and ML-inspired algorithms
 * - Movement event clustering with noise filtering and pattern recognition
 * - Micro-movement detection and sleep quality correlation
 * - Positional change analysis and sleep disruption scoring
 */
class MovementAnalyzer {

    companion object {
        private const val TAG = "MovementAnalyzer"

        // Movement intensity thresholds
        private const val MICRO_MOVEMENT_THRESHOLD = 0.5f
        private const val LIGHT_MOVEMENT_THRESHOLD = 1.5f
        private const val MODERATE_MOVEMENT_THRESHOLD = 3.0f
        private const val SIGNIFICANT_MOVEMENT_THRESHOLD = 5.0f
        private const val MAJOR_MOVEMENT_THRESHOLD = 8.0f

        // Sleep phase detection thresholds
        private const val AWAKE_MOVEMENT_THRESHOLD = 4.0f
        private const val LIGHT_SLEEP_MOVEMENT_THRESHOLD = 2.0f
        private const val DEEP_SLEEP_MOVEMENT_THRESHOLD = 1.0f
        private const val REM_MOVEMENT_MIN = 1.5f
        private const val REM_MOVEMENT_MAX = 3.5f

        // Temporal analysis windows (in milliseconds)
        private const val SHORT_WINDOW = 2 * 60 * 1000L // 2 minutes
        private const val MEDIUM_WINDOW = 10 * 60 * 1000L // 10 minutes
        private const val LONG_WINDOW = 30 * 60 * 1000L // 30 minutes

        // Clustering parameters
        private const val CLUSTER_TIME_THRESHOLD = 5 * 60 * 1000L // 5 minutes
        private const val CLUSTER_INTENSITY_THRESHOLD = 1.0f
        private const val MIN_CLUSTER_SIZE = 3

        // Restlessness scoring weights
        private const val FREQUENCY_WEIGHT = 0.35f
        private const val INTENSITY_WEIGHT = 0.30f
        private const val DISTRIBUTION_WEIGHT = 0.20f
        private const val CONSISTENCY_WEIGHT = 0.15f
    }

    // ========== MOVEMENT INTENSITY ANALYSIS ==========

    /**
     * Comprehensive movement intensity analysis with adaptive algorithms
     */
    suspend fun analyzeMovementIntensity(
        movements: List<MovementEvent>,
        sessionDuration: Long
    ): MovementIntensityAnalysis = withContext(Dispatchers.Default) {
        try {
            if (movements.isEmpty()) {
                return@withContext MovementIntensityAnalysis()
            }

            // Basic intensity metrics
            val intensities = movements.map { it.intensity }
            val averageIntensity = intensities.average().toFloat()
            val maxIntensity = intensities.maxOrNull() ?: 0f
            val minIntensity = intensities.minOrNull() ?: 0f
            val intensityVariance = calculateVariance(intensities)

            // Movement frequency analysis
            val movementFrequency = if (sessionDuration > 0) {
                (movements.size.toFloat() / sessionDuration) * 3600000f // per hour
            } else 0f

            // Movement classification
            val movementClassification = classifyMovements(movements)

            // Temporal intensity patterns
            val temporalPatterns = analyzeTemporalIntensityPatterns(movements, sessionDuration)

            // Intensity distribution analysis
            val distribution = analyzeIntensityDistribution(movements)

            // Movement bursts detection
            val bursts = detectMovementBursts(movements)

            // Adaptive threshold calculation
            val adaptiveThresholds = calculateAdaptiveThresholds(movements)

            MovementIntensityAnalysis(
                averageIntensity = averageIntensity,
                maxIntensity = maxIntensity,
                minIntensity = minIntensity,
                intensityVariance = intensityVariance,
                movementFrequency = movementFrequency,
                classification = movementClassification,
                temporalPatterns = temporalPatterns,
                distribution = distribution,
                bursts = bursts,
                adaptiveThresholds = adaptiveThresholds,
                intensityGrade = getIntensityGrade(averageIntensity),
                stabilityScore = calculateStabilityScore(intensityVariance, movementFrequency),
                recommendations = generateIntensityRecommendations(averageIntensity, movementFrequency, bursts.size)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing movement intensity", e)
            MovementIntensityAnalysis()
        }
    }

    /**
     * Real-time movement intensity calculation with smoothing
     */
    suspend fun calculateRealTimeIntensity(
        recentMovements: List<MovementEvent>,
        windowSize: Long = SHORT_WINDOW
    ): RealTimeIntensityData = withContext(Dispatchers.Default) {
        try {
            if (recentMovements.isEmpty()) {
                return@withContext RealTimeIntensityData()
            }

            val currentTime = System.currentTimeMillis()
            val windowStart = currentTime - windowSize

            // Filter movements within window
            val windowMovements = recentMovements.filter { it.timestamp >= windowStart }

            if (windowMovements.isEmpty()) {
                return@withContext RealTimeIntensityData()
            }

            // Calculate raw intensity
            val rawIntensity = windowMovements.map { it.intensity }.average().toFloat()

            // Apply smoothing algorithms
            val smoothedIntensity = applySmoothingFilter(windowMovements)
            val exponentialSmoothed = applyExponentialSmoothing(windowMovements)

            // Detect intensity trends
            val trend = detectIntensityTrend(windowMovements)

            // Movement activity level
            val activityLevel = determineActivityLevel(smoothedIntensity)

            // Predict next phase based on current pattern
            val phasePredicteion = predictSleepPhaseFromMovement(windowMovements)

            RealTimeIntensityData(
                rawIntensity = rawIntensity,
                smoothedIntensity = smoothedIntensity,
                exponentialSmoothed = exponentialSmoothed,
                trend = trend,
                activityLevel = activityLevel,
                predictedPhase = phasePredicteion,
                confidence = calculatePredictionConfidence(windowMovements),
                timestamp = currentTime
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error calculating real-time intensity", e)
            RealTimeIntensityData()
        }
    }

    // ========== RESTLESSNESS SCORING ==========

    /**
     * Multi-dimensional restlessness analysis with temporal components
     */
    suspend fun analyzeRestlessness(
        movements: List<MovementEvent>,
        sessionDuration: Long
    ): RestlessnessAnalysis = withContext(Dispatchers.Default) {
        try {
            if (movements.isEmpty() || sessionDuration == 0L) {
                return@withContext RestlessnessAnalysis()
            }

            // Frequency-based restlessness
            val frequencyScore = calculateFrequencyRestlessness(movements, sessionDuration)

            // Intensity-based restlessness
            val intensityScore = calculateIntensityRestlessness(movements)

            // Distribution-based restlessness (how evenly spread movements are)
            val distributionScore = calculateDistributionRestlessness(movements, sessionDuration)

            // Consistency-based restlessness (pattern regularity)
            val consistencyScore = calculateConsistencyRestlessness(movements)

            // Weighted overall restlessness score (0-10 scale)
            val overallScore = (
                    frequencyScore * FREQUENCY_WEIGHT +
                            intensityScore * INTENSITY_WEIGHT +
                            distributionScore * DISTRIBUTION_WEIGHT +
                            consistencyScore * CONSISTENCY_WEIGHT
                    ).coerceIn(0f, 10f)

            // Temporal restlessness analysis
            val temporalAnalysis = analyzeTemporalRestlessness(movements, sessionDuration)

            // Movement clustering for restlessness
            val clusters = clusterMovementsForRestlessness(movements)

            // Sleep disruption analysis
            val disruptionAnalysis = analyzeSleepDisruption(movements, sessionDuration)

            RestlessnessAnalysis(
                overallScore = overallScore,
                frequencyScore = frequencyScore,
                intensityScore = intensityScore,
                distributionScore = distributionScore,
                consistencyScore = consistencyScore,
                temporalAnalysis = temporalAnalysis,
                movementClusters = clusters,
                disruptionAnalysis = disruptionAnalysis,
                restlessnessGrade = getRestlessnessGrade(overallScore),
                primaryCause = identifyPrimaryRestlessnessCause(frequencyScore, intensityScore, distributionScore, consistencyScore),
                recommendations = generateRestlessnessRecommendations(overallScore, frequencyScore, intensityScore)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing restlessness", e)
            RestlessnessAnalysis()
        }
    }

    // ========== SLEEP PHASE DETECTION ==========

    /**
     * Advanced sleep phase detection based on movement patterns
     */
    suspend fun detectSleepPhase(
        movements: List<MovementEvent>,
        timeWindow: Long = MEDIUM_WINDOW,
        currentTime: Long = System.currentTimeMillis()
    ): SleepPhaseDetection = withContext(Dispatchers.Default) {
        try {
            val windowStart = currentTime - timeWindow
            val windowMovements = movements.filter { it.timestamp >= windowStart }

            if (windowMovements.isEmpty()) {
                return@withContext SleepPhaseDetection(
                    detectedPhase = SleepPhase.UNKNOWN,
                    confidence = 0f
                )
            }

            // Calculate movement metrics for phase detection
            val avgIntensity = windowMovements.map { it.intensity }.average().toFloat()
            val movementCount = windowMovements.size
            val movementRate = (movementCount.toFloat() / timeWindow) * 60000f // per minute
            val intensityVariability = calculateVariance(windowMovements.map { it.intensity })

            // Phase detection algorithms
            val phaseScores = calculatePhaseScores(avgIntensity, movementRate, intensityVariability, windowMovements)

            // Determine most likely phase
            val detectedPhase = phaseScores.maxByOrNull { it.value }?.key ?: SleepPhase.UNKNOWN
            val confidence = phaseScores[detectedPhase] ?: 0f

            // Additional analysis
            val transitionProbabilities = calculateTransitionProbabilities(windowMovements)
            val stabilityMetrics = calculatePhaseStability(windowMovements)

            SleepPhaseDetection(
                detectedPhase = detectedPhase,
                confidence = confidence,
                phaseScores = phaseScores,
                transitionProbabilities = transitionProbabilities,
                stabilityMetrics = stabilityMetrics,
                analysisWindow = timeWindow,
                movementCount = movementCount,
                averageIntensity = avgIntensity,
                algorithm = "Movement Pattern Analysis v2.0"
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error detecting sleep phase", e)
            SleepPhaseDetection(detectedPhase = SleepPhase.UNKNOWN, confidence = 0f)
        }
    }

    /**
     * Track sleep phase transitions over time
     */
    suspend fun trackPhaseTransitions(
        movements: List<MovementEvent>,
        sessionDuration: Long
    ): PhaseTransitionAnalysis = withContext(Dispatchers.Default) {
        try {
            val phaseDetections = mutableListOf<TimestampedPhaseDetection>()
            val stepSize = MEDIUM_WINDOW / 2 // 50% overlap

            var currentTime = movements.firstOrNull()?.timestamp ?: System.currentTimeMillis()
            val endTime = movements.lastOrNull()?.timestamp ?: currentTime

            // Analyze phases throughout the session
            while (currentTime <= endTime) {
                val detection = detectSleepPhase(movements, MEDIUM_WINDOW, currentTime)
                phaseDetections.add(
                    TimestampedPhaseDetection(
                        timestamp = currentTime,
                        detection = detection
                    )
                )
                currentTime += stepSize
            }

            // Analyze transitions
            val transitions = mutableListOf<PhaseTransition>()
            var currentPhase: SleepPhase? = null

            for (detection in phaseDetections) {
                if (detection.detection.confidence > 0.6f) { // Only high-confidence detections
                    if (currentPhase != null && currentPhase != detection.detection.detectedPhase) {
                        transitions.add(
                            PhaseTransition(
                                timestamp = detection.timestamp,
                                fromPhase = currentPhase,
                                toPhase = detection.detection.detectedPhase,
                                confidence = detection.detection.confidence
                            )
                        )
                    }
                    currentPhase = detection.detection.detectedPhase
                }
            }

            // Calculate phase durations
            val phaseDurations = calculatePhaseDurations(transitions, movements.firstOrNull()?.timestamp ?: 0L, endTime)

            // Transition quality analysis
            val transitionQuality = analyzeTransitionQuality(transitions)

            PhaseTransitionAnalysis(
                transitions = transitions,
                phaseDurations = phaseDurations,
                transitionQuality = transitionQuality,
                totalTransitions = transitions.size,
                averagePhaseLength = if (transitions.isNotEmpty()) sessionDuration / (transitions.size + 1) else sessionDuration,
                stabilityScore = calculateTransitionStabilityScore(transitions),
                recommendations = generatePhaseRecommendations(transitions, phaseDurations)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error tracking phase transitions", e)
            PhaseTransitionAnalysis()
        }
    }

    // ========== MOVEMENT CLUSTERING AND FILTERING ==========

    /**
     * Advanced movement event clustering with pattern recognition
     */
    suspend fun clusterMovements(
        movements: List<MovementEvent>,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.TEMPORAL_INTENSITY
    ): MovementClusterAnalysis = withContext(Dispatchers.Default) {
        try {
            if (movements.isEmpty()) {
                return@withContext MovementClusterAnalysis()
            }

            val clusters = when (algorithm) {
                ClusteringAlgorithm.TEMPORAL_INTENSITY -> performTemporalIntensityClustering(movements)
                ClusteringAlgorithm.SPATIAL_PATTERN -> performSpatialPatternClustering(movements)
                ClusteringAlgorithm.HYBRID -> performHybridClustering(movements)
            }

            // Analyze cluster characteristics
            val clusterAnalysis = analyzeClusters(clusters)

            // Identify cluster patterns
            val patterns = identifyClusterPatterns(clusters)

            // Filter noise and artifacts
            val filteredClusters = filterNoiseClusters(clusters)

            // Generate cluster insights
            val insights = generateClusterInsights(clusters, patterns)

            MovementClusterAnalysis(
                clusters = filteredClusters,
                totalClusters = clusters.size,
                filteredClusters = filteredClusters.size,
                patterns = patterns,
                insights = insights,
                algorithm = algorithm,
                clusterQuality = calculateClusterQuality(clusters),
                recommendations = generateClusterRecommendations(patterns, insights)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error clustering movements", e)
            MovementClusterAnalysis()
        }
    }

    /**
     * Real-time movement filtering and noise reduction
     */
    suspend fun filterMovements(
        movements: List<MovementEvent>,
        filterType: MovementFilter = MovementFilter.ADAPTIVE
    ): MovementFilterResult = withContext(Dispatchers.Default) {
        try {
            val filteredMovements = when (filterType) {
                MovementFilter.BASIC -> applyBasicFilter(movements)
                MovementFilter.ADAPTIVE -> applyAdaptiveFilter(movements)
                MovementFilter.STATISTICAL -> applyStatisticalFilter(movements)
                MovementFilter.MACHINE_LEARNING -> applyMLInspiredFilter(movements)
            }

            // Calculate filter effectiveness
            val noiseReduction = calculateNoiseReduction(movements, filteredMovements)
            val signalPreservation = calculateSignalPreservation(movements, filteredMovements)

            MovementFilterResult(
                originalMovements = movements,
                filteredMovements = filteredMovements,
                filterType = filterType,
                noiseReduction = noiseReduction,
                signalPreservation = signalPreservation,
                filterEffectiveness = (noiseReduction + signalPreservation) / 2f,
                removedCount = movements.size - filteredMovements.size,
                retainedCount = filteredMovements.size
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error filtering movements", e)
            MovementFilterResult(
                originalMovements = movements,
                filteredMovements = movements,
                filterType = filterType
            )
        }
    }

    // ========== HELPER METHODS - INTENSITY ANALYSIS ==========

    private fun classifyMovements(movements: List<MovementEvent>): MovementClassification {
        val microMovements = movements.count { it.intensity <= MICRO_MOVEMENT_THRESHOLD }
        val lightMovements = movements.count { it.intensity > MICRO_MOVEMENT_THRESHOLD && it.intensity <= LIGHT_MOVEMENT_THRESHOLD }
        val moderateMovements = movements.count { it.intensity > LIGHT_MOVEMENT_THRESHOLD && it.intensity <= MODERATE_MOVEMENT_THRESHOLD }
        val significantMovements = movements.count { it.intensity > MODERATE_MOVEMENT_THRESHOLD && it.intensity <= SIGNIFICANT_MOVEMENT_THRESHOLD }
        val majorMovements = movements.count { it.intensity > SIGNIFICANT_MOVEMENT_THRESHOLD }

        return MovementClassification(
            microMovements = microMovements,
            lightMovements = lightMovements,
            moderateMovements = moderateMovements,
            significantMovements = significantMovements,
            majorMovements = majorMovements,
            dominantType = getDominantMovementType(microMovements, lightMovements, moderateMovements, significantMovements, majorMovements)
        )
    }

    private fun analyzeTemporalIntensityPatterns(
        movements: List<MovementEvent>,
        sessionDuration: Long
    ): TemporalIntensityPattern {
        if (movements.isEmpty()) return TemporalIntensityPattern()

        val sessionStart = movements.minByOrNull { it.timestamp }?.timestamp ?: 0L
        val sessionEnd = sessionStart + sessionDuration

        // Divide session into periods
        val periodCount = 6 // 6 periods for analysis
        val periodDuration = sessionDuration / periodCount

        val periodIntensities = mutableListOf<Float>()

        for (i in 0 until periodCount) {
            val periodStart = sessionStart + (i * periodDuration)
            val periodEnd = periodStart + periodDuration

            val periodMovements = movements.filter { it.timestamp >= periodStart && it.timestamp < periodEnd }
            val avgIntensity = if (periodMovements.isNotEmpty()) {
                periodMovements.map { it.intensity }.average().toFloat()
            } else 0f

            periodIntensities.add(avgIntensity)
        }

        // Analyze pattern
        val trend = if (periodIntensities.size >= 3) {
            val firstThird = periodIntensities.take(2).average()
            val lastThird = periodIntensities.takeLast(2).average()
            when {
                lastThird > firstThird + 0.5f -> IntensityTrend.INCREASING
                lastThird < firstThird - 0.5f -> IntensityTrend.DECREASING
                else -> IntensityTrend.STABLE
            }
        } else IntensityTrend.INSUFFICIENT_DATA

        return TemporalIntensityPattern(
            periodIntensities = periodIntensities,
            trend = trend,
            peakPeriod = periodIntensities.indexOf(periodIntensities.maxOrNull() ?: 0f),
            calmestPeriod = periodIntensities.indexOf(periodIntensities.minOrNull() ?: 0f),
            variability = calculateVariance(periodIntensities)
        )
    }

    private fun analyzeIntensityDistribution(movements: List<MovementEvent>): IntensityDistribution {
        val intensities = movements.map { it.intensity }

        return IntensityDistribution(
            mean = intensities.average().toFloat(),
            median = calculateMedian(intensities),
            standardDeviation = sqrt(calculateVariance(intensities)),
            skewness = calculateSkewness(intensities),
            kurtosis = calculateKurtosis(intensities),
            percentile25 = calculatePercentile(intensities, 25),
            percentile75 = calculatePercentile(intensities, 75),
            interquartileRange = calculatePercentile(intensities, 75) - calculatePercentile(intensities, 25)
        )
    }

    private fun detectMovementBursts(movements: List<MovementEvent>): List<MovementBurst> {
        val bursts = mutableListOf<MovementBurst>()
        var currentBurst: MutableList<MovementEvent>? = null
        var lastTimestamp = 0L

        for (movement in movements.sortedBy { it.timestamp }) {
            val isSignificant = movement.intensity > MODERATE_MOVEMENT_THRESHOLD
            val isConsecutive = movement.timestamp - lastTimestamp <= CLUSTER_TIME_THRESHOLD

            when {
                isSignificant && (currentBurst == null || !isConsecutive) -> {
                    // Start new burst
                    currentBurst?.let { burst ->
                        if (burst.size >= MIN_CLUSTER_SIZE) {
                            bursts.add(createMovementBurst(burst))
                        }
                    }
                    currentBurst = mutableListOf(movement)
                }
                isSignificant && isConsecutive -> {
                    // Continue current burst
                    currentBurst?.add(movement)
                }
                !isSignificant && currentBurst != null -> {
                    // End current burst
                    if (currentBurst.size >= MIN_CLUSTER_SIZE) {
                        bursts.add(createMovementBurst(currentBurst))
                    }
                    currentBurst = null
                }
            }
            lastTimestamp = movement.timestamp
        }

        // Handle final burst
        currentBurst?.let { burst ->
            if (burst.size >= MIN_CLUSTER_SIZE) {
                bursts.add(createMovementBurst(burst))
            }
        }

        return bursts
    }

    private fun calculateAdaptiveThresholds(movements: List<MovementEvent>): AdaptiveThresholds {
        val intensities = movements.map { it.intensity }
        val mean = intensities.average().toFloat()
        val stdDev = sqrt(calculateVariance(intensities))

        return AdaptiveThresholds(
            microThreshold = maxOf(MICRO_MOVEMENT_THRESHOLD, mean - 2 * stdDev),
            lightThreshold = maxOf(LIGHT_MOVEMENT_THRESHOLD, mean - stdDev),
            moderateThreshold = maxOf(MODERATE_MOVEMENT_THRESHOLD, mean),
            significantThreshold = maxOf(SIGNIFICANT_MOVEMENT_THRESHOLD, mean + stdDev),
            majorThreshold = maxOf(MAJOR_MOVEMENT_THRESHOLD, mean + 2 * stdDev)
        )
    }

    // ========== HELPER METHODS - REAL-TIME ANALYSIS ==========

    private fun applySmoothingFilter(movements: List<MovementEvent>): Float {
        if (movements.isEmpty()) return 0f

        // Moving average with exponential weighting
        val weights = movements.mapIndexed { index, _ ->
            exp(-0.1 * (movements.size - index - 1))
        }

        val weightedSum = movements.zip(weights) { movement, weight ->
            movement.intensity * weight
        }.sum()

        return weightedSum / weights.sum()
    }

    private fun applyExponentialSmoothing(movements: List<MovementEvent>): Float {
        if (movements.isEmpty()) return 0f

        val alpha = 0.3f // Smoothing factor
        var smoothed = movements.first().intensity

        for (i in 1 until movements.size) {
            smoothed = alpha * movements[i].intensity + (1 - alpha) * smoothed
        }

        return smoothed
    }

    private fun detectIntensityTrend(movements: List<MovementEvent>): IntensityTrend {
        if (movements.size < 3) return IntensityTrend.INSUFFICIENT_DATA

        val recentAvg = movements.takeLast(3).map { it.intensity }.average()
        val olderAvg = movements.dropLast(3).takeLast(3).map { it.intensity }.average()

        return when {
            recentAvg > olderAvg + 0.5 -> IntensityTrend.INCREASING
            recentAvg < olderAvg - 0.5 -> IntensityTrend.DECREASING
            else -> IntensityTrend.STABLE
        }
    }

    private fun determineActivityLevel(intensity: Float): ActivityLevel {
        return when {
            intensity <= MICRO_MOVEMENT_THRESHOLD -> ActivityLevel.MINIMAL
            intensity <= LIGHT_MOVEMENT_THRESHOLD -> ActivityLevel.LOW
            intensity <= MODERATE_MOVEMENT_THRESHOLD -> ActivityLevel.MODERATE
            intensity <= SIGNIFICANT_MOVEMENT_THRESHOLD -> ActivityLevel.HIGH
            else -> ActivityLevel.VERY_HIGH
        }
    }

    private fun predictSleepPhaseFromMovement(movements: List<MovementEvent>): SleepPhase {
        if (movements.isEmpty()) return SleepPhase.UNKNOWN

        val avgIntensity = movements.map { it.intensity }.average().toFloat()
        val movementRate = movements.size.toFloat() / SHORT_WINDOW * 60000f // per minute

        return when {
            avgIntensity > AWAKE_MOVEMENT_THRESHOLD -> SleepPhase.AWAKE
            avgIntensity <= DEEP_SLEEP_MOVEMENT_THRESHOLD && movementRate < 0.5f -> SleepPhase.DEEP_SLEEP
            avgIntensity in REM_MOVEMENT_MIN..REM_MOVEMENT_MAX && movementRate > 0.3f -> SleepPhase.REM_SLEEP
            avgIntensity <= LIGHT_SLEEP_MOVEMENT_THRESHOLD -> SleepPhase.LIGHT_SLEEP
            else -> SleepPhase.UNKNOWN
        }
    }

    private fun calculatePredictionConfidence(movements: List<MovementEvent>): Float {
        if (movements.isEmpty()) return 0f

        val intensityVariance = calculateVariance(movements.map { it.intensity })
        val consistency = 1f / (1f + intensityVariance)
        val sampleSize = minOf(movements.size / 10f, 1f) // More samples = higher confidence

        return (consistency * sampleSize * 100f).coerceIn(0f, 100f)
    }

    // ========== HELPER METHODS - RESTLESSNESS ==========

    private fun calculateFrequencyRestlessness(movements: List<MovementEvent>, sessionDuration: Long): Float {
        val movementRate = (movements.size.toFloat() / sessionDuration) * 3600000f // per hour

        return when {
            movementRate <= 10f -> 1f
            movementRate <= 20f -> 2f + (movementRate - 10f) / 10f * 2f
            movementRate <= 40f -> 4f + (movementRate - 20f) / 20f * 3f
            movementRate <= 80f -> 7f + (movementRate - 40f) / 40f * 2f
            else -> 9f + minOf((movementRate - 80f) / 40f, 1f)
        }.coerceIn(0f, 10f)
    }

    private fun calculateIntensityRestlessness(movements: List<MovementEvent>): Float {
        if (movements.isEmpty()) return 0f

        val avgIntensity = movements.map { it.intensity }.average().toFloat()
        val significantMovements = movements.count { it.intensity > MODERATE_MOVEMENT_THRESHOLD }
        val significantRatio = significantMovements.toFloat() / movements.size

        val intensityScore = (avgIntensity / MAJOR_MOVEMENT_THRESHOLD * 5f).coerceIn(0f, 5f)
        val significantScore = (significantRatio * 5f).coerceIn(0f, 5f)

        return intensityScore + significantScore
    }

    private fun calculateDistributionRestlessness(movements: List<MovementEvent>, sessionDuration: Long): Float {
        if (movements.isEmpty() || sessionDuration == 0L) return 0f

        // Divide session into time buckets and analyze distribution
        val bucketCount = 10
        val bucketDuration = sessionDuration / bucketCount
        val sessionStart = movements.minByOrNull { it.timestamp }?.timestamp ?: 0L

        val bucketCounts = IntArray(bucketCount)

        for (movement in movements) {
            val bucketIndex = ((movement.timestamp - sessionStart) / bucketDuration).toInt()
                .coerceIn(0, bucketCount - 1)
            bucketCounts[bucketIndex]++
        }

        // Calculate distribution evenness (higher evenness = higher restlessness)
        val avgBucketCount = bucketCounts.average()
        val variance = bucketCounts.map { (it - avgBucketCount).pow(2) }.average()
        val unevenness = sqrt(variance) / avgBucketCount

        // Convert to 0-10 scale (more uneven = less restless in terms of distribution)
        return (10f - (unevenness * 5f)).coerceIn(0f, 10f)
    }

    private fun calculateConsistencyRestlessness(movements: List<MovementEvent>): Float {
        if (movements.size < 3) return 5f // Neutral score for insufficient data

        // Analyze intensity consistency over time
        val intensities = movements.map { it.intensity }
        val variance = calculateVariance(intensities)
        val coefficientOfVariation = sqrt(variance) / intensities.average()

        // Higher variation = higher restlessness
        return (coefficientOfVariation * 10f).coerceIn(0f, 10f)
    }

    private fun analyzeTemporalRestlessness(
        movements: List<MovementEvent>,
        sessionDuration: Long
    ): TemporalRestlessnessAnalysis {
        // Analyze restlessness patterns over time periods
        val hourlyRestlessness = calculateHourlyRestlessness(movements, sessionDuration)
        val peakRestlessnessHour = hourlyRestlessness.indexOf(hourlyRestlessness.maxOrNull() ?: 0f)
        val calmestHour = hourlyRestlessness.indexOf(hourlyRestlessness.minOrNull() ?: 0f)

        return TemporalRestlessnessAnalysis(
            hourlyScores = hourlyRestlessness,
            peakRestlessnessHour = peakRestlessnessHour,
            calmestHour = calmestHour,
            restlessnessVariation = calculateVariance(hourlyRestlessness)
        )
    }

    private fun clusterMovementsForRestlessness(movements: List<MovementEvent>): List<RestlessnessCluster> {
        // Group movements into restlessness episodes
        val clusters = mutableListOf<RestlessnessCluster>()
        var currentCluster = mutableListOf<MovementEvent>()
        var lastTimestamp = 0L

        for (movement in movements.sortedBy { it.timestamp }) {
            val isRestless = movement.intensity > LIGHT_MOVEMENT_THRESHOLD
            val isConsecutive = movement.timestamp - lastTimestamp <= CLUSTER_TIME_THRESHOLD

            when {
                isRestless && (currentCluster.isEmpty() || isConsecutive) -> {
                    currentCluster.add(movement)
                }
                isRestless && !isConsecutive -> {
                    if (currentCluster.size >= MIN_CLUSTER_SIZE) {
                        clusters.add(createRestlessnessCluster(currentCluster))
                    }
                    currentCluster = mutableListOf(movement)
                }
                !isRestless && currentCluster.isNotEmpty() -> {
                    if (currentCluster.size >= MIN_CLUSTER_SIZE) {
                        clusters.add(createRestlessnessCluster(currentCluster))
                    }
                    currentCluster = mutableListOf()
                }
            }
            lastTimestamp = movement.timestamp
        }

        // Handle final cluster
        if (currentCluster.size >= MIN_CLUSTER_SIZE) {
            clusters.add(createRestlessnessCluster(currentCluster))
        }

        return clusters
    }

    private fun analyzeSleepDisruption(movements: List<MovementEvent>, sessionDuration: Long): SleepDisruptionAnalysis {
        val majorMovements = movements.filter { it.intensity > SIGNIFICANT_MOVEMENT_THRESHOLD }
        val disruptionEvents = detectDisruptionEvents(movements)

        return SleepDisruptionAnalysis(
            totalDisruptions = disruptionEvents.size,
            majorMovements = majorMovements.size,
            disruptionRate = (disruptionEvents.size.toFloat() / sessionDuration) * 3600000f, // per hour
            averageDisruptionIntensity = if (disruptionEvents.isNotEmpty()) {
                disruptionEvents.map { it.intensity }.average().toFloat()
            } else 0f,
            longestCalm�Period = findLongestCalmPeriod(movements),
        disruptionImpact = calculateDisruptionImpact(disruptionEvents, sessionDuration)
        )
    }

    // ========== HELPER METHODS - PHASE DETECTION ==========

    private fun calculatePhaseScores(
        avgIntensity: Float,
        movementRate: Float,
        intensityVariability: Float,
        movements: List<MovementEvent>
    ): Map<SleepPhase, Float> {
        val scores = mutableMapOf<SleepPhase, Float>()

        // Awake scoring
        scores[SleepPhase.AWAKE] = when {
            avgIntensity > AWAKE_MOVEMENT_THRESHOLD -> 0.8f + minOf(0.2f, (avgIntensity - AWAKE_MOVEMENT_THRESHOLD) / 5f)
            movementRate > 2f -> 0.6f + minOf(0.3f, movementRate / 10f)
            else -> maxOf(0f, 0.4f - (AWAKE_MOVEMENT_THRESHOLD - avgIntensity) / 2f)
        }

        // Light sleep scoring
        scores[SleepPhase.LIGHT_SLEEP] = when {
            avgIntensity <= LIGHT_SLEEP_MOVEMENT_THRESHOLD && movementRate < 1.5f ->
                0.7f + (0.3f * (1f - avgIntensity / LIGHT_SLEEP_MOVEMENT_THRESHOLD))
            avgIntensity <= MODERATE_MOVEMENT_THRESHOLD ->
                0.5f - abs(avgIntensity - LIGHT_SLEEP_MOVEMENT_THRESHOLD) / 2f
            else -> maxOf(0f, 0.3f - (avgIntensity - MODERATE_MOVEMENT_THRESHOLD) / 3f)
        }

        // Deep sleep scoring
        scores[SleepPhase.DEEP_SLEEP] = when {
            avgIntensity <= DEEP_SLEEP_MOVEMENT_THRESHOLD && movementRate < 0.5f ->
                0.9f + (0.1f * (1f - avgIntensity / DEEP_SLEEP_MOVEMENT_THRESHOLD))
            avgIntensity <= LIGHT_MOVEMENT_THRESHOLD ->
                0.4f - (avgIntensity - DEEP_SLEEP_MOVEMENT_THRESHOLD) / 2f
            else -> maxOf(0f, 0.2f - (avgIntensity - LIGHT_MOVEMENT_THRESHOLD) / 4f)
        }

        // REM sleep scoring (characteristic pattern of moderate, variable movement)
        scores[SleepPhase.REM_SLEEP] = when {
            avgIntensity in REM_MOVEMENT_MIN..REM_MOVEMENT_MAX && intensityVariability > 0.5f ->
                0.8f + minOf(0.2f, intensityVariability / 2f)
            movementRate > 0.3f && movementRate < 2f ->
                0.6f - abs(movementRate - 1f) / 2f
            else -> maxOf(0f, 0.3f - abs(avgIntensity - (REM_MOVEMENT_MIN + REM_MOVEMENT_MAX) / 2f) / 2f)
        }

        // Normalize scores to sum to 1.0
        val totalScore = scores.values.sum()
        if (totalScore > 0) {
            scores.replaceAll { _, score -> score / totalScore }
        }

        return scores
    }

    private fun calculateTransitionProbabilities(movements: List<MovementEvent>): Map<SleepPhase, Float> {
        // Simplified transition probability calculation
        // In a full implementation, this would use historical data and Markov chains
        return mapOf(
            SleepPhase.AWAKE to 0.1f,
            SleepPhase.LIGHT_SLEEP to 0.4f,
            SleepPhase.DEEP_SLEEP to 0.3f,
            SleepPhase.REM_SLEEP to 0.2f
        )
    }

    private fun calculatePhaseStability(movements: List<MovementEvent>): PhaseStabilityMetrics {
        val intensities = movements.map { it.intensity }
        val variance = calculateVariance(intensities)
        val trendStrength = calculateTrendStrength(intensities)

        return PhaseStabilityMetrics(
            varianceScore = 1f / (1f + variance), // Lower variance = higher stability
            trendStability = 1f - abs(trendStrength), // Lower trend = more stable
            overallStability = (1f / (1f + variance) + (1f - abs(trendStrength))) / 2f
        )
    }

    private fun calculatePhaseDurations(
        transitions: List<PhaseTransition>,
        sessionStart: Long,
        sessionEnd: Long
    ): Map<SleepPhase, Long> {
        val durations = mutableMapOf<SleepPhase, Long>()

        if (transitions.isEmpty()) {
            durations[SleepPhase.UNKNOWN] = sessionEnd - sessionStart
            return durations
        }

        val sortedTransitions = transitions.sortedBy { it.timestamp }
        var currentTime = sessionStart
        var currentPhase = SleepPhase.AWAKE // Assume starting awake

        for (transition in sortedTransitions) {
            val phaseDuration = transition.timestamp - currentTime
            durations[currentPhase] = (durations[currentPhase] ?: 0L) + phaseDuration
            currentTime = transition.timestamp
            currentPhase = transition.toPhase
        }

        // Add final phase duration
        val finalDuration = sessionEnd - currentTime
        durations[currentPhase] = (durations[currentPhase] ?: 0L) + finalDuration

        return durations
    }

    // ========== HELPER METHODS - CLUSTERING ==========

    private fun performTemporalIntensityClustering(movements: List<MovementEvent>): List<MovementCluster> {
        val clusters = mutableListOf<MovementCluster>()
        val sortedMovements = movements.sortedBy { it.timestamp }

        var currentCluster = mutableListOf<MovementEvent>()
        var lastTimestamp = 0L
        var lastIntensity = 0f

        for (movement in sortedMovements) {
            val timeDiff = movement.timestamp - lastTimestamp
            val intensityDiff = abs(movement.intensity - lastIntensity)

            val isTemporallyClose = timeDiff <= CLUSTER_TIME_THRESHOLD
            val isIntensitySimilar = intensityDiff <= CLUSTER_INTENSITY_THRESHOLD

            if (currentCluster.isEmpty() || (!isTemporallyClose && !isIntensitySimilar)) {
                // Start new cluster
                if (currentCluster.size >= MIN_CLUSTER_SIZE) {
                    clusters.add(createMovementCluster(currentCluster, ClusterType.TEMPORAL_INTENSITY))
                }
                currentCluster = mutableListOf(movement)
            } else {
                // Add to current cluster
                currentCluster.add(movement)
            }

            lastTimestamp = movement.timestamp
            lastIntensity = movement.intensity
        }

        // Handle final cluster
        if (currentCluster.size >= MIN_CLUSTER_SIZE) {
            clusters.add(createMovementCluster(currentCluster, ClusterType.TEMPORAL_INTENSITY))
        }

        return clusters
    }

    private fun performSpatialPatternClustering(movements: List<MovementEvent>): List<MovementCluster> {
        // Cluster based on movement direction and magnitude patterns
        val clusters = mutableListOf<MovementCluster>()

        // Implementation would analyze x, y, z patterns
        // For now, simplified clustering based on magnitude similarity

        return clusters
    }

    private fun performHybridClustering(movements: List<MovementEvent>): List<MovementCluster> {
        // Combine temporal, intensity, and spatial clustering
        val temporalClusters = performTemporalIntensityClustering(movements)
        val spatialClusters = performSpatialPatternClustering(movements)

        // Merge and optimize clusters
        return mergeClusters(temporalClusters, spatialClusters)
    }

    // ========== UTILITY METHODS ==========

    private fun calculateVariance(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val mean = values.average()
        return values.map { (it - mean).pow(2) }.average().toFloat()
    }

    private fun calculateMedian(values: List<Float>): Float {
        val sorted = values.sorted()
        return if (sorted.size % 2 == 0) {
            (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2f
        } else {
            sorted[sorted.size / 2]
        }
    }

    private fun calculateSkewness(values: List<Float>): Float {
        if (values.size < 3) return 0f
        val mean = values.average().toFloat()
        val variance = calculateVariance(values)
        val stdDev = sqrt(variance)

        if (stdDev == 0f) return 0f

        val skewness = values.map { ((it - mean) / stdDev).pow(3) }.average()
        return skewness.toFloat()
    }

    private fun calculateKurtosis(values: List<Float>): Float {
        if (values.size < 4) return 0f
        val mean = values.average().toFloat()
        val variance = calculateVariance(values)
        val stdDev = sqrt(variance)

        if (stdDev == 0f) return 0f

        val kurtosis = values.map { ((it - mean) / stdDev).pow(4) }.average() - 3
        return kurtosis.toFloat()
    }

    private fun calculatePercentile(values: List<Float>, percentile: Int): Float {
        val sorted = values.sorted()
        val index = (percentile / 100.0 * (sorted.size - 1)).toInt()
        return sorted.getOrElse(index) { 0f }
    }

    private fun getIntensityGrade(intensity: Float): String {
        return when {
            intensity <= MICRO_MOVEMENT_THRESHOLD -> "Excellent (Very Still)"
            intensity <= LIGHT_MOVEMENT_THRESHOLD -> "Very Good (Minimal Movement)"
            intensity <= MODERATE_MOVEMENT_THRESHOLD -> "Good (Light Movement)"
            intensity <= SIGNIFICANT_MOVEMENT_THRESHOLD -> "Fair (Moderate Movement)"
            else -> "Poor (High Movement)"
        }
    }

    private fun calculateStabilityScore(variance: Float, frequency: Float): Float {
        val varianceScore = 1f / (1f + variance) // Lower variance = higher stability
        val frequencyScore = 1f / (1f + frequency / 10f) // Lower frequency = higher stability
        return ((varianceScore + frequencyScore) / 2f * 10f).coerceIn(0f, 10f)
    }

    private fun generateIntensityRecommendations(intensity: Float, frequency: Float, burstCount: Int): List<String> {
        val recommendations = mutableListOf<String>()

        if (intensity > MODERATE_MOVEMENT_THRESHOLD) {
            recommendations.add("Consider relaxation techniques before bed to reduce movement intensity")
        }

        if (frequency > 30f) { // 30 movements per hour
            recommendations.add("High movement frequency detected - review sleep environment for comfort")
        }

        if (burstCount > 5) {
            recommendations.add("Multiple movement episodes detected - consider stress reduction techniques")
        }

        if (intensity <= LIGHT_MOVEMENT_THRESHOLD && frequency < 15f) {
            recommendations.add("Excellent movement control - maintain current sleep habits")
        }

        return recommendations
    }

    private fun getRestlessnessGrade(score: Float): String {
        return when {
            score <= 2f -> "Excellent (Very Calm)"
            score <= 4f -> "Good (Calm)"
            score <= 6f -> "Fair (Moderate)"
            score <= 8f -> "Poor (Restless)"
            else -> "Very Poor (Very Restless)"
        }
    }

    private fun identifyPrimaryRestlessnessCause(
        frequency: Float,
        intensity: Float,
        distribution: Float,
        consistency: Float
    ): String {
        val scores = mapOf(
            "High Movement Frequency" to frequency,
            "High Movement Intensity" to intensity,
            "Uneven Movement Distribution" to (10f - distribution),
            "Inconsistent Movement Patterns" to consistency
        )
        return scores.maxByOrNull { it.value }?.key ?: "Unknown"
    }

    private fun generateRestlessnessRecommendations(overall: Float, frequency: Float, intensity: Float): List<String> {
        val recommendations = mutableListOf<String>()

        when {
            overall <= 3f -> recommendations.add("Excellent sleep stillness - maintain current habits")
            overall <= 6f -> recommendations.add("Consider gentle stretching or relaxation before bed")
            else -> recommendations.add("High restlessness detected - review sleep environment and stress levels")
        }

        if (frequency > 7f) {
            recommendations.add("Frequent movements detected - ensure comfortable sleep temperature and bedding")
        }

        if (intensity > 7f) {
            recommendations.add("High intensity movements - consider meditation or sleep hygiene improvements")
        }

        return recommendations
    }

    // ========== ADDITIONAL HELPER METHODS ==========

    private fun getDominantMovementType(micro: Int, light: Int, moderate: Int, significant: Int, major: Int): String {
        val counts = mapOf(
            "Micro" to micro,
            "Light" to light,
            "Moderate" to moderate,
            "Significant" to significant,
            "Major" to major
        )
        return counts.maxByOrNull { it.value }?.key ?: "Unknown"
    }

    private fun createMovementBurst(movements: List<MovementEvent>): MovementBurst {
        return MovementBurst(
            startTime = movements.minByOrNull { it.timestamp }?.timestamp ?: 0L,
            endTime = movements.maxByOrNull { it.timestamp }?.timestamp ?: 0L,
            duration = (movements.maxByOrNull { it.timestamp }?.timestamp ?: 0L) -
                    (movements.minByOrNull { it.timestamp }?.timestamp ?: 0L),
            movements = movements,
            averageIntensity = movements.map { it.intensity }.average().toFloat(),
            peakIntensity = movements.maxByOrNull { it.intensity }?.intensity ?: 0f,
            movementCount = movements.size
        )
    }

    private fun calculateHourlyRestlessness(movements: List<MovementEvent>, sessionDuration: Long): List<Float> {
        val hourCount = maxOf(1, (sessionDuration / (60 * 60 * 1000L)).toInt())
        val hourlyScores = mutableListOf<Float>()

        val sessionStart = movements.minByOrNull { it.timestamp }?.timestamp ?: 0L
        val hourDuration = sessionDuration / hourCount

        for (hour in 0 until hourCount) {
            val hourStart = sessionStart + (hour * hourDuration)
            val hourEnd = hourStart + hourDuration

            val hourMovements = movements.filter { it.timestamp >= hourStart && it.timestamp < hourEnd }
            val hourScore = if (hourMovements.isNotEmpty()) {
                calculateIntensityRestlessness(hourMovements)
            } else 0f

            hourlyScores.add(hourScore)
        }

        return hourlyScores
    }

    private fun createRestlessnessCluster(movements: List<MovementEvent>): RestlessnessCluster {
        return RestlessnessCluster(
            startTime = movements.minByOrNull { it.timestamp }?.timestamp ?: 0L,
            endTime = movements.maxByOrNull { it.timestamp }?.timestamp ?: 0L,
            duration = (movements.maxByOrNull { it.timestamp }?.timestamp ?: 0L) -
                    (movements.minByOrNull { it.timestamp }?.timestamp ?: 0L),
            movements = movements,
            restlessnessScore = calculateIntensityRestlessness(movements),
            severity = when {
                movements.map { it.intensity }.average() > SIGNIFICANT_MOVEMENT_THRESHOLD -> RestlessnessSeverity.HIGH
                movements.map { it.intensity }.average() > MODERATE_MOVEMENT_THRESHOLD -> RestlessnessSeverity.MODERATE
                else -> RestlessnessSeverity.LOW
            }
        )
    }

    private fun detectDisruptionEvents(movements: List<MovementEvent>): List<MovementEvent> {
        return movements.filter { it.intensity > SIGNIFICANT_MOVEMENT_THRESHOLD }
    }

    private fun findLongestCalmPeriod(movements: List<MovementEvent>): Long {
        if (movements.size < 2) return 0L

        val calmMovements = movements.filter { it.intensity <= LIGHT_MOVEMENT_THRESHOLD }
            .sortedBy { it.timestamp }

        if (calmMovements.size < 2) return 0L

        var longestPeriod = 0L
        var currentPeriodStart = calmMovements.first().timestamp

        for (i in 1 until calmMovements.size) {
            val gap = calmMovements[i].timestamp - calmMovements[i - 1].timestamp
            if (gap > CLUSTER_TIME_THRESHOLD) {
                // Gap in calm period
                val periodLength = calmMovements[i - 1].timestamp - currentPeriodStart
                longestPeriod = maxOf(longestPeriod, periodLength)
                currentPeriodStart = calmMovements[i].timestamp
            }
        }

        // Check final period
        val finalPeriodLength = calmMovements.last().timestamp - currentPeriodStart
        return maxOf(longestPeriod, finalPeriodLength)
    }

    private fun calculateDisruptionImpact(disruptionEvents: List<MovementEvent>, sessionDuration: Long): Float {
        if (disruptionEvents.isEmpty() || sessionDuration == 0L) return 0f

        val totalDisruptionIntensity = disruptionEvents.sumOf { it.intensity.toDouble() }.toFloat()
        val disruptionRate = disruptionEvents.size.toFloat() / (sessionDuration / 3600000f) // per hour

        // Impact score based on intensity and frequency
        return ((totalDisruptionIntensity / disruptionEvents.size) * (disruptionRate / 10f)).coerceIn(0f, 10f)
    }

    private fun analyzeClusters(clusters: List<MovementCluster>): ClusterAnalysisResult {
        if (clusters.isEmpty()) return ClusterAnalysisResult()

        val avgClusterSize = clusters.map { it.movements.size }.average().toFloat()
        val avgClusterDuration = clusters.map { it.duration }.average()
        val avgClusterIntensity = clusters.map { it.averageIntensity }.average().toFloat()

        return ClusterAnalysisResult(
            totalClusters = clusters.size,
            averageClusterSize = avgClusterSize,
            averageClusterDuration = avgClusterDuration,
            averageClusterIntensity = avgClusterIntensity,
            largestCluster = clusters.maxByOrNull { it.movements.size },
            mostIntenseCluster = clusters.maxByOrNull { it.averageIntensity }
        )
    }

    private fun identifyClusterPatterns(clusters: List<MovementCluster>): List<ClusterPattern> {
        // Identify patterns in cluster timing, intensity, and distribution
        val patterns = mutableListOf<ClusterPattern>()

        // Periodic pattern detection
        if (clusters.size >= 3) {
            val intervals = mutableListOf<Long>()
            for (i in 1 until clusters.size) {
                intervals.add(clusters[i].startTime - clusters[i - 1].startTime)
            }

            val avgInterval = intervals.average()
            val intervalVariance = intervals.map { (it - avgInterval).pow(2) }.average()

            if (intervalVariance < avgInterval * 0.3) { // Low variance indicates periodicity
                patterns.add(
                    ClusterPattern(
                        type = PatternType.PERIODIC,
                        description = "Regular movement episodes every ${avgInterval / 60000}min",
                        confidence = 1f - (intervalVariance / avgInterval).toFloat()
                    )
                )
            }
        }

        return patterns
    }

    private fun filterNoiseClusters(clusters: List<MovementCluster>): List<MovementCluster> {
        return clusters.filter { cluster ->
            cluster.movements.size >= MIN_CLUSTER_SIZE &&
                    cluster.duration >= 30000L && // At least 30 seconds
                    cluster.averageIntensity >= MICRO_MOVEMENT_THRESHOLD
        }
    }

    private fun generateClusterInsights(clusters: List<MovementCluster>, patterns: List<ClusterPattern>): List<String> {
        val insights = mutableListOf<String>()

        if (clusters.isNotEmpty()) {
            val avgDuration = clusters.map { it.duration }.average() / 60000 // minutes
            insights.add("Average movement episode lasts ${avgDuration.toInt()} minutes")

            val nighttimeClusters = clusters.count { isNighttimeCluster(it) }
            if (nighttimeClusters > clusters.size * 0.7) {
                insights.add("Most movement occurs during typical sleep hours")
            }
        }

        patterns.forEach { pattern ->
            insights.add(pattern.description)
        }

        return insights
    }

    private fun generateClusterRecommendations(patterns: List<ClusterPattern>, insights: List<String>): List<String> {
        val recommendations = mutableListOf<String>()

        if (patterns.any { it.type == PatternType.PERIODIC }) {
            recommendations.add("Regular movement patterns detected - consider reviewing sleep schedule consistency")
        }

        if (insights.any { it.contains("nighttime") }) {
            recommendations.add("High nighttime movement - optimize sleep environment temperature and comfort")
        }

        return recommendations
    }

    private fun createMovementCluster(movements: List<MovementEvent>, type: ClusterType): MovementCluster {
        return MovementCluster(
            id = movements.hashCode().toLong(),
            startTime = movements.minByOrNull { it.timestamp }?.timestamp ?: 0L,
            endTime = movements.maxByOrNull { it.timestamp }?.timestamp ?: 0L,
            duration = (movements.maxByOrNull { it.timestamp }?.timestamp ?: 0L) -
                    (movements.minByOrNull { it.timestamp }?.timestamp ?: 0L),
            movements = movements,
            averageIntensity = movements.map { it.intensity }.average().toFloat(),
            peakIntensity = movements.maxByOrNull { it.intensity }?.intensity ?: 0f,
            clusterType = type
        )
    }

    private fun mergeClusters(temporal: List<MovementCluster>, spatial: List<MovementCluster>): List<MovementCluster> {
        // Simplified merge - in practice would use more sophisticated algorithms
        return (temporal + spatial).distinctBy { it.id }
    }

    private fun applyBasicFilter(movements: List<MovementEvent>): List<MovementEvent> {
        return movements.filter { it.intensity >= MICRO_MOVEMENT_THRESHOLD }
    }

    private fun applyAdaptiveFilter(movements: List<MovementEvent>): List<MovementEvent> {
        if (movements.isEmpty()) return movements

        val intensities = movements.map { it.intensity }
        val mean = intensities.average().toFloat()
        val stdDev = sqrt(calculateVariance(intensities))
        val threshold = mean - stdDev

        return movements.filter { it.intensity >= threshold }
    }

    private fun applyStatisticalFilter(movements: List<MovementEvent>): List<MovementEvent> {
        if (movements.isEmpty()) return movements

        val intensities = movements.map { it.intensity }
        val q1 = calculatePercentile(intensities, 25)
        val q3 = calculatePercentile(intensities, 75)
        val iqr = q3 - q1
        val lowerBound = q1 - 1.5f * iqr

        return movements.filter { it.intensity >= lowerBound }
    }

    private fun applyMLInspiredFilter(movements: List<MovementEvent>): List<MovementEvent> {
        // Simplified ML-inspired filter using local outlier detection
        return movements.filterIndexed { index, movement ->
            val neighbors = movements.drop(maxOf(0, index - 2)).take(5)
            val avgNeighborIntensity = neighbors.map { it.intensity }.average().toFloat()
            val deviation = abs(movement.intensity - avgNeighborIntensity)
            deviation <= avgNeighborIntensity * 2f // Keep if within 2x of local average
        }
    }

    private fun calculateNoiseReduction(original: List<MovementEvent>, filtered: List<MovementEvent>): Float {
        val removedCount = original.size - filtered.size
        return (removedCount.toFloat() / original.size * 100f).coerceIn(0f, 100f)
    }

    private fun calculateSignalPreservation(original: List<MovementEvent>, filtered: List<MovementEvent>): Float {
        val significantOriginal = original.count { it.isSignificant() }
        val significantFiltered = filtered.count { it.isSignificant() }

        return if (significantOriginal > 0) {
            (significantFiltered.toFloat() / significantOriginal * 100f).coerceIn(0f, 100f)
        } else 100f
    }

    private fun calculateTrendStrength(values: List<Float>): Float {
        if (values.size < 2) return 0f

        val firstHalf = values.take(values.size / 2).average()
        val secondHalf = values.drop(values.size / 2).average()

        return ((secondHalf - firstHalf) / firstHalf).toFloat()
    }

    private fun analyzeTransitionQuality(transitions: List<PhaseTransition>): TransitionQualityAnalysis {
        if (transitions.isEmpty()) return TransitionQualityAnalysis()

        val avgConfidence = transitions.map { it.confidence }.average().toFloat()
        val smoothTransitions = transitions.count { it.confidence > 0.7f }
        val abruptTransitions = transitions.count { it.confidence < 0.4f }

        return TransitionQualityAnalysis(
            averageConfidence = avgConfidence,
            smoothTransitions = smoothTransitions,
            abruptTransitions = abruptTransitions,
            transitionQuality = when {
                avgConfidence > 0.8f -> "Excellent"
                avgConfidence > 0.6f -> "Good"
                avgConfidence > 0.4f -> "Fair"
                else -> "Poor"
            }
        )
    }

    private fun calculateTransitionStabilityScore(transitions: List<PhaseTransition>): Float {
        if (transitions.isEmpty()) return 10f

        val confidences = transitions.map { it.confidence }
        val avgConfidence = confidences.average().toFloat()
        val confidenceVariance = calculateVariance(confidences)

        return ((avgConfidence * 10f) * (1f - confidenceVariance)).coerceIn(0f, 10f)
    }

    private fun generatePhaseRecommendations(
        transitions: List<PhaseTransition>,
        phaseDurations: Map<SleepPhase, Long>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (transitions.size > 20) {
            recommendations.add("High number of sleep phase transitions - consider stress reduction techniques")
        }

        val deepSleepDuration = phaseDurations[SleepPhase.DEEP_SLEEP] ?: 0L
        val totalDuration = phaseDurations.values.sum()
        val deepSleepRatio = if (totalDuration > 0) deepSleepDuration.toFloat() / totalDuration else 0f

        if (deepSleepRatio < 0.15f) {
            recommendations.add("Low deep sleep detected - ensure cool, dark, quiet sleep environment")
        }

        return recommendations
    }

    private fun isNighttimeCluster(cluster: MovementCluster): Boolean {
        val calendar = java.util.Calendar.getInstance()
        calendar.timeInMillis = cluster.startTime
        val hour = calendar.get(java.util.Calendar.HOUR_OF_DAY)
        return hour >= 22 || hour <= 6 // 10 PM to 6 AM
    }

    private fun calculateClusterQuality(clusters: List<MovementCluster>): Float {
        if (clusters.isEmpty()) return 0f

        val avgClusterSize = clusters.map { it.movements.size }.average()
        val avgDuration = clusters.map { it.duration }.average()

        // Quality based on cluster size and duration consistency
        val sizeConsistency = 1f - (clusters.map { it.movements.size }.let { calculateVariance(it.map { size -> size.toFloat() }) } / avgClusterSize).toFloat()
        val durationConsistency = 1f - (clusters.map { it.duration }.let { calculateVariance(it.map { dur -> dur.toFloat() }) } / avgDuration).toFloat()

        return ((sizeConsistency + durationConsistency) / 2f * 10f).coerceIn(0f, 10f)
    }
}

// ========== DATA CLASSES FOR ANALYSIS RESULTS ==========

data class MovementIntensityAnalysis(
    val averageIntensity: Float = 0f,
    val maxIntensity: Float = 0f,
    val minIntensity: Float = 0f,
    val intensityVariance: Float = 0f,
    val movementFrequency: Float = 0f,
    val classification: MovementClassification = MovementClassification(),
    val temporalPatterns: TemporalIntensityPattern = TemporalIntensityPattern(),
    val distribution: IntensityDistribution = IntensityDistribution(),
    val bursts: List<MovementBurst> = emptyList(),
    val adaptiveThresholds: AdaptiveThresholds = AdaptiveThresholds(),
    val intensityGrade: String = "Unknown",
    val stabilityScore: Float = 0f,
    val recommendations: List<String> = emptyList()
)

data class MovementClassification(
    val microMovements: Int = 0,
    val lightMovements: Int = 0,
    val moderateMovements: Int = 0,
    val significantMovements: Int = 0,
    val majorMovements: Int = 0,
    val dominantType: String = "Unknown"
)

data class TemporalIntensityPattern(
    val periodIntensities: List<Float> = emptyList(),
    val trend: IntensityTrend = IntensityTrend.INSUFFICIENT_DATA,
    val peakPeriod: Int = 0,
    val calmestPeriod: Int = 0,
    val variability: Float = 0f
)

data class IntensityDistribution(
    val mean: Float = 0f,
    val median: Float = 0f,
    val standardDeviation: Float = 0f,
    val skewness: Float = 0f,
    val kurtosis: Float = 0f,
    val percentile25: Float = 0f,
    val percentile75: Float = 0f,
    val interquartileRange: Float = 0f
)

data class MovementBurst(
    val startTime: Long,
    val endTime: Long,
    val duration: Long,
    val movements: List<MovementEvent>,
    val averageIntensity: Float,
    val peakIntensity: Float,
    val movementCount: Int
)

data class AdaptiveThresholds(
    val microThreshold: Float = 0.5f,
    val lightThreshold: Float = 1.5f,
    val moderateThreshold: Float = 3.0f,
    val significantThreshold: Float = 5.0f,
    val majorThreshold: Float = 8.0f
)

data class RealTimeIntensityData(
    val rawIntensity: Float = 0f,
    val smoothedIntensity: Float = 0f,
    val exponentialSmoothed: Float = 0f,
    val trend: IntensityTrend = IntensityTrend.STABLE,
    val activityLevel: ActivityLevel = ActivityLevel.MINIMAL,
    val predictedPhase: SleepPhase = SleepPhase.UNKNOWN,
    val confidence: Float = 0f,
    val timestamp: Long = 0L
)

data class RestlessnessAnalysis(
    val overallScore: Float = 0f,
    val frequencyScore: Float = 0f,
    val intensityScore: Float = 0f,
    val distributionScore: Float = 0f,
    val consistencyScore: Float = 0f,
    val temporalAnalysis: TemporalRestlessnessAnalysis = TemporalRestlessnessAnalysis(),
    val movementClusters: List<RestlessnessCluster> = emptyList(),
    val disruptionAnalysis: SleepDisruptionAnalysis = SleepDisruptionAnalysis(),
    val restlessnessGrade: String = "Unknown",
    val primaryCause: String = "Unknown",
    val recommendations: List<String> = emptyList()
)

data class TemporalRestlessnessAnalysis(
    val hourlyScores: List<Float> = emptyList(),
    val peakRestlessnessHour: Int = 0,
    val calmestHour: Int = 0,
    val restlessnessVariation: Float = 0f
)

data class RestlessnessCluster(
    val startTime: Long,
    val endTime: Long,
    val duration: Long,
    val movements: List<MovementEvent>,
    val restlessnessScore: Float,
    val severity: RestlessnessSeverity
)

data class SleepDisruptionAnalysis(
    val totalDisruptions: Int = 0,
    val majorMovements: Int = 0,
    val disruptionRate: Float = 0f,
    val averageDisruptionIntensity: Float = 0f,
    val longestCalmPeriod: Long = 0L,
    val disruptionImpact: Float = 0f
)

data class SleepPhaseDetection(
    val detectedPhase: SleepPhase,
    val confidence: Float,
    val phaseScores: Map<SleepPhase, Float> = emptyMap(),
    val transitionProbabilities: Map<SleepPhase, Float> = emptyMap(),
    val stabilityMetrics: PhaseStabilityMetrics = PhaseStabilityMetrics(),
    val analysisWindow: Long = 0L,
    val movementCount: Int = 0,
    val averageIntensity: Float = 0f,
    val algorithm: String = ""
)

data class PhaseStabilityMetrics(
    val varianceScore: Float = 0f,
    val trendStability: Float = 0f,
    val overallStability: Float = 0f
)

data class TimestampedPhaseDetection(
    val timestamp: Long,
    val detection: SleepPhaseDetection
)

data class PhaseTransitionAnalysis(
    val transitions: List<PhaseTransition> = emptyList(),
    val phaseDurations: Map<SleepPhase, Long> = emptyMap(),
    val transitionQuality: TransitionQualityAnalysis = TransitionQualityAnalysis(),
    val totalTransitions: Int = 0,
    val averagePhaseLength: Long = 0L,
    val stabilityScore: Float = 0f,
    val recommendations: List<String> = emptyList()
)

data class TransitionQualityAnalysis(
    val averageConfidence: Float = 0f,
    val smoothTransitions: Int = 0,
    val abruptTransitions: Int = 0,
    val transitionQuality: String = "Unknown"
)

data class MovementClusterAnalysis(
    val clusters: List<MovementCluster> = emptyList(),
    val totalClusters: Int = 0,
    val filteredClusters: Int = 0,
    val patterns: List<ClusterPattern> = emptyList(),
    val insights: List<String> = emptyList(),
    val algorithm: ClusteringAlgorithm = ClusteringAlgorithm.TEMPORAL_INTENSITY,
    val clusterQuality: Float = 0f,
    val recommendations: List<String> = emptyList()
)

data class MovementCluster(
    val id: Long,
    val startTime: Long,
    val endTime: Long,
    val duration: Long,
    val movements: List<MovementEvent>,
    val averageIntensity: Float,
    val peakIntensity: Float,
    val clusterType: ClusterType
)

data class ClusterPattern(
    val type: PatternType,
    val description: String,
    val confidence: Float
)

data class ClusterAnalysisResult(
    val totalClusters: Int = 0,
    val averageClusterSize: Float = 0f,
    val averageClusterDuration: Double = 0.0,
    val averageClusterIntensity: Float = 0f,
    val largestCluster: MovementCluster? = null,
    val mostIntenseCluster: MovementCluster? = null
)

data class MovementFilterResult(
    val originalMovements: List<MovementEvent>,
    val filteredMovements: List<MovementEvent>,
    val filterType: MovementFilter,
    val noiseReduction: Float = 0f,
    val signalPreservation: Float = 0f,
    val filterEffectiveness: Float = 0f,
    val removedCount: Int = 0,
    val retainedCount: Int = 0
)

// ========== ENUMS ==========

enum class IntensityTrend {
    INCREASING,
    DECREASING,
    STABLE,
    INSUFFICIENT_DATA
}

enum class ActivityLevel {
    MINIMAL,
    LOW,
    MODERATE,
    HIGH,
    VERY_HIGH
}

enum class RestlessnessSeverity {
    LOW,
    MODERATE,
    HIGH
}

enum class ClusteringAlgorithm {
    TEMPORAL_INTENSITY,
    SPATIAL_PATTERN,
    HYBRID
}

enum class ClusterType {
    TEMPORAL_INTENSITY,
    SPATIAL_PATTERN,
    HYBRID
}

enum class PatternType {
    PERIODIC,
    BURST,
    GRADUAL,
    RANDOM
}

enum class MovementFilter {
    BASIC,
    ADAPTIVE,
    STATISTICAL,
    MACHINE_LEARNING
}