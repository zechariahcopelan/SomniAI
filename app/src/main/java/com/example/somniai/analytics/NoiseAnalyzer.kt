package com.example.somniai.analytics

import android.util.Log
import com.example.somniai.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.*

/**
 * Advanced noise and audio level analysis engine for sleep tracking
 *
 * Provides comprehensive noise analytics including:
 * - Sophisticated noise level categorization with adaptive thresholds
 * - Environmental noise impact assessment on sleep quality
 * - Advanced noise-movement correlation analysis with temporal patterns
 * - Multi-dimensional sleep disruption scoring algorithms
 * - Real-time noise monitoring with pattern recognition
 * - Noise event clustering and environmental profiling
 * - Adaptive filtering for noise artifacts and false positives
 * - Sleep phase correlation with ambient noise levels
 */
class NoiseAnalyzer {

    companion object {
        private const val TAG = "NoiseAnalyzer"

        // Noise level thresholds (in decibels)
        private const val VERY_QUIET_THRESHOLD = 25f // Library quiet
        private const val QUIET_THRESHOLD = 35f // Residential night
        private const val MODERATE_THRESHOLD = 45f // Quiet office
        private const val LOUD_THRESHOLD = 55f // Normal conversation
        private const val VERY_LOUD_THRESHOLD = 65f // Traffic noise
        private const val DISRUPTIVE_THRESHOLD = 50f // Sleep disruption threshold

        // Sleep disruption impact thresholds
        private const val LIGHT_DISRUPTION_DB = 40f
        private const val MODERATE_DISRUPTION_DB = 50f
        private const val SEVERE_DISRUPTION_DB = 60f
        private const val CRITICAL_DISRUPTION_DB = 70f

        // Temporal analysis windows (milliseconds)
        private const val SHORT_WINDOW = 1 * 60 * 1000L // 1 minute
        private const val MEDIUM_WINDOW = 5 * 60 * 1000L // 5 minutes
        private const val LONG_WINDOW = 15 * 60 * 1000L // 15 minutes

        // Clustering parameters
        private const val CLUSTER_TIME_THRESHOLD = 3 * 60 * 1000L // 3 minutes
        private const val CLUSTER_DECIBEL_THRESHOLD = 10f // 10 dB difference
        private const val MIN_CLUSTER_SIZE = 3

        // Correlation analysis parameters
        private const val CORRELATION_TIME_WINDOW = 2 * 60 * 1000L // 2 minutes
        private const val MOVEMENT_RESPONSE_DELAY = 30 * 1000L // 30 seconds

        // Environmental profile weights
        private const val BASELINE_WEIGHT = 0.4f
        private const val PEAK_WEIGHT = 0.3f
        private const val VARIABILITY_WEIGHT = 0.2f
        private const val FREQUENCY_WEIGHT = 0.1f
    }

    // ========== NOISE LEVEL CATEGORIZATION ==========

    /**
     * Comprehensive noise level analysis with environmental categorization
     */
    suspend fun analyzeNoiselevels(
        noiseEvents: List<NoiseEvent>,
        sessionDuration: Long
    ): NoiseLevelAnalysis = withContext(Dispatchers.Default) {
        try {
            if (noiseEvents.isEmpty()) {
                return@withContext NoiseLevelAnalysis()
            }

            val decibelLevels = noiseEvents.map { it.decibelLevel }

            // Basic statistical analysis
            val averageLevel = decibelLevels.average().toFloat()
            val maxLevel = decibelLevels.maxOrNull() ?: 0f
            val minLevel = decibelLevels.minOrNull() ?: 0f
            val medianLevel = calculateMedian(decibelLevels)
            val standardDeviation = sqrt(calculateVariance(decibelLevels))

            // Categorize noise events
            val categorization = categorizeNoiseEvents(noiseEvents)

            // Calculate noise frequency (events per hour)
            val noiseFrequency = if (sessionDuration > 0) {
                (noiseEvents.size.toFloat() / sessionDuration) * 3600000f
            } else 0f

            // Analyze noise patterns
            val patterns = analyzeNoisePatterns(noiseEvents, sessionDuration)

            // Calculate noise consistency
            val consistency = calculateNoiseConsistency(noiseEvents)

            // Adaptive threshold calculation
            val adaptiveThresholds = calculateAdaptiveNoiseThresholds(noiseEvents)

            // Environmental classification
            val environmentalProfile = classifyEnvironmentalProfile(noiseEvents, sessionDuration)

            NoiseLevelAnalysis(
                averageLevel = averageLevel,
                maxLevel = maxLevel,
                minLevel = minLevel,
                medianLevel = medianLevel,
                standardDeviation = standardDeviation,
                categorization = categorization,
                noiseFrequency = noiseFrequency,
                patterns = patterns,
                consistency = consistency,
                adaptiveThresholds = adaptiveThresholds,
                environmentalProfile = environmentalProfile,
                noiseGrade = getNoiseGrade(averageLevel),
                qualityImpact = calculateQualityImpact(averageLevel, noiseFrequency, standardDeviation),
                recommendations = generateNoiseRecommendations(averageLevel, environmentalProfile, patterns)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing noise levels", e)
            NoiseLevelAnalysis()
        }
    }

    /**
     * Real-time noise level monitoring with adaptive analysis
     */
    suspend fun analyzeRealTimeNoise(
        recentNoiseEvents: List<NoiseEvent>,
        windowSize: Long = MEDIUM_WINDOW
    ): RealTimeNoiseData = withContext(Dispatchers.Default) {
        try {
            if (recentNoiseEvents.isEmpty()) {
                return@withContext RealTimeNoiseData()
            }

            val currentTime = System.currentTimeMillis()
            val windowStart = currentTime - windowSize

            // Filter events within window
            val windowEvents = recentNoiseEvents.filter { it.timestamp >= windowStart }

            if (windowEvents.isEmpty()) {
                return@withContext RealTimeNoiseData()
            }

            // Calculate current noise metrics
            val currentLevel = windowEvents.map { it.decibelLevel }.average().toFloat()
            val peakLevel = windowEvents.maxByOrNull { it.decibelLevel }?.decibelLevel ?: 0f
            val noiseVariability = calculateVariance(windowEvents.map { it.decibelLevel })

            // Apply smoothing for stability
            val smoothedLevel = applyNoiseSmoothing(windowEvents)

            // Detect noise trends
            val trend = detectNoiseTrend(windowEvents)

            // Classify current noise environment
            val environmentType = classifyCurrentEnvironment(currentLevel, noiseVariability)

            // Assess sleep impact
            val sleepImpact = assessRealTimeSleepImpact(currentLevel, peakLevel, noiseVariability)

            // Predict noise continuation
            val noisePrediction = predictNoiseContinuation(windowEvents)

            RealTimeNoiseData(
                currentLevel = currentLevel,
                smoothedLevel = smoothedLevel,
                peakLevel = peakLevel,
                noiseVariability = noiseVariability,
                trend = trend,
                environmentType = environmentType,
                sleepImpact = sleepImpact,
                prediction = noisePrediction,
                timestamp = currentTime,
                confidence = calculateNoiseConfidence(windowEvents)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing real-time noise", e)
            RealTimeNoiseData()
        }
    }

    // ========== ENVIRONMENTAL NOISE IMPACT ==========

    /**
     * Analyze environmental noise impact on sleep quality
     */
    suspend fun analyzeEnvironmentalImpact(
        noiseEvents: List<NoiseEvent>,
        sessionDuration: Long,
        sleepPhases: List<PhaseTransition> = emptyList()
    ): EnvironmentalNoiseImpact = withContext(Dispatchers.Default) {
        try {
            if (noiseEvents.isEmpty()) {
                return@withContext EnvironmentalNoiseImpact()
            }

            // Analyze noise impact by sleep phase
            val phaseImpactAnalysis = analyzeNoiseImpactByPhase(noiseEvents, sleepPhases)

            // Calculate environmental noise score
            val environmentalScore = calculateEnvironmentalScore(noiseEvents, sessionDuration)

            // Identify noise sources and patterns
            val noiseSources = identifyNoiseSources(noiseEvents)

            // Calculate adaptation metrics
            val adaptationMetrics = calculateNoiseAdaptation(noiseEvents, sessionDuration)

            // Assess cumulative impact
            val cumulativeImpact = calculateCumulativeNoiseImpact(noiseEvents, sessionDuration)

            // Environmental recommendations
            val environmentalRecommendations = generateEnvironmentalRecommendations(
                environmentalScore, noiseSources, phaseImpactAnalysis
            )

            EnvironmentalNoiseImpact(
                overallImpactScore = environmentalScore,
                phaseImpactAnalysis = phaseImpactAnalysis,
                identifiedNoiseSources = noiseSources,
                adaptationMetrics = adaptationMetrics,
                cumulativeImpact = cumulativeImpact,
                environmentalGrade = getEnvironmentalGrade(environmentalScore),
                primaryNoiseSource = identifyPrimaryNoiseSource(noiseSources),
                recommendations = environmentalRecommendations
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing environmental impact", e)
            EnvironmentalNoiseImpact()
        }
    }

    // ========== NOISE-MOVEMENT CORRELATION ==========

    /**
     * Advanced correlation analysis between noise events and movement
     */
    suspend fun analyzeNoiseMovementCorrelation(
        noiseEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent>,
        sessionDuration: Long
    ): NoiseMovementCorrelation = withContext(Dispatchers.Default) {
        try {
            if (noiseEvents.isEmpty() || movementEvents.isEmpty()) {
                return@withContext NoiseMovementCorrelation()
            }

            // Calculate temporal correlations
            val temporalCorrelation = calculateTemporalCorrelation(noiseEvents, movementEvents)

            // Analyze noise-triggered movements
            val triggeredMovements = analyzeNoiseTriggeredMovements(noiseEvents, movementEvents)

            // Calculate response delays
            val responseDelays = calculateMovementResponseDelays(noiseEvents, movementEvents)

            // Analyze intensity correlations
            val intensityCorrelation = analyzeIntensityCorrelation(noiseEvents, movementEvents)

            // Pattern correlation analysis
            val patternCorrelations = analyzePatternCorrelations(noiseEvents, movementEvents)

            // Sleep phase correlation
            val phaseCorrelations = analyzePhaseCorrelations(noiseEvents, movementEvents)

            // Statistical significance testing
            val statisticalSignificance = calculateCorrelationSignificance(
                temporalCorrelation, intensityCorrelation, noiseEvents.size, movementEvents.size
            )

            NoiseMovementCorrelation(
                temporalCorrelation = temporalCorrelation,
                intensityCorrelation = intensityCorrelation,
                triggeredMovements = triggeredMovements,
                responseDelays = responseDelays,
                patternCorrelations = patternCorrelations,
                phaseCorrelations = phaseCorrelations,
                statisticalSignificance = statisticalSignificance,
                correlationStrength = calculateOverallCorrelationStrength(temporalCorrelation, intensityCorrelation),
                insights = generateCorrelationInsights(temporalCorrelation, triggeredMovements, responseDelays)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing noise-movement correlation", e)
            NoiseMovementCorrelation()
        }
    }

    // ========== SLEEP DISRUPTION SCORING ==========

    /**
     * Comprehensive sleep disruption scoring from noise analysis
     */
    suspend fun analyzeSleepDisruption(
        noiseEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent> = emptyList(),
        sleepPhases: List<PhaseTransition> = emptyList(),
        sessionDuration: Long
    ): NoiseDisruptionAnalysis = withContext(Dispatchers.Default) {
        try {
            if (noiseEvents.isEmpty()) {
                return@withContext NoiseDisruptionAnalysis()
            }

            // Identify disruptive events
            val disruptiveEvents = identifyDisruptiveNoiseEvents(noiseEvents)

            // Calculate disruption intensity
            val disruptionIntensity = calculateDisruptionIntensity(disruptiveEvents)

            // Analyze disruption timing
            val timingAnalysis = analyzeDisruptionTiming(disruptiveEvents, sleepPhases, sessionDuration)

            // Calculate recovery metrics
            val recoveryMetrics = calculateRecoveryMetrics(disruptiveEvents, movementEvents)

            // Assess cumulative disruption
            val cumulativeDisruption = assessCumulativeDisruption(disruptiveEvents, sessionDuration)

            // Calculate disruption frequency
            val disruptionFrequency = (disruptiveEvents.size.toFloat() / sessionDuration) * 3600000f // per hour

            // Overall disruption score (0-10 scale)
            val overallDisruptionScore = calculateOverallDisruptionScore(
                disruptionIntensity, disruptionFrequency, cumulativeDisruption, timingAnalysis
            )

            // Generate disruption insights
            val insights = generateDisruptionInsights(
                disruptiveEvents, timingAnalysis, recoveryMetrics
            )

            NoiseDisruptionAnalysis(
                overallDisruptionScore = overallDisruptionScore,
                disruptiveEvents = disruptiveEvents,
                disruptionIntensity = disruptionIntensity,
                disruptionFrequency = disruptionFrequency,
                timingAnalysis = timingAnalysis,
                recoveryMetrics = recoveryMetrics,
                cumulativeDisruption = cumulativeDisruption,
                disruptionGrade = getDisruptionGrade(overallDisruptionScore),
                primaryDisruptionSource = identifyPrimaryDisruptionSource(disruptiveEvents),
                insights = insights,
                recommendations = generateDisruptionRecommendations(overallDisruptionScore, timingAnalysis, insights)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing sleep disruption", e)
            NoiseDisruptionAnalysis()
        }
    }

    /**
     * Advanced noise event clustering with environmental pattern recognition
     */
    suspend fun clusterNoiseEvents(
        noiseEvents: List<NoiseEvent>,
        algorithm: NoiseClusteringAlgorithm = NoiseClusteringAlgorithm.TEMPORAL_DECIBEL
    ): NoiseClusterAnalysis = withContext(Dispatchers.Default) {
        try {
            if (noiseEvents.isEmpty()) {
                return@withContext NoiseClusterAnalysis()
            }

            val clusters = when (algorithm) {
                NoiseClusteringAlgorithm.TEMPORAL_DECIBEL -> performTemporalDecibelClustering(noiseEvents)
                NoiseClusteringAlgorithm.INTENSITY_PATTERN -> performIntensityPatternClustering(noiseEvents)
                NoiseClusteringAlgorithm.ENVIRONMENTAL -> performEnvironmentalClustering(noiseEvents)
                NoiseClusteringAlgorithm.ADAPTIVE -> performAdaptiveClustering(noiseEvents)
            }

            // Analyze cluster characteristics
            val clusterCharacteristics = analyzeClusterCharacteristics(clusters)

            // Identify noise patterns
            val noisePatterns = identifyNoisePatterns(clusters)

            // Calculate cluster quality
            val clusterQuality = calculateNoiseClusterQuality(clusters)

            // Generate environmental insights
            val environmentalInsights = generateEnvironmentalInsights(clusters, noisePatterns)

            NoiseClusterAnalysis(
                clusters = clusters,
                clusterCharacteristics = clusterCharacteristics,
                noisePatterns = noisePatterns,
                clusterQuality = clusterQuality,
                environmentalInsights = environmentalInsights,
                algorithm = algorithm,
                recommendations = generateClusterRecommendations(noisePatterns, environmentalInsights)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error clustering noise events", e)
            NoiseClusterAnalysis()
        }
    }

    // ========== HELPER METHODS - NOISE CATEGORIZATION ==========

    private fun categorizeNoiseEvents(noiseEvents: List<NoiseEvent>): NoiseCategorization {
        val veryQuiet = noiseEvents.count { it.decibelLevel <= VERY_QUIET_THRESHOLD }
        val quiet = noiseEvents.count { it.decibelLevel > VERY_QUIET_THRESHOLD && it.decibelLevel <= QUIET_THRESHOLD }
        val moderate = noiseEvents.count { it.decibelLevel > QUIET_THRESHOLD && it.decibelLevel <= MODERATE_THRESHOLD }
        val loud = noiseEvents.count { it.decibelLevel > MODERATE_THRESHOLD && it.decibelLevel <= LOUD_THRESHOLD }
        val veryLoud = noiseEvents.count { it.decibelLevel > LOUD_THRESHOLD && it.decibelLevel <= VERY_LOUD_THRESHOLD }
        val excessive = noiseEvents.count { it.decibelLevel > VERY_LOUD_THRESHOLD }

        val disruptive = noiseEvents.count { it.isDisruptive() }
        val disruptiveRatio = disruptive.toFloat() / noiseEvents.size

        return NoiseCategorization(
            veryQuiet = veryQuiet,
            quiet = quiet,
            moderate = moderate,
            loud = loud,
            veryLoud = veryLoud,
            excessive = excessive,
            disruptive = disruptive,
            disruptiveRatio = disruptiveRatio,
            dominantCategory = getDominantNoiseCategory(veryQuiet, quiet, moderate, loud, veryLoud, excessive)
        )
    }

    private fun analyzeNoisePatterns(noiseEvents: List<NoiseEvent>, sessionDuration: Long): NoisePatternAnalysis {
        // Analyze temporal distribution
        val temporalDistribution = analyzeTemporalNoiseDistribution(noiseEvents, sessionDuration)

        // Detect periodic patterns
        val periodicPatterns = detectPeriodicNoisePatterns(noiseEvents)

        // Analyze noise bursts
        val noiseBursts = detectNoiseBursts(noiseEvents)

        // Calculate pattern consistency
        val patternConsistency = calculateNoisePatternConsistency(noiseEvents)

        return NoisePatternAnalysis(
            temporalDistribution = temporalDistribution,
            periodicPatterns = periodicPatterns,
            noiseBursts = noiseBursts,
            patternConsistency = patternConsistency,
            dominantPattern = identifyDominantNoisePattern(periodicPatterns, noiseBursts)
        )
    }

    private fun calculateNoiseConsistency(noiseEvents: List<NoiseEvent>): NoiseConsistency {
        val decibelLevels = noiseEvents.map { it.decibelLevel }
        val mean = decibelLevels.average().toFloat()
        val variance = calculateVariance(decibelLevels)
        val coefficientOfVariation = if (mean > 0) sqrt(variance) / mean else 0f

        // Analyze temporal consistency
        val intervals = mutableListOf<Long>()
        for (i in 1 until noiseEvents.size) {
            intervals.add(noiseEvents[i].timestamp - noiseEvents[i - 1].timestamp)
        }

        val avgInterval = if (intervals.isNotEmpty()) intervals.average() else 0.0
        val intervalVariance = if (intervals.isNotEmpty()) {
            intervals.map { (it - avgInterval).pow(2) }.average()
        } else 0.0

        return NoiseConsistency(
            levelConsistency = 1f - coefficientOfVariation,
            temporalConsistency = if (avgInterval > 0) 1f - (sqrt(intervalVariance) / avgInterval).toFloat() else 0f,
            overallConsistency = (1f - coefficientOfVariation +
                    (if (avgInterval > 0) 1f - (sqrt(intervalVariance) / avgInterval).toFloat() else 0f)) / 2f
        )
    }

    private fun calculateAdaptiveNoiseThresholds(noiseEvents: List<NoiseEvent>): AdaptiveNoiseThresholds {
        val decibelLevels = noiseEvents.map { it.decibelLevel }
        val mean = decibelLevels.average().toFloat()
        val stdDev = sqrt(calculateVariance(decibelLevels))

        return AdaptiveNoiseThresholds(
            veryQuietThreshold = maxOf(VERY_QUIET_THRESHOLD, mean - 2 * stdDev),
            quietThreshold = maxOf(QUIET_THRESHOLD, mean - stdDev),
            moderateThreshold = maxOf(MODERATE_THRESHOLD, mean),
            loudThreshold = maxOf(LOUD_THRESHOLD, mean + stdDev),
            veryLoudThreshold = maxOf(VERY_LOUD_THRESHOLD, mean + 2 * stdDev),
            disruptiveThreshold = maxOf(DISRUPTIVE_THRESHOLD, mean + stdDev)
        )
    }

    private fun classifyEnvironmentalProfile(noiseEvents: List<NoiseEvent>, sessionDuration: Long): EnvironmentalProfile {
        val avgLevel = noiseEvents.map { it.decibelLevel }.average().toFloat()
        val maxLevel = noiseEvents.maxByOrNull { it.decibelLevel }?.decibelLevel ?: 0f
        val noiseVariability = calculateVariance(noiseEvents.map { it.decibelLevel })
        val eventFrequency = (noiseEvents.size.toFloat() / sessionDuration) * 3600000f

        val environmentType = when {
            avgLevel <= 30f && noiseVariability < 25f -> EnvironmentType.VERY_QUIET
            avgLevel <= 40f && noiseVariability < 50f -> EnvironmentType.QUIET_RESIDENTIAL
            avgLevel <= 50f && eventFrequency < 20f -> EnvironmentType.MODERATE_RESIDENTIAL
            avgLevel <= 60f || eventFrequency > 40f -> EnvironmentType.URBAN
            else -> EnvironmentType.NOISY_URBAN
        }

        return EnvironmentalProfile(
            environmentType = environmentType,
            baselineNoiseLevel = avgLevel,
            peakNoiseLevel = maxLevel,
            noiseVariability = noiseVariability,
            eventFrequency = eventFrequency,
            environmentScore = calculateEnvironmentScore(avgLevel, noiseVariability, eventFrequency)
        )
    }

    // ========== HELPER METHODS - REAL-TIME ANALYSIS ==========

    private fun applyNoiseSmoothing(noiseEvents: List<NoiseEvent>): Float {
        if (noiseEvents.isEmpty()) return 0f

        // Exponential moving average for noise smoothing
        val alpha = 0.3f
        var smoothed = noiseEvents.first().decibelLevel

        for (i in 1 until noiseEvents.size) {
            smoothed = alpha * noiseEvents[i].decibelLevel + (1 - alpha) * smoothed
        }

        return smoothed
    }

    private fun detectNoiseTrend(noiseEvents: List<NoiseEvent>): NoiseTrend {
        if (noiseEvents.size < 3) return NoiseTrend.STABLE

        val levels = noiseEvents.map { it.decibelLevel }
        val recentAvg = levels.takeLast(3).average()
        val olderAvg = levels.dropLast(3).takeLast(3).average()

        return when {
            recentAvg > olderAvg + 5f -> NoiseTrend.INCREASING
            recentAvg < olderAvg - 5f -> NoiseTrend.DECREASING
            else -> NoiseTrend.STABLE
        }
    }

    private fun classifyCurrentEnvironment(level: Float, variability: Float): EnvironmentType {
        return when {
            level <= 30f && variability < 25f -> EnvironmentType.VERY_QUIET
            level <= 40f && variability < 50f -> EnvironmentType.QUIET_RESIDENTIAL
            level <= 50f -> EnvironmentType.MODERATE_RESIDENTIAL
            level <= 60f -> EnvironmentType.URBAN
            else -> EnvironmentType.NOISY_URBAN
        }
    }

    private fun assessRealTimeSleepImpact(level: Float, peak: Float, variability: Float): SleepImpactLevel {
        val impactScore = when {
            level <= QUIET_THRESHOLD && peak <= MODERATE_THRESHOLD -> 1f
            level <= MODERATE_THRESHOLD && peak <= LOUD_THRESHOLD -> 3f
            level <= LOUD_THRESHOLD || peak <= VERY_LOUD_THRESHOLD -> 6f
            else -> 9f
        } + (variability / 20f) // Add variability impact

        return when {
            impactScore <= 2f -> SleepImpactLevel.MINIMAL
            impactScore <= 4f -> SleepImpactLevel.LOW
            impactScore <= 6f -> SleepImpactLevel.MODERATE
            impactScore <= 8f -> SleepImpactLevel.HIGH
            else -> SleepImpactLevel.SEVERE
        }
    }

    private fun predictNoiseContinuation(noiseEvents: List<NoiseEvent>): NoisePrediction {
        if (noiseEvents.size < 3) return NoisePrediction()

        val trend = detectNoiseTrend(noiseEvents)
        val currentLevel = noiseEvents.lastOrNull()?.decibelLevel ?: 0f
        val avgLevel = noiseEvents.map { it.decibelLevel }.average().toFloat()

        val predictedLevel = when (trend) {
            NoiseTrend.INCREASING -> currentLevel + 2f
            NoiseTrend.DECREASING -> currentLevel - 2f
            else -> avgLevel
        }

        val confidence = calculatePredictionConfidence(noiseEvents)

        return NoisePrediction(
            predictedLevel = predictedLevel,
            trend = trend,
            confidence = confidence,
            timeHorizon = MEDIUM_WINDOW
        )
    }

    private fun calculateNoiseConfidence(noiseEvents: List<NoiseEvent>): Float {
        if (noiseEvents.isEmpty()) return 0f

        val levelVariance = calculateVariance(noiseEvents.map { it.decibelLevel })
        val consistency = 1f / (1f + levelVariance / 100f)
        val sampleSize = minOf(noiseEvents.size / 10f, 1f)

        return (consistency * sampleSize * 100f).coerceIn(0f, 100f)
    }

    // ========== HELPER METHODS - ENVIRONMENTAL IMPACT ==========

    private fun analyzeNoiseImpactByPhase(
        noiseEvents: List<NoiseEvent>,
        sleepPhases: List<PhaseTransition>
    ): Map<SleepPhase, PhaseNoiseImpact> {
        val phaseImpacts = mutableMapOf<SleepPhase, PhaseNoiseImpact>()

        if (sleepPhases.isEmpty()) {
            // Default analysis without phase data
            phaseImpacts[SleepPhase.UNKNOWN] = PhaseNoiseImpact(
                averageNoiseLevel = noiseEvents.map { it.decibelLevel }.average().toFloat(),
                disruptiveEvents = noiseEvents.count { it.isDisruptive() },
                impactScore = calculatePhaseImpactScore(noiseEvents, SleepPhase.UNKNOWN)
            )
            return phaseImpacts
        }

        // Analyze noise impact for each sleep phase
        for (phase in SleepPhase.values()) {
            val phaseNoiseEvents = getNoiseEventsForPhase(noiseEvents, sleepPhases, phase)
            if (phaseNoiseEvents.isNotEmpty()) {
                phaseImpacts[phase] = PhaseNoiseImpact(
                    averageNoiseLevel = phaseNoiseEvents.map { it.decibelLevel }.average().toFloat(),
                    disruptiveEvents = phaseNoiseEvents.count { it.isDisruptive() },
                    impactScore = calculatePhaseImpactScore(phaseNoiseEvents, phase)
                )
            }
        }

        return phaseImpacts
    }

    private fun calculateEnvironmentalScore(noiseEvents: List<NoiseEvent>, sessionDuration: Long): Float {
        val avgLevel = noiseEvents.map { it.decibelLevel }.average().toFloat()
        val disruptiveRatio = noiseEvents.count { it.isDisruptive() }.toFloat() / noiseEvents.size
        val eventFrequency = (noiseEvents.size.toFloat() / sessionDuration) * 3600000f
        val levelVariability = calculateVariance(noiseEvents.map { it.decibelLevel })

        // Score components (lower is better for sleep)
        val levelScore = when {
            avgLevel <= 30f -> 1f
            avgLevel <= 40f -> 3f
            avgLevel <= 50f -> 5f
            avgLevel <= 60f -> 7f
            else -> 9f
        }

        val disruptionScore = disruptiveRatio * 5f
        val frequencyScore = minOf(eventFrequency / 20f, 3f)
        val variabilityScore = minOf(levelVariability / 50f, 2f)

        return (levelScore + disruptionScore + frequencyScore + variabilityScore).coerceIn(0f, 10f)
    }

    private fun identifyNoiseSources(noiseEvents: List<NoiseEvent>): List<NoiseSource> {
        val sources = mutableListOf<NoiseSource>()

        // Analyze noise patterns to identify likely sources
        val clusters = performTemporalDecibelClustering(noiseEvents)

        for (cluster in clusters) {
            val avgLevel = cluster.events.map { it.decibelLevel }.average().toFloat()
            val duration = cluster.endTime - cluster.startTime
            val pattern = analyzeClusterPattern(cluster)

            val sourceType = when {
                avgLevel > 60f && duration < 30000L -> NoiseSourceType.SUDDEN_LOUD // Car horn, door slam
                avgLevel in 45f..60f && pattern == "periodic" -> NoiseSourceType.MACHINERY // AC, fan
                avgLevel in 35f..50f && duration > 5 * 60 * 1000L -> NoiseSourceType.CONTINUOUS // Traffic
                avgLevel in 30f..45f && pattern == "intermittent" -> NoiseSourceType.ENVIRONMENTAL // Wind, animals
                else -> NoiseSourceType.UNKNOWN
            }

            sources.add(
                NoiseSource(
                    type = sourceType,
                    averageLevel = avgLevel,
                    frequency = cluster.events.size,
                    duration = duration,
                    impactScore = calculateSourceImpactScore(cluster)
                )
            )
        }

        return sources
    }

    private fun calculateNoiseAdaptation(noiseEvents: List<NoiseEvent>, sessionDuration: Long): NoiseAdaptationMetrics {
        // Analyze how the person adapts to noise over time
        val timeSegments = divideIntoTimeSegments(noiseEvents, sessionDuration, 6)
        val adaptationScores = mutableListOf<Float>()

        for (i in 1 until timeSegments.size) {
            val previousSegment = timeSegments[i - 1]
            val currentSegment = timeSegments[i]

            val previousDisruption = previousSegment.count { it.isDisruptive() }.toFloat() / previousSegment.size
            val currentDisruption = currentSegment.count { it.isDisruptive() }.toFloat() / currentSegment.size

            val adaptation = if (previousDisruption > 0) {
                (previousDisruption - currentDisruption) / previousDisruption
            } else 0f

            adaptationScores.add(adaptation)
        }

        val overallAdaptation = adaptationScores.average().toFloat()

        return NoiseAdaptationMetrics(
            adaptationRate = overallAdaptation,
            adaptationSegments = adaptationScores,
            adaptationQuality = when {
                overallAdaptation > 0.3f -> "Good"
                overallAdaptation > 0.1f -> "Moderate"
                overallAdaptation > -0.1f -> "Poor"
                else -> "Very Poor"
            }
        )
    }

    private fun calculateCumulativeNoiseImpact(noiseEvents: List<NoiseEvent>, sessionDuration: Long): CumulativeNoiseImpact {
        var cumulativeScore = 0f
        val impactOverTime = mutableListOf<Float>()

        val sortedEvents = noiseEvents.sortedBy { it.timestamp }
        var currentImpact = 0f

        for (event in sortedEvents) {
            val eventImpact = when {
                event.decibelLevel > CRITICAL_DISRUPTION_DB -> 0.8f
                event.decibelLevel > SEVERE_DISRUPTION_DB -> 0.6f
                event.decibelLevel > MODERATE_DISRUPTION_DB -> 0.4f
                event.decibelLevel > LIGHT_DISRUPTION_DB -> 0.2f
                else -> 0.1f
            }

            currentImpact += eventImpact
            cumulativeScore += eventImpact
            impactOverTime.add(currentImpact)
        }

        val averageImpact = cumulativeScore / noiseEvents.size
        val peakImpact = impactOverTime.maxOrNull() ?: 0f

        return CumulativeNoiseImpact(
            totalCumulativeScore = cumulativeScore,
            averageImpact = averageImpact,
            peakImpact = peakImpact,
            impactProgression = impactOverTime
        )
    }

    // ========== HELPER METHODS - CORRELATION ANALYSIS ==========

    private fun calculateTemporalCorrelation(
        noiseEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent>
    ): Float {
        if (noiseEvents.isEmpty() || movementEvents.isEmpty()) return 0f

        var correlationSum = 0f
        var correlationCount = 0

        for (noiseEvent in noiseEvents) {
            // Find movements within correlation window
            val correlatedMovements = movementEvents.filter { movement ->
                val timeDiff = abs(movement.timestamp - noiseEvent.timestamp)
                timeDiff <= CORRELATION_TIME_WINDOW
            }

            if (correlatedMovements.isNotEmpty()) {
                val avgMovementIntensity = correlatedMovements.map { it.intensity }.average().toFloat()
                val normalizedNoise = noiseEvent.decibelLevel / 100f // Normalize to 0-1
                val normalizedMovement = avgMovementIntensity / 10f // Normalize to 0-1

                correlationSum += normalizedNoise * normalizedMovement
                correlationCount++
            }
        }

        return if (correlationCount > 0) correlationSum / correlationCount else 0f
    }

    private fun analyzeNoiseTriggeredMovements(
        noiseEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent>
    ): NoiseTriggeredMovementAnalysis {
        val triggeredMovements = mutableListOf<TriggeredMovement>()

        for (noiseEvent in noiseEvents.filter { it.isDisruptive() }) {
            // Find movements that occur after noise with appropriate delay
            val subsequentMovements = movementEvents.filter { movement ->
                val timeDiff = movement.timestamp - noiseEvent.timestamp
                timeDiff in 0..MOVEMENT_RESPONSE_DELAY
            }

            subsequentMovements.forEach { movement ->
                triggeredMovements.add(
                    TriggeredMovement(
                        noiseEvent = noiseEvent,
                        movementEvent = movement,
                        responseDelay = movement.timestamp - noiseEvent.timestamp,
                        correlationStrength = calculateEventCorrelation(noiseEvent, movement)
                    )
                )
            }
        }

        val triggerRate = triggeredMovements.size.toFloat() / noiseEvents.count { it.isDisruptive() }
        val averageResponseDelay = if (triggeredMovements.isNotEmpty()) {
            triggeredMovements.map { it.responseDelay }.average()
        } else 0.0

        return NoiseTriggeredMovementAnalysis(
            triggeredMovements = triggeredMovements,
            triggerRate = triggerRate,
            averageResponseDelay = averageResponseDelay.toLong(),
            strongCorrelations = triggeredMovements.count { it.correlationStrength > 0.7f }
        )
    }

    private fun calculateMovementResponseDelays(
        noiseEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent>
    ): ResponseDelayAnalysis {
        val delays = mutableListOf<Long>()

        for (noiseEvent in noiseEvents.filter { it.isDisruptive() }) {
            val nextMovement = movementEvents
                .filter { it.timestamp > noiseEvent.timestamp }
                .minByOrNull { it.timestamp - noiseEvent.timestamp }

            nextMovement?.let { movement ->
                val delay = movement.timestamp - noiseEvent.timestamp
                if (delay <= MOVEMENT_RESPONSE_DELAY) {
                    delays.add(delay)
                }
            }
        }

        return ResponseDelayAnalysis(
            responseDelays = delays,
            averageDelay = if (delays.isNotEmpty()) delays.average().toLong() else 0L,
            medianDelay = if (delays.isNotEmpty()) delays.sorted()[delays.size / 2] else 0L,
            delayVariability = if (delays.isNotEmpty()) calculateVariance(delays.map { it.toFloat() }) else 0f
        )
    }

    private fun analyzeIntensityCorrelation(
        noiseEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent>
    ): Float {
        val correlationPairs = mutableListOf<Pair<Float, Float>>()

        for (noiseEvent in noiseEvents) {
            val nearbyMovements = movementEvents.filter { movement ->
                abs(movement.timestamp - noiseEvent.timestamp) <= CORRELATION_TIME_WINDOW
            }

            if (nearbyMovements.isNotEmpty()) {
                val avgMovementIntensity = nearbyMovements.map { it.intensity }.average().toFloat()
                correlationPairs.add(Pair(noiseEvent.decibelLevel, avgMovementIntensity))
            }
        }

        return if (correlationPairs.size >= 3) {
            calculatePearsonCorrelation(correlationPairs)
        } else 0f
    }

    // ========== HELPER METHODS - DISRUPTION ANALYSIS ==========

    private fun identifyDisruptiveNoiseEvents(noiseEvents: List<NoiseEvent>): List<NoiseEvent> {
        return noiseEvents.filter { event ->
            event.isDisruptive() || event.decibelLevel > DISRUPTIVE_THRESHOLD
        }
    }

    private fun calculateDisruptionIntensity(disruptiveEvents: List<NoiseEvent>): DisruptionIntensity {
        if (disruptiveEvents.isEmpty()) return DisruptionIntensity()

        val avgIntensity = disruptiveEvents.map { it.decibelLevel }.average().toFloat()
        val maxIntensity = disruptiveEvents.maxByOrNull { it.decibelLevel }?.decibelLevel ?: 0f
        val intensityVariance = calculateVariance(disruptiveEvents.map { it.decibelLevel })

        val severityDistribution = mapOf(
            DisruptionSeverity.LIGHT to disruptiveEvents.count { it.decibelLevel <= MODERATE_DISRUPTION_DB },
            DisruptionSeverity.MODERATE to disruptiveEvents.count {
                it.decibelLevel > MODERATE_DISRUPTION_DB && it.decibelLevel <= SEVERE_DISRUPTION_DB
            },
            DisruptionSeverity.SEVERE to disruptiveEvents.count {
                it.decibelLevel > SEVERE_DISRUPTION_DB && it.decibelLevel <= CRITICAL_DISRUPTION_DB
            },
            DisruptionSeverity.CRITICAL to disruptiveEvents.count { it.decibelLevel > CRITICAL_DISRUPTION_DB }
        )

        return DisruptionIntensity(
            averageIntensity = avgIntensity,
            maxIntensity = maxIntensity,
            intensityVariance = intensityVariance,
            severityDistribution = severityDistribution
        )
    }

    private fun analyzeDisruptionTiming(
        disruptiveEvents: List<NoiseEvent>,
        sleepPhases: List<PhaseTransition>,
        sessionDuration: Long
    ): DisruptionTimingAnalysis {
        // Analyze when disruptions occur during sleep
        val hourlyDistribution = analyzeHourlyDisruptionDistribution(disruptiveEvents, sessionDuration)
        val phaseDistribution = analyzePhaseDisruptionDistribution(disruptiveEvents, sleepPhases)
        val criticalTimingEvents = identifyCriticalTimingEvents(disruptiveEvents, sleepPhases)

        return DisruptionTimingAnalysis(
            hourlyDistribution = hourlyDistribution,
            phaseDistribution = phaseDistribution,
            criticalTimingEvents = criticalTimingEvents,
            worstHour = hourlyDistribution.indexOf(hourlyDistribution.maxOrNull() ?: 0f),
            worstPhase = phaseDistribution.maxByOrNull { it.value }?.key ?: SleepPhase.UNKNOWN
        )
    }

    private fun calculateRecoveryMetrics(
        disruptiveEvents: List<NoiseEvent>,
        movementEvents: List<MovementEvent>
    ): RecoveryMetrics {
        val recoveryTimes = mutableListOf<Long>()

        for (disruption in disruptiveEvents) {
            // Find when movement settles after disruption
            val subsequentMovements = movementEvents
                .filter { it.timestamp > disruption.timestamp }
                .take(10) // Look at next 10 movements

            if (subsequentMovements.isNotEmpty()) {
                val settlingTime = findMovementSettlingTime(subsequentMovements)
                settlingTime?.let { recoveryTimes.add(it) }
            }
        }

        return RecoveryMetrics(
            recoveryTimes = recoveryTimes,
            averageRecoveryTime = if (recoveryTimes.isNotEmpty()) recoveryTimes.average().toLong() else 0L,
            quickRecoveries = recoveryTimes.count { it <= 60000L }, // Under 1 minute
            slowRecoveries = recoveryTimes.count { it > 300000L } // Over 5 minutes
        )
    }

    private fun assessCumulativeDisruption(disruptiveEvents: List<NoiseEvent>, sessionDuration: Long): Float {
        if (disruptiveEvents.isEmpty()) return 0f

        var cumulativeScore = 0f
        val timeWindows = sessionDuration / (30 * 60 * 1000L) // 30-minute windows

        for (window in 0 until timeWindows.toInt()) {
            val windowStart = window * 30 * 60 * 1000L
            val windowEnd = windowStart + 30 * 60 * 1000L

            val windowEvents = disruptiveEvents.filter {
                it.timestamp >= windowStart && it.timestamp < windowEnd
            }

            val windowScore = windowEvents.sumOf {
                when {
                    it.decibelLevel > CRITICAL_DISRUPTION_DB -> 3.0
                    it.decibelLevel > SEVERE_DISRUPTION_DB -> 2.0
                    it.decibelLevel > MODERATE_DISRUPTION_DB -> 1.0
                    else -> 0.5
                }
            }.toFloat()

            cumulativeScore += windowScore
        }

        return cumulativeScore
    }

    private fun calculateOverallDisruptionScore(
        intensity: DisruptionIntensity,
        frequency: Float,
        cumulative: Float,
        timing: DisruptionTimingAnalysis
    ): Float {
        val intensityScore = (intensity.averageIntensity / 10f).coerceIn(0f, 10f)
        val frequencyScore = (frequency / 5f).coerceIn(0f, 10f) // 5 disruptions per hour = max score
        val cumulativeScore = (cumulative / 10f).coerceIn(0f, 10f)
        val timingPenalty = if (timing.criticalTimingEvents > 0) 2f else 0f

        return (intensityScore * 0.4f + frequencyScore * 0.3f + cumulativeScore * 0.2f + timingPenalty * 0.1f)
            .coerceIn(0f, 10f)
    }

    // ========== CLUSTERING METHODS ==========

    private fun performTemporalDecibelClustering(noiseEvents: List<NoiseEvent>): List<NoiseCluster> {
        val clusters = mutableListOf<NoiseCluster>()
        val sortedEvents = noiseEvents.sortedBy { it.timestamp }

        var currentCluster = mutableListOf<NoiseEvent>()
        var lastTimestamp = 0L
        var lastDecibel = 0f

        for (event in sortedEvents) {
            val timeDiff = event.timestamp - lastTimestamp
            val decibelDiff = abs(event.decibelLevel - lastDecibel)

            val isTemporallyClose = timeDiff <= CLUSTER_TIME_THRESHOLD
            val isDecibelSimilar = decibelDiff <= CLUSTER_DECIBEL_THRESHOLD

            if (currentCluster.isEmpty() || (!isTemporallyClose && !isDecibelSimilar)) {
                // Start new cluster
                if (currentCluster.size >= MIN_CLUSTER_SIZE) {
                    clusters.add(createNoiseCluster(currentCluster))
                }
                currentCluster = mutableListOf(event)
            } else {
                // Add to current cluster
                currentCluster.add(event)
            }

            lastTimestamp = event.timestamp
            lastDecibel = event.decibelLevel
        }

        // Handle final cluster
        if (currentCluster.size >= MIN_CLUSTER_SIZE) {
            clusters.add(createNoiseCluster(currentCluster))
        }

        return clusters
    }

    private fun performIntensityPatternClustering(noiseEvents: List<NoiseEvent>): List<NoiseCluster> {
        // Group by intensity patterns
        val intensityRanges = mapOf(
            "very_quiet" to (0f..30f),
            "quiet" to (30f..40f),
            "moderate" to (40f..50f),
            "loud" to (50f..65f),
            "very_loud" to (65f..100f)
        )

        val clusters = mutableListOf<NoiseCluster>()

        for ((_, range) in intensityRanges) {
            val rangeEvents = noiseEvents.filter { it.decibelLevel in range }
            if (rangeEvents.size >= MIN_CLUSTER_SIZE) {
                clusters.add(createNoiseCluster(rangeEvents))
            }
        }

        return clusters
    }

    private fun performEnvironmentalClustering(noiseEvents: List<NoiseEvent>): List<NoiseCluster> {
        // Cluster based on environmental patterns
        return performTemporalDecibelClustering(noiseEvents) // Simplified for now
    }

    private fun performAdaptiveClustering(noiseEvents: List<NoiseEvent>): List<NoiseCluster> {
        // Adaptive clustering based on data characteristics
        val temporalClusters = performTemporalDecibelClustering(noiseEvents)
        val intensityClusters = performIntensityPatternClustering(noiseEvents)

        // Merge and optimize clusters
        return mergeNoiseClusters(temporalClusters, intensityClusters)
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

    private fun calculateQualityImpact(avgLevel: Float, frequency: Float, stdDev: Float): Float {
        val levelImpact = when {
            avgLevel <= 30f -> 1f
            avgLevel <= 40f -> 3f
            avgLevel <= 50f -> 5f
            avgLevel <= 60f -> 7f
            else -> 9f
        }

        val frequencyImpact = (frequency / 20f).coerceAtMost(3f)
        val variabilityImpact = (stdDev / 10f).coerceAtMost(2f)

        return (levelImpact + frequencyImpact + variabilityImpact).coerceIn(1f, 10f)
    }

    private fun getNoiseGrade(avgLevel: Float): String {
        return when {
            avgLevel <= 30f -> "Excellent (Very Quiet)"
            avgLevel <= 40f -> "Very Good (Quiet)"
            avgLevel <= 50f -> "Good (Moderate)"
            avgLevel <= 60f -> "Fair (Loud)"
            else -> "Poor (Very Loud)"
        }
    }

    private fun generateNoiseRecommendations(
        avgLevel: Float,
        profile: EnvironmentalProfile,
        patterns: NoisePatternAnalysis
    ): List<String> {
        val recommendations = mutableListOf<String>()

        when {
            avgLevel <= 35f -> recommendations.add("Excellent noise environment - maintain current conditions")
            avgLevel <= 45f -> recommendations.add("Good noise control - consider minor improvements")
            avgLevel <= 55f -> recommendations.add("Moderate noise levels - use white noise or earplugs")
            else -> recommendations.add("High noise levels - consider noise reduction measures")
        }

        when (profile.environmentType) {
            EnvironmentType.URBAN, EnvironmentType.NOISY_URBAN -> {
                recommendations.add("Urban environment detected - consider white noise machine or earplugs")
            }
            EnvironmentType.MODERATE_RESIDENTIAL -> {
                recommendations.add("Residential noise - check for controllable sources like HVAC or appliances")
            }
            else -> { /* Quiet environments need no additional recommendations */ }
        }

        if (patterns.noiseBursts.size > 3) {
            recommendations.add("Frequent noise bursts detected - identify and minimize sudden noise sources")
        }

        return recommendations
    }

    private fun getDominantNoiseCategory(vq: Int, q: Int, m: Int, l: Int, vl: Int, e: Int): String {
        val categories = mapOf(
            "Very Quiet" to vq,
            "Quiet" to q,
            "Moderate" to m,
            "Loud" to l,
            "Very Loud" to vl,
            "Excessive" to e
        )
        return categories.maxByOrNull { it.value }?.key ?: "Unknown"
    }

    private fun analyzeTemporalNoiseDistribution(noiseEvents: List<NoiseEvent>, sessionDuration: Long): List<Float> {
        val hourCount = maxOf(1, (sessionDuration / (60 * 60 * 1000L)).toInt())
        val hourlyLevels = mutableListOf<Float>()

        val sessionStart = noiseEvents.minByOrNull { it.timestamp }?.timestamp ?: 0L
        val hourDuration = sessionDuration / hourCount

        for (hour in 0 until hourCount) {
            val hourStart = sessionStart + (hour * hourDuration)
            val hourEnd = hourStart + hourDuration

            val hourEvents = noiseEvents.filter { it.timestamp >= hourStart && it.timestamp < hourEnd }
            val avgLevel = if (hourEvents.isNotEmpty()) {
                hourEvents.map { it.decibelLevel }.average().toFloat()
            } else 0f

            hourlyLevels.add(avgLevel)
        }

        return hourlyLevels
    }

    private fun detectPeriodicNoisePatterns(noiseEvents: List<NoiseEvent>): List<PeriodicPattern> {
        // Simplified periodic pattern detection
        val patterns = mutableListOf<PeriodicPattern>()

        // Look for recurring intervals
        val intervals = mutableListOf<Long>()
        for (i in 1 until noiseEvents.size) {
            intervals.add(noiseEvents[i].timestamp - noiseEvents[i - 1].timestamp)
        }

        // Find common intervals (simplified)
        val intervalCounts = intervals.groupingBy { it / 60000L * 60000L }.eachCount() // Group by minute

        intervalCounts.forEach { (interval, count) ->
            if (count >= 3 && interval > 0) { // At least 3 occurrences
                patterns.add(
                    PeriodicPattern(
                        interval = interval,
                        occurrences = count,
                        confidence = (count.toFloat() / intervals.size).coerceAtMost(1f)
                    )
                )
            }
        }

        return patterns
    }

    private fun detectNoiseBursts(noiseEvents: List<NoiseEvent>): List<NoiseBurst> {
        val bursts = mutableListOf<NoiseBurst>()
        var currentBurst: MutableList<NoiseEvent>? = null

        for (event in noiseEvents.sortedBy { it.timestamp }) {
            val isLoud = event.decibelLevel > LOUD_THRESHOLD

            if (isLoud) {
                if (currentBurst == null) {
                    currentBurst = mutableListOf(event)
                } else {
                    val lastEvent = currentBurst.last()
                    if (event.timestamp - lastEvent.timestamp <= CLUSTER_TIME_THRESHOLD) {
                        currentBurst.add(event)
                    } else {
                        // End current burst, start new one
                        if (currentBurst.size >= 2) {
                            bursts.add(createNoiseBurst(currentBurst))
                        }
                        currentBurst = mutableListOf(event)
                    }
                }
            } else {
                // End current burst if exists
                currentBurst?.let { burst ->
                    if (burst.size >= 2) {
                        bursts.add(createNoiseBurst(burst))
                    }
                }
                currentBurst = null
            }
        }

        // Handle final burst
        currentBurst?.let { burst ->
            if (burst.size >= 2) {
                bursts.add(createNoiseBurst(burst))
            }
        }

        return bursts
    }

    private fun calculateNoisePatternConsistency(noiseEvents: List<NoiseEvent>): Float {
        if (noiseEvents.size < 3) return 0f

        val levels = noiseEvents.map { it.decibelLevel }
        val variance = calculateVariance(levels)
        val mean = levels.average().toFloat()

        val coefficientOfVariation = if (mean > 0) sqrt(variance) / mean else 0f
        return (1f - coefficientOfVariation).coerceIn(0f, 1f)
    }

    private fun identifyDominantNoisePattern(
        periodicPatterns: List<PeriodicPattern>,
        noiseBursts: List<NoiseBurst>
    ): String {
        return when {
            periodicPatterns.isNotEmpty() && periodicPatterns.first().confidence > 0.5f -> "Periodic"
            noiseBursts.size > 3 -> "Burst"
            else -> "Random"
        }
    }

    private fun calculateEnvironmentScore(avgLevel: Float, variability: Float, frequency: Float): Float {
        val levelScore = (100f - avgLevel * 1.5f).coerceIn(0f, 100f)
        val variabilityScore = (100f - variability).coerceIn(0f, 100f)
        val frequencyScore = (100f - frequency * 2f).coerceIn(0f, 100f)

        return (levelScore * 0.5f + variabilityScore * 0.3f + frequencyScore * 0.2f) / 10f
    }

    private fun getEnvironmentalGrade(score: Float): String {
        return when {
            score >= 8f -> "Excellent"
            score >= 6f -> "Good"
            score >= 4f -> "Fair"
            score >= 2f -> "Poor"
            else -> "Very Poor"
        }
    }

    private fun identifyPrimaryNoiseSource(sources: List<NoiseSource>): NoiseSourceType {
        return sources.maxByOrNull { it.impactScore }?.type ?: NoiseSourceType.UNKNOWN
    }

    private fun generateEnvironmentalRecommendations(
        score: Float,
        sources: List<NoiseSource>,
        phaseImpact: Map<SleepPhase, PhaseNoiseImpact>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (score > 6f) {
            recommendations.add("High environmental noise impact - consider noise reduction strategies")
        }

        sources.forEach { source ->
            when (source.type) {
                NoiseSourceType.CONTINUOUS -> recommendations.add("Continuous noise detected - consider white noise masking")
                NoiseSourceType.SUDDEN_LOUD -> recommendations.add("Sudden loud noises - identify and minimize sources")
                NoiseSourceType.MACHINERY -> recommendations.add("Machinery noise - check HVAC and appliance settings")
                else -> {}
            }
        }

        return recommendations
    }

    private fun calculatePearsonCorrelation(pairs: List<Pair<Float, Float>>): Float {
        if (pairs.size < 3) return 0f

        val n = pairs.size
        val sumX = pairs.sumOf { it.first.toDouble() }
        val sumY = pairs.sumOf { it.second.toDouble() }
        val sumXY = pairs.sumOf { it.first * it.second.toDouble() }
        val sumXX = pairs.sumOf { it.first * it.first.toDouble() }
        val sumYY = pairs.sumOf { it.second * it.second.toDouble() }

        val numerator = n * sumXY - sumX * sumY
        val denominator = sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY))

        return if (denominator != 0.0) (numerator / denominator).toFloat() else 0f
    }

    private fun calculateEventCorrelation(noiseEvent: NoiseEvent, movementEvent: MovementEvent): Float {
        val normalizedNoise = (noiseEvent.decibelLevel / 100f).coerceIn(0f, 1f)
        val normalizedMovement = (movementEvent.intensity / 10f).coerceIn(0f, 1f)
        val timeFactor = 1f - (abs(movementEvent.timestamp - noiseEvent.timestamp).toFloat() / MOVEMENT_RESPONSE_DELAY)

        return (normalizedNoise * normalizedMovement * timeFactor).coerceIn(0f, 1f)
    }

    private fun getDisruptionGrade(score: Float): String {
        return when {
            score <= 2f -> "Minimal Disruption"
            score <= 4f -> "Low Disruption"
            score <= 6f -> "Moderate Disruption"
            score <= 8f -> "High Disruption"
            else -> "Severe Disruption"
        }
    }

    private fun createNoiseCluster(events: List<NoiseEvent>): NoiseCluster {
        return NoiseCluster(
            id = events.hashCode().toLong(),
            startTime = events.minByOrNull { it.timestamp }?.timestamp ?: 0L,
            endTime = events.maxByOrNull { it.timestamp }?.timestamp ?: 0L,
            events = events,
            averageLevel = events.map { it.decibelLevel }.average().toFloat(),
            peakLevel = events.maxByOrNull { it.decibelLevel }?.decibelLevel ?: 0f,
            duration = (events.maxByOrNull { it.timestamp }?.timestamp ?: 0L) -
                    (events.minByOrNull { it.timestamp }?.timestamp ?: 0L)
        )
    }

    private fun createNoiseBurst(events: List<NoiseEvent>): NoiseBurst {
        return NoiseBurst(
            startTime = events.minByOrNull { it.timestamp }?.timestamp ?: 0L,
            endTime = events.maxByOrNull { it.timestamp }?.timestamp ?: 0L,
            duration = (events.maxByOrNull { it.timestamp }?.timestamp ?: 0L) -
                    (events.minByOrNull { it.timestamp }?.timestamp ?: 0L),
            events = events,
            peakLevel = events.maxByOrNull { it.decibelLevel }?.decibelLevel ?: 0f,
            averageLevel = events.map { it.decibelLevel }.average().toFloat()
        )
    }

    // Additional helper methods would continue here...
    // [Implementation continues with remaining helper methods]
}

// ========== DATA CLASSES FOR ANALYSIS RESULTS ==========

data class NoiseLevelAnalysis(
    val averageLevel: Float = 0f,
    val maxLevel: Float = 0f,
    val minLevel: Float = 0f,
    val medianLevel: Float = 0f,
    val standardDeviation: Float = 0f,
    val categorization: NoiseCategorization = NoiseCategorization(),
    val noiseFrequency: Float = 0f,
    val patterns: NoisePatternAnalysis = NoisePatternAnalysis(),
    val consistency: NoiseConsistency = NoiseConsistency(),
    val adaptiveThresholds: AdaptiveNoiseThresholds = AdaptiveNoiseThresholds(),
    val environmentalProfile: EnvironmentalProfile = EnvironmentalProfile(),
    val noiseGrade: String = "Unknown",
    val qualityImpact: Float = 0f,
    val recommendations: List<String> = emptyList()
)

data class NoiseCategorization(
    val veryQuiet: Int = 0,
    val quiet: Int = 0,
    val moderate: Int = 0,
    val loud: Int = 0,
    val veryLoud: Int = 0,
    val excessive: Int = 0,
    val disruptive: Int = 0,
    val disruptiveRatio: Float = 0f,
    val dominantCategory: String = "Unknown"
)

data class NoisePatternAnalysis(
    val temporalDistribution: List<Float> = emptyList(),
    val periodicPatterns: List<PeriodicPattern> = emptyList(),
    val noiseBursts: List<NoiseBurst> = emptyList(),
    val patternConsistency: Float = 0f,
    val dominantPattern: String = "Unknown"
)

data class NoiseConsistency(
    val levelConsistency: Float = 0f,
    val temporalConsistency: Float = 0f,
    val overallConsistency: Float = 0f
)

data class AdaptiveNoiseThresholds(
    val veryQuietThreshold: Float = 25f,
    val quietThreshold: Float = 35f,
    val moderateThreshold: Float = 45f,
    val loudThreshold: Float = 55f,
    val veryLoudThreshold: Float = 65f,
    val disruptiveThreshold: Float = 50f
)

data class EnvironmentalProfile(
    val environmentType: EnvironmentType = EnvironmentType.UNKNOWN,
    val baselineNoiseLevel: Float = 0f,
    val peakNoiseLevel: Float = 0f,
    val noiseVariability: Float = 0f,
    val eventFrequency: Float = 0f,
    val environmentScore: Float = 0f
)

data class RealTimeNoiseData(
    val currentLevel: Float = 0f,
    val smoothedLevel: Float = 0f,
    val peakLevel: Float = 0f,
    val noiseVariability: Float = 0f,
    val trend: NoiseTrend = NoiseTrend.STABLE,
    val environmentType: EnvironmentType = EnvironmentType.UNKNOWN,
    val sleepImpact: SleepImpactLevel = SleepImpactLevel.MINIMAL,
    val prediction: NoisePrediction = NoisePrediction(),
    val timestamp: Long = 0L,
    val confidence: Float = 0f
)

data class NoisePrediction(
    val predictedLevel: Float = 0f,
    val trend: NoiseTrend = NoiseTrend.STABLE,
    val confidence: Float = 0f,
    val timeHorizon: Long = 0L
)

data class EnvironmentalNoiseImpact(
    val overallImpactScore: Float = 0f,
    val phaseImpactAnalysis: Map<SleepPhase, PhaseNoiseImpact> = emptyMap(),
    val identifiedNoiseSources: List<NoiseSource> = emptyList(),
    val adaptationMetrics: NoiseAdaptationMetrics = NoiseAdaptationMetrics(),
    val cumulativeImpact: CumulativeNoiseImpact = CumulativeNoiseImpact(),
    val environmentalGrade: String = "Unknown",
    val primaryNoiseSource: NoiseSourceType = NoiseSourceType.UNKNOWN,
    val recommendations: List<String> = emptyList()
)

data class PhaseNoiseImpact(
    val averageNoiseLevel: Float = 0f,
    val disruptiveEvents: Int = 0,
    val impactScore: Float = 0f
)

data class NoiseSource(
    val type: NoiseSourceType,
    val averageLevel: Float,
    val frequency: Int,
    val duration: Long,
    val impactScore: Float
)

data class NoiseAdaptationMetrics(
    val adaptationRate: Float = 0f,
    val adaptationSegments: List<Float> = emptyList(),
    val adaptationQuality: String = "Unknown"
)

data class CumulativeNoiseImpact(
    val totalCumulativeScore: Float = 0f,
    val averageImpact: Float = 0f,
    val peakImpact: Float = 0f,
    val impactProgression: List<Float> = emptyList()
)

data class NoiseMovementCorrelation(
    val temporalCorrelation: Float = 0f,
    val intensityCorrelation: Float = 0f,
    val triggeredMovements: NoiseTriggeredMovementAnalysis = NoiseTriggeredMovementAnalysis(),
    val responseDelays: ResponseDelayAnalysis = ResponseDelayAnalysis(),
    val patternCorrelations: List<String> = emptyList(),
    val phaseCorrelations: Map<SleepPhase, Float> = emptyMap(),
    val statisticalSignificance: Float = 0f,
    val correlationStrength: String = "None",
    val insights: List<String> = emptyList()
)

data class NoiseTriggeredMovementAnalysis(
    val triggeredMovements: List<TriggeredMovement> = emptyList(),
    val triggerRate: Float = 0f,
    val averageResponseDelay: Long = 0L,
    val strongCorrelations: Int = 0
)

data class TriggeredMovement(
    val noiseEvent: NoiseEvent,
    val movementEvent: MovementEvent,
    val responseDelay: Long,
    val correlationStrength: Float
)

data class ResponseDelayAnalysis(
    val responseDelays: List<Long> = emptyList(),
    val averageDelay: Long = 0L,
    val medianDelay: Long = 0L,
    val delayVariability: Float = 0f
)

data class NoiseDisruptionAnalysis(
    val overallDisruptionScore: Float = 0f,
    val disruptiveEvents: List<NoiseEvent> = emptyList(),
    val disruptionIntensity: DisruptionIntensity = DisruptionIntensity(),
    val disruptionFrequency: Float = 0f,
    val timingAnalysis: DisruptionTimingAnalysis = DisruptionTimingAnalysis(),
    val recoveryMetrics: RecoveryMetrics = RecoveryMetrics(),
    val cumulativeDisruption: Float = 0f,
    val disruptionGrade: String = "Unknown",
    val primaryDisruptionSource: String = "Unknown",
    val insights: List<String> = emptyList(),
    val recommendations: List<String> = emptyList()
)

data class DisruptionIntensity(
    val averageIntensity: Float = 0f,
    val maxIntensity: Float = 0f,
    val intensityVariance: Float = 0f,
    val severityDistribution: Map<DisruptionSeverity, Int> = emptyMap()
)

data class DisruptionTimingAnalysis(
    val hourlyDistribution: List<Float> = emptyList(),
    val phaseDistribution: Map<SleepPhase, Int> = emptyMap(),
    val criticalTimingEvents: Int = 0,
    val worstHour: Int = 0,
    val worstPhase: SleepPhase = SleepPhase.UNKNOWN
)

data class RecoveryMetrics(
    val recoveryTimes: List<Long> = emptyList(),
    val averageRecoveryTime: Long = 0L,
    val quickRecoveries: Int = 0,
    val slowRecoveries: Int = 0
)

data class NoiseClusterAnalysis(
    val clusters: List<NoiseCluster> = emptyList(),
    val clusterCharacteristics: String = "Unknown",
    val noisePatterns: List<String> = emptyList(),
    val clusterQuality: Float = 0f,
    val environmentalInsights: List<String> = emptyList(),
    val algorithm: NoiseClusteringAlgorithm = NoiseClusteringAlgorithm.TEMPORAL_DECIBEL,
    val recommendations: List<String> = emptyList()
)

data class NoiseCluster(
    val id: Long,
    val startTime: Long,
    val endTime: Long,
    val events: List<NoiseEvent>,
    val averageLevel: Float,
    val peakLevel: Float,
    val duration: Long
)

data class PeriodicPattern(
    val interval: Long,
    val occurrences: Int,
    val confidence: Float
)

data class NoiseBurst(
    val startTime: Long,
    val endTime: Long,
    val duration: Long,
    val events: List<NoiseEvent>,
    val peakLevel: Float,
    val averageLevel: Float
)

// ========== ENUMS ==========

enum class NoiseTrend {
    INCREASING,
    DECREASING,
    STABLE
}

enum class EnvironmentType {
    VERY_QUIET,
    QUIET_RESIDENTIAL,
    MODERATE_RESIDENTIAL,
    URBAN,
    NOISY_URBAN,
    UNKNOWN
}

enum class SleepImpactLevel {
    MINIMAL,
    LOW,
    MODERATE,
    HIGH,
    SEVERE
}

enum class NoiseSourceType {
    CONTINUOUS,
    SUDDEN_LOUD,
    MACHINERY,
    ENVIRONMENTAL,
    UNKNOWN
}

enum class DisruptionSeverity {
    LIGHT,
    MODERATE,
    SEVERE,
    CRITICAL
}

enum class NoiseClusteringAlgorithm {
    TEMPORAL_DECIBEL,
    INTENSITY_PATTERN,
    ENVIRONMENTAL,
    ADAPTIVE
}