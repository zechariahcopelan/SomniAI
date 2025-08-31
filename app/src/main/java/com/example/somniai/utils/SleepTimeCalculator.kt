package com.example.somniai.utils

import kotlin.math.*
import java.util.*

/**
 * Sleep Time Calculator for SomniAI
 *
 * Specialized calculations for sleep efficiency, phase analysis, and timing optimization.
 * Uses evidence-based sleep science methodologies for accurate analysis.
 */
object SleepTimeCalculator {

    // ========== SLEEP EFFICIENCY CALCULATIONS ==========

    /**
     * Calculate sleep efficiency using standard methodology
     * Sleep Efficiency = (Total Sleep Time / Time in Bed) × 100
     *
     * @param totalSleepTimeMs Actual time spent sleeping
     * @param timeInBedMs Total time from bedtime to wake time
     * @return Sleep efficiency percentage (0-100)
     */
    fun calculateSleepEfficiency(totalSleepTimeMs: Long, timeInBedMs: Long): Float {
        return if (timeInBedMs > 0) {
            (totalSleepTimeMs.toFloat() / timeInBedMs * 100f).coerceIn(0f, 100f)
        } else 0f
    }

    /**
     * Calculate sleep efficiency with wake periods excluded
     * More accurate when accounting for multiple wake episodes
     */
    fun calculateSleepEfficiencyAdvanced(
        sleepPeriods: List<Pair<Long, Long>>, // Start-End pairs of sleep periods
        bedTime: Long,
        wakeTime: Long
    ): Float {
        val totalTimeInBed = wakeTime - bedTime
        val totalSleepTime = sleepPeriods.sumOf { it.second - it.first }

        return calculateSleepEfficiency(totalSleepTime, totalTimeInBed)
    }

    /**
     * Categorize sleep efficiency based on clinical standards
     */
    fun categorizeEfficiency(efficiency: Float): String {
        return when {
            efficiency >= 95f -> "Excellent"
            efficiency >= 85f -> "Good"
            efficiency >= 75f -> "Fair"
            efficiency >= 65f -> "Poor"
            else -> "Very Poor"
        }
    }

    /**
     * Calculate sleep latency (time to fall asleep)
     * Normal range: 10-20 minutes
     */
    fun calculateSleepLatency(bedTime: Long, sleepOnsetTime: Long): Long {
        return maxOf(0L, sleepOnsetTime - bedTime)
    }

    /**
     * Calculate Wake After Sleep Onset (WASO)
     * Total wake time during sleep period after initial sleep onset
     */
    fun calculateWASO(
        sleepOnsetTime: Long,
        finalWakeTime: Long,
        wakePeriods: List<Pair<Long, Long>>
    ): Long {
        return wakePeriods.filter {
            it.first >= sleepOnsetTime && it.second <= finalWakeTime
        }.sumOf { it.second - it.first }
    }

    // ========== SLEEP PHASE DURATION CALCULATIONS ==========

    /**
     * Calculate sleep phase percentages based on durations
     */
    fun calculatePhasePercentages(
        lightSleepMs: Long,
        deepSleepMs: Long,
        remSleepMs: Long,
        awakeMs: Long = 0L
    ): Map<String, Float> {
        val totalSleep = lightSleepMs + deepSleepMs + remSleepMs + awakeMs

        return if (totalSleep > 0) {
            mapOf(
                "light" to (lightSleepMs.toFloat() / totalSleep * 100f),
                "deep" to (deepSleepMs.toFloat() / totalSleep * 100f),
                "rem" to (remSleepMs.toFloat() / totalSleep * 100f),
                "awake" to (awakeMs.toFloat() / totalSleep * 100f)
            )
        } else {
            mapOf("light" to 0f, "deep" to 0f, "rem" to 0f, "awake" to 0f)
        }
    }

    /**
     * Analyze sleep architecture quality based on phase distribution
     * Uses established sleep science guidelines
     */
    fun analyzeSleepArchitecture(
        lightSleepPercent: Float,
        deepSleepPercent: Float,
        remSleepPercent: Float,
        awakePercent: Float
    ): SleepArchitectureAnalysis {

        // Ideal ranges based on sleep research
        val idealRanges = mapOf(
            "light" to 45f..55f,
            "deep" to 15f..25f,
            "rem" to 20f..25f,
            "awake" to 0f..5f
        )

        val scores = mutableMapOf<String, Float>()
        val deviations = mutableMapOf<String, Float>()

        // Calculate how close each phase is to ideal range
        mapOf(
            "light" to lightSleepPercent,
            "deep" to deepSleepPercent,
            "rem" to remSleepPercent,
            "awake" to awakePercent
        ).forEach { (phase, actual) ->
            val ideal = idealRanges[phase]!!
            val deviation = when {
                actual in ideal -> 0f
                actual < ideal.start -> ideal.start - actual
                else -> actual - ideal.endInclusive
            }

            deviations[phase] = deviation
            scores[phase] = maxOf(0f, 100f - deviation * 2f) // Convert deviation to 0-100 score
        }

        val overallScore = scores.values.average().toFloat()

        return SleepArchitectureAnalysis(
            overallScore = overallScore,
            phaseScores = scores,
            deviations = deviations,
            recommendations = generateArchitectureRecommendations(deviations)
        )
    }

    /**
     * Calculate optimal sleep phase durations for target sleep duration
     */
    fun calculateOptimalPhaseDurations(totalSleepDurationMs: Long): Map<String, Long> {
        return mapOf(
            "light" to (totalSleepDurationMs * 0.50).toLong(),
            "deep" to (totalSleepDurationMs * 0.20).toLong(),
            "rem" to (totalSleepDurationMs * 0.25).toLong(),
            "awake" to (totalSleepDurationMs * 0.05).toLong()
        )
    }

    // ========== BEDTIME/WAKE TIME ANALYSIS ==========

    /**
     * Calculate ideal bedtime for target wake time and sleep duration
     * Includes buffer time for sleep latency
     */
    fun calculateIdealBedtime(
        targetWakeTime: Long,
        desiredSleepHours: Float = 8f,
        averageSleepLatencyMs: Long = 15 * 60 * 1000L // 15 minutes
    ): Long {
        val sleepDurationMs = (desiredSleepHours * 60 * 60 * 1000).toLong()
        return targetWakeTime - sleepDurationMs - averageSleepLatencyMs
    }

    /**
     * Calculate sleep timing consistency score
     * Measures how consistent bedtimes and wake times are
     */
    fun calculateTimingConsistency(
        bedtimes: List<Long>,
        waketimes: List<Long>
    ): TimingConsistencyResult {

        fun calculateVariation(times: List<Long>): Float {
            if (times.size < 2) return 0f

            val hoursOfDay = times.map { timestamp ->
                Calendar.getInstance().apply { timeInMillis = timestamp }
                    .get(Calendar.HOUR_OF_DAY) +
                        Calendar.getInstance().apply { timeInMillis = timestamp }
                            .get(Calendar.MINUTE) / 60f
            }

            val mean = hoursOfDay.average()
            val variance = hoursOfDay.map { (it - mean).pow(2) }.average()
            return sqrt(variance).toFloat()
        }

        val bedtimeVariation = calculateVariation(bedtimes)
        val waketimeVariation = calculateVariation(waketimes)

        // Convert to 0-1 consistency scores (lower variation = higher consistency)
        val bedtimeConsistency = (3f / (3f + bedtimeVariation)).coerceIn(0f, 1f)
        val waketimeConsistency = (3f / (3f + waketimeVariation)).coerceIn(0f, 1f)
        val overallConsistency = (bedtimeConsistency + waketimeConsistency) / 2f

        return TimingConsistencyResult(
            bedtimeConsistency = bedtimeConsistency,
            waketimeConsistency = waketimeConsistency,
            overallConsistency = overallConsistency,
            bedtimeVariationHours = bedtimeVariation,
            waketimeVariationHours = waketimeVariation
        )
    }

    /**
     * Analyze sleep timing patterns and provide recommendations
     */
    fun analyzeSleepTiming(
        bedtimes: List<Long>,
        waketimes: List<Long>,
        durations: List<Long>
    ): SleepTimingAnalysis {

        val consistencyResult = calculateTimingConsistency(bedtimes, waketimes)

        // Calculate average sleep and wake times
        val avgBedtimeHour = bedtimes.map {
            Calendar.getInstance().apply { timeInMillis = it }.get(Calendar.HOUR_OF_DAY)
        }.average()

        val avgWaketimeHour = waketimes.map {
            Calendar.getInstance().apply { timeInMillis = it }.get(Calendar.HOUR_OF_DAY)
        }.average()

        val avgDurationHours = durations.map { it / (1000f * 60 * 60) }.average()

        // Determine chronotype based on timing patterns
        val chronotype = determineChronotype(avgBedtimeHour, avgWaketimeHour)

        // Calculate sleep debt
        val targetSleepHours = 8f
        val avgSleepDebt = targetSleepHours - avgDurationHours.toFloat()

        return SleepTimingAnalysis(
            consistency = consistencyResult,
            averageBedtimeHour = avgBedtimeHour.toFloat(),
            averageWaketimeHour = avgWaketimeHour.toFloat(),
            averageSleepDuration = avgDurationHours.toFloat(),
            chronotype = chronotype,
            sleepDebt = maxOf(0f, avgSleepDebt),
            recommendations = generateTimingRecommendations(consistencyResult, avgBedtimeHour, avgSleepDebt)
        )
    }

    /**
     * Determine chronotype based on sleep timing preferences
     */
    fun determineChronotype(avgBedtime: Double, avgWaketime: Double): String {
        return when {
            avgBedtime <= 22.0 && avgWaketime <= 6.5 -> "Morning Lark"
            avgBedtime >= 24.0 && avgWaketime >= 8.0 -> "Night Owl"
            else -> "Intermediate"
        }
    }

    // ========== SLEEP CYCLE AND OPTIMIZATION ==========

    /**
     * Calculate sleep cycles and transitions
     * Standard cycle length: ~90 minutes
     */
    fun analyzeSleepCycles(
        sleepDurationMs: Long,
        phaseTransitions: List<Long> = emptyList()
    ): SleepCycleAnalysis {

        val cycleLength = 90 * 60 * 1000L // 90 minutes
        val expectedCycles = (sleepDurationMs / cycleLength).toInt()
        val actualTransitions = phaseTransitions.size

        // Calculate cycle efficiency based on transitions
        val cycleEfficiency = if (expectedCycles > 0) {
            val expectedTransitions = expectedCycles * 4 // Rough estimate of phase changes per cycle
            1f - abs(actualTransitions - expectedTransitions) / expectedTransitions.toFloat()
        } else 1f

        return SleepCycleAnalysis(
            totalCycles = expectedCycles,
            cycleLength = TimeUtils.msToMinutes(cycleLength),
            cycleEfficiency = cycleEfficiency.coerceIn(0f, 1f),
            phaseTransitions = actualTransitions,
            recommendations = generateCycleRecommendations(expectedCycles, cycleEfficiency)
        )
    }

    /**
     * Get optimal wake time windows based on sleep cycles
     * Avoids waking during deep sleep phases
     */
    fun getOptimalWakeWindows(
        sleepStartTime: Long,
        maxSleepHours: Float = 10f
    ): List<WakeWindow> {

        val cycleLength = 90 * 60 * 1000L // 90 minutes
        val windows = mutableListOf<WakeWindow>()

        // Start checking after minimum 4 hours of sleep
        var currentTime = sleepStartTime + (4 * 60 * 60 * 1000L)
        val maxTime = sleepStartTime + (maxSleepHours * 60 * 60 * 1000).toLong()

        while (currentTime <= maxTime) {
            val cyclesCompleted = ((currentTime - sleepStartTime) / cycleLength).toInt()
            val sleepHours = (currentTime - sleepStartTime) / (1000f * 60 * 60)

            // Optimal windows are at the end of sleep cycles (light sleep phase)
            windows.add(
                WakeWindow(
                    startTime = currentTime - (10 * 60 * 1000L), // 10 min window
                    endTime = currentTime + (10 * 60 * 1000L),
                    optimalTime = currentTime,
                    cyclesCompleted = cyclesCompleted,
                    totalSleepHours = sleepHours,
                    quality = calculateWakeQuality(sleepHours)
                )
            )

            currentTime += cycleLength
        }

        return windows.sortedByDescending { it.quality }
    }

    /**
     * Calculate how good a wake time is based on sleep duration and cycle completion
     */
    private fun calculateWakeQuality(sleepHours: Float): Float {
        return when {
            sleepHours in 7f..9f -> 1.0f // Optimal range
            sleepHours in 6f..7f || sleepHours in 9f..10f -> 0.8f
            sleepHours in 5f..6f || sleepHours in 10f..11f -> 0.6f
            sleepHours < 5f -> 0.3f
            else -> 0.4f // Too much sleep
        }
    }

    // ========== SLEEP DEBT CALCULATIONS ==========

    /**
     * Calculate cumulative sleep debt over time
     */
    fun calculateSleepDebt(
        actualSleepDurations: List<Long>, // in milliseconds
        targetSleepMs: Long = 8 * 60 * 60 * 1000L // 8 hours default
    ): SleepDebtAnalysis {

        var cumulativeDebt = 0L
        val dailyDebts = mutableListOf<Long>()

        actualSleepDurations.forEach { actual ->
            val dailyDebt = maxOf(0L, targetSleepMs - actual)
            dailyDebts.add(dailyDebt)
            cumulativeDebt += dailyDebt
        }

        val avgDailyDebt = if (dailyDebts.isNotEmpty()) dailyDebts.average().toLong() else 0L
        val debtSeverity = categorizeSleepDebt(cumulativeDebt)

        return SleepDebtAnalysis(
            cumulativeDebtMs = cumulativeDebt,
            averageDailyDebtMs = avgDailyDebt,
            dailyDebts = dailyDebts,
            severity = debtSeverity,
            recoveryDaysNeeded = calculateRecoveryDays(cumulativeDebt)
        )
    }

    private fun categorizeSleepDebt(debtMs: Long): String {
        val debtHours = debtMs / (1000f * 60 * 60)
        return when {
            debtHours < 2f -> "Minimal"
            debtHours < 5f -> "Moderate"
            debtHours < 10f -> "Significant"
            else -> "Severe"
        }
    }

    private fun calculateRecoveryDays(debtMs: Long): Int {
        // Rough estimate: can recover ~2 hours of sleep debt per day
        val recoveryRateMs = 2 * 60 * 60 * 1000L
        return ceil(debtMs.toFloat() / recoveryRateMs).toInt()
    }

    // ========== HELPER FUNCTIONS ==========

    private fun generateArchitectureRecommendations(deviations: Map<String, Float>): List<String> {
        val recommendations = mutableListOf<String>()

        deviations.forEach { (phase, deviation) ->
            if (deviation > 5f) {
                when (phase) {
                    "deep" -> recommendations.add("Consider avoiding caffeine after 2 PM and creating a cooler sleep environment to increase deep sleep")
                    "rem" -> recommendations.add("Try to maintain consistent sleep schedule and avoid alcohol before bedtime to improve REM sleep")
                    "light" -> recommendations.add("Your light sleep percentage is outside the normal range - consider sleep hygiene improvements")
                    "awake" -> recommendations.add("Reduce nighttime awakenings by minimizing noise and light disruptions")
                }
            }
        }

        return recommendations
    }

    private fun generateTimingRecommendations(
        consistency: TimingConsistencyResult,
        avgBedtime: Double,
        sleepDebt: Float
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (consistency.overallConsistency < 0.7f) {
            recommendations.add("Improve sleep schedule consistency by going to bed and waking up at the same time daily")
        }

        if (avgBedtime > 24.0) {
            recommendations.add("Consider earlier bedtime to align with natural circadian rhythms")
        }

        if (sleepDebt > 1f) {
            recommendations.add("Address sleep debt by extending sleep duration or improving sleep efficiency")
        }

        return recommendations
    }

    private fun generateCycleRecommendations(cycles: Int, efficiency: Float): List<String> {
        val recommendations = mutableListOf<String>()

        if (cycles < 4) {
            recommendations.add("Consider extending sleep duration to complete more sleep cycles")
        }

        if (efficiency < 0.7f) {
            recommendations.add("Reduce sleep disruptions to improve sleep cycle continuity")
        }

        return recommendations
    }
}

// ========== DATA CLASSES FOR RESULTS ==========

data class SleepArchitectureAnalysis(
    val overallScore: Float,
    val phaseScores: Map<String, Float>,
    val deviations: Map<String, Float>,
    val recommendations: List<String>
)

data class TimingConsistencyResult(
    val bedtimeConsistency: Float,
    val waketimeConsistency: Float,
    val overallConsistency: Float,
    val bedtimeVariationHours: Float,
    val waketimeVariationHours: Float
)

data class SleepTimingAnalysis(
    val consistency: TimingConsistencyResult,
    val averageBedtimeHour: Float,
    val averageWaketimeHour: Float,
    val averageSleepDuration: Float,
    val chronotype: String,
    val sleepDebt: Float,
    val recommendations: List<String>
)

data class SleepCycleAnalysis(
    val totalCycles: Int,
    val cycleLength: Int, // in minutes
    val cycleEfficiency: Float,
    val phaseTransitions: Int,
    val recommendations: List<String>
)

data class WakeWindow(
    val startTime: Long,
    val endTime: Long,
    val optimalTime: Long,
    val cyclesCompleted: Int,
    val totalSleepHours: Float,
    val quality: Float
)

data class SleepDebtAnalysis(
    val cumulativeDebtMs: Long,
    val averageDailyDebtMs: Long,
    val dailyDebts: List<Long>,
    val severity: String,
    val recoveryDaysNeeded: Int
)