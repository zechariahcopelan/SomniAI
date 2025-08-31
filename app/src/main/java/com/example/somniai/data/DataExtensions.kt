package com.example.somniai.data

import com.example.somniai.ai.RawInsight
import java.util.*
import kotlin.math.*

/**
 * Extension functions for SomniAI data model conversions and utilities
 * File: app/src/main/java/com/example/somniai/data/DataExtensions.kt
 */

// ========== CRITICAL EXTENSION: toSleepInsight() ==========

/**
 * Converts RawInsight to SleepInsight (CRITICAL - fixes compilation errors)
 */
fun RawInsight.toSleepInsight(): SleepInsight {
    return SleepInsight(
        id = this.id ?: UUID.randomUUID().toString(),
        sessionId = this.sessionId ?: 0L,
        category = when (this.category?.lowercase()) {
            "quality" -> InsightCategory.QUALITY
            "duration" -> InsightCategory.DURATION
            "environment" -> InsightCategory.ENVIRONMENT
            "movement" -> InsightCategory.MOVEMENT
            "consistency" -> InsightCategory.CONSISTENCY
            "trends" -> InsightCategory.TRENDS
            "schedule" -> InsightCategory.SCHEDULE
            "lifestyle" -> InsightCategory.LIFESTYLE
            "health" -> InsightCategory.HEALTH
            "optimization" -> InsightCategory.OPTIMIZATION
            else -> InsightCategory.QUALITY // Default fallback
        },
        title = this.title ?: "Sleep Insight",
        description = this.description ?: "",
        recommendation = this.recommendation ?: "",
        priority = this.priority ?: 2, // Default to medium priority
        timestamp = this.timestamp ?: System.currentTimeMillis(),
        isAiGenerated = this.isAiGenerated ?: true,
        isAcknowledged = false,
        confidence = this.confidence ?: 0.7f
    )
}

// ========== SLEEP SESSION EXTENSIONS ==========

/**
 * Extensions for SleepSession calculations and formatting
 */
fun SleepSession.calculateQualityScore(): Float {
    if (sleepQualityScore != null) return sleepQualityScore

    val movementPenalty = (averageMovementIntensity / 10f).coerceAtMost(3f)
    val noisePenalty = (averageNoiseLevel / 20f).coerceAtMost(2f)
    val durationBonus = if (actualSleepDuration > 6 * 3600000) 1f else 0f
    val efficiencyBonus = if (sleepEfficiency > 85f) 1f else 0f

    return (10f - movementPenalty - noisePenalty + durationBonus + efficiencyBonus)
        .coerceIn(1f, 10f)
}

fun SleepSession.getDurationFormatted(): String {
    val hours = duration / (1000 * 60 * 60)
    val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)
    return String.format("%dh %02dm", hours, minutes)
}

fun SleepSession.getSleepLatencyFormatted(): String {
    val minutes = sleepLatency / (1000 * 60)
    return "${minutes}m"
}

fun SleepSession.getMainSleepPhase(): SleepPhase {
    val phaseDurations = mapOf(
        SleepPhase.LIGHT_SLEEP to lightSleepDuration,
        SleepPhase.DEEP_SLEEP to deepSleepDuration,
        SleepPhase.REM_SLEEP to remSleepDuration
    )
    return phaseDurations.maxByOrNull { it.value }?.key ?: SleepPhase.UNKNOWN
}

fun SleepSession.getQualityGrade(): String {
    val score = sleepQualityScore ?: calculateQualityScore()
    return when {
        score >= 9f -> "A"
        score >= 8f -> "B+"
        score >= 7f -> "B"
        score >= 6f -> "C+"
        score >= 5f -> "C"
        else -> "D"
    }
}

fun SleepSession.getPhaseBreakdown(): Map<SleepPhase, Long> {
    return mapOf(
        SleepPhase.AWAKE to awakeDuration,
        SleepPhase.LIGHT_SLEEP to lightSleepDuration,
        SleepPhase.DEEP_SLEEP to deepSleepDuration,
        SleepPhase.REM_SLEEP to remSleepDuration
    )
}

fun SleepSession.isValidSession(): Boolean {
    return duration > 0 &&
            (sleepQualityScore?.let { it in 0.0f..10.0f } ?: true) &&
            sleepEfficiency in 0.0f..100.0f &&
            startTime < (endTime ?: System.currentTimeMillis())
}

// ========== ANALYTICS EXTENSIONS ==========

/**
 * Extensions for analytics calculations
 */
fun SleepSession.toAnalytics(): SleepSessionAnalytics {
    return SleepSessionAnalytics(
        sessionId = this.id,
        sessionDuration = this.sessionDuration,
        timestamp = this.startTime,
        qualityScore = this.sleepQualityScore ?: 0f,
        efficiencyScore = this.sleepEfficiency / 100f, // Convert to 0-1 scale
        movementScore = (10f - this.averageMovementIntensity).coerceIn(0f, 10f) / 10f,
        noiseScore = (10f - this.averageNoiseLevel / 10f).coerceIn(0f, 10f) / 10f,
        deepSleepPercentage = if (duration > 0) (deepSleepDuration.toFloat() / duration) * 100f else 0f,
        remSleepPercentage = if (duration > 0) (remSleepDuration.toFloat() / duration) * 100f else 0f,
        lightSleepPercentage = if (duration > 0) (lightSleepDuration.toFloat() / duration) * 100f else 0f,
        awakePercentage = if (duration > 0) (awakeDuration.toFloat() / duration) * 100f else 0f,
        sleepLatency = this.sleepLatency,
        awakeDuration = this.awakeDuration,
        movementCount = this.movementEvents.size,
        avgNoiseLevel = this.averageNoiseLevel,
        maxNoiseLevel = this.noiseEvents.maxOfOrNull { it.decibelLevel } ?: 0f,
        confidence = this.confidence
    )
}

fun SleepSession.toSummaryDTO(): SessionSummaryDTO {
    return SessionSummaryDTO(
        sessionId = this.id.toString(),
        date = Date(this.startTime),
        quality = this.sleepQualityScore ?: 0f,
        duration = this.duration,
        efficiency = this.sleepEfficiency,
        startTime = Date(this.startTime).toString(),
        endTime = this.endTime?.let { Date(it).toString() } ?: "",
        movementScore = (10f - this.averageMovementIntensity).coerceIn(0f, 10f),
        noiseScore = (10f - this.averageNoiseLevel / 10f).coerceIn(0f, 10f)
    )
}

fun SleepSession.toDailyTrendData(): DailyTrendData {
    return DailyTrendData(
        date = Date(this.startTime),
        quality = this.sleepQualityScore ?: 0f,
        duration = this.duration.toFloat(),
        efficiency = this.sleepEfficiency,
        movementCount = this.movementEvents.size,
        noiseLevel = this.averageNoiseLevel,
        sleepOnsetTime = (this.sleepLatency / (1000 * 60)).toInt(), // Convert to minutes
        wakeUpCount = this.phaseTransitions.count { it.toPhase == SleepPhase.AWAKE },
        deepSleepPercentage = if (duration > 0) (deepSleepDuration.toFloat() / duration) * 100f else 0f,
        lightSleepPercentage = if (duration > 0) (lightSleepDuration.toFloat() / duration) * 100f else 0f,
        remSleepPercentage = if (duration > 0) (remSleepDuration.toFloat() / duration) * 100f else 0f
    )
}

// ========== COLLECTION EXTENSIONS ==========

/**
 * Extensions for collections of sleep sessions
 */
fun List<SleepSession>.calculateAverageQuality(): Float {
    return if (isEmpty()) 0f else mapNotNull { it.sleepQualityScore }.average().toFloat()
}

fun List<SleepSession>.calculateAverageEfficiency(): Float {
    return if (isEmpty()) 0f else map { it.sleepEfficiency }.average().toFloat()
}

fun List<SleepSession>.calculateAverageDuration(): Float {
    return if (isEmpty()) 0f else map { it.duration.toFloat() }.average().toFloat()
}

fun List<SleepSession>.filterValidSessions(): List<SleepSession> {
    return filter { it.isValidSession() }
}

fun List<SleepSession>.toDailyTrendData(): List<DailyTrendData> {
    return groupBy {
        Calendar.getInstance().apply { timeInMillis = it.startTime }.get(Calendar.DAY_OF_YEAR)
    }.map { (_, sessions) ->
        val daySession = sessions.first()
        DailyTrendData(
            date = Date(daySession.startTime),
            quality = sessions.mapNotNull { it.sleepQualityScore }.average().toFloat(),
            duration = sessions.map { it.duration.toFloat() }.average(),
            efficiency = sessions.map { it.sleepEfficiency }.average().toFloat(),
            movementCount = sessions.sumOf { it.movementEvents.size },
            noiseLevel = sessions.map { it.averageNoiseLevel }.average().toFloat()
        )
    }
}

fun List<SleepSession>.analyzeHabits(): HabitAnalysis {
    if (isEmpty()) {
        return HabitAnalysis(
            recognizedHabits = emptyList(),
            bedtimeConsistency = 0f,
            durationConsistency = 0f,
            overallConsistency = 0f,
            overallProgress = 0f
        )
    }

    val recognizedHabits = mutableListOf<SleepHabit>()

    // Analyze bedtime consistency
    val bedtimes = map {
        Calendar.getInstance().apply { timeInMillis = it.startTime }.get(Calendar.HOUR_OF_DAY)
    }
    val bedtimeVariance = bedtimes.map { (it - bedtimes.average()).pow(2) }.average()
    val bedtimeConsistency = (10f - bedtimeVariance.toFloat()).coerceIn(0f, 10f) / 10f

    if (bedtimeConsistency > 0.8f) {
        recognizedHabits.add(SleepHabit.CONSISTENT_BEDTIME)
    } else if (bedtimes.average() > 23) {
        recognizedHabits.add(SleepHabit.LATE_BEDTIME)
    } else {
        recognizedHabits.add(SleepHabit.IRREGULAR_SLEEP)
    }

    // Analyze duration consistency
    val durations = map { it.duration.toFloat() }
    val durationMean = durations.average()
    val durationVariance = durations.map { (it - durationMean).pow(2) }.average()
    val durationConsistency = (1f / (1f + sqrt(durationVariance).toFloat() / durationMean.toFloat())).coerceIn(0f, 1f)

    if (durationConsistency > 0.8f && durationMean > 6 * 3600000) {
        recognizedHabits.add(SleepHabit.OPTIMAL_DURATION)
    }

    val overallConsistency = (bedtimeConsistency + durationConsistency) / 2f
    val overallProgress = calculateAverageQuality() / 10f // Convert to 0-1 scale

    return HabitAnalysis(
        recognizedHabits = recognizedHabits,
        bedtimeConsistency = bedtimeConsistency,
        durationConsistency = durationConsistency,
        overallConsistency = overallConsistency,
        overallProgress = overallProgress
    )
}

// ========== TREND ANALYSIS EXTENSIONS ==========

/**
 * Extensions for trend analysis
 */
fun List<Float>.calculateTrend(): TrendDirection {
    if (size < 2) return TrendDirection.STABLE

    val firstHalf = take(size / 2).average()
    val secondHalf = drop(size / 2).average()

    val difference = secondHalf - firstHalf
    val percentChange = if (firstHalf != 0.0) abs(difference / firstHalf) * 100 else 0.0

    return when {
        percentChange < 5 -> TrendDirection.STABLE
        difference > 0 && percentChange > 15 -> TrendDirection.IMPROVING
        difference < 0 && percentChange > 15 -> TrendDirection.DECLINING
        percentChange > 25 -> TrendDirection.VOLATILE
        else -> TrendDirection.STABLE
    }
}

fun List<SleepSession>.calculateTrendDirection(): TrendDirection {
    if (size < 2) return TrendDirection.INSUFFICIENT_DATA

    val qualityScores = mapNotNull { it.sleepQualityScore }
    if (qualityScores.size < 2) return TrendDirection.INSUFFICIENT_DATA

    return qualityScores.calculateTrend()
}

// ========== TIME AND FORMATTING EXTENSIONS ==========

/**
 * Extensions for time calculations and formatting
 */
fun Long.toDurationString(): String {
    val hours = this / (1000 * 60 * 60)
    val minutes = (this % (1000 * 60 * 60)) / (1000 * 60)
    return String.format("%dh %02dm", hours, minutes)
}

fun Long.toTimeString(): String {
    val date = Date(this)
    return String.format("%tH:%tM", date, date)
}

fun Date.toTimeRange(durationMs: Long): TimeRange {
    return TimeRange(
        startDate = this.time,
        endDate = this.time + durationMs,
        description = "Time range from ${this.toTimeString()}",
        timezone = TimeZone.getDefault().id
    )
}

fun Long.toDuration(): Duration {
    val hours = (this / (1000 * 60 * 60)).toInt()
    val minutes = ((this % (1000 * 60 * 60)) / (1000 * 60)).toInt()
    return Duration(hours, minutes)
}

fun Date.toTimeString(): String {
    return String.format("%tH:%tM", this, this)
}

// ========== QUALITY ANALYSIS EXTENSIONS ==========

/**
 * Extensions for quality factor analysis
 */
fun SleepSession.calculateQualityFactors(): QualityFactorAnalysis {
    val movementFactor = (10f - averageMovementIntensity).coerceIn(0f, 10f)
    val noiseFactor = (10f - averageNoiseLevel / 10f).coerceIn(0f, 10f)
    val durationFactor = when {
        duration < 4 * 3600000 -> 3f // Less than 4 hours
        duration < 6 * 3600000 -> 6f // 4-6 hours
        duration in 6 * 3600000..9 * 3600000 -> 10f // 6-9 hours (optimal)
        duration < 12 * 3600000 -> 8f // 9-12 hours
        else -> 5f // Over 12 hours
    }
    val consistencyFactor = if (phaseTransitions.size < 10) 8f else (10f - phaseTransitions.size / 10f).coerceIn(0f, 10f)
    val efficiencyFactor = sleepEfficiency / 10f // Convert percentage to 0-10 scale
    val overallScore = (movementFactor + noiseFactor + durationFactor + consistencyFactor + efficiencyFactor) / 5f

    return QualityFactorAnalysis(
        movementFactor = movementFactor,
        noiseFactor = noiseFactor,
        durationFactor = durationFactor,
        consistencyFactor = consistencyFactor,
        overallScore = overallScore
    )
}

// ========== VALIDATION EXTENSIONS ==========

/**
 * Extensions for data validation
 */
fun SleepSessionAnalytics.isHighQuality(): Boolean {
    return qualityScore >= 8.0f && confidence >= 0.7f
}

fun SleepSessionAnalytics.hasAnomalies(): Boolean {
    return anomaliesDetected.isNotEmpty()
}

fun SleepSessionAnalytics.getDataReliabilityScore(): Float {
    return (dataQuality.qualityScore + sensorReliability + dataCompleteness) / 3f
}

// ========== INSIGHT EXTENSIONS ==========

/**
 * Extensions for insights and recommendations
 */
fun List<ProcessedInsight>.getTopInsights(count: Int = 3): List<ProcessedInsight> {
    return sortedWith(
        compareByDescending<ProcessedInsight> { it.priority }
            .thenByDescending { it.qualityScore }
    ).take(count)
}

fun ProcessedInsight.isHighQuality(): Boolean {
    return qualityScore >= 0.8f && confidence >= 0.7f && relevanceScore >= 0.7f
}

fun ProcessedInsight.isActionable(): Boolean {
    return actionabilityScore >= 0.7f && recommendation.isNotBlank()
}

// ========== UTILITY EXTENSIONS ==========

/**
 * Utility extensions for common calculations
 */
fun Float.toPercentage(): String = "${(this * 100).toInt()}%"

fun Float.toGrade(): String {
    return when {
        this >= 9f -> "A"
        this >= 8f -> "B+"
        this >= 7f -> "B"
        this >= 6f -> "C+"
        this >= 5f -> "C"
        else -> "D"
    }
}

fun Int.toOrdinal(): String {
    return when {
        this % 100 in 11..13 -> "${this}th"
        this % 10 == 1 -> "${this}st"
        this % 10 == 2 -> "${this}nd"
        this % 10 == 3 -> "${this}rd"
        else -> "${this}th"
    }
}

// ========== MISSING DATA CLASSES REFERENCED IN EXTENSIONS ==========

/**
 * Simple Duration class if not using java.time.Duration
 */
data class Duration(
    val hours: Int,
    val minutes: Int
) {
    fun toMinutes(): Long = hours * 60L + minutes
    fun toMillis(): Long = toMinutes() * 60 * 1000
    fun toHours(): Float = hours + minutes / 60f

    override fun toString(): String = "${hours}h ${minutes}m"

    companion object {
        fun fromMillis(millis: Long): Duration {
            val hours = (millis / (1000 * 60 * 60)).toInt()
            val minutes = ((millis % (1000 * 60 * 60)) / (1000 * 60)).toInt()
            return Duration(hours, minutes)
        }
    }
}

/**
 * Simple TimeRange class for time period calculations
 */
data class TimeRange(
    val startDate: Long,
    val endDate: Long,
    val description: String,
    val timezone: String = TimeZone.getDefault().id
) {
    val durationMs: Long get() = endDate - startDate
    val durationDays: Int get() = (durationMs / (24 * 60 * 60 * 1000L)).toInt()

    fun contains(timestamp: Long): Boolean = timestamp in startDate..endDate

    companion object {
        fun lastDays(days: Int): TimeRange {
            val end = System.currentTimeMillis()
            val start = end - (days * 24 * 60 * 60 * 1000L)
            return TimeRange(start, end, "Last $days days")
        }
    }
}