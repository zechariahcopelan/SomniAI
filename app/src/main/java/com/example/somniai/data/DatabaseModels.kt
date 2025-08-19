package com.example.somniai.data

import androidx.room.ColumnInfo
import androidx.room.Embedded
import com.example.somniai.database.*
import java.util.Date
import com.example.somniai.ui.theme.SleepPhase
import com.example.somniai.data.InsightCategory

/**
 * Enhanced database models for data transfer, aggregation, and mapping
 *
 * Provides specialized data classes for:
 * - Complex database query results
 * - Data transfer between layers
 * - Aggregated statistics and analytics
 * - Database result mapping utilities
 * - Optimized data structures for UI consumption
 */

// ========== DATA TRANSFER OBJECTS (DTOs) ==========

/**
 * Lightweight session data for lists and overviews
 */

// ========== MISSING ENUM FOR INSIGHT DATA ==========


data class SessionSummaryDTO(
    @ColumnInfo(name = "id")
    val id: Long,

    @ColumnInfo(name = "start_time")
    val startTime: Long,

    @ColumnInfo(name = "end_time")
    val endTime: Long?,

    @ColumnInfo(name = "total_duration")
    val totalDuration: Long,

    @ColumnInfo(name = "quality_score")
    val qualityScore: Float?,

    @ColumnInfo(name = "sleep_efficiency")
    val sleepEfficiency: Float,

    @ColumnInfo(name = "total_movement_events")
    val totalMovementEvents: Int,

    @ColumnInfo(name = "total_noise_events")
    val totalNoiseEvents: Int,

    @ColumnInfo(name = "average_movement_intensity")
    val averageMovementIntensity: Float,

    @ColumnInfo(name = "average_noise_level")
    val averageNoiseLevel: Float
) {
    fun toDomainModel(): SleepSession {
        return SleepSession(
            id = id,
            startTime = startTime,
            endTime = endTime,
            totalDuration = totalDuration,
            sleepEfficiency = sleepEfficiency,
            sleepQualityScore = qualityScore,
            averageMovementIntensity = averageMovementIntensity,
            averageNoiseLevel = averageNoiseLevel
        )
    }

    val isCompleted: Boolean
        get() = endTime != null

    val durationHours: Float
        get() = totalDuration / (1000f * 60f * 60f)

    val formattedDuration: String
        get() {
            val hours = (totalDuration / (1000 * 60 * 60)).toInt()
            val minutes = ((totalDuration % (1000 * 60 * 60)) / (1000 * 60)).toInt()
            return "${hours}h ${minutes}m"
        }

    val qualityGrade: String
        get() = when (qualityScore) {
            null -> "N/A"
            in 9f..10f -> "A+"
            in 8f..9f -> "A"
            in 7f..8f -> "B+"
            in 6f..7f -> "B"
            in 5f..6f -> "C+"
            in 4f..5f -> "C"
            in 3f..4f -> "D+"
            in 2f..3f -> "D"
            else -> "F"
        }

    val efficiencyGrade: String
        get() = when {
            sleepEfficiency >= 90f -> "Excellent"
            sleepEfficiency >= 80f -> "Very Good"
            sleepEfficiency >= 70f -> "Good"
            sleepEfficiency >= 60f -> "Fair"
            sleepEfficiency >= 50f -> "Poor"
            else -> "Very Poor"
        }
}

/**
 * Movement event data optimized for chart visualization
 */
data class MovementEventDTO(
    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "intensity")
    val intensity: Float,

    @ColumnInfo(name = "is_significant")
    val isSignificant: Boolean,

    @ColumnInfo(name = "session_id")
    val sessionId: Long
) {
    val timeFromSessionStart: Long
        get() = timestamp // Will be calculated relative to session start in queries

    val intensityLevel: String
        get() = when {
            intensity < 1.5f -> "Very Low"
            intensity < 2.5f -> "Low"
            intensity < 4f -> "Medium"
            intensity < 6f -> "High"
            else -> "Very High"
        }
}

/**
 * Noise event data optimized for analysis
 */
data class NoiseEventDTO(
    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "decibel_level")
    val decibelLevel: Float,

    @ColumnInfo(name = "is_disruptive")
    val isDisruptive: Boolean,

    @ColumnInfo(name = "session_id")
    val sessionId: Long
) {
    val timeFromSessionStart: Long
        get() = timestamp // Will be calculated relative to session start in queries

    val noiseLevel: String
        get() = when {
            decibelLevel < 30f -> "Very Quiet"
            decibelLevel < 40f -> "Quiet"
            decibelLevel < 50f -> "Moderate"
            decibelLevel < 60f -> "Loud"
            else -> "Very Loud"
        }
}

/**
 * Phase transition data for timeline visualization
 */
data class PhaseTransitionDTO(
    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "from_phase")
    val fromPhase: SleepPhase,

    @ColumnInfo(name = "to_phase")
    val toPhase: SleepPhase,

    @ColumnInfo(name = "confidence")
    val confidence: Float,

    @ColumnInfo(name = "duration_in_phase")
    val durationInPhase: Long,

    @ColumnInfo(name = "session_id")
    val sessionId: Long
) {
    val timeFromSessionStart: Long
        get() = timestamp // Will be calculated relative to session start in queries

    val phaseDurationMinutes: Int
        get() = (durationInPhase / (1000 * 60)).toInt()

    val confidenceLevel: String
        get() = when {
            confidence >= 0.8f -> "High"
            confidence >= 0.6f -> "Medium"
            confidence >= 0.4f -> "Low"
            else -> "Very Low"
        }
}

// ========== AGGREGATION RESULT MODELS ==========

/**
 * Daily sleep trends for chart visualization
 */
data class DailyTrendData(
    @ColumnInfo(name = "date")
    val date: Long, // Start of day timestamp

    @ColumnInfo(name = "session_count")
    val sessionCount: Int,

    @ColumnInfo(name = "total_duration")
    val totalDuration: Long,

    @ColumnInfo(name = "average_duration")
    val averageDuration: Long,

    @ColumnInfo(name = "average_quality")
    val averageQuality: Float?,

    @ColumnInfo(name = "average_efficiency")
    val averageEfficiency: Float,

    @ColumnInfo(name = "total_movements")
    val totalMovements: Int,

    @ColumnInfo(name = "total_noise_events")
    val totalNoiseEvents: Int,

    @ColumnInfo(name = "average_bedtime")
    val averageBedtime: Long?,

    @ColumnInfo(name = "sleep_debt")
    val sleepDebt: Long = 0L // Accumulated sleep debt
) {
    val durationHours: Float
        get() = averageDuration / (1000f * 60f * 60f)

    val formattedDate: String
        get() {
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = date
            return "${calendar.get(java.util.Calendar.MONTH) + 1}/${calendar.get(java.util.Calendar.DAY_OF_MONTH)}"
        }

    val weekDay: String
        get() {
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = date
            val dayNames = arrayOf("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
            return dayNames[calendar.get(java.util.Calendar.DAY_OF_WEEK) - 1]
        }

    val isWeekend: Boolean
        get() {
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = date
            val dayOfWeek = calendar.get(java.util.Calendar.DAY_OF_WEEK)
            return dayOfWeek == java.util.Calendar.SATURDAY || dayOfWeek == java.util.Calendar.SUNDAY
        }

    val hasData: Boolean
        get() = sessionCount > 0

    val sleepDebtHours: Float
        get() = sleepDebt / (1000f * 60f * 60f)
}

/**
 * Weekly aggregated statistics
 */
data class WeeklyStatsData(
    @ColumnInfo(name = "week_start")
    val weekStart: Long,

    @ColumnInfo(name = "session_count")
    val sessionCount: Int,

    @ColumnInfo(name = "average_duration")
    val averageDuration: Long,

    @ColumnInfo(name = "average_quality")
    val averageQuality: Float,

    @ColumnInfo(name = "average_efficiency")
    val averageEfficiency: Float,

    @ColumnInfo(name = "consistency_score")
    val consistencyScore: Float,

    @ColumnInfo(name = "weekday_avg_duration")
    val weekdayAverageDuration: Long,

    @ColumnInfo(name = "weekend_avg_duration")
    val weekendAverageDuration: Long,

    @ColumnInfo(name = "best_quality_date")
    val bestQualityDate: Long?,

    @ColumnInfo(name = "worst_quality_date")
    val worstQualityDate: Long?
) {
    val weekNumber: Int
        get() {
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = weekStart
            return calendar.get(java.util.Calendar.WEEK_OF_YEAR)
        }

    val weekEndDate: Long
        get() = weekStart + (6 * 24 * 60 * 60 * 1000L)

    val durationHours: Float
        get() = averageDuration / (1000f * 60f * 60f)

    val weekdayDurationHours: Float
        get() = weekdayAverageDuration / (1000f * 60f * 60f)

    val weekendDurationHours: Float
        get() = weekendAverageDuration / (1000f * 60f * 60f)

    val weekendEffect: Float
        get() = weekendDurationHours - weekdayDurationHours // Social jet lag

    val hasWeekendEffect: Boolean
        get() = kotlin.math.abs(weekendEffect) > 0.5f // More than 30 minutes difference
}

/**
 * Monthly aggregated statistics
 */
data class MonthlyStatsData(
    @ColumnInfo(name = "month")
    val month: Int,

    @ColumnInfo(name = "year")
    val year: Int,

    @ColumnInfo(name = "session_count")
    val sessionCount: Int,

    @ColumnInfo(name = "average_duration")
    val averageDuration: Long,

    @ColumnInfo(name = "average_quality")
    val averageQuality: Float,

    @ColumnInfo(name = "average_efficiency")
    val averageEfficiency: Float,

    @ColumnInfo(name = "improvement_trend")
    val improvementTrend: Float, // Positive = improving, negative = declining

    @ColumnInfo(name = "total_sleep_time")
    val totalSleepTime: Long,

    @ColumnInfo(name = "nights_tracked")
    val nightsTracked: Int
) {
    val monthName: String
        get() {
            val months = arrayOf(
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            )
            return months[month - 1]
        }

    val formattedMonth: String
        get() = "$monthName $year"

    val durationHours: Float
        get() = averageDuration / (1000f * 60f * 60f)

    val totalSleepHours: Float
        get() = totalSleepTime / (1000f * 60f * 60f)

    val trackingConsistency: Float
        get() {
            val daysInMonth = when (month) {
                2 -> if (isLeapYear(year)) 29 else 28
                4, 6, 9, 11 -> 30
                else -> 31
            }
            return (nightsTracked.toFloat() / daysInMonth) * 100f
        }

    val trendDirection: String
        get() = when {
            improvementTrend > 0.2f -> "Improving"
            improvementTrend < -0.2f -> "Declining"
            else -> "Stable"
        }

    private fun isLeapYear(year: Int): Boolean {
        return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
    }
}

// ========== QUALITY FACTOR ANALYSIS MODELS ==========

/**
 * Detailed quality factor breakdown for charts
 */
data class QualityFactorBreakdown(
    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "session_date")
    val sessionDate: Long,

    @ColumnInfo(name = "movement_score")
    val movementScore: Float,

    @ColumnInfo(name = "noise_score")
    val noiseScore: Float,

    @ColumnInfo(name = "duration_score")
    val durationScore: Float,

    @ColumnInfo(name = "consistency_score")
    val consistencyScore: Float,

    @ColumnInfo(name = "efficiency_score")
    val efficiencyScore: Float,

    @ColumnInfo(name = "phase_balance_score")
    val phaseBalanceScore: Float,

    @ColumnInfo(name = "overall_score")
    val overallScore: Float
) {
    val factors: Map<String, Float>
        get() = mapOf(
            "Movement" to movementScore,
            "Noise" to noiseScore,
            "Duration" to durationScore,
            "Consistency" to consistencyScore,
            "Efficiency" to efficiencyScore,
            "Phase Balance" to phaseBalanceScore
        )

    val strongestFactor: Pair<String, Float>
        get() = factors.maxByOrNull { it.value } ?: ("Unknown" to 0f)

    val weakestFactor: Pair<String, Float>
        get() = factors.minByOrNull { it.value } ?: ("Unknown" to 0f)

    val averageFactorScore: Float
        get() = factors.values.average().toFloat()

    val scoreVariance: Float
        get() {
            val avg = averageFactorScore
            return factors.values.map { (it - avg) * (it - avg) }.average().toFloat()
        }

    val isBalanced: Boolean
        get() = scoreVariance < 1f // Low variance indicates balanced scores
}

/**
 * Movement pattern analysis for visualization
 */
data class MovementPatternData(
    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "hour_of_session")
    val hourOfSession: Int,

    @ColumnInfo(name = "average_intensity")
    val averageIntensity: Float,

    @ColumnInfo(name = "movement_count")
    val movementCount: Int,

    @ColumnInfo(name = "significant_movements")
    val significantMovements: Int,

    @ColumnInfo(name = "max_intensity")
    val maxIntensity: Float,

    @ColumnInfo(name = "restlessness_score")
    val restlessnessScore: Float
) {
    val movementRate: Float
        get() = movementCount.toFloat() / 60f // Movements per minute

    val significantRate: Float
        get() = if (movementCount > 0) significantMovements.toFloat() / movementCount else 0f

    val intensityLevel: String
        get() = when {
            averageIntensity < 1.5f -> "Very Low"
            averageIntensity < 2.5f -> "Low"
            averageIntensity < 4f -> "Medium"
            averageIntensity < 6f -> "High"
            else -> "Very High"
        }
}

/**
 * Sleep phase distribution for pie charts
 */
data class PhaseDistributionData(
    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "awake_duration")
    val awakeDuration: Long,

    @ColumnInfo(name = "light_sleep_duration")
    val lightSleepDuration: Long,

    @ColumnInfo(name = "deep_sleep_duration")
    val deepSleepDuration: Long,

    @ColumnInfo(name = "rem_sleep_duration")
    val remSleepDuration: Long,

    @ColumnInfo(name = "total_duration")
    val totalDuration: Long
) {
    val awakePercentage: Float
        get() = if (totalDuration > 0) (awakeDuration.toFloat() / totalDuration) * 100f else 0f

    val lightSleepPercentage: Float
        get() = if (totalDuration > 0) (lightSleepDuration.toFloat() / totalDuration) * 100f else 0f

    val deepSleepPercentage: Float
        get() = if (totalDuration > 0) (deepSleepDuration.toFloat() / totalDuration) * 100f else 0f

    val remSleepPercentage: Float
        get() = if (totalDuration > 0) (remSleepDuration.toFloat() / totalDuration) * 100f else 0f

    val actualSleepPercentage: Float
        get() = 100f - awakePercentage

    val phaseDistribution: Map<SleepPhase, Float>
        get() = mapOf(
            SleepPhase.AWAKE to awakePercentage,
            SleepPhase.LIGHT to lightSleepPercentage,
            SleepPhase.DEEP to deepSleepPercentage,
            SleepPhase.REM to remSleepPercentage
        )

    val isHealthyDistribution: Boolean
        get() = deepSleepPercentage >= 15f && remSleepPercentage >= 20f && awakePercentage <= 10f

    val dominantPhase: SleepPhase
        get() = phaseDistribution.maxByOrNull { it.value }?.key ?: SleepPhase.LIGHT
}

// ========== COMPARATIVE ANALYSIS MODELS ==========

/**
 * Session comparison data for benchmarking
 */
data class SessionComparisonData(
    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "quality_score")
    val qualityScore: Float,

    @ColumnInfo(name = "personal_average")
    val personalAverage: Float,

    @ColumnInfo(name = "personal_best")
    val personalBest: Float,

    @ColumnInfo(name = "percentile_rank")
    val percentileRank: Float, // Rank among user's sessions

    @ColumnInfo(name = "improvement_from_baseline")
    val improvementFromBaseline: Float,

    @ColumnInfo(name = "days_since_best")
    val daysSinceBest: Int
) {
    val qualityGrade: String
        get() = when {
            qualityScore >= 9f -> "A+"
            qualityScore >= 8f -> "A"
            qualityScore >= 7f -> "B+"
            qualityScore >= 6f -> "B"
            qualityScore >= 5f -> "C+"
            qualityScore >= 4f -> "C"
            qualityScore >= 3f -> "D+"
            qualityScore >= 2f -> "D"
            else -> "F"
        }

    val performanceLevel: String
        get() = when {
            percentileRank >= 90f -> "Excellent"
            percentileRank >= 75f -> "Above Average"
            percentileRank >= 50f -> "Average"
            percentileRank >= 25f -> "Below Average"
            else -> "Poor"
        }

    val isBetterThanAverage: Boolean
        get() = qualityScore > personalAverage

    val isNewPersonalBest: Boolean
        get() = qualityScore >= personalBest

    val progressMessage: String
        get() = when {
            isNewPersonalBest -> "New personal best!"
            isBetterThanAverage -> "Above your average"
            improvementFromBaseline > 0 -> "Improving"
            else -> "Room for improvement"
        }
}

/**
 * Sleep efficiency trends for line charts
 */
data class EfficiencyTrendData(
    @ColumnInfo(name = "session_date")
    val sessionDate: Long,

    @ColumnInfo(name = "basic_efficiency")
    val basicEfficiency: Float,

    @ColumnInfo(name = "adjusted_efficiency")
    val adjustedEfficiency: Float,

    @ColumnInfo(name = "quality_weighted_efficiency")
    val qualityWeightedEfficiency: Float,

    @ColumnInfo(name = "sleep_latency")
    val sleepLatency: Long,

    @ColumnInfo(name = "wake_count")
    val wakeCount: Int,

    @ColumnInfo(name = "movement_disruptions")
    val movementDisruptions: Int
) {
    val formattedDate: String
        get() {
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = sessionDate
            return "${calendar.get(java.util.Calendar.MONTH) + 1}/${calendar.get(java.util.Calendar.DAY_OF_MONTH)}"
        }

    val sleepLatencyMinutes: Int
        get() = (sleepLatency / (1000 * 60)).toInt()

    val efficiencyGrade: String
        get() = when {
            adjustedEfficiency >= 90f -> "Excellent"
            adjustedEfficiency >= 80f -> "Very Good"
            adjustedEfficiency >= 70f -> "Good"
            adjustedEfficiency >= 60f -> "Fair"
            adjustedEfficiency >= 50f -> "Poor"
            else -> "Very Poor"
        }

    val disruptionLevel: String
        get() = when {
            movementDisruptions <= 3 -> "Minimal"
            movementDisruptions <= 6 -> "Low"
            movementDisruptions <= 10 -> "Moderate"
            movementDisruptions <= 15 -> "High"
            else -> "Very High"
        }
}

// ========== INSIGHT AND RECOMMENDATION MODELS ==========

/**
 * AI-generated insights with categorization
 */
data class InsightData(
    @ColumnInfo(name = "id")
    val id: Long,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "category")
    val category: InsightCategory,

    @ColumnInfo(name = "title")
    val title: String,

    @ColumnInfo(name = "description")
    val description: String,

    @ColumnInfo(name = "recommendation")
    val recommendation: String,

    @ColumnInfo(name = "priority")
    val priority: Int,

    @ColumnInfo(name = "confidence_score")
    val confidenceScore: Float,

    @ColumnInfo(name = "is_acknowledged")
    val isAcknowledged: Boolean,

    @ColumnInfo(name = "timestamp")
    val timestamp: Long
) {
    val priorityLevel: String
        get() = when (priority) {
            1 -> "High"
            2 -> "Medium"
            3 -> "Low"
            else -> "Unknown"
        }

    val confidenceLevel: String
        get() = when {
            confidenceScore >= 0.8f -> "High"
            confidenceScore >= 0.6f -> "Medium"
            confidenceScore >= 0.4f -> "Low"
            else -> "Very Low"
        }

    val categoryIcon: String
        get() = when (category) {
            InsightCategory.DURATION -> "⏰"
            InsightCategory.QUALITY -> "⭐"
            InsightCategory.MOVEMENT -> "🏃"
            InsightCategory.NOISE -> "🔊"
            InsightCategory.TIMING -> "⏱️"
            InsightCategory.PATTERN -> "📊"
            InsightCategory.HEALTH -> "🏥"
            InsightCategory.ENVIRONMENT -> "🌡️"
        }

    val isRecent: Boolean
        get() = System.currentTimeMillis() - timestamp < (24 * 60 * 60 * 1000L) // Within 24 hours

    val isHighPriority: Boolean
        get() = priority == 1 && confidenceScore >= 0.7f
}

// ========== DATABASE RESULT MAPPERS ==========

/**
 * Utility class for mapping between database entities and domain models
 */
object DatabaseMappers {

    /**
     * Convert SleepSessionEntity to SessionSummaryDTO
     */
    fun sessionEntityToSummaryDTO(entity: SleepSessionEntity): SessionSummaryDTO {
        return SessionSummaryDTO(
            id = entity.id,
            startTime = entity.startTime,
            endTime = entity.endTime,
            totalDuration = entity.totalDuration,
            qualityScore = entity.qualityScore,
            sleepEfficiency = entity.sleepEfficiency,
            totalMovementEvents = entity.totalMovementEvents,
            totalNoiseEvents = entity.totalNoiseEvents,
            averageMovementIntensity = entity.averageMovementIntensity,
            averageNoiseLevel = entity.averageNoiseLevel
        )
    }

    /**
     * Convert MovementEventEntity to MovementEventDTO
     */
    fun movementEntityToDTO(entity: MovementEventEntity, sessionStartTime: Long): MovementEventDTO {
        return MovementEventDTO(
            timestamp = entity.timestamp - sessionStartTime, // Relative to session start
            intensity = entity.intensity,
            isSignificant = entity.isSignificant,
            sessionId = entity.sessionId
        )
    }

    /**
     * Convert NoiseEventEntity to NoiseEventDTO
     */
    fun noiseEntityToDTO(entity: NoiseEventEntity, sessionStartTime: Long): NoiseEventDTO {
        return NoiseEventDTO(
            timestamp = entity.timestamp - sessionStartTime, // Relative to session start
            decibelLevel = entity.decibelLevel,
            isDisruptive = entity.isDisruptive,
            sessionId = entity.sessionId
        )
    }

    /**
     * Convert SleepPhaseEntity to PhaseTransitionDTO
     */
    fun phaseEntityToDTO(entity: SleepPhaseEntity, sessionStartTime: Long): PhaseTransitionDTO {
        return PhaseTransitionDTO(
            timestamp = entity.timestamp - sessionStartTime, // Relative to session start
            fromPhase = entity.fromPhase,
            toPhase = entity.toPhase,
            confidence = entity.confidence,
            durationInPhase = entity.durationInPhase,
            sessionId = entity.sessionId
        )
    }

    /**
     * Convert QualityFactorsEntity to QualityFactorBreakdown
     */
    fun qualityEntityToBreakdown(entity: QualityFactorsEntity, sessionDate: Long): QualityFactorBreakdown {
        return QualityFactorBreakdown(
            sessionId = entity.sessionId,
            sessionDate = sessionDate,
            movementScore = entity.movementScore,
            noiseScore = entity.noiseScore,
            durationScore = entity.durationScore,
            consistencyScore = entity.consistencyScore,
            efficiencyScore = entity.efficiencyScore,
            phaseBalanceScore = entity.phaseBalanceScore,
            overallScore = entity.overallScore
        )
    }

    /**
     * Convert SleepInsightEntity to InsightData
     */
    fun insightEntityToData(entity: SleepInsightEntity): InsightData {
        return InsightData(
            id = entity.id,
            sessionId = entity.sessionId,
            category = entity.category,
            title = entity.title,
            description = entity.description,
            recommendation = entity.recommendation,
            priority = entity.priority,
            confidenceScore = entity.confidenceScore,
            isAcknowledged = entity.isAcknowledged,
            timestamp = entity.timestamp
        )
    }

    /**
     * Create SessionComparisonData from session and statistics
     */
    fun createSessionComparison(
        session: SleepSessionEntity,
        personalStats: SleepStatistics,
        sessionRank: Float,
        baseline: Float,
        daysSinceBest: Int
    ): SessionComparisonData {
        return SessionComparisonData(
            sessionId = session.id,
            qualityScore = session.qualityScore ?: 0f,
            personalAverage = personalStats.averageQuality,
            personalBest = personalStats.averageQuality * 1.2f, // Simplified calculation
            percentileRank = sessionRank,
            improvementFromBaseline = (session.qualityScore ?: 0f) - baseline,
            daysSinceBest = daysSinceBest
        )
    }

    /**
     * Create PhaseDistributionData from session entity
     */
    fun createPhaseDistribution(session: SleepSessionEntity): PhaseDistributionData {
        return PhaseDistributionData(
            sessionId = session.id,
            awakeDuration = session.awakeDuration,
            lightSleepDuration = session.lightSleepDuration,
            deepSleepDuration = session.deepSleepDuration,
            remSleepDuration = session.remSleepDuration,
            totalDuration = session.totalDuration
        )
    }

    /**
     * Create EfficiencyTrendData from session and movement data
     */
    fun createEfficiencyTrend(
        session: SleepSessionEntity,
        wakeCount: Int,
        movementDisruptions: Int
    ): EfficiencyTrendData {
        return EfficiencyTrendData(
            sessionDate = session.startTime,
            basicEfficiency = session.sleepEfficiency,
            adjustedEfficiency = session.sleepEfficiency * 0.95f, // Simplified calculation
            qualityWeightedEfficiency = session.sleepEfficiency * 0.9f, // Simplified calculation
            sleepLatency = session.sleepLatency,
            wakeCount = wakeCount,
            movementDisruptions = movementDisruptions
        )
    }
}

// ========== CHART DATA MODELS ==========

/**
 * Generic chart data point for line charts
 */
data class ChartDataPoint(
    val x: Float,
    val y: Float,
    val label: String = "",
    val timestamp: Long = 0L,
    val metadata: Map<String, Any> = emptyMap()
) {
    val formattedValue: String
        get() = String.format("%.1f", y)

    val formattedTimestamp: String
        get() {
            if (timestamp == 0L) return label
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = timestamp
            return "${calendar.get(java.util.Calendar.MONTH) + 1}/${calendar.get(java.util.Calendar.DAY_OF_MONTH)}"
        }
}

/**
 * Bar chart data for quality factor breakdown
 */
data class BarChartData(
    val label: String,
    val value: Float,
    val color: Int = 0,
    val description: String = ""
) {
    val formattedValue: String
        get() = String.format("%.1f", value)

    val percentage: String
        get() = "${(value * 10).toInt()}%"
}

/**
 * Pie chart data for phase distribution
 */
data class PieChartData(
    val label: String,
    val value: Float,
    val color: Int = 0,
    val phase: SleepPhase? = null
) {
    val percentage: Float
        get() = value

    val formattedPercentage: String
        get() = "${percentage.toInt()}%"

    val duration: String
        get() = if (phase != null) {
            val hours = (value * 8 / 100).toInt() // Assuming 8-hour sleep
            val minutes = ((value * 8 / 100 - hours) * 60).toInt()
            "${hours}h ${minutes}m"
        } else ""
}

// ========== ANALYTICS AGGREGATION MODELS ==========

/**
 * Sleep analytics summary for dashboard
 */
data class SleepAnalyticsSummary(
    val totalSessions: Int,
    val totalSleepTime: Long,
    val averageDuration: Float,
    val averageQuality: Float,
    val averageEfficiency: Float,
    val bestStreak: Int,
    val currentStreak: Int,
    val improvementTrend: Float,
    val lastSessionDate: Long
) {
    val totalSleepHours: Float
        get() = totalSleepTime / (1000f * 60f * 60f)

    val averageDurationHours: Float
        get() = averageDuration / (1000f * 60f * 60f)

    val qualityGrade: String
        get() = when {
            averageQuality >= 8f -> "Excellent"
            averageQuality >= 6f -> "Good"
            averageQuality >= 4f -> "Fair"
            else -> "Poor"
        }

    val efficiencyGrade: String
        get() = when {
            averageEfficiency >= 85f -> "Excellent"
            averageEfficiency >= 75f -> "Good"
            averageEfficiency >= 65f -> "Fair"
            else -> "Poor"
        }

    val isImproving: Boolean
        get() = improvementTrend > 0.1f

    val daysSinceLastSession: Int
        get() = ((System.currentTimeMillis() - lastSessionDate) / (24 * 60 * 60 * 1000L)).toInt()

    val isConsistent: Boolean
        get() = currentStreak >= 7 // At least a week streak
}