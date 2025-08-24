package com.example.somniai.data

import android.hardware.SensorManager
import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import com.example.somniai.data.*
import com.example.somniai.data.InsightCategory

// Add this to SensorDataModels.kt after the existing enums:
enum class DataIntegrityStatus {
    HEALTHY,
    WARNING,
    CRITICAL,
    UNKNOWN
}


/**
 * Sleep phase indicators based on movement and noise patterns
 */
enum class SleepPhase {
    AWAKE,          // High movement/noise
    LIGHT_SLEEP,    // Moderate movement, some noise sensitivity
    DEEP_SLEEP,     // Minimal movement, low noise sensitivity
    REM_SLEEP,      // Increased movement, stable noise level
    UNKNOWN;        // Insufficient data

    fun getDisplayName(): String {
        return when (this) {
            AWAKE -> "Awake"
            LIGHT_SLEEP -> "Light Sleep"
            DEEP_SLEEP -> "Deep Sleep"
            REM_SLEEP -> "REM Sleep"
            UNKNOWN -> "Unknown"
        }
    }

    fun getColor(): String {
        return when (this) {
            AWAKE -> "#FF6B6B"        // Red
            LIGHT_SLEEP -> "#4ECDC4"  // Teal
            DEEP_SLEEP -> "#45B7D1"   // Blue
            REM_SLEEP -> "#96CEB4"    // Green
            UNKNOWN -> "#95A5A6"      // Gray
        }
    }
}

/**
 * Sleep trend indicators

enum class SleepTrend(val displayName: String) {
    IMPROVING("Improving"),
    STABLE("Stable"),
    DECLINING("Declining"),
    INSUFFICIENT_DATA("Insufficient Data");

    fun getColor(): String {
        return when (this) {
            IMPROVING -> "#4CAF50"           // Green
            STABLE -> "#2196F3"             // Blue
            DECLINING -> "#FF5722"          // Red
            INSUFFICIENT_DATA -> "#757575"   // Gray
        }
    }
}*/


/**
 * Movement event from accelerometer sensor
 */
@Parcelize
data class MovementEvent(
    val id: Long = 0,
    val sessionId: Long = 0,
    val timestamp: Long,
    val intensity: Float,
    val x: Float,
    val y: Float,
    val z: Float
) : Parcelable {

    fun getIntensityLevel(): String {
        return when {
            intensity <= 1.0f -> "Very Low"
            intensity <= 2.0f -> "Low"
            intensity <= 4.0f -> "Medium"
            intensity <= 6.0f -> "High"
            else -> "Very High"
        }
    }

    fun isSignificant(): Boolean = intensity > 2.0f
}

/**
 * Noise event from microphone sensor
 */
@Parcelize
data class NoiseEvent(
    val id: Long = 0,
    val sessionId: Long = 0,
    val timestamp: Long,
    val decibelLevel: Float,
    val amplitude: Int
) : Parcelable {

    fun getNoiseLevel(): String {
        return when {
            decibelLevel <= 30f -> "Very Quiet"
            decibelLevel <= 40f -> "Quiet"
            decibelLevel <= 50f -> "Moderate"
            decibelLevel <= 60f -> "Loud"
            else -> "Very Loud"
        }
    }

    fun isDisruptive(): Boolean = decibelLevel > 50f
}

/**
 * Real-time sensor status for UI updates
 */
@Parcelize
data class SensorStatus(
    val isAccelerometerActive: Boolean = false,
    val isMicrophoneActive: Boolean = false,
    val currentMovementIntensity: Float = 0f,
    val currentNoiseLevel: Float = 0f,
    val movementThreshold: Float = 2.0f,
    val noiseThreshold: Int = 1000,
    val accelerometerStatus: String = "Inactive",
    val microphoneStatus: String = "Inactive",
    val totalMovementEvents: Int = 0,
    val totalNoiseEvents: Int = 0,
    val sessionStartTime: Long? = null,
    val currentPhase: SleepPhase = SleepPhase.UNKNOWN,
    val phaseConfidence: Float = 0f
) : Parcelable {

    val isFullyActive: Boolean
        get() = isAccelerometerActive && isMicrophoneActive

    val sessionDuration: Long
        get() = sessionStartTime?.let { System.currentTimeMillis() - it } ?: 0L

    fun getOverallStatus(): String {
        return when {
            !isAccelerometerActive && !isMicrophoneActive -> "Sensors Inactive"
            !isAccelerometerActive -> "Accelerometer Issue"
            !isMicrophoneActive -> "Microphone Issue"
            else -> "All Sensors Active"
        }
    }

    fun getSessionDurationFormatted(): String {
        val duration = sessionDuration
        val hours = duration / (1000 * 60 * 60)
        val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)
        return String.format("%02d:%02d", hours, minutes)
    }
}

/**
 * Configuration settings for sensor sensitivity and behavior
 */
@Parcelize
data class SensorSettings(
    val movementThreshold: Float = 2.0f,
    val noiseThreshold: Int = 1000,
    val movementSamplingRate: Int = SensorManager.SENSOR_DELAY_NORMAL,
    val noiseSamplingInterval: Long = 1000L,
    val enableMovementDetection: Boolean = true,
    val enableNoiseDetection: Boolean = true,
    val enableSmartFiltering: Boolean = true,
    val autoAdjustSensitivity: Boolean = false
) : Parcelable {

    fun getMovementSensitivityLevel(): String {
        return when {
            movementThreshold <= 1.0f -> "Very High"
            movementThreshold <= 2.0f -> "High"
            movementThreshold <= 3.0f -> "Medium"
            movementThreshold <= 4.0f -> "Low"
            else -> "Very Low"
        }
    }

    fun getNoiseSensitivityLevel(): String {
        return when {
            noiseThreshold <= 500 -> "Very High"
            noiseThreshold <= 1000 -> "High"
            noiseThreshold <= 1500 -> "Medium"
            noiseThreshold <= 2000 -> "Low"
            else -> "Very Low"
        }
    }

    fun isOptimalConfiguration(): Boolean {
        return movementThreshold in 1.5f..3.0f &&
                noiseThreshold in 800..1500 &&
                enableMovementDetection && enableNoiseDetection
    }
}

/**
 * Sleep quality factors breakdown
 */
@Parcelize
data class SleepQualityFactors(
    val movementScore: Float,      // 0-10 (10 = very still)
    val noiseScore: Float,         // 0-10 (10 = very quiet)
    val durationScore: Float,      // 0-10 (10 = optimal duration)
    val consistencyScore: Float,   // 0-10 (10 = consistent sleep)
    val overallScore: Float        // Weighted average
) : Parcelable {

    fun getOverallGrade(): String {
        return when {
            overallScore >= 9f -> "Excellent"
            overallScore >= 8f -> "Very Good"
            overallScore >= 7f -> "Good"
            overallScore >= 6f -> "Fair"
            overallScore >= 5f -> "Poor"
            else -> "Very Poor"
        }
    }

    fun getWorstFactor(): String {
        val factors = mapOf(
            "Movement" to movementScore,
            "Noise" to noiseScore,
            "Duration" to durationScore,
            "Consistency" to consistencyScore
        )
        return factors.minByOrNull { it.value }?.key ?: "Unknown"
    }

    fun getBestFactor(): String {
        val factors = mapOf(
            "Movement" to movementScore,
            "Noise" to noiseScore,
            "Duration" to durationScore,
            "Consistency" to consistencyScore
        )
        return factors.maxByOrNull { it.value }?.key ?: "Unknown"
    }
}

/**
 * Sleep phase transition event
 */
@Parcelize
data class PhaseTransition(
    val timestamp: Long,
    val fromPhase: SleepPhase,
    val toPhase: SleepPhase,
    val confidence: Float
) : Parcelable {

    fun getTransitionType(): String {
        return "${fromPhase.getDisplayName()} → ${toPhase.getDisplayName()}"
    }

    fun isImprovement(): Boolean {
        val phaseOrder = listOf(SleepPhase.AWAKE, SleepPhase.LIGHT_SLEEP, SleepPhase.DEEP_SLEEP, SleepPhase.REM_SLEEP)
        val fromIndex = phaseOrder.indexOf(fromPhase)
        val toIndex = phaseOrder.indexOf(toPhase)
        return toIndex > fromIndex
    }
}

/**
 * Real-time sleep metrics during active tracking
 */
@Parcelize
data class LiveSleepMetrics(
    val currentPhase: SleepPhase,
    val phaseConfidence: Float,    // 0-1 confidence in phase detection
    val movementIntensity: Float,  // Current movement level
    val noiseLevel: Float,         // Current noise level
    val timeInCurrentPhase: Long,  // Time spent in current phase (ms)
    val totalRestlessness: Float,  // Overall restlessness score (0-10)
    val sleepEfficiency: Float     // Estimated sleep efficiency percentage
) : Parcelable {

    fun getPhaseConfidenceLevel(): String {
        return when {
            phaseConfidence >= 0.8f -> "High"
            phaseConfidence >= 0.6f -> "Medium"
            phaseConfidence >= 0.4f -> "Low"
            else -> "Very Low"
        }
    }

    fun getRestlessnessLevel(): String {
        return when {
            totalRestlessness <= 2f -> "Very Calm"
            totalRestlessness <= 4f -> "Calm"
            totalRestlessness <= 6f -> "Moderate"
            totalRestlessness <= 8f -> "Restless"
            else -> "Very Restless"
        }
    }

    fun getSleepEfficiencyGrade(): String {
        return when {
            sleepEfficiency >= 95f -> "Excellent"
            sleepEfficiency >= 85f -> "Very Good"
            sleepEfficiency >= 75f -> "Good"
            sleepEfficiency >= 65f -> "Fair"
            else -> "Poor"
        }
    }
}

/**
 * Comprehensive sleep session with all data and analytics
 */
@Parcelize
data class SleepSession(
    val id: Long = 0,
    val startTime: Long,
    val endTime: Long? = null,
    val totalDuration: Long = 0,
    val sessionDuration: Long = 0L,
    val sleepLatency: Long = 0,        // Time to fall asleep
    val awakeDuration: Long = 0,       // Time spent awake during session
    val lightSleepDuration: Long = 0,
    val deepSleepDuration: Long = 0,
    val remSleepDuration: Long = 0,
    val sleepEfficiency: Float = 0f,   // Percentage of time actually sleeping
    val confidence: Float = 0.0f,
    val movementEvents: List<MovementEvent> = emptyList(),
    val noiseEvents: List<NoiseEvent> = emptyList(),
    val phaseTransitions: List<PhaseTransition> = emptyList(),
    val sleepQualityScore: Float? = null,
    val qualityFactors: SleepQualityFactors? = null,
    val averageMovementIntensity: Float = 0f,
    val averageNoiseLevel: Float = 0f,
    val movementFrequency: Float = 0f, // Movements per hour
    val notes: String = "",
    val settings: SensorSettings? = null
) : Parcelable {

    val isActive: Boolean
        get() = endTime == null

    val duration: Long
        get() = if (endTime != null) endTime - startTime else System.currentTimeMillis() - startTime

    val actualSleepDuration: Long
        get() = duration - awakeDuration

    fun calculateQualityScore(): Float {
        // Simple quality calculation if not already set
        if (sleepQualityScore != null) return sleepQualityScore

        val movementPenalty = (averageMovementIntensity / 10f).coerceAtMost(3f)
        val noisePenalty = (averageNoiseLevel / 20f).coerceAtMost(2f)
        val durationBonus = if (actualSleepDuration > 6 * 3600000) 1f else 0f
        val efficiencyBonus = if (sleepEfficiency > 85f) 1f else 0f

        return (10f - movementPenalty - noisePenalty + durationBonus + efficiencyBonus)
            .coerceIn(1f, 10f)
    }

    fun getDurationFormatted(): String {
        val hours = duration / (1000 * 60 * 60)
        val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)
        return String.format("%dh %02dm", hours, minutes)
    }

    fun getSleepLatencyFormatted(): String {
        val minutes = sleepLatency / (1000 * 60)
        return "${minutes}m"
    }

    fun getMainSleepPhase(): SleepPhase {
        val phaseDurations = mapOf(
            SleepPhase.LIGHT_SLEEP to lightSleepDuration,
            SleepPhase.DEEP_SLEEP to deepSleepDuration,
            SleepPhase.REM_SLEEP to remSleepDuration
        )
        return phaseDurations.maxByOrNull { it.value }?.key ?: SleepPhase.UNKNOWN
    }

    fun getQualityGrade(): String {
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

    fun getPhaseBreakdown(): Map<SleepPhase, Long> {
        return mapOf(
            SleepPhase.AWAKE to awakeDuration,
            SleepPhase.LIGHT_SLEEP to lightSleepDuration,
            SleepPhase.DEEP_SLEEP to deepSleepDuration,
            SleepPhase.REM_SLEEP to remSleepDuration
        )
    }

    fun getTotalMovements(): Int = movementEvents.size
    fun getTotalNoiseEvents(): Int = noiseEvents.size
    fun getTotalPhaseChanges(): Int = phaseTransitions.size
}

/**
 * Sleep analytics for trends and insights
 */
@Parcelize
data class SleepAnalytics(
    val sessions: List<SleepSession>,
    val averageDuration: Long,
    val averageQuality: Float,
    val averageSleepEfficiency: Float,
    val totalMovementEvents: Int,
    val totalNoiseEvents: Int,
    val bestSleepDate: Long? = null,
    val worstSleepDate: Long? = null,
    val sleepTrend: SleepTrend,
    val recommendations: List<String> = emptyList()
) : Parcelable {

    fun getAverageDurationFormatted(): String {
        val hours = averageDuration / (1000 * 60 * 60)
        val minutes = (averageDuration % (1000 * 60 * 60)) / (1000 * 60)
        return String.format("%dh %02dm", hours, minutes)
    }

    fun getQualityGrade(): String {
        return when {
            averageQuality >= 9f -> "Excellent"
            averageQuality >= 8f -> "Very Good"
            averageQuality >= 7f -> "Good"
            averageQuality >= 6f -> "Fair"
            else -> "Needs Improvement"
        }
    }

    fun getEfficiencyGrade(): String {
        return when {
            averageSleepEfficiency >= 95f -> "Excellent"
            averageSleepEfficiency >= 85f -> "Very Good"
            averageSleepEfficiency >= 75f -> "Good"
            averageSleepEfficiency >= 65f -> "Fair"
            else -> "Poor"
        }
    }

    fun getTopRecommendation(): String? {
        return recommendations.firstOrNull()
    }

    fun hasImprovedOverTime(): Boolean {
        return sleepTrend == SleepTrend.IMPROVING
    }
}


/**
 * AI-generated sleep insight
 */
@Parcelize
data class SleepInsight(
    val id: Long = 0,
    val sessionId: Long = 0,
    val category: InsightCategory,
    val title: String,
    val description: String,
    val recommendation: String,
    val priority: Int = 1, // 1 = high, 2 = medium, 3 = low
    val timestamp: Long = System.currentTimeMillis(),
    val isAiGenerated: Boolean = false
) : Parcelable {

    fun getPriorityLevel(): String {
        return when (priority) {
            1 -> "High"
            2 -> "Medium"
            else -> "Low"
        }
    }

    fun getPriorityColor(): String {
        return when (priority) {
            1 -> "#FF5722" // Red
            2 -> "#FF9800" // Orange
            else -> "#4CAF50" // Green
        }
    }

    /**
     * Simple quality factor analysis for basic analytics
     */
    @Parcelize
    data class QualityFactorAnalysis(
        val movementQuality: Float = 0.0f,
        val noiseQuality: Float = 0.0f,
        val durationQuality: Float = 0.0f,
        val consistencyQuality: Float = 0.0f
    ) : Parcelable

    /**
     * Daily trend data for charts and analytics
     */
    @Parcelize
    data class DailyTrendData(
        val date: String,
        val qualityScore: Float,
        val duration: Long,
        val efficiency: Float
    ) : Parcelable

    /**
     * Session summary for data transfer
     */
    @Parcelize
    data class SessionSummaryDTO(
        val id: Long,
        val startTime: Long,
        val endTime: Long,
        val duration: Long,
        val qualityScore: Float,
        val efficiency: Float
    ) : Parcelable
}