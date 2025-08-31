package com.example.somniai.utils

import com.example.somniai.data.*
import org.json.JSONObject
import org.json.JSONArray
import org.json.JSONException
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.*
import java.util.regex.Pattern
import kotlin.math.*

/**
 * Format Utilities for SomniAI
 *
 * Specialized formatting utilities for AI prompts, response parsing,
 * and unit conversions. Provides consistent formatting across the app
 * and robust parsing for various data formats.
 */
object FormatUtils {

    // ========== FORMAT CONSTANTS ==========

    private val DECIMAL_FORMAT_1 = DecimalFormat("#.#")
    private val DECIMAL_FORMAT_2 = DecimalFormat("#.##")
    private val PERCENTAGE_FORMAT = DecimalFormat("#.#'%'")
    private val CURRENCY_FORMAT = DecimalFormat("$#.##")

    private val DATE_FORMAT_SHORT = SimpleDateFormat("MMM dd", Locale.getDefault())
    private val DATE_FORMAT_MEDIUM = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
    private val TIME_FORMAT_12H = SimpleDateFormat("h:mm a", Locale.getDefault())
    private val TIME_FORMAT_24H = SimpleDateFormat("HH:mm", Locale.getDefault())

    // Regex patterns for parsing
    private val DURATION_PATTERN = Pattern.compile("(\\d+)h\\s*(\\d+)m")
    private val PERCENTAGE_PATTERN = Pattern.compile("(\\d+(?:\\.\\d+)?)%")
    private val DECIMAL_PATTERN = Pattern.compile("(\\d+(?:\\.\\d+)?)")
    private val QUALITY_SCORE_PATTERN = Pattern.compile("(?:score|quality)\\s*:?\\s*(\\d+(?:\\.\\d+)?)")

    // ========== AI PROMPT FORMATTING ==========

    /**
     * Format sleep metrics for AI prompt inclusion
     */
    fun formatSleepMetricsForAI(
        quality: Float?,
        efficiency: Float,
        duration: Long,
        latency: Long
    ): String {
        return buildString {
            appendLine("SLEEP METRICS:")
            appendLine("• Quality Score: ${formatQualityScore(quality)}")
            appendLine("• Sleep Efficiency: ${formatPercentage(efficiency)}")
            appendLine("• Total Sleep Duration: ${formatDurationReadable(duration)}")
            appendLine("• Sleep Latency: ${formatDurationReadable(latency)}")
        }
    }

    /**
     * Format sleep architecture for AI analysis
     */
    fun formatSleepArchitectureForAI(
        lightSleep: Long,
        deepSleep: Long,
        remSleep: Long,
        awakeTime: Long
    ): String {
        val total = lightSleep + deepSleep + remSleep + awakeTime
        if (total == 0L) return "SLEEP ARCHITECTURE: No data available"

        return buildString {
            appendLine("SLEEP ARCHITECTURE:")
            appendLine("• Light Sleep: ${formatPhaseInfo(lightSleep, total)}")
            appendLine("• Deep Sleep: ${formatPhaseInfo(deepSleep, total)}")
            appendLine("• REM Sleep: ${formatPhaseInfo(remSleep, total)}")
            appendLine("• Awake Time: ${formatPhaseInfo(awakeTime, total)}")
            appendLine()
            appendLine("PHASE ANALYSIS:")
            appendLine("• Light Sleep Percentage: ${formatPhasePercentage(lightSleep, total)}")
            appendLine("• Deep Sleep Percentage: ${formatPhasePercentage(deepSleep, total)}")
            appendLine("• REM Sleep Percentage: ${formatPhasePercentage(remSleep, total)}")
            appendLine("• Sleep Efficiency: ${formatEfficiency(total - awakeTime, total)}")
        }
    }

    /**
     * Format environmental data for AI context
     */
    fun formatEnvironmentalDataForAI(
        avgMovement: Float,
        movementEvents: Int,
        avgNoise: Float,
        noiseEvents: Int,
        maxNoise: Float? = null
    ): String {
        return buildString {
            appendLine("ENVIRONMENTAL CONDITIONS:")
            appendLine("• Movement Analysis:")
            appendLine("  - Average Intensity: ${formatMovementIntensity(avgMovement)}")
            appendLine("  - Movement Events: ${movementEvents} detected")
            appendLine("  - Movement Level: ${categorizeMovementLevel(avgMovement)}")
            appendLine("• Noise Analysis:")
            appendLine("  - Average Level: ${formatNoiseLevel(avgNoise)}")
            appendLine("  - Noise Events: ${noiseEvents} detected")
            maxNoise?.let {
                appendLine("  - Peak Level: ${formatNoiseLevel(it)}")
            }
            appendLine("  - Noise Category: ${categorizeNoiseLevel(avgNoise)}")
        }
    }

    /**
     * Format sleep trends for AI analysis
     */
    fun formatTrendDataForAI(
        sessions: List<SleepSession>,
        period: String = "recent"
    ): String {
        if (sessions.isEmpty()) return "TREND DATA: No sessions available"

        val qualities = sessions.mapNotNull { it.sleepQualityScore }
        val durations = sessions.map { it.duration }
        val efficiencies = sessions.map { it.sleepEfficiency }

        return buildString {
            appendLine("SLEEP TRENDS ($period period):")
            appendLine("• Data Points: ${sessions.size} sessions")
            appendLine("• Date Range: ${formatDateRange(sessions)}")
            appendLine()
            appendLine("TREND ANALYSIS:")

            if (qualities.isNotEmpty()) {
                val avgQuality = qualities.average()
                val qualityTrend = calculateTrendDirection(qualities)
                appendLine("• Quality Score: ${formatDecimal(avgQuality.toFloat())} average (${qualityTrend})")
            }

            val avgDuration = durations.average()
            val durationTrend = calculateTrendDirection(durations.map { it.toFloat() })
            appendLine("• Sleep Duration: ${formatDurationReadable(avgDuration.toLong())} average (${durationTrend})")

            val avgEfficiency = efficiencies.average()
            val efficiencyTrend = calculateTrendDirection(efficiencies)
            appendLine("• Sleep Efficiency: ${formatPercentage(avgEfficiency.toFloat())} average (${efficiencyTrend})")

            appendLine()
            appendLine("CONSISTENCY METRICS:")
            appendLine("• Duration Variability: ${formatVariability(durations.map { it.toFloat() })}")
            appendLine("• Quality Consistency: ${formatConsistency(qualities)}")
            appendLine("• Overall Pattern: ${determineOverallPattern(sessions)}")
        }
    }

    /**
     * Format comparative data for AI analysis
     */
    fun formatComparativeDataForAI(
        currentValue: Float,
        benchmarkValue: Float,
        metric: String,
        unit: String = ""
    ): String {
        val difference = currentValue - benchmarkValue
        val percentChange = if (benchmarkValue != 0f) {
            (difference / benchmarkValue) * 100f
        } else 0f

        val comparisonText = when {
            abs(percentChange) < 5f -> "similar to"
            percentChange > 0 -> "above"
            else -> "below"
        }

        return "$metric: ${formatWithUnit(currentValue, unit)} " +
                "($comparisonText benchmark of ${formatWithUnit(benchmarkValue, unit)}, " +
                "${if (difference >= 0) "+" else ""}${formatPercentage(abs(percentChange))}"
    }

    // ========== RESPONSE PARSING UTILITIES ==========

    /**
     * Extract numerical values from AI responses
     */
    fun extractNumericValues(text: String): Map<String, Float> {
        val values = mutableMapOf<String, Float>()

        // Extract quality scores
        QUALITY_SCORE_PATTERN.matcher(text.lowercase()).let { matcher ->
            while (matcher.find()) {
                matcher.group(1)?.toFloatOrNull()?.let { score ->
                    values["quality"] = score
                }
            }
        }

        // Extract percentages
        PERCENTAGE_PATTERN.matcher(text).let { matcher ->
            val percentages = mutableListOf<Float>()
            while (matcher.find()) {
                matcher.group(1)?.toFloatOrNull()?.let { percentage ->
                    percentages.add(percentage)
                }
            }
            if (percentages.isNotEmpty()) {
                values["percentage_avg"] = percentages.average().toFloat()
            }
        }

        // Extract duration values
        DURATION_PATTERN.matcher(text).let { matcher ->
            while (matcher.find()) {
                val hours = matcher.group(1)?.toIntOrNull() ?: 0
                val minutes = matcher.group(2)?.toIntOrNull() ?: 0
                val totalMs = (hours * 60 + minutes) * 60 * 1000L
                values["duration_ms"] = totalMs.toFloat()
            }
        }

        return values
    }

    /**
     * Parse structured sections from AI response
     */
    fun parseResponseSections(response: String): Map<String, List<String>> {
        val sections = mutableMapOf<String, MutableList<String>>()
        val lines = response.lines()

        var currentSection: String? = null

        for (line in lines) {
            val trimmed = line.trim()

            when {
                isSectionHeader(trimmed) -> {
                    currentSection = extractSectionName(trimmed)
                    sections[currentSection] = mutableListOf()
                }
                trimmed.isNotEmpty() && currentSection != null -> {
                    sections[currentSection]?.add(trimmed)
                }
                trimmed.isNotEmpty() && currentSection == null -> {
                    // Default section for unstructured content
                    sections.getOrPut("general") { mutableListOf() }.add(trimmed)
                }
            }
        }

        return sections.mapValues { it.value }
    }

    /**
     * Extract confidence scores from AI responses
     */
    fun extractConfidenceScore(text: String): Float {
        val confidencePatterns = listOf(
            Pattern.compile("confidence[:\\s]+(\\d+(?:\\.\\d+)?)%"),
            Pattern.compile("confidence[:\\s]+(\\d+(?:\\.\\d+)?)"),
            Pattern.compile("(\\d+(?:\\.\\d+)?)%\\s+confidence"),
            Pattern.compile("certainty[:\\s]+(\\d+(?:\\.\\d+)?)%")
        )

        for (pattern in confidencePatterns) {
            val matcher = pattern.matcher(text.lowercase())
            if (matcher.find()) {
                matcher.group(1)?.toFloatOrNull()?.let { confidence ->
                    return if (confidence > 1f) confidence / 100f else confidence
                }
            }
        }

        return 0.7f // Default confidence
    }

    /**
     * Extract recommendations from AI response
     */
    fun extractRecommendations(response: String): List<String> {
        val recommendations = mutableListOf<String>()
        val lines = response.lines()

        var inRecommendationSection = false

        for (line in lines) {
            val trimmed = line.trim()

            when {
                isRecommendationHeader(trimmed) -> {
                    inRecommendationSection = true
                }
                inRecommendationSection && (trimmed.startsWith("•") ||
                        trimmed.startsWith("-") || trimmed.startsWith("*") ||
                        trimmed.matches("\\d+\\..*".toRegex())) -> {
                    val cleaned = cleanRecommendationText(trimmed)
                    if (cleaned.isNotEmpty()) {
                        recommendations.add(cleaned)
                    }
                }
                inRecommendationSection && trimmed.isEmpty() -> {
                    // Continue in section
                }
                inRecommendationSection && isSectionHeader(trimmed) -> {
                    inRecommendationSection = false
                }
                !inRecommendationSection && containsRecommendationKeywords(trimmed) -> {
                    val cleaned = cleanRecommendationText(trimmed)
                    if (cleaned.isNotEmpty()) {
                        recommendations.add(cleaned)
                    }
                }
            }
        }

        return recommendations
    }

    /**
     * Parse JSON response with error handling
     */
    fun parseJsonResponse(jsonString: String): JSONObject? {
        return try {
            JSONObject(jsonString.trim())
        } catch (e: JSONException) {
            // Try to extract JSON from text
            extractJsonFromText(jsonString)
        }
    }

    // ========== UNIT CONVERSION HELPERS ==========

    /**
     * Convert time units
     */
    fun convertTime(
        value: Number,
        fromUnit: TimeUnit,
        toUnit: TimeUnit
    ): Long {
        val milliseconds = when (fromUnit) {
            TimeUnit.MILLISECONDS -> value.toLong()
            TimeUnit.SECONDS -> value.toLong() * 1000L
            TimeUnit.MINUTES -> value.toLong() * 60 * 1000L
            TimeUnit.HOURS -> value.toLong() * 60 * 60 * 1000L
            TimeUnit.DAYS -> value.toLong() * 24 * 60 * 60 * 1000L
        }

        return when (toUnit) {
            TimeUnit.MILLISECONDS -> milliseconds
            TimeUnit.SECONDS -> milliseconds / 1000L
            TimeUnit.MINUTES -> milliseconds / (60 * 1000L)
            TimeUnit.HOURS -> milliseconds / (60 * 60 * 1000L)
            TimeUnit.DAYS -> milliseconds / (24 * 60 * 60 * 1000L)
        }
    }

    /**
     * Convert temperature units
     */
    fun convertTemperature(value: Float, fromUnit: TemperatureUnit, toUnit: TemperatureUnit): Float {
        if (fromUnit == toUnit) return value

        // Convert to Celsius first
        val celsius = when (fromUnit) {
            TemperatureUnit.CELSIUS -> value
            TemperatureUnit.FAHRENHEIT -> (value - 32f) * 5f / 9f
            TemperatureUnit.KELVIN -> value - 273.15f
        }

        // Convert from Celsius to target
        return when (toUnit) {
            TemperatureUnit.CELSIUS -> celsius
            TemperatureUnit.FAHRENHEIT -> celsius * 9f / 5f + 32f
            TemperatureUnit.KELVIN -> celsius + 273.15f
        }
    }

    /**
     * Convert length units
     */
    fun convertLength(value: Float, fromUnit: LengthUnit, toUnit: LengthUnit): Float {
        if (fromUnit == toUnit) return value

        // Convert to meters first
        val meters = when (fromUnit) {
            LengthUnit.MILLIMETERS -> value / 1000f
            LengthUnit.CENTIMETERS -> value / 100f
            LengthUnit.METERS -> value
            LengthUnit.KILOMETERS -> value * 1000f
            LengthUnit.INCHES -> value * 0.0254f
            LengthUnit.FEET -> value * 0.3048f
            LengthUnit.YARDS -> value * 0.9144f
            LengthUnit.MILES -> value * 1609.344f
        }

        // Convert from meters to target
        return when (toUnit) {
            LengthUnit.MILLIMETERS -> meters * 1000f
            LengthUnit.CENTIMETERS -> meters * 100f
            LengthUnit.METERS -> meters
            LengthUnit.KILOMETERS -> meters / 1000f
            LengthUnit.INCHES -> meters / 0.0254f
            LengthUnit.FEET -> meters / 0.3048f
            LengthUnit.YARDS -> meters / 0.9144f
            LengthUnit.MILES -> meters / 1609.344f
        }
    }

    /**
     * Convert weight units
     */
    fun convertWeight(value: Float, fromUnit: WeightUnit, toUnit: WeightUnit): Float {
        if (fromUnit == toUnit) return value

        // Convert to grams first
        val grams = when (fromUnit) {
            WeightUnit.GRAMS -> value
            WeightUnit.KILOGRAMS -> value * 1000f
            WeightUnit.POUNDS -> value * 453.592f
            WeightUnit.OUNCES -> value * 28.3495f
        }

        // Convert from grams to target
        return when (toUnit) {
            WeightUnit.GRAMS -> grams
            WeightUnit.KILOGRAMS -> grams / 1000f
            WeightUnit.POUNDS -> grams / 453.592f
            WeightUnit.OUNCES -> grams / 28.3495f
        }
    }

    // ========== STRING FORMATTING UTILITIES ==========

    /**
     * Format quality score with descriptive text
     */
    fun formatQualityScore(score: Float?): String {
        return if (score != null) {
            val grade = when {
                score >= 9f -> "Excellent"
                score >= 8f -> "Very Good"
                score >= 7f -> "Good"
                score >= 6f -> "Fair"
                score >= 5f -> "Poor"
                else -> "Very Poor"
            }
            "${DECIMAL_FORMAT_1.format(score)}/10 ($grade)"
        } else {
            "Not calculated"
        }
    }

    /**
     * Format percentage values
     */
    fun formatPercentage(value: Float): String {
        return PERCENTAGE_FORMAT.format(value)
    }

    /**
     * Format decimal numbers
     */
    fun formatDecimal(value: Float, places: Int = 1): String {
        return when (places) {
            1 -> DECIMAL_FORMAT_1.format(value)
            2 -> DECIMAL_FORMAT_2.format(value)
            else -> String.format("%.${places}f", value)
        }
    }

    /**
     * Format currency values
     */
    fun formatCurrency(value: Double): String {
        return CURRENCY_FORMAT.format(value)
    }

    /**
     * Format duration in human-readable format
     */
    fun formatDurationReadable(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)

        return when {
            hours > 0 -> "${hours}h ${minutes}m"
            minutes > 0 -> "${minutes}m"
            else -> "${durationMs / 1000}s"
        }
    }

    /**
     * Format noise level with unit
     */
    fun formatNoiseLevel(decibelLevel: Float): String {
        return "${DECIMAL_FORMAT_1.format(decibelLevel)} dB"
    }

    /**
     * Format movement intensity
     */
    fun formatMovementIntensity(intensity: Float): String {
        return "${DECIMAL_FORMAT_1.format(intensity)}/10"
    }

    /**
     * Format value with unit
     */
    fun formatWithUnit(value: Float, unit: String): String {
        return "${DECIMAL_FORMAT_1.format(value)}${if (unit.isNotEmpty()) " $unit" else ""}"
    }

    /**
     * Format large numbers with K/M suffixes
     */
    fun formatLargeNumber(value: Long): String {
        return when {
            value >= 1_000_000 -> "${DECIMAL_FORMAT_1.format(value / 1_000_000.0)}M"
            value >= 1_000 -> "${DECIMAL_FORMAT_1.format(value / 1_000.0)}K"
            else -> value.toString()
        }
    }

    // ========== VALIDATION UTILITIES ==========

    /**
     * Validate sleep quality score
     */
    fun isValidQualityScore(score: Float): Boolean {
        return score in 0f..10f
    }

    /**
     * Validate sleep efficiency percentage
     */
    fun isValidEfficiency(efficiency: Float): Boolean {
        return efficiency in 0f..100f
    }

    /**
     * Validate duration (reasonable sleep duration)
     */
    fun isValidSleepDuration(durationMs: Long): Boolean {
        val hours = durationMs / (1000f * 60 * 60)
        return hours in 0.5f..16f // 30 minutes to 16 hours
    }

    /**
     * Validate noise level
     */
    fun isValidNoiseLevel(decibelLevel: Float): Boolean {
        return decibelLevel in 0f..120f // 0 to 120 dB
    }

    /**
     * Validate timestamp
     */
    fun isValidTimestamp(timestamp: Long): Boolean {
        val now = System.currentTimeMillis()
        val oneYearAgo = now - (365L * 24 * 60 * 60 * 1000L)
        val oneDayFuture = now + (24 * 60 * 60 * 1000L)
        return timestamp in oneYearAgo..oneDayFuture
    }

    // ========== HELPER METHODS ==========

    private fun formatPhaseInfo(duration: Long, total: Long): String {
        val percentage = if (total > 0) (duration.toFloat() / total * 100f) else 0f
        return "${formatDurationReadable(duration)} (${DECIMAL_FORMAT_1.format(percentage)}%)"
    }

    private fun formatPhasePercentage(duration: Long, total: Long): String {
        val percentage = if (total > 0) (duration.toFloat() / total * 100f) else 0f
        return "${DECIMAL_FORMAT_1.format(percentage)}%"
    }

    private fun formatEfficiency(sleepTime: Long, totalTime: Long): String {
        val efficiency = if (totalTime > 0) (sleepTime.toFloat() / totalTime * 100f) else 0f
        return "${DECIMAL_FORMAT_1.format(efficiency)}%"
    }

    private fun categorizeMovementLevel(intensity: Float): String {
        return when {
            intensity < 2f -> "Very Low"
            intensity < 4f -> "Low"
            intensity < 6f -> "Moderate"
            intensity < 8f -> "High"
            else -> "Very High"
        }
    }

    private fun categorizeNoiseLevel(noiseLevel: Float): String {
        return when {
            noiseLevel < 30f -> "Very Quiet"
            noiseLevel < 40f -> "Quiet"
            noiseLevel < 50f -> "Moderate"
            noiseLevel < 60f -> "Noisy"
            else -> "Very Noisy"
        }
    }

    private fun formatDateRange(sessions: List<SleepSession>): String {
        if (sessions.isEmpty()) return "No data"
        val earliest = sessions.minOf { it.startTime }
        val latest = sessions.maxOf { it.startTime }
        return "${DATE_FORMAT_SHORT.format(Date(earliest))} - ${DATE_FORMAT_SHORT.format(Date(latest))}"
    }

    private fun calculateTrendDirection(values: List<Float>): String {
        if (values.size < 2) return "Insufficient data"

        val firstHalf = values.take(values.size / 2).average()
        val secondHalf = values.drop(values.size / 2).average()
        val change = ((secondHalf - firstHalf) / firstHalf * 100).toFloat()

        return when {
            abs(change) < 5f -> "Stable"
            change > 10f -> "Improving"
            change > 5f -> "Slightly improving"
            change < -10f -> "Declining"
            change < -5f -> "Slightly declining"
            else -> "Variable"
        }
    }

    private fun formatVariability(values: List<Float>): String {
        if (values.size < 2) return "N/A"

        val mean = values.average()
        val variance = values.map { (it - mean).pow(2) }.average()
        val coefficientOfVariation = sqrt(variance) / mean

        return when {
            coefficientOfVariation < 0.1 -> "Very Low"
            coefficientOfVariation < 0.2 -> "Low"
            coefficientOfVariation < 0.3 -> "Moderate"
            coefficientOfVariation < 0.5 -> "High"
            else -> "Very High"
        }
    }

    private fun formatConsistency(values: List<Float>): String {
        return formatVariability(values)
    }

    private fun determineOverallPattern(sessions: List<SleepSession>): String {
        // Simplified pattern detection
        val qualities = sessions.mapNotNull { it.sleepQualityScore }
        val durations = sessions.map { it.duration }

        val qualityTrend = if (qualities.isNotEmpty()) calculateTrendDirection(qualities) else "Unknown"
        val durationTrend = calculateTrendDirection(durations.map { it.toFloat() })

        return when {
            qualityTrend == "Improving" && durationTrend == "Stable" -> "Positive development"
            qualityTrend == "Stable" && durationTrend == "Stable" -> "Consistent patterns"
            qualityTrend == "Declining" -> "Needs attention"
            else -> "Mixed patterns"
        }
    }

    private fun isSectionHeader(line: String): Boolean {
        val upperLine = line.uppercase()
        return (upperLine.endsWith(":") || upperLine.matches(".*[A-Z]{2,}.*".toRegex())) &&
                (upperLine.contains("RECOMMENDATION") || upperLine.contains("QUALITY") ||
                        upperLine.contains("FACTOR") || upperLine.contains("PATTERN") ||
                        upperLine.contains("ANALYSIS") || upperLine.contains("SUMMARY"))
    }

    private fun extractSectionName(line: String): String {
        return line.replace(":", "").trim().lowercase()
    }

    private fun isRecommendationHeader(line: String): Boolean {
        val lower = line.lowercase()
        return lower.contains("recommendation") || lower.contains("suggest") ||
                lower.contains("advice") || lower.contains("improve")
    }

    private fun cleanRecommendationText(text: String): String {
        return text.replace(Regex("^[•\\-*\\d+\\.]\\s*"), "").trim()
    }

    private fun containsRecommendationKeywords(text: String): Boolean {
        val lower = text.lowercase()
        val keywords = listOf("recommend", "suggest", "consider", "try", "avoid", "improve")
        return keywords.any { lower.contains(it) }
    }

    private fun extractJsonFromText(text: String): JSONObject? {
        // Try to find JSON within text
        val jsonPattern = Pattern.compile("\\{[^{}]*\\}")
        val matcher = jsonPattern.matcher(text)

        while (matcher.find()) {
            try {
                return JSONObject(matcher.group())
            } catch (e: JSONException) {
                // Continue searching
            }
        }

        return null
    }
}

// ========== ENUM CLASSES FOR UNITS ==========

enum class TimeUnit {
    MILLISECONDS,
    SECONDS,
    MINUTES,
    HOURS,
    DAYS
}

enum class TemperatureUnit {
    CELSIUS,
    FAHRENHEIT,
    KELVIN
}

enum class LengthUnit {
    MILLIMETERS,
    CENTIMETERS,
    METERS,
    KILOMETERS,
    INCHES,
    FEET,
    YARDS,
    MILES
}

enum class WeightUnit {
    GRAMS,
    KILOGRAMS,
    POUNDS,
    OUNCES
}