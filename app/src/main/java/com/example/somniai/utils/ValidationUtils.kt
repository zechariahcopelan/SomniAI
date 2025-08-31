package com.example.somniai.utils

import com.example.somniai.data.*
import com.example.somniai.ai.*
import org.json.JSONObject
import org.json.JSONException
import java.util.*
import java.util.regex.Pattern
import kotlin.math.*

/**
 * Validation Utilities for SomniAI
 *
 * Comprehensive validation system for input data, AI processing, and output validation.
 * Ensures data integrity, prevents system errors, and maintains data quality standards
 * throughout the sleep analysis pipeline.
 */
object ValidationUtils {

    // ========== VALIDATION CONSTANTS ==========

    // Sleep data ranges based on sleep science
    private const val MIN_SLEEP_QUALITY = 0f
    private const val MAX_SLEEP_QUALITY = 10f
    private const val MIN_SLEEP_EFFICIENCY = 0f
    private const val MAX_SLEEP_EFFICIENCY = 100f

    // Duration limits (in milliseconds)
    private const val MIN_SLEEP_DURATION = 30 * 60 * 1000L // 30 minutes
    private const val MAX_SLEEP_DURATION = 16 * 60 * 60 * 1000L // 16 hours
    private const val MIN_SLEEP_LATENCY = 0L
    private const val MAX_SLEEP_LATENCY = 4 * 60 * 60 * 1000L // 4 hours

    // Environmental limits
    private const val MIN_NOISE_LEVEL = 0f
    private const val MAX_NOISE_LEVEL = 120f // dB
    private const val MIN_MOVEMENT_INTENSITY = 0f
    private const val MAX_MOVEMENT_INTENSITY = 100f
    private const val MIN_TEMPERATURE = -20f // Celsius
    private const val MAX_TEMPERATURE = 50f // Celsius

    // Time validation
    private const val ONE_YEAR_MS = 365L * 24 * 60 * 60 * 1000L
    private const val ONE_DAY_MS = 24 * 60 * 60 * 1000L

    // AI validation
    private const val MAX_PROMPT_LENGTH = 10000
    private const val MIN_CONFIDENCE_THRESHOLD = 0f
    private const val MAX_CONFIDENCE_THRESHOLD = 1f
    private const val MAX_RESPONSE_LENGTH = 50000

    // Regex patterns for validation
    private val EMAIL_PATTERN = Pattern.compile(
        "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    )
    private val MALICIOUS_PATTERN = Pattern.compile(
        "(?i)(script|javascript|vbscript|onload|onerror|eval|exec|system|cmd|shell)",
        Pattern.CASE_INSENSITIVE
    )

    // ========== INPUT DATA VALIDATION FOR AI PROCESSING ==========

    /**
     * Validate sleep session data before AI processing
     */
    fun validateSleepSession(session: SleepSession): ValidationResult {
        val errors = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        var score = 1.0f

        // Core validation
        if (session.id <= 0) {
            errors.add("Invalid session ID")
            score -= 0.2f
        }

        if (!isValidTimestamp(session.startTime)) {
            errors.add("Invalid start time")
            score -= 0.3f
        }

        session.endTime?.let { endTime ->
            if (!isValidTimestamp(endTime)) {
                errors.add("Invalid end time")
                score -= 0.3f
            } else if (endTime <= session.startTime) {
                errors.add("End time must be after start time")
                score -= 0.3f
            }
        }

        // Duration validation
        if (!isValidDuration(session.duration)) {
            errors.add("Duration outside valid range (30min - 16h)")
            score -= 0.2f
        }

        // Quality validation
        session.sleepQualityScore?.let { quality ->
            if (!isValidQualityScore(quality)) {
                errors.add("Sleep quality score outside valid range (0-10)")
                score -= 0.2f
            }
        }

        // Efficiency validation
        if (!isValidEfficiency(session.sleepEfficiency)) {
            errors.add("Sleep efficiency outside valid range (0-100%)")
            score -= 0.2f
        }

        // Sleep latency validation
        if (!isValidLatency(session.sleepLatency)) {
            errors.add("Sleep latency outside valid range (0-4h)")
            score -= 0.1f
        }

        // Sleep phases validation
        val totalPhases = session.lightSleepDuration + session.deepSleepDuration +
                session.remSleepDuration + session.awakeDuration
        if (totalPhases > session.duration * 1.1) { // Allow 10% tolerance
            warnings.add("Sleep phases sum exceeds total duration")
            score -= 0.1f
        }

        // Environmental data validation
        if (!isValidMovementIntensity(session.averageMovementIntensity)) {
            warnings.add("Movement intensity outside expected range")
            score -= 0.05f
        }

        if (!isValidNoiseLevel(session.averageNoiseLevel)) {
            warnings.add("Noise level outside expected range")
            score -= 0.05f
        }

        // Confidence validation
        if (session.confidence < 0f || session.confidence > 1f) {
            warnings.add("Confidence score outside valid range (0-1)")
            score -= 0.1f
        }

        return ValidationResult(
            isValid = errors.isEmpty(),
            validationScore = score.coerceIn(0f, 1f),
            validationChecks = mapOf(
                "core_data" to errors.isEmpty(),
                "duration_valid" to isValidDuration(session.duration),
                "quality_valid" to (session.sleepQualityScore?.let { isValidQualityScore(it) } ?: true),
                "efficiency_valid" to isValidEfficiency(session.sleepEfficiency),
                "timestamps_valid" to isValidTimestamp(session.startTime)
            ),
            validationMessages = errors + warnings,
            validationDate = System.currentTimeMillis()
        )
    }

    /**
     * Validate AI prompt data before sending
     */
    fun validateAIPrompt(prompt: String, includesSensitiveData: Boolean = false): PromptValidationResult {
        val issues = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        var score = 1.0f

        // Length validation
        if (prompt.isBlank()) {
            issues.add("Prompt cannot be empty")
            return PromptValidationResult(false, 0f, issues, warnings, false)
        }

        if (prompt.length > MAX_PROMPT_LENGTH) {
            issues.add("Prompt exceeds maximum length ($MAX_PROMPT_LENGTH characters)")
            score -= 0.3f
        }

        // Content safety validation
        if (containsMaliciousContent(prompt)) {
            issues.add("Prompt contains potentially malicious content")
            score -= 0.5f
        }

        // PII detection if sensitive data included
        if (includesSensitiveData) {
            val piiDetected = detectPII(prompt)
            if (piiDetected.isNotEmpty()) {
                warnings.add("Potential PII detected: ${piiDetected.joinToString()}")
                score -= 0.2f
            }
        }

        // Structure validation
        if (!hasValidStructure(prompt)) {
            warnings.add("Prompt lacks clear structure")
            score -= 0.1f
        }

        // Language validation
        if (!isValidLanguage(prompt)) {
            warnings.add("Prompt may contain inappropriate language")
            score -= 0.1f
        }

        return PromptValidationResult(
            isValid = issues.isEmpty(),
            validationScore = score.coerceIn(0f, 1f),
            errors = issues,
            warnings = warnings,
            isSafe = !containsMaliciousContent(prompt) &&
                    (detectPII(prompt).isEmpty() || !includesSensitiveData)
        )
    }

    /**
     * Validate batch data for AI processing
     */
    fun validateBatchData(sessions: List<SleepSession>): BatchValidationResult {
        if (sessions.isEmpty()) {
            return BatchValidationResult(
                isValid = false,
                overallScore = 0f,
                validSessions = emptyList(),
                invalidSessions = emptyList(),
                issues = listOf("No sessions provided"),
                recommendations = listOf("Provide at least one session for analysis")
            )
        }

        val validSessions = mutableListOf<SleepSession>()
        val invalidSessions = mutableListOf<Pair<SleepSession, String>>()
        val allIssues = mutableListOf<String>()
        var totalScore = 0f

        sessions.forEach { session ->
            val validation = validateSleepSession(session)
            if (validation.isValid) {
                validSessions.add(session)
                totalScore += validation.validationScore
            } else {
                invalidSessions.add(session to validation.validationMessages.firstOrNull().orEmpty())
                allIssues.addAll(validation.validationMessages)
            }
        }

        val overallScore = if (sessions.isNotEmpty()) totalScore / sessions.size else 0f
        val validPercentage = (validSessions.size.toFloat() / sessions.size * 100).toInt()

        val recommendations = mutableListOf<String>()
        if (validPercentage < 80) {
            recommendations.add("Consider reviewing data collection process - only $validPercentage% of sessions are valid")
        }
        if (invalidSessions.size > sessions.size * 0.2) {
            recommendations.add("High number of invalid sessions detected - check sensor calibration")
        }

        return BatchValidationResult(
            isValid = validSessions.isNotEmpty() && validPercentage >= 50,
            overallScore = overallScore,
            validSessions = validSessions,
            invalidSessions = invalidSessions,
            issues = allIssues.distinct(),
            recommendations = recommendations
        )
    }

    // ========== OUTPUT VALIDATION FROM AI RESPONSES ==========

    /**
     * Validate AI response content and format
     */
    fun validateAIResponse(
        response: String,
        expectedFormat: ResponseFormat = ResponseFormat.TEXT,
        model: String = "Unknown"
    ): AIResponseValidationResult {
        val issues = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        var score = 1.0f

        // Basic validation
        if (response.isBlank()) {
            issues.add("AI response is empty")
            return AIResponseValidationResult(
                isValid = false,
                validationScore = 0f,
                contentQuality = 0f,
                structureValid = false,
                safetyCheck = true,
                extractedData = emptyMap(),
                issues = issues,
                warnings = warnings
            )
        }

        if (response.length > MAX_RESPONSE_LENGTH) {
            warnings.add("Response exceeds maximum expected length")
            score -= 0.1f
        }

        // Format validation
        val formatValid = when (expectedFormat) {
            ResponseFormat.JSON -> validateJSONFormat(response)
            ResponseFormat.STRUCTURED_TEXT -> validateStructuredText(response)
            ResponseFormat.TEXT -> true // Always valid for plain text
        }

        if (!formatValid) {
            issues.add("Response format does not match expected format: $expectedFormat")
            score -= 0.3f
        }

        // Content quality assessment
        val contentQuality = assessContentQuality(response)
        if (contentQuality < 0.5f) {
            warnings.add("Response content quality is below acceptable threshold")
            score -= 0.2f
        }

        // Safety validation
        val safetyIssues = validateResponseSafety(response)
        if (safetyIssues.isNotEmpty()) {
            issues.addAll(safetyIssues)
            score -= 0.4f
        }

        // Extract structured data
        val extractedData = extractValidatedData(response, expectedFormat)

        // Validate extracted insights if present
        if (extractedData.containsKey("insights") && extractedData["insights"] is List<*>) {
            val insights = extractedData["insights"] as List<*>
            if (insights.isEmpty()) {
                warnings.add("No insights extracted from response")
                score -= 0.1f
            }
        }

        return AIResponseValidationResult(
            isValid = issues.isEmpty(),
            validationScore = score.coerceIn(0f, 1f),
            contentQuality = contentQuality,
            structureValid = formatValid,
            safetyCheck = safetyIssues.isEmpty(),
            extractedData = extractedData,
            issues = issues,
            warnings = warnings
        )
    }

    /**
     * Validate processed insights from AI
     */
    fun validateProcessedInsights(insights: List<ProcessedInsight>): InsightValidationResult {
        if (insights.isEmpty()) {
            return InsightValidationResult(
                isValid = false,
                overallQuality = 0f,
                validInsights = emptyList(),
                invalidInsights = emptyList(),
                qualityDistribution = emptyMap(),
                issues = listOf("No insights provided")
            )
        }

        val validInsights = mutableListOf<ProcessedInsight>()
        val invalidInsights = mutableListOf<Pair<ProcessedInsight, String>>()
        var totalQuality = 0f
        val qualityBuckets = mutableMapOf<String, Int>()

        insights.forEach { insight ->
            val validation = validateSingleInsight(insight)

            if (validation.isValid) {
                validInsights.add(insight)
                totalQuality += insight.qualityScore

                // Categorize quality
                val qualityCategory = when {
                    insight.qualityScore >= 0.8f -> "High"
                    insight.qualityScore >= 0.6f -> "Medium"
                    else -> "Low"
                }
                qualityBuckets[qualityCategory] = qualityBuckets.getOrDefault(qualityCategory, 0) + 1
            } else {
                invalidInsights.add(insight to validation.validationMessages.firstOrNull().orEmpty())
            }
        }

        val overallQuality = if (validInsights.isNotEmpty()) totalQuality / validInsights.size else 0f
        val issues = mutableListOf<String>()

        // Quality checks
        if (overallQuality < 0.5f) {
            issues.add("Overall insight quality is below acceptable threshold")
        }

        if (validInsights.size < insights.size * 0.7) {
            issues.add("High number of invalid insights (${insights.size - validInsights.size}/${insights.size})")
        }

        return InsightValidationResult(
            isValid = validInsights.isNotEmpty() && overallQuality >= 0.5f,
            overallQuality = overallQuality,
            validInsights = validInsights,
            invalidInsights = invalidInsights,
            qualityDistribution = qualityBuckets,
            issues = issues
        )
    }

    // ========== DATA INTEGRITY CHECKS ==========

    /**
     * Comprehensive data integrity validation
     */
    fun validateDataIntegrity(
        sessions: List<SleepSession>,
        insights: List<ProcessedInsight> = emptyList(),
        analytics: List<SleepSessionAnalytics> = emptyList()
    ): DataIntegrityResult {
        val issues = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val checks = mutableMapOf<String, Boolean>()

        // Temporal consistency
        val temporalCheck = validateTemporalConsistency(sessions)
        checks["temporal_consistency"] = temporalCheck.isValid
        if (!temporalCheck.isValid) {
            issues.addAll(temporalCheck.validationMessages)
        }

        // Data completeness
        val completenessScore = calculateDataCompleteness(sessions)
        checks["data_completeness"] = completenessScore >= 0.8f
        if (completenessScore < 0.8f) {
            warnings.add("Data completeness is ${(completenessScore * 100).toInt()}% (below 80% threshold)")
        }

        // Statistical consistency
        val statisticalCheck = validateStatisticalConsistency(sessions)
        checks["statistical_consistency"] = statisticalCheck
        if (!statisticalCheck) {
            warnings.add("Statistical anomalies detected in dataset")
        }

        // Cross-reference validation
        if (analytics.isNotEmpty()) {
            val crossRefCheck = validateCrossReferences(sessions, analytics)
            checks["cross_reference"] = crossRefCheck
            if (!crossRefCheck) {
                issues.add("Mismatch between session data and analytics")
            }
        }

        // Insight consistency
        if (insights.isNotEmpty()) {
            val insightCheck = validateInsightConsistency(sessions, insights)
            checks["insight_consistency"] = insightCheck
            if (!insightCheck) {
                warnings.add("Insights may be inconsistent with session data")
            }
        }

        val overallIntegrity = checks.values.count { it }.toFloat() / checks.size

        return DataIntegrityResult(
            overallIntegrity = overallIntegrity,
            integrityChecks = checks,
            dataCompleteness = completenessScore,
            temporalConsistency = temporalCheck.isValid,
            statisticalConsistency = statisticalCheck,
            issues = issues,
            warnings = warnings,
            recommendations = generateIntegrityRecommendations(checks, issues, warnings)
        )
    }

    /**
     * Validate sensor data consistency
     */
    fun validateSensorData(
        movementData: List<Float>,
        noiseData: List<Float>,
        heartRateData: List<Float> = emptyList(),
        temperatureData: List<Float> = emptyList()
    ): SensorValidationResult {
        val issues = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val sensorChecks = mutableMapOf<String, Float>()

        // Movement data validation
        if (movementData.isNotEmpty()) {
            val movementQuality = validateMovementSensorData(movementData)
            sensorChecks["movement"] = movementQuality
            if (movementQuality < 0.7f) {
                warnings.add("Movement sensor data quality is low")
            }
        } else {
            issues.add("No movement data available")
            sensorChecks["movement"] = 0f
        }

        // Noise data validation
        if (noiseData.isNotEmpty()) {
            val noiseQuality = validateNoiseSensorData(noiseData)
            sensorChecks["noise"] = noiseQuality
            if (noiseQuality < 0.7f) {
                warnings.add("Noise sensor data quality is low")
            }
        } else {
            issues.add("No noise data available")
            sensorChecks["noise"] = 0f
        }

        // Heart rate data validation (optional)
        if (heartRateData.isNotEmpty()) {
            val hrQuality = validateHeartRateData(heartRateData)
            sensorChecks["heart_rate"] = hrQuality
            if (hrQuality < 0.7f) {
                warnings.add("Heart rate data quality is low")
            }
        }

        // Temperature data validation (optional)
        if (temperatureData.isNotEmpty()) {
            val tempQuality = validateTemperatureData(temperatureData)
            sensorChecks["temperature"] = tempQuality
            if (tempQuality < 0.7f) {
                warnings.add("Temperature data quality is low")
            }
        }

        val overallQuality = if (sensorChecks.isNotEmpty()) {
            sensorChecks.values.average().toFloat()
        } else 0f

        return SensorValidationResult(
            overallQuality = overallQuality,
            sensorQuality = sensorChecks,
            dataPoints = movementData.size + noiseData.size + heartRateData.size + temperatureData.size,
            issues = issues,
            warnings = warnings,
            isReliable = overallQuality >= 0.7f && issues.isEmpty()
        )
    }

    // ========== INDIVIDUAL VALIDATION METHODS ==========

    private fun isValidTimestamp(timestamp: Long): Boolean {
        val now = System.currentTimeMillis()
        return timestamp in (now - ONE_YEAR_MS)..(now + ONE_DAY_MS)
    }

    private fun isValidDuration(duration: Long): Boolean {
        return duration in MIN_SLEEP_DURATION..MAX_SLEEP_DURATION
    }

    private fun isValidQualityScore(score: Float): Boolean {
        return score in MIN_SLEEP_QUALITY..MAX_SLEEP_QUALITY
    }

    private fun isValidEfficiency(efficiency: Float): Boolean {
        return efficiency in MIN_SLEEP_EFFICIENCY..MAX_SLEEP_EFFICIENCY
    }

    private fun isValidLatency(latency: Long): Boolean {
        return latency in MIN_SLEEP_LATENCY..MAX_SLEEP_LATENCY
    }

    private fun isValidMovementIntensity(intensity: Float): Boolean {
        return intensity in MIN_MOVEMENT_INTENSITY..MAX_MOVEMENT_INTENSITY
    }

    private fun isValidNoiseLevel(level: Float): Boolean {
        return level in MIN_NOISE_LEVEL..MAX_NOISE_LEVEL
    }

    private fun containsMaliciousContent(text: String): Boolean {
        return MALICIOUS_PATTERN.matcher(text).find()
    }

    private fun detectPII(text: String): List<String> {
        val piiDetected = mutableListOf<String>()

        // Email detection
        if (EMAIL_PATTERN.matcher(text).find()) {
            piiDetected.add("email")
        }

        // Phone number detection (simple pattern)
        if (text.contains(Regex("\\b\\d{3}[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b"))) {
            piiDetected.add("phone")
        }

        // SSN detection (simple pattern)
        if (text.contains(Regex("\\b\\d{3}[-.\\s]?\\d{2}[-.\\s]?\\d{4}\\b"))) {
            piiDetected.add("ssn")
        }

        return piiDetected
    }

    private fun hasValidStructure(prompt: String): Boolean {
        // Check for basic structure indicators
        val lines = prompt.lines()
        val hasHeaders = lines.any { it.contains(":") || it.uppercase() == it }
        val hasOrganization = lines.size > 3 && lines.count { it.trim().isNotEmpty() } > lines.size * 0.5

        return hasHeaders || hasOrganization
    }

    private fun isValidLanguage(text: String): Boolean {
        // Basic profanity and inappropriate content detection
        val inappropriateWords = listOf("hate", "kill", "die", "stupid", "idiot")
        val lowerText = text.lowercase()
        return inappropriateWords.none { lowerText.contains(it) }
    }

    private fun validateJSONFormat(response: String): Boolean {
        return try {
            JSONObject(response.trim())
            true
        } catch (e: JSONException) {
            false
        }
    }

    private fun validateStructuredText(response: String): Boolean {
        val lines = response.lines()
        val hasHeaders = lines.count { it.contains(":") || it.uppercase() == it } >= 2
        val hasContent = lines.count { it.trim().isNotEmpty() } >= 5
        return hasHeaders && hasContent
    }

    private fun assessContentQuality(response: String): Float {
        var score = 1.0f

        // Length assessment
        if (response.length < 100) score -= 0.3f
        if (response.length < 50) score -= 0.3f

        // Structure assessment
        val sentences = response.split(Regex("[.!?]+")).filter { it.trim().isNotEmpty() }
        if (sentences.size < 3) score -= 0.2f

        // Content diversity
        val words = response.split(Regex("\\s+")).map { it.lowercase() }.distinct()
        val uniqueWordRatio = words.size.toFloat() / response.split(Regex("\\s+")).size
        if (uniqueWordRatio < 0.5f) score -= 0.2f

        // Sleep-related content
        val sleepKeywords = listOf("sleep", "quality", "duration", "efficiency", "rest", "tired", "awake")
        val containsSleepContent = sleepKeywords.any { response.lowercase().contains(it) }
        if (!containsSleepContent) score -= 0.3f

        return score.coerceIn(0f, 1f)
    }

    private fun validateResponseSafety(response: String): List<String> {
        val safetyIssues = mutableListOf<String>()

        if (containsMaliciousContent(response)) {
            safetyIssues.add("Response contains potentially unsafe content")
        }

        if (detectPII(response).isNotEmpty()) {
            safetyIssues.add("Response may contain personal information")
        }

        // Check for medical advice
        val medicalKeywords = listOf("diagnose", "prescribe", "medication", "treatment", "cure")
        if (medicalKeywords.any { response.lowercase().contains(it) }) {
            safetyIssues.add("Response may contain medical advice")
        }

        return safetyIssues
    }

    private fun extractValidatedData(response: String, format: ResponseFormat): Map<String, Any> {
        val data = mutableMapOf<String, Any>()

        try {
            when (format) {
                ResponseFormat.JSON -> {
                    val json = JSONObject(response)
                    // Extract and validate JSON data
                    if (json.has("quality")) {
                        val quality = json.getDouble("quality").toFloat()
                        if (isValidQualityScore(quality)) {
                            data["quality"] = quality
                        }
                    }
                    if (json.has("insights")) {
                        data["insights"] = json.getJSONArray("insights")
                    }
                }
                ResponseFormat.STRUCTURED_TEXT -> {
                    // Extract structured data from text
                    val numericValues = FormatUtils.extractNumericValues(response)
                    data.putAll(numericValues)

                    val recommendations = FormatUtils.extractRecommendations(response)
                    if (recommendations.isNotEmpty()) {
                        data["recommendations"] = recommendations
                    }
                }
                ResponseFormat.TEXT -> {
                    // Basic text analysis
                    data["word_count"] = response.split(Regex("\\s+")).size
                    data["sentence_count"] = response.split(Regex("[.!?]+")).size
                }
            }
        } catch (e: Exception) {
            // If extraction fails, return basic data
            data["raw_response"] = response
            data["extraction_error"] = e.message ?: "Unknown error"
        }

        return data
    }

    // Additional helper methods for comprehensive validation...
    private fun validateSingleInsight(insight: ProcessedInsight): ValidationResult {
        val errors = mutableListOf<String>()
        var score = 1.0f

        if (insight.title.isBlank()) {
            errors.add("Insight title cannot be empty")
            score -= 0.3f
        }

        if (insight.description.isBlank()) {
            errors.add("Insight description cannot be empty")
            score -= 0.3f
        }

        if (insight.confidence < 0f || insight.confidence > 1f) {
            errors.add("Confidence score outside valid range")
            score -= 0.2f
        }

        if (insight.qualityScore < 0f || insight.qualityScore > 1f) {
            errors.add("Quality score outside valid range")
            score -= 0.2f
        }

        return ValidationResult(
            isValid = errors.isEmpty(),
            validationScore = score.coerceIn(0f, 1f),
            validationChecks = mapOf("basic_validation" to errors.isEmpty()),
            validationMessages = errors
        )
    }

    private fun validateTemporalConsistency(sessions: List<SleepSession>): ValidationResult {
        val errors = mutableListOf<String>()

        // Check for overlapping sessions
        val sortedSessions = sessions.sortedBy { it.startTime }
        for (i in 0 until sortedSessions.size - 1) {
            val current = sortedSessions[i]
            val next = sortedSessions[i + 1]
            val currentEnd = current.endTime ?: (current.startTime + current.duration)

            if (next.startTime < currentEnd) {
                errors.add("Overlapping sessions detected")
                break
            }
        }

        return ValidationResult(
            isValid = errors.isEmpty(),
            validationScore = if (errors.isEmpty()) 1f else 0.5f,
            validationChecks = mapOf("no_overlaps" to errors.isEmpty()),
            validationMessages = errors
        )
    }

    private fun calculateDataCompleteness(sessions: List<SleepSession>): Float {
        if (sessions.isEmpty()) return 0f

        var totalFields = 0
        var completedFields = 0

        sessions.forEach { session ->
            totalFields += 10 // Expected number of key fields

            if (session.sleepQualityScore != null) completedFields++
            if (session.sleepEfficiency > 0) completedFields++
            if (session.duration > 0) completedFields++
            if (session.sleepLatency >= 0) completedFields++
            if (session.lightSleepDuration > 0) completedFields++
            if (session.deepSleepDuration > 0) completedFields++
            if (session.remSleepDuration > 0) completedFields++
            if (session.averageMovementIntensity >= 0) completedFields++
            if (session.averageNoiseLevel >= 0) completedFields++
            if (session.confidence > 0) completedFields++
        }

        return if (totalFields > 0) completedFields.toFloat() / totalFields else 0f
    }

    private fun validateStatisticalConsistency(sessions: List<SleepSession>): Boolean {
        if (sessions.size < 3) return true // Need minimum data for statistical analysis

        // Check for statistical outliers in key metrics
        val qualities = sessions.mapNotNull { it.sleepQualityScore }
        val durations = sessions.map { it.duration.toFloat() }
        val efficiencies = sessions.map { it.sleepEfficiency }

        // Simple outlier detection using IQR method
        fun hasOutliers(values: List<Float>): Boolean {
            if (values.size < 4) return false
            val sorted = values.sorted()
            val q1 = sorted[sorted.size / 4]
            val q3 = sorted[3 * sorted.size / 4]
            val iqr = q3 - q1
            val outlierThreshold = 1.5f * iqr

            return values.any { it < q1 - outlierThreshold || it > q3 + outlierThreshold }
        }

        val qualityOutliers = if (qualities.isNotEmpty()) hasOutliers(qualities) else false
        val durationOutliers = hasOutliers(durations)
        val efficiencyOutliers = hasOutliers(efficiencies)

        // Allow some outliers but flag if too many
        val outlierCount = listOf(qualityOutliers, durationOutliers, efficiencyOutliers).count { it }
        return outlierCount <= 1 // Allow at most one metric to have outliers
    }

    private fun validateCrossReferences(
        sessions: List<SleepSession>,
        analytics: List<SleepSessionAnalytics>
    ): Boolean {
        // Check if session IDs match
        val sessionIds = sessions.map { it.id }.toSet()
        val analyticsIds = analytics.map { it.sessionId }.toSet()

        return sessionIds == analyticsIds
    }

    private fun validateInsightConsistency(
        sessions: List<SleepSession>,
        insights: List<ProcessedInsight>
    ): Boolean {
        // Check if insights are consistent with session data
        val avgQuality = sessions.mapNotNull { it.sleepQualityScore }.average().toFloat()
        val avgEfficiency = sessions.map { it.sleepEfficiency }.average().toFloat()

        // Look for insights that contradict the data
        val qualityInsights = insights.filter { it.category == InsightCategory.QUALITY }
        val hasContradictoryInsights = qualityInsights.any { insight ->
            // Simple check: if average quality is high but insights suggest poor quality
            (avgQuality > 7f && insight.description.lowercase().contains("poor")) ||
                    (avgQuality < 4f && insight.description.lowercase().contains("excellent"))
        }

        return !hasContradictoryInsights
    }

    private fun validateMovementSensorData(data: List<Float>): Float {
        if (data.isEmpty()) return 0f

        var score = 1f

        // Check for valid range
        val invalidValues = data.count { !isValidMovementIntensity(it) }
        if (invalidValues > 0) {
            score -= (invalidValues.toFloat() / data.size) * 0.5f
        }

        // Check for data consistency (not all zeros or constant values)
        val uniqueValues = data.distinct().size
        if (uniqueValues == 1) score -= 0.3f

        // Check for reasonable variance
        if (data.size > 1) {
            val mean = data.average()
            val variance = data.map { (it - mean).pow(2) }.average()
            if (variance < 0.1) score -= 0.2f // Too little variance suggests sensor issues
        }

        return score.coerceIn(0f, 1f)
    }

    private fun validateNoiseSensorData(data: List<Float>): Float {
        if (data.isEmpty()) return 0f

        var score = 1f

        // Check for valid range
        val invalidValues = data.count { !isValidNoiseLevel(it) }
        if (invalidValues > 0) {
            score -= (invalidValues.toFloat() / data.size) * 0.5f
        }

        // Check for realistic noise patterns
        val mean = data.average()
        if (mean < 10f || mean > 80f) score -= 0.2f // Unrealistic average noise levels

        return score.coerceIn(0f, 1f)
    }

    private fun validateHeartRateData(data: List<Float>): Float {
        if (data.isEmpty()) return 0f

        var score = 1f

        // Check for valid range (resting HR typically 40-100 bpm)
        val invalidValues = data.count { it < 30f || it > 150f }
        if (invalidValues > 0) {
            score -= (invalidValues.toFloat() / data.size) * 0.5f
        }

        // Check for realistic variance
        val mean = data.average()
        val variance = data.map { (it - mean).pow(2) }.average()
        if (variance > 400f) score -= 0.3f // Too much variance suggests measurement errors

        return score.coerceIn(0f, 1f)
    }

    private fun validateTemperatureData(data: List<Float>): Float {
        if (data.isEmpty()) return 0f

        var score = 1f

        // Check for valid range
        val invalidValues = data.count { it < MIN_TEMPERATURE || it > MAX_TEMPERATURE }
        if (invalidValues > 0) {
            score -= (invalidValues.toFloat() / data.size) * 0.5f
        }

        return score.coerceIn(0f, 1f)
    }

    private fun generateIntegrityRecommendations(
        checks: Map<String, Boolean>,
        issues: List<String>,
        warnings: List<String>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (!checks.getOrDefault("temporal_consistency", true)) {
            recommendations.add("Review data collection timing to prevent overlapping sessions")
        }

        if (!checks.getOrDefault("data_completeness", true)) {
            recommendations.add("Improve sensor coverage to increase data completeness")
        }

        if (!checks.getOrDefault("statistical_consistency", true)) {
            recommendations.add("Investigate potential sensor calibration issues")
        }

        if (issues.isNotEmpty()) {
            recommendations.add("Address critical data integrity issues before proceeding with analysis")
        }

        if (warnings.size > 3) {
            recommendations.add("Consider data quality improvements to reduce validation warnings")
        }

        return recommendations
    }
}

// ========== VALIDATION RESULT DATA CLASSES ==========

data class PromptValidationResult(
    val isValid: Boolean,
    val validationScore: Float,
    val errors: List<String>,
    val warnings: List<String>,
    val isSafe: Boolean
)

data class BatchValidationResult(
    val isValid: Boolean,
    val overallScore: Float,
    val validSessions: List<SleepSession>,
    val invalidSessions: List<Pair<SleepSession, String>>,
    val issues: List<String>,
    val recommendations: List<String>
)

data class AIResponseValidationResult(
    val isValid: Boolean,
    val validationScore: Float,
    val contentQuality: Float,
    val structureValid: Boolean,
    val safetyCheck: Boolean,
    val extractedData: Map<String, Any>,
    val issues: List<String>,
    val warnings: List<String>
)

data class InsightValidationResult(
    val isValid: Boolean,
    val overallQuality: Float,
    val validInsights: List<ProcessedInsight>,
    val invalidInsights: List<Pair<ProcessedInsight, String>>,
    val qualityDistribution: Map<String, Int>,
    val issues: List<String>
)

data class DataIntegrityResult(
    val overallIntegrity: Float,
    val integrityChecks: Map<String, Boolean>,
    val dataCompleteness: Float,
    val temporalConsistency: Boolean,
    val statisticalConsistency: Boolean,
    val issues: List<String>,
    val warnings: List<String>,
    val recommendations: List<String>
)

data class SensorValidationResult(
    val overallQuality: Float,
    val sensorQuality: Map<String, Float>,
    val dataPoints: Int,
    val issues: List<String>,
    val warnings: List<String>,
    val isReliable: Boolean
)

// ========== ENUMS ==========

enum class ResponseFormat {
    JSON,
    STRUCTURED_TEXT,
    TEXT
}