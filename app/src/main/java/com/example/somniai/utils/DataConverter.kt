package com.example.somniai.utils

import com.example.somniai.data.*
import com.example.somniai.ai.*
import com.example.somniai.utils.TimeUtils
import org.json.JSONObject
import org.json.JSONArray
import org.json.JSONException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

/**
 * Data Conversion Utilities for SomniAI
 *
 * Comprehensive data conversion system that bridges raw sensor data,
 * AI processing formats, and app data models. Ensures type safety,
 * data validation, and consistent formatting across the application.
 */
object DataConverter {

    // ========== AI FORMAT CONVERSION ==========

    /**
     * Convert sleep session data to AI-readable prompt format
     * Creates structured text suitable for AI analysis
     */
    fun sleepSessionToAIPrompt(session: SleepSession, includeContext: Boolean = true): String {
        val prompt = StringBuilder()

        // Header with session overview
        prompt.appendLine("=== SLEEP SESSION ANALYSIS REQUEST ===")
        prompt.appendLine("Session ID: ${session.id}")
        prompt.appendLine("Date: ${TimeUtils.formatDate(session.startTime)}")
        prompt.appendLine("Duration: ${TimeUtils.formatDuration(session.duration)}")
        prompt.appendLine()

        // Core metrics
        prompt.appendLine("CORE METRICS:")
        prompt.appendLine("- Sleep Quality Score: ${session.sleepQualityScore ?: "Not calculated"}")
        prompt.appendLine("- Sleep Efficiency: ${session.sleepEfficiency}%")
        prompt.appendLine("- Sleep Latency: ${TimeUtils.formatDuration(session.sleepLatency)}")
        prompt.appendLine("- Total Sleep Time: ${TimeUtils.formatDuration(session.actualSleepDuration)}")
        prompt.appendLine()

        // Sleep phases breakdown
        prompt.appendLine("SLEEP ARCHITECTURE:")
        val totalSleep = session.lightSleepDuration + session.deepSleepDuration +
                session.remSleepDuration + session.awakeDuration
        if (totalSleep > 0) {
            val lightPercent = (session.lightSleepDuration.toFloat() / totalSleep * 100).toInt()
            val deepPercent = (session.deepSleepDuration.toFloat() / totalSleep * 100).toInt()
            val remPercent = (session.remSleepDuration.toFloat() / totalSleep * 100).toInt()
            val awakePercent = (session.awakeDuration.toFloat() / totalSleep * 100).toInt()

            prompt.appendLine("- Light Sleep: ${lightPercent}% (${TimeUtils.formatDuration(session.lightSleepDuration)})")
            prompt.appendLine("- Deep Sleep: ${deepPercent}% (${TimeUtils.formatDuration(session.deepSleepDuration)})")
            prompt.appendLine("- REM Sleep: ${remPercent}% (${TimeUtils.formatDuration(session.remSleepDuration)})")
            prompt.appendLine("- Awake Time: ${awakePercent}% (${TimeUtils.formatDuration(session.awakeDuration)})")
        }
        prompt.appendLine()

        // Movement and noise data
        prompt.appendLine("ENVIRONMENTAL FACTORS:")
        prompt.appendLine("- Average Movement Intensity: ${session.averageMovementIntensity}")
        prompt.appendLine("- Movement Events: ${session.movementEvents.size}")
        prompt.appendLine("- Average Noise Level: ${session.averageNoiseLevel} dB")
        prompt.appendLine("- Noise Events: ${session.noiseEvents.size}")
        if (session.noiseEvents.isNotEmpty()) {
            val maxNoise = session.noiseEvents.maxOf { it.decibelLevel }
            prompt.appendLine("- Peak Noise Level: ${maxNoise} dB")
        }
        prompt.appendLine()

        // Sleep timing
        prompt.appendLine("SLEEP TIMING:")
        prompt.appendLine("- Bedtime: ${TimeUtils.formatTime(session.startTime)}")
        session.endTime?.let {
            prompt.appendLine("- Wake Time: ${TimeUtils.formatTime(it)}")
        }
        prompt.appendLine("- Sleep Onset Time: ${TimeUtils.formatTime(session.startTime + session.sleepLatency)}")
        prompt.appendLine()

        // Context information if requested
        if (includeContext) {
            prompt.appendLine("ANALYSIS REQUEST:")
            prompt.appendLine("Please analyze this sleep session and provide:")
            prompt.appendLine("1. Overall sleep quality assessment")
            prompt.appendLine("2. Key factors affecting sleep quality")
            prompt.appendLine("3. Specific recommendations for improvement")
            prompt.appendLine("4. Any concerning patterns or anomalies")
            prompt.appendLine("5. Comparison to optimal sleep patterns")
            prompt.appendLine()
            prompt.appendLine("Focus on actionable insights and evidence-based recommendations.")
        }

        return prompt.toString()
    }

    /**
     * Convert multiple sleep sessions to comparative AI prompt
     */
    fun sleepSessionsToTrendPrompt(
        sessions: List<SleepSession>,
        analysisType: String = "trend"
    ): String {
        val prompt = StringBuilder()

        prompt.appendLine("=== SLEEP TREND ANALYSIS REQUEST ===")
        prompt.appendLine("Analysis Type: ${analysisType.uppercase()}")
        prompt.appendLine("Sessions Count: ${sessions.size}")
        prompt.appendLine("Date Range: ${getDateRange(sessions)}")
        prompt.appendLine()

        // Summary statistics
        val avgQuality = sessions.mapNotNull { it.sleepQualityScore }.average().toFloat()
        val avgDuration = sessions.map { it.duration }.average().toLong()
        val avgEfficiency = sessions.map { it.sleepEfficiency }.average().toFloat()

        prompt.appendLine("TREND SUMMARY:")
        prompt.appendLine("- Average Quality: ${String.format("%.1f", avgQuality)}/10")
        prompt.appendLine("- Average Duration: ${TimeUtils.formatDuration(avgDuration)}")
        prompt.appendLine("- Average Efficiency: ${String.format("%.1f", avgEfficiency)}%")
        prompt.appendLine()

        // Individual session data (condensed)
        prompt.appendLine("SESSION DATA:")
        sessions.take(10).forEachIndexed { index, session ->
            prompt.appendLine("${index + 1}. ${TimeUtils.formatDate(session.startTime)}: " +
                    "Q=${session.sleepQualityScore ?: "N/A"}, " +
                    "D=${TimeUtils.formatDurationShort(session.duration)}, " +
                    "E=${session.sleepEfficiency}%")
        }
        if (sessions.size > 10) {
            prompt.appendLine("... (${sessions.size - 10} more sessions)")
        }
        prompt.appendLine()

        prompt.appendLine("ANALYSIS REQUEST:")
        prompt.appendLine("Please analyze these sleep sessions and provide:")
        prompt.appendLine("1. Overall trend direction and strength")
        prompt.appendLine("2. Key patterns and cycles identified")
        prompt.appendLine("3. Factors contributing to improvements/declines")
        prompt.appendLine("4. Recommendations for sustained improvement")
        prompt.appendLine("5. Any concerning trends requiring attention")

        return prompt.toString()
    }

    /**
     * Convert sleep data to structured JSON for AI APIs
     */
    fun sleepSessionToAIJson(session: SleepSession): JSONObject {
        return JSONObject().apply {
            put("sessionId", session.id)
            put("timestamp", session.startTime)
            put("duration", session.duration)
            put("sleepQuality", session.sleepQualityScore ?: JSONObject.NULL)
            put("sleepEfficiency", session.sleepEfficiency)
            put("sleepLatency", session.sleepLatency)

            put("sleepPhases", JSONObject().apply {
                put("lightSleep", session.lightSleepDuration)
                put("deepSleep", session.deepSleepDuration)
                put("remSleep", session.remSleepDuration)
                put("awakeTime", session.awakeDuration)
            })

            put("environmental", JSONObject().apply {
                put("avgMovement", session.averageMovementIntensity)
                put("movementEvents", session.movementEvents.size)
                put("avgNoise", session.averageNoiseLevel)
                put("noiseEvents", session.noiseEvents.size)
            })

            put("timing", JSONObject().apply {
                put("bedtime", session.startTime)
                put("wakeTime", session.endTime ?: JSONObject.NULL)
                put("sleepOnset", session.startTime + session.sleepLatency)
            })
        }
    }

    // ========== AI RESPONSE PARSING ==========

    /**
     * Parse AI response text into structured sleep insights
     */
    fun parseAIResponseToInsights(
        response: String,
        sessionId: Long,
        aiModel: String = "AI Generated"
    ): List<ProcessedInsight> {
        val insights = mutableListOf<ProcessedInsight>()

        try {
            // Try to parse as JSON first
            if (response.trim().startsWith("{") || response.trim().startsWith("[")) {
                return parseJsonResponse(response, sessionId, aiModel)
            }

            // Parse structured text response
            val sections = parseStructuredTextResponse(response)

            sections.forEach { section ->
                val insight = createInsightFromSection(section, sessionId, aiModel)
                if (insight != null) {
                    insights.add(insight)
                }
            }

            // If no structured sections found, create general insight
            if (insights.isEmpty() && response.isNotBlank()) {
                insights.add(createGeneralInsight(response, sessionId, aiModel))
            }

        } catch (e: Exception) {
            // Fallback: create single insight with raw response
            insights.add(createFallbackInsight(response, sessionId, aiModel, e.message))
        }

        return insights
    }

    /**
     * Parse JSON AI response format
     */
    private fun parseJsonResponse(
        jsonResponse: String,
        sessionId: Long,
        aiModel: String
    ): List<ProcessedInsight> {
        val insights = mutableListOf<ProcessedInsight>()

        try {
            val jsonObject = JSONObject(jsonResponse)

            // Handle array of insights
            if (jsonObject.has("insights")) {
                val insightsArray = jsonObject.getJSONArray("insights")
                for (i in 0 until insightsArray.length()) {
                    val insightJson = insightsArray.getJSONObject(i)
                    insights.add(parseJsonInsight(insightJson, sessionId, aiModel))
                }
            } else {
                // Handle single insight object
                insights.add(parseJsonInsight(jsonObject, sessionId, aiModel))
            }

        } catch (e: JSONException) {
            // If JSON parsing fails, treat as text
            return parseStructuredTextToInsights(jsonResponse, sessionId, aiModel)
        }

        return insights
    }

    private fun parseJsonInsight(
        json: JSONObject,
        sessionId: Long,
        aiModel: String
    ): ProcessedInsight {
        val category = json.optString("category", "general").let { categoryStr ->
            when (categoryStr.lowercase()) {
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
                else -> InsightCategory.QUALITY
            }
        }

        return ProcessedInsight(
            id = UUID.randomUUID().toString(),
            originalInsight = SleepInsight(
                id = UUID.randomUUID().toString(),
                sessionId = sessionId,
                category = category,
                title = json.optString("title", "Sleep Insight"),
                description = json.optString("description", ""),
                recommendation = json.optString("recommendation", ""),
                priority = json.optInt("priority", 2),
                timestamp = System.currentTimeMillis(),
                isAiGenerated = true,
                confidence = json.optDouble("confidence", 0.7).toFloat()
            ),
            category = category,
            priority = json.optInt("priority", 2),
            title = json.optString("title", "Sleep Insight"),
            description = json.optString("description", ""),
            recommendation = json.optString("recommendation", ""),
            evidence = emptyList(),
            dataPoints = emptyList(),
            confidence = json.optDouble("confidence", 0.7).toFloat(),
            qualityScore = json.optDouble("quality", 0.7).toFloat(),
            relevanceScore = json.optDouble("relevance", 0.8).toFloat(),
            actionabilityScore = json.optDouble("actionability", 0.6).toFloat(),
            noveltyScore = json.optDouble("novelty", 0.5).toFloat(),
            personalizationScore = json.optDouble("personalization", 0.3).toFloat(),
            validationResults = ValidationResult(
                isValid = true,
                validationScore = 0.8f,
                validationChecks = mapOf("parsed_successfully" to true),
                validationMessages = emptyList()
            ),
            aiGenerated = true,
            aiModelUsed = aiModel,
            processingMetadata = ProcessingMetadata(
                processingVersion = "1.0",
                processingTimeMs = 0L,
                algorithmUsed = "AI Response Parser",
                parametersUsed = emptyMap(),
                dataSourcesCount = 1,
                validationSteps = listOf("JSON parsing", "field validation")
            ),
            implementationDifficulty = ImplementationDifficulty.MEDIUM,
            expectedImpact = ExpectedImpact.MEDIUM,
            timeToImpact = TimeToImpact.SHORT_TERM,
            timestamp = System.currentTimeMillis()
        )
    }

    /**
     * Parse structured text response into sections
     */
    private fun parseStructuredTextResponse(response: String): List<ResponseSection> {
        val sections = mutableListOf<ResponseSection>()
        val lines = response.lines()
        var currentSection: ResponseSection? = null

        for (line in lines) {
            val trimmedLine = line.trim()

            // Detect section headers
            when {
                isQualitySection(trimmedLine) -> {
                    currentSection?.let { sections.add(it) }
                    currentSection = ResponseSection("quality", mutableListOf())
                }
                isRecommendationSection(trimmedLine) -> {
                    currentSection?.let { sections.add(it) }
                    currentSection = ResponseSection("recommendation", mutableListOf())
                }
                isFactorSection(trimmedLine) -> {
                    currentSection?.let { sections.add(it) }
                    currentSection = ResponseSection("factors", mutableListOf())
                }
                isPatternSection(trimmedLine) -> {
                    currentSection?.let { sections.add(it) }
                    currentSection = ResponseSection("patterns", mutableListOf())
                }
                trimmedLine.isNotEmpty() -> {
                    if (currentSection == null) {
                        currentSection = ResponseSection("general", mutableListOf())
                    }
                    currentSection.content.add(trimmedLine)
                }
            }
        }

        currentSection?.let { sections.add(it) }
        return sections
    }

    private fun createInsightFromSection(
        section: ResponseSection,
        sessionId: Long,
        aiModel: String
    ): ProcessedInsight? {
        if (section.content.isEmpty()) return null

        val category = when (section.type) {
            "quality" -> InsightCategory.QUALITY
            "recommendation" -> InsightCategory.OPTIMIZATION
            "factors" -> InsightCategory.ENVIRONMENT
            "patterns" -> InsightCategory.TRENDS
            else -> InsightCategory.QUALITY
        }

        val title = generateTitleFromSection(section)
        val description = section.content.take(3).joinToString(" ")
        val recommendation = if (section.type == "recommendation") {
            section.content.joinToString(". ")
        } else ""

        return ProcessedInsight(
            id = UUID.randomUUID().toString(),
            originalInsight = SleepInsight(
                id = UUID.randomUUID().toString(),
                sessionId = sessionId,
                category = category,
                title = title,
                description = description,
                recommendation = recommendation,
                priority = determinePriority(section),
                timestamp = System.currentTimeMillis(),
                isAiGenerated = true,
                confidence = 0.7f
            ),
            category = category,
            priority = determinePriority(section),
            title = title,
            description = description,
            recommendation = recommendation,
            evidence = emptyList(),
            dataPoints = emptyList(),
            confidence = 0.7f,
            qualityScore = 0.75f,
            relevanceScore = 0.8f,
            actionabilityScore = if (recommendation.isNotEmpty()) 0.8f else 0.4f,
            noveltyScore = 0.5f,
            personalizationScore = 0.4f,
            validationResults = ValidationResult(
                isValid = true,
                validationScore = 0.7f,
                validationChecks = mapOf("section_parsed" to true),
                validationMessages = emptyList()
            ),
            aiGenerated = true,
            aiModelUsed = aiModel,
            processingMetadata = ProcessingMetadata(
                processingVersion = "1.0",
                processingTimeMs = 0L,
                algorithmUsed = "Section Parser",
                parametersUsed = emptyMap(),
                dataSourcesCount = 1,
                validationSteps = listOf("section detection", "content extraction")
            ),
            implementationDifficulty = ImplementationDifficulty.MEDIUM,
            expectedImpact = ExpectedImpact.MEDIUM,
            timeToImpact = TimeToImpact.SHORT_TERM,
            timestamp = System.currentTimeMillis()
        )
    }

    // ========== DATA MODEL TRANSFORMATIONS ==========

    /**
     * Convert raw sensor data to SleepSession
     */
    fun rawSensorDataToSleepSession(rawData: RawSensorData): SleepSession {
        return SleepSession(
            id = rawData.sessionId,
            startTime = rawData.startTimestamp,
            endTime = rawData.endTimestamp,
            duration = rawData.endTimestamp - rawData.startTimestamp,
            sleepQualityScore = null, // Will be calculated later
            sleepEfficiency = calculateEfficiencyFromRawData(rawData),
            sleepLatency = rawData.sleepLatency,
            actualSleepDuration = rawData.totalSleepTime,
            lightSleepDuration = rawData.lightSleepDuration,
            deepSleepDuration = rawData.deepSleepDuration,
            remSleepDuration = rawData.remSleepDuration,
            awakeDuration = rawData.awakeDuration,
            averageMovementIntensity = rawData.movementData.average().toFloat(),
            movementEvents = convertMovementEvents(rawData.movementEvents),
            averageNoiseLevel = rawData.noiseData.average().toFloat(),
            noiseEvents = convertNoiseEvents(rawData.noiseEvents),
            phaseTransitions = convertPhaseTransitions(rawData.phaseTransitions),
            confidence = validateRawDataQuality(rawData)
        )
    }

    /**
     * Convert ProcessedInsight back to simple SleepInsight
     */
    fun processedInsightToSleepInsight(processed: ProcessedInsight): SleepInsight {
        return SleepInsight(
            id = processed.id,
            sessionId = processed.originalInsight.sessionId,
            category = processed.category,
            title = processed.title,
            description = processed.description,
            recommendation = processed.recommendation,
            priority = processed.priority,
            timestamp = processed.timestamp,
            isAiGenerated = processed.aiGenerated,
            isAcknowledged = false,
            confidence = processed.confidence
        )
    }

    /**
     * Convert SleepSession to analytics format
     */
    fun sleepSessionToAnalytics(session: SleepSession): SleepSessionAnalytics {
        return session.toAnalytics() // Use extension function from DataExtensions.kt
    }

    // ========== DATA NORMALIZATION ==========

    /**
     * Normalize sleep quality score to 0-10 scale
     */
    fun normalizeQualityScore(score: Float, sourceRange: ClosedFloatingPointRange<Float> = 0f..100f): Float {
        val normalized = ((score - sourceRange.start) / (sourceRange.endInclusive - sourceRange.start)) * 10f
        return normalized.coerceIn(0f, 10f)
    }

    /**
     * Normalize duration to standardized format (milliseconds)
     */
    fun normalizeDuration(
        value: Number,
        sourceUnit: TimeUnit = TimeUnit.MILLISECONDS
    ): Long {
        return when (sourceUnit) {
            TimeUnit.SECONDS -> value.toLong() * 1000L
            TimeUnit.MINUTES -> value.toLong() * 60 * 1000L
            TimeUnit.HOURS -> value.toLong() * 60 * 60 * 1000L
            else -> value.toLong()
        }
    }

    /**
     * Normalize noise levels to decibel scale
     */
    fun normalizeNoiseLevel(
        value: Float,
        sourceRange: ClosedFloatingPointRange<Float> = 0f..100f
    ): Float {
        // Convert arbitrary 0-100 scale to approximate decibel range (30-90 dB)
        val dbRange = 30f..90f
        val normalized = ((value - sourceRange.start) / (sourceRange.endInclusive - sourceRange.start))
        return dbRange.start + (normalized * (dbRange.endInclusive - dbRange.start))
    }

    /**
     * Normalize movement intensity to 0-10 scale
     */
    fun normalizeMovementIntensity(
        value: Float,
        sourceRange: ClosedFloatingPointRange<Float> = 0f..1000f
    ): Float {
        val normalized = ((value - sourceRange.start) / (sourceRange.endInclusive - sourceRange.start)) * 10f
        return normalized.coerceIn(0f, 10f)
    }

    /**
     * Validate and sanitize data
     */
    fun validateAndSanitizeData(data: Map<String, Any?>): Map<String, Any> {
        val sanitized = mutableMapOf<String, Any>()

        data.forEach { (key, value) ->
            when (key) {
                "duration" -> {
                    val duration = (value as? Number)?.toLong() ?: 0L
                    sanitized[key] = duration.coerceIn(0L, 24 * 60 * 60 * 1000L) // Max 24 hours
                }
                "efficiency" -> {
                    val efficiency = (value as? Number)?.toFloat() ?: 0f
                    sanitized[key] = efficiency.coerceIn(0f, 100f)
                }
                "quality" -> {
                    val quality = (value as? Number)?.toFloat() ?: 0f
                    sanitized[key] = quality.coerceIn(0f, 10f)
                }
                "noise" -> {
                    val noise = (value as? Number)?.toFloat() ?: 0f
                    sanitized[key] = noise.coerceIn(0f, 120f) // Max 120 dB
                }
                "movement" -> {
                    val movement = (value as? Number)?.toFloat() ?: 0f
                    sanitized[key] = movement.coerceIn(0f, 100f)
                }
                "timestamp" -> {
                    val timestamp = (value as? Number)?.toLong() ?: System.currentTimeMillis()
                    // Validate timestamp is reasonable (not in future, not too old)
                    val now = System.currentTimeMillis()
                    val oneYearAgo = now - (365L * 24 * 60 * 60 * 1000L)
                    sanitized[key] = timestamp.coerceIn(oneYearAgo, now)
                }
                else -> {
                    value?.let { sanitized[key] = it }
                }
            }
        }

        return sanitized
    }

    // ========== UTILITY METHODS ==========

    private fun getDateRange(sessions: List<SleepSession>): String {
        if (sessions.isEmpty()) return "No sessions"
        val earliest = sessions.minOf { it.startTime }
        val latest = sessions.maxOf { it.startTime }
        return "${TimeUtils.formatDate(earliest)} - ${TimeUtils.formatDate(latest)}"
    }

    private fun parseStructuredTextToInsights(
        text: String,
        sessionId: Long,
        aiModel: String
    ): List<ProcessedInsight> {
        return listOf(createGeneralInsight(text, sessionId, aiModel))
    }

    private fun createGeneralInsight(
        content: String,
        sessionId: Long,
        aiModel: String
    ): ProcessedInsight {
        val title = extractTitle(content)
        val description = content.take(200)

        return ProcessedInsight(
            id = UUID.randomUUID().toString(),
            originalInsight = SleepInsight(
                id = UUID.randomUUID().toString(),
                sessionId = sessionId,
                category = InsightCategory.QUALITY,
                title = title,
                description = description,
                recommendation = "",
                priority = 2,
                timestamp = System.currentTimeMillis(),
                isAiGenerated = true,
                confidence = 0.6f
            ),
            category = InsightCategory.QUALITY,
            priority = 2,
            title = title,
            description = description,
            recommendation = "",
            evidence = emptyList(),
            dataPoints = emptyList(),
            confidence = 0.6f,
            qualityScore = 0.6f,
            relevanceScore = 0.7f,
            actionabilityScore = 0.3f,
            noveltyScore = 0.5f,
            personalizationScore = 0.2f,
            validationResults = ValidationResult(
                isValid = true,
                validationScore = 0.6f,
                validationChecks = mapOf("general_parsing" to true),
                validationMessages = emptyList()
            ),
            aiGenerated = true,
            aiModelUsed = aiModel,
            processingMetadata = ProcessingMetadata(
                processingVersion = "1.0",
                processingTimeMs = 0L,
                algorithmUsed = "General Text Parser",
                parametersUsed = emptyMap(),
                dataSourcesCount = 1,
                validationSteps = listOf("text extraction")
            ),
            implementationDifficulty = ImplementationDifficulty.MEDIUM,
            expectedImpact = ExpectedImpact.MEDIUM,
            timeToImpact = TimeToImpact.SHORT_TERM,
            timestamp = System.currentTimeMillis()
        )
    }

    private fun createFallbackInsight(
        content: String,
        sessionId: Long,
        aiModel: String,
        errorMessage: String?
    ): ProcessedInsight {
        return createGeneralInsight("AI Response: $content", sessionId, aiModel)
    }

    // Helper methods for text parsing
    private fun isQualitySection(line: String): Boolean {
        val qualityKeywords = listOf("quality", "overall", "assessment", "rating", "score")
        return qualityKeywords.any { line.lowercase().contains(it) } && line.contains(":")
    }

    private fun isRecommendationSection(line: String): Boolean {
        val recKeywords = listOf("recommendation", "suggest", "improve", "advice", "consider")
        return recKeywords.any { line.lowercase().contains(it) } && line.contains(":")
    }

    private fun isFactorSection(line: String): Boolean {
        val factorKeywords = listOf("factor", "cause", "reason", "affect", "influence")
        return factorKeywords.any { line.lowercase().contains(it) } && line.contains(":")
    }

    private fun isPatternSection(line: String): Boolean {
        val patternKeywords = listOf("pattern", "trend", "cycle", "habit", "behavior")
        return patternKeywords.any { line.lowercase().contains(it) } && line.contains(":")
    }

    private fun generateTitleFromSection(section: ResponseSection): String {
        return when (section.type) {
            "quality" -> "Sleep Quality Analysis"
            "recommendation" -> "Sleep Improvement Recommendations"
            "factors" -> "Factors Affecting Sleep"
            "patterns" -> "Sleep Pattern Analysis"
            else -> "Sleep Insight"
        }
    }

    private fun determinePriority(section: ResponseSection): Int {
        return when (section.type) {
            "recommendation" -> 1 // High priority
            "quality" -> 2 // Medium priority
            "patterns" -> 2 // Medium priority
            "factors" -> 3 // Lower priority
            else -> 2
        }
    }

    private fun extractTitle(content: String): String {
        val firstLine = content.lines().firstOrNull { it.trim().isNotEmpty() }
        return firstLine?.take(50) ?: "Sleep Analysis"
    }

    // Raw data conversion helpers
    private fun calculateEfficiencyFromRawData(rawData: RawSensorData): Float {
        val totalTimeInBed = rawData.endTimestamp - rawData.startTimestamp
        return if (totalTimeInBed > 0) {
            (rawData.totalSleepTime.toFloat() / totalTimeInBed * 100f).coerceIn(0f, 100f)
        } else 0f
    }

    private fun convertMovementEvents(rawEvents: List<RawMovementEvent>): List<MovementEvent> {
        return rawEvents.map { raw ->
            MovementEvent(
                timestamp = raw.timestamp,
                intensity = raw.intensity,
                duration = raw.duration,
                type = raw.type
            )
        }
    }

    private fun convertNoiseEvents(rawEvents: List<RawNoiseEvent>): List<NoiseEvent> {
        return rawEvents.map { raw ->
            NoiseEvent(
                timestamp = raw.timestamp,
                decibelLevel = raw.decibelLevel,
                duration = raw.duration,
                frequency = raw.frequency
            )
        }
    }

    private fun convertPhaseTransitions(rawTransitions: List<RawPhaseTransition>): List<PhaseTransition> {
        return rawTransitions.map { raw ->
            PhaseTransition(
                timestamp = raw.timestamp,
                fromPhase = raw.fromPhase,
                toPhase = raw.toPhase,
                confidence = raw.confidence
            )
        }
    }

    private fun validateRawDataQuality(rawData: RawSensorData): Float {
        var score = 1.0f

        // Penalize for missing data
        if (rawData.totalSleepTime <= 0) score -= 0.3f
        if (rawData.movementData.isEmpty()) score -= 0.2f
        if (rawData.noiseData.isEmpty()) score -= 0.2f

        // Penalize for unrealistic values
        val duration = rawData.endTimestamp - rawData.startTimestamp
        if (duration > 16 * 60 * 60 * 1000L) score -= 0.2f // > 16 hours
        if (duration < 2 * 60 * 60 * 1000L) score -= 0.2f // < 2 hours

        return score.coerceIn(0f, 1f)
    }
}

// ========== SUPPORTING DATA CLASSES ==========

data class ResponseSection(
    val type: String,
    val content: MutableList<String>
)

enum class TimeUnit {
    MILLISECONDS,
    SECONDS,
    MINUTES,
    HOURS
}

// Mock data classes for raw sensor data (replace with actual implementations)
data class RawSensorData(
    val sessionId: Long,
    val startTimestamp: Long,
    val endTimestamp: Long,
    val totalSleepTime: Long,
    val sleepLatency: Long,
    val lightSleepDuration: Long,
    val deepSleepDuration: Long,
    val remSleepDuration: Long,
    val awakeDuration: Long,
    val movementData: List<Float>,
    val movementEvents: List<RawMovementEvent>,
    val noiseData: List<Float>,
    val noiseEvents: List<RawNoiseEvent>,
    val phaseTransitions: List<RawPhaseTransition>
)

data class RawMovementEvent(
    val timestamp: Long,
    val intensity: Float,
    val duration: Long,
    val type: String
)

data class RawNoiseEvent(
    val timestamp: Long,
    val decibelLevel: Float,
    val duration: Long,
    val frequency: Float
)

data class RawPhaseTransition(
    val timestamp: Long,
    val fromPhase: SleepPhase,
    val toPhase: SleepPhase,
    val confidence: Float
