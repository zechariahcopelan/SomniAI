package com.example.somniai.ai

import android.content.Context
import android.util.Log
import com.example.somniai.data.*
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.math.pow

/**
 * OpenAI API integration for AI-powered sleep insights
 *
 * Features:
 * - OpenAI Chat Completions API integration with GPT-3.5/GPT-4
 * - Rich prompt generation from comprehensive sleep analytics
 * - Intelligent response parsing into structured SleepInsight objects
 * - Robust error handling with exponential backoff retry
 * - Rate limiting and usage monitoring
 * - Configurable models and parameters
 * - Secure API key management
 * - Fallback mechanisms for service disruptions
 */
class OpenAIIntegration(
    private val context: Context,
    private val apiKey: String,
    private val configuration: OpenAIConfiguration = OpenAIConfiguration()
) {

    companion object {
        private const val TAG = "OpenAIIntegration"
        private const val BASE_URL = "https://api.openai.com/"

        // Rate limiting
        private const val DEFAULT_REQUESTS_PER_MINUTE = 20
        private const val DEFAULT_TOKENS_PER_MINUTE = 40000

        // Retry configuration
        private const val MAX_RETRIES = 3
        private const val INITIAL_RETRY_DELAY = 1000L
        private const val MAX_RETRY_DELAY = 8000L

        // Request timeouts
        private const val CONNECT_TIMEOUT = 30L
        private const val READ_TIMEOUT = 60L
        private const val WRITE_TIMEOUT = 60L

        // Token limits
        private const val MAX_PROMPT_TOKENS = 3500 // Leave room for response
        private const val MAX_RESPONSE_TOKENS = 1500
    }

    private val apiService: OpenAIApiService
    private val gson = Gson()
    private val requestLimiter = RequestLimiter()

    // Usage tracking
    private var totalRequestsToday = 0
    private var totalTokensUsedToday = 0
    private var lastResetDate = System.currentTimeMillis()

    init {
        apiService = createApiService()
        Log.d(TAG, "OpenAI Integration initialized with model: ${configuration.model}")
    }

    // ========== PUBLIC API ==========

    /**
     * Test connection to OpenAI API
     */
    suspend fun testConnection(): Boolean = withContext(Dispatchers.IO) {
        try {
            val testRequest = ChatCompletionRequest(
                model = configuration.model,
                messages = listOf(
                    ChatMessage(
                        role = "user",
                        content = "Test connection. Respond with 'OK' only."
                    )
                ),
                maxTokens = 5,
                temperature = 0.0
            )

            val response = apiService.createChatCompletion(testRequest)
            val isSuccessful = response.isSuccessful &&
                    response.body()?.choices?.isNotEmpty() == true

            Log.d(TAG, "Connection test result: $isSuccessful")
            isSuccessful

        } catch (e: Exception) {
            Log.e(TAG, "Connection test failed", e)
            false
        }
    }

    /**
     * Generate AI insights from sleep analytics
     */
    suspend fun generateInsights(
        prompt: String,
        context: InsightGenerationContext
    ): String = withContext(Dispatchers.IO) {
        try {
            // Check rate limits
            requestLimiter.checkRateLimit()

            // Reset daily counters if needed
            resetDailyCountersIfNeeded()

            // Check daily usage limits
            checkDailyLimits()

            // Build AI request
            val request = buildChatCompletionRequest(prompt, context)

            // Execute request with retry logic
            val response = executeWithRetry {
                apiService.createChatCompletion(request)
            }

            if (response.isSuccessful) {
                val chatResponse = response.body()
                if (chatResponse != null && chatResponse.choices.isNotEmpty()) {
                    val content = chatResponse.choices.first().message.content

                    // Update usage tracking
                    updateUsageTracking(chatResponse.usage)

                    Log.d(TAG, "AI insights generated successfully. Tokens used: ${chatResponse.usage?.totalTokens ?: 0}")
                    return@withContext content
                }
            }

            // Handle API errors
            val errorBody = response.errorBody()?.string() ?: "Unknown error"
            val error = try {
                gson.fromJson(errorBody, OpenAIError::class.java)
            } catch (e: Exception) {
                null
            }

            val errorMessage = error?.error?.message ?: errorBody
            Log.e(TAG, "OpenAI API error: $errorMessage")
            throw OpenAIException("OpenAI API error: $errorMessage", response.code())

        } catch (e: OpenAIException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error generating AI insights", e)
            throw OpenAIException("Failed to generate insights: ${e.message}", 0, e)
        }
    }

    /**
     * Generate insights with structured response parsing
     */
    suspend fun generateStructuredInsights(
        context: InsightGenerationContext
    ): List<SleepInsight> = withContext(Dispatchers.IO) {
        try {
            // Build comprehensive prompt
            val prompt = InsightsPromptBuilder.buildStructuredPrompt(context)

            // Get AI response
            val aiResponse = generateInsights(prompt, context)

            // Parse response into structured insights
            val insights = InsightsParser.parseAIResponse(aiResponse, context)

            Log.d(TAG, "Generated ${insights.size} structured insights")
            insights

        } catch (e: Exception) {
            Log.e(TAG, "Error generating structured insights", e)
            emptyList()
        }
    }

    /**
     * Get current usage statistics
     */
    fun getUsageStats(): OpenAIUsageStats {
        return OpenAIUsageStats(
            requestsToday = totalRequestsToday,
            tokensUsedToday = totalTokensUsedToday,
            requestsRemaining = configuration.maxRequestsPerDay - totalRequestsToday,
            tokensRemaining = configuration.maxTokensPerDay - totalTokensUsedToday,
            lastResetDate = lastResetDate
        )
    }

    /**
     * Update configuration
     */
    fun updateConfiguration(newConfig: OpenAIConfiguration) {
        configuration.apply {
            model = newConfig.model
            temperature = newConfig.temperature
            maxTokens = newConfig.maxTokens
            topP = newConfig.topP
            frequencyPenalty = newConfig.frequencyPenalty
            presencePenalty = newConfig.presencePenalty
            maxRequestsPerDay = newConfig.maxRequestsPerDay
            maxTokensPerDay = newConfig.maxTokensPerDay
        }
        Log.d(TAG, "Configuration updated: model=${configuration.model}")
    }

    // ========== PRIVATE IMPLEMENTATION ==========

    private fun createApiService(): OpenAIApiService {
        val loggingInterceptor = HttpLoggingInterceptor { message ->
            Log.d("$TAG-HTTP", message)
        }.apply {
            level = if (BuildConfig.DEBUG) {
                HttpLoggingInterceptor.Level.BODY
            } else {
                HttpLoggingInterceptor.Level.BASIC
            }
        }

        val authInterceptor = Interceptor { chain ->
            val request = chain.request().newBuilder()
                .addHeader("Authorization", "Bearer $apiKey")
                .addHeader("Content-Type", "application/json")
                .addHeader("User-Agent", "SomniAI-Android/1.0")
                .build()
            chain.proceed(request)
        }

        val client = OkHttpClient.Builder()
            .addInterceptor(authInterceptor)
            .addInterceptor(loggingInterceptor)
            .connectTimeout(CONNECT_TIMEOUT, TimeUnit.SECONDS)
            .readTimeout(READ_TIMEOUT, TimeUnit.SECONDS)
            .writeTimeout(WRITE_TIMEOUT, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build()

        val retrofit = Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()

        return retrofit.create(OpenAIApiService::class.java)
    }

    private fun buildChatCompletionRequest(
        prompt: String,
        context: InsightGenerationContext
    ): ChatCompletionRequest {
        // Build system message for context
        val systemMessage = buildSystemMessage(context)

        // Truncate prompt if too long
        val truncatedPrompt = truncatePrompt(prompt, MAX_PROMPT_TOKENS)

        return ChatCompletionRequest(
            model = configuration.model,
            messages = listOf(
                ChatMessage(role = "system", content = systemMessage),
                ChatMessage(role = "user", content = truncatedPrompt)
            ),
            temperature = configuration.temperature,
            maxTokens = min(configuration.maxTokens, MAX_RESPONSE_TOKENS),
            topP = configuration.topP,
            frequencyPenalty = configuration.frequencyPenalty,
            presencePenalty = configuration.presencePenalty,
            stop = listOf("END_INSIGHTS", "---")
        )
    }

    private fun buildSystemMessage(context: InsightGenerationContext): String {
        return when (context.generationType) {
            InsightGenerationType.POST_SESSION -> {
                """You are an expert sleep analyst providing personalized insights for a completed sleep session. 
                |
                |Guidelines:
                |- Focus on actionable, specific recommendations
                |- Prioritize the most impactful insights (1-3 key points)
                |- Use encouraging, supportive language
                |- Base insights on the provided sleep data
                |- Avoid medical advice, focus on lifestyle and behavioral recommendations
                |
                |Response format: Provide 2-4 insights in this exact JSON format:
                |[
                |  {
                |    "category": "DURATION|QUALITY|ENVIRONMENT|MOVEMENT|CONSISTENCY|GENERAL",
                |    "title": "Clear, descriptive title",
                |    "description": "Detailed explanation based on the data",
                |    "recommendation": "Specific, actionable advice",
                |    "priority": 1-3 (1=high, 2=medium, 3=low)
                |  }
                |]""".trimMargin()
            }
            InsightGenerationType.DAILY_ANALYSIS -> {
                """You are an expert sleep analyst providing daily pattern analysis and trend insights.
                |
                |Guidelines:
                |- Focus on patterns, trends, and consistency
                |- Identify both positive changes and areas needing attention
                |- Provide strategic recommendations for long-term improvement
                |- Consider weekly and monthly patterns
                |
                |Response format: Provide 3-5 insights in JSON format as specified.""".trimMargin()
            }
            InsightGenerationType.PERSONALIZED_ANALYSIS -> {
                """You are an expert sleep coach providing deep, personalized analysis based on extensive sleep history.
                |
                |Guidelines:
                |- Leverage personal baselines and individual patterns
                |- Identify unique personal habits and tendencies
                |- Provide customized strategies based on individual data
                |- Consider goal progress and habit formation
                |
                |Response format: Provide 4-7 insights in JSON format as specified.""".trimMargin()
            }
            else -> {
                """You are an expert sleep analyst. Provide helpful, evidence-based sleep insights.
                |Response format: Provide insights in JSON format as specified.""".trimMargin()
            }
        }
    }

    private fun truncatePrompt(prompt: String, maxTokens: Int): String {
        // Rough token estimation: ~4 characters per token
        val maxChars = maxTokens * 4
        return if (prompt.length > maxChars) {
            prompt.take(maxChars - 100) + "\n\n[Data truncated for length...]"
        } else {
            prompt
        }
    }

    private suspend fun <T> executeWithRetry(
        operation: suspend () -> retrofit2.Response<T>
    ): retrofit2.Response<T> {
        var lastException: Exception? = null
        var delay = INITIAL_RETRY_DELAY

        repeat(MAX_RETRIES) { attempt ->
            try {
                val response = operation()

                // Don't retry on client errors (4xx) except rate limiting
                if (response.isSuccessful ||
                    (response.code() in 400..499 && response.code() != 429)) {
                    return response
                }

                // Log retry attempt
                Log.w(TAG, "Request failed (attempt ${attempt + 1}/${MAX_RETRIES}): ${response.code()}")

            } catch (e: Exception) {
                lastException = e
                Log.w(TAG, "Request failed with exception (attempt ${attempt + 1}/${MAX_RETRIES})", e)
            }

            // Don't delay after the last attempt
            if (attempt < MAX_RETRIES - 1) {
                delay(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY) // Exponential backoff with cap
            }
        }

        // All retries failed
        throw lastException ?: IOException("All retry attempts failed")
    }

    private fun checkDailyLimits() {
        if (totalRequestsToday >= configuration.maxRequestsPerDay) {
            throw OpenAIException("Daily request limit exceeded", 429)
        }
        if (totalTokensUsedToday >= configuration.maxTokensPerDay) {
            throw OpenAIException("Daily token limit exceeded", 429)
        }
    }

    private fun resetDailyCountersIfNeeded() {
        val currentDate = System.currentTimeMillis()
        val daysSinceReset = (currentDate - lastResetDate) / (24 * 60 * 60 * 1000)

        if (daysSinceReset >= 1) {
            totalRequestsToday = 0
            totalTokensUsedToday = 0
            lastResetDate = currentDate
            Log.d(TAG, "Daily usage counters reset")
        }
    }

    private fun updateUsageTracking(usage: Usage?) {
        totalRequestsToday++
        if (usage != null) {
            totalTokensUsedToday += usage.totalTokens
        }
    }

    // ========== RATE LIMITING ==========

    private inner class RequestLimiter {
        private val requestTimes = mutableListOf<Long>()

        suspend fun checkRateLimit() {
            val currentTime = System.currentTimeMillis()
            val oneMinuteAgo = currentTime - 60_000

            // Remove requests older than 1 minute
            requestTimes.removeAll { it < oneMinuteAgo }

            // Check if we're at the rate limit
            if (requestTimes.size >= DEFAULT_REQUESTS_PER_MINUTE) {
                val oldestRequest = requestTimes.first()
                val delayUntil = oldestRequest + 60_000
                val delay = delayUntil - currentTime

                if (delay > 0) {
                    Log.d(TAG, "Rate limit reached, waiting ${delay}ms")
                    delay(delay)
                }
            }

            requestTimes.add(currentTime)
        }
    }
}

// ========== API INTERFACES ==========

private interface OpenAIApiService {
    @POST("v1/chat/completions")
    suspend fun createChatCompletion(
        @Body request: ChatCompletionRequest
    ): retrofit2.Response<ChatCompletionResponse>
}

// ========== DATA CLASSES ==========

/**
 * OpenAI Chat Completions API request
 */
private data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val temperature: Double = 0.7,
    @SerializedName("max_tokens")
    val maxTokens: Int = 1000,
    @SerializedName("top_p")
    val topP: Double = 1.0,
    @SerializedName("frequency_penalty")
    val frequencyPenalty: Double = 0.0,
    @SerializedName("presence_penalty")
    val presencePenalty: Double = 0.0,
    val stop: List<String>? = null
)

/**
 * Chat message for OpenAI API
 */
private data class ChatMessage(
    val role: String, // "system", "user", "assistant"
    val content: String
)

/**
 * OpenAI Chat Completions API response
 */
private data class ChatCompletionResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage?
)

/**
 * Choice in OpenAI response
 */
private data class Choice(
    val index: Int,
    val message: ChatMessage,
    @SerializedName("finish_reason")
    val finishReason: String
)

/**
 * Usage statistics from OpenAI
 */
private data class Usage(
    @SerializedName("prompt_tokens")
    val promptTokens: Int,
    @SerializedName("completion_tokens")
    val completionTokens: Int,
    @SerializedName("total_tokens")
    val totalTokens: Int
)

/**
 * OpenAI API error response
 */
private data class OpenAIError(
    val error: ErrorDetail
)

private data class ErrorDetail(
    val message: String,
    val type: String,
    val param: String?,
    val code: String?
)

/**
 * OpenAI configuration
 */
data class OpenAIConfiguration(
    var model: String = "gpt-3.5-turbo",
    var temperature: Double = 0.7,
    var maxTokens: Int = 1000,
    var topP: Double = 1.0,
    var frequencyPenalty: Double = 0.0,
    var presencePenalty: Double = 0.0,
    var maxRequestsPerDay: Int = 100,
    var maxTokensPerDay: Int = 50000
) {
    companion object {
        fun forGPT4(): OpenAIConfiguration {
            return OpenAIConfiguration(
                model = "gpt-4",
                temperature = 0.6,
                maxTokens = 1500,
                maxRequestsPerDay = 50, // Lower for GPT-4 due to cost
                maxTokensPerDay = 25000
            )
        }

        fun forGPT35Turbo(): OpenAIConfiguration {
            return OpenAIConfiguration(
                model = "gpt-3.5-turbo",
                temperature = 0.7,
                maxTokens = 1000,
                maxRequestsPerDay = 200,
                maxTokensPerDay = 100000
            )
        }
    }
}

/**
 * Usage statistics tracking
 */
data class OpenAIUsageStats(
    val requestsToday: Int,
    val tokensUsedToday: Int,
    val requestsRemaining: Int,
    val tokensRemaining: Int,
    val lastResetDate: Long
) {
    val usagePercentage: Float
        get() = if (requestsRemaining + requestsToday > 0) {
            (requestsToday.toFloat() / (requestsToday + requestsRemaining)) * 100f
        } else 0f
}

/**
 * Custom OpenAI exception
 */
class OpenAIException(
    message: String,
    val httpCode: Int = 0,
    cause: Throwable? = null
) : Exception(message, cause) {

    val isRateLimited: Boolean
        get() = httpCode == 429

    val isAuthError: Boolean
        get() = httpCode == 401

    val isQuotaExceeded: Boolean
        get() = message?.contains("quota", ignoreCase = true) == true

    val isServiceUnavailable: Boolean
        get() = httpCode in 500..599
}

// ========== PROMPT BUILDER ==========

/**
 * Builds sophisticated prompts from sleep analytics data
 */
object InsightsPromptBuilder {

    fun buildStructuredPrompt(context: InsightGenerationContext): String {
        val promptBuilder = StringBuilder()

        promptBuilder.append("# Sleep Analysis Data\n\n")

        // Add session data if available
        context.sessionData?.let { session ->
            promptBuilder.append("## Recent Sleep Session\n")
            promptBuilder.append("- Duration: ${formatDuration(session.duration)}\n")
            promptBuilder.append("- Sleep Efficiency: ${String.format("%.1f", session.sleepEfficiency)}%\n")
            promptBuilder.append("- Quality Score: ${session.sleepQualityScore ?: "Not calculated"}\n")
            promptBuilder.append("- Movement Intensity: ${String.format("%.1f", session.averageMovementIntensity)}\n")
            promptBuilder.append("- Noise Level: ${String.format("%.1f", session.averageNoiseLevel)} dB\n")
            promptBuilder.append("- Sleep Phases:\n")
            promptBuilder.append("  - Deep Sleep: ${formatDuration(session.deepSleepDuration)}\n")
            promptBuilder.append("  - REM Sleep: ${formatDuration(session.remSleepDuration)}\n")
            promptBuilder.append("  - Light Sleep: ${formatDuration(session.lightSleepDuration)}\n")
            promptBuilder.append("  - Awake Time: ${formatDuration(session.awakeDuration)}\n")
            promptBuilder.append("\n")
        }

        // Add trend analysis if available
        context.trendAnalysis?.let { trends ->
            if (trends.hasSufficientData) {
                promptBuilder.append("## Trend Analysis (${trends.periodAnalyzed} sessions)\n")
                promptBuilder.append("- Overall Trend: ${trends.overallTrend.displayName}\n")
                promptBuilder.append("- Quality Trend: ${trends.qualityTrend.displayName}\n")
                promptBuilder.append("- Duration Trend: ${trends.durationTrend.displayName}\n")
                promptBuilder.append("- Efficiency Trend: ${trends.efficiencyTrend.displayName}\n")
                if (trends.keyInsights.isNotEmpty()) {
                    promptBuilder.append("- Key Insights: ${trends.keyInsights.joinToString("; ")}\n")
                }
                promptBuilder.append("\n")
            }
        }

        // Add pattern analysis if available
        context.patternAnalysis?.let { patterns ->
            promptBuilder.append("## Sleep Patterns\n")
            promptBuilder.append("- Bedtime Consistency: ${String.format("%.1f", patterns.bedtimeConsistency.consistencyScore)}/10\n")
            promptBuilder.append("- Duration Consistency: ${String.format("%.1f", patterns.durationConsistency.consistencyScore)}/10\n")
            promptBuilder.append("- Overall Consistency: ${String.format("%.1f", patterns.overallConsistency)}/10\n")
            if (patterns.recognizedHabits.isNotEmpty()) {
                promptBuilder.append("- Recognized Habits: ${patterns.recognizedHabits.joinToString(", ") { it.name }}\n")
            }
            promptBuilder.append("\n")
        }

        // Add personal baseline if available
        context.personalBaseline?.let { baseline ->
            promptBuilder.append("## Personal Baseline (${baseline.sessionCount} sessions)\n")
            promptBuilder.append("- Average Duration: ${formatDuration(baseline.averageDuration)}\n")
            promptBuilder.append("- Average Quality: ${String.format("%.1f", baseline.averageQuality)}\n")
            promptBuilder.append("- Average Efficiency: ${String.format("%.1f", baseline.averageEfficiency)}%\n")
            promptBuilder.append("\n")
        }

        // Add context-specific instructions
        promptBuilder.append("## Analysis Request\n")
        when (context.generationType) {
            InsightGenerationType.POST_SESSION -> {
                promptBuilder.append("Please analyze this sleep session and provide 2-4 specific insights about what went well and what could be improved. Focus on actionable recommendations for tonight and future nights.\n")
            }
            InsightGenerationType.DAILY_ANALYSIS -> {
                promptBuilder.append("Please analyze the recent sleep patterns and trends. Provide 3-5 insights about consistency, trends, and areas for improvement over the coming week.\n")
            }
            InsightGenerationType.PERSONALIZED_ANALYSIS -> {
                promptBuilder.append("Please provide a comprehensive analysis of this person's sleep patterns, including personal strengths, areas for improvement, and customized recommendations based on their individual baseline and habits.\n")
            }
            else -> {
                promptBuilder.append("Please analyze the provided sleep data and provide helpful insights and recommendations.\n")
            }
        }

        return promptBuilder.toString()
    }

    private fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        return "${hours}h ${minutes}m"
    }
}

// ========== RESPONSE PARSER ==========

/**
 * Parses AI responses into structured SleepInsight objects
 */
object InsightsParser {

    private val gson = Gson()

    fun parseAIResponse(
        aiResponse: String,
        context: InsightGenerationContext
    ): List<SleepInsight> {
        return try {
            // Try to parse as JSON first
            parseJSONResponse(aiResponse, context)
        } catch (e: Exception) {
            Log.w("InsightsParser", "Failed to parse as JSON, trying text parsing", e)
            // Fallback to text parsing
            parseTextResponse(aiResponse, context)
        }
    }

    private fun parseJSONResponse(
        response: String,
        context: InsightGenerationContext
    ): List<SleepInsight> {
        // Extract JSON from response (handle markdown code blocks)
        val jsonStart = response.indexOf('[')
        val jsonEnd = response.lastIndexOf(']') + 1

        if (jsonStart == -1 || jsonEnd <= jsonStart) {
            throw IllegalArgumentException("No JSON array found in response")
        }

        val jsonString = response.substring(jsonStart, jsonEnd)
        val insights = gson.fromJson(jsonString, Array<AIInsightResponse>::class.java)

        return insights.mapIndexed { index, aiInsight ->
            SleepInsight(
                sessionId = context.sessionData?.id ?: 0L,
                category = parseCategory(aiInsight.category),
                title = aiInsight.title,
                description = aiInsight.description,
                recommendation = aiInsight.recommendation,
                priority = aiInsight.priority.coerceIn(1, 3),
                isAiGenerated = true,
                timestamp = System.currentTimeMillis()
            )
        }
    }

    private fun parseTextResponse(
        response: String,
        context: InsightGenerationContext
    ): List<SleepInsight> {
        // Basic text parsing as fallback
        val lines = response.split('\n').filter { it.isNotBlank() }
        val insights = mutableListOf<SleepInsight>()

        var currentTitle = ""
        var currentDescription = ""
        var currentRecommendation = ""

        for (line in lines) {
            when {
                line.startsWith("Title:", ignoreCase = true) ||
                        line.startsWith("Insight:", ignoreCase = true) -> {
                    currentTitle = line.substringAfter(':').trim()
                }
                line.startsWith("Description:", ignoreCase = true) ||
                        line.startsWith("Analysis:", ignoreCase = true) -> {
                    currentDescription = line.substringAfter(':').trim()
                }
                line.startsWith("Recommendation:", ignoreCase = true) ||
                        line.startsWith("Advice:", ignoreCase = true) -> {
                    currentRecommendation = line.substringAfter(':').trim()

                    // Create insight when we have all components
                    if (currentTitle.isNotEmpty() && currentDescription.isNotEmpty()) {
                        insights.add(
                            SleepInsight(
                                sessionId = context.sessionData?.id ?: 0L,
                                category = inferCategory(currentTitle, currentDescription),
                                title = currentTitle,
                                description = currentDescription,
                                recommendation = currentRecommendation,
                                priority = 2, // Default to medium priority
                                isAiGenerated = true,
                                timestamp = System.currentTimeMillis()
                            )
                        )

                        // Reset for next insight
                        currentTitle = ""
                        currentDescription = ""
                        currentRecommendation = ""
                    }
                }
            }
        }

        // If no structured insights found, create a general one
        if (insights.isEmpty() && response.isNotBlank()) {
            insights.add(
                SleepInsight(
                    sessionId = context.sessionData?.id ?: 0L,
                    category = InsightCategory.GENERAL,
                    title = "AI Sleep Analysis",
                    description = response.take(200) + if (response.length > 200) "..." else "",
                    recommendation = "Review the analysis and consider the suggested improvements.",
                    priority = 2,
                    isAiGenerated = true,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        return insights
    }

    private fun parseCategory(categoryString: String): InsightCategory {
        return when (categoryString.uppercase()) {
            "DURATION" -> InsightCategory.DURATION
            "QUALITY" -> InsightCategory.QUALITY
            "ENVIRONMENT" -> InsightCategory.ENVIRONMENT
            "MOVEMENT" -> InsightCategory.MOVEMENT
            "CONSISTENCY" -> InsightCategory.CONSISTENCY
            else -> InsightCategory.GENERAL
        }
    }

    private fun inferCategory(title: String, description: String): InsightCategory {
        val text = "$title $description".lowercase()
        return when {
            text.contains("duration") || text.contains("hours") || text.contains("sleep time") -> InsightCategory.DURATION
            text.contains("quality") || text.contains("efficiency") -> InsightCategory.QUALITY
            text.contains("noise") || text.contains("environment") || text.contains("temperature") -> InsightCategory.ENVIRONMENT
            text.contains("movement") || text.contains("restless") || text.contains("tossing") -> InsightCategory.MOVEMENT
            text.contains("consistency") || text.contains("schedule") || text.contains("bedtime") -> InsightCategory.CONSISTENCY
            else -> InsightCategory.GENERAL
        }
    }

    private data class AIInsightResponse(
        val category: String,
        val title: String,
        val description: String,
        val recommendation: String,
        val priority: Int
    )
}