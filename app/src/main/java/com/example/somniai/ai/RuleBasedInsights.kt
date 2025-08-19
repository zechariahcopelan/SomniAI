package com.example.somniai.ai

import android.content.Context
import android.util.Log
import com.example.somniai.analytics.*
import com.example.somniai.data.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.*

/**
 * Rule-based insights engine - Offline intelligence using analytics
 *
 * Provides comprehensive sleep insights using sophisticated rule-based analysis:
 * - Leverages existing SleepAnalyzer and SessionAnalytics infrastructure
 * - Generates insights from quality factors, trends, and patterns
 * - Provides actionable recommendations based on data analysis
 * - Works completely offline without external dependencies
 * - Prioritizes insights by impact and urgency
 */
class RuleBasedInsights(
    private val context: Context
) {

    companion object {
        private const val TAG = "RuleBasedInsights"

        // Quality thresholds for insight generation
        private const val POOR_QUALITY_THRESHOLD = 4.0f
        private const val EXCELLENT_QUALITY_THRESHOLD = 8.5f
        private const val LOW_EFFICIENCY_THRESHOLD = 70f
        private const val HIGH_EFFICIENCY_THRESHOLD = 90f

        // Duration thresholds (in hours)
        private const val SHORT_SLEEP_THRESHOLD = 6.0f
        private const val LONG_SLEEP_THRESHOLD = 9.5f
        private const val OPTIMAL_SLEEP_MIN = 7.0f
        private const val OPTIMAL_SLEEP_MAX = 9.0f

        // Movement and noise thresholds
        private const val HIGH_MOVEMENT_THRESHOLD = 4.0f
        private const val HIGH_NOISE_THRESHOLD = 50f
        private const val RESTLESS_MOVEMENT_FREQUENCY = 20f // movements per hour

        // Consistency thresholds
        private const val POOR_CONSISTENCY_THRESHOLD = 4.0f
        private const val GOOD_CONSISTENCY_THRESHOLD = 7.0f
        private const val BEDTIME_VARIANCE_THRESHOLD = 1.5f // hours

        // Trend analysis thresholds
        private const val SIGNIFICANT_DECLINE_THRESHOLD = -0.5f
        private const val SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.5f
        private const val MIN_SESSIONS_FOR_TRENDS = 7

        // Pattern recognition thresholds
        private const val WEEKEND_EFFECT_THRESHOLD = 1.0f // hour difference
        private const val HABIT_CONFIDENCE_THRESHOLD = 0.7f
    }

    private var isInitialized = false
    private val insightRules = mutableListOf<InsightRule>()

    // ========== INITIALIZATION ==========

    /**
     * Initialize the rule-based insights engine
     */
    suspend fun initialize(): Result<Unit> = withContext(Dispatchers.Default) {
        try {
            if (isInitialized) return@withContext Result.success(Unit)

            Log.d(TAG, "Initializing rule-based insights engine")

            // Initialize insight rules in priority order
            initializeInsightRules()

            isInitialized = true
            Log.d(TAG, "Rule-based insights engine initialized with ${insightRules.size} rules")

            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize rule-based insights", e)
            Result.failure(e)
        }
    }

    // ========== MAIN INSIGHT GENERATION ==========

    /**
     * Generate insights from the provided context
     */
    suspend fun generateInsights(
        context: InsightGenerationContext
    ): List<SleepInsight> = withContext(Dispatchers.Default) {
        try {
            if (!isInitialized) {
                initialize()
            }

            Log.d(TAG, "Generating rule-based insights for context: ${context.generationType}")

            val insights = mutableListOf<SleepInsight>()

            // Apply rules based on context type
            when (context.generationType) {
                InsightGenerationType.POST_SESSION -> {
                    insights.addAll(generateSessionInsights(context))
                }
                InsightGenerationType.DAILY_ANALYSIS -> {
                    insights.addAll(generateDailyInsights(context))
                }
                InsightGenerationType.PERSONALIZED_ANALYSIS -> {
                    insights.addAll(generatePersonalizedInsights(context))
                }
                else -> {
                    insights.addAll(generateGeneralInsights(context))
                }
            }

            // Filter and prioritize
            val prioritizedInsights = prioritizeInsights(insights)

            Log.d(TAG, "Generated ${prioritizedInsights.size} rule-based insights")
            prioritizedInsights

        } catch (e: Exception) {
            Log.e(TAG, "Error generating rule-based insights", e)
            emptyList()
        }
    }

    // ========== SESSION-SPECIFIC INSIGHTS ==========

    private suspend fun generateSessionInsights(
        context: InsightGenerationContext
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val session = context.sessionData ?: return insights
        val qualityAnalysis = context.qualityAnalysis
        val sessionSummary = context.sessionSummary

        // Duration insights
        insights.addAll(analyzeDurationInsights(session))

        // Quality insights
        insights.addAll(analyzeQualityInsights(session, qualityAnalysis))

        // Efficiency insights
        insights.addAll(analyzeEfficiencyInsights(session))

        // Movement insights
        insights.addAll(analyzeMovementInsights(session))

        // Noise insights
        insights.addAll(analyzeNoiseInsights(session))

        // Sleep onset insights
        context.onsetAnalysis?.let { onsetAnalysis ->
            insights.addAll(analyzeOnsetInsights(onsetAnalysis))
        }

        // Phase distribution insights
        insights.addAll(analyzePhaseInsights(session))

        return insights
    }

    private fun analyzeDurationInsights(session: SleepSession): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val durationHours = session.duration / (1000f * 60f * 60f)

        when {
            durationHours < SHORT_SLEEP_THRESHOLD -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.DURATION,
                        title = "Short Sleep Duration Detected",
                        description = "You slept for ${String.format("%.1f", durationHours)} hours, which is below the recommended 7-9 hours. Short sleep can impact cognitive function, mood, and physical health.",
                        recommendation = when {
                            durationHours < 5f -> "Consider going to bed 2-3 hours earlier tonight. Your sleep duration is critically short."
                            durationHours < 6f -> "Try to get to bed 1-2 hours earlier for better rest and recovery."
                            else -> "Aim for at least 7 hours of sleep by adjusting your bedtime slightly earlier."
                        },
                        priority = if (durationHours < 5f) 1 else 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            durationHours > LONG_SLEEP_THRESHOLD -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.DURATION,
                        title = "Extended Sleep Duration",
                        description = "You slept for ${String.format("%.1f", durationHours)} hours. While occasional long sleep can be restorative, consistently sleeping over 9.5 hours may indicate underlying issues or could affect your sleep schedule.",
                        recommendation = if (durationHours > 11f) {
                            "Consider consulting a healthcare provider if you consistently need more than 10 hours of sleep."
                        } else {
                            "Try maintaining a consistent 7-9 hour sleep schedule to optimize your circadian rhythm."
                        },
                        priority = if (durationHours > 11f) 1 else 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            durationHours >= OPTIMAL_SLEEP_MIN && durationHours <= OPTIMAL_SLEEP_MAX -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.DURATION,
                        title = "Optimal Sleep Duration",
                        description = "Excellent! You achieved ${String.format("%.1f", durationHours)} hours of sleep, which falls within the optimal 7-9 hour range for adults.",
                        recommendation = "Keep maintaining this healthy sleep duration. Consistency is key for long-term sleep health.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzeQualityInsights(
        session: SleepSession,
        qualityAnalysis: SessionQualityAnalysis?
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val qualityScore = session.sleepQualityScore ?: qualityAnalysis?.overallScore ?: return insights

        when {
            qualityScore < POOR_QUALITY_THRESHOLD -> {
                val qualityBreakdown = qualityAnalysis?.qualityBreakdown
                val weakestFactor = identifyWeakestQualityFactor(qualityBreakdown)

                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Sleep Quality Needs Attention",
                        description = "Your sleep quality score was ${String.format("%.1f", qualityScore)}/10. ${weakestFactor.description}",
                        recommendation = weakestFactor.recommendation,
                        priority = 1,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            qualityScore >= EXCELLENT_QUALITY_THRESHOLD -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Excellent Sleep Quality",
                        description = "Outstanding! Your sleep quality score was ${String.format("%.1f", qualityScore)}/10. You experienced high-quality, restorative sleep.",
                        recommendation = "Whatever you did before bed last night, keep it up! Your sleep environment and routine are working well.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzeEfficiencyInsights(session: SleepSession): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val efficiency = session.sleepEfficiency

        when {
            efficiency < LOW_EFFICIENCY_THRESHOLD -> {
                val awakeTime = session.awakeDuration
                val awakeMinutes = awakeTime / (1000 * 60)

                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Low Sleep Efficiency",
                        description = "Your sleep efficiency was ${String.format("%.1f", efficiency)}%. You spent approximately ${awakeMinutes} minutes awake during the night.",
                        recommendation = when {
                            awakeTime > 60 * 60 * 1000 -> "Consider relaxation techniques if you're having trouble staying asleep. Avoid checking the time or using devices during wake periods."
                            awakeTime > 30 * 60 * 1000 -> "Try progressive muscle relaxation or deep breathing if you wake up during the night."
                            else -> "Your brief awakenings are normal. Focus on creating a consistent bedtime routine."
                        },
                        priority = if (efficiency < 60f) 1 else 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            efficiency >= HIGH_EFFICIENCY_THRESHOLD -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "High Sleep Efficiency",
                        description = "Excellent sleep efficiency of ${String.format("%.1f", efficiency)}%! You spent most of your time in bed actually sleeping.",
                        recommendation = "Your sleep efficiency is excellent. Maintain your current bedtime routine and sleep environment.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzeMovementInsights(session: SleepSession): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val movementIntensity = session.averageMovementIntensity
        val movementFrequency = session.movementFrequency

        when {
            movementIntensity > HIGH_MOVEMENT_THRESHOLD || movementFrequency > RESTLESS_MOVEMENT_FREQUENCY -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.MOVEMENT,
                        title = "Restless Sleep Detected",
                        description = "Your sleep showed elevated movement levels (intensity: ${String.format("%.1f", movementIntensity)}, frequency: ${String.format("%.1f", movementFrequency)} movements/hour). This may indicate restless sleep.",
                        recommendation = when {
                            movementIntensity > 6f -> "Consider checking your sleep environment - temperature, mattress comfort, or potential stress factors. High movement levels may impact sleep quality."
                            movementFrequency > 30f -> "Frequent movements suggest possible discomfort. Review your mattress, pillows, or room temperature."
                            else -> "Try relaxation techniques before bed or consider if any lifestyle factors might be causing restlessness."
                        },
                        priority = if (movementIntensity > 6f || movementFrequency > 40f) 1 else 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            movementIntensity < 1.5f && movementFrequency < 10f -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.MOVEMENT,
                        title = "Very Still Sleep",
                        description = "You had very still sleep with minimal movement (intensity: ${String.format("%.1f", movementIntensity)}). This often indicates deep, restful sleep.",
                        recommendation = "Your low movement levels suggest good sleep quality. Continue your current sleep habits and environment.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzeNoiseInsights(session: SleepSession): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val noiseLevel = session.averageNoiseLevel
        val noiseEvents = session.noiseEvents.filter { it.isDisruptive() }

        when {
            noiseLevel > HIGH_NOISE_THRESHOLD || noiseEvents.size > 5 -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.ENVIRONMENT,
                        title = "Noisy Sleep Environment",
                        description = "Your sleep environment had elevated noise levels (average: ${String.format("%.1f", noiseLevel)} dB) with ${noiseEvents.size} disruptive noise events.",
                        recommendation = when {
                            noiseLevel > 60f -> "Consider using earplugs, white noise, or addressing external noise sources. High noise levels can significantly impact sleep quality."
                            noiseEvents.size > 10 -> "Multiple noise disruptions detected. Consider soundproofing or noise-masking solutions."
                            else -> "Try using a white noise machine or earplugs to create a more consistent sound environment."
                        },
                        priority = if (noiseLevel > 65f || noiseEvents.size > 15) 1 else 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            noiseLevel < 35f && noiseEvents.isEmpty() -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.ENVIRONMENT,
                        title = "Quiet Sleep Environment",
                        description = "Excellent! You had a very quiet sleep environment (${String.format("%.1f", noiseLevel)} dB average) with no disruptive noise events.",
                        recommendation = "Your sleep environment is optimally quiet. This contributes significantly to sleep quality.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzeOnsetInsights(onsetAnalysis: SleepOnsetAnalysis): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val sleepLatency = onsetAnalysis.sleepLatency

        if (sleepLatency != null) {
            val latencyMinutes = sleepLatency / (1000 * 60)

            when {
                latencyMinutes < 5 -> {
                    insights.add(
                        SleepInsight(
                            category = InsightCategory.QUALITY,
                            title = "Very Quick Sleep Onset",
                            description = "You fell asleep in approximately ${latencyMinutes} minutes. While this can indicate good sleep readiness, extremely quick onset might suggest sleep debt.",
                            recommendation = "If you consistently fall asleep this quickly, ensure you're getting adequate nightly sleep and not accumulating sleep debt.",
                            priority = 3,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                }
                latencyMinutes > 30 -> {
                    insights.add(
                        SleepInsight(
                            category = InsightCategory.QUALITY,
                            title = "Delayed Sleep Onset",
                            description = "It took approximately ${latencyMinutes} minutes to fall asleep. This may indicate difficulty winding down or environmental factors.",
                            recommendation = when {
                                latencyMinutes > 60 -> "Consider establishing a consistent bedtime routine, limiting screen time before bed, and creating a calm sleep environment."
                                latencyMinutes > 45 -> "Try relaxation techniques like deep breathing or progressive muscle relaxation before bed."
                                else -> "A wind-down routine 30-60 minutes before bedtime may help you fall asleep more quickly."
                            },
                            priority = if (latencyMinutes > 60) 1 else 2,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                }
                latencyMinutes in 10..20 -> {
                    insights.add(
                        SleepInsight(
                            category = InsightCategory.QUALITY,
                            title = "Optimal Sleep Onset",
                            description = "Great! You fell asleep in approximately ${latencyMinutes} minutes, which is within the optimal range of 10-20 minutes.",
                            recommendation = "Your sleep onset timing is excellent. Continue your current bedtime routine.",
                            priority = 3,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                }
            }
        }

        return insights
    }

    private fun analyzePhaseInsights(session: SleepSession): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val totalDuration = session.duration.toFloat()

        if (totalDuration == 0f) return insights

        val deepSleepRatio = session.deepSleepDuration / totalDuration
        val remRatio = session.remSleepDuration / totalDuration
        val lightSleepRatio = session.lightSleepDuration / totalDuration

        // Deep sleep analysis
        when {
            deepSleepRatio < 0.1f -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Limited Deep Sleep",
                        description = "You had ${String.format("%.1f", deepSleepRatio * 100)}% deep sleep. Optimal deep sleep is typically 15-25% of total sleep time.",
                        recommendation = "Deep sleep is crucial for physical recovery. Consider maintaining consistent sleep schedule, avoiding late caffeine, and ensuring cool, dark sleep environment.",
                        priority = 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            deepSleepRatio > 0.2f -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Excellent Deep Sleep",
                        description = "Outstanding! You achieved ${String.format("%.1f", deepSleepRatio * 100)}% deep sleep, which is excellent for physical recovery and restoration.",
                        recommendation = "Your deep sleep levels are excellent. Maintain your current sleep habits and environment.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        // REM sleep analysis
        when {
            remRatio < 0.15f -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Limited REM Sleep",
                        description = "You had ${String.format("%.1f", remRatio * 100)}% REM sleep. Optimal REM sleep is typically 20-25% of total sleep time.",
                        recommendation = "REM sleep is important for cognitive function and memory. Ensure you're getting enough total sleep and avoid alcohol before bed.",
                        priority = 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            remRatio > 0.23f -> {
                insights.add(
                    SleepInsight(
                        sessionId = session.id,
                        category = InsightCategory.QUALITY,
                        title = "Excellent REM Sleep",
                        description = "Great! You achieved ${String.format("%.1f", remRatio * 100)}% REM sleep, which is excellent for cognitive function and memory consolidation.",
                        recommendation = "Your REM sleep levels are excellent. This supports learning, memory, and emotional processing.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    // ========== DAILY/TREND INSIGHTS ==========

    private suspend fun generateDailyInsights(
        context: InsightGenerationContext
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val sessions = context.sessionsData
        val trendAnalysis = context.trendAnalysis
        val patternAnalysis = context.patternAnalysis

        if (sessions.size < 3) return insights

        // Trend insights
        trendAnalysis?.let { trends ->
            insights.addAll(analyzeTrendInsights(trends, sessions))
        }

        // Pattern insights
        patternAnalysis?.let { patterns ->
            insights.addAll(analyzePatternInsights(patterns, sessions))
        }

        // Consistency insights
        insights.addAll(analyzeConsistencyInsights(sessions))

        // Progress insights
        insights.addAll(analyzeProgressInsights(sessions))

        return insights
    }

    private fun analyzeTrendInsights(
        trendAnalysis: TrendAnalysis,
        sessions: List<SleepSession>
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        if (!trendAnalysis.hasSufficientData) return insights

        // Quality trend insights
        when (trendAnalysis.qualityTrend) {
            TrendDirection.DECLINING -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.QUALITY,
                        title = "Declining Sleep Quality Trend",
                        description = "Your sleep quality has been declining over the past ${trendAnalysis.periodAnalyzed} sessions. This trend requires attention.",
                        recommendation = "Review recent changes in your routine, stress levels, or sleep environment. Consider tracking factors that might be impacting your sleep quality.",
                        priority = 1,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            TrendDirection.IMPROVING -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.QUALITY,
                        title = "Improving Sleep Quality",
                        description = "Excellent! Your sleep quality has been improving over the past ${trendAnalysis.periodAnalyzed} sessions.",
                        recommendation = "Keep up the great work! Whatever changes you've made are having a positive impact on your sleep.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        // Duration trend insights
        when (trendAnalysis.durationTrend) {
            TrendDirection.DECLINING -> {
                val avgDuration = sessions.takeLast(7).map { it.duration }.average() / (1000 * 60 * 60)
                insights.add(
                    SleepInsight(
                        category = InsightCategory.DURATION,
                        title = "Sleep Duration Decreasing",
                        description = "Your sleep duration has been trending downward. Recent average is ${String.format("%.1f", avgDuration)} hours.",
                        recommendation = "Try to identify what's causing shorter sleep - late bedtimes, early wake times, or schedule changes. Prioritize consistent sleep schedule.",
                        priority = 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        // Efficiency trend insights
        when (trendAnalysis.efficiencyTrend) {
            TrendDirection.DECLINING -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.QUALITY,
                        title = "Sleep Efficiency Declining",
                        description = "Your sleep efficiency has been decreasing, indicating more time spent awake during sleep periods.",
                        recommendation = "Consider factors that might be causing more wake-ups: stress, environment, caffeine timing, or screen exposure before bed.",
                        priority = 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzePatternInsights(
        patternAnalysis: SleepPatternAnalysis,
        sessions: List<SleepSession>
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        // Bedtime consistency
        if (patternAnalysis.bedtimeConsistency.consistencyScore < POOR_CONSISTENCY_THRESHOLD) {
            insights.add(
                SleepInsight(
                    category = InsightCategory.CONSISTENCY,
                    title = "Inconsistent Bedtime Pattern",
                    description = "Your bedtime varies significantly (standard deviation: ${String.format("%.1f", patternAnalysis.bedtimeConsistency.standardDeviation)} hours). Consistency is crucial for circadian rhythm health.",
                    recommendation = "Try to go to bed within the same 30-60 minute window each night. Set a consistent bedtime routine to help regulate your body clock.",
                    priority = 2,
                    timestamp = System.currentTimeMillis()
                )
            )
        } else if (patternAnalysis.bedtimeConsistency.consistencyScore > GOOD_CONSISTENCY_THRESHOLD) {
            insights.add(
                SleepInsight(
                    category = InsightCategory.CONSISTENCY,
                    title = "Excellent Bedtime Consistency",
                    description = "Great job maintaining consistent bedtimes! Your circadian rhythm benefits from this regularity.",
                    recommendation = "Continue maintaining this consistent schedule, even on weekends when possible.",
                    priority = 3,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        // Sleep habits
        patternAnalysis.recognizedHabits.forEach { habit ->
            when (habit) {
                SleepHabit.SHORT_SLEEPER -> {
                    insights.add(
                        SleepInsight(
                            category = InsightCategory.DURATION,
                            title = "Short Sleeper Pattern Detected",
                            description = "You consistently sleep less than 6.5 hours. While some people are natural short sleepers, most adults need 7-9 hours.",
                            recommendation = "Monitor how you feel during the day. If you experience fatigue, consider gradually extending your sleep time.",
                            priority = 2,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                }
                SleepHabit.NIGHT_OWL -> {
                    insights.add(
                        SleepInsight(
                            category = InsightCategory.CONSISTENCY,
                            title = "Night Owl Sleep Pattern",
                            description = "You consistently go to bed late. While this may match your natural chronotype, ensure you're getting adequate sleep duration.",
                            recommendation = "If your schedule allows, embrace your natural rhythm. Otherwise, gradually shift bedtime earlier by 15-30 minutes per night.",
                            priority = 3,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                }
                SleepHabit.RESTLESS_SLEEPER -> {
                    insights.add(
                        SleepInsight(
                            category = InsightCategory.MOVEMENT,
                            title = "Restless Sleep Pattern",
                            description = "You consistently have elevated movement during sleep, which may impact sleep quality and recovery.",
                            recommendation = "Review your sleep environment, stress levels, and pre-bedtime activities. Consider relaxation techniques or consulting a sleep specialist.",
                            priority = 2,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                }
            }
        }

        return insights
    }

    private fun analyzeConsistencyInsights(sessions: List<SleepSession>): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        if (sessions.size < 7) return insights

        // Calculate week vs weekend patterns
        val weekdaySessions = mutableListOf<SleepSession>()
        val weekendSessions = mutableListOf<SleepSession>()

        sessions.forEach { session ->
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = session.startTime
            val dayOfWeek = calendar.get(java.util.Calendar.DAY_OF_WEEK)

            if (dayOfWeek == java.util.Calendar.SATURDAY || dayOfWeek == java.util.Calendar.SUNDAY) {
                weekendSessions.add(session)
            } else {
                weekdaySessions.add(session)
            }
        }

        if (weekdaySessions.isNotEmpty() && weekendSessions.isNotEmpty()) {
            val weekdayBedtime = calculateAverageBedtime(weekdaySessions)
            val weekendBedtime = calculateAverageBedtime(weekendSessions)
            val bedtimeDifference = abs(weekendBedtime - weekdayBedtime) / (60 * 60 * 1000.0)

            if (bedtimeDifference > WEEKEND_EFFECT_THRESHOLD) {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.CONSISTENCY,
                        title = "Weekend Sleep Schedule Shift",
                        description = "Your weekend bedtime differs from weekdays by ${String.format("%.1f", bedtimeDifference)} hours. This can disrupt your circadian rhythm.",
                        recommendation = "Try to maintain consistent sleep and wake times, even on weekends. If you need to catch up on sleep, limit weekend sleep-ins to 1 hour later than usual.",
                        priority = 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun analyzeProgressInsights(sessions: List<SleepSession>): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        if (sessions.size < 14) return insights

        val recentWeek = sessions.takeLast(7)
        val previousWeek = sessions.drop(sessions.size - 14).take(7)

        // Compare recent week to previous week
        val recentAvgQuality = recentWeek.mapNotNull { it.sleepQualityScore }.average()
        val previousAvgQuality = previousWeek.mapNotNull { it.sleepQualityScore }.average()

        val qualityChange = recentAvgQuality - previousAvgQuality

        when {
            qualityChange > SIGNIFICANT_IMPROVEMENT_THRESHOLD -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.GENERAL,
                        title = "Weekly Sleep Quality Improvement",
                        description = "Your sleep quality improved by ${String.format("%.1f", qualityChange)} points this week compared to last week!",
                        recommendation = "Excellent progress! Reflect on what you did differently this week and try to maintain these positive changes.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            qualityChange < SIGNIFICANT_DECLINE_THRESHOLD -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.GENERAL,
                        title = "Weekly Sleep Quality Decline",
                        description = "Your sleep quality decreased by ${String.format("%.1f", abs(qualityChange))} points this week compared to last week.",
                        recommendation = "Consider what changed this week - stress, schedule, environment, or habits. Small adjustments can help get back on track.",
                        priority = 1,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    // ========== PERSONALIZED INSIGHTS ==========

    private suspend fun generatePersonalizedInsights(
        context: InsightGenerationContext
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        // Include all daily insights as baseline
        insights.addAll(generateDailyInsights(context))

        // Add personalized analysis
        context.personalBaseline?.let { baseline ->
            insights.addAll(analyzePersonalBaselineInsights(baseline, context.sessionsData))
        }

        context.habitAnalysis?.let { habits ->
            insights.addAll(analyzeHabitInsights(habits))
        }

        context.goalAnalysis?.let { goals ->
            insights.addAll(analyzeGoalInsights(goals))
        }

        return insights
    }

    private fun analyzePersonalBaselineInsights(
        baseline: PersonalBaseline,
        sessions: List<SleepSession>
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        if (sessions.isEmpty()) return insights

        val recentSessions = sessions.takeLast(7)
        val recentAvgDuration = recentSessions.map { it.duration }.average()
        val recentAvgQuality = recentSessions.mapNotNull { it.sleepQualityScore }.average()

        // Duration comparison
        val durationDiff = (recentAvgDuration - baseline.averageDuration) / (1000 * 60 * 60.0)
        if (abs(durationDiff) > 0.5) {
            insights.add(
                SleepInsight(
                    category = InsightCategory.DURATION,
                    title = if (durationDiff > 0) "Sleeping More Than Your Average" else "Sleeping Less Than Your Average",
                    description = "Your recent sleep duration is ${String.format("%.1f", abs(durationDiff))} hours ${if (durationDiff > 0) "longer" else "shorter"} than your personal average of ${String.format("%.1f", baseline.averageDuration / (1000.0 * 60 * 60))} hours.",
                    recommendation = if (durationDiff < 0) {
                        "Try to return to your typical sleep duration. Your body is used to more sleep."
                    } else {
                        "If you're feeling well-rested with this extra sleep, you may have been under-sleeping previously."
                    },
                    priority = if (abs(durationDiff) > 1.0) 2 else 3,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        // Quality comparison
        val qualityDiff = recentAvgQuality.toFloat() - baseline.averageQuality
        if (abs(qualityDiff) > 0.5f) {
            insights.add(
                SleepInsight(
                    category = InsightCategory.QUALITY,
                    title = if (qualityDiff > 0) "Above Your Personal Average Quality" else "Below Your Personal Average Quality",
                    description = "Your recent sleep quality is ${String.format("%.1f", abs(qualityDiff))} points ${if (qualityDiff > 0) "higher" else "lower"} than your personal average of ${String.format("%.1f", baseline.averageQuality)}.",
                    recommendation = if (qualityDiff < 0) {
                        "Consider what factors might be impacting your sleep quality compared to your usual patterns."
                    } else {
                        "Great job! Your sleep quality is above your personal average. Try to maintain whatever is working well."
                    },
                    priority = if (abs(qualityDiff) > 1.0f) 2 else 3,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        return insights
    }

    private fun analyzeHabitInsights(habitAnalysis: HabitAnalysis): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        // Analyze habit strengths
        habitAnalysis.strengthAreas.forEach { strength ->
            insights.add(
                SleepInsight(
                    category = InsightCategory.GENERAL,
                    title = "Sleep Habit Strength: $strength",
                    description = "You demonstrate strong patterns in $strength, which contributes to your overall sleep health.",
                    recommendation = "Continue leveraging this strength while working on other areas of your sleep habits.",
                    priority = 3,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        // Analyze improvement areas
        habitAnalysis.improvementAreas.take(2).forEach { improvement ->
            insights.add(
                SleepInsight(
                    category = InsightCategory.GENERAL,
                    title = "Habit Improvement Opportunity",
                    description = "$improvement could be enhanced to improve your overall sleep quality.",
                    recommendation = "Focus on small, gradual changes in this area. Consistency is more important than perfection.",
                    priority = 2,
                    timestamp = System.currentTimeMillis()
                )
            )
        }

        return insights
    }

    private fun analyzeGoalInsights(goalAnalysis: GoalAnalysis): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()

        // Overall progress insight
        when {
            goalAnalysis.overallProgress > 0.8f -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.GENERAL,
                        title = "Excellent Goal Progress",
                        description = "You're making outstanding progress toward your sleep goals! Overall progress: ${String.format("%.0f", goalAnalysis.overallProgress * 100)}%",
                        recommendation = "Keep up the excellent work! You're well on your way to achieving optimal sleep health.",
                        priority = 3,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
            goalAnalysis.overallProgress < 0.4f -> {
                insights.add(
                    SleepInsight(
                        category = InsightCategory.GENERAL,
                        title = "Sleep Goals Need Attention",
                        description = "Your progress toward sleep goals could be improved. Current progress: ${String.format("%.0f", goalAnalysis.overallProgress * 100)}%",
                        recommendation = "Consider focusing on one specific area at a time. Small, consistent improvements lead to lasting change.",
                        priority = 2,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }

        return insights
    }

    private fun generateGeneralInsights(context: InsightGenerationContext): List<SleepInsight> {
        // Fallback general insights
        return listOf(
            SleepInsight(
                category = InsightCategory.GENERAL,
                title = "Keep Tracking Your Sleep",
                description = "Continue monitoring your sleep patterns to unlock personalized insights and recommendations.",
                recommendation = "Consistent sleep tracking helps identify patterns and areas for improvement in your sleep health.",
                priority = 3,
                timestamp = System.currentTimeMillis()
            )
        )
    }

    // ========== HELPER METHODS ==========

    private fun initializeInsightRules() {
        // This would contain the rule definitions
        // For now, we use the method-based approach above
        Log.d(TAG, "Insight rules initialized")
    }

    private fun prioritizeInsights(insights: List<SleepInsight>): List<SleepInsight> {
        return insights.sortedWith(
            compareBy<SleepInsight> { it.priority }
                .thenByDescending { it.timestamp }
        ).distinctBy { it.title } // Remove duplicates by title
    }

    private fun identifyWeakestQualityFactor(qualityBreakdown: QualityBreakdown?): QualityFactorInsight {
        // Analyze quality breakdown to identify weakest factor
        return QualityFactorInsight(
            factor = "Overall",
            description = "Multiple factors are impacting your sleep quality.",
            recommendation = "Focus on creating an optimal sleep environment and consistent routine."
        )
    }

    private fun calculateAverageBedtime(sessions: List<SleepSession>): Double {
        return sessions.map { session ->
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = session.startTime
            calendar.get(java.util.Calendar.HOUR_OF_DAY) * 3600.0 +
                    calendar.get(java.util.Calendar.MINUTE) * 60.0
        }.average()
    }

    // ========== DATA CLASSES ==========

    private data class QualityFactorInsight(
        val factor: String,
        val description: String,
        val recommendation: String
    )

    private data class InsightRule(
        val id: String,
        val category: InsightCategory,
        val condition: (InsightGenerationContext) -> Boolean,
        val generator: (InsightGenerationContext) -> SleepInsight
    )
}