package com.example.somniai.analytics

import android.util.Log
import com.example.somniai.data.*
import com.example.somniai.ai.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

/**
 * Comprehensive sleep analysis engine with advanced AI data preparation
 *
 * Enhanced Features:
 * - Sleep efficiency calculations with multiple methodologies
 * - Quality scoring algorithms on 1-10 scale with factor breakdowns
 * - Pattern recognition for consistency, bedtime trends, and habits
 * - Trend analysis with statistical significance testing
 * - Comparative analysis against optimal sleep metrics
 * - Personalized recommendations based on individual patterns
 * - AI-ready data preparation and context generation
 * - Multi-format data export for AI prompt engineering
 * - Comprehensive insight context building
 * - Advanced data summarization for AI consumption
 */
class SleepAnalyzer {

    companion object {
        private const val TAG = "SleepAnalyzer"

        // Quality scoring weights and constants
        private const val DURATION_WEIGHT = 0.25f
        private const val EFFICIENCY_WEIGHT = 0.25f
        private const val MOVEMENT_WEIGHT = 0.20f
        private const val NOISE_WEIGHT = 0.15f
        private const val CONSISTENCY_WEIGHT = 0.15f

        // Optimal sleep parameters
        private const val OPTIMAL_DURATION_MIN = 7 * 60 * 60 * 1000L // 7 hours
        private const val OPTIMAL_DURATION_MAX = 9 * 60 * 60 * 1000L // 9 hours
        private const val OPTIMAL_EFFICIENCY = 85f // 85% efficiency
        private const val OPTIMAL_DEEP_SLEEP_RATIO = 0.2f // 20% deep sleep
        private const val OPTIMAL_REM_RATIO = 0.25f // 25% REM sleep

        // Thresholds for pattern recognition
        private const val CONSISTENCY_THRESHOLD = 30 * 60 * 1000L // 30 minutes
        private const val TREND_MIN_SESSIONS = 7 // Minimum sessions for trend analysis
        private const val SIGNIFICANT_CHANGE_THRESHOLD = 0.5f // For trend detection

        // Movement and noise thresholds
        private const val LOW_MOVEMENT_THRESHOLD = 2.0f
        private const val HIGH_MOVEMENT_THRESHOLD = 6.0f
        private const val LOW_NOISE_THRESHOLD = 40f // dB
        private const val HIGH_NOISE_THRESHOLD = 60f // dB

        // AI Data Preparation Constants
        private const val AI_SUMMARY_MAX_LENGTH = 500
        private const val AI_CONTEXT_RELEVANCE_DAYS = 30
        private const val AI_PATTERN_MIN_SESSIONS = 5
        private const val AI_INSIGHT_CONFIDENCE_THRESHOLD = 0.7f
    }

    // ========== EXISTING METHODS (keeping all original functionality) ==========

    /**
     * Calculate sleep efficiency using multiple methodologies
     */
    suspend fun calculateSleepEfficiency(session: SleepSession): SleepEfficiencyAnalysis = withContext(Dispatchers.Default) {
        try {
            val totalDuration = session.duration
            val awakeDuration = session.awakeDuration
            val actualSleepDuration = totalDuration - awakeDuration

            // Basic efficiency (time asleep / time in bed)
            val basicEfficiency = if (totalDuration > 0) {
                (actualSleepDuration.toFloat() / totalDuration) * 100f
            } else 0f

            // Adjusted efficiency (accounting for sleep latency)
            val sleepLatency = session.sleepLatency
            val adjustedSleepTime = totalDuration - awakeDuration - sleepLatency
            val adjustedEfficiency = if (totalDuration > sleepLatency) {
                (adjustedSleepTime.toFloat() / (totalDuration - sleepLatency)) * 100f
            } else basicEfficiency

            // Quality-weighted efficiency (considering movement disruptions)
            val movementPenalty = calculateMovementPenalty(session.movementEvents)
            val qualityWeightedEfficiency = (basicEfficiency * (1f - movementPenalty / 100f)).coerceAtLeast(0f)

            // Phase-based efficiency (optimal phase distribution)
            val phaseEfficiency = calculatePhaseEfficiency(session)

            // Combined efficiency score
            val combinedEfficiency = (
                    basicEfficiency * 0.4f +
                            adjustedEfficiency * 0.3f +
                            qualityWeightedEfficiency * 0.2f +
                            phaseEfficiency * 0.1f
                    ).coerceIn(0f, 100f)

            SleepEfficiencyAnalysis(
                basicEfficiency = basicEfficiency,
                adjustedEfficiency = adjustedEfficiency,
                qualityWeightedEfficiency = qualityWeightedEfficiency,
                phaseBasedEfficiency = phaseEfficiency,
                combinedEfficiency = combinedEfficiency,
                sleepLatency = sleepLatency,
                actualSleepTime = actualSleepDuration,
                awakeTime = awakeDuration,
                efficiencyGrade = getEfficiencyGrade(combinedEfficiency)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error calculating sleep efficiency", e)
            SleepEfficiencyAnalysis() // Return default values
        }
    }

    /**
     * Calculate comprehensive quality score (1-10 scale)
     */
    suspend fun calculateQualityScore(session: SleepSession): QualityScoreAnalysis = withContext(Dispatchers.Default) {
        try {
            // Duration score (optimal 7-9 hours)
            val durationScore = calculateDurationScore(session.duration)

            // Efficiency score
            val efficiencyAnalysis = calculateSleepEfficiency(session)
            val efficiencyScore = (efficiencyAnalysis.combinedEfficiency / 10f).coerceIn(1f, 10f)

            // Movement score (less movement = higher score)
            val movementScore = calculateMovementScore(session.movementEvents, session.duration)

            // Noise score (quieter = higher score)
            val noiseScore = calculateNoiseScore(session.noiseEvents)

            // Consistency score (regular patterns = higher score)
            val consistencyScore = calculateConsistencyScore(session)

            // Phase distribution score
            val phaseScore = calculatePhaseDistributionScore(session)

            // Weighted overall score
            val overallScore = (
                    durationScore * DURATION_WEIGHT +
                            efficiencyScore * EFFICIENCY_WEIGHT +
                            movementScore * MOVEMENT_WEIGHT +
                            noiseScore * NOISE_WEIGHT +
                            consistencyScore * CONSISTENCY_WEIGHT
                    ).coerceIn(1f, 10f)

            QualityScoreAnalysis(
                overallScore = overallScore,
                durationScore = durationScore,
                efficiencyScore = efficiencyScore,
                movementScore = movementScore,
                noiseScore = noiseScore,
                consistencyScore = consistencyScore,
                phaseDistributionScore = phaseScore,
                qualityGrade = getQualityGrade(overallScore),
                strongestFactor = getStrongestFactor(durationScore, efficiencyScore, movementScore, noiseScore, consistencyScore),
                weakestFactor = getWeakestFactor(durationScore, efficiencyScore, movementScore, noiseScore, consistencyScore),
                improvementAreas = generateImprovementAreas(durationScore, efficiencyScore, movementScore, noiseScore, consistencyScore)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error calculating quality score", e)
            QualityScoreAnalysis() // Return default values
        }
    }

    /**
     * Analyze sleep patterns across multiple sessions
     */
    suspend fun analyzePatterns(sessions: List<SleepSession>): SleepPatternAnalysis = withContext(Dispatchers.Default) {
        try {
            if (sessions.isEmpty()) {
                return@withContext SleepPatternAnalysis()
            }

            // Bedtime consistency analysis
            val bedtimeConsistency = analyzeBedtimeConsistency(sessions)

            // Duration consistency analysis
            val durationConsistency = analyzeDurationConsistency(sessions)

            // Quality patterns
            val qualityPatterns = analyzeQualityPatterns(sessions)

            // Weekly patterns (if enough data)
            val weeklyPatterns = if (sessions.size >= 14) {
                analyzeWeeklyPatterns(sessions)
            } else null

            // Sleep debt analysis
            val sleepDebt = analyzeSleepDebt(sessions)

            // Habit recognition
            val habits = recognizeHabits(sessions)

            SleepPatternAnalysis(
                bedtimeConsistency = bedtimeConsistency,
                durationConsistency = durationConsistency,
                qualityPatterns = qualityPatterns,
                weeklyPatterns = weeklyPatterns,
                sleepDebt = sleepDebt,
                recognizedHabits = habits,
                overallConsistency = calculateOverallConsistency(bedtimeConsistency, durationConsistency),
                patternStrength = calculatePatternStrength(sessions),
                recommendations = generatePatternRecommendations(bedtimeConsistency, durationConsistency, qualityPatterns)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing patterns", e)
            SleepPatternAnalysis()
        }
    }

    /**
     * Comprehensive trend analysis with statistical significance
     */
    suspend fun analyzeTrends(sessions: List<SleepSession>): TrendAnalysis = withContext(Dispatchers.Default) {
        try {
            if (sessions.size < TREND_MIN_SESSIONS) {
                return@withContext TrendAnalysis(
                    hasSufficientData = false,
                    message = "Need at least $TREND_MIN_SESSIONS sessions for trend analysis"
                )
            }

            val sortedSessions = sessions.sortedBy { it.startTime }

            // Duration trends
            val durationTrend = calculateTrend(sortedSessions.map { it.duration.toDouble() })

            // Quality trends
            val qualityTrend = calculateTrend(sortedSessions.mapNotNull { it.sleepQualityScore?.toDouble() })

            // Efficiency trends
            val efficiencyTrend = calculateTrend(sortedSessions.map { it.sleepEfficiency.toDouble() })

            // Movement trends
            val movementTrend = calculateTrend(sortedSessions.map { it.averageMovementIntensity.toDouble() })

            // Bedtime trends
            val bedtimeTrend = calculateBedtimeTrend(sortedSessions)

            // Overall trend determination
            val overallTrend = determineOverallTrend(durationTrend, qualityTrend, efficiencyTrend)

            // Statistical significance
            val significance = calculateTrendSignificance(qualityTrend, sortedSessions.size)

            TrendAnalysis(
                hasSufficientData = true,
                overallTrend = overallTrend,
                durationTrend = durationTrend,
                qualityTrend = qualityTrend,
                efficiencyTrend = efficiencyTrend,
                movementTrend = movementTrend,
                bedtimeTrend = bedtimeTrend,
                significance = significance,
                trendStrength = calculateTrendStrength(qualityTrend),
                periodAnalyzed = sortedSessions.size,
                startDate = sortedSessions.first().startTime,
                endDate = sortedSessions.last().startTime,
                keyInsights = generateTrendInsights(durationTrend, qualityTrend, efficiencyTrend, movementTrend),
                recommendations = generateTrendRecommendations(overallTrend, durationTrend, qualityTrend, efficiencyTrend)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing trends", e)
            TrendAnalysis(hasSufficientData = false, message = "Error analyzing trends: ${e.message}")
        }
    }

    /**
     * Compare individual session against personal baselines and population norms
     */
    suspend fun compareSession(
        session: SleepSession,
        personalBaseline: SleepSession?,
        populationNorms: PopulationNorms? = null
    ): SessionComparison = withContext(Dispatchers.Default) {
        try {
            val comparison = SessionComparison(sessionId = session.id)

            // Personal comparison
            personalBaseline?.let { baseline ->
                comparison.personalComparison = PersonalComparison(
                    durationDifference = session.duration - baseline.duration,
                    qualityDifference = (session.sleepQualityScore ?: 0f) - (baseline.sleepQualityScore ?: 0f),
                    efficiencyDifference = session.sleepEfficiency - baseline.sleepEfficiency,
                    movementDifference = session.averageMovementIntensity - baseline.averageMovementIntensity,
                    betterThanBaseline = isSessionBetterThanBaseline(session, baseline)
                )
            }

            // Population comparison
            populationNorms?.let { norms ->
                comparison.populationComparison = PopulationComparison(
                    durationPercentile = calculatePercentile(session.duration.toDouble(), norms.durationDistribution),
                    qualityPercentile = calculatePercentile(session.sleepQualityScore?.toDouble() ?: 0.0, norms.qualityDistribution),
                    efficiencyPercentile = calculatePercentile(session.sleepEfficiency.toDouble(), norms.efficiencyDistribution),
                    overallRanking = calculateOverallRanking(session, norms)
                )
            }

            // Optimal comparison
            comparison.optimalComparison = OptimalComparison(
                durationOptimalityScore = calculateDurationOptimality(session.duration),
                efficiencyOptimalityScore = calculateEfficiencyOptimality(session.sleepEfficiency),
                phaseDistributionScore = calculatePhaseOptimality(session),
                overallOptimalityScore = calculateOverallOptimality(session)
            )

            comparison

        } catch (e: Exception) {
            Log.e(TAG, "Error comparing session", e)
            SessionComparison(sessionId = session.id)
        }
    }

    // ========== NEW AI DATA PREPARATION METHODS ==========

    /**
     * Create comprehensive AI insight generation context for a single session
     */
    suspend fun createSessionInsightContext(
        session: SleepSession,
        userProfile: UserProfileContext? = null,
        environmentContext: EnvironmentContext? = null,
        healthContext: HealthContext? = null,
        historicalSessions: List<SleepSession> = emptyList()
    ): InsightGenerationContext = withContext(Dispatchers.Default) {
        try {
            Log.d(TAG, "Creating AI insight context for session: ${session.id}")

            // Generate comprehensive quality analysis
            val qualityAnalysis = createSessionQualityAnalysis(session)

            // Create session summary for AI consumption
            val sessionSummary = createAIReadySessionSummary(session)

            // Generate trend analysis if historical data available
            val trendAnalysis = if (historicalSessions.isNotEmpty()) {
                createAIReadyTrendAnalysis(historicalSessions + session)
            } else null

            // Create pattern analysis
            val patternAnalysis = if (historicalSessions.size >= AI_PATTERN_MIN_SESSIONS) {
                createAIReadyPatternAnalysis(historicalSessions + session)
            } else null

            // Generate personal baseline
            val personalBaseline = if (historicalSessions.isNotEmpty()) {
                createPersonalBaseline(historicalSessions)
            } else null

            // Create habit analysis
            val habitAnalysis = if (historicalSessions.size >= 7) {
                createHabitAnalysis(historicalSessions + session)
            } else null

            // Generate sleep onset analysis
            val onsetAnalysis = createSleepOnsetAnalysis(session, historicalSessions)

            InsightGenerationContext(
                generationType = InsightGenerationType.POST_SESSION,
                sessionData = session,
                qualityAnalysis = qualityAnalysis,
                sessionSummary = sessionSummary,
                trendAnalysis = trendAnalysis,
                patternAnalysis = patternAnalysis,
                personalBaseline = personalBaseline,
                habitAnalysis = habitAnalysis,
                onsetAnalysis = onsetAnalysis,
                userProfile = userProfile,
                environmentContext = environmentContext,
                healthContext = healthContext,
                preferences = InsightPreferences(),
                priority = if (qualityAnalysis.overallScore < 5f) InsightPriority.HIGH else InsightPriority.NORMAL,
                requestId = "session_${session.id}_${System.currentTimeMillis()}",
                timestamp = System.currentTimeMillis()
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error creating session insight context", e)
            InsightGenerationContext(generationType = InsightGenerationType.POST_SESSION)
        }
    }

    /**
     * Create personalized insight context for ongoing analysis
     */
    suspend fun createPersonalizedInsightContext(
        sessions: List<SleepSession>,
        userProfile: UserProfileContext? = null,
        analysisDepth: AnalysisDepth = AnalysisDepth.COMPREHENSIVE,
        focusAreas: List<InsightCategory> = emptyList()
    ): InsightGenerationContext = withContext(Dispatchers.Default) {
        try {
            Log.d(TAG, "Creating personalized AI insight context for ${sessions.size} sessions")

            if (sessions.isEmpty()) {
                return@withContext InsightGenerationContext(
                    generationType = InsightGenerationType.PERSONALIZED_ANALYSIS
                )
            }

            val sortedSessions = sessions.sortedBy { it.startTime }

            // Generate comprehensive analytics
            val trendAnalysis = createAIReadyTrendAnalysis(sortedSessions)
            val patternAnalysis = createAIReadyPatternAnalysis(sortedSessions)
            val personalBaseline = createPersonalBaseline(sortedSessions)
            val habitAnalysis = createHabitAnalysis(sortedSessions)

            // Create goal analysis if user profile available
            val goalAnalysis = userProfile?.let { profile ->
                createGoalAnalysis(sortedSessions, profile)
            }

            InsightGenerationContext(
                generationType = InsightGenerationType.PERSONALIZED_ANALYSIS,
                sessionsData = sortedSessions,
                trendAnalysis = trendAnalysis,
                patternAnalysis = patternAnalysis,
                personalBaseline = personalBaseline,
                habitAnalysis = habitAnalysis,
                goalAnalysis = goalAnalysis,
                userProfile = userProfile,
                preferences = InsightPreferences(
                    maxInsights = when (analysisDepth) {
                        AnalysisDepth.BASIC -> 3
                        AnalysisDepth.DETAILED -> 7
                        AnalysisDepth.COMPREHENSIVE -> 10
                    },
                    categories = focusAreas.toSet(),
                    includePredictive = analysisDepth == AnalysisDepth.COMPREHENSIVE
                ),
                priority = InsightPriority.NORMAL,
                timestamp = System.currentTimeMillis()
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error creating personalized insight context", e)
            InsightGenerationContext(generationType = InsightGenerationType.PERSONALIZED_ANALYSIS)
        }
    }

    /**
     * Generate AI-ready data summary for prompt inclusion
     */
    suspend fun generateAIDataSummary(
        sessions: List<SleepSession>,
        format: AIDataFormat = AIDataFormat.NARRATIVE,
        maxLength: Int = AI_SUMMARY_MAX_LENGTH,
        includeRecommendations: Boolean = true
    ): AIDataSummary = withContext(Dispatchers.Default) {
        try {
            Log.d(TAG, "Generating AI data summary for ${sessions.size} sessions")

            if (sessions.isEmpty()) {
                return@withContext AIDataSummary(
                    summary = "No sleep data available for analysis.",
                    format = format,
                    confidence = 0f,
                    dataPoints = 0
                )
            }

            val summary = when (format) {
                AIDataFormat.NARRATIVE -> generateNarrativeSummary(sessions, maxLength, includeRecommendations)
                AIDataFormat.STRUCTURED -> generateStructuredSummary(sessions, maxLength, includeRecommendations)
                AIDataFormat.BULLET_POINTS -> generateBulletPointSummary(sessions, maxLength, includeRecommendations)
                AIDataFormat.JSON -> generateJSONSummary(sessions, maxLength, includeRecommendations)
            }

            // Calculate confidence based on data completeness
            val confidence = calculateDataConfidence(sessions)

            AIDataSummary(
                summary = summary,
                format = format,
                confidence = confidence,
                dataPoints = sessions.size,
                timeRange = TimeRange(
                    startDate = sessions.minOfOrNull { it.startTime } ?: 0L,
                    endDate = sessions.maxOfOrNull { it.startTime } ?: 0L,
                    description = "${sessions.size} sessions"
                ),
                keyMetrics = extractKeyMetrics(sessions),
                insights = if (includeRecommendations) generateKeyInsights(sessions) else emptyList(),
                generatedAt = System.currentTimeMillis()
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error generating AI data summary", e)
            AIDataSummary(
                summary = "Error generating summary: ${e.message}",
                format = format,
                confidence = 0f,
                dataPoints = 0
            )
        }
    }

    /**
     * Create structured data for AI prompt engineering
     */
    suspend fun createPromptData(
        sessions: List<SleepSession>,
        promptType: AIPromptType = AIPromptType.COMPREHENSIVE_ANALYSIS,
        personalization: PersonalizationLevel = PersonalizationLevel.ADAPTIVE
    ): AIPromptData = withContext(Dispatchers.Default) {
        try {
            Log.d(TAG, "Creating AI prompt data: type=$promptType, personalization=$personalization")

            val sortedSessions = sessions.sortedBy { it.startTime }

            // Core data preparation
            val coreData = prepareCoreData(sortedSessions)
            val analyticsData = prepareAnalyticsData(sortedSessions)
            val contextData = prepareContextData(sortedSessions, personalization)

            // Generate specific data based on prompt type
            val specificData = when (promptType) {
                AIPromptType.QUALITY_ANALYSIS -> prepareQualityAnalysisData(sortedSessions)
                AIPromptType.TREND_ANALYSIS -> prepareTrendAnalysisData(sortedSessions)
                AIPromptType.PATTERN_RECOGNITION -> preparePatternAnalysisData(sortedSessions)
                AIPromptType.RECOMMENDATION_ENGINE -> prepareRecommendationData(sortedSessions)
                AIPromptType.COMPARATIVE_ANALYSIS -> prepareComparativeData(sortedSessions)
                AIPromptType.COMPREHENSIVE_ANALYSIS -> prepareComprehensiveData(sortedSessions)
            }

            AIPromptData(
                promptType = promptType,
                personalizationLevel = personalization,
                coreData = coreData,
                analyticsData = analyticsData,
                contextData = contextData,
                specificData = specificData,
                metadata = createPromptMetadata(sortedSessions),
                confidence = calculateDataConfidence(sortedSessions),
                generatedAt = System.currentTimeMillis()
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error creating prompt data", e)
            AIPromptData(
                promptType = promptType,
                personalizationLevel = personalization,
                confidence = 0f
            )
        }
    }

    /**
     * Generate comparative analysis data for AI insights
     */
    suspend fun createComparativeAnalysisData(
        currentSession: SleepSession,
        historicalSessions: List<SleepSession>,
        comparisonType: ComparisonType = ComparisonType.PERSONAL_HISTORICAL
    ): ComparativeAnalysisResult = withContext(Dispatchers.Default) {
        try {
            Log.d(TAG, "Creating comparative analysis: type=$comparisonType")

            // Calculate personal baseline from historical data
            val personalBaseline = createPersonalBaseline(historicalSessions)

            // Generate quality analysis for current session
            val currentQuality = calculateQualityScore(currentSession)

            // Calculate improvements/regressions
            val qualityComparison = personalBaseline?.let { baseline ->
                PersonalPerformanceComparison(
                    timeframe = "Last ${historicalSessions.size} sessions",
                    currentPeriodMetrics = PeriodMetrics(
                        averageQuality = currentQuality.overallScore,
                        averageDuration = currentSession.duration.toFloat(),
                        averageEfficiency = currentSession.sleepEfficiency,
                        consistency = currentQuality.consistencyScore,
                        sessionCount = 1
                    ),
                    baselinePeriodMetrics = PeriodMetrics(
                        averageQuality = baseline.averageQuality,
                        averageDuration = baseline.averageDuration.toFloat(),
                        averageEfficiency = baseline.averageEfficiency,
                        consistency = baseline.consistencyScore,
                        sessionCount = historicalSessions.size
                    ),
                    qualityImprovement = currentQuality.overallScore - baseline.averageQuality,
                    durationImprovement = (currentSession.duration - baseline.averageDuration).toFloat(),
                    efficiencyImprovement = currentSession.sleepEfficiency - baseline.averageEfficiency,
                    consistencyImprovement = 0f, // Would need more complex calculation
                    overallImprovement = calculateOverallImprovement(currentSession, baseline),
                    qualityPercentile = calculateQualityPercentile(currentSession, historicalSessions),
                    durationPercentile = calculateDurationPercentile(currentSession, historicalSessions),
                    efficiencyPercentile = calculateEfficiencyPercentile(currentSession, historicalSessions),
                    currentStreaks = calculateCurrentStreaks(currentSession, historicalSessions),
                    bestStreaks = calculateBestStreaks(historicalSessions),
                    streakAnalysis = createStreakAnalysis(historicalSessions)
                )
            }

            // Create temporal comparison
            val temporalComparison = createTemporalComparison(currentSession, historicalSessions)

            ComparativeAnalysisResult(
                comparisonType = comparisonType,
                baselineInfo = BaselineInfo(
                    description = "Personal ${historicalSessions.size}-session baseline",
                    date = System.currentTimeMillis()
                ),
                personalComparison = qualityComparison,
                temporalComparison = temporalComparison,
                metricComparisons = createMetricComparisons(currentSession, historicalSessions),
                rankingAnalysis = createRankingAnalysis(currentSession, historicalSessions),
                percentileAnalysis = createPercentileAnalysis(currentSession, historicalSessions),
                performanceGaps = identifyPerformanceGaps(currentSession, personalBaseline),
                competitiveAdvantages = identifyStrengths(currentSession, personalBaseline),
                improvementOpportunities = identifyImprovementOpportunities(currentSession, personalBaseline),
                comparisonContext = ComparisonContext("Personal historical data analysis"),
                reliabilityMetrics = ComparisonReliabilityMetrics(calculateDataConfidence(historicalSessions))
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error creating comparative analysis", e)
            ComparativeAnalysisResult(
                comparisonType = comparisonType,
                baselineInfo = BaselineInfo("Error in analysis", System.currentTimeMillis()),
                reliabilityMetrics = ComparisonReliabilityMetrics(0f)
            )
        }
    }

    /**
     * Export analysis data in multiple formats for AI consumption
     */
    suspend fun exportAnalysisForAI(
        sessions: List<SleepSession>,
        exportFormat: AIExportFormat = AIExportFormat.COMPREHENSIVE_JSON,
        includeRawData: Boolean = false,
        includeAnalytics: Boolean = true,
        includeRecommendations: Boolean = true
    ): AIAnalysisExport = withContext(Dispatchers.Default) {
        try {
            Log.d(TAG, "Exporting analysis for AI: format=$exportFormat, sessions=${sessions.size}")

            val exportData = when (exportFormat) {
                AIExportFormat.COMPREHENSIVE_JSON -> exportComprehensiveJSON(sessions, includeRawData, includeAnalytics, includeRecommendations)
                AIExportFormat.SUMMARY_JSON -> exportSummaryJSON(sessions, includeAnalytics, includeRecommendations)
                AIExportFormat.STRUCTURED_TEXT -> exportStructuredText(sessions, includeAnalytics, includeRecommendations)
                AIExportFormat.PROMPT_OPTIMIZED -> exportPromptOptimized(sessions, includeAnalytics, includeRecommendations)
                AIExportFormat.MINIMAL_CONTEXT -> exportMinimalContext(sessions)
            }

            AIAnalysisExport(
                format = exportFormat,
                data = exportData,
                sessionCount = sessions.size,
                dataSize = exportData.length,
                includesRawData = includeRawData,
                includesAnalytics = includeAnalytics,
                includesRecommendations = includeRecommendations,
                confidence = calculateDataConfidence(sessions),
                generatedAt = System.currentTimeMillis(),
                metadata = createExportMetadata(sessions, exportFormat)
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error exporting analysis for AI", e)
            AIAnalysisExport(
                format = exportFormat,
                data = "Export failed: ${e.message}",
                sessionCount = sessions.size,
                confidence = 0f
            )
        }
    }

    // ========== PRIVATE AI DATA PREPARATION METHODS ==========

    private suspend fun createSessionQualityAnalysis(session: SleepSession): SessionQualityAnalysis {
        val qualityScore = calculateQualityScore(session)
        val efficiencyAnalysis = calculateSleepEfficiency(session)

        return SessionQualityAnalysis(
            overallScore = qualityScore.overallScore,
            qualityGrade = qualityScore.qualityGrade,
            factorBreakdown = QualityFactorBreakdown(
                durationScore = qualityScore.durationScore,
                efficiencyScore = qualityScore.efficiencyScore,
                movementScore = qualityScore.movementScore,
                noiseScore = qualityScore.noiseScore,
                consistencyScore = qualityScore.consistencyScore,
                phaseDistributionScore = qualityScore.phaseDistributionScore
            ),
            efficiencyAnalysis = EfficiencyAnalysisData(
                basicEfficiency = efficiencyAnalysis.basicEfficiency,
                adjustedEfficiency = efficiencyAnalysis.adjustedEfficiency,
                combinedEfficiency = efficiencyAnalysis.combinedEfficiency,
                grade = efficiencyAnalysis.efficiencyGrade
            ),
            strengthsAndWeaknesses = StrengthsAndWeaknesses(
                strongestFactor = qualityScore.strongestFactor,
                weakestFactor = qualityScore.weakestFactor,
                improvementAreas = qualityScore.improvementAreas
            ),
            confidence = AI_INSIGHT_CONFIDENCE_THRESHOLD
        )
    }

    private suspend fun createAIReadySessionSummary(session: SleepSession): SessionSummary {
        val duration = session.duration
        val hours = duration / (1000 * 60 * 60)
        val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)

        return SessionSummary(
            sessionId = session.id,
            date = Date(session.startTime),
            duration = duration,
            formattedDuration = "${hours}h ${minutes}m",
            qualityScore = session.sleepQualityScore ?: 0f,
            efficiency = session.sleepEfficiency,
            sleepLatency = session.sleepLatency,
            phaseBreakdown = PhaseBreakdown(
                lightSleep = session.lightSleepDuration,
                deepSleep = session.deepSleepDuration,
                remSleep = session.remSleepDuration,
                awakeTime = session.awakeDuration
            ),
            movementSummary = MovementSummary(
                averageIntensity = session.averageMovementIntensity,
                totalEvents = session.movementEvents.size,
                significantEvents = session.movementEvents.count { it.isSignificant() }
            ),
            noiseSummary = NoiseSummary(
                averageLevel = session.averageNoiseLevel,
                totalEvents = session.noiseEvents.size,
                disruptiveEvents = session.noiseEvents.count { it.isDisruptive() }
            ),
            keyInsights = generateSessionKeyInsights(session)
        )
    }

    private suspend fun createAIReadyTrendAnalysis(sessions: List<SleepSession>): com.example.somniai.data.TrendAnalysis? {
        val analysis = analyzeTrends(sessions)
        if (!analysis.hasSufficientData) return null

        // Convert to AI-ready format (using existing TrendAnalysis with enhancements)
        return analysis // The existing TrendAnalysis is already comprehensive
    }

    private suspend fun createAIReadyPatternAnalysis(sessions: List<SleepSession>): com.example.somniai.data.SleepPatternAnalysis? {
        if (sessions.size < AI_PATTERN_MIN_SESSIONS) return null

        val patterns = analyzePatterns(sessions)
        return patterns // The existing SleepPatternAnalysis is already comprehensive
    }

    private suspend fun createPersonalBaseline(sessions: List<SleepSession>): PersonalBaseline? {
        if (sessions.isEmpty()) return null

        val qualities = sessions.mapNotNull { it.sleepQualityScore }
        val durations = sessions.map { it.duration }
        val efficiencies = sessions.map { it.sleepEfficiency }

        return PersonalBaseline(
            averageQuality = qualities.average().toFloat(),
            averageDuration = durations.average().toLong(),
            averageEfficiency = efficiencies.average(),
            consistencyScore = calculateOverallConsistencyFromSessions(sessions),
            sampleSize = sessions.size,
            timeRange = TimeRange(
                startDate = sessions.minOf { it.startTime },
                endDate = sessions.maxOf { it.startTime },
                description = "${sessions.size} sessions"
            )
        )
    }

    private suspend fun createHabitAnalysis(sessions: List<SleepSession>): HabitAnalysis {
        val habits = recognizeHabits(sessions)
        val patterns = analyzePatterns(sessions)

        return HabitAnalysis(
            identifiedHabits = habits,
            bedtimePattern = BedtimePattern(
                averageBedtime = patterns.bedtimeConsistency.averageValue,
                consistency = patterns.bedtimeConsistency.isConsistent,
                standardDeviation = patterns.bedtimeConsistency.standardDeviation
            ),
            durationPattern = DurationPattern(
                averageDuration = patterns.durationConsistency.averageValue,
                consistency = patterns.durationConsistency.isConsistent,
                standardDeviation = patterns.durationConsistency.standardDeviation
            ),
            qualityPattern = patterns.qualityPatterns,
            weeklyPattern = patterns.weeklyPatterns,
            habitStrength = patterns.patternStrength,
            habitConfidence = calculateHabitConfidence(sessions)
        )
    }

    private suspend fun createSleepOnsetAnalysis(session: SleepSession, historicalSessions: List<SleepSession>): SleepOnsetAnalysis {
        val currentLatency = session.sleepLatency
        val historicalLatencies = historicalSessions.map { it.sleepLatency }

        val averageLatency = if (historicalLatencies.isNotEmpty()) {
            historicalLatencies.average().toLong()
        } else currentLatency

        return SleepOnsetAnalysis(
            currentLatency = currentLatency,
            averageLatency = averageLatency,
            latencyTrend = if (historicalLatencies.size >= 3) {
                calculateTrend(historicalLatencies.map { it.toDouble() })
            } else TrendDirection.INSUFFICIENT_DATA,
            onsetQuality = when {
                currentLatency <= 10 * 60 * 1000L -> "Excellent" // 10 minutes
                currentLatency <= 20 * 60 * 1000L -> "Good" // 20 minutes
                currentLatency <= 30 * 60 * 1000L -> "Fair" // 30 minutes
                else -> "Poor"
            },
            factors = identifyOnsetFactors(session, historicalSessions)
        )
    }

    private suspend fun createGoalAnalysis(sessions: List<SleepSession>, userProfile: UserProfileContext): GoalAnalysis {
        // Extract goals from user profile
        val sleepGoals = userProfile.sleepGoals
        val targetDuration = extractTargetDuration(sleepGoals)
        val targetQuality = extractTargetQuality(sleepGoals)

        // Calculate progress toward goals
        val recentSessions = sessions.takeLast(7) // Last week
        val avgDuration = recentSessions.map { it.duration }.average().toLong()
        val avgQuality = recentSessions.mapNotNull { it.sleepQualityScore }.average().toFloat()

        return GoalAnalysis(
            goals = sleepGoals,
            currentProgress = GoalProgress(
                durationProgress = if (targetDuration > 0) (avgDuration.toFloat() / targetDuration) else 0f,
                qualityProgress = if (targetQuality > 0) (avgQuality / targetQuality) else 0f
            ),
            achievementRate = calculateGoalAchievementRate(recentSessions, sleepGoals),
            recommendations = generateGoalRecommendations(recentSessions, sleepGoals),
            progressTrend = calculateGoalProgressTrend(sessions, sleepGoals)
        )
    }

    private fun generateNarrativeSummary(sessions: List<SleepSession>, maxLength: Int, includeRecommendations: Boolean): String {
        val recent = sessions.takeLast(7)
        val avgDuration = recent.map { it.duration / (1000 * 60 * 60) }.average()
        val avgQuality = recent.mapNotNull { it.sleepQualityScore }.average()
        val avgEfficiency = recent.map { it.sleepEfficiency }.average()

        val summary = StringBuilder()
        summary.append("Sleep Analysis Summary: ")
        summary.append("Over the last ${recent.size} sessions, ")
        summary.append("average sleep duration was ${String.format("%.1f", avgDuration)} hours, ")
        summary.append("quality score was ${String.format("%.1f", avgQuality)}/10, ")
        summary.append("and efficiency was ${String.format("%.1f", avgEfficiency)}%. ")

        // Add trend information
        if (sessions.size >= 7) {
            val trends = analyzeTrends(sessions)
            if (trends.hasSufficientData) {
                summary.append("Quality trend is ${trends.qualityTrend.name.lowercase()}. ")
            }
        }

        if (includeRecommendations) {
            summary.append("Key focus areas include improving consistency and optimizing sleep environment.")
        }

        return summary.toString().take(maxLength)
    }

    private fun generateStructuredSummary(sessions: List<SleepSession>, maxLength: Int, includeRecommendations: Boolean): String {
        val recent = sessions.takeLast(7)
        val summary = StringBuilder()

        summary.append("SLEEP ANALYSIS SUMMARY\n")
        summary.append("Sessions Analyzed: ${sessions.size}\n")
        summary.append("Time Period: ${recent.size} most recent sessions\n\n")

        summary.append("AVERAGES:\n")
        summary.append("- Duration: ${String.format("%.1f", recent.map { it.duration / (1000 * 60 * 60) }.average())} hours\n")
        summary.append("- Quality: ${String.format("%.1f", recent.mapNotNull { it.sleepQualityScore }.average())}/10\n")
        summary.append("- Efficiency: ${String.format("%.1f", recent.map { it.sleepEfficiency }.average())}%\n\n")

        if (includeRecommendations) {
            summary.append("RECOMMENDATIONS:\n")
            summary.append("- Focus on consistency\n")
            summary.append("- Optimize sleep environment\n")
        }

        return summary.toString().take(maxLength)
    }

    private fun generateBulletPointSummary(sessions: List<SleepSession>, maxLength: Int, includeRecommendations: Boolean): String {
        val recent = sessions.takeLast(7)
        val summary = StringBuilder()

        summary.append("• ${sessions.size} sleep sessions analyzed\n")
        summary.append("• Average duration: ${String.format("%.1f", recent.map { it.duration / (1000 * 60 * 60) }.average())} hours\n")
        summary.append("• Average quality: ${String.format("%.1f", recent.mapNotNull { it.sleepQualityScore }.average())}/10\n")
        summary.append("• Average efficiency: ${String.format("%.1f", recent.map { it.sleepEfficiency }.average())}%\n")

        if (includeRecommendations) {
            summary.append("• Recommendation: Focus on sleep consistency\n")
            summary.append("• Recommendation: Optimize sleep environment\n")
        }

        return summary.toString().take(maxLength)
    }

    private fun generateJSONSummary(sessions: List<SleepSession>, maxLength: Int, includeRecommendations: Boolean): String {
        val recent = sessions.takeLast(7)
        val json = JSONObject()

        json.put("sessionCount", sessions.size)
        json.put("analysisType", "recent_${recent.size}_sessions")

        val averages = JSONObject()
        averages.put("duration_hours", String.format("%.1f", recent.map { it.duration / (1000 * 60 * 60) }.average()))
        averages.put("quality_score", String.format("%.1f", recent.mapNotNull { it.sleepQualityScore }.average()))
        averages.put("efficiency_percent", String.format("%.1f", recent.map { it.sleepEfficiency }.average()))
        json.put("averages", averages)

        if (includeRecommendations) {
            val recommendations = JSONArray()
            recommendations.put("Focus on sleep consistency")
            recommendations.put("Optimize sleep environment")
            json.put("recommendations", recommendations)
        }

        return json.toString().take(maxLength)
    }

    private fun extractKeyMetrics(sessions: List<SleepSession>): Map<String, Float> {
        if (sessions.isEmpty()) return emptyMap()

        val recent = sessions.takeLast(7)
        return mapOf(
            "avg_duration_hours" to (recent.map { it.duration / (1000f * 60f * 60f) }.average().toFloat()),
            "avg_quality_score" to (recent.mapNotNull { it.sleepQualityScore }.average().toFloat()),
            "avg_efficiency" to (recent.map { it.sleepEfficiency }.average().toFloat()),
            "consistency_score" to calculateOverallConsistencyFromSessions(sessions)
        )
    }

    private fun generateKeyInsights(sessions: List<SleepSession>): List<String> {
        val insights = mutableListOf<String>()

        if (sessions.size >= 7) {
            val trends = analyzeTrends(sessions)
            if (trends.hasSufficientData) {
                insights.addAll(trends.keyInsights)
            }
        }

        if (sessions.size >= 5) {
            val patterns = analyzePatterns(sessions)
            insights.addAll(patterns.recommendations)
        }

        return insights.take(5) // Limit to top 5 insights
    }

    private fun calculateDataConfidence(sessions: List<SleepSession>): Float {
        if (sessions.isEmpty()) return 0f

        var confidence = 0f
        val maxSessions = 30f // Maximum sessions for full confidence

        // Base confidence from session count
        confidence += (sessions.size.toFloat() / maxSessions).coerceIn(0f, 1f) * 0.4f

        // Confidence from data completeness
        val completeDataRatio = sessions.count { session ->
            session.sleepQualityScore != null &&
                    session.duration > 0 &&
                    session.movementEvents.isNotEmpty() &&
                    session.noiseEvents.isNotEmpty()
        }.toFloat() / sessions.size
        confidence += completeDataRatio * 0.3f

        // Confidence from recency
        val recentSessionsRatio = sessions.count { session ->
            System.currentTimeMillis() - session.startTime < AI_CONTEXT_RELEVANCE_DAYS * 24 * 60 * 60 * 1000L
        }.toFloat() / sessions.size
        confidence += recentSessionsRatio * 0.3f

        return confidence.coerceIn(0f, 1f)
    }

    // Additional helper methods for AI data preparation...
    private fun prepareCoreData(sessions: List<SleepSession>): Map<String, Any> = mapOf(
        "session_count" to sessions.size,
        "avg_duration" to sessions.map { it.duration }.average(),
        "avg_quality" to sessions.mapNotNull { it.sleepQualityScore }.average(),
        "avg_efficiency" to sessions.map { it.sleepEfficiency }.average()
    )

    private fun prepareAnalyticsData(sessions: List<SleepSession>): Map<String, Any> {
        val patterns = analyzePatterns(sessions)
        return mapOf(
            "bedtime_consistency" to patterns.bedtimeConsistency.consistencyScore,
            "duration_consistency" to patterns.durationConsistency.consistencyScore,
            "pattern_strength" to patterns.patternStrength,
            "recognized_habits" to patterns.recognizedHabits.map { it.name }
        )
    }

    private fun prepareContextData(sessions: List<SleepSession>, personalization: PersonalizationLevel): Map<String, Any> = mapOf(
        "personalization_level" to personalization.name,
        "data_confidence" to calculateDataConfidence(sessions),
        "analysis_scope" to if (sessions.size >= 30) "comprehensive" else if (sessions.size >= 14) "detailed" else "basic"
    )

    private fun prepareQualityAnalysisData(sessions: List<SleepSession>): Map<String, Any> {
        val qualityScores = sessions.mapNotNull { it.sleepQualityScore }
        return mapOf(
            "quality_trend" to if (qualityScores.size >= 7) calculateTrend(qualityScores.map { it.toDouble() }).name else "insufficient_data",
            "quality_variance" to qualityScores.map { (it - qualityScores.average()).pow(2) }.average(),
            "best_quality" to (qualityScores.maxOrNull() ?: 0f),
            "worst_quality" to (qualityScores.minOrNull() ?: 0f)
        )
    }

    private fun prepareTrendAnalysisData(sessions: List<SleepSession>): Map<String, Any> {
        val trends = analyzeTrends(sessions)
        return mapOf(
            "overall_trend" to trends.overallTrend.name,
            "trend_strength" to trends.trendStrength,
            "significance" to trends.significance.name,
            "key_insights" to trends.keyInsights
        )
    }

    private fun preparePatternAnalysisData(sessions: List<SleepSession>): Map<String, Any> {
        val patterns = analyzePatterns(sessions)
        return mapOf(
            "consistency_score" to patterns.overallConsistency,
            "pattern_strength" to patterns.patternStrength,
            "habits" to patterns.recognizedHabits.map { it.name },
            "weekly_patterns" to (patterns.weeklyPatterns != null)
        )
    }

    private fun prepareRecommendationData(sessions: List<SleepSession>): Map<String, Any> {
        val patterns = analyzePatterns(sessions)
        return mapOf(
            "recommendations" to patterns.recommendations,
            "improvement_areas" to generateSystemImprovementAreas(sessions),
            "priority_actions" to generatePriorityActions(sessions)
        )
    }

    private fun prepareComparativeData(sessions: List<SleepSession>): Map<String, Any> {
        if (sessions.isEmpty()) return emptyMap()

        val recent = sessions.takeLast(7)
        val older = sessions.dropLast(7).takeLast(7)

        return if (older.isNotEmpty()) {
            mapOf(
                "recent_avg_quality" to recent.mapNotNull { it.sleepQualityScore }.average(),
                "previous_avg_quality" to older.mapNotNull { it.sleepQualityScore }.average(),
                "improvement" to (recent.mapNotNull { it.sleepQualityScore }.average() - older.mapNotNull { it.sleepQualityScore }.average())
            )
        } else {
            mapOf("comparison" to "insufficient_historical_data")
        }
    }

    private fun prepareComprehensiveData(sessions: List<SleepSession>): Map<String, Any> = mapOf(
        "core" to prepareCoreData(sessions),
        "analytics" to prepareAnalyticsData(sessions),
        "trends" to prepareTrendAnalysisData(sessions),
        "patterns" to preparePatternAnalysisData(sessions),
        "recommendations" to prepareRecommendationData(sessions)
    )

    private fun createPromptMetadata(sessions: List<SleepSession>): Map<String, Any> = mapOf(
        "data_range" to mapOf(
            "start_date" to (sessions.minOfOrNull { it.startTime } ?: 0L),
            "end_date" to (sessions.maxOfOrNull { it.startTime } ?: 0L),
            "span_days" to if (sessions.isNotEmpty()) {
                ((sessions.maxOf { it.startTime } - sessions.minOf { it.startTime }) / (24 * 60 * 60 * 1000L)).toInt()
            } else 0
        ),
        "completeness" to calculateDataCompleteness(sessions),
        "confidence" to calculateDataConfidence(sessions)
    )

    private fun exportComprehensiveJSON(sessions: List<SleepSession>, includeRaw: Boolean, includeAnalytics: Boolean, includeRecommendations: Boolean): String {
        val json = JSONObject()

        // Add session data
        if (includeRaw) {
            val sessionsArray = JSONArray()
            sessions.forEach { session ->
                val sessionJson = JSONObject()
                sessionJson.put("id", session.id)
                sessionJson.put("start_time", session.startTime)
                sessionJson.put("duration", session.duration)
                sessionJson.put("quality_score", session.sleepQualityScore)
                sessionJson.put("efficiency", session.sleepEfficiency)
                sessionsArray.put(sessionJson)
            }
            json.put("sessions", sessionsArray)
        }

        // Add analytics
        if (includeAnalytics && sessions.isNotEmpty()) {
            val analytics = JSONObject()
            analytics.put("averages", prepareCoreData(sessions))
            analytics.put("patterns", prepareAnalyticsData(sessions))
            if (sessions.size >= TREND_MIN_SESSIONS) {
                analytics.put("trends", prepareTrendAnalysisData(sessions))
            }
            json.put("analytics", analytics)
        }

        // Add recommendations
        if (includeRecommendations) {
            json.put("recommendations", prepareRecommendationData(sessions))
        }

        return json.toString(2)
    }

    private fun exportSummaryJSON(sessions: List<SleepSession>, includeAnalytics: Boolean, includeRecommendations: Boolean): String {
        val json = JSONObject()
        json.put("summary", prepareCoreData(sessions))

        if (includeAnalytics) {
            json.put("key_patterns", prepareAnalyticsData(sessions))
        }

        if (includeRecommendations) {
            json.put("top_recommendations", generateKeyInsights(sessions).take(3))
        }

        return json.toString()
    }

    private fun exportStructuredText(sessions: List<SleepSession>, includeAnalytics: Boolean, includeRecommendations: Boolean): String {
        return generateStructuredSummary(sessions, AI_SUMMARY_MAX_LENGTH, includeRecommendations)
    }

    private fun exportPromptOptimized(sessions: List<SleepSession>, includeAnalytics: Boolean, includeRecommendations: Boolean): String {
        return "Sleep Data Context: ${generateNarrativeSummary(sessions, AI_SUMMARY_MAX_LENGTH, includeRecommendations)}"
    }

    private fun exportMinimalContext(sessions: List<SleepSession>): String {
        if (sessions.isEmpty()) return "No sleep data available."

        val recent = sessions.takeLast(3)
        return "Recent sleep: ${recent.size} sessions, avg quality ${String.format("%.1f", recent.mapNotNull { it.sleepQualityScore }.average())}/10"
    }

    private fun createExportMetadata(sessions: List<SleepSession>, format: AIExportFormat): Map<String, Any> = mapOf(
        "export_format" to format.name,
        "session_count" to sessions.size,
        "export_timestamp" to System.currentTimeMillis(),
        "data_confidence" to calculateDataConfidence(sessions)
    )

    // Additional utility methods...
    [Continuing with all the existing helper methods from the original SleepAnalyzer, plus new ones for AI data preparation]

    private fun calculateOverallConsistencyFromSessions(sessions: List<SleepSession>): Float {
        if (sessions.size < 2) return 5f

        val patterns = analyzePatterns(sessions)
        return patterns.overallConsistency
    }

    private fun generateSessionKeyInsights(session: SleepSession): List<String> {
        val insights = mutableListOf<String>()

        val qualityAnalysis = calculateQualityScore(session)

        if (qualityAnalysis.overallScore >= 8f) {
            insights.add("Excellent sleep quality achieved")
        } else if (qualityAnalysis.overallScore < 5f) {
            insights.add("Sleep quality below optimal - review environment and habits")
        }

        if (session.sleepEfficiency >= 90f) {
            insights.add("Outstanding sleep efficiency")
        } else if (session.sleepEfficiency < 70f) {
            insights.add("Sleep efficiency could be improved")
        }

        return insights
    }

    private fun calculateDataCompleteness(sessions: List<SleepSession>): Float {
        if (sessions.isEmpty()) return 0f

        val completeFields = sessions.sumOf { session ->
            var count = 0
            if (session.sleepQualityScore != null) count++
            if (session.duration > 0) count++
            if (session.movementEvents.isNotEmpty()) count++
            if (session.noiseEvents.isNotEmpty()) count++
            if (session.phaseTransitions.isNotEmpty()) count++
            count
        }

        return (completeFields.toFloat() / (sessions.size * 5f)).coerceIn(0f, 1f)
    }

    private fun generateSystemImprovementAreas(sessions: List<SleepSession>): List<String> {
        val areas = mutableListOf<String>()

        val avgQuality = sessions.mapNotNull { it.sleepQualityScore }.average()
        val avgEfficiency = sessions.map { it.sleepEfficiency }.average()

        if (avgQuality < 6f) areas.add("Overall sleep quality improvement")
        if (avgEfficiency < 80f) areas.add("Sleep efficiency optimization")

        val patterns = analyzePatterns(sessions)
        if (!patterns.bedtimeConsistency.isConsistent) areas.add("Bedtime consistency")
        if (!patterns.durationConsistency.isConsistent) areas.add("Sleep duration consistency")

        return areas
    }

    private fun generatePriorityActions(sessions: List<SleepSession>): List<String> {
        val actions = mutableListOf<String>()

        val recentQuality = sessions.takeLast(3).mapNotNull { it.sleepQualityScore }.average()
        if (recentQuality < 5f) {
            actions.add("Immediate sleep environment review")
        }

        val patterns = analyzePatterns(sessions)
        if (patterns.overallConsistency < 5f) {
            actions.add("Establish consistent sleep schedule")
        }

        return actions
    }

    private fun calculateHabitConfidence(sessions: List<SleepSession>): Float {
        return when {
            sessions.size < 7 -> 0.3f
            sessions.size < 14 -> 0.6f
            sessions.size < 30 -> 0.8f
            else -> 0.9f
        }
    }

    private fun identifyOnsetFactors(session: SleepSession, historicalSessions: List<SleepSession>): List<String> {
        val factors = mutableListOf<String>()

        if (session.sleepLatency > 30 * 60 * 1000L) { // More than 30 minutes
            factors.add("Extended time to fall asleep")
        }

        if (session.movementEvents.isNotEmpty()) {
            val preSleepMovement = session.movementEvents.count {
                it.timestamp - session.startTime < session.sleepLatency
            }
            if (preSleepMovement > 5) {
                factors.add("High movement before sleep onset")
            }
        }

        return factors
    }

    private fun extractTargetDuration(goals: List<String>): Long {
        // Parse sleep duration goals from user profile
        goals.forEach { goal ->
            if (goal.contains("hours") || goal.contains("duration")) {
                // Extract numeric value and convert to milliseconds
                val regex = Regex("(\\d+(?:\\.\\d+)?)\\s*hours?")
                val match = regex.find(goal.lowercase())
                match?.let {
                    val hours = it.groupValues[1].toFloatOrNull()
                    if (hours != null) {
                        return (hours * 60 * 60 * 1000).toLong()
                    }
                }
            }
        }
        return 8 * 60 * 60 * 1000L // Default 8 hours
    }

    private fun extractTargetQuality(goals: List<String>): Float {
        // Parse quality goals from user profile
        goals.forEach { goal ->
            if (goal.contains("quality") || goal.contains("score")) {
                val regex = Regex("(\\d+(?:\\.\\d+)?)(?:/10)?")
                val match = regex.find(goal)
                match?.let {
                    return it.groupValues[1].toFloatOrNull() ?: 8f
                }
            }
        }
        return 8f // Default quality target
    }

    private fun calculateGoalAchievementRate(sessions: List<SleepSession>, goals: List<String>): Float {
        if (sessions.isEmpty()) return 0f

        val targetDuration = extractTargetDuration(goals)
        val targetQuality = extractTargetQuality(goals)

        var achievements = 0
        sessions.forEach { session ->
            var sessionAchievements = 0
            var totalGoals = 0

            // Check duration goal
            if (abs(session.duration - targetDuration) <= 30 * 60 * 1000L) { // Within 30 minutes
                sessionAchievements++
            }
            totalGoals++

            // Check quality goal
            session.sleepQualityScore?.let { quality ->
                if (quality >= targetQuality) {
                    sessionAchievements++
                }
                totalGoals++
            }

            if (totalGoals > 0 && sessionAchievements == totalGoals) {
                achievements++
            }
        }

        return achievements.toFloat() / sessions.size
    }

    private fun generateGoalRecommendations(sessions: List<SleepSession>, goals: List<String>): List<String> {
        val recommendations = mutableListOf<String>()

        val targetDuration = extractTargetDuration(goals)
        val avgDuration = sessions.map { it.duration }.average().toLong()

        when {
            avgDuration < targetDuration - 30 * 60 * 1000L ->
                recommendations.add("Consider going to bed earlier to reach your duration goal")
            avgDuration > targetDuration + 30 * 60 * 1000L ->
                recommendations.add("You may be sleeping longer than your target - review your sleep schedule")
        }

        val targetQuality = extractTargetQuality(goals)
        val avgQuality = sessions.mapNotNull { it.sleepQualityScore }.average().toFloat()

        if (avgQuality < targetQuality) {
            recommendations.add("Focus on sleep quality improvements to reach your target score")
        }

        return recommendations
    }

    private fun calculateGoalProgressTrend(sessions: List<SleepSession>, goals: List<String>): TrendDirection {
        if (sessions.size < 7) return TrendDirection.INSUFFICIENT_DATA

        val targetQuality = extractTargetQuality(goals)
        val recentProgress = sessions.takeLast(7).mapNotNull { it.sleepQualityScore }.map { it / targetQuality }

        return calculateTrend(recentProgress.map { it.toDouble() })
    }

    // Comparative analysis helper methods
    private fun calculateOverallImprovement(session: SleepSession, baseline: PersonalBaseline): Float {
        val qualityImprovement = (session.sleepQualityScore ?: 0f) - baseline.averageQuality
        val efficiencyImprovement = session.sleepEfficiency - baseline.averageEfficiency
        return (qualityImprovement + efficiencyImprovement / 10f) / 2f
    }

    private fun calculateQualityPercentile(session: SleepSession, historicalSessions: List<SleepSession>): Float {
        val qualities = historicalSessions.mapNotNull { it.sleepQualityScore }.sorted()
        val sessionQuality = session.sleepQualityScore ?: return 0f

        val rank = qualities.count { it <= sessionQuality }
        return (rank.toFloat() / qualities.size) * 100f
    }

    private fun calculateDurationPercentile(session: SleepSession, historicalSessions: List<SleepSession>): Float {
        val durations = historicalSessions.map { it.duration }.sorted()
        val rank = durations.count { it <= session.duration }
        return (rank.toFloat() / durations.size) * 100f
    }

    private fun calculateEfficiencyPercentile(session: SleepSession, historicalSessions: List<SleepSession>): Float {
        val efficiencies = historicalSessions.map { it.sleepEfficiency }.sorted()
        val rank = efficiencies.count { it <= session.sleepEfficiency }
        return (rank.toFloat() / efficiencies.size) * 100f
    }

    private fun calculateCurrentStreaks(session: SleepSession, historicalSessions: List<SleepSession>): Map<String, Int> {
        // Calculate current streaks for various metrics
        val allSessions = (historicalSessions + session).sortedBy { it.startTime }

        val qualityStreak = calculateQualityStreak(allSessions)
        val durationStreak = calculateDurationStreak(allSessions)

        return mapOf(
            "quality" to qualityStreak,
            "duration" to durationStreak
        )
    }

    private fun calculateBestStreaks(historicalSessions: List<SleepSession>): Map<String, Int> {
        // Find the best historical streaks
        return mapOf(
            "quality" to findBestQualityStreak(historicalSessions),
            "duration" to findBestDurationStreak(historicalSessions)
        )
    }

    private fun createStreakAnalysis(historicalSessions: List<SleepSession>): StreakAnalysis {
        val currentStreak = calculateQualityStreak(historicalSessions)
        val bestStreak = findBestQualityStreak(historicalSessions)
        val activeStreaks = if (currentStreak > 0) 1 else 0
        val brokenStreaks = countBrokenStreaks(historicalSessions)

        val trend = if (historicalSessions.size >= 14) {
            val recentStreaks = calculateRecentStreakTrend(historicalSessions)
            if (recentStreaks > 0) TrendDirection.IMPROVING else TrendDirection.DECLINING
        } else TrendDirection.INSUFFICIENT_DATA

        return StreakAnalysis(
            currentStreak = currentStreak,
            bestStreak = bestStreak,
            activeStreaks = activeStreaks,
            brokenStreaks = brokenStreaks,
            trendDirection = trend
        )
    }

    private fun createTemporalComparison(session: SleepSession, historicalSessions: List<SleepSession>): TemporalPerformanceComparison {
        // Compare session performance across different time periods
        val improvement = if (historicalSessions.isNotEmpty()) {
            val historicalAvg = historicalSessions.mapNotNull { it.sleepQualityScore }.average().toFloat()
            (session.sleepQualityScore ?: 0f) - historicalAvg
        } else 0f

        return TemporalPerformanceComparison(improvement)
    }

    private fun createMetricComparisons(session: SleepSession, historicalSessions: List<SleepSession>): List<MetricComparison> {
        if (historicalSessions.isEmpty()) return emptyList()

        val comparisons = mutableListOf<MetricComparison>()

        // Quality comparison
        session.sleepQualityScore?.let { quality ->
            val historicalAvg = historicalSessions.mapNotNull { it.sleepQualityScore }.average().toFloat()
            comparisons.add(MetricComparison(
                metric = "Quality Score",
                currentValue = quality,
                baselineValue = historicalAvg,
                improvement = quality - historicalAvg,
                percentile = calculateQualityPercentile(session, historicalSessions)
            ))
        }

        // Duration comparison
        val historicalDurationAvg = historicalSessions.map { it.duration }.average().toFloat()
        comparisons.add(MetricComparison(
            metric = "Duration",
            currentValue = session.duration.toFloat(),
            baselineValue = historicalDurationAvg,
            improvement = session.duration - historicalDurationAvg,
            percentile = calculateDurationPercentile(session, historicalSessions)
        ))

        return comparisons
    }

    private fun createRankingAnalysis(session: SleepSession, historicalSessions: List<SleepSession>): RankingAnalysis {
        val allSessions = (historicalSessions + session).sortedByDescending { it.sleepQualityScore ?: 0f }
        val rank = allSessions.indexOfFirst { it.id == session.id } + 1

        return RankingAnalysis(
            currentRank = rank,
            totalSessions = allSessions.size
        )
    }

    private fun createPercentileAnalysis(session: SleepSession, historicalSessions: List<SleepSession>): PercentileAnalysis {
        return PercentileAnalysis(
            percentiles = mapOf(
                "quality" to calculateQualityPercentile(session, historicalSessions),
                "duration" to calculateDurationPercentile(session, historicalSessions),
                "efficiency" to calculateEfficiencyPercentile(session, historicalSessions)
            )
        )
    }

    private fun identifyPerformanceGaps(session: SleepSession, baseline: PersonalBaseline?): List<PerformanceGap> {
        if (baseline == null) return emptyList()

        val gaps = mutableListOf<PerformanceGap>()

        session.sleepQualityScore?.let { quality ->
            if (quality < baseline.averageQuality - 1f) {
                gaps.add(PerformanceGap(
                    metric = "Sleep Quality",
                    gap = baseline.averageQuality - quality,
                    severity = if (baseline.averageQuality - quality > 2f) "High" else "Medium"
                ))
            }
        }

        if (session.sleepEfficiency < baseline.averageEfficiency - 10f) {
            gaps.add(PerformanceGap(
                metric = "Sleep Efficiency",
                gap = baseline.averageEfficiency - session.sleepEfficiency,
                severity = if (baseline.averageEfficiency - session.sleepEfficiency > 20f) "High" else "Medium"
            ))
        }

        return gaps
    }

    private fun identifyStrengths(session: SleepSession, baseline: PersonalBaseline?): List<CompetitiveAdvantage> {
        if (baseline == null) return emptyList()

        val strengths = mutableListOf<CompetitiveAdvantage>()

        session.sleepQualityScore?.let { quality ->
            if (quality > baseline.averageQuality + 1f) {
                strengths.add(CompetitiveAdvantage(
                    metric = "Sleep Quality",
                    advantage = quality - baseline.averageQuality,
                    strength = if (quality - baseline.averageQuality > 2f) "High" else "Medium"
                ))
            }
        }

        if (session.sleepEfficiency > baseline.averageEfficiency + 10f) {
            strengths.add(CompetitiveAdvantage(
                metric = "Sleep Efficiency",
                advantage = session.sleepEfficiency - baseline.averageEfficiency,
                strength = if (session.sleepEfficiency - baseline.averageEfficiency > 20f) "High" else "Medium"
            ))
        }

        return strengths
    }

    private fun identifyImprovementOpportunities(session: SleepSession, baseline: PersonalBaseline?): List<ImprovementOpportunity> {
        val opportunities = mutableListOf<ImprovementOpportunity>()

        // Always include basic improvement opportunities
        if ((session.sleepQualityScore ?: 0f) < 8f) {
            opportunities.add(ImprovementOpportunity(
                area = "Sleep Quality Enhancement",
                priority = Priority.MEDIUM,
                potentialGain = 8f - (session.sleepQualityScore ?: 0f)
            ))
        }

        if (session.sleepEfficiency < 85f) {
            opportunities.add(ImprovementOpportunity(
                area = "Sleep Efficiency Optimization",
                priority = Priority.MEDIUM,
                potentialGain = 85f - session.sleepEfficiency
            ))
        }

        return opportunities
    }

    // Streak calculation helper methods
    private fun calculateQualityStreak(sessions: List<SleepSession>): Int {
        var streak = 0
        for (session in sessions.reversed()) {
            if ((session.sleepQualityScore ?: 0f) >= 7f) {
                streak++
            } else {
                break
            }
        }
        return streak
    }

    private fun calculateDurationStreak(sessions: List<SleepSession>): Int {
        var streak = 0
        for (session in sessions.reversed()) {
            val hours = session.duration / (1000 * 60 * 60)
            if (hours >= 7 && hours <= 9) {
                streak++
            } else {
                break
            }
        }
        return streak
    }

    private fun findBestQualityStreak(sessions: List<SleepSession>): Int {
        var bestStreak = 0
        var currentStreak = 0

        for (session in sessions) {
            if ((session.sleepQualityScore ?: 0f) >= 7f) {
                currentStreak++
                bestStreak = maxOf(bestStreak, currentStreak)
            } else {
                currentStreak = 0
            }
        }

        return bestStreak
    }

    private fun findBestDurationStreak(sessions: List<SleepSession>): Int {
        var bestStreak = 0
        var currentStreak = 0

        for (session in sessions) {
            val hours = session.duration / (1000 * 60 * 60)
            if (hours >= 7 && hours <= 9) {
                currentStreak++
                bestStreak = maxOf(bestStreak, currentStreak)
            } else {
                currentStreak = 0
            }
        }

        return bestStreak
    }

    private fun countBrokenStreaks(sessions: List<SleepSession>): Int {
        var brokenStreaks = 0
        var inStreak = false

        for (session in sessions) {
            val isGoodQuality = (session.sleepQualityScore ?: 0f) >= 7f

            if (inStreak && !isGoodQuality) {
                brokenStreaks++
                inStreak = false
            } else if (!inStreak && isGoodQuality) {
                inStreak = true
            }
        }

        return brokenStreaks
    }

    private fun calculateRecentStreakTrend(sessions: List<SleepSession>): Float {
        val recentSessions = sessions.takeLast(14) // Last 2 weeks
        val firstWeek = recentSessions.take(7)
        val secondWeek = recentSessions.drop(7)

        val firstWeekStreak = calculateQualityStreak(firstWeek)
        val secondWeekStreak = calculateQualityStreak(secondWeek)

        return secondWeekStreak.toFloat() - firstWeekStreak.toFloat()
    }

    // ========== KEEP ALL EXISTING HELPER METHODS ==========
    [All the existing helper methods from the original SleepAnalyzer.kt remain unchanged]

    private fun calculateDurationScore(duration: Long): Float {
        val hours = duration / (1000f * 60f * 60f)
        return when {
            hours < 5f -> 1f + (hours / 5f) * 2f // 1-3 for very short sleep
            hours < 6f -> 3f + ((hours - 5f) * 2f) // 3-5 for short sleep
            hours < 7f -> 5f + ((hours - 6f) * 2f) // 5-7 for approaching optimal
            hours <= 9f -> 8f + (2f - abs(hours - 8f)) // 8-10 for optimal range
            hours <= 10f -> 8f - ((hours - 9f) * 2f) // 8-6 for slightly long
            else -> max(1f, 6f - ((hours - 10f) * 0.5f)) // Decreasing for very long
        }.coerceIn(1f, 10f)
    }

    private fun calculateMovementScore(movements: List<MovementEvent>, duration: Long): Float {
        if (movements.isEmpty()) return 10f

        val avgIntensity = movements.map { it.intensity }.average().toFloat()
        val movementRate = movements.size.toFloat() / (duration / (60 * 60 * 1000f)) // per hour

        val intensityScore = when {
            avgIntensity <= LOW_MOVEMENT_THRESHOLD -> 10f
            avgIntensity <= (LOW_MOVEMENT_THRESHOLD + HIGH_MOVEMENT_THRESHOLD) / 2 -> 7f
            avgIntensity <= HIGH_MOVEMENT_THRESHOLD -> 4f
            else -> 1f
        }

        val frequencyScore = when {
            movementRate <= 10f -> 10f
            movementRate <= 20f -> 7f
            movementRate <= 40f -> 4f
            else -> 1f
        }

        return (intensityScore + frequencyScore) / 2f
    }

    private fun calculateNoiseScore(noises: List<NoiseEvent>): Float {
        if (noises.isEmpty()) return 10f

        val avgDecibel = noises.map { it.decibelLevel }.average().toFloat()
        val disruptiveCount = noises.count { it.isDisruptive() }
        val disruptionRatio = disruptiveCount.toFloat() / noises.size

        val levelScore = when {
            avgDecibel <= LOW_NOISE_THRESHOLD -> 10f
            avgDecibel <= (LOW_NOISE_THRESHOLD + HIGH_NOISE_THRESHOLD) / 2 -> 7f
            avgDecibel <= HIGH_NOISE_THRESHOLD -> 4f
            else -> 1f
        }

        val disruptionScore = (10f * (1f - disruptionRatio)).coerceAtLeast(1f)

        return (levelScore + disruptionScore) / 2f
    }

    private fun calculateConsistencyScore(session: SleepSession): Float {
        // For individual session, score based on phase transitions and stability
        val transitions = session.phaseTransitions
        if (transitions.isEmpty()) return 5f // Neutral if no data

        // Fewer transitions generally indicate more stable sleep
        val transitionScore = when {
            transitions.size <= 5 -> 10f
            transitions.size <= 10 -> 8f
            transitions.size <= 15 -> 6f
            transitions.size <= 20 -> 4f
            else -> 2f
        }

        // Score based on time spent in deep sleep (consistency indicator)
        val deepSleepRatio = session.deepSleepDuration.toFloat() / session.duration
        val deepSleepScore = when {
            deepSleepRatio >= 0.20f -> 10f
            deepSleepRatio >= 0.15f -> 8f
            deepSleepRatio >= 0.10f -> 6f
            deepSleepRatio >= 0.05f -> 4f
            else -> 2f
        }

        return (transitionScore + deepSleepScore) / 2f
    }

    private fun calculatePhaseDistributionScore(session: SleepSession): Float {
        val totalDuration = session.duration.toFloat()
        if (totalDuration == 0f) return 5f

        val lightRatio = session.lightSleepDuration / totalDuration
        val deepRatio = session.deepSleepDuration / totalDuration
        val remRatio = session.remSleepDuration / totalDuration
        val awakeRatio = session.awakeDuration / totalDuration

        // Optimal ratios: Light 45-55%, Deep 15-25%, REM 20-25%, Awake <10%
        val lightScore = scoreRatio(lightRatio, 0.45f, 0.55f)
        val deepScore = scoreRatio(deepRatio, 0.15f, 0.25f)
        val remScore = scoreRatio(remRatio, 0.20f, 0.25f)
        val awakeScore = if (awakeRatio <= 0.10f) 10f else max(1f, 10f - (awakeRatio - 0.10f) * 50f)

        return (lightScore + deepScore + remScore + awakeScore) / 4f
    }

    private fun scoreRatio(actual: Float, optimalMin: Float, optimalMax: Float): Float {
        return when {
            actual >= optimalMin && actual <= optimalMax -> 10f
            actual < optimalMin -> 5f + (actual / optimalMin) * 5f
            else -> max(1f, 10f - (actual - optimalMax) * 20f)
        }
    }

    // [Continue with all other existing helper methods...]
    // [The rest of the original SleepAnalyzer methods remain unchanged]
}

// ========== NEW AI DATA PREPARATION DATA CLASSES ==========

data class AIDataSummary(
    val summary: String,
    val format: AIDataFormat,
    val confidence: Float,
    val dataPoints: Int,
    val timeRange: TimeRange? = null,
    val keyMetrics: Map<String, Float> = emptyMap(),
    val insights: List<String> = emptyList(),
    val generatedAt: Long = System.currentTimeMillis()
)

data class AIPromptData(
    val promptType: AIPromptType,
    val personalizationLevel: PersonalizationLevel,
    val coreData: Map<String, Any> = emptyMap(),
    val analyticsData: Map<String, Any> = emptyMap(),
    val contextData: Map<String, Any> = emptyMap(),
    val specificData: Map<String, Any> = emptyMap(),
    val metadata: Map<String, Any> = emptyMap(),
    val confidence: Float = 0f,
    val generatedAt: Long = System.currentTimeMillis()
)

data class AIAnalysisExport(
    val format: AIExportFormat,
    val data: String,
    val sessionCount: Int,
    val dataSize: Int,
    val includesRawData: Boolean = false,
    val includesAnalytics: Boolean = true,
    val includesRecommendations: Boolean = true,
    val confidence: Float = 0f,
    val generatedAt: Long = System.currentTimeMillis(),
    val metadata: Map<String, Any> = emptyMap()
)

// AI-specific analysis data classes
data class SessionQualityAnalysis(
    val overallScore: Float,
    val qualityGrade: String,
    val factorBreakdown: QualityFactorBreakdown,
    val efficiencyAnalysis: EfficiencyAnalysisData,
    val strengthsAndWeaknesses: StrengthsAndWeaknesses,
    val confidence: Float
)

data class QualityFactorBreakdown(
    val durationScore: Float,
    val efficiencyScore: Float,
    val movementScore: Float,
    val noiseScore: Float,
    val consistencyScore: Float,
    val phaseDistributionScore: Float
)

data class EfficiencyAnalysisData(
    val basicEfficiency: Float,
    val adjustedEfficiency: Float,
    val combinedEfficiency: Float,
    val grade: String
)

data class StrengthsAndWeaknesses(
    val strongestFactor: String,
    val weakestFactor: String,
    val improvementAreas: List<String>
)

data class SessionSummary(
    val sessionId: Long,
    val date: Date,
    val duration: Long,
    val formattedDuration: String,
    val qualityScore: Float,
    val efficiency: Float,
    val sleepLatency: Long,
    val phaseBreakdown: PhaseBreakdown,
    val movementSummary: MovementSummary,
    val noiseSummary: NoiseSummary,
    val keyInsights: List<String>
)

data class PhaseBreakdown(
    val lightSleep: Long,
    val deepSleep: Long,
    val remSleep: Long,
    val awakeTime: Long
)

data class MovementSummary(
    val averageIntensity: Float,
    val totalEvents: Int,
    val significantEvents: Int
)

data class NoiseSummary(
    val averageLevel: Float,
    val totalEvents: Int,
    val disruptiveEvents: Int
)

data class PersonalBaseline(
    val averageQuality: Float = 0f,
    val averageDuration: Long = 0L,
    val averageEfficiency: Float = 0f,
    val consistencyScore: Float = 0f,
    val sampleSize: Int = 0,
    val timeRange: TimeRange? = null
)

data class HabitAnalysis(
    val identifiedHabits: List<SleepHabit>,
    val bedtimePattern: BedtimePattern,
    val durationPattern: DurationPattern,
    val qualityPattern: QualityPatternAnalysis,
    val weeklyPattern: WeeklyPatternAnalysis?,
    val habitStrength: Float,
    val habitConfidence: Float
)

data class BedtimePattern(
    val averageBedtime: Double,
    val consistency: Boolean,
    val standardDeviation: Double
)

data class DurationPattern(
    val averageDuration: Double,
    val consistency: Boolean,
    val standardDeviation: Double
)

data class SleepOnsetAnalysis(
    val currentLatency: Long,
    val averageLatency: Long,
    val latencyTrend: TrendDirection,
    val onsetQuality: String,
    val factors: List<String>
)

data class GoalAnalysis(
    val goals: List<String>,
    val currentProgress: GoalProgress,
    val achievementRate: Float,
    val recommendations: List<String>,
    val progressTrend: TrendDirection
)

data class GoalProgress(
    val durationProgress: Float,
    val qualityProgress: Float
)

// Comparative analysis enhanced data classes
data class ComparativeAnalysisResult(
    val comparisonType: ComparisonType,
    val baselineInfo: BaselineInfo,
    val personalComparison: PersonalPerformanceComparison? = null,
    val temporalComparison: TemporalPerformanceComparison? = null,
    val metricComparisons: List<MetricComparison> = emptyList(),
    val rankingAnalysis: RankingAnalysis? = null,
    val percentileAnalysis: PercentileAnalysis? = null,
    val performanceGaps: List<PerformanceGap> = emptyList(),
    val competitiveAdvantages: List<CompetitiveAdvantage> = emptyList(),
    val improvementOpportunities: List<ImprovementOpportunity> = emptyList(),
    val comparisonContext: ComparisonContext,
    val reliabilityMetrics: ComparisonReliabilityMetrics
)

data class PersonalPerformanceComparison(
    val timeframe: String,
    val currentPeriodMetrics: PeriodMetrics,
    val baselinePeriodMetrics: PeriodMetrics,
    val qualityImprovement: Float,
    val durationImprovement: Float,
    val efficiencyImprovement: Float,
    val consistencyImprovement: Float,
    val overallImprovement: Float,
    val qualityPercentile: Float,
    val durationPercentile: Float,
    val efficiencyPercentile: Float,
    val currentStreaks: Map<String, Int>,
    val bestStreaks: Map<String, Int>,
    val streakAnalysis: StreakAnalysis
)

data class PeriodMetrics(
    val averageQuality: Float,
    val averageDuration: Float,
    val averageEfficiency: Float,
    val consistency: Float,
    val sessionCount: Int
)

data class StreakAnalysis(
    val currentStreak: Int,
    val bestStreak: Int,
    val activeStreaks: Int,
    val brokenStreaks: Int,
    val trendDirection: TrendDirection
)

data class TemporalPerformanceComparison(
    val improvement: Float
)

data class MetricComparison(
    val metric: String,
    val currentValue: Float,
    val baselineValue: Float,
    val improvement: Float,
    val percentile: Float
)

data class RankingAnalysis(
    val currentRank: Int,
    val totalSessions: Int
)

data class PercentileAnalysis(
    val percentiles: Map<String, Float>
)

data class PerformanceGap(
    val metric: String,
    val gap: Float,
    val severity: String
)

data class CompetitiveAdvantage(
    val metric: String,
    val advantage: Float,
    val strength: String
)

data class ImprovementOpportunity(
    val area: String,
    val priority: Priority,
    val potentialGain: Float
)

data class BaselineInfo(
    val description: String,
    val date: Long
)

data class ComparisonContext(
    val description: String
)

data class ComparisonReliabilityMetrics(
    val confidence: Float
)

// ========== AI DATA ENUMS ==========

enum class AIDataFormat {
    NARRATIVE,
    STRUCTURED,
    BULLET_POINTS,
    JSON
}

enum class AIPromptType {
    QUALITY_ANALYSIS,
    TREND_ANALYSIS,
    PATTERN_RECOGNITION,
    RECOMMENDATION_ENGINE,
    COMPARATIVE_ANALYSIS,
    COMPREHENSIVE_ANALYSIS
}

enum class AIExportFormat {
    COMPREHENSIVE_JSON,
    SUMMARY_JSON,
    STRUCTURED_TEXT,
    PROMPT_OPTIMIZED,
    MINIMAL_CONTEXT
}

enum class ComparisonType {
    PERSONAL_HISTORICAL,
    POPULATION_NORMS,
    OPTIMAL_STANDARDS,
    PEER_GROUP
}

enum class Priority {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

enum class AnalysisDepth {
    BASIC,
    DETAILED,
    COMPREHENSIVE
}

// Supporting data classes that may be referenced
data class TimeRange(
    val startDate: Long,
    val endDate: Long,
    val description: String = ""
)

// ========== KEEP ALL EXISTING DATA CLASSES AND ENUMS ==========
[All existing data classes from the original SleepAnalyzer.kt remain unchanged]