package com.example.somniai.ui.theme.models

import android.graphics.drawable.Drawable
import androidx.annotation.ColorInt
import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.ui.theme.ChartTheme
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*
import com.example.somniai.data.SessionSummaryDTO
import com.example.somniai.data.DailyTrendData
import com.example.somniai.data.PhaseDistributionData
import com.example.somniai.data.SleepInsight  // From AnalyticsModels.kt
import com.example.somniai.ui.theme.SleepPhase
import com.example.somniai.ui.theme.TrendDirection


/**
 * Enterprise UI Data Models
 *
 * Comprehensive UI-specific data classes that transform complex analytics data
 * into display-ready formats optimized for:
 * - RecyclerView performance with stable IDs and change detection
 * - Chart visualization with themed colors and formatted values
 * - Accessibility with semantic descriptions and labels
 * - Multi-language support with localized formatting
 * - Animation and transition support with metadata
 * - Loading states and error handling
 * - Export and sharing functionality
 */

// ========== MISSING DATA CLASSES FOR MAINACTIVITY ==========

/**
 * Sleep session analytics for MainActivity integration
 */
data class SleepSessionAnalytics(
    val totalDuration: Long,
    val qualityFactors: QualityFactorAnalysis,
    val sleepEfficiency: Float,
    val averageMovementIntensity: Float,
    val averageNoiseLevel: Float
) {
    fun toSessionSummaryDTO(): SessionSummaryDTO {
        return SessionSummaryDTO(
            id = System.currentTimeMillis(),
            startTime = System.currentTimeMillis() - totalDuration,
            endTime = System.currentTimeMillis(),
            totalDuration = totalDuration,
            qualityScore = qualityFactors.overallScore,
            sleepEfficiency = sleepEfficiency,
            totalMovementEvents = 0,
            totalNoiseEvents = 0,
            averageMovementIntensity = averageMovementIntensity,
            averageNoiseLevel = averageNoiseLevel
        )
    }
}

/**
 * Quality factor analysis for MainActivity
 */
data class QualityFactorAnalysis(
    val overallScore: Float,
    val movementScore: Float,
    val noiseScore: Float,
    val durationScore: Float,
    val consistencyScore: Float
)

// ========== ENUMS ==========

enum class InsightCategory {
    QUALITY, DURATION, ENVIRONMENT, MOVEMENT, CONSISTENCY, PATTERN
}

enum class RecommendationCategory {
    ENVIRONMENT, BEHAVIOR, TIMING, HEALTH
}

enum class Priority {
    LOW, MEDIUM, HIGH, CRITICAL
}

enum class ImplementationDifficulty {
    EASY, MODERATE, HARD
}

enum class SortField {
    DATE, QUALITY, DURATION, EFFICIENCY
}

enum class SortDirection {
    ASC, DESC
}


enum class ChartThemeStyle {
    QUALITY_TREND, EFFICIENCY_TREND, MOVEMENT_ANALYSIS, NOISE_ANALYSIS,
    PHASE_DISTRIBUTION, COMPARISON, GENERIC
}

enum class ButtonStyle {
    PRIMARY, SECONDARY, OUTLINE, TEXT, FLOATING_ACTION
}

enum class AnimationPriority {
    LOW, NORMAL, HIGH, CRITICAL
}

// ========== MISSING SUPPORTING CLASSES ==========

/**
 * Session summary DTO referenced throughout the file
 */
data class SessionSummaryDTO(
    val id: Long,
    val startTime: Long,
    val endTime: Long?,
    val totalDuration: Long,
    val qualityScore: Float?,
    val sleepEfficiency: Float,
    val totalMovementEvents: Int = 0,
    val totalNoiseEvents: Int = 0,
    val averageMovementIntensity: Float = 0f,
    val averageNoiseLevel: Float = 0f,
    val isCompleted: Boolean = endTime != null,
    val qualityGrade: String = "C",
    val efficiencyGrade: String = "C",
    val formattedDuration: String = "${totalDuration / (1000 * 60 * 60)}h ${(totalDuration % (1000 * 60 * 60)) / (1000 * 60)}m"
)

/**
 * Daily trend data for charts
 */
data class DailyTrendData(
    val date: Long,
    val formattedDate: String,
    val averageQuality: Float?,
    val averageEfficiency: Float,
    val sessionCount: Int
)

/**
 * Phase distribution data for pie charts
 */
data class PhaseDistributionData(
    val sessionId: Long,
    val deepSleepPercentage: Float,
    val remSleepPercentage: Float,
    val lightSleepPercentage: Float,
    val awakePercentage: Float
)
// ========== SESSION DISPLAY MODELS ==========

/**
 * Comprehensive session display model for RecyclerView items
 */
data class SessionDisplayModel(
    val id: Long,
    val sessionData: SessionSummaryDTO,

    // Display Formatting
    val formattedDate: String,
    val formattedTime: String,
    val formattedDuration: String,
    val formattedQuality: String,
    val formattedEfficiency: String,

    // Visual Indicators
    @ColorInt val qualityColor: Int,
    @ColorInt val efficiencyColor: Int,
    @ColorInt val statusColor: Int,
    @DrawableRes val statusIcon: Int,
    val qualityGrade: String,
    val efficiencyGrade: String,

    // Chart Data
    val miniChartData: MiniChartDisplayData?,
    val trendIndicator: TrendIndicator,

    // Metadata
    val isCompleted: Boolean,
    val isOngoing: Boolean,
    val hasInsights: Boolean,
    val insightCount: Int,
    val canBeCompared: Boolean,

    // Accessibility
    val contentDescription: String,
    val qualityDescription: String,
    val statusDescription: String,

    // Animation Support
    val animationMetadata: AnimationMetadata
) {
    companion object {
        fun fromSessionSummary(
            session: SessionSummaryDTO,
            insights: List<SleepInsight> = emptyList(),
            chartData: List<Float> = emptyList()
        ): SessionDisplayModel {
            val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
            val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())

            val qualityScore = session.qualityScore ?: 0f
            val efficiency = session.sleepEfficiency

            return SessionDisplayModel(
                id = session.id,
                sessionData = session,

                // Format display strings
                formattedDate = dateFormat.format(Date(session.startTime)),
                formattedTime = "${timeFormat.format(Date(session.startTime))} - ${
                    session.endTime?.let { timeFormat.format(Date(it)) } ?: "Ongoing"
                }",
                formattedDuration = session.formattedDuration,
                formattedQuality = String.format("%.1f", qualityScore),
                formattedEfficiency = "${efficiency.toInt()}%",

                // Assign theme colors
                qualityColor = ChartTheme.getQualityColor(qualityScore),
                efficiencyColor = ChartTheme.getEfficiencyColor(efficiency),
                statusColor = if (session.isCompleted) ChartTheme.getQualityColor(8f) else ChartTheme.getQualityColor(4f),
                statusIcon = if (session.isCompleted) android.R.drawable.ic_dialog_info else android.R.drawable.ic_dialog_alert,
                qualityGrade = session.qualityGrade,
                efficiencyGrade = session.efficiencyGrade,

                // Generate chart data
                miniChartData = if (chartData.isNotEmpty()) {
                    MiniChartDisplayData.fromValues(chartData, qualityScore)
                } else null,
                trendIndicator = TrendIndicator.fromQualityScore(qualityScore),

                // Set metadata flags
                isCompleted = session.isCompleted,
                isOngoing = !session.isCompleted,
                hasInsights = insights.isNotEmpty(),
                insightCount = insights.size,
                canBeCompared = session.isCompleted && qualityScore > 0f,

                // Generate accessibility strings
                contentDescription = buildContentDescription(session),
                qualityDescription = buildQualityDescription(qualityScore, session.qualityGrade),
                statusDescription = if (session.isCompleted) "Session completed" else "Session ongoing",

                // Animation metadata
                animationMetadata = AnimationMetadata(
                    itemType = "session",
                    priority = if (insights.isNotEmpty()) AnimationPriority.HIGH else AnimationPriority.NORMAL,
                    hasStateChange = !session.isCompleted
                )
            )
        }

        private fun buildContentDescription(session: SessionSummaryDTO): String {
            val dateFormat = SimpleDateFormat("EEEE, MMMM dd, yyyy", Locale.getDefault())
            return buildString {
                append("Sleep session from ${dateFormat.format(Date(session.startTime))}. ")
                append("Duration: ${session.formattedDuration}. ")
                append("Quality score: ${String.format("%.1f", session.qualityScore ?: 0f)} out of 10. ")
                append("Sleep efficiency: ${session.sleepEfficiency.toInt()} percent. ")
                if (session.isCompleted) append("Session completed.") else append("Session ongoing.")
            }
        }

        private fun buildQualityDescription(score: Float, grade: String): String {
            return when {
                score >= 8f -> "Excellent sleep quality - $grade grade"
                score >= 6f -> "Good sleep quality - $grade grade"
                score >= 4f -> "Fair sleep quality - $grade grade"
                else -> "Poor sleep quality - $grade grade"
            }
        }
    }
}

/**
 * Detailed session model for expanded views
 */
data class SessionDetailDisplayModel(
    val basicInfo: SessionDisplayModel,

    // Detailed Analytics
    val qualityBreakdown: QualityBreakdownDisplay,
    val phaseAnalysis: PhaseAnalysisDisplay,
    val activityAnalysis: ActivityAnalysisDisplay,
    val comparisonMetrics: ComparisonMetricsDisplay,

    // Charts and Visualizations
    val detailCharts: List<ChartDisplayModel>,
    val trendAnalysis: TrendAnalysisDisplay?,

    // Insights and Recommendations
    val insights: List<InsightDisplayModel>,
    val recommendations: List<RecommendationDisplayModel>,
    val achievements: List<AchievementDisplayModel>,

    // Export and Sharing
    val exportOptions: List<ExportOptionDisplay>,
    val shareableContent: ShareableContentDisplay
)

// ========== CHART DISPLAY MODELS ==========

/**
 * Generic chart display model for all chart types
 */
data class ChartDisplayModel(
    val id: String,
    val title: String,
    val subtitle: String?,
    val chartType: String,

    val type: String = "",

    // Data
    val dataPoints: List<ChartDataPoint>,
    val labels: List<String>,
    val values: List<Float>,

    // Styling
    @ColorInt val primaryColor: Int,
    @ColorInt val secondaryColor: Int,
    val gradientColors: List<Int>?,
    val themeStyle: ChartThemeStyle,

    val strokeColor: Int = 0,

    // Configuration
    val showLegend: Boolean,
    val showGrid: Boolean,
    val showValues: Boolean,
    val isInteractive: Boolean,
    val animationDuration: Int,

    // Formatters
    val valueFormatter: String, // Formatter identifier
    val axisFormatter: String?,

    // Accessibility
    val accessibilityLabel: String,
    val dataDescription: String,

    // Metadata
    val isEmpty: Boolean,
    val hasError: Boolean,
    val errorMessage: String?,
    val lastUpdated: Long
) {
    companion object {
        fun createQualityTrendChart(
            data: List<DailyTrendData>,
            timeRange: String = "Last 30 days"
        ): ChartDisplayModel {
            val dataPoints = data.mapIndexed { index, trend ->
                ChartDataPoint(
                    x = index.toFloat(),
                    y = trend.averageQuality ?: 0f,
                    label = trend.formattedDate,
                    timestamp = trend.date,
                    metadata = mapOf(
                        "sessionCount" to trend.sessionCount,
                        "efficiency" to trend.averageEfficiency
                    )
                )
            }

            return ChartDisplayModel(
                id = "quality_trend_${System.currentTimeMillis()}",
                title = "Sleep Quality Trend",
                subtitle = timeRange,
                chartType = "LINE",
                dataPoints = dataPoints,
                labels = data.map { it.formattedDate },
                values = data.map { it.averageQuality ?: 0f },
                primaryColor = ChartTheme.getQualityColor(7f),
                secondaryColor = ChartTheme.getQualityColor(8f),
                gradientColors = listOf(
                    ChartTheme.getQualityColor(8f),
                    ChartTheme.getQualityColor(6f)
                ),
                themeStyle = ChartThemeStyle.QUALITY_TREND,
                showLegend = false,
                showGrid = true,
                showValues = false,
                isInteractive = true,
                animationDuration = 800,
                valueFormatter = "quality",
                axisFormatter = "date",
                accessibilityLabel = "Sleep quality trend chart showing $timeRange",
                dataDescription = generateQualityTrendDescription(data),
                isEmpty = data.isEmpty(),
                hasError = false,
                errorMessage = null,
                lastUpdated = System.currentTimeMillis()
            )
        }

        fun createPhaseDistributionChart(
            phaseData: PhaseDistributionData
        ): ChartDisplayModel {
            val phases = listOf(
                Triple("Deep Sleep", phaseData.deepSleepPercentage, ChartTheme.getQualityColor(8f)),
                Triple("REM Sleep", phaseData.remSleepPercentage, ChartTheme.getQualityColor(7f)),
                Triple("Light Sleep", phaseData.lightSleepPercentage, ChartTheme.getQualityColor(6f)),
                Triple("Awake", phaseData.awakePercentage, ChartTheme.getQualityColor(4f))
            ).filter { it.second > 0f }

            val dataPoints = phases.mapIndexed { index, (label, percentage, color) ->
                ChartDataPoint(
                    x = index.toFloat(),
                    y = percentage,
                    label = label,
                    timestamp = 0L,
                    metadata = mapOf("color" to color, "duration" to (percentage * 8 / 100))
                )
            }

            return ChartDisplayModel(
                id = "phase_distribution_${phaseData.sessionId}",
                title = "Sleep Phase Distribution",
                subtitle = null,
                chartType = "PIE",
                dataPoints = dataPoints,
                labels = phases.map { it.first },
                values = phases.map { it.second },
                primaryColor = ChartTheme.getQualityColor(8f),
                secondaryColor = ChartTheme.getQualityColor(7f),
                gradientColors = null,
                themeStyle = ChartThemeStyle.PHASE_DISTRIBUTION,
                showLegend = true,
                showGrid = false,
                showValues = true,
                isInteractive = true,
                animationDuration = 1000,
                valueFormatter = "percentage",
                axisFormatter = null,
                accessibilityLabel = "Sleep phase distribution pie chart",
                dataDescription = generatePhaseDistributionDescription(phaseData),
                isEmpty = phases.isEmpty(),
                hasError = false,
                errorMessage = null,
                lastUpdated = System.currentTimeMillis()
            )
        }

        private fun generateQualityTrendDescription(data: List<DailyTrendData>): String {
            val avgQuality = data.mapNotNull { it.averageQuality }.average()
            val trend = if (data.size >= 2) {
                val recent = data.takeLast(7).mapNotNull { it.averageQuality }.average()
                val older = data.dropLast(7).mapNotNull { it.averageQuality }.average()
                when {
                    recent > older + 0.5 -> "improving"
                    recent < older - 0.5 -> "declining"
                    else -> "stable"
                }
            } else "insufficient data"

            return "Average quality: ${String.format("%.1f", avgQuality)} out of 10. Trend: $trend."
        }

        private fun generatePhaseDistributionDescription(data: PhaseDistributionData): String {
            return buildString {
                append("Sleep phases: ")
                append("${data.deepSleepPercentage.toInt()}% deep sleep, ")
                append("${data.remSleepPercentage.toInt()}% REM sleep, ")
                append("${data.lightSleepPercentage.toInt()}% light sleep, ")
                append("${data.awakePercentage.toInt()}% awake time.")
            }
        }
    }
}

/**
 * Mini chart data for session list items
 */
data class MiniChartDisplayData(
    val sparklineData: List<Float>,
    val trendDirection: TrendDirection,
    @ColorInt val trendColor: Int,
    val formattedTrend: String,
    val accessibilityDescription: String,
    val type: String = "sparkline",
    val strokeColor: Int = 0
) {
    companion object {
        fun fromValues(values: List<Float>, currentValue: Float): MiniChartDisplayData {
            val trend = if (values.size >= 2) {
                val recent = values.takeLast(3).average()
                val older = values.dropLast(3).average()
                when {
                    recent > older + 0.3 -> TrendDirection.IMPROVING
                    recent < older - 0.3 -> TrendDirection.DECLINING
                    else -> TrendDirection.STABLE
                }
            } else TrendDirection.INSUFFICIENT_DATA

            return MiniChartDisplayData(
                sparklineData = values,
                trendDirection = trend,
                trendColor = ChartTheme.getTrendColor(trend),
                formattedTrend = when (trend) {
                    TrendDirection.IMPROVING -> "↗ Improving"
                    TrendDirection.DECLINING -> "↘ Declining"
                    TrendDirection.STABLE -> "→ Stable"
                    else -> "— No trend"
                },
                accessibilityDescription = "Quality trend: ${trend.name.lowercase()}"
            )
        }
    }
}

// ========== ANALYTICS DISPLAY MODELS ==========

/**
 * Quality breakdown for detailed session view
 */
data class QualityBreakdownDisplay(
    val overallScore: Float,
    val overallGrade: String,
    @ColorInt val overallColor: Int,

    val factors: List<QualityFactorDisplay>,
    val strengths: List<String>,
    val improvements: List<String>,

    val comparisonToAverage: Float, // Difference from personal average
    val comparisonText: String,
    @ColorInt val comparisonColor: Int,

    val accessibilityDescription: String
)

/**
 * Individual quality factor display
 */
data class QualityFactorDisplay(
    val name: String,
    val score: Float,
    val maxScore: Float,
    val percentage: Int,
    @ColorInt val color: Int,
    val grade: String,
    val description: String,
    val trend: TrendIndicator?,
    val isStrength: Boolean,
    val needsImprovement: Boolean
)

/**
 * Phase analysis display model
 */
data class PhaseAnalysisDisplay(
    val phases: List<PhaseDisplay>,
    val totalSleepTime: String,
    val actualSleepTime: String,
    val sleepEfficiency: Float,
    val sleepOnset: String,
    val wakeUpCount: Int,
    val restlessnessScore: Float,
    val phaseBalance: PhaseBalanceDisplay,
    val insights: List<String>
)

/**
 * Individual sleep phase display
 */
data class PhaseDisplay(
    val phase: SleepPhase,
    val duration: String,
    val percentage: Float,
    @ColorInt val color: Int,
    val isOptimal: Boolean,
    val recommendation: String?
)

/**
 * Phase balance analysis
 */
data class PhaseBalanceDisplay(
    val score: Float,
    val grade: String,
    @ColorInt val color: Int,
    val description: String,
    val isHealthy: Boolean
)

/**
 * Activity analysis display
 */
data class ActivityAnalysisDisplay(
    val movementAnalysis: MovementAnalysisDisplay,
    val noiseAnalysis: NoiseAnalysisDisplay,
    val restlessnessAnalysis: RestlessnessAnalysisDisplay,
    val environmentalScore: Float
)

/**
 * Movement analysis display
 */
data class MovementAnalysisDisplay(
    val totalEvents: Int,
    val significantEvents: Int,
    val averageIntensity: Float,
    val maxIntensity: Float,
    val intensityLevel: String,
    @ColorInt val intensityColor: Int,
    val movementRate: String, // Events per hour
    val restlessnessScore: Float,
    val chartData: List<Float>,
    val insights: List<String>
)

/**
 * Noise analysis display
 */
data class NoiseAnalysisDisplay(
    val totalEvents: Int,
    val disruptiveEvents: Int,
    val averageLevel: Float,
    val maxLevel: Float,
    val noiseLevel: String,
    @ColorInt val noiseColor: Int,
    val disruptionRate: String,
    val environmentScore: Float,
    val chartData: List<Float>,
    val insights: List<String>
)

/**
 * Restlessness analysis display
 */
data class RestlessnessAnalysisDisplay(
    val score: Float,
    val level: String,
    @ColorInt val color: Int,
    val description: String,
    val factors: List<String>,
    val recommendations: List<String>
)

/**
 * Comparison metrics display
 */
data class ComparisonMetricsDisplay(
    val personalComparison: PersonalComparisonDisplay,
    val historicalComparison: HistoricalComparisonDisplay,
    val benchmarkComparison: BenchmarkComparisonDisplay?,
    val rankingInfo: RankingInfoDisplay
)

/**
 * Personal comparison display
 */
data class PersonalComparisonDisplay(
    val vsAverage: ComparisonItemDisplay,
    val vsBest: ComparisonItemDisplay,
    val vsWorst: ComparisonItemDisplay,
    val percentileRank: Float,
    val performanceLevel: String,
    @ColorInt val performanceColor: Int
)

/**
 * Individual comparison item
 */
data class ComparisonItemDisplay(
    val label: String,
    val difference: Float,
    val differenceText: String,
    val isBetter: Boolean,
    @ColorInt val color: Int,
    val icon: String // Unicode arrow or symbol
)

/**
 * Historical comparison display
 */
data class HistoricalComparisonDisplay(
    val weekComparison: ComparisonItemDisplay,
    val monthComparison: ComparisonItemDisplay,
    val yearComparison: ComparisonItemDisplay,
    val allTimeComparison: ComparisonItemDisplay,
    val streak: StreakDisplay
)

/**
 * Benchmark comparison display
 */
data class BenchmarkComparisonDisplay(
    val populationRank: Float,
    val ageGroupRank: Float,
    val generalRecommendation: String,
    val achievements: List<String>
)

/**
 * Ranking information display
 */
data class RankingInfoDisplay(
    val currentRank: Int,
    val totalSessions: Int,
    val topPercentile: Float,
    val rankingTrend: TrendDirection,
    val badgeLevel: String,
    @DrawableRes val badgeIcon: Int
)

/**
 * Streak information display
 */
data class StreakDisplay(
    val currentStreak: Int,
    val bestStreak: Int,
    val streakType: String, // "quality", "consistency", etc.
    val isActive: Boolean,
    val nextMilestone: Int,
    val description: String
)

// ========== TREND ANALYSIS DISPLAY ==========

/**
 * Trend analysis display model
 */
data class TrendAnalysisDisplay(
    val overallTrend: TrendItemDisplay,
    val qualityTrend: TrendItemDisplay,
    val durationTrend: TrendItemDisplay,
    val efficiencyTrend: TrendItemDisplay,
    val consistencyTrend: TrendItemDisplay,

    val seasonalPatterns: List<SeasonalPatternDisplay>,
    val projections: List<ProjectionDisplay>,
    val insights: List<TrendInsightDisplay>,

    val confidence: Float,
    val reliability: String,
    val dataQuality: String
)

/**
 * Individual trend item display
 */
data class TrendItemDisplay(
    val name: String,
    val direction: TrendDirection,
    val strength: Float,
    val changeRate: Float,
    val changeText: String,
    @ColorInt val color: Int,
    val icon: String,
    val description: String,
    val significance: String
)

/**
 * Seasonal pattern display
 */
data class SeasonalPatternDisplay(
    val pattern: String,
    val strength: Float,
    val description: String,
    val bestPeriod: String,
    val worstPeriod: String
)

/**
 * Trend projection display
 */
data class ProjectionDisplay(
    val timeframe: String,
    val projectedValue: Float,
    val confidence: Float,
    val direction: TrendDirection,
    val description: String,
    @ColorInt val color: Int
)

/**
 * Trend insight display
 */
data class TrendInsightDisplay(
    val title: String,
    val description: String,
    val significance: Float,
    val actionable: Boolean,
    val recommendation: String?,
    @ColorInt val priorityColor: Int
)

// ========== INSIGHT AND RECOMMENDATION MODELS ==========

/**
 * Insight display model
 */
data class InsightDisplayModel(
    val id: String,
    val title: String,
    val description: String,
    val category: InsightCategory,
    val priority: Priority,
    @ColorInt val priorityColor: Int,
    @DrawableRes val categoryIcon: Int,
    val confidence: Float,
    val confidenceText: String,
    val isActionable: Boolean,
    val isNew: Boolean,
    val timestamp: String,
    val accessibilityLabel: String
)

/**
 * Recommendation display model
 */
data class RecommendationDisplayModel(
    val id: String,
    val title: String,
    val description: String,
    val category: RecommendationCategory,
    val priority: Priority,
    @ColorInt val priorityColor: Int,
    @DrawableRes val categoryIcon: Int,
    val difficulty: ImplementationDifficulty,
    val difficultyText: String,
    val expectedImpact: String,
    val timeToSee: String,
    val actionItems: List<ActionItemDisplay>,
    val isCompleted: Boolean,
    val completionDate: Long?
)

/**
 * Action item display
 */
data class ActionItemDisplay(
    val title: String,
    val description: String,
    val difficulty: String,
    val timeRequired: String,
    val isCompleted: Boolean
)

/**
 * Achievement display model
 */
data class AchievementDisplayModel(
    val id: String,
    val title: String,
    val description: String,
    val category: String,
    @DrawableRes val icon: Int,
    @ColorInt val badgeColor: Int,
    val level: Int,
    val progress: Float,
    val nextLevel: String?,
    val unlockedDate: Long,
    val isNew: Boolean,
    val rarity: String
)

// ========== FILTER AND SORT MODELS ==========

/**
 * Filter option display model
 */
data class FilterOptionDisplay(
    val id: String,
    val title: String,
    val description: String,
    @DrawableRes val icon: Int,
    val isSelected: Boolean,
    val count: Int,
    val category: String,
    val quickFilter: Boolean = false
)

/**
 * Sort option display model
 */
data class SortOptionDisplay(
    val id: String,
    val title: String,
    val field: SortField,
    val direction: SortDirection,
    @DrawableRes val icon: Int,
    val isSelected: Boolean,
    val description: String
)

/**
 * Filter group display model
 */
data class FilterGroupDisplay(
    val title: String,
    val options: List<FilterOptionDisplay>,
    val multiSelect: Boolean,
    val isExpanded: Boolean
)

// ========== LOADING AND ERROR MODELS ==========

/**
 * Loading state display model
 */
data class LoadingStateDisplay(
    val isLoading: Boolean,
    val loadingText: String,
    val progress: Float? = null,
    val showProgress: Boolean = false,
    val canCancel: Boolean = false
)

/**
 * Error state display model
 */
data class ErrorStateDisplay(
    val title: String,
    val message: String,
    val details: String? = null,
    @DrawableRes val icon: Int,
    val canRetry: Boolean = false,
    val retryText: String = "Retry",
    val canDismiss: Boolean = true,
    val actionItems: List<ErrorActionDisplay> = emptyList()
)

/**
 * Error action display
 */
data class ErrorActionDisplay(
    val title: String,
    val description: String,
    val action: String, // Action identifier
    @DrawableRes val icon: Int
)

// ========== EXPORT AND SHARING MODELS ==========

/**
 * Export option display model
 */
data class ExportOptionDisplay(
    val id: String,
    val title: String,
    val description: String,
    val format: String,
    @DrawableRes val icon: Int,
    val fileSize: String?,
    val isRecommended: Boolean = false
)

/**
 * Shareable content display model
 */
data class ShareableContentDisplay(
    val title: String,
    val summary: String,
    val detailedText: String,
    val hashtags: List<String>,
    val imageUrl: String? = null,
    val shareUrl: String? = null
)

// ========== NAVIGATION AND INTERACTION MODELS ==========

/**
 * Navigation item display model
 */
data class NavigationItemDisplay(
    val id: String,
    val title: String,
    val subtitle: String? = null,
    @DrawableRes val icon: Int,
    val badge: BadgeDisplay? = null,
    val isEnabled: Boolean = true,
    val hasNotification: Boolean = false
)

/**
 * Badge display model
 */
data class BadgeDisplay(
    val text: String,
    @ColorInt val backgroundColor: Int,
    @ColorInt val textColor: Int,
    val isVisible: Boolean = true
)

/**
 * Action button display model
 */
data class ActionButtonDisplay(
    val id: String,
    val title: String,
    val description: String? = null,
    @DrawableRes val icon: Int,
    @ColorInt val backgroundColor: Int,
    @ColorInt val textColor: Int,
    val isEnabled: Boolean = true,
    val isLoading: Boolean = false,
    val style: ButtonStyle = ButtonStyle.PRIMARY
)

// ========== SUMMARY AND STATISTICS MODELS ==========

/**
 * Statistics summary display model
 */
data class StatisticsSummaryDisplay(
    val totalSessions: String,
    val averageDuration: String,
    val averageQuality: String,
    val averageEfficiency: String,
    val currentStreak: String,
    val bestStreak: String,
    val totalSleepTime: String,
    val improvementRate: String,

    val keyMetrics: List<KeyMetricDisplay>,
    val achievements: List<AchievementSummaryDisplay>,
    val trends: List<TrendSummaryDisplay>
)

/**
 * Key metric display model
 */
data class KeyMetricDisplay(
    val name: String,
    val value: String,
    val unit: String? = null,
    val change: String? = null,
    val changeDirection: TrendDirection? = null,
    @ColorInt val color: Int,
    @DrawableRes val icon: Int,
    val description: String
)

/**
 * Achievement summary display
 */
data class AchievementSummaryDisplay(
    val title: String,
    val description: String,
    @DrawableRes val icon: Int,
    val isRecent: Boolean = false
)

/**
 * Trend summary display
 */
data class TrendSummaryDisplay(
    val metric: String,
    val direction: TrendDirection,
    val changeText: String,
    @ColorInt val color: Int,
    val description: String
)

// ========== HELPER MODELS ==========

/**
 * Chart data point for all chart types
 */
data class ChartDataPoint(
    val x: Float,
    val y: Float,
    val label: String = "",
    val timestamp: Long = 0L,
    val type: String = "",  // ADD THIS LINE
    val metadata: Map<String, Any> = emptyMap()
)

/**
 * Trend indicator for mini displays
 */
data class TrendIndicator(
    val direction: TrendDirection,
    val icon: String,
    @ColorInt val color: Int,
    val text: String
) {
    companion object {
        fun fromQualityScore(score: Float): TrendIndicator {
            return when {
                score >= 8f -> TrendIndicator(
                    TrendDirection.IMPROVING,
                    "↗",
                    ChartTheme.getQualityColor(8f),
                    "Excellent"
                )
                score >= 6f -> TrendIndicator(
                    TrendDirection.STABLE,
                    "→",
                    ChartTheme.getQualityColor(6f),
                    "Good"
                )
                else -> TrendIndicator(
                    TrendDirection.DECLINING,
                    "↘",
                    ChartTheme.getQualityColor(3f),
                    "Needs Work"
                )
            }
        }
    }
}

/**
 * Animation metadata for smooth transitions
 */
data class AnimationMetadata(
    val itemType: String,
    val priority: AnimationPriority,
    val hasStateChange: Boolean = false,
    val duration: Int = 300,
    val delay: Int = 0
)

// ========== EXTENSION FUNCTIONS ==========

/**
 * Extension functions for easy UI model creation
 */

fun SessionSummaryDTO.toDisplayModel(
    insights: List<SleepInsight> = emptyList(),
    chartData: List<Float> = emptyList()
): SessionDisplayModel {
    return SessionDisplayModel.fromSessionSummary(this, insights, chartData)
}

fun List<DailyTrendData>.toQualityTrendChart(timeRange: String = "Last 30 days"): ChartDisplayModel {
    return ChartDisplayModel.createQualityTrendChart(this, timeRange)
}

fun PhaseDistributionData.toPhaseChart(): ChartDisplayModel {
    return ChartDisplayModel.createPhaseDistributionChart(this)
}

fun TrendDirection.toIndicator(value: Float): TrendIndicator {
    return TrendIndicator(
        direction = this,
        icon = when (this) {
            TrendDirection.IMPROVING, TrendDirection.STRONGLY_IMPROVING -> "↗"
            TrendDirection.DECLINING, TrendDirection.STRONGLY_DECLINING -> "↘"
            TrendDirection.STABLE -> "→"
            else -> "—"
        },
        color = ChartTheme.getTrendColor(this),
        text = this.name
    )
}

/**
 * Create formatted strings for common display patterns
 */
object DisplayFormatters {
    fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        return when {
            hours > 0 -> "${hours}h ${minutes}m"
            minutes > 0 -> "${minutes}m"
            else -> "< 1m"
        }
    }

    fun formatQuality(score: Float): String {
        return String.format("%.1f", score)
    }

    fun formatEfficiency(efficiency: Float): String {
        return "${efficiency.toInt()}%"
    }

    fun formatTrend(change: Float): String {
        val sign = if (change >= 0) "+" else ""
        return "$sign${String.format("%.1f", change)}"
    }

    fun formatConfidence(confidence: Float): String {
        return when {
            confidence >= 0.9f -> "Very High"
            confidence >= 0.8f -> "High"
            confidence >= 0.6f -> "Medium"
            confidence >= 0.4f -> "Low"
            else -> "Very Low"
        }
    }
}