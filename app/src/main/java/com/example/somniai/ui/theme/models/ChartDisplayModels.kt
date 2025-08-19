package com.example.somniai.ui.theme.models

import android.graphics.Color
import android.graphics.Paint
import androidx.annotation.ColorInt
import androidx.annotation.DrawableRes
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.ui.theme.ChartTheme
import com.github.mikephil.charting.components.LimitLine
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.interfaces.datasets.*
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

/**
 * Enterprise Chart Display Models
 *
 * Comprehensive chart-specific models that transform analytics data into
 * visualization-ready formats optimized for:
 * - MPAndroidChart integration with full customization
 * - Sleep analytics visualization with domain-specific styling
 * - Performance optimization with pre-computed chart data
 * - Accessibility with screen reader support and semantic descriptions
 * - Interactive features with touch handling and animations
 * - Export capabilities with high-quality image generation
 * - Real-time updates with smooth data transitions
 * - Multi-chart dashboards with synchronized interactions
 */

// ========== CORE CHART MODELS ==========

/**
 * Comprehensive chart configuration model
 */
data class ChartConfig(
    val id: String,
    val title: String,
    val subtitle: String? = null,
    val chartType: ChartType,
    val analyticsType: AnalyticsChartType,

    // Data Configuration
    val dataSource: ChartDataSource,
    val timeRange: TimeRange? = null,
    val aggregationLevel: AggregationLevel = AggregationLevel.DAY,

    // Visual Configuration
    val themeConfig: ChartThemeConfig,
    val layoutConfig: ChartLayoutConfig,
    val interactionConfig: ChartInteractionConfig,

    // Performance Configuration
    val performanceConfig: ChartPerformanceConfig,

    // Accessibility Configuration
    val accessibilityConfig: ChartAccessibilityConfig,

    // Export Configuration
    val exportConfig: ChartExportConfig,

    // Metadata
    val lastUpdated: Long = System.currentTimeMillis(),
    val isStale: Boolean = false,
    val updateFrequency: UpdateFrequency = UpdateFrequency.MANUAL
) {
    val uniqueKey: String
        get() = "${chartType.name}_${analyticsType.name}_${dataSource.hashCode()}_${timeRange?.hashCode() ?: 0}"
}

/**
 * Chart theme configuration
 */
data class ChartThemeConfig(
    val primaryColor: Int = ChartTheme.Colors.QUALITY_GOOD.toInt(),
    val secondaryColor: Int = ChartTheme.Colors.QUALITY_EXCELLENT.toInt(),
    val backgroundColor: Int = ChartTheme.Colors.BACKGROUND_SECONDARY.toInt(),
    val textColor: Int = ChartTheme.Colors.TEXT_PRIMARY.toInt(),
    val gridColor: Int = ChartTheme.Colors.GRID_PRIMARY.toInt(),
    val gradientColors: List<Int>? = null,
    val useCustomColors: Boolean = false,
    val colorPalette: ColorPalette = ColorPalette.QUALITY_FOCUSED,
    val nightMode: Boolean = true
)

/**
 * Chart layout configuration
 */
data class ChartLayoutConfig(
    val showLegend: Boolean = true,
    val legendPosition: LegendPosition = LegendPosition.BOTTOM,
    val showGrid: Boolean = true,
    val showAxisLabels: Boolean = true,
    val showValues: Boolean = false,
    val showMarkers: Boolean = true,
    val showLimitLines: Boolean = true,
    val margins: ChartMargins = ChartMargins(),
    val aspectRatio: Float? = null,
    val minimumHeight: Float = 200f
)

/**
 * Chart interaction configuration
 */
data class ChartInteractionConfig(
    val isInteractive: Boolean = true,
    val enableZoom: Boolean = false,
    val enablePan: Boolean = false,
    val enableHighlight: Boolean = true,
    val enableSelection: Boolean = true,
    val touchGestureThreshold: Float = 10f,
    val longPressEnabled: Boolean = true,
    val doubleClickEnabled: Boolean = false,
    val multiSelectEnabled: Boolean = false
)

/**
 * Chart performance configuration
 */
data class ChartPerformanceConfig(
    val maxDataPoints: Int = 100,
    val animationDuration: Int = 800,
    val enableAnimations: Boolean = true,
    val useHardwareAcceleration: Boolean = true,
    val cacheEnabled: Boolean = true,
    val precomputeValues: Boolean = true,
    val memoryOptimized: Boolean = true
)

/**
 * Chart accessibility configuration
 */
data class ChartAccessibilityConfig(
    val enableAccessibility: Boolean = true,
    val screenReaderSupport: Boolean = true,
    val highContrastMode: Boolean = false,
    val largeFontMode: Boolean = false,
    val colorBlindSupport: Boolean = true,
    val audioDescriptionEnabled: Boolean = false,
    val hapticFeedbackEnabled: Boolean = false
)

/**
 * Chart export configuration
 */
data class ChartExportConfig(
    val enableExport: Boolean = true,
    val supportedFormats: List<ExportFormat> = listOf(ExportFormat.PNG, ExportFormat.PDF),
    val defaultFormat: ExportFormat = ExportFormat.PNG,
    val exportQuality: ExportQuality = ExportQuality.HIGH,
    val includeMetadata: Boolean = true,
    val includeWatermark: Boolean = false
)

// ========== SPECIALIZED CHART MODELS ==========

/**
 * Sleep quality trend chart model
 */
data class QualityTrendChartModel(
    val config: ChartConfig,
    val trendData: List<QualityTrendPoint>,
    val benchmarkLines: List<BenchmarkLine>,
    val annotations: List<ChartAnnotation>,
    val statisticalData: TrendStatistics,
    val comparisonData: ComparisonTrendData? = null
) {

    fun toLineDataSet(): LineDataSet {
        val entries = trendData.mapIndexed { index, point ->
            Entry(index.toFloat(), point.qualityScore, point)
        }

        val dataSet = LineDataSet(entries, "Sleep Quality")
        return ChartTheme.styleQualityTrendDataSet(dataSet)
    }

    fun createAccessibilityDescription(): String {
        val avgQuality = trendData.map { it.qualityScore }.average()
        val trend = statisticalData.overallTrend
        val dataPoints = trendData.size

        return buildString {
            append("Sleep quality trend chart with $dataPoints data points. ")
            append("Average quality: ${String.format("%.1f", avgQuality)} out of 10. ")
            append("Overall trend: ${trend.displayName.lowercase()}. ")
            if (benchmarkLines.isNotEmpty()) {
                append("Chart includes ${benchmarkLines.size} benchmark lines for reference. ")
            }
        }
    }

    companion object {
        fun fromTrendAnalysis(
            trendAnalysis: TrendAnalysisResult,
            sessions: List<SessionSummaryDTO>
        ): QualityTrendChartModel {
            val trendData = sessions.mapIndexed { index, session ->
                QualityTrendPoint(
                    timestamp = session.startTime,
                    qualityScore = session.qualityScore ?: 0f,
                    sessionId = session.id,
                    confidence = 0.8f, // Would come from analytics
                    metadata = mapOf(
                        "duration" to session.totalDuration,
                        "efficiency" to session.sleepEfficiency
                    )
                )
            }

            val benchmarkLines = listOf(
                BenchmarkLine(
                    value = 8f,
                    label = "Excellent",
                    color = ChartTheme.Colors.QUALITY_EXCELLENT.toInt(),
                    style = LineStyle.SOLID
                ),
                BenchmarkLine(
                    value = 6f,
                    label = "Good",
                    color = ChartTheme.Colors.QUALITY_GOOD.toInt(),
                    style = LineStyle.DASHED
                )
            )

            return QualityTrendChartModel(
                config = ChartConfig(
                    id = "quality_trend_${System.currentTimeMillis()}",
                    title = "Sleep Quality Trend",
                    chartType = ChartType.LINE,
                    analyticsType = AnalyticsChartType.QUALITY_TREND,
                    dataSource = ChartDataSource.QUALITY_ANALYSIS,
                    themeConfig = ChartThemeConfig(),
                    layoutConfig = ChartLayoutConfig(),
                    interactionConfig = ChartInteractionConfig(),
                    performanceConfig = ChartPerformanceConfig(),
                    accessibilityConfig = ChartAccessibilityConfig(),
                    exportConfig = ChartExportConfig()
                ),
                trendData = trendData,
                benchmarkLines = benchmarkLines,
                annotations = emptyList(),
                statisticalData = TrendStatistics.fromTrendAnalysis(trendAnalysis)
            )
        }
    }
}

/**
 * Sleep efficiency chart model
 */
data class EfficiencyChartModel(
    val config: ChartConfig,
    val efficiencyData: List<EfficiencyDataPoint>,
    val targetLines: List<TargetLine>,
    val correlationData: CorrelationData? = null,
    val performanceMetrics: EfficiencyMetrics
) {

    fun toLineDataSet(): LineDataSet {
        val entries = efficiencyData.mapIndexed { index, point ->
            Entry(index.toFloat(), point.efficiency, point)
        }

        val dataSet = LineDataSet(entries, "Sleep Efficiency")
        return ChartTheme.styleEfficiencyTrendDataSet(dataSet)
    }

    fun createTargetLimitLines(): List<LimitLine> {
        return targetLines.map { target ->
            LimitLine(target.value, target.label).apply {
                lineColor = target.color
                lineWidth = target.lineWidth
                textColor = target.color
                textSize = ChartTheme.Typography.ANNOTATION_SIZE
            }
        }
    }
}

/**
 * Movement analysis chart model
 */
data class MovementAnalysisChartModel(
    val config: ChartConfig,
    val movementData: List<MovementDataPoint>,
    val intensityThresholds: List<IntensityThreshold>,
    val patternAnalysis: MovementPatternAnalysis,
    val correlations: MovementCorrelationData
) {

    fun toLineDataSet(): LineDataSet {
        val entries = movementData.mapIndexed { index, point ->
            Entry(index.toFloat(), point.intensity, point)
        }

        val dataSet = LineDataSet(entries, "Movement Intensity")
        return ChartTheme.styleMovementAnalysisDataSet(dataSet)
    }

    fun createIntensityZones(): List<LimitLine> {
        return intensityThresholds.map { threshold ->
            LimitLine(threshold.value, threshold.label).apply {
                lineColor = ChartTheme.getMovementColor(threshold.value)
                lineWidth = 2f
                textColor = ChartTheme.getMovementColor(threshold.value)
                textSize = ChartTheme.Typography.ANNOTATION_SIZE
            }
        }
    }
}

/**
 * Phase distribution pie chart model
 */
data class PhaseDistributionChartModel(
    val config: ChartConfig,
    val phaseData: List<PhaseSlice>,
    val totalDuration: Long,
    val phaseAnalysis: PhaseAnalysisMetrics,
    val recommendations: List<PhaseRecommendation>
) {

    fun toPieDataSet(): PieDataSet {
        val entries = phaseData.map { phase ->
            PieEntry(phase.percentage, phase.label, phase)
        }

        val dataSet = PieDataSet(entries, "Sleep Phases")
        return ChartTheme.stylePhaseDistributionDataSet(dataSet)
    }

    fun createCenterText(): String {
        val hours = totalDuration / (1000 * 60 * 60)
        val minutes = (totalDuration % (1000 * 60 * 60)) / (1000 * 60)
        return "Total Sleep\n${hours}h ${minutes}m"
    }

    companion object {
        fun fromPhaseDistribution(
            phaseDistribution: PhaseDistributionData
        ): PhaseDistributionChartModel {
            val phaseSlices = listOf(
                PhaseSlice(
                    phase = SleepPhase.DEEP_SLEEP,
                    percentage = phaseDistribution.deepSleepPercentage,
                    duration = phaseDistribution.deepSleepDuration,
                    color = ChartTheme.Colors.PHASE_DEEP_SLEEP.toInt(),
                    label = "Deep Sleep",
                    isOptimal = phaseDistribution.deepSleepPercentage >= 15f
                ),
                PhaseSlice(
                    phase = SleepPhase.REM_SLEEP,
                    percentage = phaseDistribution.remSleepPercentage,
                    duration = phaseDistribution.remSleepDuration,
                    color = ChartTheme.Colors.PHASE_REM_SLEEP.toInt(),
                    label = "REM Sleep",
                    isOptimal = phaseDistribution.remSleepPercentage >= 20f
                ),
                PhaseSlice(
                    phase = SleepPhase.LIGHT_SLEEP,
                    percentage = phaseDistribution.lightSleepPercentage,
                    duration = phaseDistribution.lightSleepDuration,
                    color = ChartTheme.Colors.PHASE_LIGHT_SLEEP.toInt(),
                    label = "Light Sleep",
                    isOptimal = true
                ),
                PhaseSlice(
                    phase = SleepPhase.AWAKE,
                    percentage = phaseDistribution.awakePercentage,
                    duration = phaseDistribution.awakeDuration,
                    color = ChartTheme.Colors.PHASE_AWAKE.toInt(),
                    label = "Awake",
                    isOptimal = phaseDistribution.awakePercentage <= 10f
                )
            ).filter { it.percentage > 0f }

            return PhaseDistributionChartModel(
                config = ChartConfig(
                    id = "phase_distribution_${phaseDistribution.sessionId}",
                    title = "Sleep Phase Distribution",
                    chartType = ChartType.PIE,
                    analyticsType = AnalyticsChartType.PHASE_DISTRIBUTION,
                    dataSource = ChartDataSource.PHASE_ANALYSIS,
                    themeConfig = ChartThemeConfig(
                        colorPalette = ColorPalette.PHASE_FOCUSED
                    ),
                    layoutConfig = ChartLayoutConfig(
                        showLegend = true,
                        legendPosition = LegendPosition.RIGHT
                    ),
                    interactionConfig = ChartInteractionConfig(),
                    performanceConfig = ChartPerformanceConfig(),
                    accessibilityConfig = ChartAccessibilityConfig(),
                    exportConfig = ChartExportConfig()
                ),
                phaseData = phaseSlices,
                totalDuration = phaseDistribution.totalDuration,
                phaseAnalysis = PhaseAnalysisMetrics.fromDistribution(phaseDistribution),
                recommendations = generatePhaseRecommendations(phaseSlices)
            )
        }

        private fun generatePhaseRecommendations(phases: List<PhaseSlice>): List<PhaseRecommendation> {
            val recommendations = mutableListOf<PhaseRecommendation>()

            phases.forEach { phase ->
                if (!phase.isOptimal) {
                    val recommendation = when (phase.phase) {
                        SleepPhase.DEEP_SLEEP -> PhaseRecommendation(
                            phase = phase.phase,
                            message = "Consider improving sleep environment to increase deep sleep",
                            priority = RecommendationPriority.HIGH,
                            actionable = true
                        )
                        SleepPhase.REM_SLEEP -> PhaseRecommendation(
                            phase = phase.phase,
                            message = "REM sleep may improve with consistent sleep schedule",
                            priority = RecommendationPriority.MEDIUM,
                            actionable = true
                        )
                        SleepPhase.AWAKE -> PhaseRecommendation(
                            phase = phase.phase,
                            message = "Reduce sleep disruptions for better sleep continuity",
                            priority = RecommendationPriority.HIGH,
                            actionable = true
                        )
                        else -> null
                    }
                    recommendation?.let { recommendations.add(it) }
                }
            }

            return recommendations
        }
    }
}

/**
 * Quality factor radar chart model
 */
data class QualityFactorRadarChartModel(
    val config: ChartConfig,
    val factorData: List<QualityFactorDataPoint>,
    val benchmarkData: List<QualityFactorDataPoint>? = null,
    val factorAnalysis: QualityFactorAnalysisData
) {

    fun toRadarDataSet(): RadarDataSet {
        val entries = factorData.map { factor ->
            RadarEntry(factor.score, factor.label)
        }

        val dataSet = RadarDataSet(entries, "Quality Factors")
        dataSet.apply {
            color = ChartTheme.Colors.QUALITY_GOOD.toInt()
            fillColor = ChartTheme.Colors.QUALITY_GOOD.toInt()
            setDrawFilled(true)
            fillAlpha = 100
            lineWidth = 2f
            setDrawHighlightCircleEnabled(true)
            setDrawHighlightIndicators(false)
        }

        return dataSet
    }
}

/**
 * Combined analytics dashboard chart model
 */
data class CombinedAnalyticsChartModel(
    val config: ChartConfig,
    val qualityLineData: List<Entry>,
    val efficiencyBarData: List<BarEntry>,
    val movementData: List<Entry>,
    val correlationMatrix: CorrelationMatrix,
    val insights: List<MultiMetricInsight>
) {

    fun toCombinedData(): CombinedData {
        val combinedData = CombinedData()

        // Add line data for quality
        val qualityDataSet = LineDataSet(qualityLineData, "Quality")
        ChartTheme.styleQualityTrendDataSet(qualityDataSet)
        combinedData.setData(LineData(qualityDataSet))

        // Add bar data for efficiency
        val efficiencyDataSet = BarDataSet(efficiencyBarData, "Efficiency")
        efficiencyDataSet.apply {
            color = ChartTheme.Colors.EFFICIENCY_EXCELLENT.toInt()
            setDrawValues(false)
        }
        combinedData.setData(BarData(efficiencyDataSet))

        return combinedData
    }
}

// ========== DATA POINT MODELS ==========

/**
 * Quality trend data point
 */
data class QualityTrendPoint(
    val timestamp: Long,
    val qualityScore: Float,
    val sessionId: Long,
    val confidence: Float,
    val metadata: Map<String, Any> = emptyMap()
) {
    val formattedDate: String
        get() = SimpleDateFormat("MMM dd", Locale.getDefault()).format(Date(timestamp))

    val qualityGrade: String
        get() = when {
            qualityScore >= 9f -> "A+"
            qualityScore >= 8f -> "A"
            qualityScore >= 7f -> "B+"
            qualityScore >= 6f -> "B"
            qualityScore >= 5f -> "C+"
            qualityScore >= 4f -> "C"
            else -> "D"
        }
}

/**
 * Efficiency data point
 */
data class EfficiencyDataPoint(
    val timestamp: Long,
    val efficiency: Float,
    val basicEfficiency: Float,
    val adjustedEfficiency: Float,
    val sleepLatency: Long,
    val wakeCount: Int,
    val sessionId: Long
)

/**
 * Movement data point
 */
data class MovementDataPoint(
    val timestamp: Long,
    val intensity: Float,
    val isSignificant: Boolean,
    val sessionRelativeTime: Long, // Time within the session
    val context: MovementContext
)

/**
 * Phase slice for pie charts
 */
data class PhaseSlice(
    val phase: SleepPhase,
    val percentage: Float,
    val duration: Long,
    @ColorInt val color: Int,
    val label: String,
    val isOptimal: Boolean
) {
    val formattedDuration: String
        get() {
            val hours = duration / (1000 * 60 * 60)
            val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)
            return "${hours}h ${minutes}m"
        }

    val formattedPercentage: String
        get() = "${percentage.toInt()}%"
}

/**
 * Quality factor data point for radar charts
 */
data class QualityFactorDataPoint(
    val factorName: String,
    val score: Float,
    val maxScore: Float = 10f,
    val weight: Float = 1f,
    val label: String = factorName,
    val description: String = "",
    val trend: TrendDirection = TrendDirection.STABLE
) {
    val normalizedScore: Float
        get() = (score / maxScore) * 100f

    val grade: String
        get() = when {
            normalizedScore >= 90f -> "A+"
            normalizedScore >= 80f -> "A"
            normalizedScore >= 70f -> "B"
            normalizedScore >= 60f -> "C"
            normalizedScore >= 50f -> "D"
            else -> "F"
        }
}

// ========== CHART ENHANCEMENT MODELS ==========

/**
 * Benchmark line for charts
 */
data class BenchmarkLine(
    val value: Float,
    val label: String,
    @ColorInt val color: Int,
    val style: LineStyle = LineStyle.SOLID,
    val lineWidth: Float = 2f,
    val description: String = "",
    val priority: Int = 1
)

/**
 * Target line for goal-based charts
 */
data class TargetLine(
    val value: Float,
    val label: String,
    @ColorInt val color: Int,
    val lineWidth: Float = 2f,
    val isAchieved: Boolean = false,
    val targetDate: Long? = null
)

/**
 * Intensity threshold for movement charts
 */
data class IntensityThreshold(
    val value: Float,
    val label: String,
    val level: IntensityLevel,
    val description: String
)

/**
 * Chart annotation for important events
 */
data class ChartAnnotation(
    val x: Float,
    val y: Float,
    val text: String,
    val icon: String? = null,
    @ColorInt val color: Int = ChartTheme.Colors.ACCENT_PRIMARY.toInt(),
    val priority: AnnotationPriority = AnnotationPriority.NORMAL
)

/**
 * Chart marker for data point details
 */
data class ChartMarker(
    val timestamp: Long,
    val primaryValue: Float,
    val secondaryValue: Float? = null,
    val title: String,
    val description: String,
    val additionalInfo: Map<String, String> = emptyMap(),
    @ColorInt val backgroundColor: Int = ChartTheme.Colors.SURFACE_ELEVATED.toInt(),
    @ColorInt val textColor: Int = ChartTheme.Colors.TEXT_PRIMARY.toInt()
)

// ========== CHART ANALYTICS MODELS ==========

/**
 * Trend statistics for analytical insights
 */
data class TrendStatistics(
    val overallTrend: TrendDirection,
    val trendStrength: Float,
    val correlation: Float,
    val seasonality: Float,
    val volatility: Float,
    val predictability: Float,
    val significantChanges: List<SignificantChange>,
    val projections: List<TrendProjection>
) {
    companion object {
        fun fromTrendAnalysis(analysis: TrendAnalysisResult): TrendStatistics {
            return TrendStatistics(
                overallTrend = analysis.overallTrend,
                trendStrength = analysis.trendStrength.ordinal.toFloat() / 4f,
                correlation = 0.75f, // Would be calculated from actual data
                seasonality = 0.3f,
                volatility = 0.4f,
                predictability = analysis.trendConfidence,
                significantChanges = emptyList(),
                projections = emptyList()
            )
        }
    }
}

/**
 * Comparison trend data
 */
data class ComparisonTrendData(
    val personalAverage: Float,
    val personalBest: Float,
    val populationAverage: Float? = null,
    val ageGroupAverage: Float? = null,
    val targetValue: Float? = null,
    val improvementRate: Float = 0f
)

/**
 * Efficiency metrics
 */
data class EfficiencyMetrics(
    val averageEfficiency: Float,
    val bestEfficiency: Float,
    val worstEfficiency: Float,
    val consistencyScore: Float,
    val improvementTrend: TrendDirection,
    val targetAchievementRate: Float
)

/**
 * Movement pattern analysis
 */
data class MovementPatternAnalysis(
    val peakMovementTimes: List<TimeRange>,
    val restfulPeriods: List<TimeRange>,
    val movementFrequency: Float,
    val intensityDistribution: Map<IntensityLevel, Float>,
    val correlationWithPhases: Map<SleepPhase, Float>
)

/**
 * Movement correlation data
 */
data class MovementCorrelationData(
    val qualityCorrelation: Float,
    val efficiencyCorrelation: Float,
    val phaseCorrelations: Map<SleepPhase, Float>,
    val environmentalFactors: Map<String, Float>
)

/**
 * Phase analysis metrics
 */
data class PhaseAnalysisMetrics(
    val phaseBalance: Float,
    val transitionQuality: Float,
    val sleepContinuity: Float,
    val phaseEfficiency: Map<SleepPhase, Float>,
    val recommendations: List<String>
) {
    companion object {
        fun fromDistribution(distribution: PhaseDistributionData): PhaseAnalysisMetrics {
            val balance = calculatePhaseBalance(distribution)
            return PhaseAnalysisMetrics(
                phaseBalance = balance,
                transitionQuality = 0.8f, // Would be calculated from actual data
                sleepContinuity = 1f - (distribution.awakePercentage / 100f),
                phaseEfficiency = mapOf(
                    SleepPhase.DEEP_SLEEP to distribution.deepSleepPercentage / 25f, // Optimal ~25%
                    SleepPhase.REM_SLEEP to distribution.remSleepPercentage / 25f,   // Optimal ~25%
                    SleepPhase.LIGHT_SLEEP to 1f, // Light sleep is flexible
                    SleepPhase.AWAKE to maxOf(0f, 1f - distribution.awakePercentage / 10f) // Target <10%
                ),
                recommendations = emptyList()
            )
        }

        private fun calculatePhaseBalance(distribution: PhaseDistributionData): Float {
            val deepOptimal = 20f
            val remOptimal = 25f
            val awakeMax = 10f

            val deepScore = 1f - abs(distribution.deepSleepPercentage - deepOptimal) / deepOptimal
            val remScore = 1f - abs(distribution.remSleepPercentage - remOptimal) / remOptimal
            val awakeScore = maxOf(0f, 1f - distribution.awakePercentage / awakeMax)

            return (deepScore + remScore + awakeScore) / 3f
        }
    }
}

/**
 * Quality factor analysis data
 */
data class QualityFactorAnalysisData(
    val factorWeights: Map<String, Float>,
    val factorCorrelations: Map<String, Map<String, Float>>,
    val strengthAreas: List<String>,
    val improvementAreas: List<String>,
    val recommendedFocus: String
)

/**
 * Correlation matrix for multi-metric analysis
 */
data class CorrelationMatrix(
    val metrics: List<String>,
    val correlations: Array<FloatArray>,
    val significanceLevel: Float = 0.05f
) {
    fun getCorrelation(metric1: String, metric2: String): Float? {
        val index1 = metrics.indexOf(metric1)
        val index2 = metrics.indexOf(metric2)

        return if (index1 >= 0 && index2 >= 0) {
            correlations[index1][index2]
        } else null
    }
}

/**
 * Multi-metric insight
 */
data class MultiMetricInsight(
    val title: String,
    val description: String,
    val involvedMetrics: List<String>,
    val correlationStrength: Float,
    val actionable: Boolean,
    val priority: InsightPriority
)

// ========== CHART INTERACTION MODELS ==========

/**
 * Chart selection state
 */
data class ChartSelectionState(
    val selectedDataPoints: List<ChartDataPoint>,
    val selectionRange: SelectionRange? = null,
    val highlightedEntry: Entry? = null,
    val activeTooltip: ChartMarker? = null
)

/**
 * Chart zoom state
 */
data class ChartZoomState(
    val zoomLevel: Float = 1f,
    val centerX: Float = 0f,
    val centerY: Float = 0f,
    val isZoomed: Boolean = false,
    val minZoom: Float = 0.5f,
    val maxZoom: Float = 5f
)

/**
 * Chart animation state
 */
data class ChartAnimationState(
    val isAnimating: Boolean = false,
    val animationType: AnimationType = AnimationType.NONE,
    val progress: Float = 0f,
    val duration: Long = 800L
)

// ========== CHART EXPORT MODELS ==========

/**
 * Chart export request
 */
data class ChartExportRequest(
    val chartId: String,
    val format: ExportFormat,
    val quality: ExportQuality,
    val includeMetadata: Boolean = true,
    val includeWatermark: Boolean = false,
    val dimensions: ExportDimensions = ExportDimensions.STANDARD,
    val backgroundColor: Int? = null
)

/**
 * Chart export result
 */
data class ChartExportResult(
    val success: Boolean,
    val filePath: String? = null,
    val fileSize: Long = 0L,
    val format: ExportFormat,
    val dimensions: Pair<Int, Int>,
    val error: String? = null,
    val exportTime: Long = System.currentTimeMillis()
)

// ========== SUPPORTING MODELS ==========

/**
 * Chart margins
 */
data class ChartMargins(
    val left: Float = 16f,
    val top: Float = 16f,
    val right: Float = 16f,
    val bottom: Float = 40f
)

/**
 * Selection range
 */
data class SelectionRange(
    val startX: Float,
    val endX: Float,
    val startY: Float? = null,
    val endY: Float? = null
)

/**
 * Time range for chart data
 */
data class TimeRange(
    val startTime: Long,
    val endTime: Long,
    val description: String = ""
) {
    val durationMs: Long
        get() = endTime - startTime

    val durationDays: Int
        get() = (durationMs / (24 * 60 * 60 * 1000L)).toInt()
}

/**
 * Significant change in trend
 */
data class SignificantChange(
    val timestamp: Long,
    val magnitude: Float,
    val direction: TrendDirection,
    val description: String,
    val confidence: Float
)

/**
 * Trend projection
 */
data class TrendProjection(
    val futureTimestamp: Long,
    val projectedValue: Float,
    val confidence: Float,
    val scenario: ProjectionScenario
)

/**
 * Movement context
 */
data class MovementContext(
    val sessionPhase: SleepPhase,
    val environmentalFactors: Map<String, Float>,
    val isAnomalous: Boolean = false
)

/**
 * Phase recommendation
 */
data class PhaseRecommendation(
    val phase: SleepPhase,
    val message: String,
    val priority: RecommendationPriority,
    val actionable: Boolean
)

// ========== ENUMS ==========

enum class ChartType {
    LINE, BAR, PIE, RADAR, COMBINED, SCATTER, CANDLESTICK
}

enum class AnalyticsChartType {
    QUALITY_TREND, EFFICIENCY_TREND, MOVEMENT_ANALYSIS, NOISE_ANALYSIS,
    PHASE_DISTRIBUTION, QUALITY_FACTORS, COMPARATIVE_ANALYSIS,
    CORRELATION_MATRIX, SLEEP_DEBT, CONSISTENCY_ANALYSIS
}

enum class ChartDataSource {
    QUALITY_ANALYSIS, EFFICIENCY_ANALYSIS, MOVEMENT_ANALYSIS, NOISE_ANALYSIS,
    PHASE_ANALYSIS, TREND_ANALYSIS, COMPARATIVE_ANALYSIS, STATISTICAL_ANALYSIS
}

enum class ColorPalette {
    QUALITY_FOCUSED, EFFICIENCY_FOCUSED, PHASE_FOCUSED, MOVEMENT_FOCUSED,
    NOISE_FOCUSED, CUSTOM, ACCESSIBILITY_HIGH_CONTRAST
}

enum class LegendPosition {
    TOP, BOTTOM, LEFT, RIGHT, CENTER, NONE
}

enum class UpdateFrequency {
    REAL_TIME, MANUAL, HOURLY, DAILY, WEEKLY
}

enum class AggregationLevel {
    RAW, MINUTE, HOUR, DAY, WEEK, MONTH
}

enum class LineStyle {
    SOLID, DASHED, DOTTED
}

enum class IntensityLevel {
    VERY_LOW, LOW, MODERATE, HIGH, VERY_HIGH
}

enum class AnnotationPriority {
    LOW, NORMAL, HIGH, CRITICAL
}

enum class AnimationType {
    NONE, FADE_IN, SLIDE_IN, ZOOM_IN, DATA_CHANGE, HIGHLIGHT
}

enum class ExportFormat {
    PNG, JPG, PDF, SVG
}

enum class ExportQuality {
    LOW, MEDIUM, HIGH, ULTRA
}

enum class ExportDimensions(val width: Int, val height: Int) {
    THUMBNAIL(400, 300),
    STANDARD(800, 600),
    HIGH_RESOLUTION(1600, 1200),
    PRINT_QUALITY(2400, 1800),
    CUSTOM(0, 0)
}

enum class ProjectionScenario {
    OPTIMISTIC, REALISTIC, PESSIMISTIC
}

enum class RecommendationPriority {
    LOW, MEDIUM, HIGH, CRITICAL
}

enum class InsightPriority {
    LOW, MEDIUM, HIGH, CRITICAL
}

// ========== FACTORY METHODS ==========

/**
 * Chart model factory for creating different chart types
 */
object ChartModelFactory {

    fun createQualityTrendChart(
        trendAnalysis: TrendAnalysisResult,
        sessions: List<SessionSummaryDTO>,
        timeRange: String = "Last 30 days"
    ): QualityTrendChartModel {
        return QualityTrendChartModel.fromTrendAnalysis(trendAnalysis, sessions)
    }

    fun createPhaseDistributionChart(
        phaseDistribution: PhaseDistributionData
    ): PhaseDistributionChartModel {
        return PhaseDistributionChartModel.fromPhaseDistribution(phaseDistribution)
    }

    fun createEfficiencyChart(
        sessions: List<SessionSummaryDTO>,
        targetEfficiency: Float = 85f
    ): EfficiencyChartModel {
        val efficiencyData = sessions.mapIndexed { index, session ->
            EfficiencyDataPoint(
                timestamp = session.startTime,
                efficiency = session.sleepEfficiency,
                basicEfficiency = session.sleepEfficiency,
                adjustedEfficiency = session.sleepEfficiency * 0.95f,
                sleepLatency = 0L, // Would come from detailed session data
                wakeCount = 0, // Would come from detailed session data
                sessionId = session.id
            )
        }

        val targetLines = listOf(
            TargetLine(
                value = targetEfficiency,
                label = "Target",
                color = ChartTheme.Colors.EFFICIENCY_EXCELLENT.toInt(),
                isAchieved = sessions.any { it.sleepEfficiency >= targetEfficiency }
            )
        )

        return EfficiencyChartModel(
            config = ChartConfig(
                id = "efficiency_${System.currentTimeMillis()}",
                title = "Sleep Efficiency Trend",
                chartType = ChartType.LINE,
                analyticsType = AnalyticsChartType.EFFICIENCY_TREND,
                dataSource = ChartDataSource.EFFICIENCY_ANALYSIS,
                themeConfig = ChartThemeConfig(
                    primaryColor = ChartTheme.Colors.EFFICIENCY_EXCELLENT.toInt(),
                    colorPalette = ColorPalette.EFFICIENCY_FOCUSED
                ),
                layoutConfig = ChartLayoutConfig(),
                interactionConfig = ChartInteractionConfig(),
                performanceConfig = ChartPerformanceConfig(),
                accessibilityConfig = ChartAccessibilityConfig(),
                exportConfig = ChartExportConfig()
            ),
            efficiencyData = efficiencyData,
            targetLines = targetLines,
            performanceMetrics = EfficiencyMetrics(
                averageEfficiency = sessions.map { it.sleepEfficiency }.average().toFloat(),
                bestEfficiency = sessions.maxOfOrNull { it.sleepEfficiency } ?: 0f,
                worstEfficiency = sessions.minOfOrNull { it.sleepEfficiency } ?: 0f,
                consistencyScore = 0.8f, // Would be calculated
                improvementTrend = TrendDirection.STABLE, // Would be calculated
                targetAchievementRate = sessions.count { it.sleepEfficiency >= targetEfficiency }.toFloat() / sessions.size
            )
        )
    }
}

// ========== EXTENSION FUNCTIONS ==========

/**
 * Extension functions for easy chart creation
 */
fun List<SessionSummaryDTO>.toQualityTrendChart(): QualityTrendChartModel {
    // Create a simple trend analysis from sessions
    val trendAnalysis = TrendAnalysisResult(
        timeRange = TimeRange(
            startDate = this.minOfOrNull { it.startTime } ?: 0L,
            endDate = this.maxOfOrNull { it.startTime } ?: 0L,
            description = "Session range"
        ),
        overallTrend = SleepTrend.STABLE,
        trendStrength = TrendStrength.MODERATE,
        trendConfidence = 0.8f,
        statisticalSignificance = StatisticalSignificance.SIGNIFICANT,
        qualityTrend = MetricTrend("quality", TrendDirection.STABLE, 0f, 0f, 0f, 0f, 0f, 0f, 0f, emptyList()),
        durationTrend = MetricTrend("duration", TrendDirection.STABLE, 0f, 0f, 0f, 0f, 0f, 0f, 0f, emptyList()),
        efficiencyTrend = MetricTrend("efficiency", TrendDirection.STABLE, 0f, 0f, 0f, 0f, 0f, 0f, 0f, emptyList()),
        consistencyTrend = MetricTrend("consistency", TrendDirection.STABLE, 0f, 0f, 0f, 0f, 0f, 0f, 0f, emptyList()),
        movementTrend = MetricTrend("movement", TrendDirection.STABLE, 0f, 0f, 0f, 0f, 0f, 0f, 0f, emptyList()),
        timingTrend = MetricTrend("timing", TrendDirection.STABLE, 0f, 0f, 0f, 0f, 0f, 0f, 0f, emptyList()),
        seasonalPatterns = null,
        cyclicalBehaviors = emptyList(),
        changepoints = emptyList(),
        projections = TrendProjections(
            shortTermProjection = Projection(0L, 0f, TrendDirection.STABLE, 0f, emptyList()),
            mediumTermProjection = Projection(0L, 0f, TrendDirection.STABLE, 0f, emptyList()),
            longTermProjection = Projection(0L, 0f, TrendDirection.STABLE, 0f, emptyList()),
            projectionConfidence = 0f,
            uncertaintyBounds = UncertaintyBounds(0f, 0f, 0f),
            assumptions = emptyList()
        ),
        trendInsights = emptyList(),
        contributingFactors = emptyList(),
        recommendations = emptyList(),
        sampleSize = this.size,
        dataCompleteness = 1f,
        analysisReliability = AnalysisReliability.HIGH
    )

    return ChartModelFactory.createQualityTrendChart(trendAnalysis, this)
}

fun PhaseDistributionData.toChartModel(): PhaseDistributionChartModel {
    return ChartModelFactory.createPhaseDistributionChart(this)
}