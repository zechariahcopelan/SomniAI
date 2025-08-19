package com.example.somniai.data

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import java.util.*
import kotlin.math.*
import com.example.somniai.data.SessionSummaryDTO

/**
 * Comprehensive analytics result models for sleep data analysis with AI integration
 *
 * Provides sophisticated data classes for:
 * - Sleep quality reports with detailed scoring and insights
 * - Trend analysis with statistical significance testing
 * - Statistical summaries across multiple time periods
 * - Comparative metrics against personal and population baselines
 * - AI-generated insights and recommendations with full metadata
 * - Visualization-ready analytics data
 * - Advanced pattern recognition and anomaly detection
 * - Personalization and user context integration
 * - Performance monitoring and optimization
 */

// ========== ENHANCED AI INSIGHT RESULT MODELS ==========

/**
 * Basic sleep session analytics for UI compatibility
 */
@Parcelize
data class SleepSessionAnalytics(
    val sessionId: Long,
    val sessionDuration: Long = 0L,
    val actionableRecommendations: List<String> = emptyList(),
    val confidence: Float = 0.0f,
    val qualityScore: Float = 0f,
    val efficiencyScore: Float = 0f,
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable

/**
 * Comprehensive AI insight generation result with full metadata
 */
@Parcelize
data class AIInsightGenerationResult(
    val jobId: String,
    val sessionId: Long? = null,
    val generationType: InsightGenerationType,
    val generationStartTime: Long,
    val generationEndTime: Long,
    val processingTimeMs: Long,
    val insights: List<ProcessedInsight>,
    val qualityMetrics: InsightQualityMetrics,
    val aiModelUsed: String,
    val promptVersion: String,
    val tokenUsage: AITokenUsage,
    val confidenceDistribution: Map<String, Float>,
    val categoryDistribution: Map<InsightCategory, Int>,
    val priorityDistribution: Map<Int, Int>,
    val validationResults: InsightValidationResults,
    val personalizationApplied: PersonalizationMetadata,
    val fallbackUsed: Boolean = false,
    val errorMessages: List<String> = emptyList(),
    val performanceMetrics: GenerationPerformanceMetrics,
    val dataSourcesUsed: List<String>,
    val contextData: Map<String, Any> = emptyMap(),
    val generatedAt: Long = System.currentTimeMillis()
) : Parcelable {

    val isSuccessful: Boolean
        get() = insights.isNotEmpty() && errorMessages.isEmpty()

    val averageConfidence: Float
        get() = if (insights.isNotEmpty()) insights.map { it.confidence }.average().toFloat() else 0f

    val averageQuality: Float
        get() = if (insights.isNotEmpty()) insights.map { it.qualityScore }.average().toFloat() else 0f

    val highQualityInsightCount: Int
        get() = insights.count { it.qualityScore >= 0.8f && it.confidence >= 0.7f }

    val actionableInsightCount: Int
        get() = insights.count { it.actionabilityScore >= 0.7f }

    fun getInsightsByPriority(): Map<Int, List<ProcessedInsight>> {
        return insights.groupBy { it.priority }
    }

    fun getInsightsByCategory(): Map<InsightCategory, List<ProcessedInsight>> {
        return insights.groupBy { it.category }
    }

    fun getTopInsights(count: Int = 5): List<ProcessedInsight> {
        return insights
            .sortedWith(
                compareByDescending<ProcessedInsight> { it.priority }
                    .thenByDescending { it.qualityScore }
                    .thenByDescending { it.confidence }
            )
            .take(count)
    }
}


// ========== BASIC INSIGHT MODEL (for UI compatibility) ==========

/**
 * Basic sleep insight for UI display (simplified version of ProcessedInsight)
 */
@Parcelize
data class SleepInsight(
    val id: String,
    val title: String,
    val description: String,
    val category: String,
    val priority: Int,
    val confidence: Float,
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable


/**
 * Processed insight with comprehensive validation and scoring
 */
@Parcelize
data class ProcessedInsight(
    val id: String,
    val originalInsight: SleepInsight,
    val category: InsightCategory,
    val subcategory: String? = null,
    val priority: Int,
    val title: String,
    val description: String,
    val recommendation: String,
    val evidence: List<EvidencePoint>,
    val dataPoints: List<InsightDataPoint>,
    val confidence: Float,
    val qualityScore: Float,
    val relevanceScore: Float,
    val actionabilityScore: Float,
    val noveltyScore: Float,
    val personalizationScore: Float,
    val validationResults: ValidationResult,
    val aiGenerated: Boolean,
    val aiModelUsed: String? = null,
    val processingMetadata: ProcessingMetadata,
    val userContext: UserContextSummary? = null,
    val relatedInsights: List<String> = emptyList(),
    val tags: List<String> = emptyList(),
    val validityPeriod: Long = 0L,
    val implementationDifficulty: ImplementationDifficulty,
    val expectedImpact: ExpectedImpact,
    val timeToImpact: TimeToImpact,
    val successMetrics: List<String> = emptyList(),
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable {

    val isHighQuality: Boolean
        get() = qualityScore >= 0.8f && confidence >= 0.7f && relevanceScore >= 0.7f

    val isActionable: Boolean
        get() = actionabilityScore >= 0.7f && recommendation.isNotBlank()

    val isPersonalized: Boolean
        get() = personalizationScore >= 0.5f && userContext != null

    val overallScore: Float
        get() = (qualityScore + confidence + relevanceScore + actionabilityScore) / 4f

    fun getScoreBreakdown(): Map<String, Float> {
        return mapOf(
            "quality" to qualityScore,
            "confidence" to confidence,
            "relevance" to relevanceScore,
            "actionability" to actionabilityScore,
            "novelty" to noveltyScore,
            "personalization" to personalizationScore
        )
    }
}

/**
 * Evidence supporting an insight with confidence scoring
 */
@Parcelize
data class EvidencePoint(
    val type: EvidenceType,
    val description: String,
    val dataSource: String,
    val value: String,
    val confidence: Float,
    val weight: Float,
    val statisticalSignificance: Float = 0f,
    val timestamp: Long? = null,
    val metadata: Map<String, String> = emptyMap()
) : Parcelable

enum class EvidenceType {
    STATISTICAL_TREND,
    PATTERN_ANALYSIS,
    COMPARATIVE_DATA,
    MEDICAL_REFERENCE,
    PERSONAL_BASELINE,
    BEHAVIORAL_CORRELATION,
    ENVIRONMENTAL_FACTOR,
    AI_PREDICTION
}

enum class DataIntegrityStatus {
    EXCELLENT,
    GOOD,
    FAIR,
    POOR
}

/**
 * Data point referenced in insight with visualization support
 */
@Parcelize
data class InsightDataPoint(
    val metric: String,
    val value: Float,
    val unit: String,
    val timestamp: Long,
    val context: String? = null,
    val benchmark: Float? = null,
    val percentileRank: Float? = null,
    val trend: TrendDirection? = null,
    val significance: StatisticalSignificance? = null,
    val visualization: VisualizationHint? = null
) : Parcelable

@Parcelize
data class VisualizationHint(
    val chartType: String,
    val colorScheme: String? = null,
    val highlightValue: Boolean = false
) : Parcelable

/**
 * AI token usage tracking for cost management
 */
@Parcelize
data class AITokenUsage(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int,
    val estimatedCost: Double,
    val model: String,
    val efficiency: Float = if (promptTokens > 0) completionTokens.toFloat() / promptTokens else 0f
) : Parcelable

/**
 * Insight quality metrics for AI validation
 */
@Parcelize
data class InsightQualityMetrics(
    val averageQuality: Float,
    val averageConfidence: Float,
    val averageRelevance: Float,
    val averageActionability: Float,
    val qualityDistribution: Map<String, Int>,
    val outlierCount: Int,
    val validationScore: Float
) : Parcelable

/**
 * Comprehensive insight validation results
 */
@Parcelize
data class InsightValidationResults(
    val overallValid: Boolean,
    val validationScore: Float,
    val contentValidation: ContentValidationResult,
    val qualityValidation: QualityValidationResult,
    val consistencyValidation: ConsistencyValidationResult,
    val relevanceValidation: RelevanceValidationResult,
    val validationMessages: List<String>,
    val recommendedActions: List<String>
) : Parcelable

@Parcelize
data class ContentValidationResult(
    val hasTitle: Boolean,
    val hasDescription: Boolean,
    val hasRecommendation: Boolean,
    val appropriateLength: Boolean,
    val noHarmfulContent: Boolean,
    val factuallyConsistent: Boolean
) : Parcelable

@Parcelize
data class QualityValidationResult(
    val qualityScore: Float,
    val meetsQualityThreshold: Boolean,
    val hasEvidence: Boolean,
    val evidenceQuality: Float,
    val actionabilityScore: Float
) : Parcelable

@Parcelize
data class ConsistencyValidationResult(
    val internalConsistency: Float,
    val consistentWithData: Boolean,
    val consistentWithPriorInsights: Boolean,
    val noContradictions: Boolean
) : Parcelable

@Parcelize
data class RelevanceValidationResult(
    val relevanceScore: Float,
    val userRelevant: Boolean,
    val contextuallyAppropriate: Boolean,
    val timelyRelevant: Boolean
) : Parcelable

/**
 * Personalization metadata for insights
 */
@Parcelize
data class PersonalizationMetadata(
    val personalizationLevel: Float,
    val userFactorsConsidered: List<String>,
    val personalizationStrategy: String,
    val adaptationScore: Float,
    val userContextUsed: Map<String, String>,
    val personalizationConfidence: Float
) : Parcelable

/**
 * User context summary for personalization
 */
@Parcelize
data class UserContextSummary(
    val userId: String,
    val age: Int?,
    val sleepGoals: List<String>,
    val healthConditions: List<String>,
    val lifestyle: String?,
    val preferences: Map<String, String>,
    val historicalPatterns: Map<String, Float>,
    val contextScore: Float
) : Parcelable

/**
 * Generation performance metrics
 */
@Parcelize
data class GenerationPerformanceMetrics(
    val totalGenerationTime: Long,
    val aiProcessingTime: Long,
    val validationTime: Long,
    val personalizationTime: Long,
    val memoryUsage: Long,
    val tokensPerSecond: Float,
    val insightsPerSecond: Float,
    val efficiency: Float
) : Parcelable

/**
 * Processing metadata for various analysis processes
 */
@Parcelize
data class ProcessingMetadata(
    val processingVersion: String,
    val processingTimeMs: Long,
    val algorithmUsed: String,
    val parametersUsed: Map<String, String>,
    val dataSourcesCount: Int,
    val validationSteps: List<String>,
    val qualityChecks: Map<String, Boolean>,
    val performance: Map<String, Float> = emptyMap()
) : Parcelable

/**
 * Enhanced validation result with detailed scoring
 */
@Parcelize
data class ValidationResult(
    val isValid: Boolean,
    val validationScore: Float,
    val validationChecks: Map<String, Boolean>,
    val validationMessages: List<String>,
    val validationDate: Long = System.currentTimeMillis()
) : Parcelable

// ========== ENHANCED SLEEP QUALITY REPORTS ==========

/**
 * Comprehensive sleep quality report for individual sessions or periods
 */
@Parcelize
data class SleepQualityReport(
    val reportId: String = generateReportId(),
    val reportType: ReportType,
    val timeRange: TimeRange,
    val generatedAt: Long = System.currentTimeMillis(),

    // Core quality metrics
    val overallQualityScore: Float,
    val qualityGrade: QualityGrade,
    val qualityFactors: QualityFactorAnalysis,

    // Detailed breakdowns
    val durationAnalysis: DurationAnalysis,
    val efficiencyAnalysis: EfficiencyAnalysis,
    val movementAnalysis: MovementAnalysis,
    val noiseAnalysis: NoiseAnalysis,
    val consistencyAnalysis: ConsistencyAnalysis,
    val timingAnalysis: TimingAnalysis,

    // Comparative context
    val personalComparison: PersonalComparisonMetrics,
    val populationComparison: PopulationComparisonMetrics? = null,
    val benchmarkComparison: BenchmarkComparisonMetrics? = null,

    // Enhanced analytics
    val trendAnalysis: QualityTrendAnalysis,
    val patternAnalysis: SleepPatternAnalysis? = null,
    val anomalyDetection: AnomalyDetectionResult? = null,

    // Insights and recommendations
    val keyInsights: List<QualityInsight>,
    val recommendations: List<QualityRecommendation>,
    val strengthAreas: List<StrengthArea>,
    val improvementOpportunities: List<ImprovementOpportunity>,
    val goalProgress: GoalProgressAnalysis? = null,

    // AI integration
    val aiGeneratedInsights: List<ProcessedInsight> = emptyList(),
    val aiAnalysisMetadata: AIAnalysisMetadata? = null,

    // Data reliability
    val dataQuality: DataQualityMetrics,
    val confidenceLevel: ConfidenceLevel,
    val reportMetadata: ReportMetadata
) : Parcelable {
    val formattedScore: String
        get() = "${(overallQualityScore * 10).toInt()}/100"

    val isHighQuality: Boolean
        get() = overallQualityScore >= 7.5f

    val needsAttention: Boolean
        get() = overallQualityScore < 5.0f || improvementOpportunities.any { it.priority == Priority.HIGH }

    val reportSummary: String
        get() = "Sleep quality: ${qualityGrade.displayName} (${formattedScore}). " +
                "${if (isHighQuality) "Excellent sleep patterns maintained." else "Room for improvement identified."}"

    val hasExcellentQuality: Boolean
        get() = qualityGrade == QualityGrade.A_PLUS

    val hasImprovedFromBaseline: Boolean
        get() = personalComparison.qualityImprovement > 0f

    fun getTopRecommendations(count: Int = 5): List<QualityRecommendation> {
        return recommendations
            .sortedWith(
                compareByDescending<QualityRecommendation> { it.priority.value }
                    .thenByDescending { it.expectedImpact.value }
            )
            .take(count)
    }

    fun getQuickWins(): List<QualityRecommendation> {
        return recommendations.filter {
            it.implementationDifficulty == ImplementationDifficulty.LOW &&
                    it.expectedImpact.value >= 0.5f
        }
    }

    companion object {
        fun generateReportId(): String = "SQR-${System.currentTimeMillis()}"
    }
}

/**
 * Enhanced quality factor analysis with detailed scoring
 */
@Parcelize
data class QualityFactorAnalysis(
    val movementFactor: QualityFactor,
    val noiseFactor: QualityFactor,
    val durationFactor: QualityFactor,
    val consistencyFactor: QualityFactor,
    val efficiencyFactor: QualityFactor,
    val timingFactor: QualityFactor,
    val phaseBalanceFactor: QualityFactor,
    val environmentalFactor: QualityFactor? = null,

    // Weighted scores
    val weightedOverallScore: Float,
    val factorWeights: Map<String, Float>,
    val factorInteractions: List<FactorInteraction> = emptyList()
) : Parcelable {
    val allFactors: List<QualityFactor>
        get() = listOfNotNull(
            movementFactor, noiseFactor, durationFactor, consistencyFactor,
            efficiencyFactor, timingFactor, phaseBalanceFactor, environmentalFactor
        )

    val strongestFactor: QualityFactor
        get() = allFactors.maxByOrNull { it.score } ?: movementFactor

    val weakestFactor: QualityFactor
        get() = allFactors.minByOrNull { it.score } ?: movementFactor

    val factorBalance: Float
        get() {
            val scores = allFactors.map { it.score }
            val mean = scores.average()
            val variance = scores.map { (it - mean).pow(2) }.average()
            return (10f - variance.toFloat()).coerceIn(0f, 10f) // Higher = more balanced
        }

    val improvementPotential: Float
        get() = 10f - allFactors.map { it.score }.average().toFloat()

    fun getAllFactors(): List<QualityFactor> = allFactors

    fun getStrongestFactor(): QualityFactor = strongestFactor

    fun getWeakestFactor(): QualityFactor = weakestFactor
}

/**
 * Enhanced individual quality factor with detailed analysis
 */
@Parcelize
data class QualityFactor(
    val name: String,
    val score: Float,
    val grade: QualityGrade,
    val rawValue: Float,
    val benchmarkValue: Float,
    val percentileRank: Float = 0f,
    val trend: TrendDirection,
    val impact: FactorImpact,
    val confidence: Float,
    val dataPoints: Int = 0,
    val insights: List<String>,
    val recommendations: List<String>,
    val historicalComparison: Float = 0f
) : Parcelable {
    val scorePercentage: Int
        get() = (score * 10).toInt()

    val isExcellent: Boolean
        get() = score >= 8.5f

    val needsImprovement: Boolean
        get() = score < 6.0f

    val isAboveBenchmark: Boolean
        get() = rawValue > benchmarkValue

    val improvement: Float
        get() = rawValue - benchmarkValue

    val improvementPercentage: Float
        get() = if (benchmarkValue != 0f) (improvement / benchmarkValue) * 100f else 0f

    val performanceLevel: String
        get() = when {
            score >= 9f -> "Exceptional"
            score >= 8f -> "Excellent"
            score >= 7f -> "Very Good"
            score >= 6f -> "Good"
            score >= 5f -> "Average"
            score >= 4f -> "Below Average"
            score >= 3f -> "Poor"
            else -> "Very Poor"
        }

    val deviationFromBenchmark: Float
        get() = rawValue - benchmarkValue

    val improvementSuggestion: String
        get() = recommendations.firstOrNull() ?: "Continue current practices"
}

/**
 * Factor interaction analysis
 */
@Parcelize
data class FactorInteraction(
    val factor1: String,
    val factor2: String,
    val correlationStrength: Float,
    val interactionType: InteractionType,
    val description: String
) : Parcelable

enum class InteractionType {
    SYNERGISTIC,
    COMPETITIVE,
    COMPENSATORY,
    INDEPENDENT
}

/**
 * AI analysis metadata for quality reports
 */
@Parcelize
data class AIAnalysisMetadata(
    val aiModelsUsed: List<String>,
    val totalProcessingTime: Long,
    val confidenceScore: Float,
    val personalizationApplied: Boolean,
    val dataSourcesAnalyzed: List<String>,
    val analysisVersion: String
) : Parcelable

/**
 * Report metadata
 */
@Parcelize
data class ReportMetadata(
    val version: String,
    val generator: String = "SomniAI",
    val analysisEngine: String = "Advanced Analytics Engine",
    val reportFormat: String = "Comprehensive",
    val customizations: List<String> = emptyList()
) : Parcelable

// ========== ENHANCED TREND ANALYSIS RESULTS ==========

/**
 * Comprehensive trend analysis with statistical validation and AI enhancements
 */
@Parcelize
data class TrendAnalysisResult(
    val analysisId: String = generateAnalysisId(),
    val analysisType: TrendAnalysisType,
    val timeRange: TimeRange,
    val analysisDate: Long = System.currentTimeMillis(),

    // Primary trend metrics
    val overallTrend: SleepTrend,
    val trendStrength: TrendStrength,
    val trendConfidence: Float,
    val statisticalSignificance: StatisticalSignificance,

    // Enhanced data points and metrics
    val dataPoints: List<TrendDataPoint>,
    val trendMetrics: TrendMetrics,
    val trendDirection: TrendDirection,

    // Individual metric trends
    val qualityTrend: MetricTrend,
    val durationTrend: MetricTrend,
    val efficiencyTrend: MetricTrend,
    val consistencyTrend: MetricTrend,
    val movementTrend: MetricTrend,
    val timingTrend: MetricTrend,

    // Advanced analytics
    val seasonalPatterns: List<SeasonalPattern>,
    val cyclicPatterns: List<CyclicPattern>,
    val seasonalPatternAnalysis: SeasonalPatternAnalysis?,
    val cyclicalBehaviors: List<CyclicalBehavior>,
    val changepoints: List<Changepoint>,
    val anomalies: List<TrendAnomaly>,
    val correlations: List<TrendCorrelation>,
    val projections: TrendProjections,
    val predictions: List<TrendPrediction>,

    // AI integration
    val aiPredictions: List<AITrendPrediction> = emptyList(),
    val patternRecognition: AIPatternRecognition? = null,

    // Statistical validation
    val statisticalValidation: StatisticalValidation,
    val modelMetadata: TrendModelMetadata,

    // Insights and explanations
    val trendInsights: List<TrendInsight>,
    val contributingFactors: List<ContributingFactor>,
    val recommendations: List<TrendRecommendation>,

    // Data quality and reliability
    val sampleSize: Int,
    val dataCompleteness: Float,
    val analysisReliability: AnalysisReliability
) : Parcelable {
    val trendSummary: String
        get() = "${overallTrend.displayName} trend with ${trendStrength.displayName} strength " +
                "(${(trendConfidence * 100).toInt()}% confidence)"

    val isSignificantTrend: Boolean
        get() = statisticalSignificance == StatisticalSignificance.SIGNIFICANT ||
                statisticalSignificance == StatisticalSignificance.HIGHLY_SIGNIFICANT

    val isSignificant: Boolean
        get() = isSignificantTrend

    val isReliable: Boolean
        get() = trendConfidence >= 0.7f && analysisReliability in listOf(
            AnalysisReliability.HIGH, AnalysisReliability.VERY_HIGH
        )

    val isPredictiveReliable: Boolean
        get() = trendConfidence >= 0.7f && sampleSize >= 14 && dataCompleteness >= 0.8f

    val keyFinding: String
        get() = trendInsights.firstOrNull()?.description ?: "Trend analysis in progress"

    fun getLatestValue(): Float? = dataPoints.maxByOrNull { it.timestamp }?.value

    fun getValueChange(): Float? {
        if (dataPoints.size < 2) return null
        val latest = dataPoints.maxByOrNull { it.timestamp }?.value ?: return null
        val earliest = dataPoints.minByOrNull { it.timestamp }?.value ?: return null
        return latest - earliest
    }

    fun getPercentageChange(): Float? {
        val change = getValueChange() ?: return null
        val earliest = dataPoints.minByOrNull { it.timestamp }?.value ?: return null
        return if (earliest != 0f) (change / earliest) * 100f else null
    }

    companion object {
        fun generateAnalysisId(): String = "TA-${System.currentTimeMillis()}"
    }
}

enum class TrendAnalysisType {
    QUALITY_TREND,
    DURATION_TREND,
    EFFICIENCY_TREND,
    MOVEMENT_TREND,
    NOISE_TREND,
    CONSISTENCY_TREND,
    TIMING_TREND,
    COMPOSITE_TREND
}

@Parcelize
data class TrendDataPoint(
    val timestamp: Long,
    val value: Float,
    val confidence: Float,
    val dataQuality: Float,
    val outlier: Boolean = false,
    val interpolated: Boolean = false,
    val metadata: Map<String, String> = emptyMap()
) : Parcelable

@Parcelize
data class TrendMetrics(
    val slope: Float,
    val rSquared: Float,
    val correlation: Float,
    val variance: Float,
    val standardDeviation: Float,
    val meanAbsoluteError: Float,
    val trendPersistence: Float,
    val volatility: Float
) : Parcelable

/**
 * Enhanced individual metric trend analysis
 */
@Parcelize
data class MetricTrend(
    val metricName: String,
    val direction: TrendDirection,
    val magnitude: Float, // Rate of change
    val significance: Float,
    val rSquared: Float, // Goodness of fit
    val startValue: Float,
    val endValue: Float,
    val changeRate: Float, // Change per unit time
    val volatility: Float,
    val outliers: List<OutlierPoint>
) : Parcelable {
    val totalChange: Float
        get() = endValue - startValue

    val percentageChange: Float
        get() = if (startValue != 0f) (totalChange / startValue) * 100f else 0f

    val isVolatile: Boolean
        get() = volatility > 0.3f

    val trendReliability: Float
        get() = (rSquared * (1f - volatility)).coerceIn(0f, 1f)

    val changeDescription: String
        get() = when {
            abs(percentageChange) < 5f -> "Stable"
            percentageChange > 20f -> "Significant increase"
            percentageChange > 10f -> "Moderate increase"
            percentageChange > 5f -> "Slight increase"
            percentageChange < -20f -> "Significant decrease"
            percentageChange < -10f -> "Moderate decrease"
            else -> "Slight decrease"
        }
}

/**
 * Enhanced seasonal pattern analysis
 */
@Parcelize
data class SeasonalPatternAnalysis(
    val hasSeasonalPattern: Boolean,
    val dominantPattern: SeasonalPattern,
    val patternStrength: Float,
    val seasonalComponents: Map<SeasonalComponent, Float>,
    val cycleLengths: List<Int>, // In days
    val seasonalTrends: Map<Season, Float>
) : Parcelable {
    val strongestSeason: Season
        get() = seasonalTrends.maxByOrNull { it.value }?.key ?: Season.UNKNOWN

    val weakestSeason: Season
        get() = seasonalTrends.minByOrNull { it.value }?.key ?: Season.UNKNOWN

    val seasonalVariability: Float
        get() = seasonalTrends.values.let { values ->
            if (values.isEmpty()) 0f
            else {
                val mean = values.average()
                sqrt(values.map { (it - mean).pow(2) }.average()).toFloat()
            }
        }
}

/**
 * Enhanced trend projections and forecasting
 */
@Parcelize
data class TrendProjections(
    val shortTermProjection: Projection, // Next 7 days
    val mediumTermProjection: Projection, // Next 30 days
    val longTermProjection: Projection, // Next 90 days
    val projectionConfidence: Float,
    val uncertaintyBounds: UncertaintyBounds,
    val assumptions: List<String>
) : Parcelable {
    val mostReliableProjection: Projection
        get() = when {
            projectionConfidence >= 0.8f -> longTermProjection
            projectionConfidence >= 0.6f -> mediumTermProjection
            else -> shortTermProjection
        }

    val projectionSummary: String
        get() = "Expected ${mostReliableProjection.trendDirection.displayName} trend " +
                "with ${(projectionConfidence * 100).toInt()}% confidence"
}

// ========== ENHANCED PATTERN ANALYSIS ==========

/**
 * Comprehensive pattern analysis result with AI-powered recognition
 */
@Parcelize
data class SleepPatternAnalysis(
    val analysisId: String = generatePatternId(),
    val patternType: PatternType,
    val timeRange: TimeRange,
    val patterns: List<DetectedPattern>,
    val patternStrength: PatternStrength,
    val patternConsistency: Float,
    val patternFrequency: PatternFrequency,
    val seasonalComponents: List<SeasonalComponent>,
    val cyclicComponents: List<CyclicComponent>,
    val irregularComponents: List<IrregularComponent>,
    val patternCorrelations: List<PatternCorrelation>,
    val patternPredictions: List<PatternPrediction>,
    val anomalyPatterns: List<AnomalyPattern>,
    val behavioralPatterns: List<BehavioralPattern>,
    val environmentalPatterns: List<EnvironmentalPattern>,
    val aiRecognitionMetadata: AIRecognitionMetadata,
    val validationResults: PatternValidationResults,
    val confidenceLevel: ConfidenceLevel,
    val insights: List<PatternInsight>,
    val analysisDate: Long = System.currentTimeMillis()
) : Parcelable {

    val hasStrongPatterns: Boolean
        get() = patternStrength in listOf(PatternStrength.STRONG, PatternStrength.VERY_STRONG)

    fun getPatternsByType(type: PatternType): List<DetectedPattern> {
        return patterns.filter { it.type == type }
    }

    fun getHighConfidencePatterns(): List<DetectedPattern> {
        return patterns.filter { it.confidence >= 0.8f }
    }

    companion object {
        fun generatePatternId(): String = "PA-${System.currentTimeMillis()}"
    }
}

enum class PatternType {
    SLEEP_SCHEDULE,
    QUALITY_PATTERN,
    DURATION_PATTERN,
    MOVEMENT_PATTERN,
    NOISE_PATTERN,
    SEASONAL_PATTERN,
    WEEKLY_PATTERN,
    MONTHLY_PATTERN,
    BEHAVIORAL_PATTERN,
    ENVIRONMENTAL_PATTERN
}

enum class PatternStrength {
    VERY_STRONG,
    STRONG,
    MODERATE,
    WEAK,
    VERY_WEAK,
    NO_PATTERN
}

enum class PatternFrequency {
    DAILY,
    WEEKLY,
    MONTHLY,
    SEASONAL,
    IRREGULAR,
    SPORADIC
}

@Parcelize
data class DetectedPattern(
    val id: String,
    val type: PatternType,
    val description: String,
    val strength: PatternStrength,
    val confidence: Float,
    val frequency: PatternFrequency,
    val duration: Long,
    val occurrences: List<PatternOccurrence>,
    val characteristics: Map<String, Float>,
    val triggers: List<PatternTrigger>,
    val consequences: List<PatternConsequence>,
    val stability: Float,
    val predictability: Float,
    val firstDetected: Long,
    val lastSeen: Long
) : Parcelable

// ========== ENHANCED STATISTICAL SUMMARIES ==========

/**
 * Comprehensive statistical summary for sleep data with AI insights
 */
@Parcelize
data class SleepStatisticalSummary(
    val summaryId: String = generateSummaryId(),
    val timeRange: TimeRange,
    val generatedAt: Long = System.currentTimeMillis(),

    // Descriptive statistics
    val descriptiveStats: DescriptiveStatistics,
    val distributionAnalysis: DistributionAnalysis,
    val correlationAnalysis: CorrelationAnalysis,

    // Quality metrics
    val qualityStatistics: QualityStatistics,
    val consistencyMetrics: ConsistencyMetrics,
    val performanceMetrics: PerformanceMetrics,

    // Behavioral patterns
    val patternAnalysis: PatternAnalysis,
    val habitAnalysis: HabitAnalysis,
    val anomalyDetection: AnomalyDetection,

    // Comparative analysis
    val periodComparisons: List<PeriodComparison>,
    val benchmarkAnalysis: BenchmarkAnalysis,

    // Enhanced AI insights
    val aiStatisticalInsights: List<AIStatisticalInsight> = emptyList(),
    val predictiveModeling: PredictiveModelingResult? = null,

    // Insights and findings
    val statisticalInsights: List<StatisticalInsight>,
    val significantFindings: List<SignificantFinding>,
    val dataQualityAssessment: DataQualityAssessment
) : Parcelable {
    val overallPerformanceScore: Float
        get() = (qualityStatistics.meanQuality + performanceMetrics.overallPerformance) / 2f

    val dataReliabilityScore: Float
        get() = dataQualityAssessment.overallReliability

    val summarySentence: String
        get() = "Analysis of ${descriptiveStats.sampleSize} sessions shows " +
                "${qualityStatistics.performanceLevel} sleep quality with " +
                "${consistencyMetrics.consistencyLevel} patterns"

    companion object {
        fun generateSummaryId(): String = "SS-${System.currentTimeMillis()}"
    }
}

/**
 * Enhanced descriptive statistics for sleep metrics
 */
@Parcelize
data class DescriptiveStatistics(
    val sampleSize: Int,
    val timeSpanDays: Int,

    // Duration statistics
    val durationStats: MetricStatistics,
    val qualityStats: MetricStatistics,
    val efficiencyStats: MetricStatistics,
    val latencyStats: MetricStatistics,

    // Movement and noise statistics
    val movementStats: MetricStatistics,
    val noiseStats: MetricStatistics,

    // Timing statistics
    val bedtimeStats: MetricStatistics,
    val waketimeStats: MetricStatistics
) : Parcelable {
    val hasAdequateSample: Boolean
        get() = sampleSize >= 7 && timeSpanDays >= 7

    val averageSessionsPerWeek: Float
        get() = if (timeSpanDays > 0) (sampleSize.toFloat() / timeSpanDays) * 7f else 0f

    val dataFrequency: DataFrequency
        get() = when {
            averageSessionsPerWeek >= 6f -> DataFrequency.DAILY
            averageSessionsPerWeek >= 3f -> DataFrequency.REGULAR
            averageSessionsPerWeek >= 1f -> DataFrequency.OCCASIONAL
            else -> DataFrequency.SPARSE
        }
}

/**
 * Enhanced statistics for individual metrics
 */
@Parcelize
data class MetricStatistics(
    val metricName: String,
    val count: Int,
    val mean: Float,
    val median: Float,
    val mode: Float?,
    val standardDeviation: Float,
    val variance: Float,
    val minimum: Float,
    val maximum: Float,
    val range: Float,
    val quartiles: Quartiles,
    val percentiles: Map<Int, Float>,
    val skewness: Float,
    val kurtosis: Float,
    val outliers: List<Float>
) : Parcelable {
    val coefficientOfVariation: Float
        get() = if (mean != 0f) standardDeviation / mean else Float.MAX_VALUE

    val isNormallyDistributed: Boolean
        get() = abs(skewness) < 1f && abs(kurtosis - 3f) < 2f

    val variabilityLevel: VariabilityLevel
        get() = when {
            coefficientOfVariation < 0.1f -> VariabilityLevel.VERY_LOW
            coefficientOfVariation < 0.2f -> VariabilityLevel.LOW
            coefficientOfVariation < 0.3f -> VariabilityLevel.MODERATE
            coefficientOfVariation < 0.5f -> VariabilityLevel.HIGH
            else -> VariabilityLevel.VERY_HIGH
        }

    val stabilityScore: Float
        get() = (1f / (1f + coefficientOfVariation)).coerceIn(0f, 1f)

    val outlierPercentage: Float
        get() = if (count > 0) (outliers.size.toFloat() / count) * 100f else 0f
}

// ========== ENHANCED COMPARATIVE METRICS ==========

/**
 * Comprehensive comparative analysis results with AI enhancements
 */
@Parcelize
data class ComparativeAnalysisResult(
    val analysisId: String = generateComparisonId(),
    val comparisonType: ComparisonType,
    val baselineInfo: BaselineInfo,
    val comparisonDate: Long = System.currentTimeMillis(),

    // Primary comparisons
    val personalComparison: PersonalPerformanceComparison,
    val populationComparison: PopulationPerformanceComparison?,
    val temporalComparison: TemporalPerformanceComparison,
    val cohortComparison: CohortPerformanceComparison? = null,
    val goalComparison: GoalPerformanceComparison?,

    // Detailed metric comparisons
    val metricComparisons: List<MetricComparison>,
    val rankingAnalysis: RankingAnalysis,
    val percentileAnalysis: PercentileAnalysis,

    // Performance insights
    val performanceGaps: List<PerformanceGap>,
    val competitiveAdvantages: List<CompetitiveAdvantage>,
    val improvementOpportunities: List<ComparisonBasedRecommendation>,

    // AI enhancements
    val aiComparativeInsights: List<AIComparativeInsight> = emptyList(),
    val predictiveComparison: PredictiveComparisonResult? = null,

    // Context and reliability
    val comparisonContext: ComparisonContext,
    val reliabilityMetrics: ComparisonReliabilityMetrics,
    val visualizationData: ComparisonVisualizationData
) : Parcelable {
    val overallPerformanceRating: PerformanceRating
        get() = when {
            personalComparison.overallImprovement >= 20f -> PerformanceRating.EXCEPTIONAL
            personalComparison.overallImprovement >= 10f -> PerformanceRating.EXCELLENT
            personalComparison.overallImprovement >= 5f -> PerformanceRating.GOOD
            personalComparison.overallImprovement >= 0f -> PerformanceRating.AVERAGE
            personalComparison.overallImprovement >= -10f -> PerformanceRating.BELOW_AVERAGE
            else -> PerformanceRating.POOR
        }

    val keyComparisonInsight: String
        get() = when (overallPerformanceRating) {
            PerformanceRating.EXCEPTIONAL -> "Outstanding performance across all metrics"
            PerformanceRating.EXCELLENT -> "Strong performance with consistent improvements"
            PerformanceRating.GOOD -> "Above-average performance with growth potential"
            PerformanceRating.AVERAGE -> "Performance in line with expectations"
            PerformanceRating.BELOW_AVERAGE -> "Performance below baseline with improvement needed"
            PerformanceRating.POOR -> "Significant performance challenges identified"
        }

    fun getOverallPerformanceScore(): Float {
        return metricComparisons.map { it.performanceRatio }.average().toFloat()
    }

    fun getTopPerformanceGaps(count: Int = 3): List<PerformanceGap> {
        return performanceGaps
            .sortedByDescending { it.impact.value }
            .take(count)
    }

    fun getTopOpportunities(count: Int = 3): List<ComparisonBasedRecommendation> {
        return improvementOpportunities
            .sortedWith(
                compareByDescending<ComparisonBasedRecommendation> { it.priority.value }
                    .thenByDescending { it.expectedImprovement }
            )
            .take(count)
    }

    companion object {
        fun generateComparisonId(): String = "CA-${System.currentTimeMillis()}"
    }
}

/**
 * Enhanced personal performance comparison against historical data
 */
@Parcelize
data class PersonalPerformanceComparison(
    val timeframe: String,
    val currentPeriodMetrics: PeriodMetrics,
    val baselinePeriodMetrics: PeriodMetrics,

    // Improvement calculations
    val qualityImprovement: Float,
    val durationImprovement: Float,
    val efficiencyImprovement: Float,
    val consistencyImprovement: Float,
    val overallImprovement: Float,

    // Ranking within personal history
    val qualityPercentile: Float,
    val durationPercentile: Float,
    val efficiencyPercentile: Float,

    // Streak analysis
    val currentStreaks: Map<String, Int>,
    val bestStreaks: Map<String, Int>,
    val streakAnalysis: StreakAnalysis,

    // Enhanced metrics
    val personalBests: Map<String, PersonalBest>,
    val improvementRate: Float,
    val consistencyScore: Float
) : Parcelable {
    val isImproving: Boolean
        get() = overallImprovement > 5f

    val isDeclining: Boolean
        get() = overallImprovement < -5f

    val isStable: Boolean
        get() = !isImproving && !isDeclining

    val bestMetric: String
        get() = mapOf(
            "Quality" to qualityImprovement,
            "Duration" to durationImprovement,
            "Efficiency" to efficiencyImprovement,
            "Consistency" to consistencyImprovement
        ).maxByOrNull { it.value }?.key ?: "None"

    val improvementSummary: String
        get() = when {
            isImproving -> "Sleep quality improving by ${overallImprovement.toInt()}%"
            isDeclining -> "Sleep quality declining by ${abs(overallImprovement).toInt()}%"
            else -> "Sleep quality remains stable"
        }
}

/**
 * Enhanced individual metric comparison details
 */
@Parcelize
data class MetricComparison(
    val metricName: String,
    val currentValue: Float,
    val baselineValue: Float,
    val populationMean: Float?,
    val populationPercentile: Float?,
    val improvement: Float,
    val improvementPercentage: Float,
    val trend: TrendDirection,
    val significance: ComparisonSignificance,
    val contextualRanking: ContextualRanking,
    val performanceRatio: Float = if (baselineValue != 0f) currentValue / baselineValue else 1f
) : Parcelable {
    val isSignificantImprovement: Boolean
        get() = improvement > 0 && significance == ComparisonSignificance.SIGNIFICANT

    val isSignificantDecline: Boolean
        get() = improvement < 0 && significance == ComparisonSignificance.SIGNIFICANT

    val performanceLevel: String
        get() = when (populationPercentile) {
            null -> "Unknown"
            in 90f..100f -> "Top 10%"
            in 75f..90f -> "Above Average"
            in 25f..75f -> "Average"
            in 10f..25f -> "Below Average"
            else -> "Bottom 10%"
        }

    val comparisonSummary: String
        get() = "$metricName: ${if (improvement >= 0) "+" else ""}${improvement.toInt()}% " +
                "vs baseline (${performanceLevel})"
}

// ========== ENHANCED INSIGHT AND RECOMMENDATION MODELS ==========

/**
 * Enhanced quality insight with detailed analysis
 */
@Parcelize
data class QualityInsight(
    val insightId: String,
    val category: InsightCategory,
    val type: InsightType,
    val priority: Priority,
    val confidence: Float,
    val title: String,
    val description: String,
    val evidence: List<String>,
    val dataPoints: List<DataPoint>,
    val relatedMetrics: List<String>,
    val implications: List<String> = emptyList(),
    val contextualFactors: List<String> = emptyList(),
    val timeRelevance: TimeRelevance = TimeRelevance.ONGOING,
    val personalizationScore: Float = 0f,
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable {
    val isHighPriority: Boolean
        get() = priority == Priority.HIGH && confidence >= 0.7f

    val isActionable: Boolean
        get() = type == InsightType.ACTIONABLE && confidence >= 0.6f

    val evidenceStrength: EvidenceStrength
        get() = when {
            evidence.size >= 3 && confidence >= 0.8f -> EvidenceStrength.STRONG
            evidence.size >= 2 && confidence >= 0.6f -> EvidenceStrength.MODERATE
            evidence.isNotEmpty() && confidence >= 0.4f -> EvidenceStrength.WEAK
            else -> EvidenceStrength.INSUFFICIENT
        }
}

/**
 * Enhanced quality recommendation with implementation guidance
 */
@Parcelize
data class QualityRecommendation(
    val recommendationId: String,
    val category: RecommendationCategory,
    val priority: Priority,
    val title: String,
    val description: String,
    val actionItems: List<ActionItem>,
    val expectedImpact: ExpectedImpact,
    val implementationDifficulty: ImplementationDifficulty,
    val timeToImpact: TimeToImpact,
    val confidence: Float,
    val personalizationScore: Float,
    val relatedInsights: List<String>,
    val successMetrics: List<String>,
    val prerequisites: List<String> = emptyList(),
    val resources: List<RecommendationResource> = emptyList(),
    val alternatives: List<AlternativeRecommendation> = emptyList(),
    val contraindications: List<String> = emptyList()
) : Parcelable {
    val isHighImpact: Boolean
        get() = expectedImpact.value >= 0.7f && confidence >= 0.7f

    val isEasyToImplement: Boolean
        get() = implementationDifficulty == ImplementationDifficulty.LOW

    val quickWin: Boolean
        get() = isEasyToImplement && expectedImpact.value >= 0.5f

    val priorityScore: Float
        get() = (priority.value + expectedImpact.value - implementationDifficulty.value) * confidence

    val implementationSummary: String
        get() = "${implementationDifficulty.displayName} implementation, " +
                "${expectedImpact.displayName} impact, " +
                "${timeToImpact.displayName} results"
}

// ========== ENHANCED ANOMALY DETECTION ==========

/**
 * Comprehensive anomaly detection result
 */
@Parcelize
data class AnomalyDetectionResult(
    val anomaliesDetected: List<DetectedAnomaly>,
    val anomalyScore: Float,
    val anomalyType: AnomalyType,
    val severity: AnomalySeverity,
    val confidence: Float,
    val analysisMethod: String,
    val recommendations: List<String>
) : Parcelable

@Parcelize
data class DetectedAnomaly(
    val timestamp: Long,
    val metric: String,
    val expectedValue: Float,
    val actualValue: Float,
    val deviation: Float,
    val severity: AnomalySeverity,
    val confidence: Float,
    val possibleCauses: List<String>
) : Parcelable

enum class AnomalyType {
    POINT_ANOMALY,
    CONTEXTUAL_ANOMALY,
    COLLECTIVE_ANOMALY
}

enum class AnomalySeverity {
    MINOR,
    MODERATE,
    SIGNIFICANT,
    CRITICAL
}

// ========== SUPPORTING DATA CLASSES ==========

@Parcelize
data class TimeRange(
    val startDate: Long,
    val endDate: Long,
    val description: String,
    val timezone: String = TimeZone.getDefault().id
) : Parcelable {
    val durationDays: Int
        get() = ((endDate - startDate) / (24 * 60 * 60 * 1000L)).toInt()

    val durationWeeks: Int
        get() = durationDays / 7

    val durationMs: Long
        get() = endDate - startDate

    val isRecentPeriod: Boolean
        get() = endDate >= System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000L)

    companion object {
        fun lastDays(days: Int): TimeRange {
            val end = System.currentTimeMillis()
            val start = end - (days * 24 * 60 * 60 * 1000L)
            return TimeRange(start, end, "Last $days days")
        }

        fun lastWeeks(weeks: Int): TimeRange {
            val end = System.currentTimeMillis()
            val start = end - (weeks * 7 * 24 * 60 * 60 * 1000L)
            return TimeRange(start, end, "Last $weeks weeks")
        }

        fun lastMonths(months: Int): TimeRange {
            val end = System.currentTimeMillis()
            val start = end - (months * 30L * 24 * 60 * 60 * 1000L)
            return TimeRange(start, end, "Last $months months")
        }
    }
}

@Parcelize
data class Quartiles(
    val q1: Float,
    val q2: Float, // Median
    val q3: Float
) : Parcelable {
    val iqr: Float
        get() = q3 - q1
}

@Parcelize
data class UncertaintyBounds(
    val lowerBound: Float,
    val upperBound: Float,
    val confidenceInterval: Float
) : Parcelable {
    val range: Float
        get() = upperBound - lowerBound
}

@Parcelize
data class OutlierPoint(
    val value: Float,
    val timestamp: Long,
    val severity: OutlierSeverity,
    val explanation: String?
) : Parcelable

@Parcelize
data class Changepoint(
    val timestamp: Long,
    val beforeValue: Float,
    val afterValue: Float,
    val magnitude: Float,
    val confidence: Float,
    val explanation: String
) : Parcelable

@Parcelize
data class Projection(
    val targetDate: Long,
    val projectedValue: Float,
    val trendDirection: TrendDirection,
    val confidence: Float,
    val factors: List<String>
) : Parcelable

@Parcelize
data class PeriodMetrics(
    val averageQuality: Float,
    val averageDuration: Float,
    val averageEfficiency: Float,
    val consistency: Float,
    val sessionCount: Int,
    val dataCompleteness: Float = 1.0f
) : Parcelable

@Parcelize
data class StreakAnalysis(
    val longestQualityStreak: Int,
    val longestConsistencyStreak: Int,
    val currentQualityStreak: Int,
    val currentConsistencyStreak: Int,
    val streakTrend: TrendDirection,
    val streakDirection: TrendDirection,
    val streakProbability: Float = 0f
) : Parcelable

@Parcelize
data class PersonalBest(
    val value: Float,
    val date: Long,
    val context: String,
    val daysAgo: Int
) : Parcelable

@Parcelize
data class ActionItem(
    val action: String,
    val description: String? = null,
    val difficulty: ImplementationDifficulty,
    val timeRequired: String,
    val frequency: String? = null,
    val expectedOutcome: String,
    val measurableGoal: String? = null,
    val reminders: List<String> = emptyList()
) : Parcelable

@Parcelize
data class ExpectedImpact(
    val magnitude: Float,
    val confidence: Float,
    val timeframe: String,
    val affectedMetrics: List<String>
) : Parcelable {
    val value: Float
        get() = magnitude * confidence

    val displayName: String
        get() = when {
            magnitude >= 0.8f -> "High"
            magnitude >= 0.5f -> "Medium"
            else -> "Low"
        }

    companion object {
        val HIGH = ExpectedImpact(0.8f, 0.8f, "2-4 weeks", emptyList())
        val MEDIUM = ExpectedImpact(0.5f, 0.7f, "1-3 weeks", emptyList())
        val LOW = ExpectedImpact(0.3f, 0.6f, "3-6 weeks", emptyList())
    }
}

@Parcelize
data class RecommendationResource(
    val type: ResourceType,
    val title: String,
    val description: String,
    val url: String? = null,
    val internalReference: String? = null
) : Parcelable

enum class ResourceType {
    ARTICLE,
    VIDEO,
    TOOL,
    APP_FEATURE,
    EXTERNAL_SERVICE,
    PROFESSIONAL_CONSULTATION
}

@Parcelize
data class AlternativeRecommendation(
    val title: String,
    val description: String,
    val whenToUse: String,
    val expectedImpact: ExpectedImpact
) : Parcelable

@Parcelize
data class DataPoint(
    val value: Float,
    val timestamp: Long,
    val metric: String = "",
    val unit: String = ""
) : Parcelable

// ========== ENHANCED ENUMS ==========

enum class ReportType(val displayName: String) {
    DAILY("Daily Report"),
    WEEKLY("Weekly Report"),
    MONTHLY("Monthly Report"),
    QUARTERLY("Quarterly Report"),
    YEARLY("Yearly Report"),
    CUSTOM("Custom Period Report"),
    SESSION("Individual Session Report"),
    CUSTOM_RANGE("Custom Range")
}

enum class QualityGrade(val displayName: String, val range: ClosedFloatingPointRange<Float>) {
    EXCELLENT("Excellent", 9.0f..10.0f),
    VERY_GOOD("Very Good", 8.0f..8.9f),
    GOOD("Good", 7.0f..7.9f),
    FAIR("Fair", 6.0f..6.9f),
    POOR("Poor", 4.0f..5.9f),
    VERY_POOR("Very Poor", 0.0f..3.9f),
    A_PLUS("A+", 9.5f..10f),
    A("A", 8.5f..9.5f),
    B_PLUS("B+", 7.5f..8.5f),
    B("B", 6.5f..7.5f),
    C_PLUS("C+", 5.5f..6.5f),
    C("C", 4.5f..5.5f),
    D_PLUS("D+", 3.5f..4.5f),
    D("D", 2.5f..3.5f),
    F("F", 0f..2.5f);

    companion object {
        fun fromScore(score: Float): QualityGrade {
            return values().find { score in it.range } ?: POOR
        }
    }
}

enum class TrendStrength(val displayName: String) {
    VERY_STRONG("Very Strong"),
    STRONG("Strong"),
    MODERATE("Moderate"),
    WEAK("Weak"),
    NEGLIGIBLE("Negligible"),
    NO_TREND("No Trend")
}

enum class StatisticalSignificance(val displayName: String) {
    HIGHLY_SIGNIFICANT("Highly Significant"),
    SIGNIFICANT("Significant"),
    MARGINALLY_SIGNIFICANT("Marginally Significant"),
    NOT_SIGNIFICANT("Not Significant"),
    INSUFFICIENT_DATA("Insufficient Data")
}

enum class ConfidenceLevel(val displayName: String, val threshold: Float) {
    VERY_HIGH("Very High", 0.9f),
    HIGH("High", 0.8f),
    MODERATE("Moderate", 0.6f),
    LOW("Low", 0.4f),
    VERY_LOW("Very Low", 0.0f);

    companion object {
        fun fromValue(confidence: Float): ConfidenceLevel {
            return values().findLast { confidence >= it.threshold } ?: VERY_LOW
        }

        fun fromScore(score: Float): ConfidenceLevel = fromValue(score)
    }
}

enum class Priority(val displayName: String, val value: Float) {
    HIGH("High", 3f),
    MEDIUM("Medium", 2f),
    LOW("Low", 1f),
    URGENT("Urgent", 4f),
    INFO("Info", 0.5f)
}

enum class FactorImpact(val displayName: String, val value: Float) {
    CRITICAL("Critical", 1f),
    HIGH("High", 0.8f),
    MEDIUM("Medium", 0.6f),
    LOW("Low", 0.4f),
    MINIMAL("Minimal", 0.2f),
    VERY_HIGH("Very High", 0.9f)
}

enum class TrendDirection(val displayName: String) {
    STRONGLY_IMPROVING("Strongly Improving"),
    IMPROVING("Improving"),
    STABLE("Stable"),
    DECLINING("Declining"),
    STRONGLY_DECLINING("Strongly Declining"),
    INSUFFICIENT_DATA("Insufficient Data"),
    VOLATILE("Volatile"),
    UNKNOWN("Unknown")
}

enum class SleepTrend(val displayName: String) {
    IMPROVING("Improving"),
    STABLE("Stable"),
    DECLINING("Declining"),
    INSUFFICIENT_DATA("Insufficient Data")
}

enum class SeasonalPattern(val displayName: String) {
    WEEKLY("Weekly Pattern"),
    MONTHLY("Monthly Pattern"),
    SEASONAL("Seasonal Pattern"),
    NONE("No Pattern")
}

enum class Season {
    SPRING, SUMMER, FALL, WINTER, UNKNOWN
}

enum class SeasonalComponent {
    TREND, SEASONAL, RESIDUAL
}

enum class VariabilityLevel(val displayName: String) {
    VERY_LOW("Very Low"),
    LOW("Low"),
    MODERATE("Moderate"),
    HIGH("High"),
    VERY_HIGH("Very High")
}

enum class DataFrequency(val displayName: String) {
    DAILY("Daily"),
    REGULAR("Regular"),
    OCCASIONAL("Occasional"),
    SPARSE("Sparse")
}

enum class ComparisonType(val displayName: String) {
    PERSONAL_HISTORICAL("Personal Historical"),
    PERSONAL_GOALS("Personal Goals"),
    POPULATION_AVERAGE("Population Average"),
    AGE_COHORT("Age Cohort"),
    DEMOGRAPHIC_SIMILAR("Demographic Similar"),
    TOP_PERFORMERS("Top Performers"),
    TEMPORAL_PERIODS("Temporal Periods"),
    POPULATION_BENCHMARK("Population Benchmark"),
    GOAL_BASED("Goal-Based"),
    PEER_GROUP("Peer Group")
}

enum class PerformanceRating(val displayName: String) {
    EXCEPTIONAL("Exceptional"),
    EXCELLENT("Excellent"),
    GOOD("Good"),
    AVERAGE("Average"),
    BELOW_AVERAGE("Below Average"),
    POOR("Poor")
}

enum class ComparisonSignificance(val displayName: String) {
    HIGHLY_SIGNIFICANT("Highly Significant"),
    SIGNIFICANT("Significant"),
    NOT_SIGNIFICANT("Not Significant")
}

enum class InsightType(val displayName: String) {
    ACTIONABLE("Actionable"),
    INFORMATIONAL("Informational"),
    WARNING("Warning"),
    ACHIEVEMENT("Achievement"),
    TREND_ALERT("Trend Alert"),
    PATTERN_RECOGNITION("Pattern Recognition"),
    ANOMALY_DETECTION("Anomaly Detection"),
    RECOMMENDATION_TRIGGER("Recommendation Trigger"),
    GOAL_PROGRESS("Goal Progress"),
    COMPARATIVE_ANALYSIS("Comparative Analysis")
}

enum class RecommendationCategory(val displayName: String) {
    SLEEP_HYGIENE("Sleep Hygiene"),
    ENVIRONMENT_OPTIMIZATION("Environment Optimization"),
    SCHEDULE_ADJUSTMENT("Schedule Adjustment"),
    LIFESTYLE_CHANGE("Lifestyle Change"),
    BEHAVIORAL_MODIFICATION("Behavioral Modification"),
    STRESS_MANAGEMENT("Stress Management"),
    EXERCISE_TIMING("Exercise Timing"),
    NUTRITION_TIMING("Nutrition Timing"),
    TECHNOLOGY_USAGE("Technology Usage"),
    MEDICAL_CONSULTATION("Medical Consultation"),
    ENVIRONMENT("Environment"),
    SCHEDULE("Schedule"),
    LIFESTYLE("Lifestyle"),
    MEDICAL("Medical")
}

enum class ImplementationDifficulty(val displayName: String, val value: Float) {
    LOW("Easy", 1f),
    MEDIUM("Moderate", 2f),
    HIGH("Difficult", 3f),
    VERY_EASY("Very Easy", 0.5f),
    VERY_DIFFICULT("Very Difficult", 4f)
}

enum class TimeToImpact(val displayName: String) {
    IMMEDIATE("Immediate"),
    SHORT_TERM("Short-term"),
    MEDIUM_TERM("Medium-term"),
    LONG_TERM("Long-term")
}

enum class TimeRelevance {
    IMMEDIATE,
    SHORT_TERM,
    ONGOING,
    LONG_TERM,
    HISTORICAL
}

enum class EvidenceStrength(val displayName: String) {
    STRONG("Strong"),
    MODERATE("Moderate"),
    WEAK("Weak"),
    INSUFFICIENT("Insufficient")
}

enum class OutlierSeverity(val displayName: String) {
    EXTREME("Extreme"),
    MODERATE("Moderate"),
    MILD("Mild")
}

enum class AnalysisReliability(val displayName: String) {
    VERY_HIGH("Very High"),
    HIGH("High"),
    MODERATE("Moderate"),
    LOW("Low"),
    VERY_LOW("Very Low")
}

// ========== MISSING ENUM CLASSES (ADD HERE) ==========

enum class InsightGenerationType(val displayName: String) {
    AUTOMATED("Automated"),
    ON_DEMAND("On Demand"),
    SCHEDULED("Scheduled"),
    TRIGGERED("Triggered"),
    MANUAL("Manual")
}

enum class InsightCategory(val displayName: String) {
    DURATION("Duration"),
    QUALITY("Quality"),
    ENVIRONMENT("Environment"),
    MOVEMENT("Movement"),
    CONSISTENCY("Consistency"),
    TRENDS("Trends"),
    SCHEDULE("Schedule"),
    LIFESTYLE("Lifestyle"),
    HEALTH("Health"),
    OPTIMIZATION("Optimization")
}

// ========== ADDITIONAL SUPPORTING CLASSES ==========
// Enhanced analysis components
@Parcelize data class DurationAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val durationScore: Float = score) : Parcelable
// ... rest of your existing supporting classes ...

// ========== ADDITIONAL SUPPORTING CLASSES ==========

// Enhanced analysis components
@Parcelize data class DurationAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val durationScore: Float = score) : Parcelable
@Parcelize data class EfficiencyAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val combinedEfficiency: Float = score) : Parcelable
@Parcelize data class MovementAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val movementScore: Float = score) : Parcelable
@Parcelize data class NoiseAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val noiseScore: Float = score) : Parcelable
@Parcelize data class ConsistencyAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val consistencyScore: Float = score) : Parcelable
@Parcelize data class TimingAnalysis(val score: Float = 0f, val insights: List<String> = emptyList(), val timingScore: Float = score) : Parcelable

// Enhanced comparison metrics
@Parcelize data class PersonalComparisonMetrics(val improvement: Float = 0f, val qualityImprovement: Float = improvement) : Parcelable
@Parcelize data class PopulationComparisonMetrics(val percentile: Float = 50f) : Parcelable
@Parcelize data class BenchmarkComparisonMetrics(val percentile: Float = 50f) : Parcelable
@Parcelize data class QualityTrendAnalysis(val trend: TrendDirection, val trendStrength: TrendStrength = TrendStrength.MODERATE) : Parcelable

// Enhanced opportunity and strength tracking
@Parcelize data class StrengthArea(val area: String, val score: Float) : Parcelable
@Parcelize data class ImprovementOpportunity(val area: String, val priority: Priority, val impact: Float, val expectedImprovement: Float = impact) : Parcelable
@Parcelize data class GoalProgressAnalysis(val progress: Float) : Parcelable

// Enhanced data quality and validation
@Parcelize data class DataQualityMetrics(
    val completeness: Float = 1f,
    val accuracy: Float = 1f,
    val consistency: Float = 1f,
    val timeliness: Float = 1.0f,
    val validity: Float = 1.0f,
    val dataPointCount: Int = 0,
    val missingDataPercentage: Float = 0f,
    val outlierPercentage: Float = 0f,
    val qualityScore: Float = (completeness + accuracy + consistency + timeliness + validity) / 5f
) : Parcelable {
    val isHighQuality: Boolean
        get() = qualityScore >= 0.8f

    val isReliable: Boolean
        get() = completeness >= 0.7f && accuracy >= 0.8f
}

// AI-specific analysis components
@Parcelize data class AIStatisticalInsight(val insight: String, val confidence: Float, val aiModel: String) : Parcelable
@Parcelize data class PredictiveModelingResult(val predictions: List<String>, val accuracy: Float) : Parcelable
@Parcelize data class AIComparativeInsight(val insight: String, val confidence: Float, val category: String) : Parcelable
@Parcelize data class PredictiveComparisonResult(val prediction: String, val confidence: Float) : Parcelable
@Parcelize data class AITrendPrediction(val prediction: String, val confidence: Float, val timeframe: String) : Parcelable
@Parcelize data class AIPatternRecognition(val patterns: List<String>, val confidence: Float) : Parcelable

// Pattern analysis components
@Parcelize data class CyclicPattern(val pattern: String, val strength: Float) : Parcelable
@Parcelize data class TrendAnomaly(val timestamp: Long, val severity: Float) : Parcelable
@Parcelize data class TrendCorrelation(val metric1: String, val metric2: String, val correlation: Float) : Parcelable
@Parcelize data class TrendPrediction(val timestamp: Long, val predictedValue: Float, val confidence: Float) : Parcelable
@Parcelize data class StatisticalValidation(val pValue: Float, val isSignificant: Boolean) : Parcelable
@Parcelize data class TrendModelMetadata(val modelType: String, val accuracy: Float) : Parcelable
@Parcelize data class TrendInsight(val description: String, val confidence: Float) : Parcelable
@Parcelize data class ContributingFactor(val factor: String, val impact: Float) : Parcelable
@Parcelize data class TrendRecommendation(val recommendation: String, val priority: Priority) : Parcelable

// Pattern detection components
@Parcelize data class CyclicalBehavior(val pattern: String, val strength: Float) : Parcelable
@Parcelize data class SeasonalComponent(val component: String, val strength: Float) : Parcelable
@Parcelize data class CyclicComponent(val component: String, val frequency: Float) : Parcelable
@Parcelize data class IrregularComponent(val component: String, val variance: Float) : Parcelable
@Parcelize data class PatternCorrelation(val pattern1: String, val pattern2: String, val correlation: Float) : Parcelable
@Parcelize data class PatternPrediction(val pattern: String, val probability: Float) : Parcelable
@Parcelize data class AnomalyPattern(val pattern: String, val frequency: Float) : Parcelable
@Parcelize data class BehavioralPattern(val behavior: String, val correlation: Float) : Parcelable
@Parcelize data class EnvironmentalPattern(val factor: String, val impact: Float) : Parcelable
@Parcelize data class AIRecognitionMetadata(val model: String, val confidence: Float) : Parcelable
@Parcelize data class PatternValidationResults(val isValid: Boolean, val score: Float) : Parcelable
@Parcelize data class PatternInsight(val insight: String, val confidence: Float) : Parcelable
@Parcelize data class PatternOccurrence(val timestamp: Long, val strength: Float) : Parcelable
@Parcelize data class PatternTrigger(val trigger: String, val probability: Float) : Parcelable
@Parcelize data class PatternConsequence(val consequence: String, val impact: Float) : Parcelable

// Statistical analysis components
@Parcelize data class DistributionAnalysis(val type: String = "Normal") : Parcelable
@Parcelize data class CorrelationAnalysis(val correlations: Map<String, Float> = emptyMap()) : Parcelable
@Parcelize data class QualityStatistics(val meanQuality: Float = 0f, val performanceLevel: String = "Average") : Parcelable
@Parcelize data class ConsistencyMetrics(val consistencyLevel: String = "Moderate") : Parcelable
@Parcelize data class PerformanceMetrics(val overallPerformance: Float = 0f) : Parcelable
@Parcelize data class PatternAnalysis(val patterns: List<String> = emptyList()) : Parcelable
@Parcelize data class HabitAnalysis(val habits: List<String> = emptyList()) : Parcelable
@Parcelize data class AnomalyDetection(val anomalies: List<String> = emptyList()) : Parcelable
@Parcelize data class PeriodComparison(val period: String, val improvement: Float) : Parcelable
@Parcelize data class BenchmarkAnalysis(val benchmarks: Map<String, Float> = emptyMap()) : Parcelable
@Parcelize data class StatisticalInsight(val insight: String, val significance: Float) : Parcelable
@Parcelize data class SignificantFinding(val finding: String, val pValue: Float) : Parcelable
@Parcelize data class DataQualityAssessment(val overallReliability: Float = 0.8f) : Parcelable

// Comparison analysis components
@Parcelize data class BaselineInfo(val description: String, val date: Long) : Parcelable
@Parcelize data class PopulationPerformanceComparison(val percentile: Float = 50f) : Parcelable
@Parcelize data class TemporalPerformanceComparison(val improvement: Float = 0f) : Parcelable
@Parcelize data class CohortPerformanceComparison(val ranking: Int, val total: Int) : Parcelable
@Parcelize data class GoalPerformanceComparison(val progress: Float = 0f) : Parcelable
@Parcelize data class RankingAnalysis(val rank: Int = 1, val totalPopulation: Int = 1, val totalParticipants: Int = totalPopulation) : Parcelable
@Parcelize data class PercentileAnalysis(val percentiles: Map<String, Float> = emptyMap()) : Parcelable
@Parcelize data class PerformanceGap(val metric: String, val gap: Float, val impact: ExpectedImpact) : Parcelable
@Parcelize data class CompetitiveAdvantage(val advantage: String, val strength: Float) : Parcelable
@Parcelize data class ComparisonBasedRecommendation(val recommendation: String, val rationale: String, val priority: Priority = Priority.MEDIUM, val expectedImprovement: Float = 0.5f) : Parcelable
@Parcelize data class ComparisonContext(val context: String, val description: String = context) : Parcelable
@Parcelize data class ComparisonReliabilityMetrics(val reliability: Float = 0.8f) : Parcelable
@Parcelize data class ComparisonVisualizationData(val chartType: String = "comparison", val data: Map<String, Float> = emptyMap()) : Parcelable
@Parcelize data class ContextualRanking(val rank: String, val context: String) : Parcelable

// Confidence and validation
@Parcelize
data class ConfidenceInterval(
    val lowerBound: Float,
    val upperBound: Float,
    val confidenceLevel: Float,
    val meanValue: Float
) : Parcelable {
    val intervalWidth: Float
        get() = upperBound - lowerBound

    val marginOfError: Float
        get() = intervalWidth / 2f

    fun contains(value: Float): Boolean = value in lowerBound..upperBound
}