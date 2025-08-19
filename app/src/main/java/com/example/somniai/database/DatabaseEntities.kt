package com.example.somniai.database

import androidx.room.*
import com.example.somniai.data.*
import java.util.Date



/**
 * Primary sleep session entity with AI analysis integration
 * Stores complete sleep session data with analytics and AI-generated insights
 */

/**
 * Enhanced enum for data integrity status tracking
 */
enum class DataIntegrityStatus {
    EXCELLENT,
    GOOD,
    FAIR,
    POOR
}

/**
 * Sleep phase enum
 */
enum class SleepPhase {
    AWAKE,
    LIGHT_SLEEP,
    DEEP_SLEEP,
    REM_SLEEP,
    UNKNOWN
}

@Entity(
    tableName = "sleep_sessions",
    indices = [
        Index(value = ["start_time"], name = "idx_session_start_time"),
        Index(value = ["end_time"], name = "idx_session_end_time"),
        Index(value = ["quality_score"], name = "idx_session_quality"),
        Index(value = ["ai_analysis_status"], name = "idx_session_ai_status"),
        Index(value = ["ai_analysis_completed_at"], name = "idx_session_ai_completed")
    ]
)
data class SleepSessionEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    // Basic session information
    @ColumnInfo(name = "start_time")
    val startTime: Long,

    @ColumnInfo(name = "end_time")
    val endTime: Long? = null,

    @ColumnInfo(name = "total_duration")
    val totalDuration: Long = 0,

    @ColumnInfo(name = "session_duration")
    val sessionDuration: Long = 0L,

    // Sleep timing metrics
    @ColumnInfo(name = "sleep_latency")
    val sleepLatency: Long = 0, // Time to fall asleep

    @ColumnInfo(name = "wake_duration")
    val awakeDuration: Long = 0,

    @ColumnInfo(name = "light_sleep_duration")
    val lightSleepDuration: Long = 0,

    @ColumnInfo(name = "deep_sleep_duration")
    val deepSleepDuration: Long = 0,

    @ColumnInfo(name = "rem_sleep_duration")
    val remSleepDuration: Long = 0,

    // Quality and efficiency metrics
    @ColumnInfo(name = "sleep_efficiency")
    val sleepEfficiency: Float = 0f,

    @ColumnInfo(name = "quality_score")
    val qualityScore: Float? = null,

    // Movement and noise analytics
    @ColumnInfo(name = "average_movement_intensity")
    val averageMovementIntensity: Float = 0f,

    @ColumnInfo(name = "average_noise_level")
    val averageNoiseLevel: Float = 0f,

    @ColumnInfo(name = "movement_frequency")
    val movementFrequency: Float = 0f, // Movements per hour

    @ColumnInfo(name = "total_movement_events")
    val totalMovementEvents: Int = 0,

    @ColumnInfo(name = "total_noise_events")
    val totalNoiseEvents: Int = 0,

    @ColumnInfo(name = "total_phase_transitions")
    val totalPhaseTransitions: Int = 0,

    // AI Analysis Integration
    @ColumnInfo(name = "ai_analysis_status")
    val aiAnalysisStatus: String = "pending", // pending, in_progress, completed, failed


    @ColumnInfo(name = "confidence")
    val confidence: Float = 0.0f,

    @ColumnInfo(name = "ai_analysis_started_at")
    val aiAnalysisStartedAt: Long? = null,

    @ColumnInfo(name = "ai_analysis_completed_at")
    val aiAnalysisCompletedAt: Long? = null,

    @ColumnInfo(name = "ai_insights_generated")
    val aiInsightsGenerated: Int = 0,

    @ColumnInfo(name = "ai_model_used")
    val aiModelUsed: String? = null, // GPT-4, Claude, etc.

    @ColumnInfo(name = "ai_analysis_version")
    val aiAnalysisVersion: String? = null,

    @ColumnInfo(name = "ai_confidence_score")
    val aiConfidenceScore: Float = 0f,

    @ColumnInfo(name = "ai_processing_time_ms")
    val aiProcessingTimeMs: Long = 0L,

    @ColumnInfo(name = "personalization_applied")
    val personalizationApplied: Boolean = false,

    // Session metadata
    @ColumnInfo(name = "notes")
    val notes: String = "",

    @ColumnInfo(name = "actionable_recommendations")
    val actionableRecommendations: String = "", // JSON array of recommendations

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date(),

    @ColumnInfo(name = "updated_at")
    val updatedAt: Date = Date()
) {
    // Computed properties for convenience
    val isActive: Boolean
        get() = endTime == null

    val actualSleepDuration: Long
        get() = totalDuration - awakeDuration

    val hasAIAnalysis: Boolean
        get() = aiAnalysisStatus == "completed"

    fun toDomainModel(): SleepSession {
        return SleepSession(
            id = id,
            startTime = startTime,
            endTime = endTime,
            totalDuration = totalDuration,
            sessionDuration = sessionDuration,  // ADD THIS LINE
            sleepLatency = sleepLatency,
            awakeDuration = awakeDuration,
            lightSleepDuration = lightSleepDuration,
            deepSleepDuration = deepSleepDuration,
            remSleepDuration = remSleepDuration,
            sleepEfficiency = sleepEfficiency,
            sleepQualityScore = qualityScore,
            confidence = confidence,  // ADD THIS LINE
            averageMovementIntensity = averageMovementIntensity,
            averageNoiseLevel = averageNoiseLevel,
            movementFrequency = movementFrequency,
            notes = notes
        )
    }
}

/**
 * Enhanced sleep insights entity with comprehensive AI metadata
 * Stores AI-generated recommendations with feedback tracking
 */
@Entity(
    tableName = "sleep_insights",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE,
            onUpdate = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_insight_session"),
        Index(value = ["category"], name = "idx_insight_category"),
        Index(value = ["priority"], name = "idx_insight_priority"),
        Index(value = ["timestamp"], name = "idx_insight_timestamp"),
        Index(value = ["is_ai_generated"], name = "idx_insight_ai_generated"),
        Index(value = ["effectiveness_score"], name = "idx_insight_effectiveness"),
        Index(value = ["user_rating"], name = "idx_insight_rating")
    ]
)
data class SleepInsightEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "category")
    val category: InsightCategory,

    @ColumnInfo(name = "title")
    val title: String,

    @ColumnInfo(name = "description")
    val description: String,

    @ColumnInfo(name = "recommendation")
    val recommendation: String,

    @ColumnInfo(name = "priority")
    val priority: Int = 1, // 1 = high, 2 = medium, 3 = low

    @ColumnInfo(name = "is_ai_generated")
    val isAiGenerated: Boolean = false,

    @ColumnInfo(name = "confidence_score")
    val confidenceScore: Float = 0f,

    @ColumnInfo(name = "is_acknowledged")
    val isAcknowledged: Boolean = false,

    @ColumnInfo(name = "timestamp")
    val timestamp: Long = System.currentTimeMillis(),

    // AI Generation Metadata
    @ColumnInfo(name = "ai_model_used")
    val aiModelUsed: String? = null,

    @ColumnInfo(name = "ai_prompt_version")
    val aiPromptVersion: String? = null,

    @ColumnInfo(name = "ai_generation_job_id")
    val aiGenerationJobId: String? = null,

    @ColumnInfo(name = "ai_processing_time_ms")
    val aiProcessingTimeMs: Long = 0L,

    @ColumnInfo(name = "ai_tokens_used")
    val aiTokensUsed: Int = 0,

    @ColumnInfo(name = "personalization_factors")
    val personalizationFactors: String? = null, // JSON array of factors

    @ColumnInfo(name = "data_sources_used")
    val dataSourcesUsed: String? = null, // JSON array of data sources

    // User Feedback and Effectiveness Tracking
    @ColumnInfo(name = "user_rating")
    val userRating: Int? = null, // 1-5 stars

    @ColumnInfo(name = "was_helpful")
    val wasHelpful: Boolean? = null,

    @ColumnInfo(name = "was_implemented")
    val wasImplemented: Boolean? = null,

    @ColumnInfo(name = "feedback_text")
    val feedbackText: String? = null,

    @ColumnInfo(name = "effectiveness_score")
    val effectivenessScore: Float = 0f, // Calculated based on user engagement

    @ColumnInfo(name = "view_count")
    val viewCount: Int = 0,

    @ColumnInfo(name = "share_count")
    val shareCount: Int = 0,

    @ColumnInfo(name = "last_viewed_at")
    val lastViewedAt: Long? = null,

    @ColumnInfo(name = "acknowledged_at")
    val acknowledgedAt: Long? = null,

    @ColumnInfo(name = "feedback_submitted_at")
    val feedbackSubmittedAt: Long? = null,

    // ML Learning Integration
    @ColumnInfo(name = "ml_features")
    val mlFeatures: String? = null, // JSON of features used for generation

    @ColumnInfo(name = "similarity_score")
    val similarityScore: Float = 0f, // Similarity to other insights

    @ColumnInfo(name = "relevance_score")
    val relevanceScore: Float = 0f, // Calculated relevance to user

    @ColumnInfo(name = "predicted_usefulness")
    val predictedUsefulness: Float = 0f, // ML prediction of usefulness

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
) {
    fun toDomainModel(): SleepInsight {
        return SleepInsight(
            id = id,
            sessionId = sessionId,
            category = category,
            title = title,
            description = description,
            recommendation = recommendation,
            priority = priority,
            timestamp = timestamp,
            isAiGenerated = isAiGenerated,
            isAcknowledged = isAcknowledged
        )
    }
}

/**
 * AI Generation Job tracking entity
 * Tracks AI insight generation requests and their status
 */
@Entity(
    tableName = "ai_generation_jobs",
    indices = [
        Index(value = ["job_id"], name = "idx_ai_job_id", unique = true),
        Index(value = ["status"], name = "idx_ai_job_status"),
        Index(value = ["created_at"], name = "idx_ai_job_created"),
        Index(value = ["generation_type"], name = "idx_ai_job_type")
    ]
)
data class AIGenerationJobEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "job_id")
    val jobId: String,

    @ColumnInfo(name = "generation_type")
    val generationType: String, // POST_SESSION, DAILY_ANALYSIS, PERSONALIZED, etc.

    @ColumnInfo(name = "status")
    val status: String = "pending", // pending, in_progress, completed, failed, cancelled

    @ColumnInfo(name = "ai_model")
    val aiModel: String,

    @ColumnInfo(name = "prompt_version")
    val promptVersion: String,

    @ColumnInfo(name = "session_id")
    val sessionId: Long? = null,

    @ColumnInfo(name = "request_data")
    val requestData: String? = null, // JSON of request parameters

    @ColumnInfo(name = "insights_generated")
    val insightsGenerated: Int = 0,

    @ColumnInfo(name = "tokens_used")
    val tokensUsed: Int = 0,

    @ColumnInfo(name = "processing_time_ms")
    val processingTimeMs: Long = 0L,

    @ColumnInfo(name = "cost_cents")
    val costCents: Int = 0,

    @ColumnInfo(name = "quality_score")
    val qualityScore: Float = 0f,

    @ColumnInfo(name = "error_message")
    val errorMessage: String? = null,

    @ColumnInfo(name = "retry_count")
    val retryCount: Int = 0,

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date(),

    @ColumnInfo(name = "started_at")
    val startedAt: Date? = null,

    @ColumnInfo(name = "completed_at")
    val completedAt: Date? = null
)

/**
 * User interaction tracking entity
 * Tracks user engagement with insights for ML learning
 */
@Entity(
    tableName = "user_interactions",
    foreignKeys = [
        ForeignKey(
            entity = SleepInsightEntity::class,
            parentColumns = ["id"],
            childColumns = ["insight_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["insight_id"], name = "idx_interaction_insight"),
        Index(value = ["interaction_type"], name = "idx_interaction_type"),
        Index(value = ["timestamp"], name = "idx_interaction_timestamp"),
        Index(value = ["user_session"], name = "idx_interaction_session")
    ]
)
data class UserInteractionEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "insight_id")
    val insightId: Long,

    @ColumnInfo(name = "interaction_type")
    val interactionType: String, // VIEWED, ACKNOWLEDGED, SHARED, RATED, IMPLEMENTED, etc.

    @ColumnInfo(name = "interaction_value")
    val interactionValue: String? = null, // Rating value, share platform, etc.

    @ColumnInfo(name = "duration_ms")
    val durationMs: Long = 0L, // Time spent viewing/interacting

    @ColumnInfo(name = "user_session")
    val userSession: String? = null, // Session identifier for grouping

    @ColumnInfo(name = "context_data")
    val contextData: String? = null, // JSON of additional context

    @ColumnInfo(name = "timestamp")
    val timestamp: Long = System.currentTimeMillis(),

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
)

/**
 * User personalization preferences entity
 * Stores user preferences for AI insight generation
 */
@Entity(
    tableName = "user_preferences",
    indices = [
        Index(value = ["updated_at"], name = "idx_preferences_updated"),
        Index(value = ["preference_type"], name = "idx_preferences_type")
    ]
)
data class UserPreferencesEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "preference_type")
    val preferenceType: String, // INSIGHT_CATEGORIES, COMMUNICATION_STYLE, etc.

    @ColumnInfo(name = "preference_value")
    val preferenceValue: String, // JSON value

    @ColumnInfo(name = "weight")
    val weight: Float = 1.0f, // Preference strength

    @ColumnInfo(name = "learned_from_behavior")
    val learnedFromBehavior: Boolean = false,

    @ColumnInfo(name = "user_set")
    val userSet: Boolean = true,

    @ColumnInfo(name = "confidence_score")
    val confidenceScore: Float = 1.0f,

    @ColumnInfo(name = "updated_at")
    val updatedAt: Date = Date(),

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
)

/**
 * AI Model Performance tracking entity
 * Tracks performance metrics for different AI models
 */
@Entity(
    tableName = "ai_model_performance",
    indices = [
        Index(value = ["model_name"], name = "idx_model_name"),
        Index(value = ["metric_date"], name = "idx_model_metric_date"),
        Index(value = ["generation_type"], name = "idx_model_generation_type")
    ]
)
data class AIModelPerformanceEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "model_name")
    val modelName: String,

    @ColumnInfo(name = "generation_type")
    val generationType: String,

    @ColumnInfo(name = "metric_date")
    val metricDate: Long, // Date bucket for aggregation

    @ColumnInfo(name = "total_requests")
    val totalRequests: Int = 0,

    @ColumnInfo(name = "successful_requests")
    val successfulRequests: Int = 0,

    @ColumnInfo(name = "failed_requests")
    val failedRequests: Int = 0,

    @ColumnInfo(name = "average_processing_time_ms")
    val averageProcessingTimeMs: Long = 0L,

    @ColumnInfo(name = "total_tokens_used")
    val totalTokensUsed: Int = 0,

    @ColumnInfo(name = "total_cost_cents")
    val totalCostCents: Int = 0,

    @ColumnInfo(name = "average_quality_score")
    val averageQualityScore: Float = 0f,

    @ColumnInfo(name = "average_user_rating")
    val averageUserRating: Float = 0f,

    @ColumnInfo(name = "insights_implemented_rate")
    val insightsImplementedRate: Float = 0f,

    @ColumnInfo(name = "user_satisfaction_rate")
    val userSatisfactionRate: Float = 0f,

    @ColumnInfo(name = "updated_at")
    val updatedAt: Date = Date(),

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
)

/**
 * Insight feedback entity for detailed user feedback
 * Stores structured feedback on insights for ML improvement
 */
@Entity(
    tableName = "insight_feedback",
    foreignKeys = [
        ForeignKey(
            entity = SleepInsightEntity::class,
            parentColumns = ["id"],
            childColumns = ["insight_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["insight_id"], name = "idx_feedback_insight"),
        Index(value = ["feedback_type"], name = "idx_feedback_type"),
        Index(value = ["timestamp"], name = "idx_feedback_timestamp")
    ]
)
data class InsightFeedbackEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "insight_id")
    val insightId: Long,

    @ColumnInfo(name = "feedback_type")
    val feedbackType: String, // RATING, IMPLEMENTATION, HELPFULNESS, ACCURACY, etc.

    @ColumnInfo(name = "rating")
    val rating: Int? = null, // 1-5 scale

    @ColumnInfo(name = "was_helpful")
    val wasHelpful: Boolean? = null,

    @ColumnInfo(name = "was_accurate")
    val wasAccurate: Boolean? = null,

    @ColumnInfo(name = "was_implemented")
    val wasImplemented: Boolean? = null,

    @ColumnInfo(name = "implementation_result")
    val implementationResult: String? = null, // SUCCESS, PARTIAL, FAILED

    @ColumnInfo(name = "feedback_text")
    val feedbackText: String? = null,

    @ColumnInfo(name = "improvement_suggestions")
    val improvementSuggestions: String? = null,

    @ColumnInfo(name = "context_data")
    val contextData: String? = null, // JSON of contextual information

    @ColumnInfo(name = "engagement_metrics")
    val engagementMetrics: String? = null, // JSON of engagement data

    @ColumnInfo(name = "timestamp")
    val timestamp: Long = System.currentTimeMillis(),

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
)

/**
 * Movement event entity with AI analysis capabilities
 * Enhanced with pattern recognition and significance scoring
 */
@Entity(
    tableName = "movement_events",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_movement_session"),
        Index(value = ["timestamp"], name = "idx_movement_timestamp"),
        Index(value = ["intensity"], name = "idx_movement_intensity"),
        Index(value = ["ai_significance_score"], name = "idx_movement_ai_significance")
    ]
)
data class MovementEventEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "intensity")
    val intensity: Float,

    @ColumnInfo(name = "x_axis")
    val x: Float,

    @ColumnInfo(name = "y_axis")
    val y: Float,

    @ColumnInfo(name = "z_axis")
    val z: Float,

    @ColumnInfo(name = "magnitude")
    val magnitude: Float = 0f,

    @ColumnInfo(name = "is_significant")
    val isSignificant: Boolean = false,

    // AI Enhancement Fields
    @ColumnInfo(name = "ai_significance_score")
    val aiSignificanceScore: Float = 0f, // AI-calculated significance

    @ColumnInfo(name = "pattern_type")
    val patternType: String? = null, // RESTLESS, POSITION_CHANGE, AWAKENING, etc.

    @ColumnInfo(name = "confidence_score")
    val confidenceScore: Float = 0f,

    @ColumnInfo(name = "context_phase")
    val contextPhase: String? = null, // Sleep phase when movement occurred

    @ColumnInfo(name = "related_events")
    val relatedEvents: String? = null, // JSON array of related event IDs

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
) {
    fun toDomainModel(): MovementEvent {
        return MovementEvent(
            id = id,
            sessionId = sessionId,
            timestamp = timestamp,
            intensity = intensity,
            x = x,
            y = y,
            z = z
        )
    }
}

/**
 * Noise event entity with AI-powered analysis
 * Enhanced with sound classification and impact assessment
 */
@Entity(
    tableName = "noise_events",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_noise_session"),
        Index(value = ["timestamp"], name = "idx_noise_timestamp"),
        Index(value = ["decibel_level"], name = "idx_noise_level"),
        Index(value = ["ai_impact_score"], name = "idx_noise_ai_impact")
    ]
)
data class NoiseEventEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "decibel_level")
    val decibelLevel: Float,

    @ColumnInfo(name = "amplitude")
    val amplitude: Int,

    @ColumnInfo(name = "is_disruptive")
    val isDisruptive: Boolean = false,

    @ColumnInfo(name = "duration_ms")
    val durationMs: Long = 0,

    @ColumnInfo(name = "noise_type")
    val noiseType: String = "unknown", // AI-classified noise type

    // AI Enhancement Fields
    @ColumnInfo(name = "ai_impact_score")
    val aiImpactScore: Float = 0f, // AI-calculated sleep impact

    @ColumnInfo(name = "sound_classification")
    val soundClassification: String? = null, // SNORING, TRAFFIC, APPLIANCE, etc.

    @ColumnInfo(name = "classification_confidence")
    val classificationConfidence: Float = 0f,

    @ColumnInfo(name = "frequency_profile")
    val frequencyProfile: String? = null, // JSON of frequency analysis

    @ColumnInfo(name = "context_phase")
    val contextPhase: String? = null, // Sleep phase when noise occurred

    @ColumnInfo(name = "arousal_likelihood")
    val arousalLikelihood: Float = 0f, // Likelihood of causing awakening

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
) {
    fun toDomainModel(): NoiseEvent {
        return NoiseEvent(
            id = id,
            sessionId = sessionId,
            timestamp = timestamp,
            decibelLevel = decibelLevel,
            amplitude = amplitude
        )
    }
}

/**
 * Sleep phase transition entity with AI predictions
 * Enhanced with transition predictions and confidence scoring
 */
@Entity(
    tableName = "sleep_phases",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_phase_session"),
        Index(value = ["timestamp"], name = "idx_phase_timestamp"),
        Index(value = ["to_phase"], name = "idx_phase_type"),
        Index(value = ["ai_confidence"], name = "idx_phase_ai_confidence")
    ]
)
data class SleepPhaseEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "timestamp")
    val timestamp: Long,

    @ColumnInfo(name = "from_phase")
    val fromPhase: SleepPhase,

    @ColumnInfo(name = "to_phase")
    val toPhase: SleepPhase,

    @ColumnInfo(name = "confidence")
    val confidence: Float,

    @ColumnInfo(name = "duration_in_phase")
    val durationInPhase: Long = 0, // Time spent in previous phase

    @ColumnInfo(name = "transition_trigger")
    val transitionTrigger: String = "automatic", // movement, noise, time, etc.

    // AI Enhancement Fields
    @ColumnInfo(name = "ai_confidence")
    val aiConfidence: Float = 0f, // AI model confidence in classification

    @ColumnInfo(name = "prediction_model")
    val predictionModel: String? = null, // Model used for phase detection

    @ColumnInfo(name = "feature_vector")
    val featureVector: String? = null, // JSON of features used

    @ColumnInfo(name = "transition_probability")
    val transitionProbability: Float = 0f, // Probability of this transition

    @ColumnInfo(name = "expected_duration")
    val expectedDuration: Long = 0L, // AI prediction of phase duration

    @ColumnInfo(name = "quality_impact")
    val qualityImpact: Float = 0f, // Impact on sleep quality

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
) {
    fun toDomainModel(): PhaseTransition {
        return PhaseTransition(
            timestamp = timestamp,
            fromPhase = fromPhase,
            toPhase = toPhase,
            confidence = confidence
        )
    }
}

/**
 * Quality factors entity with AI-enhanced scoring
 * Enhanced with detailed factor analysis and AI explanations
 */
@Entity(
    tableName = "quality_factors",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_quality_session"),
        Index(value = ["overall_score"], name = "idx_quality_overall"),
        Index(value = ["ai_enhanced"], name = "idx_quality_ai_enhanced")
    ]
)
data class QualityFactorsEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "movement_score")
    val movementScore: Float,

    @ColumnInfo(name = "noise_score")
    val noiseScore: Float,

    @ColumnInfo(name = "duration_score")
    val durationScore: Float,

    @ColumnInfo(name = "consistency_score")
    val consistencyScore: Float,

    @ColumnInfo(name = "overall_score")
    val overallScore: Float,

    @ColumnInfo(name = "efficiency_score")
    val efficiencyScore: Float = 0f,

    @ColumnInfo(name = "phase_balance_score")
    val phaseBalanceScore: Float = 0f,

    // AI Enhancement Fields
    @ColumnInfo(name = "ai_enhanced")
    val aiEnhanced: Boolean = false,

    @ColumnInfo(name = "ai_model_version")
    val aiModelVersion: String? = null,

    @ColumnInfo(name = "factor_explanations")
    val factorExplanations: String? = null, // JSON of AI explanations

    @ColumnInfo(name = "improvement_suggestions")
    val improvementSuggestions: String? = null, // JSON of suggestions

    @ColumnInfo(name = "comparative_percentile")
    val comparativePercentile: Float = 0f, // Percentile vs personal history

    @ColumnInfo(name = "trend_analysis")
    val trendAnalysis: String? = null, // JSON of trend information

    @ColumnInfo(name = "confidence_intervals")
    val confidenceIntervals: String? = null, // JSON of confidence ranges

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
) {
    fun toDomainModel(): SleepQualityFactors {
        return SleepQualityFactors(
            movementScore = movementScore,
            noiseScore = noiseScore,
            durationScore = durationScore,
            consistencyScore = consistencyScore,
            overallScore = overallScore
        )
    }
}

/**
 * Sensor settings entity with AI optimization
 * Enhanced with adaptive AI-driven sensitivity adjustments
 */
@Entity(
    tableName = "sensor_settings",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_settings_session"),
        Index(value = ["created_at"], name = "idx_settings_created"),
        Index(value = ["ai_optimized"], name = "idx_settings_ai_optimized")
    ]
)
data class SensorSettingsEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "movement_threshold")
    val movementThreshold: Float = 2.0f,

    @ColumnInfo(name = "noise_threshold")
    val noiseThreshold: Int = 1000,

    @ColumnInfo(name = "movement_sampling_rate")
    val movementSamplingRate: Int = 3, // SensorManager.SENSOR_DELAY_NORMAL

    @ColumnInfo(name = "noise_sampling_interval")
    val noiseSamplingInterval: Long = 1000L,

    @ColumnInfo(name = "enable_movement_detection")
    val enableMovementDetection: Boolean = true,

    @ColumnInfo(name = "enable_noise_detection")
    val enableNoiseDetection: Boolean = true,

    @ColumnInfo(name = "enable_smart_filtering")
    val enableSmartFiltering: Boolean = true,

    @ColumnInfo(name = "auto_adjust_sensitivity")
    val autoAdjustSensitivity: Boolean = false,

    // AI Optimization Fields
    @ColumnInfo(name = "ai_optimized")
    val aiOptimized: Boolean = false,

    @ColumnInfo(name = "ai_recommended_movement_threshold")
    val aiRecommendedMovementThreshold: Float? = null,

    @ColumnInfo(name = "ai_recommended_noise_threshold")
    val aiRecommendedNoiseThreshold: Int? = null,

    @ColumnInfo(name = "optimization_confidence")
    val optimizationConfidence: Float = 0f,

    @ColumnInfo(name = "optimization_reasoning")
    val optimizationReasoning: String? = null, // AI explanation

    @ColumnInfo(name = "learning_data")
    val learningData: String? = null, // JSON of data used for optimization

    @ColumnInfo(name = "performance_prediction")
    val performancePrediction: String? = null, // JSON of predicted outcomes

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
) {
    fun toDomainModel(): SensorSettings {
        return SensorSettings(
            movementThreshold = movementThreshold,
            noiseThreshold = noiseThreshold,
            movementSamplingRate = movementSamplingRate,
            noiseSamplingInterval = noiseSamplingInterval,
            enableMovementDetection = enableMovementDetection,
            enableNoiseDetection = enableNoiseDetection,
            enableSmartFiltering = enableSmartFiltering,
            autoAdjustSensitivity = autoAdjustSensitivity
        )
    }
}

/**
 * Data integrity tracking entity for quality assurance
 */
@Entity(
    tableName = "data_integrity",
    foreignKeys = [
        ForeignKey(
            entity = SleepSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["session_id"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["session_id"], name = "idx_integrity_session"),
        Index(value = ["integrity_status"], name = "idx_integrity_status")
    ]
)
data class DataIntegrityEntity(
    @PrimaryKey(autoGenerate = true)
    @ColumnInfo(name = "id")
    val id: Long = 0,

    @ColumnInfo(name = "session_id")
    val sessionId: Long,

    @ColumnInfo(name = "integrity_status")
    val integrityStatus: DataIntegrityStatus,

    @ColumnInfo(name = "completeness_score")
    val completenessScore: Float = 0f,

    @ColumnInfo(name = "accuracy_score")
    val accuracyScore: Float = 0f,

    @ColumnInfo(name = "consistency_score")
    val consistencyScore: Float = 0f,

    @ColumnInfo(name = "overall_score")
    val overallScore: Float = 0f,

    @ColumnInfo(name = "issues_detected")
    val issuesDetected: String? = null, // JSON array of issues

    @ColumnInfo(name = "created_at")
    val createdAt: Date = Date()
)

/**
 * Relations for complex queries with AI data
 */

/**
 * Complete sleep session with all related data including AI insights
 */
data class SleepSessionWithDetailsAndAI(
    @Embedded
    val session: SleepSessionEntity,

    @Relation(
        parentColumn = "id",
        entityColumn = "session_id"
    )
    val movementEvents: List<MovementEventEntity>,

    @Relation(
        parentColumn = "id",
        entityColumn = "session_id"
    )
    val noiseEvents: List<NoiseEventEntity>,

    @Relation(
        parentColumn = "id",
        entityColumn = "session_id"
    )
    val phaseTransitions: List<SleepPhaseEntity>,

    @Relation(
        parentColumn = "id",
        entityColumn = "session_id"
    )
    val qualityFactors: QualityFactorsEntity?,

    @Relation(
        parentColumn = "id",
        entityColumn = "session_id"
    )
    val insights: List<SleepInsightEntity>,

    @Relation(
        parentColumn = "id",
        entityColumn = "session_id"
    )
    val settings: SensorSettingsEntity?
) {
    fun toDomainModel(): SleepSession {
        return SleepSession(
            id = session.id,
            startTime = session.startTime,
            endTime = session.endTime,
            totalDuration = session.totalDuration,
            sessionDuration = session.sessionDuration,  // ADDED
            sleepLatency = session.sleepLatency,
            awakeDuration = session.awakeDuration,
            lightSleepDuration = session.lightSleepDuration,
            deepSleepDuration = session.deepSleepDuration,
            remSleepDuration = session.remSleepDuration,
            sleepEfficiency = session.sleepEfficiency,
            confidence = session.confidence,  // ADDED
            movementEvents = movementEvents.map { it.toDomainModel() },
            noiseEvents = noiseEvents.map { it.toDomainModel() },
            phaseTransitions = phaseTransitions.map { it.toDomainModel() },
            sleepQualityScore = qualityFactors?.overallScore,
            qualityFactors = qualityFactors?.toDomainModel(),
            averageMovementIntensity = session.averageMovementIntensity,
            averageNoiseLevel = session.averageNoiseLevel,
            movementFrequency = session.movementFrequency,
            notes = session.notes,
            settings = settings?.toDomainModel()
        )
    }
}

/**
 * Insight with user feedback and interaction data
 */
data class InsightWithFeedback(
    @Embedded
    val insight: SleepInsightEntity,

    @Relation(
        parentColumn = "id",
        entityColumn = "insight_id"
    )
    val feedback: List<InsightFeedbackEntity>,

    @Relation(
        parentColumn = "id",
        entityColumn = "insight_id"
    )
    val interactions: List<UserInteractionEntity>
)

/**
 * AI Generation Job with generated insights
 */
data class AIJobWithInsights(
    @Embedded
    val job: AIGenerationJobEntity,

    @Relation(
        parentColumn = "job_id",
        entityColumn = "ai_generation_job_id"
    )
    val insights: List<SleepInsightEntity>
)

/**
 * Enhanced session summary for AI-powered analytics
 */
data class SleepSessionSummaryWithAI(
    @ColumnInfo(name = "id")
    val id: Long,

    @ColumnInfo(name = "start_time")
    val startTime: Long,

    @ColumnInfo(name = "end_time")
    val endTime: Long?,

    @ColumnInfo(name = "total_duration")
    val totalDuration: Long,

    @ColumnInfo(name = "quality_score")
    val qualityScore: Float?,

    @ColumnInfo(name = "sleep_efficiency")
    val sleepEfficiency: Float,

    @ColumnInfo(name = "total_movement_events")
    val totalMovementEvents: Int,

    @ColumnInfo(name = "total_noise_events")
    val totalNoiseEvents: Int,

    @ColumnInfo(name = "ai_analysis_status")
    val aiAnalysisStatus: String,

    @ColumnInfo(name = "ai_insights_generated")
    val aiInsightsGenerated: Int,

    @ColumnInfo(name = "ai_confidence_score")
    val aiConfidenceScore: Float,

    @ColumnInfo(name = "personalization_applied")
    val personalizationApplied: Boolean
)

/**
 * Domain model for sleep session (separate from database entity)
 */

/**
 * Helper functions for AI-enhanced entity creation
 */
object AIEntityHelper {

    fun createAIGenerationJob(
        jobId: String,
        generationType: String,
        aiModel: String,
        promptVersion: String,
        sessionId: Long? = null
    ): AIGenerationJobEntity {
        return AIGenerationJobEntity(
            jobId = jobId,
            generationType = generationType,
            aiModel = aiModel,
            promptVersion = promptVersion,
            sessionId = sessionId,
            status = "pending"
        )
    }

    fun createUserInteraction(
        insightId: Long,
        interactionType: String,
        value: String? = null,
        duration: Long = 0L
    ): UserInteractionEntity {
        return UserInteractionEntity(
            insightId = insightId,
            interactionType = interactionType,
            interactionValue = value,
            durationMs = duration
        )
    }

    fun createInsightFeedback(
        insightId: Long,
        feedbackType: String,
        rating: Int? = null,
        wasHelpful: Boolean? = null,
        wasImplemented: Boolean? = null,
        feedbackText: String? = null
    ): InsightFeedbackEntity {
        return InsightFeedbackEntity(
            insightId = insightId,
            feedbackType = feedbackType,
            rating = rating,
            wasHelpful = wasHelpful,
            wasImplemented = wasImplemented,
            feedbackText = feedbackText
        )
    }

    fun createUserPreference(
        preferenceType: String,
        preferenceValue: String,
        weight: Float = 1.0f,
        learnedFromBehavior: Boolean = false
    ): UserPreferencesEntity {
        return UserPreferencesEntity(
            preferenceType = preferenceType,
            preferenceValue = preferenceValue,
            weight = weight,
            learnedFromBehavior = learnedFromBehavior,
            userSet = !learnedFromBehavior
        )
    }

    fun updateSessionWithAIAnalysis(
        session: SleepSessionEntity,
        status: String,
        modelUsed: String,
        insightsGenerated: Int,
        confidenceScore: Float,
        processingTime: Long
    ): SleepSessionEntity {
        return session.copy(
            aiAnalysisStatus = status,
            aiAnalysisCompletedAt = if (status == "completed") System.currentTimeMillis() else null,
            aiModelUsed = modelUsed,
            aiInsightsGenerated = insightsGenerated,
            aiConfidenceScore = confidenceScore,
            aiProcessingTimeMs = processingTime,
            updatedAt = Date()
        )
    }

    fun enhanceInsightWithAIMetadata(
        insight: SleepInsightEntity,
        modelUsed: String,
        promptVersion: String,
        jobId: String,
        processingTime: Long,
        tokensUsed: Int,
        personalizationFactors: List<String>
    ): SleepInsightEntity {
        return insight.copy(
            aiModelUsed = modelUsed,
            aiPromptVersion = promptVersion,
            aiGenerationJobId = jobId,
            aiProcessingTimeMs = processingTime,
            aiTokensUsed = tokensUsed,
            personalizationFactors = personalizationFactors.joinToString(",")
        )
    }
}

data class SleepSession(
    val id: Long,
    val startTime: Long,
    val endTime: Long?,
    val totalDuration: Long,
    val sessionDuration: Long,  // NEW PROPERTY
    val sleepLatency: Long,
    val awakeDuration: Long,
    val lightSleepDuration: Long,
    val deepSleepDuration: Long,
    val remSleepDuration: Long,
    val sleepEfficiency: Float,
    val confidence: Float,  // NEW PROPERTY
    val movementEvents: List<MovementEvent>,
    val noiseEvents: List<NoiseEvent>,
    val phaseTransitions: List<PhaseTransition>,
    val sleepQualityScore: Float?,
    val qualityFactors: SleepQualityFactors?,
    val averageMovementIntensity: Float,
    val averageNoiseLevel: Float,
    val movementFrequency: Float,
    val notes: String,
    val settings: SensorSettings?
)

/**
 * Domain model for movement event
 */
data class MovementEvent(
    val id: Long,
    val sessionId: Long,
    val timestamp: Long,
    val intensity: Float,
    val x: Float,
    val y: Float,
    val z: Float
)

/**
 * Domain model for noise event
 */
data class NoiseEvent(
    val id: Long,
    val sessionId: Long,
    val timestamp: Long,
    val decibelLevel: Float,
    val amplitude: Int
)

/**
 * Domain model for phase transition
 */
data class PhaseTransition(
    val timestamp: Long,
    val fromPhase: SleepPhase,
    val toPhase: SleepPhase,
    val confidence: Float
)

/**
 * Domain model for sleep quality factors
 */
data class SleepQualityFactors(
    val movementScore: Float,
    val noiseScore: Float,
    val durationScore: Float,
    val consistencyScore: Float,
    val overallScore: Float
)

/**
 * Domain model for sensor settings
 */
data class SensorSettings(
    val movementThreshold: Float,
    val noiseThreshold: Int,
    val movementSamplingRate: Int,
    val noiseSamplingInterval: Long,
    val enableMovementDetection: Boolean,
    val enableNoiseDetection: Boolean,
    val enableSmartFiltering: Boolean,
    val autoAdjustSensitivity: Boolean
)