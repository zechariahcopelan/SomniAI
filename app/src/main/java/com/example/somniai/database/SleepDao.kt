package com.example.somniai.database

import androidx.room.*
import androidx.lifecycle.LiveData
import com.example.somniai.data.*
import kotlinx.coroutines.flow.Flow

/**
 * Enhanced Sleep Session DAO with AI integration
 * Handles all operations for sleep sessions including AI analysis tracking
 */
@Dao
interface SleepSessionDao {

    // ========== BASIC CRUD OPERATIONS ==========

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSession(session: SleepSessionEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSessions(sessions: List<SleepSessionEntity>): List<Long>

    @Update
    suspend fun updateSession(session: SleepSessionEntity)

    @Delete
    suspend fun deleteSession(session: SleepSessionEntity)

    @Query("DELETE FROM sleep_sessions WHERE id = :sessionId")
    suspend fun deleteSessionById(sessionId: Long)

    @Query("DELETE FROM sleep_sessions")
    suspend fun deleteAllSessions()

    // ========== BASIC QUERIES ==========

    @Query("SELECT * FROM sleep_sessions WHERE id = :sessionId")
    suspend fun getSessionById(sessionId: Long): SleepSessionEntity?

    @Query("SELECT * FROM sleep_sessions WHERE id = :sessionId")
    fun getSessionByIdLiveData(sessionId: Long): LiveData<SleepSessionEntity?>

    @Query("SELECT * FROM sleep_sessions ORDER BY start_time DESC")
    fun getAllSessions(): Flow<List<SleepSessionEntity>>

    @Query("SELECT * FROM sleep_sessions ORDER BY start_time DESC")
    suspend fun getAllSessionsList(): List<SleepSessionEntity>

    @Query("SELECT * FROM sleep_sessions WHERE end_time IS NULL LIMIT 1")
    suspend fun getActiveSession(): SleepSessionEntity?

    @Query("SELECT * FROM sleep_sessions WHERE end_time IS NULL LIMIT 1")
    fun getActiveSessionLiveData(): LiveData<SleepSessionEntity?>

    @Query("SELECT * FROM sleep_sessions ORDER BY start_time DESC LIMIT 1")
    suspend fun getLatestSession(): SleepSessionEntity?

    @Query("SELECT * FROM sleep_sessions ORDER BY start_time DESC LIMIT 1")
    fun getLatestSessionLiveData(): LiveData<SleepSessionEntity?>

    // ========== AI ANALYSIS QUERIES ==========

    @Query("SELECT * FROM sleep_sessions WHERE ai_analysis_status = :status ORDER BY start_time DESC")
    suspend fun getSessionsByAIStatus(status: String): List<SleepSessionEntity>

    @Query("SELECT * FROM sleep_sessions WHERE ai_analysis_status = 'pending' AND end_time IS NOT NULL ORDER BY start_time ASC")
    suspend fun getSessionsPendingAIAnalysis(): List<SleepSessionEntity>

    @Query("SELECT * FROM sleep_sessions WHERE ai_analysis_status = 'completed' ORDER BY ai_analysis_completed_at DESC")
    suspend fun getSessionsWithCompletedAIAnalysis(): List<SleepSessionEntity>

    @Query("SELECT * FROM sleep_sessions WHERE ai_analysis_status = 'failed' ORDER BY start_time DESC")
    suspend fun getFailedAIAnalysisSessions(): List<SleepSessionEntity>

    @Query("SELECT COUNT(*) FROM sleep_sessions WHERE ai_analysis_status = 'completed'")
    suspend fun getAIAnalyzedSessionCount(): Int

    @Query("SELECT AVG(ai_confidence_score) FROM sleep_sessions WHERE ai_analysis_status = 'completed'")
    suspend fun getAverageAIConfidenceScore(): Float?

    @Query("SELECT AVG(ai_processing_time_ms) FROM sleep_sessions WHERE ai_analysis_status = 'completed'")
    suspend fun getAverageAIProcessingTime(): Long?

    @Query("""
        UPDATE sleep_sessions 
        SET ai_analysis_status = :status, 
            ai_analysis_started_at = :startedAt,
            ai_model_used = :modelUsed,
            ai_analysis_version = :version
        WHERE id = :sessionId
    """)
    suspend fun updateAIAnalysisStatus(
        sessionId: Long,
        status: String,
        startedAt: Long?,
        modelUsed: String?,
        version: String?
    )

    @Query("""
        UPDATE sleep_sessions 
        SET ai_analysis_status = 'completed',
            ai_analysis_completed_at = :completedAt,
            ai_insights_generated = :insightsGenerated,
            ai_confidence_score = :confidenceScore,
            ai_processing_time_ms = :processingTime,
            personalization_applied = :personalizationApplied
        WHERE id = :sessionId
    """)
    suspend fun completeAIAnalysis(
        sessionId: Long,
        completedAt: Long,
        insightsGenerated: Int,
        confidenceScore: Float,
        processingTime: Long,
        personalizationApplied: Boolean
    )

    // ========== COMPLEX QUERIES WITH RELATIONSHIPS ==========

    @Transaction
    @Query("SELECT * FROM sleep_sessions WHERE id = :sessionId")
    suspend fun getSessionWithDetails(sessionId: Long): SleepSessionWithDetailsAndAI?

    @Transaction
    @Query("SELECT * FROM sleep_sessions ORDER BY start_time DESC")
    fun getAllSessionsWithDetails(): Flow<List<SleepSessionWithDetailsAndAI>>

    @Transaction
    @Query("SELECT * FROM sleep_sessions ORDER BY start_time DESC LIMIT :limit")
    suspend fun getRecentSessionsWithDetails(limit: Int = 10): List<SleepSessionWithDetailsAndAI>

    @Transaction
    @Query("SELECT * FROM sleep_sessions WHERE ai_analysis_status = 'completed' ORDER BY start_time DESC LIMIT :limit")
    suspend fun getAIAnalyzedSessionsWithDetails(limit: Int = 10): List<SleepSessionWithDetailsAndAI>

    // ========== DATE RANGE FILTERING ==========

    @Query("""
        SELECT * FROM sleep_sessions 
        WHERE start_time >= :startDate AND start_time <= :endDate 
        ORDER BY start_time DESC
    """)
    suspend fun getSessionsInDateRange(startDate: Long, endDate: Long): List<SleepSessionEntity>

    @Query("""
        SELECT * FROM sleep_sessions 
        WHERE start_time >= :startDate AND start_time <= :endDate 
        ORDER BY start_time DESC
    """)
    fun getSessionsInDateRangeFlow(startDate: Long, endDate: Long): Flow<List<SleepSessionEntity>>

    @Query("""
        SELECT * FROM sleep_sessions 
        WHERE start_time >= :currentTime - (:daysAgo * 24 * 60 * 60 * 1000)
        ORDER BY start_time DESC
    """)
    suspend fun getSessionsFromLastDays(daysAgo: Int, currentTime: Long = System.currentTimeMillis()): List<SleepSessionEntity>

    @Query("""
        SELECT * FROM sleep_sessions 
        WHERE start_time >= :weekStartTime AND start_time <= :weekEndTime
        ORDER BY start_time ASC
    """)
    suspend fun getSessionsForWeek(weekStartTime: Long, weekEndTime: Long): List<SleepSessionEntity>

    // ========== STATISTICS AND AGGREGATIONS ==========

    @Query("SELECT COUNT(*) FROM sleep_sessions")
    suspend fun getSessionCount(): Int

    @Query("SELECT COUNT(*) FROM sleep_sessions WHERE end_time IS NOT NULL")
    suspend fun getCompletedSessionCount(): Int

    @Query("SELECT AVG(total_duration) FROM sleep_sessions WHERE end_time IS NOT NULL")
    suspend fun getAverageSessionDuration(): Long?

    @Query("SELECT AVG(quality_score) FROM sleep_sessions WHERE quality_score IS NOT NULL")
    suspend fun getAverageQualityScore(): Float?

    @Query("SELECT AVG(sleep_efficiency) FROM sleep_sessions WHERE end_time IS NOT NULL")
    suspend fun getAverageEfficiency(): Float?

    @Query("""
        SELECT 
            COUNT(*) as session_count,
            AVG(total_duration) as avg_duration,
            AVG(quality_score) as avg_quality,
            AVG(sleep_efficiency) as avg_efficiency,
            SUM(total_movement_events) as total_movements,
            SUM(total_noise_events) as total_noise_events,
            MIN(start_time) as earliest_session,
            MAX(start_time) as latest_session
        FROM sleep_sessions 
        WHERE end_time IS NOT NULL
    """)
    suspend fun getOverallStatistics(): SleepStatistics?

    @Query("""
        SELECT 
            COUNT(*) as session_count,
            AVG(total_duration) as avg_duration,
            AVG(quality_score) as avg_quality,
            AVG(sleep_efficiency) as avg_efficiency,
            SUM(total_movement_events) as total_movements,
            SUM(total_noise_events) as total_noise_events,
            MIN(start_time) as best_quality_date,
            MAX(start_time) as worst_quality_date
        FROM sleep_sessions 
        WHERE end_time IS NOT NULL 
        AND start_time >= :startDate AND start_time <= :endDate
    """)
    suspend fun getStatisticsForDateRange(startDate: Long, endDate: Long): SleepStatistics?

    // ========== AI PERFORMANCE STATISTICS ==========

    @Query("""
        SELECT 
            COUNT(*) as total_sessions,
            COUNT(CASE WHEN ai_analysis_status = 'completed' THEN 1 END) as analyzed_sessions,
            COUNT(CASE WHEN ai_analysis_status = 'failed' THEN 1 END) as failed_sessions,
            AVG(ai_confidence_score) as avg_confidence,
            AVG(ai_processing_time_ms) as avg_processing_time,
            AVG(ai_insights_generated) as avg_insights_generated
        FROM sleep_sessions 
        WHERE end_time IS NOT NULL
    """)
    suspend fun getAIPerformanceStatistics(): AIPerformanceStatistics?

    // ========== QUALITY AND EFFICIENCY QUERIES ==========

    @Query("""
        SELECT * FROM sleep_sessions 
        WHERE quality_score IS NOT NULL 
        ORDER BY quality_score DESC 
        LIMIT :limit
    """)
    suspend fun getBestQualitySessions(limit: Int = 10): List<SleepSessionEntity>

    @Query("""
        SELECT * FROM sleep_sessions 
        WHERE quality_score IS NOT NULL 
        ORDER BY quality_score ASC 
        LIMIT :limit
    """)
    suspend fun getWorstQualitySessions(limit: Int = 10): List<SleepSessionEntity>

    @Query("SELECT * FROM sleep_sessions WHERE sleep_efficiency >= :minEfficiency ORDER BY sleep_efficiency DESC")
    suspend fun getHighEfficiencySessions(minEfficiency: Float = 85f): List<SleepSessionEntity>

    @Query("SELECT * FROM sleep_sessions WHERE sleep_efficiency < :maxEfficiency ORDER BY sleep_efficiency ASC")
    suspend fun getLowEfficiencySessions(maxEfficiency: Float = 70f): List<SleepSessionEntity>

    // ========== DURATION ANALYSIS ==========

    @Query("SELECT * FROM sleep_sessions WHERE total_duration >= :minDuration AND total_duration <= :maxDuration ORDER BY start_time DESC")
    suspend fun getSessionsByDurationRange(minDuration: Long, maxDuration: Long): List<SleepSessionEntity>

    @Query("SELECT AVG(total_duration) FROM sleep_sessions WHERE start_time >= :startDate AND start_time <= :endDate AND end_time IS NOT NULL")
    suspend fun getAverageDurationForPeriod(startDate: Long, endDate: Long): Long?

    // ========== TREND ANALYSIS ==========

    @Query("""
        SELECT 
            (start_time / (24 * 60 * 60 * 1000)) as day,
            AVG(quality_score) as avg_quality,
            AVG(sleep_efficiency) as avg_efficiency,
            AVG(total_duration) as avg_duration,
            COUNT(*) as session_count
        FROM sleep_sessions 
        WHERE quality_score IS NOT NULL 
        AND start_time >= :startDate 
        GROUP BY (start_time / (24 * 60 * 60 * 1000))
        ORDER BY day ASC
    """)
    suspend fun getDailyTrends(startDate: Long): List<DailyTrendData>

    @Query("""
        SELECT 
            ((start_time / (7 * 24 * 60 * 60 * 1000))) as week,
            AVG(quality_score) as avg_quality,
            AVG(sleep_efficiency) as avg_efficiency,
            AVG(total_duration) as avg_duration,
            COUNT(*) as session_count
        FROM sleep_sessions 
        WHERE quality_score IS NOT NULL 
        AND start_time >= :startDate 
        GROUP BY ((start_time / (7 * 24 * 60 * 60 * 1000)))
        ORDER BY week ASC
    """)
    suspend fun getWeeklyTrends(startDate: Long): List<WeeklyTrendData>

    // ========== MAINTENANCE OPERATIONS ==========

    @Query("DELETE FROM sleep_sessions WHERE start_time < :cutoffDate")
    suspend fun deleteSessionsOlderThan(cutoffDate: Long): Int

    @Query("DELETE FROM sleep_sessions WHERE end_time IS NULL AND start_time < :cutoffTime")
    suspend fun deleteAbandonedSessions(cutoffTime: Long): Int

    @Query("SELECT * FROM sleep_sessions WHERE end_time IS NULL AND start_time < :cutoffTime")
    suspend fun getAbandonedSessions(cutoffTime: Long): List<SleepSessionEntity>
}

/**
 * Enhanced Sleep Insight DAO with comprehensive AI features
 * Handles AI-generated recommendations with feedback tracking
 */
@Dao
interface SleepInsightDao {

    // ========== BASIC CRUD OPERATIONS ==========

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertInsight(insight: SleepInsightEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertInsights(insights: List<SleepInsightEntity>): List<Long>

    @Update
    suspend fun updateInsight(insight: SleepInsightEntity)

    @Delete
    suspend fun deleteInsight(insight: SleepInsightEntity)

    @Query("DELETE FROM sleep_insights WHERE id = :insightId")
    suspend fun deleteInsightById(insightId: Long)

    @Query("DELETE FROM sleep_insights WHERE session_id = :sessionId")
    suspend fun deleteInsightsForSession(sessionId: Long)

    // ========== BASIC QUERIES ==========

    @Query("SELECT * FROM sleep_insights WHERE id = :insightId")
    suspend fun getInsightById(insightId: Long): SleepInsightEntity?

    @Query("SELECT * FROM sleep_insights ORDER BY timestamp DESC")
    suspend fun getAllInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights ORDER BY timestamp DESC")
    fun getAllInsightsFlow(): Flow<List<SleepInsightEntity>>

    @Query("SELECT * FROM sleep_insights WHERE session_id = :sessionId ORDER BY priority ASC, timestamp DESC")
    suspend fun getInsightsForSession(sessionId: Long): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE category = :category ORDER BY priority ASC, timestamp DESC")
    suspend fun getInsightsByCategory(category: InsightCategory): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE priority = :priority ORDER BY timestamp DESC")
    suspend fun getInsightsByPriority(priority: Int): List<SleepInsightEntity>

    // ========== AI-SPECIFIC QUERIES ==========

    @Query("SELECT * FROM sleep_insights WHERE is_ai_generated = 1 ORDER BY timestamp DESC")
    suspend fun getAiGeneratedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE is_ai_generated = 0 ORDER BY timestamp DESC")
    suspend fun getRuleBasedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE ai_model_used = :modelName ORDER BY timestamp DESC")
    suspend fun getInsightsByAIModel(modelName: String): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE ai_generation_job_id = :jobId ORDER BY timestamp DESC")
    suspend fun getInsightsByGenerationJob(jobId: String): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE confidence_score >= :minConfidence ORDER BY confidence_score DESC")
    suspend fun getHighConfidenceInsights(minConfidence: Float): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE effectiveness_score >= :minEffectiveness ORDER BY effectiveness_score DESC")
    suspend fun getHighEffectivenessInsights(minEffectiveness: Float): List<SleepInsightEntity>

    // ========== USER ENGAGEMENT QUERIES ==========

    @Query("SELECT * FROM sleep_insights WHERE is_acknowledged = 0 ORDER BY priority ASC, timestamp DESC")
    suspend fun getUnacknowledgedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE is_acknowledged = 1 ORDER BY acknowledged_at DESC")
    suspend fun getAcknowledgedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE user_rating IS NOT NULL ORDER BY user_rating DESC, timestamp DESC")
    suspend fun getRatedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE was_helpful = 1 ORDER BY timestamp DESC")
    suspend fun getHelpfulInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE was_implemented = 1 ORDER BY timestamp DESC")
    suspend fun getImplementedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE view_count > 0 ORDER BY view_count DESC")
    suspend fun getMostViewedInsights(): List<SleepInsightEntity>

    @Query("SELECT * FROM sleep_insights WHERE share_count > 0 ORDER BY share_count DESC")
    suspend fun getMostSharedInsights(): List<SleepInsightEntity>

    // ========== FILTERING AND SEARCH ==========

    @Query("""
        SELECT * FROM sleep_insights 
        WHERE (title LIKE '%' || :query || '%' OR description LIKE '%' || :query || '%' OR recommendation LIKE '%' || :query || '%')
        ORDER BY timestamp DESC
    """)
    suspend fun searchInsights(query: String): List<SleepInsightEntity>

    @Query("""
        SELECT * FROM sleep_insights 
        WHERE category IN (:categories) 
        ORDER BY priority ASC, timestamp DESC
    """)
    suspend fun getInsightsByCategories(categories: List<InsightCategory>): List<SleepInsightEntity>

    @Query("""
        SELECT * FROM sleep_insights 
        WHERE timestamp >= :startDate AND timestamp <= :endDate 
        ORDER BY timestamp DESC
    """)
    suspend fun getInsightsInDateRange(startDate: Long, endDate: Long): List<SleepInsightEntity>

    @Query("""
        SELECT * FROM sleep_insights 
        WHERE timestamp >= :cutoffTime 
        ORDER BY timestamp DESC
    """)
    suspend fun getRecentInsights(cutoffTime: Long): List<SleepInsightEntity>

    // ========== UPDATE OPERATIONS ==========

    @Query("UPDATE sleep_insights SET is_acknowledged = 1, acknowledged_at = :acknowledgedAt WHERE id = :insightId")
    suspend fun acknowledgeInsight(insightId: Long, acknowledgedAt: Long = System.currentTimeMillis())

    @Query("UPDATE sleep_insights SET view_count = view_count + 1, last_viewed_at = :viewedAt WHERE id = :insightId")
    suspend fun incrementViewCount(insightId: Long, viewedAt: Long = System.currentTimeMillis())

    @Query("UPDATE sleep_insights SET share_count = share_count + 1 WHERE id = :insightId")
    suspend fun incrementShareCount(insightId: Long)

    @Query("""
        UPDATE sleep_insights 
        SET user_rating = :rating, 
            was_helpful = :wasHelpful, 
            was_implemented = :wasImplemented,
            feedback_text = :feedbackText,
            feedback_submitted_at = :submittedAt,
            effectiveness_score = :effectivenessScore
        WHERE id = :insightId
    """)
    suspend fun updateInsightFeedback(
        insightId: Long,
        rating: Int?,
        wasHelpful: Boolean?,
        wasImplemented: Boolean?,
        feedbackText: String?,
        submittedAt: Long,
        effectivenessScore: Float
    )

    // ========== ANALYTICS QUERIES ==========

    @Query("""
        SELECT 
            COUNT(*) as total_insights,
            COUNT(CASE WHEN is_acknowledged = 1 THEN 1 END) as acknowledged_count,
            COUNT(CASE WHEN is_ai_generated = 1 THEN 1 END) as ai_generated_count,
            COUNT(CASE WHEN user_rating IS NOT NULL THEN 1 END) as rated_count,
            AVG(user_rating) as avg_rating,
            AVG(effectiveness_score) as avg_effectiveness,
            COUNT(CASE WHEN was_helpful = 1 THEN 1 END) as helpful_count,
            COUNT(CASE WHEN was_implemented = 1 THEN 1 END) as implemented_count,
            SUM(view_count) as total_views,
            SUM(share_count) as total_shares
        FROM sleep_insights
    """)
    suspend fun getInsightAnalytics(): InsightAnalytics?

    @Query("""
        SELECT 
            category,
            COUNT(*) as count,
            AVG(user_rating) as avg_rating,
            AVG(effectiveness_score) as avg_effectiveness,
            COUNT(CASE WHEN is_acknowledged = 1 THEN 1 END) as acknowledged_count
        FROM sleep_insights 
        GROUP BY category
    """)
    suspend fun getInsightAnalyticsByCategory(): List<CategoryAnalytics>

    @Query("""
        SELECT 
            ai_model_used,
            COUNT(*) as count,
            AVG(user_rating) as avg_rating,
            AVG(confidence_score) as avg_confidence,
            AVG(effectiveness_score) as avg_effectiveness
        FROM sleep_insights 
        WHERE ai_model_used IS NOT NULL
        GROUP BY ai_model_used
    """)
    suspend fun getInsightAnalyticsByAIModel(): List<AIModelAnalytics>

    // ========== RELATIONS ==========

    @Transaction
    @Query("SELECT * FROM sleep_insights WHERE id = :insightId")
    suspend fun getInsightWithFeedback(insightId: Long): InsightWithFeedback?

    @Transaction
    @Query("SELECT * FROM sleep_insights ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentInsightsWithFeedback(limit: Int = 20): List<InsightWithFeedback>

    // ========== MAINTENANCE OPERATIONS ==========

    @Query("DELETE FROM sleep_insights WHERE timestamp < :cutoffDate")
    suspend fun deleteOldInsights(cutoffDate: Long): Int

    @Query("DELETE FROM sleep_insights WHERE effectiveness_score < :minEffectiveness AND timestamp < :cutoffDate")
    suspend fun deletePoorPerformingInsights(minEffectiveness: Float, cutoffDate: Long): Int
}

/**
 * AI Generation Job DAO
 * Tracks AI insight generation requests and performance
 */
@Dao
interface AIGenerationJobDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertJob(job: AIGenerationJobEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertJobs(jobs: List<AIGenerationJobEntity>): List<Long>

    @Update
    suspend fun updateJob(job: AIGenerationJobEntity)

    @Delete
    suspend fun deleteJob(job: AIGenerationJobEntity)

    @Query("SELECT * FROM ai_generation_jobs WHERE job_id = :jobId")
    suspend fun getJobById(jobId: String): AIGenerationJobEntity?

    @Query("SELECT * FROM ai_generation_jobs WHERE status = :status ORDER BY created_at DESC")
    suspend fun getJobsByStatus(status: String): List<AIGenerationJobEntity>

    @Query("SELECT * FROM ai_generation_jobs WHERE generation_type = :type ORDER BY created_at DESC")
    suspend fun getJobsByType(type: String): List<AIGenerationJobEntity>

    @Query("SELECT * FROM ai_generation_jobs WHERE session_id = :sessionId ORDER BY created_at DESC")
    suspend fun getJobsForSession(sessionId: Long): List<AIGenerationJobEntity>

    @Query("SELECT * FROM ai_generation_jobs ORDER BY created_at DESC LIMIT :limit")
    suspend fun getRecentJobs(limit: Int = 50): List<AIGenerationJobEntity>

    @Query("UPDATE ai_generation_jobs SET status = :status, started_at = :startedAt WHERE job_id = :jobId")
    suspend fun updateJobStatus(jobId: String, status: String, startedAt: Long?)

    @Query("""
        UPDATE ai_generation_jobs 
        SET status = 'completed',
            completed_at = :completedAt,
            insights_generated = :insightsGenerated,
            tokens_used = :tokensUsed,
            processing_time_ms = :processingTime,
            cost_cents = :costCents,
            quality_score = :qualityScore
        WHERE job_id = :jobId
    """)
    suspend fun completeJob(
        jobId: String,
        completedAt: Long,
        insightsGenerated: Int,
        tokensUsed: Int,
        processingTime: Long,
        costCents: Int,
        qualityScore: Float
    )

    @Query("""
        UPDATE ai_generation_jobs 
        SET status = 'failed',
            error_message = :errorMessage,
            retry_count = retry_count + 1
        WHERE job_id = :jobId
    """)
    suspend fun failJob(jobId: String, errorMessage: String)

    @Query("""
        SELECT 
            COUNT(*) as total_jobs,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
            AVG(processing_time_ms) as avg_processing_time,
            SUM(tokens_used) as total_tokens,
            SUM(cost_cents) as total_cost,
            AVG(quality_score) as avg_quality
        FROM ai_generation_jobs
    """)
    suspend fun getJobStatistics(): JobStatistics?

    @Transaction
    @Query("SELECT * FROM ai_generation_jobs WHERE job_id = :jobId")
    suspend fun getJobWithInsights(jobId: String): AIJobWithInsights?

    @Query("DELETE FROM ai_generation_jobs WHERE created_at < :cutoffDate")
    suspend fun deleteOldJobs(cutoffDate: Long): Int
}

/**
 * User Interaction DAO
 * Tracks user engagement with insights for ML learning
 */
@Dao
interface UserInteractionDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertInteraction(interaction: UserInteractionEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertInteractions(interactions: List<UserInteractionEntity>): List<Long>

    @Update
    suspend fun updateInteraction(interaction: UserInteractionEntity)

    @Delete
    suspend fun deleteInteraction(interaction: UserInteractionEntity)

    @Query("SELECT * FROM user_interactions WHERE insight_id = :insightId ORDER BY timestamp DESC")
    suspend fun getInteractionsForInsight(insightId: Long): List<UserInteractionEntity>

    @Query("SELECT * FROM user_interactions WHERE interaction_type = :type ORDER BY timestamp DESC")
    suspend fun getInteractionsByType(type: String): List<UserInteractionEntity>

    @Query("SELECT * FROM user_interactions WHERE user_session = :sessionId ORDER BY timestamp ASC")
    suspend fun getInteractionsForUserSession(sessionId: String): List<UserInteractionEntity>

    @Query("""
        SELECT * FROM user_interactions 
        WHERE timestamp >= :startDate AND timestamp <= :endDate 
        ORDER BY timestamp DESC
    """)
    suspend fun getInteractionsInDateRange(startDate: Long, endDate: Long): List<UserInteractionEntity>

    @Query("""
        SELECT 
            interaction_type,
            COUNT(*) as count,
            AVG(duration_ms) as avg_duration
        FROM user_interactions 
        GROUP BY interaction_type
    """)
    suspend fun getInteractionAnalytics(): List<InteractionTypeAnalytics>

    @Query("""
        SELECT 
            insight_id,
            COUNT(*) as interaction_count,
            SUM(duration_ms) as total_duration
        FROM user_interactions 
        GROUP BY insight_id
        ORDER BY interaction_count DESC
    """)
    suspend fun getInsightEngagementMetrics(): List<InsightEngagementMetrics>

    @Query("DELETE FROM user_interactions WHERE timestamp < :cutoffDate")
    suspend fun deleteOldInteractions(cutoffDate: Long): Int
}

/**
 * Insight Feedback DAO
 * Handles structured feedback for AI improvement
 */
@Dao
interface InsightFeedbackDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertFeedback(feedback: InsightFeedbackEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertFeedbacks(feedbacks: List<InsightFeedbackEntity>): List<Long>

    @Update
    suspend fun updateFeedback(feedback: InsightFeedbackEntity)

    @Delete
    suspend fun deleteFeedback(feedback: InsightFeedbackEntity)

    @Query("SELECT * FROM insight_feedback WHERE insight_id = :insightId ORDER BY timestamp DESC")
    suspend fun getFeedbackForInsight(insightId: Long): List<InsightFeedbackEntity>

    @Query("SELECT * FROM insight_feedback WHERE feedback_type = :type ORDER BY timestamp DESC")
    suspend fun getFeedbackByType(type: String): List<InsightFeedbackEntity>

    @Query("SELECT * FROM insight_feedback WHERE rating >= :minRating ORDER BY rating DESC")
    suspend fun getHighRatedFeedback(minRating: Int): List<InsightFeedbackEntity>

    @Query("SELECT * FROM insight_feedback WHERE was_implemented = 1 ORDER BY timestamp DESC")
    suspend fun getImplementedFeedback(): List<InsightFeedbackEntity>

    @Query("""
        SELECT 
            AVG(rating) as avg_rating,
            COUNT(CASE WHEN was_helpful = 1 THEN 1 END) as helpful_count,
            COUNT(CASE WHEN was_implemented = 1 THEN 1 END) as implemented_count,
            COUNT(*) as total_feedback
        FROM insight_feedback
    """)
    suspend fun getFeedbackStatistics(): FeedbackStatistics?

    @Query("DELETE FROM insight_feedback WHERE timestamp < :cutoffDate")
    suspend fun deleteOldFeedback(cutoffDate: Long): Int
}

/**
 * User Preferences DAO
 * Manages personalization preferences and learning
 */
@Dao
interface UserPreferencesDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPreference(preference: UserPreferencesEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPreferences(preferences: List<UserPreferencesEntity>): List<Long>

    @Update
    suspend fun updatePreference(preference: UserPreferencesEntity)

    @Delete
    suspend fun deletePreference(preference: UserPreferencesEntity)

    @Query("SELECT * FROM user_preferences WHERE preference_type = :type ORDER BY updated_at DESC LIMIT 1")
    suspend fun getLatestPreference(type: String): UserPreferencesEntity?

    @Query("SELECT * FROM user_preferences WHERE preference_type = :type ORDER BY updated_at DESC")
    suspend fun getPreferenceHistory(type: String): List<UserPreferencesEntity>

    @Query("SELECT * FROM user_preferences WHERE learned_from_behavior = 1 ORDER BY confidence_score DESC")
    suspend fun getLearnedPreferences(): List<UserPreferencesEntity>

    @Query("SELECT * FROM user_preferences WHERE user_set = 1 ORDER BY updated_at DESC")
    suspend fun getUserSetPreferences(): List<UserPreferencesEntity>

    @Query("SELECT DISTINCT preference_type FROM user_preferences")
    suspend fun getAllPreferenceTypes(): List<String>

    @Query("DELETE FROM user_preferences WHERE preference_type = :type AND updated_at < :cutoffDate")
    suspend fun deleteOldPreferences(type: String, cutoffDate: Long): Int
}

/**
 * AI Model Performance DAO
 * Tracks performance metrics for different AI models
 */
@Dao
interface AIModelPerformanceDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPerformance(performance: AIModelPerformanceEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPerformances(performances: List<AIModelPerformanceEntity>): List<Long>

    @Update
    suspend fun updatePerformance(performance: AIModelPerformanceEntity)

    @Delete
    suspend fun deletePerformance(performance: AIModelPerformanceEntity)

    @Query("SELECT * FROM ai_model_performance WHERE model_name = :modelName ORDER BY metric_date DESC")
    suspend fun getPerformanceForModel(modelName: String): List<AIModelPerformanceEntity>

    @Query("SELECT * FROM ai_model_performance WHERE generation_type = :type ORDER BY metric_date DESC")
    suspend fun getPerformanceByType(type: String): List<AIModelPerformanceEntity>

    @Query("""
        SELECT * FROM ai_model_performance 
        WHERE metric_date >= :startDate AND metric_date <= :endDate 
        ORDER BY metric_date ASC
    """)
    suspend fun getPerformanceInDateRange(startDate: Long, endDate: Long): List<AIModelPerformanceEntity>

    @Query("""
        SELECT 
            model_name,
            AVG(average_quality_score) as avg_quality,
            AVG(average_user_rating) as avg_rating,
            SUM(total_requests) as total_requests,
            SUM(successful_requests) as successful_requests,
            AVG(user_satisfaction_rate) as satisfaction_rate
        FROM ai_model_performance 
        GROUP BY model_name
    """)
    suspend fun getModelComparisonMetrics(): List<ModelComparisonMetrics>

    @Query("DELETE FROM ai_model_performance WHERE metric_date < :cutoffDate")
    suspend fun deleteOldPerformanceData(cutoffDate: Long): Int
}

/**
 * Enhanced Movement Event DAO with AI analysis
 */
@Dao
interface MovementEventDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMovementEvent(event: MovementEventEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMovementEvents(events: List<MovementEventEntity>): List<Long>

    @Update
    suspend fun updateMovementEvent(event: MovementEventEntity)

    @Delete
    suspend fun deleteMovementEvent(event: MovementEventEntity)

    @Query("DELETE FROM movement_events WHERE session_id = :sessionId")
    suspend fun deleteMovementEventsForSession(sessionId: Long)

    @Query("SELECT * FROM movement_events WHERE session_id = :sessionId ORDER BY timestamp ASC")
    suspend fun getMovementEventsForSession(sessionId: Long): List<MovementEventEntity>

    @Query("SELECT * FROM movement_events WHERE session_id = :sessionId ORDER BY timestamp ASC")
    fun getMovementEventsForSessionFlow(sessionId: Long): Flow<List<MovementEventEntity>>

    @Query("SELECT COUNT(*) FROM movement_events WHERE session_id = :sessionId")
    suspend fun getMovementCountForSession(sessionId: Long): Int

    @Query("SELECT AVG(intensity) FROM movement_events WHERE session_id = :sessionId")
    suspend fun getAverageIntensityForSession(sessionId: Long): Float?

    @Query("SELECT * FROM movement_events WHERE session_id = :sessionId AND is_significant = 1 ORDER BY timestamp ASC")
    suspend fun getSignificantMovementsForSession(sessionId: Long): List<MovementEventEntity>

    @Query("SELECT * FROM movement_events WHERE pattern_type = :patternType ORDER BY timestamp DESC")
    suspend fun getMovementsByPattern(patternType: String): List<MovementEventEntity>

    @Query("SELECT * FROM movement_events WHERE ai_significance_score >= :minScore ORDER BY ai_significance_score DESC")
    suspend fun getHighSignificanceMovements(minScore: Float): List<MovementEventEntity>

    @Query("SELECT COUNT(*) FROM movement_events WHERE is_significant = 1")
    suspend fun getTotalSignificantMovements(): Int

    @Query("SELECT COUNT(*) FROM movement_events")
    suspend fun getTotalMovementEvents(): Int

    @Query("""
        SELECT 
            session_id,
            COUNT(*) as movement_count,
            AVG(intensity) as avg_intensity,
            MAX(intensity) as max_intensity,
            COUNT(CASE WHEN is_significant = 1 THEN 1 END) as significant_count,
            AVG(ai_significance_score) as avg_ai_significance
        FROM movement_events 
        WHERE session_id = :sessionId
        GROUP BY session_id
    """)
    suspend fun getMovementAnalyticsForSession(sessionId: Long): EnhancedMovementAnalytics?

    @Query("""
        SELECT * FROM movement_events 
        WHERE timestamp >= :startTime AND timestamp <= :endTime 
        ORDER BY timestamp ASC
    """)
    suspend fun getMovementEventsInTimeRange(startTime: Long, endTime: Long): List<MovementEventEntity>

    @Query("DELETE FROM movement_events WHERE timestamp < :cutoffDate")
    suspend fun deleteOldMovementEvents(cutoffDate: Long): Int
}

/**
 * Enhanced Noise Event DAO with AI analysis
 */
@Dao
interface NoiseEventDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertNoiseEvent(event: NoiseEventEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertNoiseEvents(events: List<NoiseEventEntity>): List<Long>

    @Update
    suspend fun updateNoiseEvent(event: NoiseEventEntity)

    @Delete
    suspend fun deleteNoiseEvent(event: NoiseEventEntity)

    @Query("DELETE FROM noise_events WHERE session_id = :sessionId")
    suspend fun deleteNoiseEventsForSession(sessionId: Long)

    @Query("SELECT * FROM noise_events WHERE session_id = :sessionId ORDER BY timestamp ASC")
    suspend fun getNoiseEventsForSession(sessionId: Long): List<NoiseEventEntity>

    @Query("SELECT * FROM noise_events WHERE session_id = :sessionId ORDER BY timestamp ASC")
    fun getNoiseEventsForSessionFlow(sessionId: Long): Flow<List<NoiseEventEntity>>

    @Query("SELECT COUNT(*) FROM noise_events WHERE session_id = :sessionId")
    suspend fun getNoiseCountForSession(sessionId: Long): Int

    @Query("SELECT AVG(decibel_level) FROM noise_events WHERE session_id = :sessionId")
    suspend fun getAverageNoiseLevelForSession(sessionId: Long): Float?

    @Query("SELECT * FROM noise_events WHERE session_id = :sessionId AND is_disruptive = 1 ORDER BY timestamp ASC")
    suspend fun getDisruptiveNoiseForSession(sessionId: Long): List<NoiseEventEntity>

    @Query("SELECT * FROM noise_events WHERE sound_classification = :classification ORDER BY timestamp DESC")
    suspend fun getNoiseByClassification(classification: String): List<NoiseEventEntity>

    @Query("SELECT * FROM noise_events WHERE ai_impact_score >= :minScore ORDER BY ai_impact_score DESC")
    suspend fun getHighImpactNoise(minScore: Float): List<NoiseEventEntity>

    @Query("SELECT COUNT(*) FROM noise_events WHERE is_disruptive = 1")
    suspend fun getTotalDisruptiveNoiseEvents(): Int

    @Query("SELECT COUNT(*) FROM noise_events")
    suspend fun getTotalNoiseEvents(): Int

    @Query("""
        SELECT 
            session_id,
            COUNT(*) as noise_count,
            AVG(decibel_level) as avg_decibel,
            MAX(decibel_level) as max_decibel,
            COUNT(CASE WHEN is_disruptive = 1 THEN 1 END) as disruptive_count,
            AVG(ai_impact_score) as avg_ai_impact
        FROM noise_events 
        WHERE session_id = :sessionId
        GROUP BY session_id
    """)
    suspend fun getNoiseAnalyticsForSession(sessionId: Long): EnhancedNoiseAnalytics?

    @Query("""
        SELECT * FROM noise_events 
        WHERE timestamp >= :startTime AND timestamp <= :endTime 
        ORDER BY timestamp ASC
    """)
    suspend fun getNoiseEventsInTimeRange(startTime: Long, endTime: Long): List<NoiseEventEntity>

    @Query("DELETE FROM noise_events WHERE timestamp < :cutoffDate")
    suspend fun deleteOldNoiseEvents(cutoffDate: Long): Int
}

/**
 * Enhanced Sleep Phase DAO with AI predictions
 */
@Dao
interface SleepPhaseDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPhaseTransition(phase: SleepPhaseEntity): Long

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertPhaseTransitions(phases: List<SleepPhaseEntity>): List<Long>

    @Update
    suspend fun updatePhaseTransition(phase: SleepPhaseEntity)

    @Delete
    suspend fun deletePhaseTransition(phase: SleepPhaseEntity)

    @Query("DELETE FROM sleep_phases WHERE session_id = :sessionId")
    suspend fun deletePhaseTransitionsForSession(sessionId: Long)

    @Query("SELECT * FROM sleep_phases WHERE session_id = :sessionId ORDER BY timestamp ASC")
    suspend fun getPhaseTransitionsForSession(sessionId: Long): List<SleepPhaseEntity>

    @Query("SELECT * FROM sleep_phases WHERE session_id = :sessionId ORDER BY timestamp DESC LIMIT 1")
    suspend fun getLatestPhaseForSession(sessionId: Long): SleepPhaseEntity?

    @Query("SELECT COUNT(*) FROM sleep_phases WHERE session_id = :sessionId")
    suspend fun getPhaseTransitionCountForSession(sessionId: Long): Int

    @Query("SELECT COUNT(*) FROM sleep_phases")
    suspend fun getTotalPhaseTransitions(): Int

    @Query("SELECT * FROM sleep_phases WHERE prediction_model = :modelName ORDER BY timestamp DESC")
    suspend fun getPhasesByPredictionModel(modelName: String): List<SleepPhaseEntity>

    @Query("SELECT * FROM sleep_phases WHERE ai_confidence >= :minConfidence ORDER BY ai_confidence DESC")
    suspend fun getHighConfidencePhases(minConfidence: Float): List<SleepPhaseEntity>

    @Query("""
        SELECT 
            to_phase,
            COUNT(*) as transition_count,
            AVG(confidence) as avg_confidence,
            AVG(duration_in_phase) as avg_duration,
            AVG(ai_confidence) as avg_ai_confidence
        FROM sleep_phases 
        WHERE session_id = :sessionId
        GROUP BY to_phase
        ORDER BY transition_count DESC
    """)
    suspend fun getPhaseAnalyticsForSession(sessionId: Long): List<EnhancedPhaseAnalytics>

    @Query("""
        SELECT 
            to_phase,
            SUM(duration_in_phase) as total_duration
        FROM sleep_phases 
        WHERE session_id = :sessionId
        GROUP BY to_phase
    """)
    suspend fun getPhaseDurationsForSession(sessionId: Long): List<PhaseDuration>

    @Query("DELETE FROM sleep_phases WHERE timestamp < :cutoffDate")
    suspend fun deleteOldPhaseTransitions(cutoffDate: Long): Int
}

/**
 * Enhanced Quality Factors DAO with AI insights
 */
@Dao
interface QualityFactorsDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertQualityFactors(factors: QualityFactorsEntity): Long

    @Update
    suspend fun updateQualityFactors(factors: QualityFactorsEntity)

    @Delete
    suspend fun deleteQualityFactors(factors: QualityFactorsEntity)

    @Query("DELETE FROM quality_factors WHERE session_id = :sessionId")
    suspend fun deleteQualityFactorsForSession(sessionId: Long)

    @Query("SELECT * FROM quality_factors WHERE session_id = :sessionId")
    suspend fun getQualityFactorsForSession(sessionId: Long): QualityFactorsEntity?

    @Query("SELECT * FROM quality_factors WHERE ai_enhanced = 1 ORDER BY overall_score DESC")
    suspend fun getAIEnhancedQualityFactors(): List<QualityFactorsEntity>

    @Query("SELECT AVG(overall_score) FROM quality_factors")
    suspend fun getAverageOverallScore(): Float?

    @Query("SELECT AVG(movement_score) FROM quality_factors")
    suspend fun getAverageMovementScore(): Float?

    @Query("SELECT AVG(noise_score) FROM quality_factors")
    suspend fun getAverageNoiseScore(): Float?

    @Query("SELECT * FROM quality_factors ORDER BY overall_score DESC LIMIT :limit")
    suspend fun getBestQualityFactors(limit: Int = 10): List<QualityFactorsEntity>

    @Query("SELECT * FROM quality_factors ORDER BY overall_score ASC LIMIT :limit")
    suspend fun getWorstQualityFactors(limit: Int = 10): List<QualityFactorsEntity>

    @Query("""
        SELECT 
            AVG(overall_score) as avg_overall,
            AVG(movement_score) as avg_movement,
            AVG(noise_score) as avg_noise,
            AVG(duration_score) as avg_duration,
            AVG(efficiency_score) as avg_efficiency,
            COUNT(CASE WHEN ai_enhanced = 1 THEN 1 END) as ai_enhanced_count
        FROM quality_factors
    """)
    suspend fun getQualityAnalytics(): QualityAnalytics?
}

/**
 * Enhanced Sensor Settings DAO with AI optimization
 */
@Dao
interface SensorSettingsDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSettings(settings: SensorSettingsEntity): Long

    @Update
    suspend fun updateSettings(settings: SensorSettingsEntity)

    @Delete
    suspend fun deleteSettings(settings: SensorSettingsEntity)

    @Query("SELECT * FROM sensor_settings WHERE session_id = :sessionId")
    suspend fun getSettingsForSession(sessionId: Long): SensorSettingsEntity?

    @Query("SELECT * FROM sensor_settings ORDER BY created_at DESC LIMIT 1")
    suspend fun getLatestSettings(): SensorSettingsEntity?

    @Query("SELECT * FROM sensor_settings WHERE ai_optimized = 1 ORDER BY created_at DESC")
    suspend fun getAIOptimizedSettings(): List<SensorSettingsEntity>

    @Query("SELECT AVG(movement_threshold) FROM sensor_settings")
    suspend fun getAverageMovementThreshold(): Float?

    @Query("SELECT AVG(noise_threshold) FROM sensor_settings")
    suspend fun getAverageNoiseThreshold(): Int?

    @Query("SELECT AVG(optimization_confidence) FROM sensor_settings WHERE ai_optimized = 1")
    suspend fun getAverageOptimizationConfidence(): Float?
}

// ========== ENHANCED DATA CLASSES FOR QUERY RESULTS ==========

data class AIPerformanceStatistics(
    val totalSessions: Int,
    val analyzedSessions: Int,
    val failedSessions: Int,
    val avgConfidence: Float,
    val avgProcessingTime: Long,
    val avgInsightsGenerated: Float
)

data class InsightAnalytics(
    val totalInsights: Int,
    val acknowledgedCount: Int,
    val aiGeneratedCount: Int,
    val ratedCount: Int,
    val avgRating: Float,
    val avgEffectiveness: Float,
    val helpfulCount: Int,
    val implementedCount: Int,
    val totalViews: Int,
    val totalShares: Int
)

data class CategoryAnalytics(
    val category: InsightCategory,
    val count: Int,
    val avgRating: Float,
    val avgEffectiveness: Float,
    val acknowledgedCount: Int
)

data class AIModelAnalytics(
    val aiModelUsed: String,
    val count: Int,
    val avgRating: Float,
    val avgConfidence: Float,
    val avgEffectiveness: Float
)

data class JobStatistics(
    val totalJobs: Int,
    val completedJobs: Int,
    val failedJobs: Int,
    val avgProcessingTime: Long,
    val totalTokens: Int,
    val totalCost: Int,
    val avgQuality: Float
)

data class InteractionTypeAnalytics(
    val interactionType: String,
    val count: Int,
    val avgDuration: Long
)

data class InsightEngagementMetrics(
    val insightId: Long,
    val interactionCount: Int,
    val totalDuration: Long
)

data class FeedbackStatistics(
    val avgRating: Float,
    val helpfulCount: Int,
    val implementedCount: Int,
    val totalFeedback: Int
)

data class ModelComparisonMetrics(
    val modelName: String,
    val avgQuality: Float,
    val avgRating: Float,
    val totalRequests: Int,
    val successfulRequests: Int,
    val satisfactionRate: Float
)

data class EnhancedMovementAnalytics(
    val sessionId: Long,
    val movementCount: Int,
    val avgIntensity: Float,
    val maxIntensity: Float,
    val significantCount: Int,
    val avgAiSignificance: Float
)

data class EnhancedNoiseAnalytics(
    val sessionId: Long,
    val noiseCount: Int,
    val avgDecibel: Float,
    val maxDecibel: Float,
    val disruptiveCount: Int,
    val avgAiImpact: Float
)

data class EnhancedPhaseAnalytics(
    val toPhase: SleepPhase,
    val transitionCount: Int,
    val avgConfidence: Float,
    val avgDuration: Long,
    val avgAiConfidence: Float
)

data class QualityAnalytics(
    val avgOverall: Float,
    val avgMovement: Float,
    val avgNoise: Float,
    val avgDuration: Float,
    val avgEfficiency: Float,
    val aiEnhancedCount: Int
)

// Keep existing data classes for compatibility
data class MovementAnalytics(
    val sessionId: Long,
    val movementCount: Int,
    val avgIntensity: Float,
    val maxIntensity: Float,
    val significantCount: Int
)

data class NoiseAnalytics(
    val sessionId: Long,
    val noiseCount: Int,
    val avgDecibel: Float,
    val maxDecibel: Float,
    val disruptiveCount: Int
)

data class PhaseAnalytics(
    val toPhase: SleepPhase,
    val transitionCount: Int,
    val avgConfidence: Float,
    val avgDuration: Long
)

data class PhaseDuration(
    val toPhase: SleepPhase,
    val totalDuration: Long
)

data class DailyTrendData(
    val day: Long,
    val avgQuality: Float,
    val avgEfficiency: Float,
    val avgDuration: Long,
    val sessionCount: Int
)

data class WeeklyTrendData(
    val week: Long,
    val avgQuality: Float,
    val avgEfficiency: Float,
    val avgDuration: Long,
    val sessionCount: Int
)