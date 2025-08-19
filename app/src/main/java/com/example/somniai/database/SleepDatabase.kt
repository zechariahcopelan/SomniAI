package com.example.somniai.database

import android.content.Context
import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.example.somniai.data.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Date

/**
 * Enhanced Room Database for SomniAI with comprehensive AI integration
 *
 * Manages persistent storage of:
 * - Sleep sessions and their metadata with AI analysis tracking
 * - Movement/noise events with AI pattern recognition
 * - Sleep phase transitions with AI predictions
 * - Quality factors with AI-enhanced scoring
 * - AI-generated insights with feedback tracking
 * - AI generation jobs and performance monitoring
 * - User interactions and personalization preferences
 * - Model performance analytics and optimization
 * - Sensor configuration with AI recommendations
 *
 * Features:
 * - Comprehensive AI workflow tracking
 * - Advanced analytics and machine learning support
 * - User engagement and feedback collection
 * - Model performance optimization
 * - Intelligent caching and data management
 * - Robust migration system for schema evolution
 */
@Database(
    entities = [
        // Core sleep tracking entities (enhanced with AI fields)
        SleepSessionEntity::class,
        MovementEventEntity::class,
        NoiseEventEntity::class,
        SleepPhaseEntity::class,
        QualityFactorsEntity::class,
        SleepInsightEntity::class,
        SensorSettingsEntity::class,

        // New AI integration entities
        AIGenerationJobEntity::class,
        UserInteractionEntity::class,
        UserPreferencesEntity::class,
        AIModelPerformanceEntity::class,
        InsightFeedbackEntity::class
    ],
    version = 2, // Bumped from 1 to 2 for AI integration
    exportSchema = true,
    autoMigrations = [
        // Auto-migrations for minor schema changes
        // AutoMigration(from = 2, to = 3) // Future auto-migrations
    ]
)
@TypeConverters(DatabaseConverters::class)
abstract class SleepDatabase : RoomDatabase() {

    // Core Data Access Objects
    abstract fun sleepSessionDao(): SleepSessionDao
    abstract fun movementEventDao(): MovementEventDao
    abstract fun noiseEventDao(): NoiseEventDao
    abstract fun sleepPhaseDao(): SleepPhaseDao
    abstract fun qualityFactorsDao(): QualityFactorsDao
    abstract fun sleepInsightDao(): SleepInsightDao
    abstract fun sensorSettingsDao(): SensorSettingsDao

    // AI Integration Data Access Objects
    abstract fun aiGenerationJobDao(): AIGenerationJobDao
    abstract fun userInteractionDao(): UserInteractionDao
    abstract fun userPreferencesDao(): UserPreferencesDao
    abstract fun aiModelPerformanceDao(): AIModelPerformanceDao
    abstract fun insightFeedbackDao(): InsightFeedbackDao

    companion object {
        // Database configuration constants
        const val DATABASE_NAME = "somniai_sleep_database"
        const val DATABASE_VERSION = 2 // Updated for AI integration
        const val AI_INTEGRATION_VERSION = 2

        // Singleton instance
        @Volatile
        private var INSTANCE: SleepDatabase? = null

        /**
         * Get database instance using singleton pattern
         * Thread-safe implementation with double-checked locking
         */
        fun getDatabase(
            context: Context,
            scope: CoroutineScope = CoroutineScope(Dispatchers.IO)
        ): SleepDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    SleepDatabase::class.java,
                    DATABASE_NAME
                )
                    .addMigrations(*getAllMigrations())
                    .addCallback(SleepDatabaseCallback(scope))
                    .setJournalMode(RoomDatabase.JournalMode.WAL) // Write-Ahead Logging for performance
                    .fallbackToDestructiveMigration() // Only for development - remove for production
                    .enableMultiInstanceInvalidation() // Enable multi-instance invalidation for AI workers
                    .build()

                INSTANCE = instance
                instance
            }
        }

        /**
         * Get all database migrations
         * Comprehensive migration system for AI integration
         */
        private fun getAllMigrations(): Array<Migration> {
            return arrayOf(
                MIGRATION_1_2, // Core AI integration migration
                // Future migrations will be added here
                // MIGRATION_2_3,
                // MIGRATION_3_4,
            )
        }

        /**
         * Clear database instance (for testing)
         */
        fun clearInstance() {
            INSTANCE?.close()
            INSTANCE = null
        }
    }

    /**
     * Enhanced database callback for AI initialization
     * Handles database creation, seeding, and AI system setup
     */
    private class SleepDatabaseCallback(
        private val scope: CoroutineScope
    ) : RoomDatabase.Callback() {

        override fun onCreate(db: SupportSQLiteDatabase) {
            super.onCreate(db)

            // Initialize database with AI-ready configuration
            INSTANCE?.let { database ->
                scope.launch {
                    initializeAIDatabase(database)
                }
            }
        }

        override fun onOpen(db: SupportSQLiteDatabase) {
            super.onOpen(db)

            // Perform AI system maintenance on database open
            INSTANCE?.let { database ->
                scope.launch {
                    performAIMaintenanceTasks(database)
                }
            }
        }

        /**
         * Initialize database with AI-ready configuration and seed data
         */
        private suspend fun initializeAIDatabase(database: SleepDatabase) {
            android.util.Log.d("SleepDatabase", "Initializing AI-enhanced database")

            // Initialize default user preferences for AI personalization
            initializeDefaultPreferences(database)

            // Set up AI model performance tracking
            initializeAIModelTracking(database)

            // Create initial sensor settings with AI optimization enabled
            initializeAIOptimizedSettings(database)

            android.util.Log.d("SleepDatabase", "AI database initialization completed")
        }

        /**
         * Perform AI-specific maintenance tasks
         */
        private suspend fun performAIMaintenanceTasks(database: SleepDatabase) {
            android.util.Log.d("SleepDatabase", "Performing AI maintenance tasks")

            // Clean up old AI generation jobs (older than 30 days)
            val cutoffDate = System.currentTimeMillis() - (30L * 24L * 60L * 60L * 1000L)
            database.aiGenerationJobDao().deleteOldJobs(cutoffDate)

            // Clean up old user interactions (older than 90 days)
            val interactionCutoff = System.currentTimeMillis() - (90L * 24L * 60L * 60L * 1000L)
            database.userInteractionDao().deleteOldInteractions(interactionCutoff)

            // Archive old model performance data (older than 180 days)
            val performanceCutoff = System.currentTimeMillis() - (180L * 24L * 60L * 60L * 1000L)
            database.aiModelPerformanceDao().deleteOldPerformanceData(performanceCutoff)

            // Update AI model performance aggregations
            updateAIPerformanceAggregations(database)

            android.util.Log.d("SleepDatabase", "AI maintenance completed")
        }

        private suspend fun initializeDefaultPreferences(database: SleepDatabase) {
            val preferencesDao = database.userPreferencesDao()

            // Default insight categories preference
            val defaultCategories = listOf(
                InsightCategory.QUALITY.name,
                InsightCategory.DURATION.name,
                InsightCategory.EFFICIENCY.name
            ).joinToString(",")

            val categoryPreference = UserPreferencesEntity(
                preferenceType = "PREFERRED_INSIGHT_CATEGORIES",
                preferenceValue = defaultCategories,
                weight = 1.0f,
                userSet = false,
                learnedFromBehavior = false,
                confidenceScore = 0.5f
            )

            // Default communication style
            val communicationPreference = UserPreferencesEntity(
                preferenceType = "COMMUNICATION_STYLE",
                preferenceValue = "supportive",
                weight = 1.0f,
                userSet = false,
                learnedFromBehavior = false,
                confidenceScore = 0.8f
            )

            // Default AI generation preferences
            val aiPreference = UserPreferencesEntity(
                preferenceType = "AI_GENERATION_ENABLED",
                preferenceValue = "true",
                weight = 1.0f,
                userSet = true,
                learnedFromBehavior = false,
                confidenceScore = 1.0f
            )

            preferencesDao.insertPreferences(listOf(
                categoryPreference,
                communicationPreference,
                aiPreference
            ))
        }

        private suspend fun initializeAIModelTracking(database: SleepDatabase) {
            val performanceDao = database.aiModelPerformanceDao()
            val currentDate = System.currentTimeMillis()

            // Initialize performance tracking for supported models
            val models = listOf("GPT-4", "Claude", "Gemini", "Local-Rule-Based")
            val generationTypes = listOf("POST_SESSION", "DAILY_ANALYSIS", "PERSONALIZED", "PREDICTIVE")

            for (model in models) {
                for (type in generationTypes) {
                    val performance = AIModelPerformanceEntity(
                        modelName = model,
                        generationType = type,
                        metricDate = currentDate,
                        totalRequests = 0,
                        successfulRequests = 0,
                        failedRequests = 0
                    )
                    performanceDao.insertPerformance(performance)
                }
            }
        }

        private suspend fun initializeAIOptimizedSettings(database: SleepDatabase) {
            // This would be called when creating initial sensor settings
            // The settings would include AI optimization flags enabled by default
        }

        private suspend fun updateAIPerformanceAggregations(database: SleepDatabase) {
            // Update daily/weekly/monthly performance aggregations
            // This could include calculating rolling averages, success rates, etc.
        }
    }
}

/**
 * Enhanced type converters for AI integration
 * Handles conversion between Kotlin types and SQLite types for AI data
 */
class DatabaseConverters {

    // ========== EXISTING CONVERTERS ==========

    @TypeConverter
    fun fromTimestamp(value: Long?): Date? {
        return value?.let { Date(it) }
    }

    @TypeConverter
    fun dateToTimestamp(date: Date?): Long? {
        return date?.time
    }

    @TypeConverter
    fun fromSleepPhase(phase: SleepPhase): String {
        return phase.name
    }

    @TypeConverter
    fun toSleepPhase(phase: String): SleepPhase {
        return try {
            SleepPhase.valueOf(phase)
        } catch (e: IllegalArgumentException) {
            SleepPhase.UNKNOWN
        }
    }

    @TypeConverter
    fun fromInsightCategory(category: InsightCategory): String {
        return category.name
    }

    @TypeConverter
    fun toInsightCategory(category: String): InsightCategory {
        return try {
            InsightCategory.valueOf(category)
        } catch (e: IllegalArgumentException) {
            InsightCategory.GENERAL
        }
    }

    @TypeConverter
    fun fromFloatList(value: List<Float>?): String? {
        return value?.joinToString(",")
    }

    @TypeConverter
    fun toFloatList(value: String?): List<Float>? {
        return value?.split(",")?.mapNotNull {
            try {
                it.toFloat()
            } catch (e: NumberFormatException) {
                null
            }
        }
    }

    @TypeConverter
    fun fromStringList(value: List<String>?): String? {
        return value?.joinToString("|")
    }

    @TypeConverter
    fun toStringList(value: String?): List<String>? {
        return value?.split("|")?.filter { it.isNotEmpty() }
    }

    // ========== NEW AI-SPECIFIC CONVERTERS ==========

    @TypeConverter
    fun fromStringMap(value: Map<String, String>?): String? {
        return value?.entries?.joinToString(";") { "${it.key}:${it.value}" }
    }

    @TypeConverter
    fun toStringMap(value: String?): Map<String, String>? {
        return value?.split(";")?.mapNotNull { entry ->
            val parts = entry.split(":", limit = 2)
            if (parts.size == 2) parts[0] to parts[1] else null
        }?.toMap()
    }

    @TypeConverter
    fun fromStringFloatMap(value: Map<String, Float>?): String? {
        return value?.entries?.joinToString(";") { "${it.key}:${it.value}" }
    }

    @TypeConverter
    fun toStringFloatMap(value: String?): Map<String, Float>? {
        return value?.split(";")?.mapNotNull { entry ->
            val parts = entry.split(":", limit = 2)
            if (parts.size == 2) {
                try {
                    parts[0] to parts[1].toFloat()
                } catch (e: NumberFormatException) {
                    null
                }
            } else null
        }?.toMap()
    }

    @TypeConverter
    fun fromStringIntMap(value: Map<String, Int>?): String? {
        return value?.entries?.joinToString(";") { "${it.key}:${it.value}" }
    }

    @TypeConverter
    fun toStringIntMap(value: String?): Map<String, Int>? {
        return value?.split(";")?.mapNotNull { entry ->
            val parts = entry.split(":", limit = 2)
            if (parts.size == 2) {
                try {
                    parts[0] to parts[1].toInt()
                } catch (e: NumberFormatException) {
                    null
                }
            } else null
        }?.toMap()
    }

    @TypeConverter
    fun fromStringAnyMap(value: Map<String, Any>?): String? {
        return value?.entries?.joinToString(";") { "${it.key}:${it.value}" }
    }

    @TypeConverter
    fun toStringAnyMap(value: String?): Map<String, Any>? {
        return value?.split(";")?.mapNotNull { entry ->
            val parts = entry.split(":", limit = 2)
            if (parts.size == 2) parts[0] to parts[1] as Any else null
        }?.toMap()
    }

    @TypeConverter
    fun fromLongList(value: List<Long>?): String? {
        return value?.joinToString(",")
    }

    @TypeConverter
    fun toLongList(value: String?): List<Long>? {
        return value?.split(",")?.mapNotNull {
            try {
                it.toLong()
            } catch (e: NumberFormatException) {
                null
            }
        }
    }
}

/**
 * Comprehensive migration from version 1 to 2 - AI Integration
 * Adds all AI-related tables and enhances existing tables with AI fields
 */
val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(database: SupportSQLiteDatabase) {
        android.util.Log.d("SleepDatabase", "Starting migration 1->2: AI Integration")

        // ========== ENHANCE EXISTING TABLES ==========

        // Add AI fields to sleep_sessions
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_analysis_status TEXT DEFAULT 'pending'")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_analysis_started_at INTEGER")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_analysis_completed_at INTEGER")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_insights_generated INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_model_used TEXT")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_analysis_version TEXT")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_confidence_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN ai_processing_time_ms INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_sessions ADD COLUMN personalization_applied INTEGER DEFAULT 0")

        // Add AI fields to sleep_insights
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN ai_model_used TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN ai_prompt_version TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN ai_generation_job_id TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN ai_processing_time_ms INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN ai_tokens_used INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN personalization_factors TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN data_sources_used TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN user_rating INTEGER")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN was_helpful INTEGER")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN was_implemented INTEGER")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN feedback_text TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN effectiveness_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN view_count INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN share_count INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN last_viewed_at INTEGER")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN acknowledged_at INTEGER")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN feedback_submitted_at INTEGER")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN ml_features TEXT")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN similarity_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN relevance_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sleep_insights ADD COLUMN predicted_usefulness REAL DEFAULT 0.0")

        // Add AI fields to movement_events
        database.execSQL("ALTER TABLE movement_events ADD COLUMN ai_significance_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE movement_events ADD COLUMN pattern_type TEXT")
        database.execSQL("ALTER TABLE movement_events ADD COLUMN confidence_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE movement_events ADD COLUMN context_phase TEXT")
        database.execSQL("ALTER TABLE movement_events ADD COLUMN related_events TEXT")

        // Add AI fields to noise_events
        database.execSQL("ALTER TABLE noise_events ADD COLUMN ai_impact_score REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE noise_events ADD COLUMN sound_classification TEXT")
        database.execSQL("ALTER TABLE noise_events ADD COLUMN classification_confidence REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE noise_events ADD COLUMN frequency_profile TEXT")
        database.execSQL("ALTER TABLE noise_events ADD COLUMN context_phase TEXT")
        database.execSQL("ALTER TABLE noise_events ADD COLUMN arousal_likelihood REAL DEFAULT 0.0")

        // Add AI fields to sleep_phases
        database.execSQL("ALTER TABLE sleep_phases ADD COLUMN ai_confidence REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sleep_phases ADD COLUMN prediction_model TEXT")
        database.execSQL("ALTER TABLE sleep_phases ADD COLUMN feature_vector TEXT")
        database.execSQL("ALTER TABLE sleep_phases ADD COLUMN transition_probability REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sleep_phases ADD COLUMN expected_duration INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sleep_phases ADD COLUMN quality_impact REAL DEFAULT 0.0")

        // Add AI fields to quality_factors
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN ai_enhanced INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN ai_model_version TEXT")
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN factor_explanations TEXT")
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN improvement_suggestions TEXT")
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN comparative_percentile REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN trend_analysis TEXT")
        database.execSQL("ALTER TABLE quality_factors ADD COLUMN confidence_intervals TEXT")

        // Add AI fields to sensor_settings
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN ai_optimized INTEGER DEFAULT 0")
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN ai_recommended_movement_threshold REAL")
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN ai_recommended_noise_threshold INTEGER")
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN optimization_confidence REAL DEFAULT 0.0")
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN optimization_reasoning TEXT")
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN learning_data TEXT")
        database.execSQL("ALTER TABLE sensor_settings ADD COLUMN performance_prediction TEXT")

        // ========== CREATE NEW AI TABLES ==========

        // AI Generation Jobs table
        database.execSQL("""
            CREATE TABLE IF NOT EXISTS ai_generation_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                job_id TEXT NOT NULL,
                generation_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                ai_model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                session_id INTEGER,
                request_data TEXT,
                insights_generated INTEGER NOT NULL DEFAULT 0,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                processing_time_ms INTEGER NOT NULL DEFAULT 0,
                cost_cents INTEGER NOT NULL DEFAULT 0,
                quality_score REAL NOT NULL DEFAULT 0.0,
                error_message TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT 0,
                started_at INTEGER,
                completed_at INTEGER
            )
        """.trimIndent())

        // User Interactions table
        database.execSQL("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                insight_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL,
                interaction_value TEXT,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                user_session TEXT,
                context_data TEXT,
                timestamp INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(insight_id) REFERENCES sleep_insights(id) ON DELETE CASCADE
            )
        """.trimIndent())

        // User Preferences table
        database.execSQL("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                preference_type TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                learned_from_behavior INTEGER NOT NULL DEFAULT 0,
                user_set INTEGER NOT NULL DEFAULT 1,
                confidence_score REAL NOT NULL DEFAULT 1.0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent())

        // AI Model Performance table
        database.execSQL("""
            CREATE TABLE IF NOT EXISTS ai_model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model_name TEXT NOT NULL,
                generation_type TEXT NOT NULL,
                metric_date INTEGER NOT NULL,
                total_requests INTEGER NOT NULL DEFAULT 0,
                successful_requests INTEGER NOT NULL DEFAULT 0,
                failed_requests INTEGER NOT NULL DEFAULT 0,
                average_processing_time_ms INTEGER NOT NULL DEFAULT 0,
                total_tokens_used INTEGER NOT NULL DEFAULT 0,
                total_cost_cents INTEGER NOT NULL DEFAULT 0,
                average_quality_score REAL NOT NULL DEFAULT 0.0,
                average_user_rating REAL NOT NULL DEFAULT 0.0,
                insights_implemented_rate REAL NOT NULL DEFAULT 0.0,
                user_satisfaction_rate REAL NOT NULL DEFAULT 0.0,
                updated_at INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent())

        // Insight Feedback table
        database.execSQL("""
            CREATE TABLE IF NOT EXISTS insight_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                insight_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER,
                was_helpful INTEGER,
                was_accurate INTEGER,
                was_implemented INTEGER,
                implementation_result TEXT,
                feedback_text TEXT,
                improvement_suggestions TEXT,
                context_data TEXT,
                engagement_metrics TEXT,
                timestamp INTEGER NOT NULL,
                created_at INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(insight_id) REFERENCES sleep_insights(id) ON DELETE CASCADE
            )
        """.trimIndent())

        // ========== CREATE INDEXES FOR PERFORMANCE ==========

        // AI-specific indexes on existing tables
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_session_ai_status ON sleep_sessions(ai_analysis_status)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_session_ai_completed ON sleep_sessions(ai_analysis_completed_at)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_insight_ai_generated ON sleep_insights(is_ai_generated)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_insight_effectiveness ON sleep_insights(effectiveness_score)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_insight_rating ON sleep_insights(user_rating)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_movement_ai_significance ON movement_events(ai_significance_score)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_noise_ai_impact ON noise_events(ai_impact_score)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_phase_ai_confidence ON sleep_phases(ai_confidence)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_quality_ai_enhanced ON quality_factors(ai_enhanced)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_settings_ai_optimized ON sensor_settings(ai_optimized)")

        // Indexes on new tables
        database.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_job_id ON ai_generation_jobs(job_id)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_ai_job_status ON ai_generation_jobs(status)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_ai_job_created ON ai_generation_jobs(created_at)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_ai_job_type ON ai_generation_jobs(generation_type)")

        database.execSQL("CREATE INDEX IF NOT EXISTS idx_interaction_insight ON user_interactions(insight_id)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_interaction_type ON user_interactions(interaction_type)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_interaction_timestamp ON user_interactions(timestamp)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_interaction_session ON user_interactions(user_session)")

        database.execSQL("CREATE INDEX IF NOT EXISTS idx_preferences_updated ON user_preferences(updated_at)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_preferences_type ON user_preferences(preference_type)")

        database.execSQL("CREATE INDEX IF NOT EXISTS idx_model_name ON ai_model_performance(model_name)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_model_metric_date ON ai_model_performance(metric_date)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_model_generation_type ON ai_model_performance(generation_type)")

        database.execSQL("CREATE INDEX IF NOT EXISTS idx_feedback_insight ON insight_feedback(insight_id)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_feedback_type ON insight_feedback(feedback_type)")
        database.execSQL("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON insight_feedback(timestamp)")

        android.util.Log.d("SleepDatabase", "Migration 1->2 completed successfully: AI Integration")
    }
}

/**
 * Enhanced database utilities with AI-specific functionality
 */
object DatabaseUtils {

    /**
     * Get database file size in bytes
     */
    fun getDatabaseSize(context: Context): Long {
        val dbFile = context.getDatabasePath(SleepDatabase.DATABASE_NAME)
        return if (dbFile.exists()) dbFile.length() else 0L
    }

    /**
     * Check if database exists
     */
    fun databaseExists(context: Context): Boolean {
        val dbFile = context.getDatabasePath(SleepDatabase.DATABASE_NAME)
        return dbFile.exists()
    }

    /**
     * Format database size for display
     */
    fun formatDatabaseSize(sizeBytes: Long): String {
        return when {
            sizeBytes < 1024 -> "$sizeBytes B"
            sizeBytes < 1024 * 1024 -> "${sizeBytes / 1024} KB"
            else -> "${sizeBytes / (1024 * 1024)} MB"
        }
    }

    /**
     * Get comprehensive database info including AI metrics
     */
    fun getDatabaseInfo(context: Context): String {
        val size = getDatabaseSize(context)
        val exists = databaseExists(context)
        return "Database exists: $exists, Size: ${formatDatabaseSize(size)}, Version: ${SleepDatabase.DATABASE_VERSION}"
    }

    /**
     * Clear all data from database (for testing)
     */
    suspend fun clearAllData(database: SleepDatabase) {
        database.runInTransaction {
            database.clearAllTables()
        }
    }

    /**
     * Export comprehensive database statistics including AI metrics
     */
    suspend fun getQuickStats(database: SleepDatabase): Map<String, Any> {
        val sessionCount = database.sleepSessionDao().getSessionCount()
        val movementCount = database.movementEventDao().getTotalMovementEvents()
        val noiseCount = database.noiseEventDao().getTotalNoiseEvents()
        val phaseCount = database.sleepPhaseDao().getTotalPhaseTransitions()
        val insightCount = database.sleepInsightDao().getAllInsights().size
        val aiJobCount = database.aiGenerationJobDao().getRecentJobs(1000).size
        val aiAnalyzedCount = database.sleepSessionDao().getAIAnalyzedSessionCount()

        return mapOf(
            "total_sessions" to sessionCount,
            "ai_analyzed_sessions" to aiAnalyzedCount,
            "total_movements" to movementCount,
            "total_noise_events" to noiseCount,
            "total_phase_transitions" to phaseCount,
            "total_insights" to insightCount,
            "total_ai_jobs" to aiJobCount,
            "database_size" to getDatabaseSize(database.openHelper.context),
            "database_version" to SleepDatabase.DATABASE_VERSION
        )
    }

    /**
     * Get AI-specific performance statistics
     */
    suspend fun getAIPerformanceStats(database: SleepDatabase): Map<String, Any> {
        return try {
            val aiPerformance = database.sleepSessionDao().getAIPerformanceStatistics()
            val jobStats = database.aiGenerationJobDao().getJobStatistics()
            val insightAnalytics = database.sleepInsightDao().getInsightAnalytics()

            mapOf(
                "ai_analysis_success_rate" to (aiPerformance?.let {
                    if (it.totalSessions > 0) it.analyzedSessions.toFloat() / it.totalSessions else 0f
                } ?: 0f),
                "avg_ai_confidence" to (aiPerformance?.avgConfidence ?: 0f),
                "avg_processing_time" to (aiPerformance?.avgProcessingTime ?: 0L),
                "total_ai_jobs" to (jobStats?.totalJobs ?: 0),
                "ai_job_success_rate" to (jobStats?.let {
                    if (it.totalJobs > 0) it.completedJobs.toFloat() / it.totalJobs else 0f
                } ?: 0f),
                "total_insights" to (insightAnalytics?.totalInsights ?: 0),
                "insight_engagement_rate" to (insightAnalytics?.let {
                    if (it.totalInsights > 0) it.acknowledgedCount.toFloat() / it.totalInsights else 0f
                } ?: 0f),
                "avg_insight_rating" to (insightAnalytics?.avgRating ?: 0f)
            )
        } catch (e: Exception) {
            android.util.Log.e("DatabaseUtils", "Failed to get AI performance stats", e)
            emptyMap()
        }
    }

    /**
     * Backup database for AI training data export
     */
    suspend fun exportTrainingData(database: SleepDatabase): Map<String, Any> {
        return try {
            mapOf(
                "sessions" to database.sleepSessionDao().getAllSessionsList(),
                "insights" to database.sleepInsightDao().getAllInsights(),
                "feedback" to database.insightFeedbackDao().getFeedbackByType("RATING"),
                "interactions" to database.userInteractionDao().getInteractionsByType("ACKNOWLEDGED"),
                "preferences" to database.userPreferencesDao().getUserSetPreferences(),
                "performance" to database.aiModelPerformanceDao().getModelComparisonMetrics()
            )
        } catch (e: Exception) {
            android.util.Log.e("DatabaseUtils", "Failed to export training data", e)
            emptyMap()
        }
    }
}

/**
 * Enhanced database configuration for AI operations
 */
object DatabaseConfig {

    /**
     * Whether to enable destructive migrations (development only)
     */
    const val ENABLE_DESTRUCTIVE_MIGRATION = true // Set to false for production

    /**
     * Whether to export database schema
     */
    const val EXPORT_SCHEMA = true

    /**
     * Database journal mode for AI workloads
     */
    val JOURNAL_MODE = RoomDatabase.JournalMode.WAL // Optimal for concurrent AI operations

    /**
     * AI-specific configuration
     */
    const val MAX_SESSIONS_TO_KEEP = 2000 // Increased for AI training data
    const val MAX_AI_JOBS_TO_KEEP = 1000
    const val MAX_INTERACTIONS_TO_KEEP = 10000
    const val AI_DATA_RETENTION_DAYS = 365 // 1 year for AI learning

    /**
     * Performance configuration for AI operations
     */
    const val AI_BATCH_SIZE = 50
    const val AI_PROCESSING_TIMEOUT_MS = 60000L
    const val AI_CACHE_SIZE_MB = 100

    /**
     * Whether to enable query logging (development only)
     */
    const val ENABLE_QUERY_LOGGING = true // Set to false for production

    /**
     * AI model configuration
     */
    val SUPPORTED_AI_MODELS = listOf("GPT-4", "Claude", "Gemini", "Local-Rule-Based")
    val DEFAULT_AI_MODEL = "GPT-4"
    const val AI_CONFIDENCE_THRESHOLD = 0.7f
}

/**
 * Enhanced database health monitoring with AI metrics
 */
object DatabaseHealth {

    /**
     * Check comprehensive database integrity including AI tables
     */
    suspend fun checkIntegrity(database: SleepDatabase): Boolean {
        return try {
            database.runInTransaction {
                // Check core table integrity
                val sessionCount = database.sleepSessionDao().getSessionCount()
                val insightCount = database.sleepInsightDao().getAllInsights().size

                // Check AI table integrity
                val aiJobCount = database.aiGenerationJobDao().getRecentJobs(10).size
                val preferenceCount = database.userPreferencesDao().getAllPreferenceTypes().size

                // Basic integrity validation
                sessionCount >= 0 && insightCount >= 0 && aiJobCount >= 0 && preferenceCount >= 0
            }
        } catch (e: Exception) {
            android.util.Log.e("DatabaseHealth", "Integrity check failed", e)
            false
        }
    }

    /**
     * Get comprehensive performance metrics including AI operations
     */
    suspend fun getPerformanceMetrics(database: SleepDatabase): Map<String, Any> {
        return try {
            val basicStats = DatabaseUtils.getQuickStats(database)
            val aiStats = DatabaseUtils.getAIPerformanceStats(database)

            basicStats + aiStats + mapOf(
                "health_check_timestamp" to System.currentTimeMillis(),
                "database_version" to SleepDatabase.DATABASE_VERSION,
                "ai_integration_version" to SleepDatabase.AI_INTEGRATION_VERSION
            )
        } catch (e: Exception) {
            android.util.Log.e("DatabaseHealth", "Failed to get performance metrics", e)
            mapOf(
                "error" to true,
                "error_message" to e.message,
                "timestamp" to System.currentTimeMillis()
            )
        }
    }

    /**
     * Check AI system health specifically
     */
    suspend fun checkAISystemHealth(database: SleepDatabase): Map<String, Any> {
        return try {
            val recentJobs = database.aiGenerationJobDao().getRecentJobs(50)
            val completedJobs = recentJobs.count { it.status == "completed" }
            val failedJobs = recentJobs.count { it.status == "failed" }

            val successRate = if (recentJobs.isNotEmpty()) {
                completedJobs.toFloat() / recentJobs.size
            } else 1.0f

            val avgProcessingTime = recentJobs.filter { it.status == "completed" }
                .mapNotNull { it.processingTimeMs }
                .average()
                .takeIf { !it.isNaN() } ?: 0.0

            mapOf(
                "ai_system_healthy" to (successRate >= 0.8f),
                "recent_job_count" to recentJobs.size,
                "success_rate" to successRate,
                "failed_jobs" to failedJobs,
                "avg_processing_time_ms" to avgProcessingTime,
                "last_successful_job" to (recentJobs.firstOrNull { it.status == "completed" }?.completedAt ?: 0L)
            )
        } catch (e: Exception) {
            android.util.Log.e("DatabaseHealth", "AI health check failed", e)
            mapOf(
                "ai_system_healthy" to false,
                "error" to e.message
            )
        }
    }
}