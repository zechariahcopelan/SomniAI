package com.example.somniai.ai

/**
 * AI Constants for SomniAI Sleep Analysis
 *
 * Contains all configuration constants, thresholds, and settings
 * used throughout the AI analysis pipeline
 */
object AIConstants {

    // ========== CONFIDENCE THRESHOLDS ==========
    const val MIN_CONFIDENCE_SCORE = 0.3f
    const val HIGH_CONFIDENCE_THRESHOLD = 0.8f
    const val MEDIUM_CONFIDENCE_THRESHOLD = 0.6f
    const val LOW_CONFIDENCE_THRESHOLD = 0.4f
    const val VERY_HIGH_CONFIDENCE_THRESHOLD = 0.9f

    // ========== QUALITY SCORING ==========
    const val MIN_QUALITY_SCORE = 0.0f
    const val MAX_QUALITY_SCORE = 10.0f
    const val HIGH_QUALITY_THRESHOLD = 7.5f
    const val GOOD_QUALITY_THRESHOLD = 6.0f
    const val FAIR_QUALITY_THRESHOLD = 4.5f
    const val POOR_QUALITY_THRESHOLD = 3.0f

    // ========== INSIGHT GENERATION ==========
    const val MAX_INSIGHTS_PER_SESSION = 10
    const val MIN_INSIGHTS_PER_SESSION = 1
    const val HIGH_PRIORITY_INSIGHT_LIMIT = 3
    const val MEDIUM_PRIORITY_INSIGHT_LIMIT = 5
    const val LOW_PRIORITY_INSIGHT_LIMIT = 7

    // ========== DATA VALIDATION ==========
    const val MIN_DATA_COMPLETENESS = 0.7f
    const val HIGH_DATA_COMPLETENESS = 0.9f
    const val MIN_SAMPLE_SIZE = 7
    const val RECOMMENDED_SAMPLE_SIZE = 30
    const val MAX_OUTLIER_PERCENTAGE = 0.15f

    // ========== TREND ANALYSIS ==========
    const val MIN_TREND_CONFIDENCE = 0.6f
    const val SIGNIFICANT_TREND_THRESHOLD = 0.8f
    const val TREND_ANALYSIS_MIN_DAYS = 7
    const val TREND_ANALYSIS_RECOMMENDED_DAYS = 30
    const val SEASONAL_ANALYSIS_MIN_DAYS = 90

    // ========== SLEEP METRICS THRESHOLDS ==========
    const val MIN_SLEEP_DURATION_HOURS = 4.0f
    const val RECOMMENDED_SLEEP_DURATION_HOURS = 8.0f
    const val MAX_SLEEP_DURATION_HOURS = 12.0f
    const val HIGH_EFFICIENCY_THRESHOLD = 0.85f
    const val GOOD_EFFICIENCY_THRESHOLD = 0.75f
    const val POOR_EFFICIENCY_THRESHOLD = 0.65f

    // ========== MOVEMENT AND NOISE ==========
    const val LOW_MOVEMENT_THRESHOLD = 10
    const val MODERATE_MOVEMENT_THRESHOLD = 30
    const val HIGH_MOVEMENT_THRESHOLD = 60
    const val QUIET_NOISE_THRESHOLD = 40.0f // dB
    const val MODERATE_NOISE_THRESHOLD = 55.0f // dB
    const val LOUD_NOISE_THRESHOLD = 70.0f // dB

    // ========== AI MODEL CONFIGURATION ==========
    const val DEFAULT_AI_MODEL = "GPT3_5_TURBO"
    const val FALLBACK_AI_MODEL = "LOCAL_MODEL"
    const val MAX_PROCESSING_TIME_MS = 30000L // 30 seconds
    const val AI_REQUEST_TIMEOUT_MS = 15000L // 15 seconds
    const val MAX_RETRY_ATTEMPTS = 3

    // ========== TOKEN LIMITS ==========
    const val MAX_PROMPT_TOKENS = 2000
    const val MAX_COMPLETION_TOKENS = 1000
    const val ESTIMATED_COST_PER_TOKEN = 0.0001 // $0.0001

    // ========== PERFORMANCE THRESHOLDS ==========
    const val FAST_PROCESSING_THRESHOLD_MS = 1000L
    const val ACCEPTABLE_PROCESSING_THRESHOLD_MS = 5000L
    const val SLOW_PROCESSING_THRESHOLD_MS = 10000L

    // ========== PATTERN RECOGNITION ==========
    const val MIN_PATTERN_STRENGTH = 0.5f
    const val STRONG_PATTERN_THRESHOLD = 0.8f
    const val PATTERN_FREQUENCY_THRESHOLD = 0.6f
    const val MIN_PATTERN_OCCURRENCES = 3

    // ========== ANOMALY DETECTION ==========
    const val ANOMALY_DETECTION_THRESHOLD = 2.0f // Standard deviations
    const val SEVERE_ANOMALY_THRESHOLD = 3.0f
    const val MAX_ANOMALIES_PER_SESSION = 5
    const val ANOMALY_CONFIDENCE_THRESHOLD = 0.7f

    // ========== RECOMMENDATION SCORING ==========
    const val HIGH_IMPACT_RECOMMENDATION_THRESHOLD = 0.8f
    const val MEDIUM_IMPACT_RECOMMENDATION_THRESHOLD = 0.5f
    const val LOW_IMPACT_RECOMMENDATION_THRESHOLD = 0.3f
    const val ACTIONABLE_RECOMMENDATION_THRESHOLD = 0.7f

    // ========== TIME-BASED CONSTANTS ==========
    const val MINUTES_PER_HOUR = 60
    const val SECONDS_PER_MINUTE = 60
    const val MILLISECONDS_PER_SECOND = 1000L
    const val HOURS_PER_DAY = 24
    const val DAYS_PER_WEEK = 7
    const val WEEKS_PER_MONTH = 4
    const val MONTHS_PER_YEAR = 12

    // ========== SLEEP PHASE PERCENTAGES ==========
    const val IDEAL_DEEP_SLEEP_PERCENTAGE = 0.20f // 20%
    const val IDEAL_REM_SLEEP_PERCENTAGE = 0.25f // 25%
    const val IDEAL_LIGHT_SLEEP_PERCENTAGE = 0.50f // 50%
    const val MAX_AWAKE_PERCENTAGE = 0.10f // 10%

    // ========== CONSISTENCY THRESHOLDS ==========
    const val HIGH_CONSISTENCY_THRESHOLD = 0.9f
    const val GOOD_CONSISTENCY_THRESHOLD = 0.7f
    const val POOR_CONSISTENCY_THRESHOLD = 0.5f
    const val BEDTIME_CONSISTENCY_WINDOW_MINUTES = 30
    const val WAKETIME_CONSISTENCY_WINDOW_MINUTES = 30

    // ========== PERSONALIZATION ==========
    const val MIN_PERSONALIZATION_SCORE = 0.0f
    const val HIGH_PERSONALIZATION_THRESHOLD = 0.8f
    const val PERSONALIZATION_LEARNING_PERIOD_DAYS = 14
    const val USER_FEEDBACK_WEIGHT = 0.3f

    // ========== ERROR HANDLING ==========
    const val MAX_ERROR_RETRIES = 3
    const val ERROR_RETRY_DELAY_MS = 1000L
    const val CACHE_EXPIRY_HOURS = 24
    const val MAX_CACHE_SIZE = 1000

    // ========== STATISTICAL CONSTANTS ==========
    const val STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05f // p-value
    const val MIN_R_SQUARED_FOR_TREND = 0.5f
    const val CORRELATION_STRENGTH_THRESHOLD = 0.7f
    const val OUTLIER_Z_SCORE_THRESHOLD = 2.5f

    // ========== VERSION AND COMPATIBILITY ==========
    const val AI_ENGINE_VERSION = "1.0.0"
    const val MINIMUM_API_VERSION = 21
    const val SUPPORTED_DATA_FORMAT_VERSION = "2.0"

    // ========== BATCH PROCESSING ==========
    const val MAX_BATCH_SIZE = 50
    const val BATCH_PROCESSING_TIMEOUT_MS = 60000L // 1 minute
    const val PARALLEL_PROCESSING_THREAD_COUNT = 4

    // ========== DEBUGGING AND LOGGING ==========
    const val ENABLE_DEBUG_LOGGING = true
    const val LOG_AI_PROCESSING_TIMES = true
    const val LOG_CONFIDENCE_SCORES = false
    const val MAX_LOG_ENTRIES = 1000

    // ========== MODEL-SPECIFIC SETTINGS ==========
    object OpenAI {
        const val GPT35_MAX_TOKENS = 4096
        const val GPT4_MAX_TOKENS = 8192
        const val TEMPERATURE = 0.7f
        const val TOP_P = 0.9f
    }

    object Anthropic {
        const val CLAUDE_MAX_TOKENS = 100000
        const val TEMPERATURE = 0.7f
    }

    object Google {
        const val GEMINI_MAX_TOKENS = 30720
        const val TEMPERATURE = 0.7f
        const val TOP_K = 40
    }

    // ========== FEATURE FLAGS ==========
    const val ENABLE_ADVANCED_ANALYTICS = true
    const val ENABLE_PREDICTIVE_INSIGHTS = true
    const val ENABLE_ANOMALY_DETECTION = true
    const val ENABLE_PATTERN_RECOGNITION = true
    const val ENABLE_PERSONALIZATION = true
    const val ENABLE_COMPARATIVE_ANALYSIS = true
}