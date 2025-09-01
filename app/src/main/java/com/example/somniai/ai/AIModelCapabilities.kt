package com.example.somniai.ai

import kotlinx.serialization.Serializable
import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * Core AI model capabilities for SomniAI sleep analysis platform
 */
object AIModelCapabilities {

    // ========== CORE TEXT CAPABILITIES ==========
    const val TEXT_GENERATION = "text_generation"
    const val TEXT_COMPLETION = "text_completion"
    const val TEXT_EDITING = "text_editing"
    const val LANGUAGE_TRANSLATION = "language_translation"
    const val GRAMMAR_CORRECTION = "grammar_correction"
    const val STYLE_ADAPTATION = "style_adaptation"

    // ========== ANALYSIS CAPABILITIES ==========
    const val ANALYSIS = "analysis"
    const val DATA_ANALYSIS = "data_analysis"
    const val STATISTICAL_ANALYSIS = "statistical_analysis"
    const val TREND_ANALYSIS = "trend_analysis"
    const val COMPARATIVE_ANALYSIS = "comparative_analysis"
    const val CORRELATION_ANALYSIS = "correlation_analysis"
    const val ANOMALY_DETECTION = "anomaly_detection"
    const val ROOT_CAUSE_ANALYSIS = "root_cause_analysis"

    // ========== SLEEP-SPECIFIC CAPABILITIES ==========
    const val SLEEP_ANALYSIS = "sleep_analysis"
    const val SLEEP_PATTERN_RECOGNITION = "sleep_pattern_recognition"
    const val SLEEP_QUALITY_ASSESSMENT = "sleep_quality_assessment"
    const val SLEEP_EFFICIENCY_CALCULATION = "sleep_efficiency_calculation"
    const val SLEEP_PHASE_DETECTION = "sleep_phase_detection"
    const val SLEEP_DISTURBANCE_ANALYSIS = "sleep_disturbance_analysis"
    const val CIRCADIAN_RHYTHM_ANALYSIS = "circadian_rhythm_analysis"
    const val SLEEP_HABIT_ANALYSIS = "sleep_habit_analysis"
    const val INSOMNIA_ASSESSMENT = "insomnia_assessment"
    const val SLEEP_HYGIENE_EVALUATION = "sleep_hygiene_evaluation"

    // ========== HEALTH & BEHAVIORAL CAPABILITIES ==========
    const val HEALTH_ASSESSMENT = "health_assessment"
    const val BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    const val LIFESTYLE_ANALYSIS = "lifestyle_analysis"
    const val WELLNESS_EVALUATION = "wellness_evaluation"
    const val STRESS_ANALYSIS = "stress_analysis"
    const val MOOD_ANALYSIS = "mood_analysis"
    const val ENERGY_LEVEL_ANALYSIS = "energy_level_analysis"
    const val RECOVERY_ANALYSIS = "recovery_analysis"

    // ========== PATTERN RECOGNITION ==========
    const val PATTERN_RECOGNITION = "pattern_recognition"
    const val TEMPORAL_PATTERN_ANALYSIS = "temporal_pattern_analysis"
    const val CYCLICAL_PATTERN_DETECTION = "cyclical_pattern_detection"
    const val BEHAVIORAL_PATTERN_ANALYSIS = "behavioral_pattern_analysis"
    const val ENVIRONMENTAL_PATTERN_ANALYSIS = "environmental_pattern_analysis"
    const val SEASONAL_PATTERN_ANALYSIS = "seasonal_pattern_analysis"

    // ========== INSIGHT GENERATION ==========
    const val INSIGHTS = "insights"
    const val INSIGHT_GENERATION = "insight_generation"
    const val ACTIONABLE_INSIGHTS = "actionable_insights"
    const val PERSONALIZED_INSIGHTS = "personalized_insights"
    const val PREDICTIVE_INSIGHTS = "predictive_insights"
    const val CONTEXTUAL_INSIGHTS = "contextual_insights"
    const val COMPARATIVE_INSIGHTS = "comparative_insights"

    // ========== RECOMMENDATION CAPABILITIES ==========
    const val RECOMMENDATION = "recommendation"
    const val RECOMMENDATION_ENGINE = "recommendation_engine"
    const val PERSONALIZED_RECOMMENDATIONS = "personalized_recommendations"
    const val BEHAVIORAL_RECOMMENDATIONS = "behavioral_recommendations"
    const val LIFESTYLE_RECOMMENDATIONS = "lifestyle_recommendations"
    const val TREATMENT_RECOMMENDATIONS = "treatment_recommendations"
    const val INTERVENTION_RECOMMENDATIONS = "intervention_recommendations"

    // ========== PREDICTION CAPABILITIES ==========
    const val PREDICTION = "prediction"
    const val OUTCOME_PREDICTION = "outcome_prediction"
    const val TREND_PREDICTION = "trend_prediction"
    const val RISK_ASSESSMENT = "risk_assessment"
    const val PROGRESS_PREDICTION = "progress_prediction"
    const val SLEEP_QUALITY_FORECASTING = "sleep_quality_forecasting"
    const val OPTIMAL_TIMING_PREDICTION = "optimal_timing_prediction"

    // ========== REASONING CAPABILITIES ==========
    const val REASONING = "reasoning"
    const val LOGICAL_REASONING = "logical_reasoning"
    const val CAUSAL_REASONING = "causal_reasoning"
    const val DIAGNOSTIC_REASONING = "diagnostic_reasoning"
    const val CLINICAL_REASONING = "clinical_reasoning"
    const val EVIDENCE_BASED_REASONING = "evidence_based_reasoning"

    // ========== CLASSIFICATION CAPABILITIES ==========
    const val CLASSIFICATION = "classification"
    const val SLEEP_STAGE_CLASSIFICATION = "sleep_stage_classification"
    const val QUALITY_CLASSIFICATION = "quality_classification"
    const val RISK_CLASSIFICATION = "risk_classification"
    const val SEVERITY_CLASSIFICATION = "severity_classification"
    const val CATEGORY_CLASSIFICATION = "category_classification"

    // ========== CONVERSATION & INTERACTION ==========
    const val CONVERSATION = "conversation"
    const val DIALOGUE_MANAGEMENT = "dialogue_management"
    const val CONTEXT_AWARENESS = "context_awareness"
    const val EMPATHETIC_RESPONSE = "empathetic_response"
    const val MOTIVATIONAL_COACHING = "motivational_coaching"
    const val EDUCATIONAL_GUIDANCE = "educational_guidance"

    // ========== SUMMARIZATION ==========
    const val SUMMARIZATION = "summarization"
    const val DATA_SUMMARIZATION = "data_summarization"
    const val TREND_SUMMARIZATION = "trend_summarization"
    const val PROGRESS_SUMMARIZATION = "progress_summarization"
    const val SESSION_SUMMARIZATION = "session_summarization"
    const val WEEKLY_SUMMARIZATION = "weekly_summarization"
    const val MONTHLY_SUMMARIZATION = "monthly_summarization"

    // ========== MULTIMODAL CAPABILITIES ==========
    const val MULTIMODAL_PROCESSING = "multimodal_processing"
    const val IMAGE_ANALYSIS = "image_analysis"
    const val CHART_INTERPRETATION = "chart_interpretation"
    const val GRAPH_ANALYSIS = "graph_analysis"
    const val VISUAL_DATA_ANALYSIS = "visual_data_analysis"

    // ========== ADVANCED AI CAPABILITIES ==========
    const val NATURAL_LANGUAGE_UNDERSTANDING = "natural_language_understanding"
    const val SEMANTIC_ANALYSIS = "semantic_analysis"
    const val SENTIMENT_ANALYSIS = "sentiment_analysis"
    const val INTENT_RECOGNITION = "intent_recognition"
    const val ENTITY_EXTRACTION = "entity_extraction"
    const val KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"

    // ========== QUALITY & VALIDATION ==========
    const val QUALITY_ASSURANCE = "quality_assurance"
    const val DATA_VALIDATION = "data_validation"
    const val CONSISTENCY_CHECK = "consistency_check"
    const val ACCURACY_VERIFICATION = "accuracy_verification"
    const val RELIABILITY_ASSESSMENT = "reliability_assessment"

    // ========== SAFETY & COMPLIANCE ==========
    const val SAFETY_ASSESSMENT = "safety_assessment"
    const val PRIVACY_PRESERVATION = "privacy_preservation"
    const val BIAS_DETECTION = "bias_detection"
    const val ETHICAL_EVALUATION = "ethical_evaluation"
    const val COMPLIANCE_CHECKING = "compliance_checking"

    /**
     * Get all available capabilities as a list
     */
    fun getAllCapabilities(): List<String> {
        return AIModelCapabilities::class.java.declaredFields
            .filter { it.type == String::class.java }
            .map { it.get(null) as String }
            .sorted()
    }
}

/**
 * Categorized capability groups for better organization
 */
@Serializable
enum class CapabilityCategory(
    val displayName: String,
    val description: String,
    val capabilities: List<String>
) {
    CORE_TEXT(
        displayName = "Core Text Processing",
        description = "Basic text generation and manipulation capabilities",
        capabilities = listOf(
            AIModelCapabilities.TEXT_GENERATION,
            AIModelCapabilities.TEXT_COMPLETION,
            AIModelCapabilities.TEXT_EDITING,
            AIModelCapabilities.LANGUAGE_TRANSLATION,
            AIModelCapabilities.GRAMMAR_CORRECTION,
            AIModelCapabilities.STYLE_ADAPTATION
        )
    ),

    SLEEP_ANALYSIS(
        displayName = "Sleep Analysis",
        description = "Specialized sleep tracking and analysis capabilities",
        capabilities = listOf(
            AIModelCapabilities.SLEEP_ANALYSIS,
            AIModelCapabilities.SLEEP_PATTERN_RECOGNITION,
            AIModelCapabilities.SLEEP_QUALITY_ASSESSMENT,
            AIModelCapabilities.SLEEP_EFFICIENCY_CALCULATION,
            AIModelCapabilities.SLEEP_PHASE_DETECTION,
            AIModelCapabilities.SLEEP_DISTURBANCE_ANALYSIS,
            AIModelCapabilities.CIRCADIAN_RHYTHM_ANALYSIS,
            AIModelCapabilities.SLEEP_HABIT_ANALYSIS,
            AIModelCapabilities.INSOMNIA_ASSESSMENT,
            AIModelCapabilities.SLEEP_HYGIENE_EVALUATION
        )
    ),

    HEALTH_BEHAVIORAL(
        displayName = "Health & Behavioral Analysis",
        description = "Health assessment and behavioral analysis capabilities",
        capabilities = listOf(
            AIModelCapabilities.HEALTH_ASSESSMENT,
            AIModelCapabilities.BEHAVIORAL_ANALYSIS,
            AIModelCapabilities.LIFESTYLE_ANALYSIS,
            AIModelCapabilities.WELLNESS_EVALUATION,
            AIModelCapabilities.STRESS_ANALYSIS,
            AIModelCapabilities.MOOD_ANALYSIS,
            AIModelCapabilities.ENERGY_LEVEL_ANALYSIS,
            AIModelCapabilities.RECOVERY_ANALYSIS
        )
    ),

    PATTERN_RECOGNITION(
        displayName = "Pattern Recognition",
        description = "Advanced pattern detection and analysis capabilities",
        capabilities = listOf(
            AIModelCapabilities.PATTERN_RECOGNITION,
            AIModelCapabilities.TEMPORAL_PATTERN_ANALYSIS,
            AIModelCapabilities.CYCLICAL_PATTERN_DETECTION,
            AIModelCapabilities.BEHAVIORAL_PATTERN_ANALYSIS,
            AIModelCapabilities.ENVIRONMENTAL_PATTERN_ANALYSIS,
            AIModelCapabilities.SEASONAL_PATTERN_ANALYSIS
        )
    ),

    INSIGHTS_RECOMMENDATIONS(
        displayName = "Insights & Recommendations",
        description = "Intelligent insight generation and recommendation capabilities",
        capabilities = listOf(
            AIModelCapabilities.INSIGHTS,
            AIModelCapabilities.INSIGHT_GENERATION,
            AIModelCapabilities.ACTIONABLE_INSIGHTS,
            AIModelCapabilities.PERSONALIZED_INSIGHTS,
            AIModelCapabilities.PREDICTIVE_INSIGHTS,
            AIModelCapabilities.CONTEXTUAL_INSIGHTS,
            AIModelCapabilities.COMPARATIVE_INSIGHTS,
            AIModelCapabilities.RECOMMENDATION,
            AIModelCapabilities.RECOMMENDATION_ENGINE,
            AIModelCapabilities.PERSONALIZED_RECOMMENDATIONS,
            AIModelCapabilities.BEHAVIORAL_RECOMMENDATIONS,
            AIModelCapabilities.LIFESTYLE_RECOMMENDATIONS
        )
    ),

    PREDICTIVE_ANALYTICS(
        displayName = "Predictive Analytics",
        description = "Advanced prediction and forecasting capabilities",
        capabilities = listOf(
            AIModelCapabilities.PREDICTION,
            AIModelCapabilities.OUTCOME_PREDICTION,
            AIModelCapabilities.TREND_PREDICTION,
            AIModelCapabilities.RISK_ASSESSMENT,
            AIModelCapabilities.PROGRESS_PREDICTION,
            AIModelCapabilities.SLEEP_QUALITY_FORECASTING,
            AIModelCapabilities.OPTIMAL_TIMING_PREDICTION
        )
    ),

    REASONING_ANALYSIS(
        displayName = "Reasoning & Analysis",
        description = "Advanced reasoning and analytical capabilities",
        capabilities = listOf(
            AIModelCapabilities.REASONING,
            AIModelCapabilities.LOGICAL_REASONING,
            AIModelCapabilities.CAUSAL_REASONING,
            AIModelCapabilities.DIAGNOSTIC_REASONING,
            AIModelCapabilities.CLINICAL_REASONING,
            AIModelCapabilities.EVIDENCE_BASED_REASONING,
            AIModelCapabilities.ANALYSIS,
            AIModelCapabilities.DATA_ANALYSIS,
            AIModelCapabilities.STATISTICAL_ANALYSIS,
            AIModelCapabilities.TREND_ANALYSIS
        )
    ),

    INTERACTION_COMMUNICATION(
        displayName = "Interaction & Communication",
        description = "Conversational and user interaction capabilities",
        capabilities = listOf(
            AIModelCapabilities.CONVERSATION,
            AIModelCapabilities.DIALOGUE_MANAGEMENT,
            AIModelCapabilities.CONTEXT_AWARENESS,
            AIModelCapabilities.EMPATHETIC_RESPONSE,
            AIModelCapabilities.MOTIVATIONAL_COACHING,
            AIModelCapabilities.EDUCATIONAL_GUIDANCE,
            AIModelCapabilities.NATURAL_LANGUAGE_UNDERSTANDING,
            AIModelCapabilities.SENTIMENT_ANALYSIS
        )
    ),

    MULTIMODAL(
        displayName = "Multimodal Processing",
        description = "Cross-modal data processing capabilities",
        capabilities = listOf(
            AIModelCapabilities.MULTIMODAL_PROCESSING,
            AIModelCapabilities.IMAGE_ANALYSIS,
            AIModelCapabilities.CHART_INTERPRETATION,
            AIModelCapabilities.GRAPH_ANALYSIS,
            AIModelCapabilities.VISUAL_DATA_ANALYSIS
        )
    ),

    QUALITY_SAFETY(
        displayName = "Quality & Safety",
        description = "Quality assurance and safety capabilities",
        capabilities = listOf(
            AIModelCapabilities.QUALITY_ASSURANCE,
            AIModelCapabilities.DATA_VALIDATION,
            AIModelCapabilities.CONSISTENCY_CHECK,
            AIModelCapabilities.ACCURACY_VERIFICATION,
            AIModelCapabilities.RELIABILITY_ASSESSMENT,
            AIModelCapabilities.SAFETY_ASSESSMENT,
            AIModelCapabilities.PRIVACY_PRESERVATION,
            AIModelCapabilities.BIAS_DETECTION,
            AIModelCapabilities.ETHICAL_EVALUATION
        )
    );

    /**
     * Check if this category contains a specific capability
     */
    fun hasCapability(capability: String): Boolean {
        return capabilities.contains(capability)
    }

    /**
     * Get capabilities count for this category
     */
    val capabilityCount: Int
        get() = capabilities.size
}

/**
 * Capability requirement levels for different tasks
 */
@Serializable
enum class CapabilityRequirement(
    val level: String,
    val description: String,
    val confidenceThreshold: Float
) {
    ESSENTIAL(
        level = "essential",
        description = "Must have this capability to perform the task",
        confidenceThreshold = 0.9f
    ),
    RECOMMENDED(
        level = "recommended",
        description = "Strongly recommended for optimal performance",
        confidenceThreshold = 0.8f
    ),
    OPTIONAL(
        level = "optional",
        description = "Nice to have but not required",
        confidenceThreshold = 0.6f
    ),
    EXPERIMENTAL(
        level = "experimental",
        description = "Experimental capability for advanced use cases",
        confidenceThreshold = 0.5f
    )
}

/**
 * Task-specific capability requirements for SomniAI operations
 */
@Parcelize
@Serializable
data class TaskCapabilityRequirements(
    val taskName: String,
    val taskDescription: String,
    val essential: List<String>,
    val recommended: List<String> = emptyList(),
    val optional: List<String> = emptyList(),
    val experimental: List<String> = emptyList(),
    val minimumModelTier: String = "STANDARD",
    val estimatedComplexity: Float = 1.0f
) : Parcelable {

    /**
     * Get all required capabilities regardless of level
     */
    val allRequiredCapabilities: List<String>
        get() = essential + recommended + optional + experimental

    /**
     * Get capability requirements by level
     */
    fun getCapabilitiesByLevel(requirement: CapabilityRequirement): List<String> {
        return when (requirement) {
            CapabilityRequirement.ESSENTIAL -> essential
            CapabilityRequirement.RECOMMENDED -> recommended
            CapabilityRequirement.OPTIONAL -> optional
            CapabilityRequirement.EXPERIMENTAL -> experimental
        }
    }

    /**
     * Check if a model meets the essential requirements
     */
    fun doesModelMeetEssentialRequirements(modelCapabilities: List<String>): Boolean {
        return essential.all { modelCapabilities.contains(it) }
    }

    /**
     * Calculate compatibility score with a model
     */
    fun calculateCompatibilityScore(modelCapabilities: List<String>): Float {
        val essentialScore = essential.count { modelCapabilities.contains(it) }.toFloat() / essential.size
        val recommendedScore = if (recommended.isNotEmpty()) {
            recommended.count { modelCapabilities.contains(it) }.toFloat() / recommended.size
        } else 1.0f
        val optionalScore = if (optional.isNotEmpty()) {
            optional.count { modelCapabilities.contains(it) }.toFloat() / optional.size
        } else 1.0f

        return (essentialScore * 0.6f) + (recommendedScore * 0.3f) + (optionalScore * 0.1f)
    }

    companion object {
        /**
         * Predefined task requirements for common SomniAI operations
         */
        val SLEEP_SESSION_ANALYSIS = TaskCapabilityRequirements(
            taskName = "Sleep Session Analysis",
            taskDescription = "Analyze individual sleep sessions for quality and patterns",
            essential = listOf(
                AIModelCapabilities.SLEEP_ANALYSIS,
                AIModelCapabilities.SLEEP_QUALITY_ASSESSMENT,
                AIModelCapabilities.DATA_ANALYSIS,
                AIModelCapabilities.INSIGHT_GENERATION
            ),
            recommended = listOf(
                AIModelCapabilities.SLEEP_PATTERN_RECOGNITION,
                AIModelCapabilities.BEHAVIORAL_ANALYSIS,
                AIModelCapabilities.STATISTICAL_ANALYSIS
            ),
            optional = listOf(
                AIModelCapabilities.PREDICTIVE_INSIGHTS,
                AIModelCapabilities.COMPARATIVE_ANALYSIS,
                AIModelCapabilities.VISUAL_DATA_ANALYSIS
            )
        )

        val PERSONALIZED_RECOMMENDATIONS = TaskCapabilityRequirements(
            taskName = "Personalized Recommendations",
            taskDescription = "Generate personalized sleep improvement recommendations",
            essential = listOf(
                AIModelCapabilities.RECOMMENDATION_ENGINE,
                AIModelCapabilities.PERSONALIZED_RECOMMENDATIONS,
                AIModelCapabilities.BEHAVIORAL_ANALYSIS,
                AIModelCapabilities.LIFESTYLE_ANALYSIS
            ),
            recommended = listOf(
                AIModelCapabilities.PATTERN_RECOGNITION,
                AIModelCapabilities.TREND_ANALYSIS,
                AIModelCapabilities.CONTEXTUAL_INSIGHTS,
                AIModelCapabilities.MOTIVATIONAL_COACHING
            ),
            optional = listOf(
                AIModelCapabilities.PREDICTIVE_INSIGHTS,
                AIModelCapabilities.RISK_ASSESSMENT,
                AIModelCapabilities.SEASONAL_PATTERN_ANALYSIS
            )
        )

        val CONVERSATIONAL_INSIGHTS = TaskCapabilityRequirements(
            taskName = "Conversational Insights",
            taskDescription = "Interactive conversation about sleep data and insights",
            essential = listOf(
                AIModelCapabilities.CONVERSATION,
                AIModelCapabilities.NATURAL_LANGUAGE_UNDERSTANDING,
                AIModelCapabilities.CONTEXT_AWARENESS,
                AIModelCapabilities.EMPATHETIC_RESPONSE
            ),
            recommended = listOf(
                AIModelCapabilities.DIALOGUE_MANAGEMENT,
                AIModelCapabilities.EDUCATIONAL_GUIDANCE,
                AIModelCapabilities.MOTIVATIONAL_COACHING,
                AIModelCapabilities.SENTIMENT_ANALYSIS
            ),
            optional = listOf(
                AIModelCapabilities.MULTIMODAL_PROCESSING,
                AIModelCapabilities.CHART_INTERPRETATION,
                AIModelCapabilities.VISUAL_DATA_ANALYSIS
            )
        )

        val TREND_ANALYSIS = TaskCapabilityRequirements(
            taskName = "Sleep Trend Analysis",
            taskDescription = "Long-term pattern and trend analysis of sleep data",
            essential = listOf(
                AIModelCapabilities.TREND_ANALYSIS,
                AIModelCapabilities.PATTERN_RECOGNITION,
                AIModelCapabilities.TEMPORAL_PATTERN_ANALYSIS,
                AIModelCapabilities.STATISTICAL_ANALYSIS
            ),
            recommended = listOf(
                AIModelCapabilities.CYCLICAL_PATTERN_DETECTION,
                AIModelCapabilities.SEASONAL_PATTERN_ANALYSIS,
                AIModelCapabilities.COMPARATIVE_ANALYSIS,
                AIModelCapabilities.CORRELATION_ANALYSIS
            ),
            optional = listOf(
                AIModelCapabilities.PREDICTIVE_INSIGHTS,
                AIModelCapabilities.TREND_PREDICTION,
                AIModelCapabilities.ANOMALY_DETECTION
            )
        )

        val HEALTH_ASSESSMENT = TaskCapabilityRequirements(
            taskName = "Health Assessment",
            taskDescription = "Comprehensive health and wellness assessment based on sleep data",
            essential = listOf(
                AIModelCapabilities.HEALTH_ASSESSMENT,
                AIModelCapabilities.CLINICAL_REASONING,
                AIModelCapabilities.RISK_ASSESSMENT,
                AIModelCapabilities.EVIDENCE_BASED_REASONING
            ),
            recommended = listOf(
                AIModelCapabilities.WELLNESS_EVALUATION,
                AIModelCapabilities.STRESS_ANALYSIS,
                AIModelCapabilities.RECOVERY_ANALYSIS,
                AIModelCapabilities.DIAGNOSTIC_REASONING
            ),
            optional = listOf(
                AIModelCapabilities.TREATMENT_RECOMMENDATIONS,
                AIModelCapabilities.INTERVENTION_RECOMMENDATIONS,
                AIModelCapabilities.COMPLIANCE_CHECKING
            ),
            minimumModelTier = "PREMIUM"
        )

        /**
         * Get all predefined task requirements
         */
        fun getAllTaskRequirements(): List<TaskCapabilityRequirements> {
            return listOf(
                SLEEP_SESSION_ANALYSIS,
                PERSONALIZED_RECOMMENDATIONS,
                CONVERSATIONAL_INSIGHTS,
                TREND_ANALYSIS,
                HEALTH_ASSESSMENT
            )
        }
    }
}

/**
 * Capability validation and matching utilities
 */
object CapabilityMatcher {

    /**
     * Find models that meet specific capability requirements
     */
    fun findModelsWithCapabilities(
        requiredCapabilities: List<String>,
        availableModels: List<AIModel>,
        strictMode: Boolean = false
    ): List<Pair<AIModel, Float>> {
        return availableModels.mapNotNull { model ->
            val matchScore = calculateCapabilityMatchScore(
                requiredCapabilities,
                model.capabilities,
                strictMode
            )

            if (matchScore > 0f) {
                Pair(model, matchScore)
            } else {
                null
            }
        }.sortedByDescending { it.second }
    }

    /**
     * Calculate how well a model's capabilities match requirements
     */
    private fun calculateCapabilityMatchScore(
        required: List<String>,
        available: List<String>,
        strictMode: Boolean
    ): Float {
        if (required.isEmpty()) return 1.0f

        val matchedCount = required.count { available.contains(it) }
        val matchRatio = matchedCount.toFloat() / required.size

        return if (strictMode && matchRatio < 1.0f) {
            0.0f // Must have all capabilities in strict mode
        } else {
            matchRatio
        }
    }

    /**
     * Get capability recommendations for a specific task
     */
    fun getCapabilityRecommendations(
        taskType: String,
        userContext: Map<String, Any> = emptyMap()
    ): TaskCapabilityRequirements? {
        return when (taskType.lowercase()) {
            "sleep_analysis", "session_analysis" -> TaskCapabilityRequirements.SLEEP_SESSION_ANALYSIS
            "recommendations", "personalized_recommendations" -> TaskCapabilityRequirements.PERSONALIZED_RECOMMENDATIONS
            "conversation", "chat", "interactive" -> TaskCapabilityRequirements.CONVERSATIONAL_INSIGHTS
            "trends", "pattern_analysis", "long_term" -> TaskCapabilityRequirements.TREND_ANALYSIS
            "health", "medical", "clinical" -> TaskCapabilityRequirements.HEALTH_ASSESSMENT
            else -> null
        }
    }

    /**
     * Validate that a model can handle a specific task
     */
    fun validateModelForTask(
        model: AIModel,
        taskRequirements: TaskCapabilityRequirements,
        minimumScore: Float = 0.7f
    ): ValidationResult {
        val compatibilityScore = taskRequirements.calculateCompatibilityScore(model.capabilities)
        val meetsEssential = taskRequirements.doesModelMeetEssentialRequirements(model.capabilities)

        val isValid = meetsEssential && compatibilityScore >= minimumScore

        val missingEssential = taskRequirements.essential.filter { !model.capabilities.contains(it) }
        val missingRecommended = taskRequirements.recommended.filter { !model.capabilities.contains(it) }

        return ValidationResult(
            isValid = isValid,
            compatibilityScore = compatibilityScore,
            missingEssentialCapabilities = missingEssential,
            missingRecommendedCapabilities = missingRecommended,
            modelTierSufficient = model.tier.name >= taskRequirements.minimumModelTier
        )
    }
}

/**
 * Result of capability validation
 */
@Parcelize
@Serializable
data class ValidationResult(
    val isValid: Boolean,
    val compatibilityScore: Float,
    val missingEssentialCapabilities: List<String>,
    val missingRecommendedCapabilities: List<String>,
    val modelTierSufficient: Boolean,
    val validationTimestamp: Long = System.currentTimeMillis()
) : Parcelable {

    /**
     * Get human-readable validation summary
     */
    fun getValidationSummary(): String {
        return when {
            isValid -> "Model is suitable for this task (${(compatibilityScore * 100).toInt()}% compatibility)"
            missingEssentialCapabilities.isNotEmpty() ->
                "Model is missing essential capabilities: ${missingEssentialCapabilities.joinToString()}"
            !modelTierSufficient -> "Model tier is insufficient for this task"
            else -> "Model compatibility score too low: ${(compatibilityScore * 100).toInt()}%"
        }
    }

    /**
     * Get recommendations for improvement
     */
    fun getImprovementRecommendations(): List<String> {
        val recommendations = mutableListOf<String>()

        if (missingEssentialCapabilities.isNotEmpty()) {
            recommendations.add("Consider upgrading to a model with: ${missingEssentialCapabilities.joinToString()}")
        }

        if (missingRecommendedCapabilities.isNotEmpty() && missingRecommendedCapabilities.size > 2) {
            recommendations.add("For better results, consider a model with: ${missingRecommendedCapabilities.take(3).joinToString()}")
        }

        if (!modelTierSufficient) {
            recommendations.add("Consider upgrading to a higher tier model for this task")
        }

        return recommendations
    }
}

/**
 * Constants for capability management
 */
object CapabilityConstants {
    const val MIN_COMPATIBILITY_SCORE = 0.7f
    const val EXCELLENT_COMPATIBILITY_SCORE = 0.9f
    const val MAX_MISSING_ESSENTIAL_CAPABILITIES = 0
    const val MAX_MISSING_RECOMMENDED_CAPABILITIES = 2

    // Sleep-specific capability groups
    val CORE_SLEEP_CAPABILITIES = listOf(
        AIModelCapabilities.SLEEP_ANALYSIS,
        AIModelCapabilities.SLEEP_QUALITY_ASSESSMENT,
        AIModelCapabilities.PATTERN_RECOGNITION,
        AIModelCapabilities.INSIGHT_GENERATION
    )

    val ADVANCED_SLEEP_CAPABILITIES = listOf(
        AIModelCapabilities.SLEEP_PHASE_DETECTION,
        AIModelCapabilities.CIRCADIAN_RHYTHM_ANALYSIS,
        AIModelCapabilities.PREDICTIVE_INSIGHTS,
        AIModelCapabilities.CLINICAL_REASONING
    )

    val INTERACTION_CAPABILITIES = listOf(
        AIModelCapabilities.CONVERSATION,
        AIModelCapabilities.EMPATHETIC_RESPONSE,
        AIModelCapabilities.MOTIVATIONAL_COACHING,
        AIModelCapabilities.EDUCATIONAL_GUIDANCE
    )
}