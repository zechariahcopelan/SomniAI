package com.example.somniai.ui.adapters

import android.view.View
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.example.somniai.R
import com.example.somniai.data.SleepRecommendation
import com.example.somniai.data.RecommendationType
import com.example.somniai.databinding.CardRecommendationBinding

/**
 * ViewHolder classes for sleep recommendations using a single flexible layout
 */

/**
 * Base ViewHolder for all recommendation types
 */
abstract class BaseRecommendationViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
    abstract fun bind(recommendation: SleepRecommendation, onItemClick: ((SleepRecommendation) -> Unit)?)
}

/**
 * Main ViewHolder that handles all recommendation types using the card_recommendation.xml layout
 */
class RecommendationViewHolder(
    private val binding: CardRecommendationBinding
) : BaseRecommendationViewHolder(binding.root) {

    override fun bind(recommendation: SleepRecommendation, onItemClick: ((SleepRecommendation) -> Unit)?) {
        with(binding) {
            // Set basic content
            recommendationTitle.text = recommendation.title
            recommendationDescription.text = recommendation.description
            recommendationAction.text = recommendation.actionText ?: "View Details"

            // Set icon based on recommendation type
            recommendationIcon.setImageResource(getIconForType(recommendation.type))

            // Set colors/styling based on recommendation type and priority
            setupStyling(recommendation)

            // Set click listeners
            root.setOnClickListener { onItemClick?.invoke(recommendation) }
            recommendationAction.setOnClickListener { onItemClick?.invoke(recommendation) }

            // Set priority indicator
            setPriorityIndicator(recommendation.priority)
        }
    }

    private fun getIconForType(type: RecommendationType): Int {
        return when (type) {
            RecommendationType.SLEEP_QUALITY -> R.drawable.ic_sleep_quality
            RecommendationType.SLEEP_DURATION -> R.drawable.ic_sleep_duration
            RecommendationType.BEDTIME_CONSISTENCY -> R.drawable.ic_bedtime_consistency
            RecommendationType.ENVIRONMENT -> R.drawable.ic_environment
            RecommendationType.STRESS_MANAGEMENT -> R.drawable.ic_stress_management
            RecommendationType.EXERCISE -> R.drawable.ic_exercise
            RecommendationType.NUTRITION -> R.drawable.ic_nutrition
            RecommendationType.SCREEN_TIME -> R.drawable.ic_screen_time
            RecommendationType.TEMPERATURE -> R.drawable.ic_temperature
            RecommendationType.NOISE -> R.drawable.ic_noise
            RecommendationType.LIGHTING -> R.drawable.ic_lighting
            RecommendationType.ROUTINE -> R.drawable.ic_routine
            else -> R.drawable.ic_default_recommendation
        }
    }

    private fun setupStyling(recommendation: SleepRecommendation) {
        val context = binding.root.context

        // Set background tint based on recommendation type
        val backgroundTint = when (recommendation.type) {
            RecommendationType.SLEEP_QUALITY -> ContextCompat.getColor(context, R.color.sleep_quality_tint)
            RecommendationType.SLEEP_DURATION -> ContextCompat.getColor(context, R.color.sleep_duration_tint)
            RecommendationType.BEDTIME_CONSISTENCY -> ContextCompat.getColor(context, R.color.bedtime_consistency_tint)
            RecommendationType.ENVIRONMENT -> ContextCompat.getColor(context, R.color.environment_tint)
            RecommendationType.STRESS_MANAGEMENT -> ContextCompat.getColor(context, R.color.stress_management_tint)
            RecommendationType.EXERCISE -> ContextCompat.getColor(context, R.color.exercise_tint)
            RecommendationType.NUTRITION -> ContextCompat.getColor(context, R.color.nutrition_tint)
            RecommendationType.SCREEN_TIME -> ContextCompat.getColor(context, R.color.screen_time_tint)
            RecommendationType.TEMPERATURE -> ContextCompat.getColor(context, R.color.temperature_tint)
            RecommendationType.NOISE -> ContextCompat.getColor(context, R.color.noise_tint)
            RecommendationType.LIGHTING -> ContextCompat.getColor(context, R.color.lighting_tint)
            RecommendationType.ROUTINE -> ContextCompat.getColor(context, R.color.routine_tint)
            else -> ContextCompat.getColor(context, R.color.default_recommendation_tint)
        }

        // Apply subtle background tint
        binding.recommendationCard.setCardBackgroundColor(backgroundTint)

        // Set icon tint to match
        binding.recommendationIcon.setColorFilter(
            ContextCompat.getColor(context, R.color.recommendation_icon_tint)
        )
    }

    private fun setPriorityIndicator(priority: Int) {
        val context = binding.root.context

        // Set priority indicator color and visibility
        when (priority) {
            1 -> { // High priority
                binding.priorityIndicator?.visibility = View.VISIBLE
                binding.priorityIndicator?.setBackgroundColor(
                    ContextCompat.getColor(context, R.color.priority_high)
                )
            }
            2 -> { // Medium priority
                binding.priorityIndicator?.visibility = View.VISIBLE
                binding.priorityIndicator?.setBackgroundColor(
                    ContextCompat.getColor(context, R.color.priority_medium)
                )
            }
            3 -> { // Low priority
                binding.priorityIndicator?.visibility = View.GONE
            }
            else -> {
                binding.priorityIndicator?.visibility = View.GONE
            }
        }
    }
}

/**
 * Specialized ViewHolder for compact recommendation display (if needed)
 */
class CompactRecommendationViewHolder(
    private val binding: CardRecommendationBinding
) : BaseRecommendationViewHolder(binding.root) {

    override fun bind(recommendation: SleepRecommendation, onItemClick: ((SleepRecommendation) -> Unit)?) {
        with(binding) {
            // Compact layout - hide description, show only title and icon
            recommendationTitle.text = recommendation.title
            recommendationDescription.visibility = View.GONE
            recommendationAction.visibility = View.GONE

            recommendationIcon.setImageResource(getIconForType(recommendation.type))

            root.setOnClickListener { onItemClick?.invoke(recommendation) }
        }
    }

    private fun getIconForType(type: RecommendationType): Int {
        return when (type) {
            RecommendationType.SLEEP_QUALITY -> R.drawable.ic_sleep_quality
            RecommendationType.SLEEP_DURATION -> R.drawable.ic_sleep_duration
            RecommendationType.BEDTIME_CONSISTENCY -> R.drawable.ic_bedtime_consistency
            RecommendationType.ENVIRONMENT -> R.drawable.ic_environment
            RecommendationType.STRESS_MANAGEMENT -> R.drawable.ic_stress_management
            RecommendationType.EXERCISE -> R.drawable.ic_exercise
            RecommendationType.NUTRITION -> R.drawable.ic_nutrition
            RecommendationType.SCREEN_TIME -> R.drawable.ic_screen_time
            RecommendationType.TEMPERATURE -> R.drawable.ic_temperature
            RecommendationType.NOISE -> R.drawable.ic_noise
            RecommendationType.LIGHTING -> R.drawable.ic_lighting
            RecommendationType.ROUTINE -> R.drawable.ic_routine
            else -> R.drawable.ic_default_recommendation
        }
    }
}

/**
 * ViewHolder factory for creating appropriate ViewHolder based on view type
 */
object RecommendationViewHolderFactory {

    const val VIEW_TYPE_NORMAL = 0
    const val VIEW_TYPE_COMPACT = 1

    fun createViewHolder(
        binding: CardRecommendationBinding,
        viewType: Int
    ): BaseRecommendationViewHolder {
        return when (viewType) {
            VIEW_TYPE_COMPACT -> CompactRecommendationViewHolder(binding)
            else -> RecommendationViewHolder(binding)
        }
    }
}

/**
 * ViewHolder for AI-generated recommendations with additional metadata
 */
class AIRecommendationViewHolder(
    private val binding: CardRecommendationBinding
) : BaseRecommendationViewHolder(binding.root) {

    override fun bind(recommendation: SleepRecommendation, onItemClick: ((SleepRecommendation) -> Unit)?) {
        // Use the standard ViewHolder logic
        val standardViewHolder = RecommendationViewHolder(binding)
        standardViewHolder.bind(recommendation, onItemClick)

        // Add AI-specific elements
        with(binding) {
            // Show AI badge if this is AI-generated
            if (recommendation.isAiGenerated) {
                aiGeneratedBadge?.visibility = View.VISIBLE
                aiConfidenceScore?.text = "Confidence: ${(recommendation.confidence * 100).toInt()}%"
                aiConfidenceScore?.visibility = View.VISIBLE
            } else {
                aiGeneratedBadge?.visibility = View.GONE
                aiConfidenceScore?.visibility = View.GONE
            }

            // Show effectiveness tracking if available
            recommendation.effectivenessScore?.let { score ->
                effectivenessIndicator?.visibility = View.VISIBLE
                effectivenessScore?.text = "Effectiveness: ${(score * 100).toInt()}%"
                effectivenessScore?.visibility = View.VISIBLE
            } ?: run {
                effectivenessIndicator?.visibility = View.GONE
                effectivenessScore?.visibility = View.GONE
            }
        }
    }
}