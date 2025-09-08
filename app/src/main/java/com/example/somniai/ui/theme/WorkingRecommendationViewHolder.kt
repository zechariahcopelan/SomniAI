package com.example.somniai.ui.theme

import android.view.View
import androidx.recyclerview.widget.RecyclerView
import com.example.somniai.R
import com.example.somniai.data.SleepRecommendation
import com.example.somniai.data.RecommendationType
import com.example.somniai.databinding.CardRecommendationBinding

/**
 * Simple, working ViewHolder for sleep recommendations
 */
class RecommendationViewHolder(
    private val binding: CardRecommendationBinding
) : RecyclerView.ViewHolder(binding.root) {

    fun bind(
        recommendation: SleepRecommendation,
        onItemClick: ((SleepRecommendation) -> Unit)? = null
    ) {
        // Set click listener
        binding.root.setOnClickListener { onItemClick?.invoke(recommendation) }

        // Set basic content - adjust these IDs to match your actual XML
        // You'll need to replace these with your actual view IDs from card_recommendation.xml

        // Example - replace with your actual IDs:
        // binding.titleText?.text = recommendation.title
        // binding.descriptionText?.text = recommendation.description
        // binding.typeIcon?.setImageResource(getIconForType(recommendation.type))
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
        }
    }
}