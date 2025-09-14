package com.example.somniai.ui.theme

import android.content.Context
import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.databinding.*
import com.example.somniai.utils.FormatUtils
import com.example.somniai.ai.AIPerformanceMonitor
import com.example.somniai.ai.AIOperationType
import kotlinx.coroutines.*
import kotlin.math.roundToInt

/**
 * Advanced Recommendations Adapter for SomniAI
 *
 * Enterprise-grade adapter supporting 12 distinct recommendation types with AI-driven
 * insights, confidence scoring, priority-based sorting, and comprehensive analytics integration.
 * Dynamically handles sleep quality, environmental, behavioral, and optimization recommendations
 * with sophisticated view binding and performance optimization.
 */
class RecommendationsAdapter(
    private val context: Context,
    private val onRecommendationClick: (ProcessedInsight) -> Unit = {},
    private val onRecommendationAction: (ProcessedInsight, RecommendationAction) -> Unit = { _, _ -> },
    private val onConfidenceInfoClick: (ProcessedInsight) -> Unit = {},
    private val showConfidenceScores: Boolean = true,
    private val showPriorityIndicators: Boolean = true,
    private val enableAnalytics: Boolean = true,
    private val groupByCategory: Boolean = false
) : ListAdapter<RecommendationItem, RecommendationsAdapter.BaseRecommendationViewHolder>(RecommendationDiffCallback()) {

    companion object {
        // View types for 12 different recommendation categories
        private const val VIEW_TYPE_SLEEP_QUALITY = 0
        private const val VIEW_TYPE_SLEEP_DURATION = 1
        private const val VIEW_TYPE_BEDTIME_CONSISTENCY = 2
        private const val VIEW_TYPE_ENVIRONMENT = 3
        private const val VIEW_TYPE_STRESS_MANAGEMENT = 4
        private const val VIEW_TYPE_EXERCISE = 5
        private const val VIEW_TYPE_NUTRITION = 6
        private const val VIEW_TYPE_SCREEN_TIME = 7
        private const val VIEW_TYPE_TEMPERATURE = 8
        private const val VIEW_TYPE_NOISE = 9
        private const val VIEW_TYPE_LIGHTING = 10
        private const val VIEW_TYPE_ROUTINE = 11
        private const val VIEW_TYPE_HEADER = 12
        private const val VIEW_TYPE_LOADING = 13
        private const val VIEW_TYPE_ERROR = 14

        // Confidence thresholds for visual indicators
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.8f
        private const val MEDIUM_CONFIDENCE_THRESHOLD = 0.6f
        private const val LOW_CONFIDENCE_THRESHOLD = 0.4f

        // Priority color mapping
        private const val HIGH_PRIORITY_COLOR = "#F44336" // Red
        private const val MEDIUM_PRIORITY_COLOR = "#FF9800" // Orange
        private const val LOW_PRIORITY_COLOR = "#4CAF50" // Green

        // Animation constants
        private const val CONFIDENCE_ANIMATION_DURATION = 300L
        private const val PRIORITY_PULSE_DURATION = 1000L
    }

    // Performance tracking
    private var lastInteractionTime = 0L
    private val analyticsScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // ========== BASE VIEW HOLDER ==========

    abstract class BaseRecommendationViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        abstract fun bind(item: RecommendationItem, adapter: RecommendationsAdapter)

        protected fun setupBaseRecommendationContent(
            insight: ProcessedInsight,
            titleView: android.widget.TextView,
            descriptionView: android.widget.TextView,
            iconView: android.widget.ImageView,
            adapter: RecommendationsAdapter
        ) {
            titleView.text = insight.title
            descriptionView.text = insight.description

            // Set category-specific icon
            iconView.setImageResource(getIconForCategory(insight.category))
            iconView.setColorFilter(getCategoryColor(insight.category))
        }

        protected fun setupConfidenceIndicator(
            insight: ProcessedInsight,
            confidenceView: android.widget.TextView?,
            confidenceBar: android.widget.ProgressBar?,
            adapter: RecommendationsAdapter
        ) {
            if (!adapter.showConfidenceScores) {
                confidenceView?.visibility = View.GONE
                confidenceBar?.visibility = View.GONE
                return
            }

            val confidence = insight.confidence
            confidenceView?.text = "${(confidence * 100).roundToInt()}%"
            confidenceBar?.progress = (confidence * 100).roundToInt()

            // Color coding based on confidence level
            val confidenceColor = when {
                confidence >= HIGH_CONFIDENCE_THRESHOLD -> Color.parseColor("#4CAF50")
                confidence >= MEDIUM_CONFIDENCE_THRESHOLD -> Color.parseColor("#FF9800")
                else -> Color.parseColor("#F44336")
            }

            confidenceView?.setTextColor(confidenceColor)
            confidenceBar?.progressTintList = android.content.res.ColorStateList.valueOf(confidenceColor)
        }

        protected fun setupPriorityIndicator(
            insight: ProcessedInsight,
            priorityView: android.widget.TextView?,
            priorityIndicator: View?,
            adapter: RecommendationsAdapter
        ) {
            if (!adapter.showPriorityIndicators) {
                priorityView?.visibility = View.GONE
                priorityIndicator?.visibility = View.GONE
                return
            }

            val priorityText = when (insight.priority) {
                1 -> "HIGH"
                2 -> "MEDIUM"
                3 -> "LOW"
                else -> "NORMAL"
            }

            val priorityColor = when (insight.priority) {
                1 -> Color.parseColor(HIGH_PRIORITY_COLOR)
                2 -> Color.parseColor(MEDIUM_PRIORITY_COLOR)
                3 -> Color.parseColor(LOW_PRIORITY_COLOR)
                else -> Color.parseColor("#757575")
            }

            priorityView?.text = priorityText
            priorityView?.setTextColor(priorityColor)
            priorityIndicator?.setBackgroundColor(priorityColor)

            // Add pulsing animation for high priority
            if (insight.priority == 1) {
                startPriorityPulseAnimation(priorityIndicator)
            }
        }

        protected fun setupClickListeners(
            insight: ProcessedInsight,
            rootView: View,
            actionButton: android.widget.Button?,
            adapter: RecommendationsAdapter
        ) {
            rootView.setOnClickListener {
                adapter.trackRecommendationInteraction(insight, "click")
                adapter.onRecommendationClick(insight)
            }

            actionButton?.setOnClickListener {
                adapter.trackRecommendationInteraction(insight, "action_button")
                adapter.onRecommendationAction(insight, determineRecommendationAction(insight))
            }
        }

        private fun startPriorityPulseAnimation(view: View?) {
            view?.let { v ->
                v.animate()
                    .alpha(0.3f)
                    .setDuration(PRIORITY_PULSE_DURATION / 2)
                    .withEndAction {
                        v.animate()
                            .alpha(1.0f)
                            .setDuration(PRIORITY_PULSE_DURATION / 2)
                            .withEndAction {
                                startPriorityPulseAnimation(v) // Repeat
                            }
                    }
            }
        }

        private fun determineRecommendationAction(insight: ProcessedInsight): RecommendationAction {
            return when (insight.category) {
                InsightCategory.SLEEP_QUALITY -> RecommendationAction.IMPROVE_QUALITY
                InsightCategory.DURATION -> RecommendationAction.ADJUST_DURATION
                InsightCategory.ENVIRONMENT -> RecommendationAction.OPTIMIZE_ENVIRONMENT
                InsightCategory.PATTERN -> RecommendationAction.ADJUST_SCHEDULE
                else -> RecommendationAction.GENERAL_IMPROVEMENT
            }
        }

        private fun getIconForCategory(category: InsightCategory): Int {
            return when (category) {
                InsightCategory.SLEEP_QUALITY, InsightCategory.DURATION -> R.drawable.ic_sleep
                InsightCategory.ENVIRONMENT -> R.drawable.ic_environment
                InsightCategory.PATTERN -> R.drawable.ic_routine
                InsightCategory.GENERAL -> R.drawable.ic_bedtime_consistency
                else -> R.drawable.ic_sleep
            }
        }

        private fun getCategoryColor(category: InsightCategory): Int {
            return when (category) {
                InsightCategory.SLEEP_QUALITY -> Color.parseColor("#2196F3")
                InsightCategory.DURATION -> Color.parseColor("#4CAF50")
                InsightCategory.ENVIRONMENT -> Color.parseColor("#FF9800")
                InsightCategory.PATTERN -> Color.parseColor("#9C27B0")
                InsightCategory.GENERAL -> Color.parseColor("#607D8B")
                else -> Color.parseColor("#757575")
            }
        }
    }

    // ========== SPECIFIC VIEW HOLDERS FOR EACH RECOMMENDATION TYPE ==========

    inner class SleepQualityViewHolder(
        private val binding: ItemRecommendationSleepQualityBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupConfidenceIndicator(insight, null, null, adapter) // No confidence views in basic layout
            setupPriorityIndicator(insight, null, null, adapter) // No priority views in basic layout
            setupClickListeners(insight, binding.root, null, adapter)

            // Sleep quality specific enhancements
            if (insight.qualityScore > 0) {
                binding.titleText.text = "${insight.title} (Quality: ${(insight.qualityScore * 10).roundToInt()}/10)"
            }
        }
    }

    inner class SleepDurationViewHolder(
        private val binding: ItemRecommendationSleepDurationBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Duration-specific enhancements
            binding.descriptionText.text = "${insight.description}\n\nRecommendation: ${insight.recommendation}"
        }
    }

    inner class BedtimeConsistencyViewHolder(
        private val binding: ItemRecommendationBedtimeConsistencyBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Consistency-specific styling
            if (insight.priority == 1) {
                binding.root.setBackgroundResource(R.drawable.card_background) // Highlighted background for high priority
            }
        }
    }

    inner class EnvironmentViewHolder(
        private val binding: ItemRecommendationEnvironmentBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Environment-specific icon tinting
            binding.iconImage.setColorFilter(Color.parseColor("#FF9800")) // Orange for environment
        }
    }

    inner class StressManagementViewHolder(
        private val binding: ItemRecommendationStressManagementBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Stress management specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#9C27B0")) // Purple for stress management
        }
    }

    inner class ExerciseViewHolder(
        private val binding: ItemRecommendationExerciseBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Exercise-specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#4CAF50")) // Green for exercise
        }
    }

    inner class NutritionViewHolder(
        private val binding: ItemRecommendationNutritionBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Nutrition-specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#FF5722")) // Deep orange for nutrition
        }
    }

    inner class ScreenTimeViewHolder(
        private val binding: ItemRecommendationScreenTimeBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Screen time specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#2196F3")) // Blue for screen time
        }
    }

    inner class TemperatureViewHolder(
        private val binding: ItemRecommendationTemperatureBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Temperature-specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#FF9800")) // Orange for temperature
        }
    }

    inner class NoiseViewHolder(
        private val binding: ItemRecommendationNoiseBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Noise-specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#795548")) // Brown for noise
        }
    }

    inner class LightingViewHolder(
        private val binding: ItemRecommendationLightingBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Lighting-specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#FFC107")) // Amber for lighting
        }
    }

    inner class RoutineViewHolder(
        private val binding: ItemRecommendationRoutineBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item !is RecommendationItem.Recommendation) return

            val insight = item.insight
            setupBaseRecommendationContent(insight, binding.titleText, binding.descriptionText, binding.iconImage, adapter)
            setupClickListeners(insight, binding.root, null, adapter)

            // Routine-specific styling
            binding.iconImage.setColorFilter(Color.parseColor("#673AB7")) // Deep purple for routine
        }
    }

    // ========== UTILITY VIEW HOLDERS ==========

    inner class HeaderViewHolder(itemView: View) : BaseRecommendationViewHolder(itemView) {
        private val headerText: android.widget.TextView = itemView.findViewById(android.R.id.text1)

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item is RecommendationItem.Header) {
                headerText.text = item.title
                headerText.setTextColor(Color.parseColor("#333333"))
                headerText.textSize = 16f
                headerText.typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
        }
    }

    inner class LoadingViewHolder(itemView: View) : BaseRecommendationViewHolder(itemView) {
        private val loadingText: android.widget.TextView = itemView.findViewById(android.R.id.text1)

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item is RecommendationItem.Loading) {
                loadingText.text = item.message
                loadingText.setTextColor(Color.parseColor("#757575"))
            }
        }
    }

    inner class ErrorViewHolder(itemView: View) : BaseRecommendationViewHolder(itemView) {
        private val errorText: android.widget.TextView = itemView.findViewById(android.R.id.text1)

        override fun bind(item: RecommendationItem, adapter: RecommendationsAdapter) {
            if (item is RecommendationItem.Error) {
                errorText.text = item.message
                errorText.setTextColor(Color.parseColor("#F44336"))
            }
        }
    }

    // ========== ADAPTER IMPLEMENTATION ==========

    override fun getItemViewType(position: Int): Int {
        return when (val item = getItem(position)) {
            is RecommendationItem.Recommendation -> {
                getViewTypeForRecommendationType(item.insight.getRecommendationType())
            }
            is RecommendationItem.Header -> VIEW_TYPE_HEADER
            is RecommendationItem.Loading -> VIEW_TYPE_LOADING
            is RecommendationItem.Error -> VIEW_TYPE_ERROR
        }
    }

    private fun getViewTypeForRecommendationType(type: RecommendationType): Int {
        return when (type) {
            RecommendationType.SLEEP_QUALITY -> VIEW_TYPE_SLEEP_QUALITY
            RecommendationType.SLEEP_DURATION -> VIEW_TYPE_SLEEP_DURATION
            RecommendationType.BEDTIME_CONSISTENCY -> VIEW_TYPE_BEDTIME_CONSISTENCY
            RecommendationType.ENVIRONMENT -> VIEW_TYPE_ENVIRONMENT
            RecommendationType.STRESS_MANAGEMENT -> VIEW_TYPE_STRESS_MANAGEMENT
            RecommendationType.EXERCISE -> VIEW_TYPE_EXERCISE
            RecommendationType.NUTRITION -> VIEW_TYPE_NUTRITION
            RecommendationType.SCREEN_TIME -> VIEW_TYPE_SCREEN_TIME
            RecommendationType.TEMPERATURE -> VIEW_TYPE_TEMPERATURE
            RecommendationType.NOISE -> VIEW_TYPE_NOISE
            RecommendationType.LIGHTING -> VIEW_TYPE_LIGHTING
            RecommendationType.ROUTINE -> VIEW_TYPE_ROUTINE
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): BaseRecommendationViewHolder {
        val inflater = LayoutInflater.from(parent.context)

        return when (viewType) {
            VIEW_TYPE_SLEEP_QUALITY -> {
                val binding = ItemRecommendationSleepQualityBinding.inflate(inflater, parent, false)
                SleepQualityViewHolder(binding)
            }
            VIEW_TYPE_SLEEP_DURATION -> {
                val binding = ItemRecommendationSleepDurationBinding.inflate(inflater, parent, false)
                SleepDurationViewHolder(binding)
            }
            VIEW_TYPE_BEDTIME_CONSISTENCY -> {
                val binding = ItemRecommendationBedtimeConsistencyBinding.inflate(inflater, parent, false)
                BedtimeConsistencyViewHolder(binding)
            }
            VIEW_TYPE_ENVIRONMENT -> {
                val binding = ItemRecommendationEnvironmentBinding.inflate(inflater, parent, false)
                EnvironmentViewHolder(binding)
            }
            VIEW_TYPE_STRESS_MANAGEMENT -> {
                val binding = ItemRecommendationStressManagementBinding.inflate(inflater, parent, false)
                StressManagementViewHolder(binding)
            }
            VIEW_TYPE_EXERCISE -> {
                val binding = ItemRecommendationExerciseBinding.inflate(inflater, parent, false)
                ExerciseViewHolder(binding)
            }
            VIEW_TYPE_NUTRITION -> {
                val binding = ItemRecommendationNutritionBinding.inflate(inflater, parent, false)
                NutritionViewHolder(binding)
            }
            VIEW_TYPE_SCREEN_TIME -> {
                val binding = ItemRecommendationScreenTimeBinding.inflate(inflater, parent, false)
                ScreenTimeViewHolder(binding)
            }
            VIEW_TYPE_TEMPERATURE -> {
                val binding = ItemRecommendationTemperatureBinding.inflate(inflater, parent, false)
                TemperatureViewHolder(binding)
            }
            VIEW_TYPE_NOISE -> {
                val binding = ItemRecommendationNoiseBinding.inflate(inflater, parent, false)
                NoiseViewHolder(binding)
            }
            VIEW_TYPE_LIGHTING -> {
                val binding = ItemRecommendationLightingBinding.inflate(inflater, parent, false)
                LightingViewHolder(binding)
            }
            VIEW_TYPE_ROUTINE -> {
                val binding = ItemRecommendationRoutineBinding.inflate(inflater, parent, false)
                RoutineViewHolder(binding)
            }
            VIEW_TYPE_HEADER -> {
                val view = inflater.inflate(android.R.layout.simple_list_item_1, parent, false)
                HeaderViewHolder(view)
            }
            VIEW_TYPE_LOADING -> {
                val view = inflater.inflate(android.R.layout.simple_list_item_1, parent, false)
                LoadingViewHolder(view)
            }
            VIEW_TYPE_ERROR -> {
                val view = inflater.inflate(android.R.layout.simple_list_item_1, parent, false)
                ErrorViewHolder(view)
            }
            else -> throw IllegalArgumentException("Unknown view type: $viewType")
        }
    }

    override fun onBindViewHolder(holder: BaseRecommendationViewHolder, position: Int) {
        holder.bind(getItem(position), this)
    }

    // ========== ANALYTICS AND PERFORMANCE TRACKING ==========

    private fun trackRecommendationInteraction(insight: ProcessedInsight, interactionType: String) {
        val currentTime = System.currentTimeMillis()

        // Prevent rapid-fire interactions
        if (currentTime - lastInteractionTime < 300) return
        lastInteractionTime = currentTime

        if (enableAnalytics) {
            analyticsScope.launch {
                try {
                    AIPerformanceMonitor.recordOperation(
                        operationId = "recommendation_${insight.id}_${interactionType}",
                        operationType = AIOperationType.INSIGHT_GENERATION,
                        model = com.example.somniai.ai.AIModel.NATIVE_PROCESSING,
                        processingTimeMs = 1L,
                        tokenUsage = com.example.somniai.ai.TokenUsage(0, 0, 0),
                        success = true,
                        confidenceScore = insight.confidence,
                        metadata = mapOf(
                            "recommendation_type" to insight.getRecommendationType().name,
                            "interaction_type" to interactionType,
                            "priority" to insight.priority.toString(),
                            "category" to insight.category.name
                        )
                    )
                } catch (e: Exception) {
                    // Silently handle analytics errors
                }
            }
        }
    }

    // ========== PUBLIC METHODS ==========

    /**
     * Update recommendations with proper sorting and grouping
     */
    fun updateRecommendations(
        insights: List<ProcessedInsight>,
        includeHeaders: Boolean = groupByCategory
    ) {
        val items = mutableListOf<RecommendationItem>()

        if (includeHeaders && groupByCategory) {
            // Group by category with headers
            val groupedInsights = insights.groupBy { it.category }

            groupedInsights.forEach { (category, categoryInsights) ->
                items.add(RecommendationItem.Header(category.getDisplayName()))

                categoryInsights
                    .sortedWith(compareBy<ProcessedInsight> { it.priority }.thenByDescending { it.confidence })
                    .forEach { insight ->
                        items.add(RecommendationItem.Recommendation(insight))
                    }
            }
        } else {
            // Simple list with priority sorting
            insights
                .sortedWith(compareBy<ProcessedInsight> { it.priority }.thenByDescending { it.confidence })
                .forEach { insight ->
                    items.add(RecommendationItem.Recommendation(insight))
                }
        }

        submitList(items)
    }

    /**
     * Show loading state
     */
    fun showLoading(message: String = "Generating recommendations...") {
        submitList(listOf(RecommendationItem.Loading(message)))
    }

    /**
     * Show error state
     */
    fun showError(message: String, throwable: Throwable? = null) {
        submitList(listOf(RecommendationItem.Error(message, throwable)))
    }

    /**
     * Get recommendations by type
     */
    fun getRecommendationsByType(type: RecommendationType): List<ProcessedInsight> {
        return currentList.filterIsInstance<RecommendationItem.Recommendation>()
            .map { it.insight }
            .filter { it.getRecommendationType() == type }
    }

    /**
     * Get high priority recommendations
     */
    fun getHighPriorityRecommendations(): List<ProcessedInsight> {
        return currentList.filterIsInstance<RecommendationItem.Recommendation>()
            .map { it.insight }
            .filter { it.priority == 1 }
    }

    fun onDestroy() {
        analyticsScope.cancel()
    }
}

// ========== EXTENSION FUNCTIONS ==========

private fun ProcessedInsight.getRecommendationType(): RecommendationType {
    return when {
        title.contains("quality", ignoreCase = true) -> RecommendationType.SLEEP_QUALITY
        title.contains("duration", ignoreCase = true) -> RecommendationType.SLEEP_DURATION
        title.contains("bedtime", ignoreCase = true) || title.contains("consistency", ignoreCase = true) -> RecommendationType.BEDTIME_CONSISTENCY
        title.contains("environment", ignoreCase = true) -> RecommendationType.ENVIRONMENT
        title.contains("stress", ignoreCase = true) -> RecommendationType.STRESS_MANAGEMENT
        title.contains("exercise", ignoreCase = true) -> RecommendationType.EXERCISE
        title.contains("nutrition", ignoreCase = true) || title.contains("diet", ignoreCase = true) -> RecommendationType.NUTRITION
        title.contains("screen", ignoreCase = true) -> RecommendationType.SCREEN_TIME
        title.contains("temperature", ignoreCase = true) -> RecommendationType.TEMPERATURE
        title.contains("noise", ignoreCase = true) -> RecommendationType.NOISE
        title.contains("light", ignoreCase = true) -> RecommendationType.LIGHTING
        title.contains("routine", ignoreCase = true) -> RecommendationType.ROUTINE
        else -> RecommendationType.SLEEP_QUALITY // Default
    }
}

private fun InsightCategory.getDisplayName(): String {
    return when (this) {
        InsightCategory.SLEEP_QUALITY -> "Sleep Quality"
        InsightCategory.DURATION -> "Sleep Duration"
        InsightCategory.ENVIRONMENT -> "Environment"
        InsightCategory.PATTERN -> "Sleep Patterns"
        InsightCategory.GENERAL -> "General"
        else -> "Recommendations"
    }
}

// ========== DIFF CALLBACK ==========

class RecommendationDiffCallback : DiffUtil.ItemCallback<RecommendationItem>() {
    override fun areItemsTheSame(oldItem: RecommendationItem, newItem: RecommendationItem): Boolean {
        return when {
            oldItem is RecommendationItem.Recommendation && newItem is RecommendationItem.Recommendation ->
                oldItem.insight.id == newItem.insight.id
            oldItem is RecommendationItem.Header && newItem is RecommendationItem.Header ->
                oldItem.title == newItem.title
            oldItem is RecommendationItem.Loading && newItem is RecommendationItem.Loading -> true
            oldItem is RecommendationItem.Error && newItem is RecommendationItem.Error -> true
            else -> false
        }
    }

    override fun areContentsTheSame(oldItem: RecommendationItem, newItem: RecommendationItem): Boolean {
        return when {
            oldItem is RecommendationItem.Recommendation && newItem is RecommendationItem.Recommendation ->
                oldItem.insight.title == newItem.insight.title &&
                        oldItem.insight.description == newItem.insight.description &&
                        oldItem.insight.confidence == newItem.insight.confidence &&
                        oldItem.insight.priority == newItem.insight.priority
            else -> oldItem == newItem
        }
    }
}

// ========== SUPPORTING CLASSES AND ENUMS ==========

sealed class RecommendationItem {
    data class Recommendation(val insight: ProcessedInsight) : RecommendationItem()
    data class Header(val title: String) : RecommendationItem()
    data class Loading(val message: String) : RecommendationItem()
    data class Error(val message: String, val throwable: Throwable? = null) : RecommendationItem()
}

enum class RecommendationType {
    SLEEP_QUALITY,
    SLEEP_DURATION,
    BEDTIME_CONSISTENCY,
    ENVIRONMENT,
    STRESS_MANAGEMENT,
    EXERCISE,
    NUTRITION,
    SCREEN_TIME,
    TEMPERATURE,
    NOISE,
    LIGHTING,
    ROUTINE
}

enum class RecommendationAction {
    IMPROVE_QUALITY,
    ADJUST_DURATION,
    OPTIMIZE_ENVIRONMENT,
    ADJUST_SCHEDULE,
    GENERAL_IMPROVEMENT,
    TRACK_PROGRESS,
    SET_REMINDER,
    GET_MORE_INFO
}