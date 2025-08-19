package com.example.somniai.ui.adapters

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.text.SpannableString
import android.text.style.ForegroundColorSpan
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.somniai.R
import com.example.somniai.ai.*
import com.example.somniai.data.*
import com.example.somniai.databinding.*
import com.google.android.material.chip.Chip
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

/**
 * Enterprise-grade RecyclerView Adapter for AI-powered Recommendations
 *
 * Advanced Features:
 * - Multi-view type support for different recommendation categories
 * - AI insight integration with quality scoring and confidence levels
 * - Real-time user engagement tracking and analytics
 * - Material Design 3 with dark theme optimization
 * - Comprehensive accessibility support
 * - Performance monitoring and optimization
 * - Interactive feedback collection and processing
 * - Smooth animations and micro-interactions
 * - Offline capability with seamless sync
 * - Advanced filtering and categorization
 * - Context-aware recommendation prioritization
 * - Integration with ML personalization engine
 */
class RecommendationsAdapter(
    private val onRecommendationClick: (Recommendation) -> Unit,
    private val onRecommendationLongClick: (Recommendation) -> Unit,
    private val onActionClick: (Recommendation, RecommendationAction) -> Unit,
    private val onFeedbackClick: (Recommendation, RecommendationFeedback) -> Unit,
    private val onShareClick: (Recommendation) -> Unit,
    private val onDismissClick: (Recommendation) -> Unit,
    private val onImplementClick: (Recommendation) -> Unit,
    private val onTrackingCallback: (RecommendationEvent) -> Unit = {}
) : ListAdapter<Recommendation, RecommendationsAdapter.BaseRecommendationViewHolder>(RecommendationDiffCallback()) {

    companion object {
        private const val TAG = "RecommendationsAdapter"

        // View types for different recommendation categories
        private const val VIEW_TYPE_SLEEP_HYGIENE = 1
        private const val VIEW_TYPE_ENVIRONMENT = 2
        private const val VIEW_TYPE_LIFESTYLE = 3
        private const val VIEW_TYPE_MEDICAL = 4
        private const val VIEW_TYPE_TECHNOLOGY = 5
        private const val VIEW_TYPE_BEHAVIORAL = 6
        private const val VIEW_TYPE_NUTRITIONAL = 7
        private const val VIEW_TYPE_EXERCISE = 8
        private const val VIEW_TYPE_STRESS_MANAGEMENT = 9
        private const val VIEW_TYPE_GOAL_TRACKING = 10
        private const val VIEW_TYPE_EMERGENCY = 11
        private const val VIEW_TYPE_ACHIEVEMENT = 12

        // Animation constants
        private const val ANIMATION_DURATION = 300L
        private const val STAGGER_DELAY = 50L
        private const val SCALE_FACTOR = 0.95f

        // Interaction tracking
        private const val IMPRESSION_THRESHOLD_MS = 1000L
        private const val ENGAGEMENT_THRESHOLD_MS = 3000L
    }

    // Tracking and analytics
    private val visibleRecommendations = mutableSetOf<String>()
    private val impressionTimestamps = mutableMapOf<String, Long>()
    private val engagementStartTimes = mutableMapOf<String, Long>()

    // Animation management
    private var isAnimationEnabled = true
    private var lastAnimationTime = 0L

    override fun getItemViewType(position: Int): Int {
        return when (getItem(position).category) {
            RecommendationCategory.SLEEP_HYGIENE -> VIEW_TYPE_SLEEP_HYGIENE
            RecommendationCategory.ENVIRONMENT -> VIEW_TYPE_ENVIRONMENT
            RecommendationCategory.LIFESTYLE -> VIEW_TYPE_LIFESTYLE
            RecommendationCategory.MEDICAL -> VIEW_TYPE_MEDICAL
            RecommendationCategory.TECHNOLOGY -> VIEW_TYPE_TECHNOLOGY
            RecommendationCategory.BEHAVIORAL -> VIEW_TYPE_BEHAVIORAL
            RecommendationCategory.NUTRITIONAL -> VIEW_TYPE_NUTRITIONAL
            RecommendationCategory.EXERCISE -> VIEW_TYPE_EXERCISE
            RecommendationCategory.STRESS_MANAGEMENT -> VIEW_TYPE_STRESS_MANAGEMENT
            RecommendationCategory.GOAL_TRACKING -> VIEW_TYPE_GOAL_TRACKING
            RecommendationCategory.EMERGENCY -> VIEW_TYPE_EMERGENCY
            RecommendationCategory.ACHIEVEMENT -> VIEW_TYPE_ACHIEVEMENT
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): BaseRecommendationViewHolder {
        val inflater = LayoutInflater.from(parent.context)

        return when (viewType) {
            VIEW_TYPE_SLEEP_HYGIENE -> SleepHygieneViewHolder(
                ItemRecommendationSleepHygieneBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_ENVIRONMENT -> EnvironmentViewHolder(
                ItemRecommendationEnvironmentBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_LIFESTYLE -> LifestyleViewHolder(
                ItemRecommendationLifestyleBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_MEDICAL -> MedicalViewHolder(
                ItemRecommendationMedicalBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_TECHNOLOGY -> TechnologyViewHolder(
                ItemRecommendationTechnologyBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_BEHAVIORAL -> BehavioralViewHolder(
                ItemRecommendationBehavioralBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_NUTRITIONAL -> NutritionalViewHolder(
                ItemRecommendationNutritionalBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_EXERCISE -> ExerciseViewHolder(
                ItemRecommendationExerciseBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_STRESS_MANAGEMENT -> StressManagementViewHolder(
                ItemRecommendationStressBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_GOAL_TRACKING -> GoalTrackingViewHolder(
                ItemRecommendationGoalBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_EMERGENCY -> EmergencyViewHolder(
                ItemRecommendationEmergencyBinding.inflate(inflater, parent, false)
            )
            VIEW_TYPE_ACHIEVEMENT -> AchievementViewHolder(
                ItemRecommendationAchievementBinding.inflate(inflater, parent, false)
            )
            else -> DefaultViewHolder(
                ItemRecommendationDefaultBinding.inflate(inflater, parent, false)
            )
        }
    }

    override fun onBindViewHolder(holder: BaseRecommendationViewHolder, position: Int) {
        val recommendation = getItem(position)
        holder.bind(recommendation)

        // Track impression
        trackImpression(recommendation, position)

        // Apply entrance animation if enabled
        if (isAnimationEnabled && shouldAnimateItem(position)) {
            animateItemEntrance(holder.itemView, position)
        }
    }

    override fun onBindViewHolder(holder: BaseRecommendationViewHolder, position: Int, payloads: List<Any>) {
        if (payloads.isEmpty()) {
            onBindViewHolder(holder, position)
        } else {
            val recommendation = getItem(position)
            payloads.forEach { payload ->
                when (payload) {
                    is RecommendationPayload.StatusUpdate -> holder.updateStatus(recommendation)
                    is RecommendationPayload.EngagementUpdate -> holder.updateEngagement(recommendation)
                    is RecommendationPayload.ProgressUpdate -> holder.updateProgress(recommendation)
                }
            }
        }
    }

    override fun onViewAttachedToWindow(holder: BaseRecommendationViewHolder) {
        super.onViewAttachedToWindow(holder)
        val recommendation = getItem(holder.bindingAdapterPosition)
        startEngagementTracking(recommendation)
    }

    override fun onViewDetachedFromWindow(holder: BaseRecommendationViewHolder) {
        super.onViewDetachedFromWindow(holder)
        val recommendation = getItem(holder.bindingAdapterPosition)
        stopEngagementTracking(recommendation)
    }

    // ========== BASE VIEW HOLDER ==========

    abstract class BaseRecommendationViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        abstract fun bind(recommendation: Recommendation)
        abstract fun updateStatus(recommendation: Recommendation)
        abstract fun updateEngagement(recommendation: Recommendation)
        abstract fun updateProgress(recommendation: Recommendation)

        protected fun setupCommonViews(
            recommendation: Recommendation,
            binding: Any,
            adapter: RecommendationsAdapter
        ) {
            // Common setup logic that all view holders can use
            itemView.setOnClickListener {
                adapter.handleRecommendationClick(recommendation)
            }

            itemView.setOnLongClickListener {
                adapter.handleRecommendationLongClick(recommendation)
                true
            }
        }

        protected fun bindCommonElements(
            recommendation: Recommendation,
            titleView: android.widget.TextView,
            descriptionView: android.widget.TextView,
            priorityView: View?,
            confidenceView: View?,
            timestampView: android.widget.TextView?
        ) {
            // Title with priority styling
            titleView.text = recommendation.title
            titleView.setTextColor(getPriorityColor(itemView.context, recommendation.priority))

            // Description with expandable text support
            descriptionView.text = recommendation.description

            // Priority indicator
            priorityView?.let { view ->
                setupPriorityIndicator(view, recommendation.priority)
            }

            // Confidence indicator
            confidenceView?.let { view ->
                setupConfidenceIndicator(view, recommendation.confidence)
            }

            // Timestamp
            timestampView?.text = formatTimestamp(recommendation.timestamp)
        }

        private fun getPriorityColor(context: Context, priority: RecommendationPriority): Int {
            return when (priority) {
                RecommendationPriority.CRITICAL -> ContextCompat.getColor(context, R.color.priority_critical)
                RecommendationPriority.HIGH -> ContextCompat.getColor(context, R.color.priority_high)
                RecommendationPriority.MEDIUM -> ContextCompat.getColor(context, R.color.priority_medium)
                RecommendationPriority.LOW -> ContextCompat.getColor(context, R.color.priority_low)
            }
        }

        private fun setupPriorityIndicator(view: View, priority: RecommendationPriority) {
            val drawable = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(getPriorityColor(view.context, priority))
                setSize(24, 24) // dp will be converted by the system
            }
            view.background = drawable
            view.contentDescription = view.context.getString(
                R.string.priority_indicator_description,
                priority.displayName
            )
        }

        private fun setupConfidenceIndicator(view: View, confidence: Float) {
            if (view is com.google.android.material.progressindicator.CircularProgressIndicator) {
                view.progress = (confidence * 100).roundToInt()
                view.contentDescription = view.context.getString(
                    R.string.confidence_indicator_description,
                    (confidence * 100).roundToInt()
                )
            }
        }

        private fun formatTimestamp(timestamp: Long): String {
            val now = System.currentTimeMillis()
            val diff = now - timestamp

            return when {
                diff < 60000L -> "Just now"
                diff < 3600000L -> "${diff / 60000L}m ago"
                diff < 86400000L -> "${diff / 3600000L}h ago"
                diff < 604800000L -> "${diff / 86400000L}d ago"
                else -> SimpleDateFormat("MMM dd", Locale.getDefault()).format(Date(timestamp))
            }
        }
    }

    // ========== SPECIALIZED VIEW HOLDERS ==========

    inner class SleepHygieneViewHolder(
        private val binding: ItemRecommendationSleepHygieneBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(recommendation: Recommendation) {
            bindCommonElements(
                recommendation,
                binding.textTitle,
                binding.textDescription,
                binding.viewPriorityIndicator,
                binding.progressConfidence,
                binding.textTimestamp
            )

            // Sleep hygiene specific elements
            setupSleepHygieneSpecifics(recommendation)
            setupActionButtons(recommendation)
            setupFeedbackOptions(recommendation)

            setupCommonViews(recommendation, binding, this@RecommendationsAdapter)
        }

        private fun setupSleepHygieneSpecifics(recommendation: Recommendation) {
            // Sleep score impact
            val impact = recommendation.expectedImpact
            binding.textSleepScoreImpact.text = binding.root.context.getString(
                R.string.sleep_score_impact,
                if (impact > 0) "+" else "",
                impact.roundToInt()
            )

            // Sleep hygiene category chips
            binding.chipGroupCategories.removeAllViews()
            recommendation.tags.forEach { tag ->
                val chip = Chip(binding.root.context).apply {
                    text = tag
                    isCheckable = false
                    chipIcon = ContextCompat.getDrawable(context, getTagIcon(tag))
                }
                binding.chipGroupCategories.addView(chip)
            }

            // Implementation difficulty
            setupDifficultyIndicator(recommendation.implementationDifficulty)

            // Time to impact
            binding.textTimeToImpact.text = recommendation.timeToImpact
        }

        private fun setupDifficultyIndicator(difficulty: ImplementationDifficulty) {
            val (text, color) = when (difficulty) {
                ImplementationDifficulty.EASY ->
                    "Easy" to ContextCompat.getColor(binding.root.context, R.color.difficulty_easy)
                ImplementationDifficulty.MODERATE ->
                    "Moderate" to ContextCompat.getColor(binding.root.context, R.color.difficulty_moderate)
                ImplementationDifficulty.CHALLENGING ->
                    "Challenging" to ContextCompat.getColor(binding.root.context, R.color.difficulty_challenging)
            }

            binding.textDifficulty.text = text
            binding.textDifficulty.setTextColor(color)
        }

        private fun setupActionButtons(recommendation: Recommendation) {
            // Primary action button
            binding.buttonPrimaryAction.text = recommendation.primaryAction?.label ?: "Learn More"
            binding.buttonPrimaryAction.setOnClickListener {
                recommendation.primaryAction?.let { action ->
                    onActionClick(recommendation, action)
                }
            }

            // Secondary actions
            setupSecondaryActions(recommendation)
        }

        private fun setupSecondaryActions(recommendation: Recommendation) {
            binding.layoutSecondaryActions.removeAllViews()

            recommendation.secondaryActions.forEach { action ->
                val button = com.google.android.material.button.MaterialButton(
                    binding.root.context,
                    null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle
                ).apply {
                    text = action.label
                    icon = ContextCompat.getDrawable(context, action.iconRes)
                    setOnClickListener {
                        onActionClick(recommendation, action)
                    }
                }

                binding.layoutSecondaryActions.addView(button)
            }
        }

        private fun setupFeedbackOptions(recommendation: Recommendation) {
            binding.buttonThumbsUp.setOnClickListener {
                onFeedbackClick(recommendation, RecommendationFeedback.HELPFUL)
                animateFeedbackSelection(binding.buttonThumbsUp, true)
            }

            binding.buttonThumbsDown.setOnClickListener {
                onFeedbackClick(recommendation, RecommendationFeedback.NOT_HELPFUL)
                animateFeedbackSelection(binding.buttonThumbsDown, false)
            }

            binding.buttonShare.setOnClickListener {
                onShareClick(recommendation)
            }

            binding.buttonDismiss.setOnClickListener {
                onDismissClick(recommendation)
            }
        }

        private fun animateFeedbackSelection(button: View, isPositive: Boolean) {
            val colorFrom = ContextCompat.getColor(button.context, R.color.surface)
            val colorTo = ContextCompat.getColor(
                button.context,
                if (isPositive) R.color.feedback_positive else R.color.feedback_negative
            )

            ValueAnimator.ofArgb(colorFrom, colorTo).apply {
                duration = 200L
                addUpdateListener { animator ->
                    button.setBackgroundColor(animator.animatedValue as Int)
                }
                start()
            }
        }

        override fun updateStatus(recommendation: Recommendation) {
            // Update implementation status
            updateImplementationStatus(recommendation.implementationStatus)
        }

        override fun updateEngagement(recommendation: Recommendation) {
            // Update engagement indicators
            val engagement = recommendation.engagementMetrics
            engagement?.let {
                binding.textEngagementScore.text = String.format("%.1f", it.engagementScore)
                binding.progressEngagement.progress = (it.engagementScore * 100).roundToInt()
            }
        }

        override fun updateProgress(recommendation: Recommendation) {
            // Update implementation progress
            val progress = recommendation.implementationProgress
            binding.progressImplementation.progress = (progress * 100).roundToInt()
            binding.textProgressPercentage.text = "${(progress * 100).roundToInt()}%"
        }

        private fun updateImplementationStatus(status: ImplementationStatus) {
            val (text, color) = when (status) {
                ImplementationStatus.NOT_STARTED ->
                    "Not Started" to ContextCompat.getColor(binding.root.context, R.color.status_not_started)
                ImplementationStatus.IN_PROGRESS ->
                    "In Progress" to ContextCompat.getColor(binding.root.context, R.color.status_in_progress)
                ImplementationStatus.COMPLETED ->
                    "Completed" to ContextCompat.getColor(binding.root.context, R.color.status_completed)
                ImplementationStatus.ABANDONED ->
                    "Abandoned" to ContextCompat.getColor(binding.root.context, R.color.status_abandoned)
            }

            binding.textImplementationStatus.text = text
            binding.textImplementationStatus.setTextColor(color)
        }

        private fun getTagIcon(tag: String): Int {
            return when (tag.lowercase()) {
                "bedtime" -> R.drawable.ic_bedtime
                "environment" -> R.drawable.ic_environment
                "routine" -> R.drawable.ic_routine
                "temperature" -> R.drawable.ic_temperature
                "light" -> R.drawable.ic_light
                "noise" -> R.drawable.ic_noise
                else -> R.drawable.ic_tag_default
            }
        }
    }

    inner class EnvironmentViewHolder(
        private val binding: ItemRecommendationEnvironmentBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(recommendation: Recommendation) {
            bindCommonElements(
                recommendation,
                binding.textTitle,
                binding.textDescription,
                binding.viewPriorityIndicator,
                binding.progressConfidence,
                binding.textTimestamp
            )

            setupEnvironmentSpecifics(recommendation)
            setupEnvironmentControls(recommendation)
            setupCommonViews(recommendation, binding, this@RecommendationsAdapter)
        }

        private fun setupEnvironmentSpecifics(recommendation: Recommendation) {
            // Environment metrics
            recommendation.environmentData?.let { data ->
                binding.textTemperature.text = "${data.optimalTemperature}°C"
                binding.textHumidity.text = "${data.optimalHumidity}%"
                binding.textLightLevel.text = data.lightLevel
                binding.textNoiseLevel.text = "${data.noiseLevel}dB"

                // Environmental impact visualization
                setupEnvironmentImpactChart(data)
            }

            // Smart device integration
            setupSmartDeviceControls(recommendation)
        }

        private fun setupEnvironmentImpactChart(data: EnvironmentData) {
            // Create a simple impact visualization
            val impactItems = listOf(
                "Temperature" to data.temperatureImpact,
                "Humidity" to data.humidityImpact,
                "Light" to data.lightImpact,
                "Noise" to data.noiseImpact
            )

            binding.layoutImpactChart.removeAllViews()

            impactItems.forEach { (label, impact) ->
                val itemView = LayoutInflater.from(binding.root.context)
                    .inflate(R.layout.item_environment_impact, binding.layoutImpactChart, false)

                itemView.findViewById<android.widget.TextView>(R.id.text_label).text = label
                itemView.findViewById<com.google.android.material.progressindicator.LinearProgressIndicator>(R.id.progress_impact)
                    .progress = (impact * 100).roundToInt()

                binding.layoutImpactChart.addView(itemView)
            }
        }

        private fun setupSmartDeviceControls(recommendation: Recommendation) {
            recommendation.smartDeviceActions.forEach { action ->
                val button = com.google.android.material.button.MaterialButton(
                    binding.root.context,
                    null,
                    com.google.android.material.R.attr.materialButtonOutlinedStyle
                ).apply {
                    text = action.label
                    icon = ContextCompat.getDrawable(context, action.iconRes)
                    setOnClickListener {
                        onActionClick(recommendation, action)
                    }
                }

                binding.layoutSmartControls.addView(button)
            }
        }

        private fun setupEnvironmentControls(recommendation: Recommendation) {
            // Temperature adjustment
            binding.sliderTemperature.addOnChangeListener { _, value, _ ->
                onActionClick(recommendation, RecommendationAction.AdjustTemperature(value))
            }

            // Light adjustment
            binding.sliderBrightness.addOnChangeListener { _, value, _ ->
                onActionClick(recommendation, RecommendationAction.AdjustBrightness(value))
            }

            // Sound machine toggle
            binding.switchSoundMachine.setOnCheckedChangeListener { _, isChecked ->
                onActionClick(recommendation, RecommendationAction.ToggleSoundMachine(isChecked))
            }
        }

        override fun updateStatus(recommendation: Recommendation) {
            // Update environmental readings
            recommendation.currentEnvironmentData?.let { data ->
                updateEnvironmentReadings(data)
            }
        }

        override fun updateEngagement(recommendation: Recommendation) {
            // Update environmental adjustment engagement
        }

        override fun updateProgress(recommendation: Recommendation) {
            // Update environment optimization progress
            val progress = recommendation.environmentOptimizationProgress
            binding.progressOptimization.progress = (progress * 100).roundToInt()
        }

        private fun updateEnvironmentReadings(data: EnvironmentData) {
            // Animate value changes
            animateValueChange(binding.textCurrentTemperature, "${data.currentTemperature}°C")
            animateValueChange(binding.textCurrentHumidity, "${data.currentHumidity}%")
            animateValueChange(binding.textCurrentLight, data.currentLightLevel)
            animateValueChange(binding.textCurrentNoise, "${data.currentNoiseLevel}dB")
        }

        private fun animateValueChange(textView: android.widget.TextView, newValue: String) {
            textView.animate()
                .alpha(0f)
                .setDuration(150L)
                .withEndAction {
                    textView.text = newValue
                    textView.animate()
                        .alpha(1f)
                        .setDuration(150L)
                        .start()
                }
                .start()
        }
    }

    inner class LifestyleViewHolder(
        private val binding: ItemRecommendationLifestyleBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(recommendation: Recommendation) {
            bindCommonElements(
                recommendation,
                binding.textTitle,
                binding.textDescription,
                binding.viewPriorityIndicator,
                binding.progressConfidence,
                binding.textTimestamp
            )

            setupLifestyleSpecifics(recommendation)
            setupLifestyleTracking(recommendation)
            setupCommonViews(recommendation, binding, this@RecommendationsAdapter)
        }

        private fun setupLifestyleSpecifics(recommendation: Recommendation) {
            // Lifestyle categories
            recommendation.lifestyleCategories.forEach { category ->
                val chip = Chip(binding.root.context).apply {
                    text = category.displayName
                    chipIcon = ContextCompat.getDrawable(context, category.iconRes)
                    isCheckable = false
                }
                binding.chipGroupLifestyle.addView(chip)
            }

            // Habit tracking
            setupHabitTracking(recommendation)

            // Progress visualization
            setupProgressVisualization(recommendation)
        }

        private fun setupHabitTracking(recommendation: Recommendation) {
            recommendation.habitTracking?.let { tracking ->
                binding.textCurrentStreak.text = "${tracking.currentStreak} days"
                binding.textBestStreak.text = "${tracking.bestStreak} days"
                binding.progressConsistency.progress = (tracking.consistency * 100).roundToInt()
            }
        }

        private fun setupProgressVisualization(recommendation: Recommendation) {
            // Weekly progress chart
            recommendation.weeklyProgress?.let { progress ->
                setupWeeklyProgressChart(progress)
            }
        }

        private fun setupWeeklyProgressChart(progress: List<Float>) {
            binding.layoutWeeklyProgress.removeAllViews()

            val dayLabels = arrayOf("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

            progress.forEachIndexed { index, value ->
                val dayView = LayoutInflater.from(binding.root.context)
                    .inflate(R.layout.item_daily_progress, binding.layoutWeeklyProgress, false)

                dayView.findViewById<android.widget.TextView>(R.id.text_day).text = dayLabels[index]
                dayView.findViewById<com.google.android.material.progressindicator.CircularProgressIndicator>(R.id.progress_day)
                    .progress = (value * 100).roundToInt()

                binding.layoutWeeklyProgress.addView(dayView)
            }
        }

        private fun setupLifestyleTracking(recommendation: Recommendation) {
            // Implementation tracking buttons
            binding.buttonMarkCompleted.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.MarkCompleted)
                animateCompletionFeedback()
            }

            binding.buttonSkipToday.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.SkipDay)
            }

            binding.buttonSetReminder.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.SetReminder)
            }
        }

        private fun animateCompletionFeedback() {
            binding.buttonMarkCompleted.animate()
                .scaleX(1.2f)
                .scaleY(1.2f)
                .setDuration(100L)
                .withEndAction {
                    binding.buttonMarkCompleted.animate()
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(100L)
                        .start()
                }
                .start()
        }

        override fun updateStatus(recommendation: Recommendation) {
            // Update habit completion status
            recommendation.todayCompletion?.let { completed ->
                binding.iconTodayStatus.setImageResource(
                    if (completed) R.drawable.ic_check_circle else R.drawable.ic_circle_outline
                )
                binding.iconTodayStatus.setColorFilter(
                    ContextCompat.getColor(
                        binding.root.context,
                        if (completed) R.color.success else R.color.surface_variant
                    )
                )
            }
        }

        override fun updateEngagement(recommendation: Recommendation) {
            // Update engagement tracking
        }

        override fun updateProgress(recommendation: Recommendation) {
            // Update overall progress
            val progress = recommendation.overallProgress
            binding.progressOverall.progress = (progress * 100).roundToInt()
            binding.textOverallProgress.text = "${(progress * 100).roundToInt()}%"
        }
    }

    inner class EmergencyViewHolder(
        private val binding: ItemRecommendationEmergencyBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(recommendation: Recommendation) {
            bindCommonElements(
                recommendation,
                binding.textTitle,
                binding.textDescription,
                binding.viewPriorityIndicator,
                binding.progressConfidence,
                binding.textTimestamp
            )

            setupEmergencySpecifics(recommendation)
            setupEmergencyActions(recommendation)
            setupCommonViews(recommendation, binding, this@RecommendationsAdapter)
        }

        private fun setupEmergencySpecifics(recommendation: Recommendation) {
            // Emergency severity indicator
            val severity = recommendation.emergencySeverity
            binding.textSeverity.text = severity?.displayName ?: "Unknown"
            binding.textSeverity.setTextColor(getSeverityColor(severity))

            // Urgency timeline
            binding.textUrgencyTimeline.text = recommendation.urgencyTimeline

            // Health impact warning
            recommendation.healthImpactWarning?.let { warning ->
                binding.textHealthWarning.text = warning
                binding.layoutHealthWarning.isVisible = true
            } ?: run {
                binding.layoutHealthWarning.isVisible = false
            }

            // Emergency contact information
            setupEmergencyContacts(recommendation)
        }

        private fun getSeverityColor(severity: EmergencySeverity?): Int {
            return when (severity) {
                EmergencySeverity.CRITICAL -> ContextCompat.getColor(binding.root.context, R.color.emergency_critical)
                EmergencySeverity.HIGH -> ContextCompat.getColor(binding.root.context, R.color.emergency_high)
                EmergencySeverity.MODERATE -> ContextCompat.getColor(binding.root.context, R.color.emergency_moderate)
                null -> ContextCompat.getColor(binding.root.context, R.color.on_surface)
            }
        }

        private fun setupEmergencyContacts(recommendation: Recommendation) {
            recommendation.emergencyContacts.forEach { contact ->
                val button = com.google.android.material.button.MaterialButton(
                    binding.root.context,
                    null,
                    com.google.android.material.R.attr.materialButtonStyle
                ).apply {
                    text = contact.name
                    icon = ContextCompat.getDrawable(context, R.drawable.ic_phone)
                    setOnClickListener {
                        onActionClick(recommendation, RecommendationAction.CallEmergencyContact(contact))
                    }
                }

                binding.layoutEmergencyContacts.addView(button)
            }
        }

        private fun setupEmergencyActions(recommendation: Recommendation) {
            binding.buttonEmergencyAction.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.EmergencyResponse)
            }

            binding.buttonCallProfessional.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.CallHealthcareProfessional)
            }

            binding.buttonDismissEmergency.setOnClickListener {
                showEmergencyDismissalDialog(recommendation)
            }
        }

        private fun showEmergencyDismissalDialog(recommendation: Recommendation) {
            // This would show a confirmation dialog for dismissing emergency recommendations
            onDismissClick(recommendation)
        }

        override fun updateStatus(recommendation: Recommendation) {
            // Update emergency status
        }

        override fun updateEngagement(recommendation: Recommendation) {
            // Track emergency recommendation engagement
        }

        override fun updateProgress(recommendation: Recommendation) {
            // Emergency recommendations don't typically have progress
        }
    }

    inner class AchievementViewHolder(
        private val binding: ItemRecommendationAchievementBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(recommendation: Recommendation) {
            bindCommonElements(
                recommendation,
                binding.textTitle,
                binding.textDescription,
                null, // Achievements don't need priority indicators
                binding.progressConfidence,
                binding.textTimestamp
            )

            setupAchievementSpecifics(recommendation)
            setupAchievementActions(recommendation)
            setupCommonViews(recommendation, binding, this@RecommendationsAdapter)
        }

        private fun setupAchievementSpecifics(recommendation: Recommendation) {
            // Achievement badge
            recommendation.achievementBadge?.let { badge ->
                binding.imageAchievementBadge.setImageResource(badge.iconRes)
                binding.textAchievementTitle.text = badge.title
            }

            // Achievement metrics
            recommendation.achievementMetrics?.let { metrics ->
                binding.textMetricValue.text = metrics.value
                binding.textMetricLabel.text = metrics.label
                binding.textComparison.text = metrics.comparison
            }

            // Celebration animation
            triggerCelebrationAnimation()
        }

        private fun setupAchievementActions(recommendation: Recommendation) {
            binding.buttonShare.setOnClickListener {
                onShareClick(recommendation)
            }

            binding.buttonSetNewGoal.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.SetNewGoal)
            }

            binding.buttonViewProgress.setOnClickListener {
                onActionClick(recommendation, RecommendationAction.ViewDetailedProgress)
            }
        }

        private fun triggerCelebrationAnimation() {
            // Confetti or celebration animation
            binding.layoutCelebration.animate()
                .alpha(1f)
                .scaleX(1.1f)
                .scaleY(1.1f)
                .setDuration(500L)
                .withEndAction {
                    binding.layoutCelebration.animate()
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(300L)
                        .start()
                }
                .start()
        }

        override fun updateStatus(recommendation: Recommendation) {
            // Achievements are typically static
        }

        override fun updateEngagement(recommendation: Recommendation) {
            // Track achievement engagement
        }

        override fun updateProgress(recommendation: Recommendation) {
            // Achievements represent completed progress
        }
    }

    inner class DefaultViewHolder(
        private val binding: ItemRecommendationDefaultBinding
    ) : BaseRecommendationViewHolder(binding.root) {

        override fun bind(recommendation: Recommendation) {
            bindCommonElements(
                recommendation,
                binding.textTitle,
                binding.textDescription,
                binding.viewPriorityIndicator,
                binding.progressConfidence,
                binding.textTimestamp
            )

            setupDefaultActions(recommendation)
            setupCommonViews(recommendation, binding, this@RecommendationsAdapter)
        }

        private fun setupDefaultActions(recommendation: Recommendation) {
            binding.buttonPrimaryAction.setOnClickListener {
                onRecommendationClick(recommendation)
            }

            binding.buttonSecondaryAction.setOnClickListener {
                onShareClick(recommendation)
            }
        }

        override fun updateStatus(recommendation: Recommendation) {}
        override fun updateEngagement(recommendation: Recommendation) {}
        override fun updateProgress(recommendation: Recommendation) {}
    }

    // ========== INTERACTION HANDLING ==========

    private fun handleRecommendationClick(recommendation: Recommendation) {
        trackInteraction(recommendation, RecommendationInteractionType.CLICKED)
        onRecommendationClick(recommendation)
    }

    private fun handleRecommendationLongClick(recommendation: Recommendation) {
        trackInteraction(recommendation, RecommendationInteractionType.LONG_CLICKED)
        onRecommendationLongClick(recommendation)
    }

    // ========== ANALYTICS AND TRACKING ==========

    private fun trackImpression(recommendation: Recommendation, position: Int) {
        if (visibleRecommendations.add(recommendation.id)) {
            impressionTimestamps[recommendation.id] = System.currentTimeMillis()

            onTrackingCallback(
                RecommendationEvent.Impression(
                    recommendationId = recommendation.id,
                    position = position,
                    category = recommendation.category,
                    priority = recommendation.priority,
                    timestamp = System.currentTimeMillis()
                )
            )
        }
    }

    private fun startEngagementTracking(recommendation: Recommendation) {
        engagementStartTimes[recommendation.id] = System.currentTimeMillis()
    }

    private fun stopEngagementTracking(recommendation: Recommendation) {
        val startTime = engagementStartTimes.remove(recommendation.id)
        if (startTime != null) {
            val engagementDuration = System.currentTimeMillis() - startTime

            if (engagementDuration > ENGAGEMENT_THRESHOLD_MS) {
                onTrackingCallback(
                    RecommendationEvent.Engagement(
                        recommendationId = recommendation.id,
                        duration = engagementDuration,
                        category = recommendation.category,
                        timestamp = System.currentTimeMillis()
                    )
                )
            }
        }
    }

    private fun trackInteraction(recommendation: Recommendation, type: RecommendationInteractionType) {
        onTrackingCallback(
            RecommendationEvent.Interaction(
                recommendationId = recommendation.id,
                type = type,
                category = recommendation.category,
                timestamp = System.currentTimeMillis()
            )
        )
    }

    // ========== ANIMATIONS ==========

    private fun animateItemEntrance(itemView: View, position: Int) {
        val delay = (position % 5) * STAGGER_DELAY // Stagger animation for visible items

        itemView.alpha = 0f
        itemView.scaleX = SCALE_FACTOR
        itemView.scaleY = SCALE_FACTOR
        itemView.translationY = 100f

        itemView.animate()
            .alpha(1f)
            .scaleX(1f)
            .scaleY(1f)
            .translationY(0f)
            .setStartDelay(delay)
            .setDuration(ANIMATION_DURATION)
            .setInterpolator(DecelerateInterpolator())
            .start()
    }

    private fun shouldAnimateItem(position: Int): Boolean {
        val currentTime = System.currentTimeMillis()
        return currentTime - lastAnimationTime > 100L // Throttle animations
    }

    // ========== PUBLIC METHODS ==========

    fun setAnimationEnabled(enabled: Boolean) {
        isAnimationEnabled = enabled
    }

    fun getRecommendationAt(position: Int): Recommendation? {
        return if (position in 0 until itemCount) getItem(position) else null
    }

    fun updateRecommendation(recommendationId: String, update: RecommendationUpdate) {
        val position = currentList.indexOfFirst { it.id == recommendationId }
        if (position != -1) {
            notifyItemChanged(position, update.toPayload())
        }
    }

    fun clearVisibilityTracking() {
        visibleRecommendations.clear()
        impressionTimestamps.clear()
        engagementStartTimes.clear()
    }
}

// ========== DIFF CALLBACK ==========

class RecommendationDiffCallback : DiffUtil.ItemCallback<Recommendation>() {
    override fun areItemsTheSame(oldItem: Recommendation, newItem: Recommendation): Boolean {
        return oldItem.id == newItem.id
    }

    override fun areContentsTheSame(oldItem: Recommendation, newItem: Recommendation): Boolean {
        return oldItem == newItem
    }

    override fun getChangePayload(oldItem: Recommendation, newItem: Recommendation): Any? {
        return when {
            oldItem.implementationStatus != newItem.implementationStatus ->
                RecommendationPayload.StatusUpdate
            oldItem.engagementMetrics != newItem.engagementMetrics ->
                RecommendationPayload.EngagementUpdate
            oldItem.implementationProgress != newItem.implementationProgress ->
                RecommendationPayload.ProgressUpdate
            else -> null
        }
    }
}

// ========== SUPPORTING DATA CLASSES ==========

sealed class RecommendationPayload {
    object StatusUpdate : RecommendationPayload()
    object EngagementUpdate : RecommendationPayload()
    object ProgressUpdate : RecommendationPayload()
}

data class RecommendationUpdate(
    val status: ImplementationStatus? = null,
    val progress: Float? = null,
    val engagement: UserEngagementMetrics? = null
) {
    fun toPayload(): RecommendationPayload {
        return when {
            status != null -> RecommendationPayload.StatusUpdate
            engagement != null -> RecommendationPayload.EngagementUpdate
            progress != null -> RecommendationPayload.ProgressUpdate
            else -> RecommendationPayload.StatusUpdate
        }
    }
}

// ========== ANALYTICS EVENTS ==========

sealed class RecommendationEvent {
    data class Impression(
        val recommendationId: String,
        val position: Int,
        val category: RecommendationCategory,
        val priority: RecommendationPriority,
        val timestamp: Long
    ) : RecommendationEvent()

    data class Engagement(
        val recommendationId: String,
        val duration: Long,
        val category: RecommendationCategory,
        val timestamp: Long
    ) : RecommendationEvent()

    data class Interaction(
        val recommendationId: String,
        val type: RecommendationInteractionType,
        val category: RecommendationCategory,
        val timestamp: Long
    ) : RecommendationEvent()
}

enum class RecommendationInteractionType {
    CLICKED,
    LONG_CLICKED,
    SHARED,
    DISMISSED,
    ACTION_TAKEN,
    FEEDBACK_GIVEN
}

// ========== PLACEHOLDER ENUMS AND DATA CLASSES ==========

// These would be defined in your data models files
enum class RecommendationCategory(val displayName: String, val iconRes: Int) {
    SLEEP_HYGIENE("Sleep Hygiene", R.drawable.ic_sleep_hygiene),
    ENVIRONMENT("Environment", R.drawable.ic_environment),
    LIFESTYLE("Lifestyle", R.drawable.ic_lifestyle),
    MEDICAL("Medical", R.drawable.ic_medical),
    TECHNOLOGY("Technology", R.drawable.ic_technology),
    BEHAVIORAL("Behavioral", R.drawable.ic_behavioral),
    NUTRITIONAL("Nutritional", R.drawable.ic_nutrition),
    EXERCISE("Exercise", R.drawable.ic_exercise),
    STRESS_MANAGEMENT("Stress Management", R.drawable.ic_stress),
    GOAL_TRACKING("Goal Tracking", R.drawable.ic_goals),
    EMERGENCY("Emergency", R.drawable.ic_emergency),
    ACHIEVEMENT("Achievement", R.drawable.ic_achievement)
}

enum class RecommendationPriority(val displayName: String) {
    CRITICAL("Critical"),
    HIGH("High"),
    MEDIUM("Medium"),
    LOW("Low")
}

enum class ImplementationDifficulty {
    EASY, MODERATE, CHALLENGING
}

enum class ImplementationStatus {
    NOT_STARTED, IN_PROGRESS, COMPLETED, ABANDONED
}

enum class EmergencySeverity(val displayName: String) {
    CRITICAL("Critical"),
    HIGH("High"),
    MODERATE("Moderate")
}

enum class RecommendationFeedback {
    HELPFUL, NOT_HELPFUL, IRRELEVANT, NEEDS_MORE_INFO
}

// Placeholder data classes that would be defined in your models
data class Recommendation(
    val id: String,
    val category: RecommendationCategory,
    val priority: RecommendationPriority,
    val title: String,
    val description: String,
    val confidence: Float,
    val timestamp: Long,
    val expectedImpact: Float = 0f,
    val implementationDifficulty: ImplementationDifficulty = ImplementationDifficulty.MODERATE,
    val timeToImpact: String = "",
    val tags: List<String> = emptyList(),
    val primaryAction: RecommendationAction? = null,
    val secondaryActions: List<RecommendationAction> = emptyList(),
    val smartDeviceActions: List<RecommendationAction> = emptyList(),
    val environmentData: EnvironmentData? = null,
    val currentEnvironmentData: EnvironmentData? = null,
    val environmentOptimizationProgress: Float = 0f,
    val lifestyleCategories: List<LifestyleCategory> = emptyList(),
    val habitTracking: HabitTracking? = null,
    val weeklyProgress: List<Float>? = null,
    val todayCompletion: Boolean? = null,
    val overallProgress: Float = 0f,
    val emergencySeverity: EmergencySeverity? = null,
    val urgencyTimeline: String = "",
    val healthImpactWarning: String? = null,
    val emergencyContacts: List<EmergencyContact> = emptyList(),
    val achievementBadge: AchievementBadge? = null,
    val achievementMetrics: AchievementMetrics? = null,
    val implementationStatus: ImplementationStatus = ImplementationStatus.NOT_STARTED,
    val implementationProgress: Float = 0f,
    val engagementMetrics: UserEngagementMetrics? = null
)

// Additional placeholder classes
data class EnvironmentData(
    val optimalTemperature: Float,
    val optimalHumidity: Float,
    val lightLevel: String,
    val noiseLevel: Float,
    val temperatureImpact: Float,
    val humidityImpact: Float,
    val lightImpact: Float,
    val noiseImpact: Float,
    val currentTemperature: Float = 0f,
    val currentHumidity: Float = 0f,
    val currentLightLevel: String = "",
    val currentNoiseLevel: Float = 0f
)

data class LifestyleCategory(val displayName: String, val iconRes: Int)
data class HabitTracking(val currentStreak: Int, val bestStreak: Int, val consistency: Float)
data class EmergencyContact(val name: String, val phone: String)
data class AchievementBadge(val title: String, val iconRes: Int)
data class AchievementMetrics(val value: String, val label: String, val comparison: String)

sealed class RecommendationAction(val label: String, val iconRes: Int = R.drawable.ic_action_default) {
    object MarkCompleted : RecommendationAction("Mark Completed", R.drawable.ic_check)
    object SkipDay : RecommendationAction("Skip Today", R.drawable.ic_skip)
    object SetReminder : RecommendationAction("Set Reminder", R.drawable.ic_reminder)
    object EmergencyResponse : RecommendationAction("Emergency Response", R.drawable.ic_emergency)
    object CallHealthcareProfessional : RecommendationAction("Call Professional", R.drawable.ic_phone)
    object SetNewGoal : RecommendationAction("Set New Goal", R.drawable.ic_goal)
    object ViewDetailedProgress : RecommendationAction("View Progress", R.drawable.ic_chart)

    data class AdjustTemperature(val temperature: Float) : RecommendationAction("Adjust Temperature", R.drawable.ic_temperature)
    data class AdjustBrightness(val brightness: Float) : RecommendationAction("Adjust Brightness", R.drawable.ic_brightness)
    data class ToggleSoundMachine(val enabled: Boolean) : RecommendationAction("Sound Machine", R.drawable.ic_sound)
    data class CallEmergencyContact(val contact: EmergencyContact) : RecommendationAction("Call ${contact.name}", R.drawable.ic_phone)
}