package com.example.somniai.ui.theme

import android.content.Context
import android.graphics.Color
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import com.example.somniai.data.*
import com.example.somniai.databinding.ItemSessionHistoryBinding
import com.example.somniai.databinding.ItemSessionHistoryExpandedBinding
import com.example.somniai.databinding.ItemSessionLoadingBinding
import com.example.somniai.databinding.ItemSessionErrorBinding
import com.example.somniai.utils.TimeUtils
import com.example.somniai.utils.FormatUtils
import kotlinx.coroutines.*
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

/**
 * Advanced Session History Adapter for SomniAI
 *
 * Comprehensive adapter supporting multiple view types, expandable sessions,
 * loading states, error handling, and performance optimization.
 * Integrates with the SomniAI analytics and AI processing pipeline.
 */
class SessionHistoryAdapter(
    private val context: Context,
    private val onSessionClick: (SessionSummaryDTO) -> Unit = {},
    private val onSessionLongClick: (SessionSummaryDTO) -> Boolean = { false },
    private val onExpandToggle: (SessionSummaryDTO, Boolean) -> Unit = { _, _ -> },
    private val onRetryError: () -> Unit = {},
    private val showExpandedView: Boolean = true,
    private val enableAnalytics: Boolean = true
) : ListAdapter<SessionHistoryItem, SessionHistoryAdapter.BaseViewHolder>(SessionHistoryDiffCallback()) {

    companion object {
        private const val VIEW_TYPE_SESSION = 0
        private const val VIEW_TYPE_SESSION_EXPANDED = 1
        private const val VIEW_TYPE_LOADING = 2
        private const val VIEW_TYPE_ERROR = 3
        private const val VIEW_TYPE_HEADER = 4

        // Animation constants
        private const val EXPAND_ANIMATION_DURATION = 300L
        private const val FADE_ANIMATION_DURATION = 200L

        // Quality thresholds for color coding
        private const val EXCELLENT_QUALITY_THRESHOLD = 8.0f
        private const val GOOD_QUALITY_THRESHOLD = 6.0f
        private const val FAIR_QUALITY_THRESHOLD = 4.0f
    }

    // Performance optimization
    private val expandedSessions = mutableSetOf<Long>()
    private val dateFormatter = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
    private val timeFormatter = SimpleDateFormat("h:mm a", Locale.getDefault())
    private val durationFormatter = SimpleDateFormat("H:mm", Locale.getDefault())

    // Analytics tracking
    private var lastClickTime = 0L
    private var clickCount = 0

    // ========== VIEW HOLDERS ==========

    abstract class BaseViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        abstract fun bind(item: SessionHistoryItem)
    }

    inner class SessionViewHolder(
        private val binding: ItemSessionHistoryBinding
    ) : BaseViewHolder(binding.root) {

        override fun bind(item: SessionHistoryItem) {
            if (item !is SessionHistoryItem.Session) return

            val session = item.session
            setupSessionBasicInfo(session)
            setupSessionMetrics(session)
            setupClickListeners(session)
            setupQualityIndicators(session)

            if (enableAnalytics) {
                trackSessionView(session)
            }
        }

        private fun setupSessionBasicInfo(session: SessionSummaryDTO) {
            binding.apply {
                // Date and time
                sessionDateText.text = dateFormatter.format(Date(session.startTime))
                sessionTimeText.text = "${timeFormatter.format(Date(session.startTime))} - ${
                    session.endTime?.let { timeFormatter.format(Date(it)) } ?: "Ongoing"
                }"

                // Duration
                durationText.text = TimeUtils.formatDuration(session.totalDuration)

                // Quality score with formatting
                session.qualityScore?.let { score ->
                    qualityScoreText.text = String.format("%.1f", score)
                    qualityScoreText.visibility = View.VISIBLE
                    qualityLabel.visibility = View.VISIBLE
                } ?: run {
                    qualityScoreText.visibility = View.GONE
                    qualityLabel.visibility = View.GONE
                }
            }
        }

        private fun setupSessionMetrics(session: SessionSummaryDTO) {
            binding.apply {
                // Efficiency
                efficiencyText.text = "${session.sleepEfficiency.roundToInt()}%"

                // Sleep phases (if available)
                if (session.lightSleepDuration > 0 || session.deepSleepDuration > 0 || session.remSleepDuration > 0) {
                    val total = session.lightSleepDuration + session.deepSleepDuration +
                            session.remSleepDuration + session.awakeDuration

                    if (total > 0) {
                        val lightPercent = ((session.lightSleepDuration.toFloat() / total) * 100).roundToInt()
                        val deepPercent = ((session.deepSleepDuration.toFloat() / total) * 100).roundToInt()
                        val remPercent = ((session.remSleepDuration.toFloat() / total) * 100).roundToInt()

                        phasesText.text = "L:${lightPercent}% D:${deepPercent}% R:${remPercent}%"
                        phasesText.visibility = View.VISIBLE
                    } else {
                        phasesText.visibility = View.GONE
                    }
                } else {
                    phasesText.visibility = View.GONE
                }

                // Movement indicator
                movementIndicator.text = when {
                    session.averageMovementIntensity < 2f -> "●"
                    session.averageMovementIntensity < 4f -> "●●"
                    session.averageMovementIntensity < 6f -> "●●●"
                    else -> "●●●●"
                }

                movementIndicator.setTextColor(getMovementColor(session.averageMovementIntensity))
            }
        }

        private fun setupQualityIndicators(session: SessionSummaryDTO) {
            binding.apply {
                // Quality color indicator
                session.qualityScore?.let { score ->
                    val color = getQualityColor(score)
                    qualityIndicator.setBackgroundColor(color)
                    qualityScoreText.setTextColor(color)
                    qualityIndicator.visibility = View.VISIBLE
                } ?: run {
                    qualityIndicator.visibility = View.GONE
                }

                // Session status
                val statusText = when {
                    session.endTime == null -> "In Progress"
                    session.qualityScore == null -> "Processing"
                    session.qualityScore >= EXCELLENT_QUALITY_THRESHOLD -> "Excellent"
                    session.qualityScore >= GOOD_QUALITY_THRESHOLD -> "Good"
                    session.qualityScore >= FAIR_QUALITY_THRESHOLD -> "Fair"
                    else -> "Poor"
                }

                sessionStatusText.text = statusText
                sessionStatusText.setTextColor(
                    session.qualityScore?.let { getQualityColor(it) }
                        ?: Color.parseColor("#888888")
                )
            }
        }

        private fun setupClickListeners(session: SessionSummaryDTO) {
            binding.root.setOnClickListener {
                if (showExpandedView && canExpand(session)) {
                    toggleExpansion(session)
                } else {
                    onSessionClick(session)
                }
                trackSessionInteraction(session, "click")
            }

            binding.root.setOnLongClickListener {
                trackSessionInteraction(session, "long_click")
                onSessionLongClick(session)
            }

            if (showExpandedView) {
                binding.expandIndicator?.setOnClickListener {
                    toggleExpansion(session)
                    trackSessionInteraction(session, "expand_toggle")
                }
            }
        }

        private fun toggleExpansion(session: SessionSummaryDTO) {
            val isExpanded = expandedSessions.contains(session.id)
            val newExpandedState = !isExpanded

            if (newExpandedState) {
                expandedSessions.add(session.id)
            } else {
                expandedSessions.remove(session.id)
            }

            onExpandToggle(session, newExpandedState)

            // Trigger rebind with animation
            notifyItemChanged(bindingAdapterPosition)
        }
    }

    inner class SessionExpandedViewHolder(
        private val binding: ItemSessionHistoryExpandedBinding
    ) : BaseViewHolder(binding.root) {

        override fun bind(item: SessionHistoryItem) {
            if (item !is SessionHistoryItem.Session) return

            val session = item.session
            setupExpandedContent(session)
            setupExpandedClickListeners(session)
        }

        private fun setupExpandedContent(session: SessionSummaryDTO) {
            binding.apply {
                // Copy basic info from regular view
                setupBasicExpandedInfo(session)

                // Additional detailed information
                setupDetailedMetrics(session)
                setupSleepPhaseDetails(session)
                setupEnvironmentalDetails(session)
                setupAIInsights(session)
            }
        }

        private fun setupBasicExpandedInfo(session: SessionSummaryDTO) {
            binding.apply {
                sessionDateText.text = dateFormatter.format(Date(session.startTime))
                sessionTimeText.text = "${timeFormatter.format(Date(session.startTime))} - ${
                    session.endTime?.let { timeFormatter.format(Date(it)) } ?: "Ongoing"
                }"
                durationText.text = TimeUtils.formatDuration(session.totalDuration)

                session.qualityScore?.let { score ->
                    qualityScoreText.text = String.format("%.1f/10", score)
                    qualityIndicator.setBackgroundColor(getQualityColor(score))
                }
            }
        }

        private fun setupDetailedMetrics(session: SessionSummaryDTO) {
            binding.apply {
                // Sleep efficiency with more detail
                efficiencyText.text = "${session.sleepEfficiency.roundToInt()}%"
                efficiencyProgressBar?.progress = session.sleepEfficiency.roundToInt()

                // Sleep latency
                sleepLatencyText?.text = if (session.sleepLatency > 0) {
                    TimeUtils.formatDuration(session.sleepLatency)
                } else "N/A"

                // Wake count
                wakeCountText?.text = session.wakeCount?.toString() ?: "N/A"

                // Movement events
                movementEventsText?.text = session.totalMovementEvents.toString()
            }
        }

        private fun setupSleepPhaseDetails(session: SessionSummaryDTO) {
            binding.apply {
                val total = session.lightSleepDuration + session.deepSleepDuration +
                        session.remSleepDuration + session.awakeDuration

                if (total > 0) {
                    // Phase durations
                    lightSleepText?.text = TimeUtils.formatDuration(session.lightSleepDuration)
                    deepSleepText?.text = TimeUtils.formatDuration(session.deepSleepDuration)
                    remSleepText?.text = TimeUtils.formatDuration(session.remSleepDuration)
                    awakeTimeText?.text = TimeUtils.formatDuration(session.awakeDuration)

                    // Phase percentages
                    val lightPercent = ((session.lightSleepDuration.toFloat() / total) * 100).roundToInt()
                    val deepPercent = ((session.deepSleepDuration.toFloat() / total) * 100).roundToInt()
                    val remPercent = ((session.remSleepDuration.toFloat() / total) * 100).roundToInt()
                    val awakePercent = ((session.awakeDuration.toFloat() / total) * 100).roundToInt()

                    lightPercentText?.text = "$lightPercent%"
                    deepPercentText?.text = "$deepPercent%"
                    remPercentText?.text = "$remPercent%"
                    awakePercentText?.text = "$awakePercent%"

                    // Progress bars
                    lightProgressBar?.progress = lightPercent
                    deepProgressBar?.progress = deepPercent
                    remProgressBar?.progress = remPercent
                    awakeProgressBar?.progress = awakePercent
                }
            }
        }

        private fun setupEnvironmentalDetails(session: SessionSummaryDTO) {
            binding.apply {
                // Average noise level
                avgNoiseText?.text = "${session.averageNoiseLevel.roundToInt()} dB"

                // Peak noise level
                session.peakNoiseLevel?.let { peak ->
                    peakNoiseText?.text = "${peak.roundToInt()} dB"
                } ?: run {
                    peakNoiseText?.text = "N/A"
                }

                // Temperature (if available)
                session.averageTemperature?.let { temp ->
                    temperatureText?.text = "${temp.roundToInt()}°"
                } ?: run {
                    temperatureText?.text = "N/A"
                }

                // Movement intensity
                movementIntensityText?.text = String.format("%.1f", session.averageMovementIntensity)
                movementIntensityBar?.progress = (session.averageMovementIntensity * 10).roundToInt()
            }
        }

        private fun setupAIInsights(session: SessionSummaryDTO) {
            binding.apply {
                // AI-generated insights count
                val insightCount = session.insights?.size ?: 0
                aiInsightsCountText?.text = if (insightCount > 0) {
                    "$insightCount insights available"
                } else {
                    "No insights generated"
                }

                // Show first insight if available
                session.insights?.firstOrNull()?.let { insight ->
                    firstInsightText?.text = insight.description
                    firstInsightText?.visibility = View.VISIBLE
                } ?: run {
                    firstInsightText?.visibility = View.GONE
                }
            }
        }

        private fun setupExpandedClickListeners(session: SessionSummaryDTO) {
            binding.apply {
                collapseButton?.setOnClickListener {
                    expandedSessions.remove(session.id)
                    onExpandToggle(session, false)
                    notifyItemChanged(bindingAdapterPosition)
                    trackSessionInteraction(session, "collapse")
                }

                viewDetailsButton?.setOnClickListener {
                    onSessionClick(session)
                    trackSessionInteraction(session, "view_details")
                }
            }
        }
    }

    inner class LoadingViewHolder(
        private val binding: ItemSessionLoadingBinding
    ) : BaseViewHolder(binding.root) {

        override fun bind(item: SessionHistoryItem) {
            binding.apply {
                loadingText.text = when (item) {
                    is SessionHistoryItem.Loading -> item.message
                    else -> "Loading session data..."
                }

                // Start loading animation
                loadingProgressBar.visibility = View.VISIBLE
            }
        }
    }

    inner class ErrorViewHolder(
        private val binding: ItemSessionErrorBinding
    ) : BaseViewHolder(binding.root) {

        override fun bind(item: SessionHistoryItem) {
            binding.apply {
                errorMessage.text = when (item) {
                    is SessionHistoryItem.Error -> item.message
                    else -> "Error loading session data"
                }

                retryButton.setOnClickListener {
                    onRetryError()
                    trackInteraction("error_retry")
                }

                // Error icon visibility
                errorIcon.visibility = View.VISIBLE
            }
        }
    }

    inner class HeaderViewHolder(itemView: View) : BaseViewHolder(itemView) {
        private val headerText: TextView = itemView.findViewById(android.R.id.text1)

        override fun bind(item: SessionHistoryItem) {
            if (item is SessionHistoryItem.Header) {
                headerText.text = item.title
            }
        }
    }

    // ========== ADAPTER IMPLEMENTATION ==========

    override fun getItemViewType(position: Int): Int {
        return when (val item = getItem(position)) {
            is SessionHistoryItem.Session -> {
                if (showExpandedView && expandedSessions.contains(item.session.id)) {
                    VIEW_TYPE_SESSION_EXPANDED
                } else {
                    VIEW_TYPE_SESSION
                }
            }
            is SessionHistoryItem.Loading -> VIEW_TYPE_LOADING
            is SessionHistoryItem.Error -> VIEW_TYPE_ERROR
            is SessionHistoryItem.Header -> VIEW_TYPE_HEADER
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): BaseViewHolder {
        val inflater = LayoutInflater.from(parent.context)

        return when (viewType) {
            VIEW_TYPE_SESSION -> {
                val binding = ItemSessionHistoryBinding.inflate(inflater, parent, false)
                SessionViewHolder(binding)
            }
            VIEW_TYPE_SESSION_EXPANDED -> {
                val binding = ItemSessionHistoryExpandedBinding.inflate(inflater, parent, false)
                SessionExpandedViewHolder(binding)
            }
            VIEW_TYPE_LOADING -> {
                val binding = ItemSessionLoadingBinding.inflate(inflater, parent, false)
                LoadingViewHolder(binding)
            }
            VIEW_TYPE_ERROR -> {
                val binding = ItemSessionErrorBinding.inflate(inflater, parent, false)
                ErrorViewHolder(binding)
            }
            VIEW_TYPE_HEADER -> {
                val view = inflater.inflate(android.R.layout.simple_list_item_1, parent, false)
                HeaderViewHolder(view)
            }
            else -> throw IllegalArgumentException("Unknown view type: $viewType")
        }
    }

    override fun onBindViewHolder(holder: BaseViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    // ========== UTILITY METHODS ==========

    private fun canExpand(session: SessionSummaryDTO): Boolean {
        return session.endTime != null && // Session is complete
                (session.lightSleepDuration > 0 || session.deepSleepDuration > 0 || session.remSleepDuration > 0)
    }

    private fun getQualityColor(score: Float): Int {
        return when {
            score >= EXCELLENT_QUALITY_THRESHOLD -> Color.parseColor("#4CAF50") // Green
            score >= GOOD_QUALITY_THRESHOLD -> Color.parseColor("#FFC107") // Amber
            score >= FAIR_QUALITY_THRESHOLD -> Color.parseColor("#FF9800") // Orange
            else -> Color.parseColor("#F44336") // Red
        }
    }

    private fun getMovementColor(intensity: Float): Int {
        return when {
            intensity < 2f -> Color.parseColor("#4CAF50") // Green - Low movement
            intensity < 4f -> Color.parseColor("#FFC107") // Amber - Medium movement
            intensity < 6f -> Color.parseColor("#FF9800") // Orange - High movement
            else -> Color.parseColor("#F44336") // Red - Very high movement
        }
    }

    // ========== ANALYTICS AND TRACKING ==========

    private fun trackSessionView(session: SessionSummaryDTO) {
        // Track session view for analytics
        // Implementation would integrate with your analytics system
    }

    private fun trackSessionInteraction(session: SessionSummaryDTO, interactionType: String) {
        val currentTime = System.currentTimeMillis()

        // Prevent rapid-fire clicks
        if (currentTime - lastClickTime < 500) {
            clickCount++
            if (clickCount > 3) return // Ignore rapid clicks
        } else {
            clickCount = 0
        }

        lastClickTime = currentTime

        // Track interaction for analytics
        // Implementation would integrate with your analytics system
    }

    private fun trackInteraction(interactionType: String) {
        // Track general interactions
        // Implementation would integrate with your analytics system
    }

    // ========== PUBLIC METHODS ==========

    /**
     * Toggle expansion state for a specific session
     */
    fun toggleSessionExpansion(sessionId: Long) {
        val position = currentList.indexOfFirst {
            it is SessionHistoryItem.Session && it.session.id == sessionId
        }
        if (position != -1) {
            val isExpanded = expandedSessions.contains(sessionId)
            if (isExpanded) {
                expandedSessions.remove(sessionId)
            } else {
                expandedSessions.add(sessionId)
            }
            notifyItemChanged(position)
        }
    }

    /**
     * Expand all sessions
     */
    fun expandAllSessions() {
        currentList.forEach { item ->
            if (item is SessionHistoryItem.Session && canExpand(item.session)) {
                expandedSessions.add(item.session.id)
            }
        }
        notifyDataSetChanged()
    }

    /**
     * Collapse all sessions
     */
    fun collapseAllSessions() {
        expandedSessions.clear()
        notifyDataSetChanged()
    }

    /**
     * Get count of expanded sessions
     */
    fun getExpandedSessionsCount(): Int = expandedSessions.size

    /**
     * Check if session is expanded
     */
    fun isSessionExpanded(sessionId: Long): Boolean = expandedSessions.contains(sessionId)
}

// ========== DIFF CALLBACK ==========

class SessionHistoryDiffCallback : DiffUtil.ItemCallback<SessionHistoryItem>() {
    override fun areItemsTheSame(oldItem: SessionHistoryItem, newItem: SessionHistoryItem): Boolean {
        return when {
            oldItem is SessionHistoryItem.Session && newItem is SessionHistoryItem.Session ->
                oldItem.session.id == newItem.session.id
            oldItem is SessionHistoryItem.Header && newItem is SessionHistoryItem.Header ->
                oldItem.title == newItem.title
            oldItem is SessionHistoryItem.Loading && newItem is SessionHistoryItem.Loading -> true
            oldItem is SessionHistoryItem.Error && newItem is SessionHistoryItem.Error -> true
            else -> false
        }
    }

    override fun areContentsTheSame(oldItem: SessionHistoryItem, newItem: SessionHistoryItem): Boolean {
        return when {
            oldItem is SessionHistoryItem.Session && newItem is SessionHistoryItem.Session ->
                oldItem.session == newItem.session
            oldItem is SessionHistoryItem.Header && newItem is SessionHistoryItem.Header ->
                oldItem == newItem
            oldItem is SessionHistoryItem.Loading && newItem is SessionHistoryItem.Loading ->
                oldItem.message == newItem.message
            oldItem is SessionHistoryItem.Error && newItem is SessionHistoryItem.Error ->
                oldItem.message == newItem.message
            else -> false
        }
    }
}

// ========== SEALED CLASS FOR DIFFERENT ITEM TYPES ==========

sealed class SessionHistoryItem {
    data class Session(val session: SessionSummaryDTO) : SessionHistoryItem()
    data class Header(val title: String) : SessionHistoryItem()
    data class Loading(val message: String = "Loading...") : SessionHistoryItem()
    data class Error(val message: String, val throwable: Throwable? = null) : SessionHistoryItem()
}