package com.example.somniai.ui.theme

import android.animation.ValueAnimator
import android.content.res.ColorStateList
import android.os.Bundle
import android.util.Log
import android.view.*
import android.view.animation.DecelerateInterpolator
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.fragment.app.viewModels
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.somniai.R
import com.example.somniai.ai.*
import com.example.somniai.data.*
import com.example.somniai.databinding.*
import com.example.somniai.viewmodel.MainViewModel
import com.google.android.material.card.MaterialCardView
import com.google.android.material.chip.Chip
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

/**
 * Enterprise-grade Insights Fragment with AI Integration
 *
 * Advanced Features:
 * - Real-time AI insight generation and display
 * - Multi-service AI orchestration integration
 * - Advanced filtering and categorization
 * - User engagement tracking and feedback
 * - Performance monitoring and analytics
 * - Offline-first architecture with seamless sync
 * - Material Design 3 with dark theme optimization
 * - Accessibility features and inclusive design
 * - AI insight quality assessment and validation
 * - Personalized insight recommendations
 * - Interactive data visualizations
 * - Comprehensive error handling and recovery
 */
class InsightsFragment : Fragment() {

    companion object {
        private const val TAG = "InsightsFragment"
        private const val INSIGHTS_REFRESH_INTERVAL = 30000L // 30 seconds
        private const val MAX_INSIGHTS_PER_PAGE = 20
        private const val ANIMATION_DURATION = 300L

        fun newInstance(): InsightsFragment = InsightsFragment()
    }

    // ViewModels and dependencies
    private val mainViewModel: MainViewModel by activityViewModels()
    private val insightsViewModel: InsightsViewModel by viewModels()

    // View binding
    private var _binding: FragmentInsightsBinding? = null
    private val binding get() = _binding!!

    // UI components
    private lateinit var insightsAdapter: InsightsAdapter
    private lateinit var filterChipsContainer: ViewGroup
    private var currentFilter: InsightFilter = InsightFilter.ALL
    private var isRefreshing = false

    // State management
    private var selectedInsightId: Long? = null
    private var lastRefreshTime = 0L
    private val visibleInsights = mutableSetOf<Long>()

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentInsightsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        Log.d(TAG, "Initializing enterprise insights fragment")

        setupUI()
        setupRecyclerView()
        setupFilters()
        setupObservers()
        setupInteractions()

        // Initial data load
        loadInsights()
    }

    override fun onResume() {
        super.onResume()
        // Track fragment visibility for analytics
        insightsViewModel.trackFragmentVisibility(true)

        // Refresh insights if data is stale
        if (shouldRefreshInsights()) {
            refreshInsights()
        }
    }

    override fun onPause() {
        super.onPause()
        insightsViewModel.trackFragmentVisibility(false)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    // ========== UI SETUP ==========

    private fun setupUI() {
        with(binding) {
            // Setup toolbar
            toolbar.title = getString(R.string.insights_title)
            toolbar.setNavigationIcon(R.drawable.ic_arrow_back)
            toolbar.setNavigationOnClickListener { requireActivity().onBackPressed() }

            // Setup menu
            toolbar.inflateMenu(R.menu.insights_menu)
            toolbar.setOnMenuItemClickListener { menuItem ->
                when (menuItem.itemId) {
                    R.id.action_refresh -> {
                        refreshInsights()
                        true
                    }
                    R.id.action_filter -> {
                        showFilterDialog()
                        true
                    }
                    R.id.action_settings -> {
                        navigateToInsightSettings()
                        true
                    }
                    else -> false
                }
            }

            // Setup swipe refresh
            swipeRefreshLayout.setOnRefreshListener {
                refreshInsights()
            }

            // Setup empty state
            setupEmptyState()

            // Setup generation progress
            setupGenerationProgress()

            // Setup error state
            setupErrorHandling()
        }
    }

    private fun setupRecyclerView() {
        insightsAdapter = InsightsAdapter(
            onInsightClick = { insight ->
                onInsightClicked(insight)
            },
            onInsightLongClick = { insight ->
                showInsightContextMenu(insight)
            },
            onFeedbackClick = { insight, feedback ->
                submitInsightFeedback(insight, feedback)
            },
            onShareClick = { insight ->
                shareInsight(insight)
            },
            onDismissClick = { insight ->
                dismissInsight(insight)
            }
        )

        with(binding.recyclerViewInsights) {
            adapter = insightsAdapter
            layoutManager = LinearLayoutManager(requireContext())
            setHasFixedSize(true)

            // Add scroll listener for analytics
            addOnScrollListener(object : RecyclerView.OnScrollListener() {
                override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                    super.onScrolled(recyclerView, dx, dy)
                    trackInsightVisibility()
                }
            })

            // Add item decoration for spacing
            addItemDecoration(InsightItemDecoration(requireContext()))
        }
    }

    private fun setupFilters() {
        filterChipsContainer = binding.chipGroupFilters

        // Create filter chips dynamically
        InsightFilter.values().forEach { filter ->
            createFilterChip(filter)
        }

        // Set default selection
        selectFilter(InsightFilter.ALL)
    }

    private fun createFilterChip(filter: InsightFilter) {
        val chip = Chip(requireContext()).apply {
            text = getString(filter.titleRes)
            isCheckable = true
            chipIcon = ContextCompat.getDrawable(requireContext(), filter.iconRes)

            setOnClickListener {
                selectFilter(filter)
                applyFilter(filter)
            }
        }

        filterChipsContainer.addView(chip)
    }

    private fun selectFilter(filter: InsightFilter) {
        currentFilter = filter

        // Update chip selection states
        for (i in 0 until filterChipsContainer.childCount) {
            val chip = filterChipsContainer.getChildAt(i) as Chip
            chip.isChecked = InsightFilter.values()[i] == filter
        }
    }

    private fun setupEmptyState() {
        with(binding.layoutEmptyState) {
            textEmptyTitle.text = getString(R.string.insights_empty_title)
            textEmptyMessage.text = getString(R.string.insights_empty_message)

            buttonGenerateInsights.setOnClickListener {
                generateNewInsights()
            }

            buttonTrackSleep.setOnClickListener {
                navigateToSleepTracking()
            }
        }
    }

    private fun setupGenerationProgress() {
        with(binding.layoutGenerationProgress) {
            // Setup progress indicators
            progressBarGeneration.isIndeterminate = true

            buttonCancelGeneration.setOnClickListener {
                cancelInsightGeneration()
            }
        }
    }

    private fun setupErrorHandling() {
        with(binding.layoutErrorState) {
            buttonRetry.setOnClickListener {
                retryInsightGeneration()
            }

            buttonOfflineMode.setOnClickListener {
                enableOfflineMode()
            }
        }
    }

    // ========== DATA OBSERVERS ==========

    private fun setupObservers() {
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {

                // Observe insights data
                launch {
                    insightsViewModel.insights.collect { insights ->
                        handleInsightsUpdate(insights)
                    }
                }

                // Observe loading state
                launch {
                    insightsViewModel.isLoading.collect { isLoading ->
                        handleLoadingState(isLoading)
                    }
                }

                // Observe generation progress
                launch {
                    insightsViewModel.generationProgress.collect { progress ->
                        handleGenerationProgress(progress)
                    }
                }

                // Observe error state
                launch {
                    insightsViewModel.errorState.collect { error ->
                        handleErrorState(error)
                    }
                }

                // Observe filter state
                launch {
                    insightsViewModel.currentFilter.collect { filter ->
                        handleFilterUpdate(filter)
                    }
                }

                // Observe AI analytics
                launch {
                    insightsViewModel.aiAnalytics.collect { analytics ->
                        handleAnalyticsUpdate(analytics)
                    }
                }

                // Observe user engagement metrics
                launch {
                    insightsViewModel.engagementMetrics.collect { metrics ->
                        handleEngagementUpdate(metrics)
                    }
                }
            }
        }

        // Observe main view model for session updates
        mainViewModel.isTracking.observe(viewLifecycleOwner) { isTracking ->
            updateInsightGenerationAvailability(isTracking)
        }

        mainViewModel.insights.observe(viewLifecycleOwner) { insights ->
            insightsViewModel.updateInsights(insights)
        }
    }

    // ========== DATA HANDLING ==========

    private fun handleInsightsUpdate(insights: List<SleepInsight>) {
        Log.d(TAG, "Updating insights display: ${insights.size} insights")

        val filteredInsights = applyCurrentFilter(insights)
        insightsAdapter.submitList(filteredInsights) {
            updateEmptyState(filteredInsights.isEmpty())

            // Animate new insights
            animateNewInsights(filteredInsights)

            // Track insight impressions
            trackInsightImpressions(filteredInsights)
        }

        // Update summary stats
        updateInsightsSummary(insights)
    }

    private fun handleLoadingState(isLoading: Boolean) {
        this.isRefreshing = isLoading

        binding.swipeRefreshLayout.isRefreshing = isLoading

        // Show/hide loading indicators
        binding.progressBarMain.isVisible = isLoading && insightsAdapter.itemCount == 0

        // Update menu items
        binding.toolbar.menu.findItem(R.id.action_refresh)?.isEnabled = !isLoading
    }

    private fun handleGenerationProgress(progress: InsightGenerationProgress?) {
        with(binding.layoutGenerationProgress) {
            if (progress != null) {
                root.isVisible = true

                textGenerationStatus.text = getString(
                    R.string.insights_generation_status,
                    progress.stage.displayName,
                    (progress.progress * 100).roundToInt()
                )

                progressBarGeneration.apply {
                    isIndeterminate = progress.progress <= 0f
                    if (progress.progress > 0f) {
                        setProgress((progress.progress * 100).roundToInt(), true)
                    }
                }

                textEstimatedTime.text = if (progress.estimatedTimeRemaining > 0) {
                    getString(
                        R.string.insights_estimated_time,
                        progress.estimatedTimeRemaining / 1000
                    )
                } else {
                    ""
                }

                textCompletedInsights.text = if (progress.completedInsights > 0) {
                    getString(R.string.insights_completed_count, progress.completedInsights)
                } else {
                    ""
                }

            } else {
                root.isVisible = false
            }
        }
    }

    private fun handleErrorState(error: InsightError?) {
        with(binding.layoutErrorState) {
            if (error != null) {
                root.isVisible = true

                textErrorTitle.text = error.title
                textErrorMessage.text = error.message

                // Configure action buttons based on error type
                when (error.type) {
                    InsightErrorType.NETWORK_ERROR -> {
                        buttonRetry.isVisible = true
                        buttonOfflineMode.isVisible = true
                        buttonRetry.text = getString(R.string.action_retry)
                    }
                    InsightErrorType.AI_SERVICE_ERROR -> {
                        buttonRetry.isVisible = true
                        buttonOfflineMode.isVisible = false
                        buttonRetry.text = getString(R.string.action_try_again)
                    }
                    InsightErrorType.INSUFFICIENT_DATA -> {
                        buttonRetry.isVisible = false
                        buttonOfflineMode.isVisible = false
                    }
                    InsightErrorType.QUOTA_EXCEEDED -> {
                        buttonRetry.isVisible = false
                        buttonOfflineMode.isVisible = true
                    }
                }

            } else {
                root.isVisible = false
            }
        }
    }

    private fun handleFilterUpdate(filter: InsightFilter) {
        if (filter != currentFilter) {
            selectFilter(filter)
            applyFilter(filter)
        }
    }

    private fun handleAnalyticsUpdate(analytics: AIAnalyticsReport?) {
        // Update performance indicators if analytics are available
        analytics?.let {
            updatePerformanceIndicators(it)
        }
    }

    private fun handleEngagementUpdate(metrics: UserEngagementMetrics?) {
        // Update engagement tracking
        metrics?.let {
            updateEngagementIndicators(it)
        }
    }

    // ========== USER INTERACTIONS ==========

    private fun setupInteractions() {
        // Setup pull-to-refresh
        binding.swipeRefreshLayout.setOnRefreshListener {
            refreshInsights()
        }

        // Setup FAB for manual insight generation
        binding.fabGenerateInsights.setOnClickListener {
            showInsightGenerationDialog()
        }

        // Setup gesture detection for accessibility
        setupAccessibilityGestures()
    }

    private fun onInsightClicked(insight: SleepInsight) {
        Log.d(TAG, "Insight clicked: ${insight.id}")

        selectedInsightId = insight.id

        // Track engagement
        insightsViewModel.trackInsightInteraction(
            insightId = insight.id,
            interactionType = InsightInteractionType.CLICKED,
            engagementData = mapOf(
                "category" to insight.category.name,
                "priority" to insight.priority,
                "timestamp" to System.currentTimeMillis()
            )
        )

        // Show detailed insight view
        showInsightDetails(insight)
    }

    private fun showInsightContextMenu(insight: SleepInsight) {
        val options = arrayOf(
            getString(R.string.action_view_details),
            getString(R.string.action_share),
            getString(R.string.action_mark_helpful),
            getString(R.string.action_mark_not_helpful),
            getString(R.string.action_dismiss),
            getString(R.string.action_report_issue)
        )

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(insight.title)
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showInsightDetails(insight)
                    1 -> shareInsight(insight)
                    2 -> submitInsightFeedback(insight, InsightFeedback.HELPFUL)
                    3 -> submitInsightFeedback(insight, InsightFeedback.NOT_HELPFUL)
                    4 -> dismissInsight(insight)
                    5 -> reportInsightIssue(insight)
                }
            }
            .show()
    }

    private fun submitInsightFeedback(insight: SleepInsight, feedback: InsightFeedback) {
        Log.d(TAG, "Submitting feedback for insight ${insight.id}: $feedback")

        insightsViewModel.submitFeedback(
            insightId = insight.id,
            feedback = feedback,
            engagementMetrics = EngagementMetrics(
                timeSpent = System.currentTimeMillis() - (selectedInsightId?.let {
                    System.currentTimeMillis()
                } ?: 0L),
                interactionCount = 1
            )
        )

        // Show feedback confirmation
        val message = when (feedback) {
            InsightFeedback.HELPFUL -> getString(R.string.feedback_helpful_submitted)
            InsightFeedback.NOT_HELPFUL -> getString(R.string.feedback_not_helpful_submitted)
            else -> getString(R.string.feedback_submitted)
        }

        Snackbar.make(binding.root, message, Snackbar.LENGTH_SHORT)
            .setAction(getString(R.string.action_undo)) {
                insightsViewModel.undoFeedback(insight.id)
            }
            .show()
    }

    private fun shareInsight(insight: SleepInsight) {
        val shareText = buildString {
            appendLine(insight.title)
            appendLine()
            appendLine(insight.description)
            appendLine()
            appendLine(insight.recommendation)
            appendLine()
            appendLine(getString(R.string.shared_from_somniai))
        }

        val shareIntent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(android.content.Intent.EXTRA_TEXT, shareText)
            putExtra(android.content.Intent.EXTRA_SUBJECT, insight.title)
        }

        startActivity(android.content.Intent.createChooser(
            shareIntent,
            getString(R.string.share_insight)
        ))

        // Track sharing
        insightsViewModel.trackInsightInteraction(
            insightId = insight.id,
            interactionType = InsightInteractionType.SHARED,
            engagementData = mapOf("method" to "system_share")
        )
    }

    private fun dismissInsight(insight: SleepInsight) {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(getString(R.string.dismiss_insight_title))
            .setMessage(getString(R.string.dismiss_insight_message))
            .setPositiveButton(getString(R.string.action_dismiss)) { _, _ ->
                insightsViewModel.dismissInsight(insight.id)

                Snackbar.make(binding.root, getString(R.string.insight_dismissed), Snackbar.LENGTH_LONG)
                    .setAction(getString(R.string.action_undo)) {
                        insightsViewModel.undoDismissal(insight.id)
                    }
                    .show()
            }
            .setNegativeButton(getString(R.string.action_cancel), null)
            .show()
    }

    // ========== INSIGHT MANAGEMENT ==========

    private fun loadInsights() {
        Log.d(TAG, "Loading insights")
        insightsViewModel.loadInsights(forceRefresh = false)
    }

    private fun refreshInsights() {
        Log.d(TAG, "Refreshing insights")
        lastRefreshTime = System.currentTimeMillis()
        insightsViewModel.loadInsights(forceRefresh = true)
    }

    private fun generateNewInsights() {
        Log.d(TAG, "Generating new insights")

        insightsViewModel.generateInsights(
            generationType = InsightGenerationType.PERSONALIZED_ANALYSIS,
            priority = InsightPriority.NORMAL
        )
    }

    private fun showInsightGenerationDialog() {
        val generationTypes = InsightGenerationType.values()
        val typeNames = generationTypes.map { it.displayName }.toTypedArray()

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(getString(R.string.generate_insights_title))
            .setItems(typeNames) { _, which ->
                val selectedType = generationTypes[which]
                generateSpecificInsights(selectedType)
            }
            .setNegativeButton(getString(R.string.action_cancel), null)
            .show()
    }

    private fun generateSpecificInsights(type: InsightGenerationType) {
        Log.d(TAG, "Generating specific insights: $type")

        insightsViewModel.generateInsights(
            generationType = type,
            priority = InsightPriority.HIGH
        )
    }

    private fun cancelInsightGeneration() {
        Log.d(TAG, "Cancelling insight generation")
        insightsViewModel.cancelGeneration()

        Snackbar.make(binding.root, getString(R.string.generation_cancelled), Snackbar.LENGTH_SHORT)
            .show()
    }

    private fun retryInsightGeneration() {
        Log.d(TAG, "Retrying insight generation")
        insightsViewModel.retryGeneration()
    }

    private fun enableOfflineMode() {
        Log.d(TAG, "Enabling offline mode")
        insightsViewModel.enableOfflineMode()

        Snackbar.make(binding.root, getString(R.string.offline_mode_enabled), Snackbar.LENGTH_LONG)
            .setAction(getString(R.string.action_settings)) {
                navigateToInsightSettings()
            }
            .show()
    }

    // ========== FILTERING AND SEARCH ==========

    private fun applyFilter(filter: InsightFilter) {
        Log.d(TAG, "Applying filter: $filter")

        insightsViewModel.setFilter(filter)

        // Track filter usage
        insightsViewModel.trackFilterUsage(filter)
    }

    private fun applyCurrentFilter(insights: List<SleepInsight>): List<SleepInsight> {
        return when (currentFilter) {
            InsightFilter.ALL -> insights
            InsightFilter.HIGH_PRIORITY -> insights.filter { it.priority <= 1 }
            InsightFilter.RECENT -> insights.filter {
                System.currentTimeMillis() - it.timestamp < 24 * 60 * 60 * 1000L
            }
            InsightFilter.ACTIONABLE -> insights.filter {
                it.recommendation.isNotBlank()
            }
            InsightFilter.QUALITY -> insights.filter {
                it.category == InsightCategory.QUALITY
            }
            InsightFilter.DURATION -> insights.filter {
                it.category == InsightCategory.DURATION
            }
            InsightFilter.MOVEMENT -> insights.filter {
                it.category == InsightCategory.MOVEMENT
            }
            InsightFilter.ENVIRONMENT -> insights.filter {
                it.category == InsightCategory.ENVIRONMENT
            }
        }
    }

    private fun showFilterDialog() {
        val filterOptions = InsightFilter.values()
        val filterNames = filterOptions.map { getString(it.titleRes) }.toTypedArray()
        val checkedItems = BooleanArray(filterOptions.size) { i ->
            filterOptions[i] == currentFilter
        }

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(getString(R.string.filter_insights))
            .setSingleChoiceItems(filterNames, currentFilter.ordinal) { dialog, which ->
                selectFilter(filterOptions[which])
                applyFilter(filterOptions[which])
                dialog.dismiss()
            }
            .setNegativeButton(getString(R.string.action_cancel), null)
            .show()
    }

    // ========== UI UPDATES AND ANIMATIONS ==========

    private fun updateEmptyState(isEmpty: Boolean) {
        binding.layoutEmptyState.root.isVisible = isEmpty && !isRefreshing
        binding.recyclerViewInsights.isVisible = !isEmpty

        if (isEmpty && !isRefreshing) {
            // Animate empty state appearance
            binding.layoutEmptyState.root.apply {
                alpha = 0f
                animate()
                    .alpha(1f)
                    .setDuration(ANIMATION_DURATION)
                    .setInterpolator(DecelerateInterpolator())
                    .start()
            }
        }
    }

    private fun updateInsightsSummary(insights: List<SleepInsight>) {
        val highPriorityCount = insights.count { it.priority <= 1 }
        val newInsightsCount = insights.count {
            System.currentTimeMillis() - it.timestamp < 24 * 60 * 60 * 1000L
        }

        // Update summary in toolbar subtitle
        binding.toolbar.subtitle = when {
            highPriorityCount > 0 -> getString(
                R.string.insights_summary_priority,
                insights.size,
                highPriorityCount
            )
            newInsightsCount > 0 -> getString(
                R.string.insights_summary_new,
                insights.size,
                newInsightsCount
            )
            else -> resources.getQuantityString(
                R.plurals.insights_count,
                insights.size,
                insights.size
            )
        }
    }

    private fun animateNewInsights(insights: List<SleepInsight>) {
        // Only animate if we have a reasonable number of new insights
        val newInsights = insights.filter { insight ->
            System.currentTimeMillis() - insight.timestamp < 60000L // Last minute
        }

        if (newInsights.size in 1..5) {
            // Animate the RecyclerView items
            binding.recyclerViewInsights.scheduleLayoutAnimation()
        }
    }

    private fun updateInsightGenerationAvailability(isTracking: Boolean) {
        binding.fabGenerateInsights.isVisible = !isTracking

        if (isTracking) {
            binding.fabGenerateInsights.animate()
                .scaleX(0f)
                .scaleY(0f)
                .alpha(0f)
                .setDuration(200L)
                .start()
        } else {
            binding.fabGenerateInsights.animate()
                .scaleX(1f)
                .scaleY(1f)
                .alpha(1f)
                .setDuration(200L)
                .start()
        }
    }

    private fun updatePerformanceIndicators(analytics: AIAnalyticsReport) {
        // Update performance indicators in the UI
        val performanceScore = analytics.overallMetrics.efficiency

        // Could show a subtle performance indicator
        val color = when {
            performanceScore >= 0.8f -> ContextCompat.getColor(requireContext(), R.color.performance_excellent)
            performanceScore >= 0.6f -> ContextCompat.getColor(requireContext(), R.color.performance_good)
            else -> ContextCompat.getColor(requireContext(), R.color.performance_poor)
        }

        // Apply performance color to some UI element (like toolbar background tint)
        binding.toolbar.setBackgroundColor(color)
    }

    private fun updateEngagementIndicators(metrics: UserEngagementMetrics) {
        // Update engagement-based UI elements
        // Could adjust UI recommendations based on engagement patterns
    }

    // ========== ANALYTICS AND TRACKING ==========

    private fun trackInsightVisibility() {
        val layoutManager = binding.recyclerViewInsights.layoutManager as LinearLayoutManager
        val firstVisible = layoutManager.findFirstVisibleItemPosition()
        val lastVisible = layoutManager.findLastVisibleItemPosition()

        for (position in firstVisible..lastVisible) {
            if (position >= 0 && position < insightsAdapter.itemCount) {
                val insight = insightsAdapter.getInsightAt(position)
                insight?.let {
                    if (visibleInsights.add(it.id)) {
                        insightsViewModel.trackInsightImpression(
                            insightId = it.id,
                            position = position,
                            context = mapOf(
                                "filter" to currentFilter.name,
                                "total_count" to insightsAdapter.itemCount
                            )
                        )
                    }
                }
            }
        }
    }

    private fun trackInsightImpressions(insights: List<SleepInsight>) {
        insights.forEachIndexed { index, insight ->
            insightsViewModel.trackInsightImpression(
                insightId = insight.id,
                position = index,
                context = mapOf(
                    "filter" to currentFilter.name,
                    "total_count" to insights.size,
                    "load_type" to if (isRefreshing) "refresh" else "load"
                )
            )
        }
    }

    // ========== NAVIGATION ==========

    private fun showInsightDetails(insight: SleepInsight) {
        // Create and show insight details dialog/fragment
        InsightDetailsDialogFragment.newInstance(insight)
            .show(parentFragmentManager, "insight_details")
    }

    private fun navigateToInsightSettings() {
        // Navigate to insight settings
        // Implementation depends on navigation architecture
    }

    private fun navigateToSleepTracking() {
        // Navigate back to sleep tracking
        requireActivity().onBackPressed()
    }

    private fun reportInsightIssue(insight: SleepInsight) {
        // Show issue reporting dialog
        IssueReportDialogFragment.newInstance(insight.id)
            .show(parentFragmentManager, "report_issue")
    }

    // ========== ACCESSIBILITY ==========

    private fun setupAccessibilityGestures() {
        // Setup accessibility features
        binding.recyclerViewInsights.contentDescription = getString(R.string.insights_list_description)

        // Add accessibility actions
        binding.fabGenerateInsights.contentDescription = getString(R.string.generate_insights_description)
    }

    // ========== UTILITY METHODS ==========

    private fun shouldRefreshInsights(): Boolean {
        return System.currentTimeMillis() - lastRefreshTime > INSIGHTS_REFRESH_INTERVAL
    }

    private fun formatInsightTimestamp(timestamp: Long): String {
        val formatter = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())
        return formatter.format(Date(timestamp))
    }
}

// ========== SUPPORTING CLASSES ==========

/**
 * Filter options for insights
 */
enum class InsightFilter(val titleRes: Int, val iconRes: Int) {
    ALL(R.string.filter_all, R.drawable.ic_all),
    HIGH_PRIORITY(R.string.filter_high_priority, R.drawable.ic_priority_high),
    RECENT(R.string.filter_recent, R.drawable.ic_recent),
    ACTIONABLE(R.string.filter_actionable, R.drawable.ic_action),
    QUALITY(R.string.filter_quality, R.drawable.ic_quality),
    DURATION(R.string.filter_duration, R.drawable.ic_duration),
    MOVEMENT(R.string.filter_movement, R.drawable.ic_movement),
    ENVIRONMENT(R.string.filter_environment, R.drawable.ic_environment)
}

/**
 * Insight interaction tracking
 */
enum class InsightInteractionType {
    CLICKED,
    SHARED,
    DISMISSED,
    FEEDBACK_POSITIVE,
    FEEDBACK_NEGATIVE,
    DETAILS_VIEWED,
    ACTION_TAKEN
}

/**
 * Feedback types for insights
 */
enum class InsightFeedback {
    HELPFUL,
    NOT_HELPFUL,
    IRRELEVANT,
    INACCURATE,
    TOO_TECHNICAL,
    TOO_SIMPLE
}

/**
 * Error types for insight operations
 */
enum class InsightErrorType {
    NETWORK_ERROR,
    AI_SERVICE_ERROR,
    INSUFFICIENT_DATA,
    QUOTA_EXCEEDED,
    PROCESSING_ERROR,
    UNKNOWN_ERROR
}

/**
 * Error state for insights
 */
data class InsightError(
    val type: InsightErrorType,
    val title: String,
    val message: String,
    val isRetryable: Boolean = true,
    val details: String? = null
)

/**
 * Progress tracking for insight generation
 */
data class InsightGenerationProgress(
    val stage: GenerationStage,
    val progress: Float, // 0.0 to 1.0
    val estimatedTimeRemaining: Long,
    val completedInsights: Int = 0,
    val totalExpectedInsights: Int = 0,
    val currentOperation: String = ""
)

enum class GenerationStage(val displayName: String) {
    PREPARING("Preparing"),
    ANALYZING("Analyzing"),
    GENERATING("Generating"),
    PROCESSING("Processing"),
    FINALIZING("Finalizing"),
    COMPLETED("Completed")
}

/**
 * Engagement metrics for analytics
 */
data class EngagementMetrics(
    val timeSpent: Long = 0L,
    val interactionCount: Int = 0,
    val scrollDepth: Float = 0f,
    val actionsThaken: List<String> = emptyList()
)

// Extension properties for InsightGenerationType
val InsightGenerationType.displayName: String
    get() = when (this) {
        InsightGenerationType.POST_SESSION -> "Session Analysis"
        InsightGenerationType.DAILY_ANALYSIS -> "Daily Summary"
        InsightGenerationType.WEEKLY_SUMMARY -> "Weekly Report"
        InsightGenerationType.TREND_ANALYSIS -> "Trend Analysis"
        InsightGenerationType.PERSONALIZED_ANALYSIS -> "Personal Insights"
        InsightGenerationType.COMPARATIVE_ANALYSIS -> "Comparative Analysis"
        InsightGenerationType.PREDICTIVE_ANALYSIS -> "Predictive Insights"
        InsightGenerationType.EMERGENCY_ANALYSIS -> "Emergency Analysis"
        InsightGenerationType.PATTERN_RECOGNITION -> "Pattern Recognition"
        InsightGenerationType.GOAL_TRACKING -> "Goal Progress"
        InsightGenerationType.HEALTH_ASSESSMENT -> "Health Assessment"
        InsightGenerationType.RECOMMENDATION_ENGINE -> "Recommendations"
    }