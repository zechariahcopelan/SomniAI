package com.example.somniai

import android.Manifest
import android.animation.ValueAnimator
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.somniai.activities.ChartsActivity
import com.example.somniai.activities.HistoryActivity
import com.example.somniai.activities.SettingsActivity
import com.example.somniai.data.*
import com.example.somniai.databinding.ActivityMainBinding
import com.example.somniai.service.SleepTrackingService
import com.example.somniai.ui.theme.models.*
import com.example.somniai.ui.theme.ChartTheme
import com.example.somniai.viewmodel.MainViewModel
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.*
import java.util.*
import kotlin.math.*
import com.example.somniai.ui.theme.*
import com.example.somniai.data.PerformanceMetrics
import java.io.Serializable
import com.example.somniai.data.DataIntegrityStatus
import com.example.somniai.data.SleepPhase
import com.example.somniai.data.SleepTrend
import com.example.somniai.data.SleepSession
import com.example.somniai.data.SleepAnalytics

/**
 * Enhanced MainActivity with Advanced Analytics Integration
 *
 * Features:
 * - Real-time analytics visualization with mini charts
 * - Advanced UI models integration for rich display
 * - Comprehensive session tracking with live feedback
 * - Performance-optimized real-time updates
 * - Accessibility-enhanced interface
 * - Smooth animations and micro-interactions
 * - Enterprise-grade error handling and recovery
 * - Deep integration with analytics ecosystem
 * - AI-powered insights display system
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var viewModel: MainViewModel

    // Enhanced tracking state
    private var isTracking = false
    private var currentSensorStatus: SensorStatus? = null
    private var currentLiveMetrics: LiveSleepMetrics? = null
    private var currentSessionDisplay: SessionDisplayModel? = null

    // UI Enhancement Components
    private val uiUpdateHandler = Handler(Looper.getMainLooper())
    private val animationScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var qualityTrendChart: LineChart? = null
    private var realTimeDataPoints = mutableListOf<Entry>()
    private var updateCounter = 0

    // Performance Tracking
    private var lastUpdateTime = 0L
    private val updateThrottleMs = 1000L // Throttle UI updates to 1 second

    // Analytics Display State
    private var currentAnalyticsDisplay: AnalyticsDisplayState = AnalyticsDisplayState.OVERVIEW
    private var lastQualityScore = 0f
    private var qualityTrendData = mutableListOf<Float>()

    // Insights System State
    private var currentCategorizedInsights: CategorizedInsights? = null
    private var lastInsightsUpdate = 0L

    // Data classes for insights categorization
    data class CategorizedInsights(
        val primary: SleepInsight?,
        val pattern: SleepInsight?,
        val recommendation: SleepInsight?,
        val trend: SleepInsight?,
        val all: List<SleepInsight>
    )

    enum class InsightType {
        QUALITY_IMPROVEMENT,
        PATTERN_ANALYSIS,
        RECOMMENDATION,
        TREND_ANALYSIS,
        ENVIRONMENTAL,
        BEHAVIORAL
    }

    enum class TrendDirection {
        IMPROVING,
        DECLINING,
        STABLE,
        INSUFFICIENT_DATA
    }

    // Broadcast receiver for enhanced sensor updates
    private val sensorUpdateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                SleepTrackingService.BROADCAST_SENSOR_UPDATE -> {
                    handleEnhancedSensorUpdate(intent)
                }
                SleepTrackingService.BROADCAST_SESSION_COMPLETE -> {
                    val sessionAnalytics = intent.getSerializableExtra(
                        SleepTrackingService.EXTRA_COMPLETED_SESSION
                    )
                    val completedSession = intent.getParcelableExtra<SleepSession>(
                        "completed_sleep_session"
                    )
                    handleEnhancedSessionComplete(sessionAnalytics, completedSession)
                }
                SleepTrackingService.BROADCAST_PHASE_CHANGE -> {
                    intent?.getParcelableExtra<PhaseTransition>(
                        SleepTrackingService.EXTRA_PHASE_TRANSITION
                    ) as? PhaseTransition?.let { phaseTransition: PhaseTransition ->
                        handleEnhancedPhaseChange(phaseTransition)
                    }
                }
                SleepTrackingService.BROADCAST_DATA_SAVED -> {
                    handleEnhancedDataSaved(intent)
                }
                SleepTrackingService.BROADCAST_PERFORMANCE_UPDATE -> {
                    val performanceMetrics = intent.getSerializableExtra("performance_metrics") as? Map<String, Any>
                    performanceMetrics?.let { handlePerformanceUpdate(it) }
                }
            }
        }
    }

    // Permission launcher for comprehensive permissions
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val microphoneGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
        val notificationGranted = permissions[Manifest.permission.POST_NOTIFICATIONS] ?: false

        if (microphoneGranted) {
            initializeEnhancedTracking()
            showEnhancedSuccessMessage("Permissions granted! Ready for advanced sleep tracking.")
        } else {
            showEnhancedErrorMessage(
                ErrorStateDisplay(
                    title = "Permission Required",
                    message = "Microphone access is required for sleep tracking",
                    icon = android.R.drawable.ic_btn_speak_now,
                    //icon = R.drawable.ic_microphone_off,
                    canRetry = true,
                    actionItems = listOf(
                        ErrorActionDisplay(
                            title = "Grant Permission",
                            description = "Allow microphone access in settings",
                            action = "open_settings",
                            //icon = R.drawable.ic_settings
                            icon = android.R.drawable.ic_menu_preferences
                        )
                    )
                )
            )
            handlePermissionDenied()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize ViewModel
        viewModel = ViewModelProvider(this)[MainViewModel::class.java]

        setupEnhancedUI()
        setupEnhancedObservers()
        registerEnhancedBroadcastReceivers()
        checkEnhancedPermissions()
        initializeRealTimeCharts()
        initializeInsightsSystem()
    }

    // ========== ENHANCED UI SETUP ==========

    private fun setupEnhancedUI() {
        // Set up toolbar with enhanced styling
        setSupportActionBar(binding.toolbar)
        supportActionBar?.title = "SomniAI Sleep Tracker"

        // Set up enhanced tracking button with animations
        binding.trackingButton.setOnClickListener {
            animateButtonPress(binding.trackingButton) {
                if (isTracking) {
                    stopEnhancedSleepTracking()
                } else {
                    startEnhancedSleepTracking()
                }
            }
        }

        // Set up enhanced navigation buttons with analytics integration
        binding.historyButton.setOnClickListener {
            animateButtonPress(binding.historyButton) {
                val intent = Intent(this, HistoryActivity::class.java)
                // Pass current analytics context
                currentSessionDisplay?.let { session ->
                    intent.putExtra("current_session_context", session.sessionData.id)
                }
                startActivity(intent)
            }
        }

        binding.chartsButton.setOnClickListener {
            animateButtonPress(binding.chartsButton) {
                val intent = Intent(this, ChartsActivity::class.java)
                // Pass real-time chart data
                intent.putExtra("realtime_quality_data", qualityTrendData.toFloatArray())
                startActivity(intent)
            }
        }

        binding.settingsButton.setOnClickListener {
            animateButtonPress(binding.settingsButton) {
                startActivity(Intent(this, SettingsActivity::class.java))
            }
        }

        // Set up analytics view toggle
        binding.analyticsToggleButton?.setOnClickListener {
            toggleAnalyticsView()
        }

        // Set up refresh functionality with enhanced feedback
        //setupEnhancedRefreshFunctionality()

        // Initialize UI state with animations
        updateEnhancedTrackingUI()
        updateEnhancedSensorStatusUI()
        showEnhancedLoadingIndicators()

        // Set up gesture handlers for advanced interactions
        //setupGestureHandlers()
    }

    private fun setupEnhancedRefreshFunctionality() {
        binding.refreshButton?.setOnClickListener { button ->
            // Animate refresh button
            animateRefreshButton(button) {
                refreshAllEnhancedData()
            }
        }

        // Add swipe-to-refresh if using SwipeRefreshLayout
        //binding.swipeRefreshLayout?.setOnRefreshListener {
        //    refreshAllEnhancedData()
        //}
    //}

    //private fun setupGestureHandlers() {
        // Add gesture support for analytics cards
      //  binding.qualityCard?.setOnClickListener {
        //    showDetailedQualityAnalysis()
        //}

        //binding.efficiencyCard?.setOnClickListener {
        //    showDetailedEfficiencyAnalysis()
        //}

        //binding.analyticsCard?.setOnClickListener {
        //    showComprehensiveAnalytics()
        //}
    }

    private fun initializeRealTimeCharts() {
        // Initialize quality trend mini chart
        qualityTrendChart = binding.qualityTrendMiniChart
        qualityTrendChart?.let { chart ->
            chart.applySleepTheme(this, SleepChartType.QUALITY_TREND)

            // Configure for real-time updates
            chart.apply {
                setDrawGridBackground(false)
                description.isEnabled = false
                legend.isEnabled = false
                setTouchEnabled(false)
                isDragEnabled = false
                setScaleEnabled(false)
                setPinchZoom(false)

                // Set up for streaming data
                setMaxVisibleValueCount(20)
                setDrawMarkers(false)
            }
        }
    }

    private fun initializeInsightsSystem() {
        // Initialize insights display system
        hideInsightsSection()

        // Set up default insight displays
        showDefaultInsightMessages()

        // Initialize insights update tracking
        lastInsightsUpdate = System.currentTimeMillis()
    }

    // ========== ENHANCED OBSERVERS ==========

    private fun setupEnhancedObservers() {
        // Observe initialization state with enhanced feedback
        viewModel.isInitialized.observe(this) { initialized ->
            if (initialized) {
                hideEnhancedLoadingIndicators()
                updateEnhancedStatisticsDisplay()
                showInitializationSuccess()
            } else {
                showEnhancedLoadingIndicators()
            }
        }

        // Observe loading states with granular feedback
        viewModel.isLoadingStatistics.observe(this) { loading ->
            updateEnhancedLoadingIndicator("statistics", loading)
        }

        viewModel.isLoadingSessions.observe(this) { loading ->
            updateEnhancedLoadingIndicator("sessions", loading)
        }

        viewModel.isLoadingAnalytics.observe(this) { loading ->
            updateEnhancedLoadingIndicator("analytics", loading)
        }

        // Observe tracking state with enhanced UI updates
        viewModel.isTracking.observe(this) { tracking ->
            isTracking = tracking
            updateEnhancedTrackingUI()
            if (tracking) {
                startRealTimeUpdates()
            } else {
                stopRealTimeUpdates()
            }
        }

        // Observe enhanced sleep statistics
        viewModel.totalSessions.observe(this) { total ->
            updateEnhancedSessionCount(total)
        }

        viewModel.averageSleepDuration.observe(this) { duration ->
            updateDurationDisplay(duration)
        }

        viewModel.averageSleepQuality.observe(this) { quality ->
            updateQualityDisplay(quality)
        }

        viewModel.averageSleepEfficiency.observe(this) { efficiency ->
            updateEnhancedEfficiencyDisplay(efficiency)
        }

        // Observe enhanced analytics data
        viewModel.sleepAnalytics.observe(this) { analytics ->
            updateEnhancedAnalyticsDisplay(analytics)
        }

        viewModel.sleepTrends.observe(this) { trends ->
            updateTrendsDisplay(trends)
        }

        viewModel.qualityReport.observe(this) { report ->
            updateEnhancedQualityReportDisplay(report)
        }

        viewModel.performanceComparison.observe(this) { comparison ->
            updateComparisonDisplay(comparison)
        }

        viewModel.insights.observe(this) { insights ->
            updateEnhancedInsightsDisplay(insights)
        }

        // Observe sensor status with real-time updates
        viewModel.sensorStatus.observe(this) { status ->
            currentSensorStatus = status
            updateEnhancedSensorStatusUI()
        }

        // Observe live metrics with enhanced visualization
        viewModel.liveMetrics.observe(this) { metrics ->
            currentLiveMetrics = metrics
            updateEnhancedLiveMetricsUI()
            updateRealTimeChart(metrics)
        }

        // Observe error states with enhanced error handling
        viewModel.errorMessage.observe(this) { error ->
            error?.let {
                showEnhancedErrorMessage(
                    ErrorStateDisplay(
                        title = "Error",
                        message = it,
                        icon = android.R.drawable.ic_dialog_alert,
                        canRetry = true
                    )
                )
                viewModel.clearError()
            }
        }

        // Observe data integrity with visual feedback
        viewModel.dataIntegrity.observe(this) { integrity ->
            updateEnhancedDataIntegrityIndicator(integrity)
        }

        // Observe performance metrics
        viewModel.performanceMetrics.observe(this) { metrics ->
           // updatePerformanceDisplay(metrics)
        }
    }

    private fun registerEnhancedBroadcastReceivers() {
        val intentFilter = IntentFilter().apply {
            addAction(SleepTrackingService.BROADCAST_SENSOR_UPDATE)
            addAction(SleepTrackingService.BROADCAST_SESSION_COMPLETE)
            addAction(SleepTrackingService.BROADCAST_PHASE_CHANGE)
            addAction(SleepTrackingService.BROADCAST_DATA_SAVED)
            addAction(SleepTrackingService.BROADCAST_PERFORMANCE_UPDATE)
        }
        LocalBroadcastManager.getInstance(this)
            .registerReceiver(sensorUpdateReceiver, intentFilter)
    }

    private fun checkEnhancedPermissions() {
        val permissionsToRequest = mutableListOf<String>()

        // Check microphone permission
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            permissionsToRequest.add(Manifest.permission.RECORD_AUDIO)
        }

        // Check notification permission (Android 13+)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        if (permissionsToRequest.isNotEmpty()) {
            showEnhancedPermissionRationale {
                requestPermissionLauncher.launch(permissionsToRequest.toTypedArray())
            }
        } else {
            initializeEnhancedTracking()
        }
    }

    // ========== INSIGHTS DISPLAY SYSTEM ==========

    private fun updateEnhancedInsightsDisplay(insights: List<SleepInsight>) {
        // Update the comprehensive AI Insights section
        updateComprehensiveInsightsDisplay(insights)

        // Legacy insights counter for other parts of UI
        binding.insightsCountText?.let { textView ->
            textView.text = "${insights.size} insights"
        }

        if (insights.isNotEmpty()) {
            binding.insightsIndicator?.let { indicator ->
                animateViewVisibility(indicator, true)
            }

            // Show the highest priority insight in legacy section
            val highPriorityInsight = insights.minByOrNull { it.priority }
            highPriorityInsight?.let { insight ->
                binding.latestInsightText?.let { textView ->
                    textView.text = insight.title
                    animateViewVisibility(textView, true)
                }

                // Color code by priority with enhanced styling
                val priorityColor = when (insight.priority) {
                    1 -> R.color.accent_red
                    2 -> R.color.accent_amber
                    else -> R.color.accent_green
                }
                binding.insightsIndicator?.let { indicator ->
                    indicator.backgroundTintList = ContextCompat.getColorStateList(this, priorityColor)
                }

                // Add priority badge
                val priorityText = when (insight.priority) {
                    1 -> "🚨"
                    2 -> "⚠️"
                    else -> "💡"
                }
                binding.insightPriorityBadge?.let { badge ->
                    badge.text = priorityText
                }
            }
        } else {
            binding.insightsIndicator?.let { it.visibility = View.GONE }
            binding.latestInsightText?.let { it.visibility = View.GONE }
            binding.insightPriorityBadge?.let { it.visibility = View.GONE }
            hideInsightsSection()
        }
    }

    /**
     * Updates the comprehensive AI Insights section with detailed analysis
     */
    private fun updateComprehensiveInsightsDisplay(insights: List<SleepInsight>) {
        if (insights.isEmpty()) {
            hideInsightsSection()
            return
        }

        // Update insights timestamp
        updateInsightsTimestamp()

        // Categorize insights for better display
        val categorizedInsights = categorizeInsights(insights)
        currentCategorizedInsights = categorizedInsights

        // Update primary insight (highest confidence/priority)
        updatePrimaryInsightDisplay(categorizedInsights.primary)

        // Update pattern analysis insight
        updatePatternInsightDisplay(categorizedInsights.pattern)

        // Update recommendation insight
        updateRecommendationInsightDisplay(categorizedInsights.recommendation)

        // Update weekly trend analysis
        updateWeeklyTrendInsightDisplay(categorizedInsights.trend)

        // Set up click handlers for detailed views
        setupInsightClickHandlers(categorizedInsights)
    }

    private fun categorizeInsights(insights: List<SleepInsight>): CategorizedInsights {
        val sortedInsights = insights.sortedWith(
            compareBy<SleepInsight> { it.priority }.thenByDescending { it.confidence }
        )

        return CategorizedInsights(
            primary = sortedInsights.firstOrNull { it.type == InsightType.QUALITY_IMPROVEMENT }
                ?: sortedInsights.firstOrNull(),
            pattern = sortedInsights.firstOrNull { it.type == InsightType.PATTERN_ANALYSIS },
            recommendation = sortedInsights.firstOrNull { it.type == InsightType.RECOMMENDATION },
            trend = sortedInsights.firstOrNull { it.type == InsightType.TREND_ANALYSIS },
            all = sortedInsights
        )
    }

    private fun updateInsightsTimestamp() {
        val timeAgo = getTimeAgo(Date())
        binding.insightsUpdateTime?.text = "Updated $timeAgo"
        animateValueChange(binding.insightsUpdateTime)
        lastInsightsUpdate = System.currentTimeMillis()
    }

    private fun updatePrimaryInsightDisplay(insight: SleepInsight?) {
        insight?.let { primaryInsight ->
            animateViewVisibility(binding.primaryInsightCard, true)

            // Update title and description
            binding.primaryInsightTitle?.text = primaryInsight.title
            binding.primaryInsightDescription?.text = primaryInsight.description

            // Update confidence percentage
            val confidencePercent = (primaryInsight.confidence * 100).toInt()
            binding.primaryInsightConfidence?.text = "$confidencePercent%"

            // Update icon based on insight type
            val iconRes = getInsightIcon(primaryInsight.type, primaryInsight.priority)
            binding.primaryInsightIcon?.setImageResource(iconRes)

            // Color code by confidence and priority
            val insightColor = getInsightColor(primaryInsight.confidence, primaryInsight.priority)
            binding.primaryInsightIcon?.imageTintList = ContextCompat.getColorStateList(this, insightColor)
            binding.primaryInsightConfidence?.setTextColor(ContextCompat.getColor(this, insightColor))

            // Update stroke color of card
            binding.primaryInsightCard?.strokeColor = ContextCompat.getColor(this, insightColor)

            // Animate insight update
            animateInsightUpdate(binding.primaryInsightCard)

        } ?: run {
            // Show default primary insight when no specific insights available
            showDefaultPrimaryInsight()
        }
    }

    private fun updatePatternInsightDisplay(insight: SleepInsight?) {
        insight?.let { patternInsight ->
            animateViewVisibility(binding.patternInsightCard, true)

            binding.patternInsightTitle?.text = patternInsight.title
            binding.patternInsightText?.text = patternInsight.description

            // Color code pattern insight (typically amber for patterns)
            val patternColor = R.color.accent_amber
            updateInsightCardStyling(binding.patternInsightCard, patternColor)

            animateInsightUpdate(binding.patternInsightCard)

        } ?: run {
            showDefaultPatternInsight()
        }
    }

    private fun updateRecommendationInsightDisplay(insight: SleepInsight?) {
        insight?.let { recommendationInsight ->
            animateViewVisibility(binding.recommendationInsightCard, true)

            binding.recommendationInsightTitle?.text = recommendationInsight.title
            binding.recommendationInsightText?.text = recommendationInsight.description

            // Color code recommendation insight (typically blue for tips)
            val recommendationColor = R.color.accent_blue
            updateInsightCardStyling(binding.recommendationInsightCard, recommendationColor)

            animateInsightUpdate(binding.recommendationInsightCard)

        } ?: run {
            showDefaultRecommendationInsight()
        }
    }

    private fun updateWeeklyTrendInsightDisplay(trendInsight: SleepInsight?) {
        trendInsight?.let { insight ->
            animateViewVisibility(binding.weeklyTrendCard, true)

            // Update main trend description
            binding.weeklyTrendDescription?.text = insight.description

            // Extract trend percentage from insight data
            val trendPercentage = extractTrendPercentage(insight)
            val isPositiveTrend = trendPercentage >= 0

            // Update trend indicator
            binding.trendPercentage?.text = if (isPositiveTrend) "+${trendPercentage}%" else "$trendPercentage%"
            binding.trendIcon?.setImageResource(
                if (isPositiveTrend) R.drawable.ic_arrow_up else R.drawable.ic_arrow_down
            )

            // Color code trend
            val trendColor = if (isPositiveTrend) R.color.accent_green else R.color.accent_red
            binding.trendPercentage?.setTextColor(ContextCompat.getColor(this, trendColor))
            binding.trendIcon?.imageTintList = ContextCompat.getColorStateList(this, trendColor)

            // Update mini progress indicators with sample data
            updateMiniProgressIndicators()

            animateInsightUpdate(binding.weeklyTrendCard)

        } ?: run {
            showDefaultWeeklyTrend()
        }
    }

    private fun setupInsightClickHandlers(categorizedInsights: CategorizedInsights) {
        // Primary insight click handler
        binding.primaryInsightCard?.setOnClickListener {
            showDetailedInsightView(categorizedInsights.primary, "Primary Insight")
        }

        // Pattern insight click handler
        binding.patternInsightCard?.setOnClickListener {
            showDetailedInsightView(categorizedInsights.pattern, "Pattern Analysis")
        }

        // Recommendation insight click handler
        binding.recommendationInsightCard?.setOnClickListener {
            showDetailedInsightView(categorizedInsights.recommendation, "Recommendation")
        }

        // Weekly trend click handler
        binding.weeklyTrendCard?.setOnClickListener {
            showDetailedInsightView(categorizedInsights.trend, "Weekly Trend")
        }
    }

    private fun hideInsightsSection() {
        animateViewVisibility(binding.primaryInsightCard, false)
        animateViewVisibility(binding.patternInsightCard, false)
        animateViewVisibility(binding.recommendationInsightCard, false)
        animateViewVisibility(binding.weeklyTrendCard, false)
    }

    private fun showDefaultInsightMessages() {
        // Show placeholder messages when no insights are available
        showDefaultPrimaryInsight()
        showDefaultPatternInsight()
        showDefaultRecommendationInsight()
        showDefaultWeeklyTrend()
    }

    private fun showDefaultPrimaryInsight() {
        animateViewVisibility(binding.primaryInsightCard, true)
        binding.primaryInsightTitle?.text = "Building Your Profile"
        binding.primaryInsightDescription?.text = "Track a few more sleep sessions to unlock personalized insights powered by AI analysis."
        binding.primaryInsightConfidence?.text = "..."
        binding.primaryInsightIcon?.setImageResource(R.drawable.ic_ai_brain)
        binding.primaryInsightIcon?.imageTintList = ContextCompat.getColorStateList(this, R.color.text_secondary)
    }

    private fun showDefaultPatternInsight() {
        animateViewVisibility(binding.patternInsightCard, true)
        binding.patternInsightTitle?.text = "Learning Patterns"
        binding.patternInsightText?.text = "We're analyzing your sleep behavior to identify helpful patterns and trends."
    }

    private fun showDefaultRecommendationInsight() {
        animateViewVisibility(binding.recommendationInsightCard, true)
        binding.recommendationInsightTitle?.text = "Tips Coming Soon"
        binding.recommendationInsightText?.text = "Personalized recommendations will appear here as we learn about your sleep habits."
    }

    private fun showDefaultWeeklyTrend() {
        animateViewVisibility(binding.weeklyTrendCard, true)
        binding.weeklyTrendDescription?.text = "Your weekly sleep trends will appear here once you've completed several sleep tracking sessions."
        binding.trendPercentage?.text = "..."
        binding.trendIcon?.setImageResource(R.drawable.ic_trending_up)
        binding.trendIcon?.imageTintList = ContextCompat.getColorStateList(this, R.color.text_secondary)
    }

    private fun updateMiniProgressIndicators() {
        // Update mini progress bars with current session data or defaults
        val durationProgress = currentLiveMetrics?.let { metrics ->
            ((metrics.sessionDuration / (8 * 60 * 60 * 1000f)) * 100).toInt().coerceAtMost(100)
        } ?: 0

        val deepSleepProgress = 85 // Would come from actual deep sleep analysis
        val efficiencyProgress = currentLiveMetrics?.sleepEfficiency?.toInt() ?: 0

        // Update progress bars (these would need to be implemented based on actual progress bar views)
        // binding.durationProgressBar?.progress = durationProgress
        // binding.deepSleepProgressBar?.progress = deepSleepProgress
        // binding.efficiencyProgressBar?.progress = efficiencyProgress
    }

    private fun showDetailedInsightView(insight: SleepInsight?, title: String) {
        insight?.let {
            // For now, show a detailed toast. In production, this would open a detailed insight view
            val detailMessage = buildString {
                appendLine("$title - ${it.title}")
                appendLine()
                appendLine(it.description)
                appendLine()
                appendLine("Confidence: ${(it.confidence * 100).toInt()}%")
                appendLine("Priority: ${getPriorityText(it.priority)}")
                if (it.actionableRecommendations.isNotEmpty()) {
                    appendLine()
                    appendLine("Recommendations:")
                    it.actionableRecommendations.forEach { rec ->
                        appendLine("• $rec")
                    }
                }
            }
            Toast.makeText(this, detailMessage, Toast.LENGTH_LONG).show()
        }
    }

    // ========== INSIGHT HELPER METHODS ==========

    private fun getInsightIcon(type: InsightType, priority: Int): Int {
        return when (type) {
            InsightType.QUALITY_IMPROVEMENT -> when (priority) {
                1 -> R.drawable.ic_insight_quality_urgent
                2 -> R.drawable.ic_insight_quality_moderate
                else -> R.drawable.ic_insight_quality_good
            }
            InsightType.PATTERN_ANALYSIS -> R.drawable.ic_pattern_analysis
            InsightType.RECOMMENDATION -> R.drawable.ic_lightbulb
            InsightType.TREND_ANALYSIS -> R.drawable.ic_trending_up
            InsightType.ENVIRONMENTAL -> R.drawable.ic_environment
            InsightType.BEHAVIORAL -> R.drawable.ic_behavior_analysis
        }
    }

    private fun getInsightColor(confidence: Float, priority: Int): Int {
        return when {
            priority == 1 -> R.color.accent_red
            priority == 2 -> R.color.accent_amber
            confidence >= 0.8f -> R.color.accent_green
            confidence >= 0.6f -> R.color.accent_blue
            else -> R.color.text_secondary
        }
    }

    private fun updateInsightCardStyling(cardView: View?, colorRes: Int) {
        cardView?.let { card ->
            // Update card styling based on insight type
            // This would need to be implemented based on your actual card view structure
        }
    }

    private fun animateInsightUpdate(cardView: View?) {
        cardView?.let { card ->
            card.animate()
                .scaleX(1.02f)
                .scaleY(1.02f)
                .setDuration(150)
                .withEndAction {
                    card.animate()
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(150)
                        .start()
                }
                .start()
        }
    }

    private fun extractTrendPercentage(insight: SleepInsight): Int {
        // Extract percentage from insight description or metadata
        // This is a simplified implementation - in production, this would come from structured data
        val description = insight.description
        val percentageRegex = """([+-]?\d+)%""".toRegex()
        val match = percentageRegex.find(description)
        return match?.groupValues?.get(1)?.toIntOrNull() ?: 0
    }

    private fun getPriorityText(priority: Int): String {
        return when (priority) {
            1 -> "High"
            2 -> "Medium"
            else -> "Low"
        }
    }

    private fun getTimeAgo(date: Date): String {
        val now = System.currentTimeMillis()
        val timeDiff = now - date.time

        return when {
            timeDiff < 60 * 1000 -> "just now"
            timeDiff < 60 * 60 * 1000 -> "${timeDiff / (60 * 1000)}m ago"
            timeDiff < 24 * 60 * 60 * 1000 -> "${timeDiff / (60 * 60 * 1000)}h ago"
            else -> "${timeDiff / (24 * 60 * 60 * 1000)}d ago"
        }
    }

    // ========== ENHANCED TRACKING MANAGEMENT ==========

    private fun initializeEnhancedTracking() {
        try {
            // Check if tracking service is already running
            isTracking = SleepTrackingService.isServiceRunning(this)
            updateEnhancedTrackingUI()

            // Wait for ViewModel to initialize, then load data
            if (viewModel.isInitialized.value == true) {
                loadEnhancedInitialData()
            }

            // Request current status from service if running
            if (isTracking) {
                requestEnhancedServiceStatus()
            }

            showInitializationSuccess()
        } catch (e: Exception) {
            showEnhancedErrorMessage(
                ErrorStateDisplay(
                    title = "Initialization Failed",
                    message = "Failed to initialize tracking: ${e.message}",
                    icon = R.drawable.ic_error,
                    canRetry = true
                )
            )
        }
    }

    private fun loadEnhancedInitialData() {
        // Show loading with detailed progress
        showEnhancedLoadingIndicators()

        // Load data through ViewModel with progress tracking
        viewModel.refreshAllData()
    }

    private fun refreshAllEnhancedData() {
        // Animate refresh process
        animateDataRefresh {
            viewModel.refreshAllData()
        }
    }

    private fun startEnhancedSleepTracking() {
        if (!hasRequiredPermissions()) {
            checkEnhancedPermissions()
            return
        }

        try {
            // Create enhanced sensor settings
            val sensorSettings = SensorSettings(
                movementThreshold = 2.0f,
                noiseThreshold = 1000,
                enableMovementDetection = true,
                enableNoiseDetection = true,
                enableSmartFiltering = true,
                autoAdjustSensitivity = true
            )

            // Start tracking through ViewModel
            viewModel.startTracking(sensorSettings)

            // Start the enhanced sleep tracking service
            val serviceIntent = Intent(this, SleepTrackingService::class.java).apply {
                action = SleepTrackingService.ACTION_START_TRACKING
                putExtra(SleepTrackingService.EXTRA_SENSOR_SETTINGS, sensorSettings)
            }

            startForegroundService(serviceIntent)

            // Initialize real-time tracking state
            initializeRealTimeTracking()

            showEnhancedSuccessMessage("Advanced sleep tracking started!")

        } catch (e: SecurityException) {
            showEnhancedErrorMessage(
                ErrorStateDisplay(
                    title = "Permission Denied",
                    message = "Microphone access is required for sleep tracking",
                    //icon = R.drawable.ic_microphone_off,
                    icon = R.drawable.ic_microphone_off,
                    canRetry = true
                )
            )
            checkEnhancedPermissions()
        } catch (e: Exception) {
            showEnhancedErrorMessage(
                ErrorStateDisplay(
                    title = "Tracking Start Failed",
                    message = "Failed to start tracking: ${e.message}",
                    icon = R.drawable.ic_error,
                    canRetry = true
                )
            )
        }
    }

    private fun stopEnhancedSleepTracking() {
        try {
            // Animate stopping process
            animateTrackingStop {
                // Stop tracking through ViewModel
                viewModel.stopTracking()

                val serviceIntent = Intent(this, SleepTrackingService::class.java).apply {
                    action = SleepTrackingService.ACTION_STOP_TRACKING
                }

                startService(serviceIntent)

                // Clear real-time data
                clearRealTimeTrackingData()

                showEnhancedSuccessMessage("Sleep tracking completed successfully!")
            }

        } catch (e: Exception) {
            showEnhancedErrorMessage(
                ErrorStateDisplay(
                    title = "Stop Tracking Failed",
                    message = "Failed to stop tracking: ${e.message}",
                    icon = R.drawable.ic_error,
                    canRetry = true
                )
            )
        }
    }

    private fun initializeRealTimeTracking() {
        // Clear previous data
        realTimeDataPoints.clear()
        qualityTrendData.clear()
        updateCounter = 0

        // Start real-time updates
        startRealTimeUpdates()
    }

    private fun clearRealTimeTrackingData() {
        // Stop real-time updates
        stopRealTimeUpdates()

        // Clear data
        currentSensorStatus = null
        currentLiveMetrics = null
        currentSessionDisplay = null
        realTimeDataPoints.clear()
        qualityTrendData.clear()

        // Update UI
        updateEnhancedSensorStatusUI()
        updateEnhancedLiveMetricsUI()
        clearRealTimeChart()
    }

    // ========== REAL-TIME UPDATES ==========

    private fun startRealTimeUpdates() {
        // Start periodic UI updates for smooth real-time feedback
        val updateRunnable = object : Runnable {
            override fun run() {
                if (isTracking) {
                    updateRealTimeDisplays()
                    uiUpdateHandler.postDelayed(this, updateThrottleMs)
                }
            }
        }
        uiUpdateHandler.post(updateRunnable)
    }

    private fun stopRealTimeUpdates() {
        uiUpdateHandler.removeCallbacksAndMessages(null)
    }

    private fun updateRealTimeDisplays() {
        currentSensorStatus?.let { status ->
            updateEnhancedSensorStatusUI()
        }

        currentLiveMetrics?.let { metrics ->
            updateEnhancedLiveMetricsUI()
            updateRealTimeChart(metrics)
        }
    }

    private fun updateRealTimeChart(metrics: LiveSleepMetrics?) {
        if (metrics == null) return

        try {
            // Add new data point
            val qualityEstimate = calculateRealTimeQuality(metrics)
            qualityTrendData.add(qualityEstimate)

            // Keep only last 20 points for performance
            if (qualityTrendData.size > 20) {
                qualityTrendData.removeAt(0)
            }

            // Update chart data
            val entries = qualityTrendData.mapIndexed { index, quality ->
                Entry(index.toFloat(), quality)
            }

            val dataSet = LineDataSet(entries, "Quality").apply {
                color = ChartTheme.getQualityColor(qualityEstimate)
                lineWidth = 2f
                setDrawCircles(false)
                setDrawValues(false)
                mode = LineDataSet.Mode.CUBIC_BEZIER
                cubicIntensity = 0.2f
            }

            val lineData = LineData(dataSet)
            qualityTrendChart?.data = lineData
            qualityTrendChart?.invalidate()

        } catch (e: Exception) {
            // Handle chart update errors gracefully
        }
    }

    private fun calculateRealTimeQuality(metrics: LiveSleepMetrics): Float {
        // Simplified real-time quality calculation
        val efficiencyFactor = metrics.sleepEfficiency / 100f
        val restlessnessFactor = 1f - (metrics.totalRestlessness / 10f).coerceIn(0f, 1f)
        val phaseFactor = when (metrics.currentPhase) {
            SleepPhase.DEEP_SLEEP -> 1.0f
            SleepPhase.REM_SLEEP -> 0.9f
            SleepPhase.LIGHT_SLEEP -> 0.7f
            SleepPhase.AWAKE -> 0.3f
            SleepPhase.UNKNOWN -> 0.5f
        }

        return ((efficiencyFactor + restlessnessFactor + phaseFactor) / 3f * 10f).coerceIn(0f, 10f)
    }

    private fun clearRealTimeChart() {
        qualityTrendChart?.clear()
        qualityTrendChart?.invalidate()
    }

    // ========== ENHANCED BROADCAST HANDLING ==========

    private fun handleEnhancedSensorUpdate(intent: Intent) {
        val currentTime = System.currentTimeMillis()

        // Throttle updates for performance
        if (currentTime - lastUpdateTime < updateThrottleMs) {
            return
        }
        lastUpdateTime = currentTime

        // Handle sensor status updates
        intent.getParcelableExtra<SensorStatus>(SleepTrackingService.EXTRA_SENSOR_STATUS)?.let { status ->
            currentSensorStatus = status
            viewModel.updateSensorStatus(status)
        }

        // Handle live metrics updates
        intent.getParcelableExtra<LiveSleepMetrics>(SleepTrackingService.EXTRA_LIVE_METRICS)?.let { metrics ->
            currentLiveMetrics = metrics
            viewModel.updateLiveMetrics(metrics)
        }

        // Handle tracking started event
        if (intent.getBooleanExtra("tracking_started", false)) {
            val settings = intent.getParcelableExtra<SensorSettings>(SleepTrackingService.EXTRA_SENSOR_SETTINGS)
            settings?.let {
                showEnhancedSuccessMessage("Advanced sensors initialized successfully!")
                animateTrackingStart()
            }
        }

        // Handle errors with enhanced error display
        if (intent.getBooleanExtra("error", false)) {
            val errorMessage = intent.getStringExtra("error_message")
            errorMessage?.let {
                showEnhancedErrorMessage(
                    ErrorStateDisplay(
                        title = "Sensor Error",
                        message = it,
                        icon = R.drawable.ic_sensor_error,
                        canRetry = true
                    )
                )
            }
        }
    }

    private fun handleEnhancedSessionComplete(
        sessionAnalytics: Serializable?,
        completedSession: SleepSession?
    ) {
        try {
            sessionAnalytics?.let { analytics ->
                // Create enhanced session display model
                val sessionDisplay = SessionDisplayModel.fromSessionSummary(
                    session = analytics.toSessionSummaryDTO(),
                    insights = emptyList(), // Would be loaded from ViewModel
                    chartData = qualityTrendData
                )

                currentSessionDisplay = sessionDisplay

                // Show comprehensive session summary
                showEnhancedSessionSummary(analytics, completedSession, sessionDisplay)
            }

            // Animate session completion
            animateSessionCompletion {
                // Refresh all data to show updated statistics
                refreshAllEnhancedData()
            }

        } catch (e: Exception) {
            showEnhancedErrorMessage(
                ErrorStateDisplay(
                    title = "Session Processing Error",
                    message = "Error processing completed session: ${e.message}",
                    icon = R.drawable.ic_error,
                    canRetry = false
                )
            )
        }
    }

    private fun handleEnhancedPhaseChange(phaseTransition: PhaseTransition) {
        // Animate phase change with enhanced visual feedback
        animatePhaseChange(phaseTransition) {
            // Update UI indicator with phase-specific styling
            binding.currentPhaseText?.text = phaseTransition.toPhase.getDisplayName()
            binding.currentPhaseText?.setTextColor(
                Color.parseColor(phaseTransition.toPhase.getColor())
            )

            // Show subtle phase change notification
            if (isTracking) {
                showPhaseChangeNotification(phaseTransition)
            }
        }
    }

    private fun handleEnhancedDataSaved(intent: Intent) {
        val saveStatus = intent.getStringExtra(SleepTrackingService.EXTRA_SAVE_STATUS)
        val totalSaves = intent.getIntExtra("total_saves", 0)

        // Animate data save confirmation
        animateDataSave()

        // Update data integrity indicator
        binding.dataSaveIndicator?.backgroundTintList =
            ContextCompat.getColorStateList(this, R.color.accent_green)

        // Update save status for debugging
        if (BuildConfig.DEBUG) {
            binding.dataSaveStatusText?.text = "Saves: $totalSaves"
        }
    }

    private fun handlePerformanceUpdate(performanceMetrics: Map<String, Any>) {
        // Update performance metrics in ViewModel
        viewModel.updatePerformanceMetrics(performanceMetrics)
    }

    // ========== ENHANCED UI UPDATES ==========

    private fun updateEnhancedTrackingUI() {
        if (isTracking) {
            binding.trackingButton.text = getString(R.string.stop_sleep_tracking)
            binding.trackingButton.setIconResource(R.drawable.ic_stop)
            binding.statusText.text = getString(R.string.tracking_in_progress)
            binding.trackingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.accent_amber)

            // Show enhanced tracking elements
            binding.liveTrackingLayout?.visibility = View.VISIBLE
            binding.realTimeChartsLayout?.visibility = View.VISIBLE
            binding.analyticsExpandedLayout?.visibility = View.VISIBLE

        } else {
            binding.trackingButton.text = getString(R.string.start_sleep_tracking)
            binding.trackingButton.setIconResource(R.drawable.ic_sleep)
            binding.statusText.text = getString(R.string.ready_to_track)
            binding.trackingButton.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.accent_green)

            // Hide live tracking elements
            binding.liveTrackingLayout?.visibility = View.GONE
            binding.realTimeChartsLayout?.visibility = View.GONE
            binding.analyticsExpandedLayout?.visibility = View.GONE

            // Reset quality score label when not tracking
            binding.qualityScoreLabel.text = "Quality Score"
            binding.qualityScoreText.setTextColor(ContextCompat.getColor(this, R.color.accent_green))
        }

        // Update navigation availability with visual feedback
        updateNavigationAvailability(!isTracking)
    }

    private fun updateNavigationAvailability(enabled: Boolean) {
        val alpha = if (enabled) 1.0f else 0.5f
        val clickable = enabled

        binding.historyButton.apply {
            isEnabled = clickable
            alpha = alpha
        }

        binding.chartsButton.apply {
            isEnabled = clickable
            alpha = alpha
        }

        binding.settingsButton.apply {
            isEnabled = clickable
            alpha = alpha
        }
    }

    private fun updateEnhancedSensorStatusUI() {
        val status = currentSensorStatus

        if (status == null || !isTracking) {
            // Hide live tracking elements when not tracking
            binding.liveTrackingLayout?.visibility = View.GONE
            binding.liveSensorDataLayout?.visibility = View.GONE
            binding.sensorIndicatorsLayout?.visibility = View.GONE
            return
        }

        // Show live tracking elements with animations
        animateViewVisibility(binding.liveTrackingLayout, true)
        animateViewVisibility(binding.liveSensorDataLayout, true)
        animateViewVisibility(binding.sensorIndicatorsLayout, true)

        // Update sensor indicators with enhanced styling
        updateEnhancedSensorIndicators(status)

        // Update live tracking info with rich display
        updateEnhancedLiveTrackingInfo(status)

        // Update sensor data cards with analytics
        updateEnhancedSensorDataCards(status)

        // Update main status text with rich formatting
        updateEnhancedStatusText(status)
    }

    private fun updateEnhancedSensorIndicators(status: SensorStatus) {
        // Update motion sensor indicator with animation
        val motionColor = if (status.isAccelerometerActive) R.color.accent_green else R.color.text_secondary
        animateColorChange(binding.motionSensorIndicator, motionColor)

        // Update audio sensor indicator with animation
        val audioColor = if (status.isMicrophoneActive) R.color.accent_green else R.color.text_secondary
        animateColorChange(binding.audioSensorIndicator, audioColor)

        // Update sensor health indicators
        binding.sensorHealthIndicator?.visibility = if (status.isFullyActive) View.VISIBLE else View.GONE
    }

    private fun updateEnhancedLiveTrackingInfo(status: SensorStatus) {
        // Update current phase with enhanced styling
        binding.currentPhaseText?.text = status.currentPhase.getDisplayName()
        binding.currentPhaseText?.setTextColor(
            Color.parseColor(status.currentPhase.getColor())
        )

        // Update phase confidence with visual indicator
        val confidencePercent = (status.phaseConfidence * 100).toInt()
        binding.phaseConfidenceText?.text = "$confidencePercent%"
        binding.phaseConfidenceProgressBar?.progress = confidencePercent

        // Update session duration with rich formatting
        binding.sessionDurationText?.text = status.getSessionDurationFormatted()

        // Update progress bar with target duration visualization
        updateSessionProgress(status)

        // Update session efficiency indicator
        currentLiveMetrics?.let { metrics ->
            binding.liveEfficiencyText?.text = "${metrics.sleepEfficiency.toInt()}%"

            val efficiencyColor = ChartTheme.getEfficiencyColor(metrics.sleepEfficiency)
            binding.liveEfficiencyText?.setTextColor(efficiencyColor)
        }
    }

    private fun updateEnhancedSensorDataCards(status: SensorStatus) {
        // Update movement analysis with enhanced visuals
        updateMovementAnalysisCard(status)

        // Update noise analysis with enhanced visuals
        updateNoiseAnalysisCard(status)

        // Update environmental factors
        updateEnvironmentalFactorsCard(status)
    }

    private fun updateMovementAnalysisCard(status: SensorStatus) {
        val intensity = status.currentMovementIntensity

        // Update movement level with semantic description
        val movementLevel = when {
            intensity <= 1.0f -> "Very Still"
            intensity <= 2.0f -> "Calm"
            intensity <= 4.0f -> "Moderate"
            intensity <= 6.0f -> "Restless"
            else -> "Very Restless"
        }

        binding.movementIntensityText?.text = movementLevel
        binding.movementCountText?.text = "${status.totalMovementEvents} events"
        binding.movementIntensityValue?.text = String.format("%.1f", intensity)

        // Color code movement intensity with theme integration
        val movementColor = ChartTheme.getMovementColor(intensity)
        binding.movementIntensityText?.setTextColor(movementColor)

        // Update movement progress bar
        val movementProgress = ((intensity / 6f) * 100).toInt().coerceAtMost(100)
        animateProgressBar(binding.movementProgressBar, movementProgress)
    }

    private fun updateNoiseAnalysisCard(status: SensorStatus) {
        val noiseLevel = status.currentNoiseLevel

        // Update noise level with semantic description
        val noiseLevelText = when {
            noiseLevel <= 1000f -> "Very Quiet"
            noiseLevel <= 2000f -> "Quiet"
            noiseLevel <= 4000f -> "Moderate"
            noiseLevel <= 8000f -> "Loud"
            else -> "Very Loud"
        }

        binding.noiseLevelText?.text = noiseLevelText
        binding.noiseCountText?.text = "${status.totalNoiseEvents} events"
        binding.noiseLevelValue?.text = "${noiseLevel.toInt()}"

        // Color code noise level with theme integration
        val noiseColor = ChartTheme.getNoiseColor(noiseLevel)
        binding.noiseLevelText?.setTextColor(noiseColor)

        // Update noise progress bar
        val noiseProgress = ((noiseLevel / 10000f) * 100).toInt().coerceAtMost(100)
        animateProgressBar(binding.noiseProgressBar, noiseProgress)
    }

    private fun updateEnvironmentalFactorsCard(status: SensorStatus) {
        // Calculate environmental score
        val environmentalScore = calculateEnvironmentalScore(status)

        binding.environmentalScoreText?.text = String.format("%.1f", environmentalScore)

        val environmentalGrade = when {
            environmentalScore >= 8f -> "Excellent"
            environmentalScore >= 6f -> "Good"
            environmentalScore >= 4f -> "Fair"
            else -> "Poor"
        }

        binding.environmentalGradeText?.text = environmentalGrade

        // Color code environmental score
        val environmentColor = ChartTheme.getQualityColor(environmentalScore)
        binding.environmentalScoreText?.setTextColor(environmentColor)
    }

    private fun updateEnhancedStatusText(status: SensorStatus) {
        val statusText = when {
            status.isFullyActive -> {
                val duration = status.getSessionDurationFormatted()
                "🌙 Active Tracking: $duration"
            }
            !status.isAccelerometerActive && !status.isMicrophoneActive ->
                "🔄 Initializing sensors..."
            !status.isAccelerometerActive ->
                "⚠️ Motion sensor issue"
            !status.isMicrophoneActive ->
                "⚠️ Audio sensor issue"
            else ->
                "🚀 Starting sensors..."
        }

        binding.statusText?.text = statusText
        binding.statusText?.setTextColor(
            ContextCompat.getColor(
                this,
                if (status.isFullyActive) R.color.accent_green else R.color.accent_amber
            )
        )
    }

    private fun updateEnhancedLiveMetricsUI() {
        val metrics = currentLiveMetrics ?: return

        if (isTracking) {
            // Update quality score to show live efficiency
            binding.qualityScoreLabel?.text = "Live Efficiency"
            binding.qualityScoreText?.text = "${metrics.sleepEfficiency.toInt()}%"

            // Color code efficiency with enhanced visuals
            val efficiencyColor = ChartTheme.getEfficiencyColor(metrics.sleepEfficiency)
            binding.qualityScoreText?.setTextColor(efficiencyColor)

            // Update restlessness display with rich analytics
            updateRestlessnessDisplay(metrics)

            // Update sleep quality estimation
            updateSleepQualityEstimation(metrics)

            // Update phase confidence visualization
            updatePhaseConfidenceVisualization(metrics)
        }
    }

    private fun updateRestlessnessDisplay(metrics: LiveSleepMetrics) {
        val restlessnessLevel = when {
            metrics.totalRestlessness <= 2f -> "😴 Very Calm"
            metrics.totalRestlessness <= 4f -> "😌 Calm"
            metrics.totalRestlessness <= 6f -> "😐 Moderate"
            metrics.totalRestlessness <= 8f -> "😟 Restless"
            else -> "😰 Very Restless"
        }

        binding.restlessnessLevelText?.text = restlessnessLevel

        val restlessnessColor = when {
            metrics.totalRestlessness <= 4f -> R.color.accent_green
            metrics.totalRestlessness <= 6f -> R.color.accent_amber
            else -> R.color.accent_red
        }
        binding.restlessnessLevelText?.setTextColor(ContextCompat.getColor(this, restlessnessColor))

        // Update restlessness progress bar
        val restlessnessProgress = ((metrics.totalRestlessness / 10f) * 100).toInt()
        animateProgressBar(binding.restlessnessProgressBar, restlessnessProgress)
    }

    private fun updateSleepQualityEstimation(metrics: LiveSleepMetrics) {
        val qualityEstimate = calculateRealTimeQuality(metrics)

        binding.liveQualityEstimateText?.text = String.format("%.1f", qualityEstimate)

        val qualityColor = ChartTheme.getQualityColor(qualityEstimate)
        binding.liveQualityEstimateText?.setTextColor(qualityColor)

        // Update quality progress bar
        val qualityProgress = (qualityEstimate * 10).toInt()
        animateProgressBar(binding.qualityProgressBar, qualityProgress)
    }

    private fun updatePhaseConfidenceVisualization(metrics: LiveSleepMetrics) {
        val confidence = metrics.phaseConfidence
        val confidencePercent = (confidence * 100).toInt()

        binding.phaseConfidenceText?.text = "$confidencePercent%"

        val confidenceColor = when {
            confidence >= 0.8f -> R.color.accent_green
            confidence >= 0.6f -> R.color.accent_amber
            else -> R.color.accent_red
        }
        binding.phaseConfidenceText?.setTextColor(ContextCompat.getColor(this, confidenceColor))
    }

    // ========== ENHANCED DISPLAY UPDATES ==========

    private fun updateEnhancedStatisticsDisplay() {
        val stats = viewModel.getFormattedStatistics()

        // Update main statistics with enhanced formatting
        updateStatisticCard("total_sessions", stats["total_sessions"] ?: "0")
        updateStatisticCard("average_duration", stats["average_duration"] ?: "0h 00m")
        updateStatisticCard("average_quality", stats["average_quality"] ?: "0.0/10")
        updateStatisticCard("average_efficiency", stats["average_efficiency"] ?: "0.0%")

        // Update status indicators with visual enhancements
        binding.sensorStatusText?.text = stats["sensor_status"] ?: "Unknown"
        binding.insightsCountText?.text = "${stats["insights_count"]} insights"

        // Update data integrity indicator
        updateDataIntegrityFromStats(stats)
    }

    private fun updateStatisticCard(type: String, value: String) {
        when (type) {
            "total_sessions" -> {
                binding.totalSessionsText?.text = value
                animateValueChange(binding.totalSessionsText)
            }
            "average_duration" -> {
                binding.avgSleepText?.text = value
                animateValueChange(binding.avgSleepText)
            }
            "average_quality" -> {
                binding.qualityScoreText?.text = value
                animateValueChange(binding.qualityScoreText)

                // Extract numeric value for color coding
                val numericValue = value.substringBefore("/").toFloatOrNull() ?: 0f
                val qualityColor = ChartTheme.getQualityColor(numericValue)
                binding.qualityScoreText?.setTextColor(qualityColor)
            }
            "average_efficiency" -> {
                binding.efficiencyText?.text = value
                animateValueChange(binding.efficiencyText)

                // Extract numeric value for color coding
                val numericValue = value.substringBefore("%").toFloatOrNull() ?: 0f
                val efficiencyColor = ChartTheme.getEfficiencyColor(numericValue)
                binding.efficiencyText?.setTextColor(efficiencyColor)
            }
        }
    }

    private fun updateEnhancedSessionCount(total: Int) {
        binding.totalSessionsText?.text = if (total > 0) "$total sessions tracked" else "No sessions yet"
        binding.totalSessionsText?.visibility = View.VISIBLE

        // Animate count change
        animateCounterChange(binding.totalSessionsText, total)
    }

    private fun updateDurationDisplay(duration: Long) {
        val formattedDuration = formatDuration(duration)
        binding.avgSleepText?.text = formattedDuration
        animateValueChange(binding.avgSleepText)
    }

    private fun updateQualityDisplay(quality: Float) {
        binding.qualityScoreText?.text = getString(R.string.quality_score_format, quality)

        // Color code quality with theme
        val qualityColor = ChartTheme.getQualityColor(quality)
        binding.qualityScoreText?.setTextColor(qualityColor)

        animateValueChange(binding.qualityScoreText)
    }

    private fun updateEnhancedEfficiencyDisplay(efficiency: Float) {
        binding.efficiencyText?.text = String.format("%.1f%%", efficiency)

        // Color code efficiency with theme
        val efficiencyColor = ChartTheme.getEfficiencyColor(efficiency)
        binding.efficiencyText?.setTextColor(efficiencyColor)

        animateValueChange(binding.efficiencyText)

        // Update efficiency progress bar if available
        val efficiencyProgress = efficiency.toInt()
        animateProgressBar(binding.efficiencyProgressBar, efficiencyProgress)
    }

    private fun updateEnhancedAnalyticsDisplay(analytics: SleepAnalytics?) {
        analytics?.let { data ->
            animateViewVisibility(binding.analyticsLayout, true)

            // Show trend information with enhanced styling
            updateTrendIndicator(data.sleepTrend)

            // Show recommendations with count and priority
            updateRecommendationsDisplay(data.recommendations)

            // Show analytics summary
            updateAnalyticsSummary(data)

        } ?: run {
            binding.analyticsLayout?.visibility = View.GONE
        }
    }

    private fun updateTrendIndicator(trend: SleepTrend) {
        val (trendText, trendColor, trendIcon) = when (trend) {
            SleepTrend.IMPROVING -> Triple("📈 Improving", R.color.accent_green, "📈")
            SleepTrend.STABLE -> Triple("➡️ Stable", R.color.accent_blue, "➡️")
            SleepTrend.DECLINING -> Triple("📉 Declining", R.color.accent_amber, "📉")
            SleepTrend.INSUFFICIENT_DATA -> Triple("📊 Analyzing...", R.color.text_secondary, "📊")
        }

        binding.trendIndicatorText?.text = trendText
        binding.trendIndicatorText?.setTextColor(ContextCompat.getColor(this, trendColor))
        binding.trendIconText?.text = trendIcon

        animateValueChange(binding.trendIndicatorText)
    }

    private fun updateRecommendationsDisplay(recommendations: List<String>) {
        binding.recommendationsCountText?.text = "${recommendations.size} recommendations"

        if (recommendations.isNotEmpty()) {
            binding.latestRecommendationText?.text = recommendations.first()
            animateViewVisibility(binding.latestRecommendationText, true)

            // Show recommendation priority indicator
            binding.recommendationPriorityIndicator?.visibility = View.VISIBLE
        } else {
            binding.latestRecommendationText?.visibility = View.GONE
            binding.recommendationPriorityIndicator?.visibility = View.GONE
        }
    }

    private fun updateAnalyticsSummary(analytics: SleepAnalytics) {
        // Update sessions summary
        binding.analyticsSessionCountText?.text = "${analytics.sessions.size} sessions analyzed"

        // Update quality trend
        val qualityTrendText = if (analytics.averageQuality > 0) {
            "Average quality: ${String.format("%.1f/10", analytics.averageQuality)}"
        } else {
            "Quality data unavailable"
        }
        binding.analyticsQualityText?.text = qualityTrendText

        // Update efficiency summary
        val efficiencyText = if (analytics.averageSleepEfficiency > 0) {
            "Average efficiency: ${String.format("%.1f%%", analytics.averageSleepEfficiency)}"
        } else {
            "Efficiency data unavailable"
        }
        binding.analyticsEfficiencyText?.text = efficiencyText
    }

    private fun updateTrendsDisplay(trends: List<DailyTrendData>) {
        if (trends.isNotEmpty()) {
            binding.trendsLayout?.visibility = View.VISIBLE

            // Update trend summary
            val avgQuality = trends.mapNotNull { it.averageQuality }.average()
            binding.trendsSummaryText?.text = "30-day average: ${String.format("%.1f/10", avgQuality)}"

            // Show trend direction
            val trendDirection = calculateTrendDirection(trends)
            updateTrendDirectionIndicator(trendDirection)

        } else {
            binding.trendsLayout?.visibility = View.GONE
        }
    }

    private fun updateEnhancedQualityReportDisplay(report: SleepQualityReport?) {
        report?.let { qualityReport ->
            animateViewVisibility(binding.qualityReportLayout, true)

            // Show overall grade with enhanced styling
            updateQualityGradeDisplay(qualityReport.qualityGrade, qualityReport.overallQualityScore)

            // Show factor analysis
            updateQualityFactorsDisplay(qualityReport.qualityFactors)

            // Show insights and recommendations
            updateQualityInsightsDisplay(qualityReport.keyInsights)

        } ?: run {
            binding.qualityReportLayout?.visibility = View.GONE
        }
    }

    private fun updateQualityGradeDisplay(grade: QualityGrade, score: Float) {
        binding.qualityGradeText?.text = grade.displayName
        binding.qualityScoreDetailText?.text = String.format("%.1f/10", score)

        // Color code the grade with theme
        val gradeColor = ChartTheme.getQualityColor(score)
        binding.qualityGradeText?.setTextColor(gradeColor)
        binding.qualityScoreDetailText?.setTextColor(gradeColor)

        animateValueChange(binding.qualityGradeText)
    }

    private fun updateQualityFactorsDisplay(factors: QualityFactorAnalysis) {
        // Show strongest and weakest factors
        val strongest = factors.strongestFactor
        val weakest = factors.weakestFactor

        binding.strongestFactorText?.text = "💪 ${strongest.name}: ${String.format("%.1f", strongest.score)}"
        binding.weakestFactorText?.text = "⚠️ ${weakest.name}: ${String.format("%.1f", weakest.score)}"

        // Color code factors
        binding.strongestFactorText?.setTextColor(ChartTheme.getQualityColor(strongest.score))
        binding.weakestFactorText?.setTextColor(ChartTheme.getQualityColor(weakest.score))

        animateValueChange(binding.strongestFactorText)
        animateValueChange(binding.weakestFactorText)
    }

    private fun updateQualityInsightsDisplay(insights: List<QualityInsight>) {
        if (insights.isNotEmpty()) {
            binding.qualityInsightsLayout?.visibility = View.VISIBLE
            binding.qualityInsightsCountText?.text = "${insights.size} insights"

            // Show top insight
            val topInsight = insights.firstOrNull()
            topInsight?.let { insight ->
                binding.topQualityInsightText?.text = insight.title
                binding.topQualityInsightDescriptionText?.text = insight.description
            }
        } else {
            binding.qualityInsightsLayout?.visibility = View.GONE
        }
    }

    private fun updateComparisonDisplay(comparison: ComparativeAnalysisResult?) {
        comparison?.let { result ->
            binding.comparisonLayout?.visibility = View.VISIBLE

            // Show performance rating
            val performanceText = when (result.overallPerformanceRating) {
                PerformanceRating.EXCEPTIONAL -> "🏆 Exceptional"
                PerformanceRating.EXCELLENT -> "⭐ Excellent"
                PerformanceRating.GOOD -> "✅ Good"
                PerformanceRating.AVERAGE -> "➡️ Average"
                PerformanceRating.BELOW_AVERAGE -> "⚠️ Below Average"
                PerformanceRating.POOR -> "📉 Needs Work"
            }

            binding.performanceRatingText?.text = performanceText

            // Show key comparison insight
            binding.comparisonInsightText?.text = result.keyComparisonInsight

        } ?: run {
            binding.comparisonLayout?.visibility = View.GONE
        }
    }

    private fun updateEnhancedDataIntegrityIndicator(integrity: DataIntegrityStatus?) {
        integrity?.let { status ->
            val color = if (status.isHealthy) R.color.accent_green else R.color.accent_amber
            animateColorChange(binding.dataIntegrityIndicator, color)

            val statusText = if (status.isHealthy) "✅ Data Healthy" else "⚠️ Data Issues"
            binding.dataIntegrityText?.text = statusText

            // Show detailed integrity info
            val successRateText = "Success rate: ${String.format("%.1f%%", status.successRate)}"
            binding.dataSuccessRateText?.text = successRateText

        } ?: run {
            binding.dataIntegrityIndicator?.backgroundTintList =
                ContextCompat.getColorStateList(this, R.color.text_secondary)
            binding.dataIntegrityText?.text = "Data status unknown"
        }
    }

    private fun updatePerformanceDisplay(metrics: PerformanceMetrics?) {
        metrics?.let { performance ->
            if (BuildConfig.DEBUG) {
                binding.performanceLayout?.visibility = View.VISIBLE

                // Show memory usage
                binding.memoryUsageText?.text = "Memory: ${String.format("%.1f%%", performance.memoryUsage)}"

                // Show data queue size
                binding.dataQueueSizeText?.text = "Queue: ${performance.dataQueueSize}"

                // Show average save time
                binding.avgSaveTimeText?.text = "Save time: ${performance.averageSaveTime}ms"

                // Color code performance
                val performanceColor = when {
                    performance.memoryUsage < 50f && performance.averageSaveTime < 100L -> R.color.accent_green
                    performance.memoryUsage < 80f && performance.averageSaveTime < 200L -> R.color.accent_amber
                    else -> R.color.accent_red
                }

                binding.performanceIndicator?.backgroundTintList =
                    ContextCompat.getColorStateList(this, performanceColor)
            }
        } ?: run {
            binding.performanceLayout?.visibility = View.GONE
        }
    }

    // ========== ENHANCED LOADING STATES ==========

    private fun showEnhancedLoadingIndicators() {
        animateViewVisibility(binding.loadingIndicator, true)
        animateViewVisibility(binding.statisticsLoadingIndicator, true)

        // Start loading animation
        startLoadingAnimation()
    }

    private fun hideEnhancedLoadingIndicators() {
        animateViewVisibility(binding.loadingIndicator, false)
        animateViewVisibility(binding.statisticsLoadingIndicator, false)
        animateViewVisibility(binding.sessionsLoadingIndicator, false)
        animateViewVisibility(binding.analyticsLoadingIndicator, false)

        // Stop loading animation
        stopLoadingAnimation()
    }

    private fun updateEnhancedLoadingIndicator(type: String, loading: Boolean) {
        when (type) {
            "statistics" -> animateViewVisibility(binding.statisticsLoadingIndicator, loading)
            "sessions" -> animateViewVisibility(binding.sessionsLoadingIndicator, loading)
            "analytics" -> animateViewVisibility(binding.analyticsLoadingIndicator, loading)
        }
    }

    private fun startLoadingAnimation() {
        // Implement shimmer or pulse loading animation
        binding.loadingIndicator?.alpha = 0.3f
        binding.loadingIndicator?.animate()?.alpha(1f)?.setDuration(1000)?.withEndAction {
            if (binding.loadingIndicator?.visibility == View.VISIBLE) {
                binding.loadingIndicator?.animate()?.alpha(0.3f)?.setDuration(1000)?.withEndAction {
                    startLoadingAnimation()
                }?.start()
            }
        }?.start()
    }

    private fun stopLoadingAnimation() {
        binding.loadingIndicator?.animate()?.cancel()
    }

    // ========== ENHANCED SESSION HANDLING ==========

    private fun showEnhancedSessionSummary(
        sessionAnalytics: SleepSessionAnalytics,
        completedSession: SleepSession?,
        sessionDisplay: SessionDisplayModel
    ) {
        // Create comprehensive summary with enhanced analytics
        val summary = buildString {
            appendLine("🌙 Sleep Session Complete!")
            appendLine()
            appendLine("⏱️ Duration: ${sessionDisplay.formattedDuration}")
            appendLine("⭐ Quality: ${sessionDisplay.formattedQuality}/10 (${sessionDisplay.qualityGrade})")
            appendLine("📊 Efficiency: ${sessionDisplay.formattedEfficiency}")
            appendLine()

            // Add enhanced quality breakdown
            appendLine("Quality Analysis:")
            appendLine("• Movement: ${String.format("%.1f/10", sessionAnalytics.qualityFactors.movementScore)}")
            appendLine("• Environment: ${String.format("%.1f/10", sessionAnalytics.qualityFactors.noiseScore)}")
            appendLine("• Duration: ${String.format("%.1f/10", sessionAnalytics.qualityFactors.durationScore)}")
            appendLine("• Consistency: ${String.format("%.1f/10", sessionAnalytics.qualityFactors.consistencyScore)}")
            appendLine()

            // Add personalized insights
            val qualityScore = sessionAnalytics.qualityFactors.overallScore
            when {
                qualityScore >= 8f -> {
                    appendLine("🎉 Excellent sleep! You're in the top tier.")
                    appendLine("✨ Keep up your great sleep habits!")
                }
                qualityScore >= 6f -> {
                    appendLine("😊 Good sleep quality achieved.")
                    appendLine("💡 Small improvements could make it even better.")
                }
                qualityScore >= 4f -> {
                    appendLine("😐 Average sleep quality.")
                    appendLine("🎯 Focus on reducing movement and improving environment.")
                }
                else -> {
                    appendLine("😴 Sleep quality could be improved.")
                    appendLine("💪 Try relaxation techniques and optimize your environment.")
                }
            }

            // Add movement insights
            if (sessionAnalytics.averageMovementIntensity < 2.0f) {
                appendLine()
                appendLine("🧘 Excellent stillness during sleep!")
            } else if (sessionAnalytics.averageMovementIntensity > 5.0f) {
                appendLine()
                appendLine("🌊 Consider relaxation techniques for more restful sleep.")
            }
        }

        // Show enhanced session summary dialog
        showEnhancedSessionSummaryDialog(summary, sessionAnalytics, sessionDisplay)
    }

    private fun showEnhancedSessionSummaryDialog(
        summary: String,
        sessionAnalytics: SleepSessionAnalytics,
        sessionDisplay: SessionDisplayModel
    ) {
        // For now, show as enhanced toast. In production, use a rich dialog
        Toast.makeText(this, summary, Toast.LENGTH_LONG).show()

        // Update UI with latest session info
        updateLastEnhancedSessionDisplay(sessionAnalytics, sessionDisplay)

        // Trigger session completion animation
        animateSessionCompletion {
            // Additional completion actions
        }
    }

    private fun updateLastEnhancedSessionDisplay(
        sessionAnalytics: SleepSessionAnalytics,
        sessionDisplay: SessionDisplayModel
    ) {
        // Update the last session display with comprehensive info
        val lastSessionText = buildString {
            append("Latest: ${sessionDisplay.formattedDuration}")
            append(" • ${sessionDisplay.formattedQuality}/10")
            append(" • ${sessionDisplay.formattedEfficiency} efficiency")

            // Add trend indicator
            sessionDisplay.trendIndicator.let { trend ->
                append(" ${trend.icon}")
            }
        }

        binding.lastSleepInfo?.text = lastSessionText
        animateViewVisibility(binding.lastSleepInfo, true)

        // Color code based on quality
        val qualityColor = ChartTheme.getQualityColor(sessionAnalytics.qualityFactors.overallScore)
        binding.lastSleepInfo?.setTextColor(qualityColor)
    }

    // ========== ENHANCED ANALYTICS VIEWS ==========

    private fun toggleAnalyticsView() {
        currentAnalyticsDisplay = when (currentAnalyticsDisplay) {
            AnalyticsDisplayState.OVERVIEW -> AnalyticsDisplayState.DETAILED
            AnalyticsDisplayState.DETAILED -> AnalyticsDisplayState.TRENDS
            AnalyticsDisplayState.TRENDS -> AnalyticsDisplayState.COMPARISON
            AnalyticsDisplayState.COMPARISON -> AnalyticsDisplayState.OVERVIEW
        }

        updateAnalyticsViewDisplay()
    }

    private fun updateAnalyticsViewDisplay() {
        when (currentAnalyticsDisplay) {
            AnalyticsDisplayState.OVERVIEW -> {
                binding.analyticsOverviewLayout?.visibility = View.VISIBLE
                binding.analyticsDetailedLayout?.visibility = View.GONE
                binding.analyticsTrendsLayout?.visibility = View.GONE
                binding.analyticsComparisonLayout?.visibility = View.GONE
                binding.analyticsToggleButton?.text = "Overview"
            }
            AnalyticsDisplayState.DETAILED -> {
                binding.analyticsOverviewLayout?.visibility = View.GONE
                binding.analyticsDetailedLayout?.visibility = View.VISIBLE
                binding.analyticsTrendsLayout?.visibility = View.GONE
                binding.analyticsComparisonLayout?.visibility = View.GONE
                binding.analyticsToggleButton?.text = "Detailed"
            }
            AnalyticsDisplayState.TRENDS -> {
                binding.analyticsOverviewLayout?.visibility = View.GONE
                binding.analyticsDetailedLayout?.visibility = View.GONE
                binding.analyticsTrendsLayout?.visibility = View.VISIBLE
                binding.analyticsComparisonLayout?.visibility = View.GONE
                binding.analyticsToggleButton?.text = "Trends"
            }
            AnalyticsDisplayState.COMPARISON -> {
                binding.analyticsOverviewLayout?.visibility = View.GONE
                binding.analyticsDetailedLayout?.visibility = View.GONE
                binding.analyticsTrendsLayout?.visibility = View.GONE
                binding.analyticsComparisonLayout?.visibility = View.VISIBLE
                binding.analyticsToggleButton?.text = "Comparison"
            }
        }

        // Animate view transition
        animateAnalyticsViewTransition()
    }

    private fun showDetailedQualityAnalysis() {
        // Navigate to detailed quality analysis (could be a fragment or new activity)
        currentAnalyticsDisplay = AnalyticsDisplayState.DETAILED
        updateAnalyticsViewDisplay()
    }

    private fun showDetailedEfficiencyAnalysis() {
        // Navigate to detailed efficiency analysis
        val intent = Intent(this, ChartsActivity::class.java)
        intent.putExtra("chart_type", "efficiency_detailed")
        startActivity(intent)
    }

    private fun showComprehensiveAnalytics() {
        // Show comprehensive analytics view
        currentAnalyticsDisplay = AnalyticsDisplayState.COMPARISON
        updateAnalyticsViewDisplay()
    }

    // ========== ENHANCED ANIMATIONS ==========

    private fun animateButtonPress(button: View, action: () -> Unit) {
        button.animate()
            .scaleX(0.95f)
            .scaleY(0.95f)
            .setDuration(100)
            .setInterpolator(AccelerateDecelerateInterpolator())
            .withEndAction {
                button.animate()
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(100)
                    .setInterpolator(AccelerateDecelerateInterpolator())
                    .withEndAction {
                        action()
                    }
                    .start()
            }
            .start()
    }

    private fun animateRefreshButton(button: View, action: () -> Unit) {
        button.animate()
            .rotationBy(360f)
            .setDuration(500)
            .setInterpolator(AccelerateDecelerateInterpolator())
            .withEndAction {
                action()
            }
            .start()
    }

    private fun animateViewVisibility(view: View?, visible: Boolean) {
        view?.let { v ->
            if (visible && v.visibility != View.VISIBLE) {
                v.visibility = View.VISIBLE
                v.alpha = 0f
                v.animate()
                    .alpha(1f)
                    .setDuration(300)
                    .setInterpolator(AccelerateDecelerateInterpolator())
                    .start()
            } else if (!visible && v.visibility == View.VISIBLE) {
                v.animate()
                    .alpha(0f)
                    .setDuration(300)
                    .setInterpolator(AccelerateDecelerateInterpolator())
                    .withEndAction {
                        v.visibility = View.GONE
                    }
                    .start()
            }
        }
    }

    private fun animateColorChange(view: View?, colorRes: Int) {
        view?.let { v ->
            val colorFrom = (v.backgroundTintList?.defaultColor ?: 0)
            val colorTo = ContextCompat.getColor(this, colorRes)

            ValueAnimator.ofArgb(colorFrom, colorTo).apply {
                duration = 300
                addUpdateListener { animator ->
                    v.backgroundTintList = ContextCompat.getColorStateList(
                        this@MainActivity,
                        colorRes
                    )
                }
                start()
            }
        }
    }

    private fun animateProgressBar(progressBar: View?, progress: Int) {
        // This would need to be implemented based on your progress bar type
        // For now, just a placeholder
    }

    private fun animateValueChange(textView: View?) {
        textView?.let { tv ->
            tv.animate()
                .scaleX(1.1f)
                .scaleY(1.1f)
                .setDuration(150)
                .withEndAction {
                    tv.animate()
                        .scaleX(1f)
                        .scaleY(1f)
                        .setDuration(150)
                        .start()
                }
                .start()
        }
    }

    private fun animateCounterChange(textView: View?, newValue: Int) {
        // Implement counter animation (counting up effect)
        animateValueChange(textView)
    }

    private fun animateDataRefresh(action: () -> Unit) {
        binding.mainContentLayout?.animate()
            ?.alpha(0.7f)
            ?.setDuration(200)
            ?.withEndAction {
                action()
                binding.mainContentLayout?.animate()
                    ?.alpha(1f)
                    ?.setDuration(200)
                    ?.start()
            }
            ?.start()
    }

    private fun animateTrackingStart() {
        binding.trackingButton.animate()
            .scaleX(1.1f)
            .scaleY(1.1f)
            .setDuration(200)
            .withEndAction {
                binding.trackingButton.animate()
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(200)
                    .start()
            }
            .start()
    }

    private fun animateTrackingStop(action: () -> Unit) {
        binding.liveTrackingLayout?.animate()
            ?.alpha(0f)
            ?.setDuration(300)
            ?.withEndAction {
                action()
                binding.liveTrackingLayout?.alpha = 1f
            }
            ?.start()
    }

    private fun animateSessionCompletion(action: () -> Unit) {
        // Animate session completion with celebration effect
        binding.trackingButton.animate()
            .rotationBy(360f)
            .setDuration(800)
            .withEndAction {
                action()
            }
            .start()
    }

    private fun animatePhaseChange(phaseTransition: PhaseTransition, action: () -> Unit) {
        binding.currentPhaseText?.animate()
            ?.alpha(0f)
            ?.setDuration(150)
            ?.withEndAction {
                action()
                binding.currentPhaseText?.animate()
                    ?.alpha(1f)
                    ?.setDuration(150)
                    ?.start()
            }
            ?.start()
    }

    private fun animateDataSave() {
        binding.dataSaveIndicator?.animate()
            ?.scaleX(1.2f)
            ?.scaleY(1.2f)
            ?.setDuration(100)
            ?.withEndAction {
                binding.dataSaveIndicator?.animate()
                    ?.scaleX(1f)
                    ?.scaleY(1f)
                    ?.setDuration(100)
                    ?.start()
            }
            ?.start()
    }

    private fun animateAnalyticsViewTransition() {
        binding.analyticsContentLayout?.animate()
            ?.alpha(0f)
            ?.setDuration(150)
            ?.withEndAction {
                binding.analyticsContentLayout?.animate()
                    ?.alpha(1f)
                    ?.setDuration(150)
                    ?.start()
            }
            ?.start()
    }

    // ========== ENHANCED HELPER METHODS ==========

    private fun calculateEnvironmentalScore(status: SensorStatus): Float {
        val movementFactor = 1f - (status.currentMovementIntensity / 6f).coerceIn(0f, 1f)
        val noiseFactor = 1f - (status.currentNoiseLevel / 10000f).coerceIn(0f, 1f)
        return ((movementFactor + noiseFactor) / 2f * 10f).coerceIn(0f, 10f)
    }

    private fun calculateTrendDirection(trends: List<DailyTrendData>): TrendDirection {
        if (trends.size < 7) return TrendDirection.INSUFFICIENT_DATA

        val recent = trends.takeLast(7).mapNotNull { it.averageQuality }.average()
        val older = trends.dropLast(7).mapNotNull { it.averageQuality }.average()

        return when {
            recent > older + 0.5 -> TrendDirection.IMPROVING
            recent < older - 0.5 -> TrendDirection.DECLINING
            else -> TrendDirection.STABLE
        }
    }

    private fun updateTrendDirectionIndicator(direction: TrendDirection) {
        val (icon, color) = when (direction) {
            TrendDirection.IMPROVING -> "📈" to R.color.accent_green
            TrendDirection.DECLINING -> "📉" to R.color.accent_red
            TrendDirection.STABLE -> "➡️" to R.color.accent_blue
            else -> "📊" to R.color.text_secondary
        }

        binding.trendDirectionIcon?.text = icon
        binding.trendDirectionIcon?.setTextColor(ContextCompat.getColor(this, color))
    }

    private fun updateSessionProgress(status: SensorStatus) {
        val targetDuration = 8 * 60 * 60 * 1000L // 8 hours in ms
        val currentDuration = status.sessionDuration
        val progress = ((currentDuration.toFloat() / targetDuration.toFloat()) * 100).toInt()

        binding.sessionProgressBar?.progress = progress.coerceAtMost(100)

        // Update progress text
        val progressText = "${String.format("%.1f", currentDuration / (1000f * 60f * 60f))}h / 8h"
        binding.sessionProgressText?.text = progressText
    }

    private fun updateDataIntegrityFromStats(stats: Map<String, String>) {
        val isTracking = stats["is_tracking"]?.toBoolean() ?: false
        val color = if (isTracking) R.color.accent_green else R.color.text_secondary

        binding.dataIntegrityIndicator?.backgroundTintList =
            ContextCompat.getColorStateList(this, color)
    }

    private fun showInitializationSuccess() {
        animateViewVisibility(binding.initializationSuccessIndicator, true)

        // Auto-hide after delay
        uiUpdateHandler.postDelayed({
            animateViewVisibility(binding.initializationSuccessIndicator, false)
        }, 3000)
    }

    private fun showPhaseChangeNotification(phaseTransition: PhaseTransition) {
        val message = "Sleep phase: ${phaseTransition.toPhase.getDisplayName()}"

        // Create subtle toast for phase changes
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun requestEnhancedServiceStatus() {
        val serviceIntent = Intent(this, SleepTrackingService::class.java).apply {
            action = SleepTrackingService.ACTION_GET_STATUS
        }
        startService(serviceIntent)
    }

    private fun showEnhancedPermissionRationale(onAccept: () -> Unit) {
        // Show enhanced permission rationale with better explanation
        Toast.makeText(
            this,
            "SomniAI needs microphone access to monitor your sleep environment for comprehensive analysis.",
            Toast.LENGTH_LONG
        ).show()
        onAccept()
    }

    private fun handlePermissionDenied() {
        binding.trackingButton.apply {
            isEnabled = false
            text = "🎤 Microphone Permission Required"
            backgroundTintList = ContextCompat.getColorStateList(context, R.color.text_secondary)
        }

        // Show helpful message
        binding.statusText?.text = "Grant microphone permission to enable sleep tracking"
        binding.statusText?.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
    }

    private fun showEnhancedSuccessMessage(message: String) {
        Toast.makeText(this, "✅ $message", Toast.LENGTH_SHORT).show()
    }

    private fun showEnhancedErrorMessage(errorState: ErrorStateDisplay) {
        // For now, show as toast. In production, use a proper error dialog
        Toast.makeText(this, "${errorState.title}: ${errorState.message}", Toast.LENGTH_LONG).show()
    }

    private fun hasRequiredPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        return getString(R.string.sleep_duration_format, hours, minutes)
    }

    // Extension function to convert analytics to DTO
    private fun SleepSessionAnalytics.toSessionSummaryDTO(): SessionSummaryDTO {
        return SessionSummaryDTO(
            id = System.currentTimeMillis(), // Would use actual session ID
            startTime = System.currentTimeMillis() - totalDuration,
            endTime = System.currentTimeMillis(),
            totalDuration = totalDuration,
            qualityScore = qualityFactors.overallScore,
            sleepEfficiency = sleepEfficiency,
            totalMovementEvents = 0, // Would come from actual data
            totalNoiseEvents = 0, // Would come from actual data
            averageMovementIntensity = averageMovementIntensity,
            averageNoiseLevel = averageNoiseLevel
        )
    }

    // ========== LIFECYCLE METHODS ==========

    override fun onResume() {
        super.onResume()
        // Refresh tracking state when returning to the activity
        if (hasRequiredPermissions()) {
            initializeEnhancedTracking()
        }
    }

    override fun onPause() {
        super.onPause()
        // Service continues running in background
        // Save current UI state if needed
    }

    override fun onDestroy() {
        // Cleanup resources
        animationScope.cancel()
        uiUpdateHandler.removeCallbacksAndMessages(null)

        // Unregister broadcast receiver
        try {
            LocalBroadcastManager.getInstance(this)
                .unregisterReceiver(sensorUpdateReceiver)
        } catch (e: Exception) {
            // Receiver might not be registered
        }

        super.onDestroy()
    }

    // ========== ENUMS ==========

    private enum class AnalyticsDisplayState {
        OVERVIEW, DETAILED, TRENDS, COMPARISON
    }
}