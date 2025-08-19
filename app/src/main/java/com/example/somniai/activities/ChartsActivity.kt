package com.example.somniai.activities

import android.graphics.Color
import android.os.Bundle
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.database.SleepDatabase
import com.example.somniai.repository.SleepRepository
import com.example.somniai.viewmodel.ChartsViewModel
import com.example.somniai.databinding.ActivityChartsBinding
import com.github.mikephil.charting.animation.Easing
import com.github.mikephil.charting.charts.*
import com.github.mikephil.charting.components.*
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.listener.OnChartValueSelectedListener
import com.google.android.material.tabs.TabLayout
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

/**
 * Advanced Charts Activity with MPAndroidChart integration
 *
 * Features:
 * - Multiple chart types with smooth transitions
 * - Integration with sophisticated analytics data models
 * - Dark theme optimized styling
 * - Interactive chart selection and filtering
 * - Real-time data loading with proper error handling
 * - Comprehensive visualization of sleep patterns
 */
class ChartsActivity : AppCompatActivity(), OnChartValueSelectedListener {

    companion object {
        private const val TAG = "ChartsActivity"
        private const val CHART_ANIMATION_DURATION = 1000
        private const val DEFAULT_DAYS_RANGE = 30

        // Chart type constants
        private const val CHART_DURATION_TRENDS = 0
        private const val CHART_QUALITY_FACTORS = 1
        private const val CHART_MOVEMENT_PATTERNS = 2
        private const val CHART_PHASE_DISTRIBUTION = 3
        private const val CHART_EFFICIENCY_TRENDS = 4
        private const val CHART_WEEKLY_SUMMARY = 5
    }

    private lateinit var binding: ActivityChartsBinding
    private lateinit var chartsViewModel: ChartsViewModel

    // Chart references
    private lateinit var durationLineChart: LineChart
    private lateinit var qualityBarChart: BarChart
    private lateinit var movementLineChart: LineChart
    private lateinit var phaseDistributionPieChart: PieChart
    private lateinit var efficiencyLineChart: LineChart
    private lateinit var weeklyBarChart: BarChart

    // Current state
    private var currentChartType = CHART_DURATION_TRENDS
    private var currentDataRange = DEFAULT_DAYS_RANGE
    private var isLoading = false

    // Color palette for dark theme
    private val chartColors = intArrayOf(
        Color.parseColor("#38a169"), // accent_green
        Color.parseColor("#d69e2e"), // accent_amber
        Color.parseColor("#4299e1"), // blue
        Color.parseColor("#ed8936"), // orange
        Color.parseColor("#9f7aea"), // purple
        Color.parseColor("#38b2ac")  // teal
    )

    private val phaseColors = mapOf(
        SleepPhase.AWAKE to Color.parseColor("#FF6B6B"),
        SleepPhase.LIGHT_SLEEP to Color.parseColor("#4ECDC4"),
        SleepPhase.DEEP_SLEEP to Color.parseColor("#45B7D1"),
        SleepPhase.REM_SLEEP to Color.parseColor("#96CEB4")
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityChartsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        initializeViewModel()
        setupChartViews()
        setupTabNavigation()
        setupDataRangeFilter()
        setupObservers()

        // Load initial data
        loadChartData()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setDisplayShowHomeEnabled(true)
            title = "Sleep Analytics"
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                finish()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun initializeViewModel() {
        // Initialize ViewModel with repository
        val database = SleepDatabase.getDatabase(this)
        val repository = SleepRepository(database, this)
        val factory = ChartsViewModelFactory(repository)
        chartsViewModel = ViewModelProvider(this, factory)[ChartsViewModel::class.java]
    }

    private fun setupChartViews() {
        // Initialize all chart views
        durationLineChart = binding.durationLineChart
        qualityBarChart = binding.qualityBarChart
        movementLineChart = binding.movementLineChart
        phaseDistributionPieChart = binding.phaseDistributionPieChart
        efficiencyLineChart = binding.efficiencyLineChart
        weeklyBarChart = binding.weeklyBarChart

        // Configure each chart
        setupLineChart(durationLineChart, "Sleep Duration", "Hours")
        setupBarChart(qualityBarChart, "Quality Factors", "Score (0-10)")
        setupLineChart(movementLineChart, "Movement Intensity", "Intensity Level")
        setupPieChart(phaseDistributionPieChart, "Sleep Phase Distribution")
        setupLineChart(efficiencyLineChart, "Sleep Efficiency", "Efficiency %")
        setupBarChart(weeklyBarChart, "Weekly Summary", "Value")

        // Show initial chart
        showChart(currentChartType)
    }

    private fun setupTabNavigation() {
        // Add tabs for different chart types
        val chartTabs = listOf(
            "Duration Trends",
            "Quality Factors",
            "Movement Patterns",
            "Phase Distribution",
            "Efficiency Trends",
            "Weekly Summary"
        )

        chartTabs.forEachIndexed { index, title ->
            binding.chartTabLayout.addTab(
                binding.chartTabLayout.newTab().setText(title)
            )
        }

        binding.chartTabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                tab?.position?.let { position ->
                    currentChartType = position
                    showChart(position)
                    loadChartData()
                }
            }

            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })
    }

    private fun setupDataRangeFilter() {
        binding.rangeFilterGroup.setOnCheckedChangeListener { _, checkedId ->
            currentDataRange = when (checkedId) {
                R.id.range7Days -> 7
                R.id.range30Days -> 30
                R.id.range90Days -> 90
                else -> 30
            }

            loadChartData()
        }

        // Set default selection
        binding.range30Days.isChecked = true
    }

    private fun setupObservers() {
        // Observe loading states
        chartsViewModel.isLoading.observe(this) { loading ->
            isLoading = loading
            updateLoadingState(loading)
        }

        // Observe error messages
        chartsViewModel.errorMessage.observe(this) { error ->
            error?.let {
                showError(it)
                chartsViewModel.clearError()
            }
        }

        // Observe chart data
        chartsViewModel.dailyTrends.observe(this) { trends ->
            trends?.let { updateDurationTrendsChart(it) }
        }

        chartsViewModel.qualityFactors.observe(this) { factors ->
            factors?.let { updateQualityFactorsChart(it) }
        }

        chartsViewModel.movementPatterns.observe(this) { patterns ->
            patterns?.let { updateMovementPatternsChart(it) }
        }

        chartsViewModel.phaseDistribution.observe(this) { distribution ->
            distribution?.let { updatePhaseDistributionChart(it) }
        }

        chartsViewModel.efficiencyTrends.observe(this) { efficiency ->
            efficiency?.let { updateEfficiencyTrendsChart(it) }
        }

        chartsViewModel.weeklyStats.observe(this) { weekly ->
            weekly?.let { updateWeeklySummaryChart(it) }
        }
    }

    // ========== CHART CONFIGURATION ==========

    private fun setupLineChart(chart: LineChart, title: String, yAxisLabel: String) {
        chart.apply {
            // Chart styling
            setBackgroundColor(ContextCompat.getColor(this@ChartsActivity, R.color.surface_dark))
            description.isEnabled = false
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(true)
            setPinchZoom(true)
            setDrawGridBackground(false)

            // Animation
            animateX(CHART_ANIMATION_DURATION, Easing.EaseInOutQuart)

            // Legend
            legend.apply {
                textColor = Color.WHITE
                textSize = 12f
                isWordWrapEnabled = true
            }

            // X Axis
            xAxis.apply {
                textColor = Color.WHITE
                textSize = 11f
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(false)
                granularity = 1f
                valueFormatter = DateAxisValueFormatter()
            }

            // Y Axis Left
            axisLeft.apply {
                textColor = Color.WHITE
                textSize = 11f
                setDrawGridLines(true)
                gridColor = Color.parseColor("#33FFFFFF")
                setDrawZeroLine(true)
                zeroLineColor = Color.parseColor("#66FFFFFF")
            }

            // Y Axis Right (disabled)
            axisRight.isEnabled = false

            // Chart interaction
            setOnChartValueSelectedListener(this@ChartsActivity)
        }
    }

    private fun setupBarChart(chart: BarChart, title: String, yAxisLabel: String) {
        chart.apply {
            // Chart styling
            setBackgroundColor(ContextCompat.getColor(this@ChartsActivity, R.color.surface_dark))
            description.isEnabled = false
            setTouchEnabled(true)
            setDrawGridBackground(false)
            setFitBars(true)

            // Animation
            animateY(CHART_ANIMATION_DURATION, Easing.EaseInOutQuart)

            // Legend
            legend.apply {
                textColor = Color.WHITE
                textSize = 12f
                isWordWrapEnabled = true
            }

            // X Axis
            xAxis.apply {
                textColor = Color.WHITE
                textSize = 11f
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(false)
                granularity = 1f
                valueFormatter = FactorAxisValueFormatter()
            }

            // Y Axis Left
            axisLeft.apply {
                textColor = Color.WHITE
                textSize = 11f
                setDrawGridLines(true)
                gridColor = Color.parseColor("#33FFFFFF")
                axisMinimum = 0f
                axisMaximum = 10f
            }

            // Y Axis Right (disabled)
            axisRight.isEnabled = false

            // Chart interaction
            setOnChartValueSelectedListener(this@ChartsActivity)
        }
    }

    private fun setupPieChart(chart: PieChart, title: String) {
        chart.apply {
            // Chart styling
            setBackgroundColor(ContextCompat.getColor(this@ChartsActivity, R.color.surface_dark))
            description.isEnabled = false
            setTouchEnabled(true)
            setDrawHoleEnabled(true)
            setHoleColor(ContextCompat.getColor(this@ChartsActivity, R.color.surface_dark))
            holeRadius = 40f
            transparentCircleRadius = 45f
            setDrawCenterText(true)
            setCenterText(title)
            setCenterTextColor(Color.WHITE)
            setCenterTextSize(14f)

            // Animation
            animateY(CHART_ANIMATION_DURATION, Easing.EaseInOutQuart)

            // Legend
            legend.apply {
                textColor = Color.WHITE
                textSize = 12f
                orientation = Legend.LegendOrientation.VERTICAL
                verticalAlignment = Legend.LegendVerticalAlignment.CENTER
                horizontalAlignment = Legend.LegendHorizontalAlignment.RIGHT
                setDrawInside(false)
            }

            // Chart interaction
            setOnChartValueSelectedListener(this@ChartsActivity)
        }
    }

    // ========== CHART DISPLAY MANAGEMENT ==========

    private fun showChart(chartType: Int) {
        // Hide all charts
        binding.durationLineChart.visibility = View.GONE
        binding.qualityBarChart.visibility = View.GONE
        binding.movementLineChart.visibility = View.GONE
        binding.phaseDistributionPieChart.visibility = View.GONE
        binding.efficiencyLineChart.visibility = View.GONE
        binding.weeklyBarChart.visibility = View.GONE

        // Show selected chart
        when (chartType) {
            CHART_DURATION_TRENDS -> {
                binding.durationLineChart.visibility = View.VISIBLE
                binding.chartTitle.text = "Sleep Duration Trends"
                binding.chartDescription.text = "Daily sleep duration over time"
            }
            CHART_QUALITY_FACTORS -> {
                binding.qualityBarChart.visibility = View.VISIBLE
                binding.chartTitle.text = "Quality Factor Analysis"
                binding.chartDescription.text = "Breakdown of sleep quality components"
            }
            CHART_MOVEMENT_PATTERNS -> {
                binding.movementLineChart.visibility = View.VISIBLE
                binding.chartTitle.text = "Movement Patterns"
                binding.chartDescription.text = "Movement intensity during sleep sessions"
            }
            CHART_PHASE_DISTRIBUTION -> {
                binding.phaseDistributionPieChart.visibility = View.VISIBLE
                binding.chartTitle.text = "Sleep Phase Distribution"
                binding.chartDescription.text = "Time spent in each sleep phase"
            }
            CHART_EFFICIENCY_TRENDS -> {
                binding.efficiencyLineChart.visibility = View.VISIBLE
                binding.chartTitle.text = "Sleep Efficiency Trends"
                binding.chartDescription.text = "Sleep efficiency percentage over time"
            }
            CHART_WEEKLY_SUMMARY -> {
                binding.weeklyBarChart.visibility = View.VISIBLE
                binding.chartTitle.text = "Weekly Summary"
                binding.chartDescription.text = "Comprehensive weekly sleep metrics"
            }
        }
    }

    // ========== DATA LOADING ==========

    private fun loadChartData() {
        if (isLoading) return

        val endDate = System.currentTimeMillis()
        val startDate = endDate - (currentDataRange * 24 * 60 * 60 * 1000L)

        when (currentChartType) {
            CHART_DURATION_TRENDS -> chartsViewModel.loadDurationTrends(startDate, endDate)
            CHART_QUALITY_FACTORS -> chartsViewModel.loadQualityFactors(startDate, endDate)
            CHART_MOVEMENT_PATTERNS -> chartsViewModel.loadMovementPatterns(startDate, endDate)
            CHART_PHASE_DISTRIBUTION -> chartsViewModel.loadPhaseDistribution(startDate, endDate)
            CHART_EFFICIENCY_TRENDS -> chartsViewModel.loadEfficiencyTrends(startDate, endDate)
            CHART_WEEKLY_SUMMARY -> chartsViewModel.loadWeeklyStats(startDate, endDate)
        }
    }

    // ========== CHART DATA UPDATES ==========

    private fun updateDurationTrendsChart(trends: List<DailyTrendData>) {
        if (trends.isEmpty()) {
            showEmptyState("No duration data available")
            return
        }

        val entries = trends.mapIndexed { index, trend ->
            Entry(index.toFloat(), trend.durationHours)
        }

        val dataSet = LineDataSet(entries, "Sleep Duration").apply {
            color = chartColors[0]
            setCircleColor(chartColors[0])
            lineWidth = 3f
            circleRadius = 4f
            setDrawCircleHole(false)
            valueTextColor = Color.WHITE
            valueTextSize = 10f
            mode = LineDataSet.Mode.CUBIC_BEZIER
            setDrawFilled(true)
            fillColor = chartColors[0]
            fillAlpha = 30
        }

        val lineData = LineData(dataSet)
        durationLineChart.apply {
            data = lineData
            invalidate()

            // Update X-axis labels
            xAxis.valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String {
                    val index = value.toInt()
                    return if (index >= 0 && index < trends.size) {
                        trends[index].formattedDate
                    } else ""
                }
            }
        }

        hideEmptyState()
    }

    private fun updateQualityFactorsChart(factors: List<QualityFactorBreakdown>) {
        if (factors.isEmpty()) {
            showEmptyState("No quality factor data available")
            return
        }

        // Calculate average factors across all sessions
        val avgFactors = factors.fold(
            mutableMapOf(
                "Movement" to 0f,
                "Noise" to 0f,
                "Duration" to 0f,
                "Consistency" to 0f,
                "Efficiency" to 0f,
                "Phase Balance" to 0f
            )
        ) { acc, factor ->
            acc["Movement"] = acc["Movement"]!! + factor.movementScore
            acc["Noise"] = acc["Noise"]!! + factor.noiseScore
            acc["Duration"] = acc["Duration"]!! + factor.durationScore
            acc["Consistency"] = acc["Consistency"]!! + factor.consistencyScore
            acc["Efficiency"] = acc["Efficiency"]!! + factor.efficiencyScore
            acc["Phase Balance"] = acc["Phase Balance"]!! + factor.phaseBalanceScore
            acc
        }.mapValues { it.value / factors.size }

        val entries = avgFactors.values.mapIndexed { index, value ->
            BarEntry(index.toFloat(), value)
        }

        val dataSet = BarDataSet(entries, "Quality Factors").apply {
            colors = chartColors.take(avgFactors.size)
            valueTextColor = Color.WHITE
            valueTextSize = 11f
            valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String = "${value.roundToInt()}/10"
            }
        }

        val barData = BarData(dataSet)
        qualityBarChart.apply {
            data = barData
            invalidate()

            // Update X-axis labels
            xAxis.valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String {
                    val index = value.toInt()
                    val labels = avgFactors.keys.toList()
                    return if (index >= 0 && index < labels.size) {
                        labels[index]
                    } else ""
                }
            }
        }

        hideEmptyState()
    }

    private fun updateMovementPatternsChart(patterns: List<MovementPatternData>) {
        if (patterns.isEmpty()) {
            showEmptyState("No movement pattern data available")
            return
        }

        val entries = patterns.mapIndexed { index, pattern ->
            Entry(index.toFloat(), pattern.averageIntensity)
        }

        val dataSet = LineDataSet(entries, "Movement Intensity").apply {
            color = chartColors[2]
            setCircleColor(chartColors[2])
            lineWidth = 2f
            circleRadius = 3f
            setDrawCircleHole(false)
            valueTextColor = Color.WHITE
            valueTextSize = 9f
            mode = LineDataSet.Mode.LINEAR
        }

        val lineData = LineData(dataSet)
        movementLineChart.apply {
            data = lineData
            invalidate()

            // Update X-axis to show hours
            xAxis.valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String {
                    val index = value.toInt()
                    return if (index >= 0 && index < patterns.size) {
                        "H${patterns[index].hourOfSession}"
                    } else ""
                }
            }
        }

        hideEmptyState()
    }

    private fun updatePhaseDistributionChart(distributions: List<PhaseDistributionData>) {
        if (distributions.isEmpty()) {
            showEmptyState("No sleep phase data available")
            return
        }

        // Aggregate phase distribution across all sessions
        val totalPhases = distributions.fold(
            mutableMapOf(
                SleepPhase.AWAKE to 0f,
                SleepPhase.LIGHT_SLEEP to 0f,
                SleepPhase.DEEP_SLEEP to 0f,
                SleepPhase.REM_SLEEP to 0f
            )
        ) { acc, dist ->
            acc[SleepPhase.AWAKE] = acc[SleepPhase.AWAKE]!! + dist.awakePercentage
            acc[SleepPhase.LIGHT_SLEEP] = acc[SleepPhase.LIGHT_SLEEP]!! + dist.lightSleepPercentage
            acc[SleepPhase.DEEP_SLEEP] = acc[SleepPhase.DEEP_SLEEP]!! + dist.deepSleepPercentage
            acc[SleepPhase.REM_SLEEP] = acc[SleepPhase.REM_SLEEP]!! + dist.remSleepPercentage
            acc
        }.mapValues { it.value / distributions.size }

        val entries = totalPhases.map { (phase, percentage) ->
            PieEntry(percentage, phase.getDisplayName())
        }

        val dataSet = PieDataSet(entries, "Sleep Phases").apply {
            colors = totalPhases.keys.map { phaseColors[it] ?: Color.GRAY }
            valueTextColor = Color.WHITE
            valueTextSize = 12f
            valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String = "${value.roundToInt()}%"
            }
            sliceSpace = 2f
            selectionShift = 8f
        }

        val pieData = PieData(dataSet)
        phaseDistributionPieChart.apply {
            data = pieData
            invalidate()
        }

        hideEmptyState()
    }

    private fun updateEfficiencyTrendsChart(efficiency: List<EfficiencyTrendData>) {
        if (efficiency.isEmpty()) {
            showEmptyState("No efficiency data available")
            return
        }

        val basicEntries = efficiency.mapIndexed { index, eff ->
            Entry(index.toFloat(), eff.basicEfficiency)
        }

        val adjustedEntries = efficiency.mapIndexed { index, eff ->
            Entry(index.toFloat(), eff.adjustedEfficiency)
        }

        val basicDataSet = LineDataSet(basicEntries, "Basic Efficiency").apply {
            color = chartColors[0]
            setCircleColor(chartColors[0])
            lineWidth = 2f
            circleRadius = 3f
            setDrawCircleHole(false)
            valueTextColor = Color.WHITE
            valueTextSize = 9f
        }

        val adjustedDataSet = LineDataSet(adjustedEntries, "Adjusted Efficiency").apply {
            color = chartColors[1]
            setCircleColor(chartColors[1])
            lineWidth = 2f
            circleRadius = 3f
            setDrawCircleHole(false)
            valueTextColor = Color.WHITE
            valueTextSize = 9f
        }

        val lineData = LineData(basicDataSet, adjustedDataSet)
        efficiencyLineChart.apply {
            data = lineData
            invalidate()

            // Update Y-axis to show percentage
            axisLeft.valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String = "${value.roundToInt()}%"
            }

            // Update X-axis labels
            xAxis.valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String {
                    val index = value.toInt()
                    return if (index >= 0 && index < efficiency.size) {
                        efficiency[index].formattedDate
                    } else ""
                }
            }
        }

        hideEmptyState()
    }

    private fun updateWeeklySummaryChart(weekly: List<WeeklyStatsData>) {
        if (weekly.isEmpty()) {
            showEmptyState("No weekly data available")
            return
        }

        val durationEntries = weekly.mapIndexed { index, week ->
            BarEntry(index.toFloat(), week.durationHours)
        }

        val qualityEntries = weekly.mapIndexed { index, week ->
            BarEntry(index.toFloat(), week.averageQuality)
        }

        val durationDataSet = BarDataSet(durationEntries, "Avg Duration (hours)").apply {
            color = chartColors[0]
            valueTextColor = Color.WHITE
            valueTextSize = 10f
        }

        val qualityDataSet = BarDataSet(qualityEntries, "Avg Quality (0-10)").apply {
            color = chartColors[1]
            valueTextColor = Color.WHITE
            valueTextSize = 10f
        }

        val barData = BarData(durationDataSet, qualityDataSet).apply {
            barWidth = 0.35f
        }

        weeklyBarChart.apply {
            data = barData
            groupBars(0f, 0.3f, 0.05f)
            invalidate()

            // Update X-axis labels
            xAxis.valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String {
                    val index = value.toInt()
                    return if (index >= 0 && index < weekly.size) {
                        "Week ${weekly[index].weekNumber}"
                    } else ""
                }
            }
        }

        hideEmptyState()
    }

    // ========== UI STATE MANAGEMENT ==========

    private fun updateLoadingState(loading: Boolean) {
        if (loading) {
            binding.loadingIndicator.visibility = View.VISIBLE
            binding.chartsContainer.alpha = 0.5f
            binding.chartTabLayout.isEnabled = false
        } else {
            binding.loadingIndicator.visibility = View.GONE
            binding.chartsContainer.alpha = 1f
            binding.chartTabLayout.isEnabled = true
        }
    }

    private fun showEmptyState(message: String) {
        binding.emptyStateLayout.visibility = View.VISIBLE
        binding.emptyStateText.text = message
        binding.chartsContainer.visibility = View.GONE
    }

    private fun hideEmptyState() {
        binding.emptyStateLayout.visibility = View.GONE
        binding.chartsContainer.visibility = View.VISIBLE
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    // ========== CHART INTERACTION ==========

    override fun onValueSelected(e: Entry?, h: Highlight?) {
        e?.let { entry ->
            // Handle chart value selection
            val info = when (currentChartType) {
                CHART_DURATION_TRENDS -> "Duration: ${entry.y} hours"
                CHART_QUALITY_FACTORS -> "Score: ${entry.y}/10"
                CHART_MOVEMENT_PATTERNS -> "Intensity: ${entry.y}"
                CHART_EFFICIENCY_TRENDS -> "Efficiency: ${entry.y}%"
                else -> "Value: ${entry.y}"
            }

            binding.chartValueInfo.text = info
            binding.chartValueInfo.visibility = View.VISIBLE
        }
    }

    override fun onNothingSelected() {
        binding.chartValueInfo.visibility = View.GONE
    }

    // ========== VALUE FORMATTERS ==========

    private inner class DateAxisValueFormatter : ValueFormatter() {
        private val dateFormat = SimpleDateFormat("MM/dd", Locale.getDefault())

        override fun getFormattedValue(value: Float): String {
            val daysSinceStart = value.toLong()
            val startDate = System.currentTimeMillis() - (currentDataRange * 24 * 60 * 60 * 1000L)
            val targetDate = startDate + (daysSinceStart * 24 * 60 * 60 * 1000L)
            return dateFormat.format(Date(targetDate))
        }
    }

    private inner class FactorAxisValueFormatter : ValueFormatter() {
        private val factors = listOf("Movement", "Noise", "Duration", "Consistency", "Efficiency", "Phase")

        override fun getFormattedValue(value: Float): String {
            val index = value.toInt()
            return if (index >= 0 && index < factors.size) factors[index] else ""
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        chartsViewModel.cleanup()
    }
}