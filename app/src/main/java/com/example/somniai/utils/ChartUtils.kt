package com.example.somniai.utils

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import androidx.core.content.ContextCompat
import com.example.somniai.R
import com.example.somniai.data.*
import com.github.mikephil.charting.animation.Easing
import com.github.mikephil.charting.charts.BarChart
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.components.*
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.IndexAxisValueFormatter
import com.github.mikephil.charting.formatter.PercentFormatter
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.highlight.Highlight
import com.github.mikephil.charting.listener.OnChartValueSelectedListener
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

/**
 * Comprehensive utility class for MPAndroidChart styling and configuration
 *
 * Provides:
 * - Dark theme optimized chart styling
 * - Pre-configured chart types for sleep data
 * - Data conversion and formatting utilities
 * - Animation and interaction configurations
 * - Color schemes for different metrics
 * - Accessibility and responsive design support
 */
object ChartUtils {

    // ========== COLOR SCHEMES ==========

    /**
     * Sleep quality color palette
     */
    object QualityColors {
        const val EXCELLENT = "#4CAF50"    // Green
        const val GOOD = "#2196F3"         // Blue
        const val FAIR = "#FF9800"         // Orange
        const val POOR = "#F44336"         // Red
        const val UNKNOWN = "#757575"      // Grey
    }

    /**
     * Sleep phase color palette
     */
    object PhaseColors {
        const val DEEP_SLEEP = "#1A237E"   // Deep Blue
        const val LIGHT_SLEEP = "#3F51B5"  // Indigo
        const val REM_SLEEP = "#9C27B0"    // Purple
        const val AWAKE = "#FF5722"        // Deep Orange
        const val TRANSITION = "#607D8B"   // Blue Grey
    }

    /**
     * Metric-specific colors
     */
    object MetricColors {
        const val DURATION = "#4CAF50"     // Green
        const val EFFICIENCY = "#2196F3"   // Blue
        const val MOVEMENT = "#FF9800"     // Orange
        const val NOISE = "#9C27B0"        // Purple
        const val QUALITY = "#FFD700"      // Gold
    }

    /**
     * Dark theme colors
     */
    object DarkTheme {
        const val BACKGROUND = "#121212"
        const val SURFACE = "#1E1E1E"
        const val TEXT_PRIMARY = "#FFFFFF"
        const val TEXT_SECONDARY = "#B3B3B3"
        const val GRID_LINE = "#333333"
        const val AXIS_LINE = "#555555"
    }

    // ========== CHART CONFIGURATION ==========

    /**
     * Configure LineChart for sleep trends
     */
    fun configureLineChart(
        chart: LineChart,
        context: Context,
        title: String = "",
        showLegend: Boolean = true,
        enableTouch: Boolean = true
    ) {
        chart.apply {
            // Basic configuration
            description.isEnabled = false
            setTouchEnabled(enableTouch)
            setPinchZoom(true)
            setScaleEnabled(true)
            setDrawGridBackground(false)
            setBackgroundColor(Color.parseColor(DarkTheme.BACKGROUND))

            // Animation
            animateX(1000, Easing.EaseInOutCubic)

            // Legend configuration
            legend.apply {
                isEnabled = showLegend
                textColor = Color.parseColor(DarkTheme.TEXT_PRIMARY)
                textSize = 12f
                form = Legend.LegendForm.CIRCLE
                formSize = 8f
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
                yOffset = 10f
            }

            // Configure axes
            configureLineChartAxes(this, context)

            // Marker for value display
            marker = createCustomMarker(context)
        }
    }

    /**
     * Configure BarChart for quality factors
     */
    fun configureBarChart(
        chart: BarChart,
        context: Context,
        title: String = "",
        showLegend: Boolean = true,
        enableTouch: Boolean = true
    ) {
        chart.apply {
            // Basic configuration
            description.isEnabled = false
            setTouchEnabled(enableTouch)
            setPinchZoom(false)
            setScaleEnabled(false)
            setDrawGridBackground(false)
            setBackgroundColor(Color.parseColor(DarkTheme.BACKGROUND))
            setDrawBarShadow(false)
            setDrawValueAboveBar(true)
            setFitBars(true)

            // Animation
            animateY(1000, Easing.EaseInOutCubic)

            // Legend configuration
            legend.apply {
                isEnabled = showLegend
                textColor = Color.parseColor(DarkTheme.TEXT_PRIMARY)
                textSize = 12f
                form = Legend.LegendForm.SQUARE
                formSize = 8f
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                verticalAlignment = Legend.LegendVerticalAlignment.TOP
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
                yOffset = 10f
            }

            // Configure axes
            configureBarChartAxes(this, context)

            // Marker for value display
            marker = createCustomMarker(context)
        }
    }

    /**
     * Configure PieChart for phase distribution
     */
    fun configurePieChart(
        chart: PieChart,
        context: Context,
        title: String = "",
        showLegend: Boolean = true,
        enableTouch: Boolean = true
    ) {
        chart.apply {
            // Basic configuration
            description.isEnabled = false
            setTouchEnabled(enableTouch)
            setDrawHoleEnabled(true)
            setHoleColor(Color.parseColor(DarkTheme.BACKGROUND))
            setTransparentCircleColor(Color.parseColor(DarkTheme.SURFACE))
            setHoleRadius(45f)
            setTransparentCircleRadius(50f)
            setDrawCenterText(true)
            setRotationAngle(0f)
            setRotationEnabled(true)
            setHighlightPerTapEnabled(true)
            setBackgroundColor(Color.parseColor(DarkTheme.BACKGROUND))

            // Center text
            centerText = title
            setCenterTextColor(Color.parseColor(DarkTheme.TEXT_PRIMARY))
            setCenterTextSize(16f)
            setCenterTextTypeface(Typeface.DEFAULT_BOLD)

            // Animation
            animateY(1000, Easing.EaseInOutCubic)

            // Legend configuration
            legend.apply {
                isEnabled = showLegend
                textColor = Color.parseColor(DarkTheme.TEXT_PRIMARY)
                textSize = 12f
                form = Legend.LegendForm.CIRCLE
                formSize = 8f
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
                yOffset = 10f
            }

            // Entry label styling
            setEntryLabelColor(Color.parseColor(DarkTheme.TEXT_PRIMARY))
            setEntryLabelTextSize(10f)
            setEntryLabelTypeface(Typeface.DEFAULT_BOLD)
            setDrawEntryLabels(false) // Hide labels on slices for cleaner look
        }
    }

    // ========== AXIS CONFIGURATION ==========

    private fun configureLineChartAxes(chart: LineChart, context: Context) {
        // X-Axis (bottom)
        chart.xAxis.apply {
            position = XAxis.XAxisPosition.BOTTOM
            textColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            textSize = 10f
            setDrawGridLines(true)
            gridColor = Color.parseColor(DarkTheme.GRID_LINE)
            axisLineColor = Color.parseColor(DarkTheme.AXIS_LINE)
            setDrawAxisLine(true)
            granularity = 1f
            labelRotationAngle = -45f
            setAvoidFirstLastClipping(true)
        }

        // Y-Axis (left)
        chart.axisLeft.apply {
            textColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            textSize = 10f
            setDrawGridLines(true)
            gridColor = Color.parseColor(DarkTheme.GRID_LINE)
            axisLineColor = Color.parseColor(DarkTheme.AXIS_LINE)
            setDrawAxisLine(true)
            setDrawZeroLine(true)
            zeroLineColor = Color.parseColor(DarkTheme.AXIS_LINE)
            setPosition(YAxis.YAxisLabelPosition.OUTSIDE_CHART)
        }

        // Y-Axis (right) - disabled
        chart.axisRight.isEnabled = false
    }

    private fun configureBarChartAxes(chart: BarChart, context: Context) {
        // X-Axis (bottom)
        chart.xAxis.apply {
            position = XAxis.XAxisPosition.BOTTOM
            textColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            textSize = 10f
            setDrawGridLines(false)
            axisLineColor = Color.parseColor(DarkTheme.AXIS_LINE)
            setDrawAxisLine(true)
            granularity = 1f
            labelRotationAngle = 0f
            setAvoidFirstLastClipping(true)
        }

        // Y-Axis (left)
        chart.axisLeft.apply {
            textColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            textSize = 10f
            setDrawGridLines(true)
            gridColor = Color.parseColor(DarkTheme.GRID_LINE)
            axisLineColor = Color.parseColor(DarkTheme.AXIS_LINE)
            setDrawAxisLine(true)
            setDrawZeroLine(true)
            zeroLineColor = Color.parseColor(DarkTheme.AXIS_LINE)
            setPosition(YAxis.YAxisLabelPosition.OUTSIDE_CHART)
            axisMinimum = 0f
        }

        // Y-Axis (right) - disabled
        chart.axisRight.isEnabled = false
    }

    // ========== DATA CONVERSION UTILITIES ==========

    /**
     * Convert session data to LineChart entries for duration trends
     */
    fun createDurationTrendData(sessions: List<SessionSummaryDTO>): LineData {
        val entries = sessions.mapIndexed { index, session ->
            val durationHours = session.totalDuration / (1000f * 60f * 60f)
            Entry(index.toFloat(), durationHours)
        }

        val dataSet = LineDataSet(entries, "Sleep Duration").apply {
            color = Color.parseColor(MetricColors.DURATION)
            setCircleColor(Color.parseColor(MetricColors.DURATION))
            lineWidth = 2.5f
            circleRadius = 4f
            setDrawCircleHole(false)
            valueTextSize = 9f
            valueTextColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            setDrawValues(false)
            setDrawFilled(true)
            fillColor = Color.parseColor(MetricColors.DURATION)
            fillAlpha = 30
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.2f
        }

        return LineData(dataSet)
    }

    /**
     * Convert session data to LineChart entries for quality trends
     */
    fun createQualityTrendData(sessions: List<SessionSummaryDTO>): LineData {
        val entries = sessions.mapIndexed { index, session ->
            Entry(index.toFloat(), session.qualityScore ?: 0f)
        }

        val dataSet = LineDataSet(entries, "Sleep Quality").apply {
            color = Color.parseColor(MetricColors.QUALITY)
            setCircleColor(Color.parseColor(MetricColors.QUALITY))
            lineWidth = 2.5f
            circleRadius = 4f
            setDrawCircleHole(false)
            valueTextSize = 9f
            valueTextColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            setDrawValues(false)
            setDrawFilled(true)
            fillColor = Color.parseColor(MetricColors.QUALITY)
            fillAlpha = 30
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.2f
        }

        return LineData(dataSet)
    }

    /**
     * Convert quality factors to BarChart data
     */
    fun createQualityFactorBarData(qualityFactors: QualityFactorBreakdown): BarData {
        val entries = listOf(
            BarEntry(0f, qualityFactors.durationScore),
            BarEntry(1f, qualityFactors.efficiencyScore),
            BarEntry(2f, qualityFactors.movementScore),
            BarEntry(3f, qualityFactors.noiseScore),
            BarEntry(4f, qualityFactors.consistencyScore)
        )

        val colors = listOf(
            Color.parseColor(MetricColors.DURATION),
            Color.parseColor(MetricColors.EFFICIENCY),
            Color.parseColor(MetricColors.MOVEMENT),
            Color.parseColor(MetricColors.NOISE),
            Color.parseColor(MetricColors.QUALITY)
        )

        val dataSet = BarDataSet(entries, "Quality Factors").apply {
            setColors(colors)
            valueTextSize = 10f
            valueTextColor = Color.parseColor(DarkTheme.TEXT_PRIMARY)
            valueFormatter = object : ValueFormatter() {
                override fun getFormattedValue(value: Float): String {
                    return String.format("%.1f", value)
                }
            }
        }

        return BarData(dataSet).apply {
            barWidth = 0.6f
        }
    }

    /**
     * Convert phase distribution to PieChart data
     */
    fun createPhaseDistributionData(phaseData: PhaseDistributionData): PieData {
        val entries = mutableListOf<PieEntry>()
        val colors = mutableListOf<Int>()

        phaseData.deepSleepPercentage?.let { percentage ->
            if (percentage > 0) {
                entries.add(PieEntry(percentage, "Deep Sleep"))
                colors.add(Color.parseColor(PhaseColors.DEEP_SLEEP))
            }
        }

        phaseData.lightSleepPercentage?.let { percentage ->
            if (percentage > 0) {
                entries.add(PieEntry(percentage, "Light Sleep"))
                colors.add(Color.parseColor(PhaseColors.LIGHT_SLEEP))
            }
        }

        phaseData.remSleepPercentage?.let { percentage ->
            if (percentage > 0) {
                entries.add(PieEntry(percentage, "REM Sleep"))
                colors.add(Color.parseColor(PhaseColors.REM_SLEEP))
            }
        }

        phaseData.awakePercentage?.let { percentage ->
            if (percentage > 0) {
                entries.add(PieEntry(percentage, "Awake"))
                colors.add(Color.parseColor(PhaseColors.AWAKE))
            }
        }

        val dataSet = PieDataSet(entries, "Sleep Phases").apply {
            setColors(colors)
            valueTextSize = 11f
            valueTextColor = Color.parseColor(DarkTheme.TEXT_PRIMARY)
            valueTypeface = Typeface.DEFAULT_BOLD
            valueFormatter = PercentFormatter()
            sliceSpace = 2f
            selectionShift = 8f
        }

        return PieData(dataSet)
    }

    /**
     * Create multi-line chart data for comprehensive trends
     */
    fun createMultiLineTrendData(trendData: List<DailyTrendData>): LineData {
        val dataSets = mutableListOf<LineDataSet>()

        // Duration trend
        val durationEntries = trendData.mapIndexed { index, data ->
            val hours = data.averageDuration / (1000f * 60f * 60f)
            Entry(index.toFloat(), hours)
        }
        dataSets.add(createLineDataSet(durationEntries, "Duration (hrs)", MetricColors.DURATION))

        // Quality trend
        val qualityEntries = trendData.mapIndexed { index, data ->
            Entry(index.toFloat(), data.averageQuality)
        }
        dataSets.add(createLineDataSet(qualityEntries, "Quality", MetricColors.QUALITY))

        // Efficiency trend (scaled to 0-10 range)
        val efficiencyEntries = trendData.mapIndexed { index, data ->
            Entry(index.toFloat(), data.averageEfficiency / 10f)
        }
        dataSets.add(createLineDataSet(efficiencyEntries, "Efficiency (/10)", MetricColors.EFFICIENCY))

        return LineData(dataSets)
    }

    private fun createLineDataSet(entries: List<Entry>, label: String, color: String): LineDataSet {
        return LineDataSet(entries, label).apply {
            this.color = Color.parseColor(color)
            setCircleColor(Color.parseColor(color))
            lineWidth = 2f
            circleRadius = 3f
            setDrawCircleHole(false)
            valueTextSize = 8f
            valueTextColor = Color.parseColor(DarkTheme.TEXT_SECONDARY)
            setDrawValues(false)
            setDrawFilled(false)
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.15f
        }
    }

    // ========== FORMATTERS ==========

    /**
     * Date formatter for X-axis labels
     */
    class DateAxisFormatter(private val dates: List<Date>) : IndexAxisValueFormatter() {
        private val formatter = SimpleDateFormat("MMM dd", Locale.getDefault())

        override fun getFormattedValue(value: Float): String {
            val index = value.roundToInt()
            return if (index >= 0 && index < dates.size) {
                formatter.format(dates[index])
            } else {
                ""
            }
        }
    }

    /**
     * Duration formatter for Y-axis (hours)
     */
    class DurationFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return "${value.roundToInt()}h"
        }
    }

    /**
     * Quality score formatter
     */
    class QualityFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return String.format("%.1f", value)
        }
    }

    /**
     * Percentage formatter for efficiency
     */
    class EfficiencyFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return "${value.roundToInt()}%"
        }
    }

    // ========== MARKERS ==========

    /**
     * Create custom marker for value display
     */
    private fun createCustomMarker(context: Context): IMarker {
        return object : MarkerView(context, R.layout.chart_marker) {
            override fun refreshContent(e: Entry?, highlight: Highlight?) {
                // Custom marker implementation would go here
                super.refreshContent(e, highlight)
            }
        }
    }

    // ========== HELPER METHODS ==========

    /**
     * Get quality color based on score
     */
    fun getQualityColor(score: Float): Int {
        return when {
            score >= 8f -> Color.parseColor(QualityColors.EXCELLENT)
            score >= 6f -> Color.parseColor(QualityColors.GOOD)
            score >= 4f -> Color.parseColor(QualityColors.FAIR)
            else -> Color.parseColor(QualityColors.POOR)
        }
    }

    /**
     * Get efficiency color based on percentage
     */
    fun getEfficiencyColor(efficiency: Float): Int {
        return when {
            efficiency >= 85f -> Color.parseColor(QualityColors.EXCELLENT)
            efficiency >= 70f -> Color.parseColor(QualityColors.GOOD)
            efficiency >= 50f -> Color.parseColor(QualityColors.FAIR)
            else -> Color.parseColor(QualityColors.POOR)
        }
    }

    /**
     * Set up chart interaction listeners
     */
    fun setupChartInteractions(
        chart: com.github.mikephil.charting.charts.Chart<*>,
        onValueSelected: ((Entry, Highlight) -> Unit)? = null,
        onNothingSelected: (() -> Unit)? = null
    ) {
        chart.setOnChartValueSelectedListener(object : OnChartValueSelectedListener {
            override fun onValueSelected(e: Entry?, h: Highlight?) {
                if (e != null && h != null) {
                    onValueSelected?.invoke(e, h)
                }
            }

            override fun onNothingSelected() {
                onNothingSelected?.invoke()
            }
        })
    }

    /**
     * Apply zoom and pan settings for better UX
     */
    fun configureChartGestures(chart: com.github.mikephil.charting.charts.Chart<*>) {
        chart.apply {
            setTouchEnabled(true)
            setDragEnabled(true)
            setScaleEnabled(true)
            setPinchZoom(true)
            setDoubleTapToZoomEnabled(true)
            setHighlightPerDragEnabled(false)
            setMaxHighlightDistance(500f)
        }
    }

    /**
     * Format session list into date-indexed entries
     */
    fun prepareSessionsForCharting(sessions: List<SessionSummaryDTO>): Pair<List<Date>, List<SessionSummaryDTO>> {
        val sortedSessions = sessions.sortedBy { it.startTime }
        val dates = sortedSessions.map { Date(it.startTime) }
        return Pair(dates, sortedSessions)
    }
}