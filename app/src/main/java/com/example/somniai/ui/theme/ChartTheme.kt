package com.example.somniai.ui.theme

import android.content.Context
import android.graphics.*
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import androidx.core.content.ContextCompat
import com.example.somniai.R
import com.github.mikephil.charting.charts.*
import com.github.mikephil.charting.components.*
import com.github.mikephil.charting.data.*
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.interfaces.datasets.*
import com.github.mikephil.charting.utils.ColorTemplate
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.*

/**
 * Enterprise-Grade Chart Theming System
 *
 * Features:
 * - Material Design 3 dark theme optimization
 * - Comprehensive analytics data visualization support
 * - Accessibility-first design with high contrast and semantic colors
 * - Advanced gradient and animation support
 * - Metric-specific color coding (quality, efficiency, movement, noise)
 * - Responsive typography and spacing
 * - Custom formatters for sleep data
 * - Theme variants for different chart contexts
 * - Performance-optimized rendering
 * - Extensible architecture for new chart types
 */
object ChartTheme {

    // ========== COLOR SYSTEM ==========

    /**
     * Core color palette optimized for dark theme sleep analytics
     */
    object Colors {
        // Primary Sleep Quality Colors
        const val QUALITY_EXCELLENT = 0xFF4CAF50    // Green - High quality sleep
        const val QUALITY_VERY_GOOD = 0xFF8BC34A    // Light Green
        const val QUALITY_GOOD = 0xFF2196F3        // Blue - Good quality
        const val QUALITY_FAIR = 0xFFFF9800        // Orange - Fair quality
        const val QUALITY_POOR = 0xFFFF5722        // Red Orange - Poor quality
        const val QUALITY_VERY_POOR = 0xFFF44336   // Red - Very poor quality

        // Sleep Efficiency Colors
        const val EFFICIENCY_OPTIMAL = 0xFF00E676   // Bright Green - 90%+
        const val EFFICIENCY_EXCELLENT = 0xFF4CAF50 // Green - 85-90%
        const val EFFICIENCY_GOOD = 0xFF2196F3     // Blue - 75-85%
        const val EFFICIENCY_FAIR = 0xFFFF9800     // Orange - 65-75%
        const val EFFICIENCY_POOR = 0xFFFF5722     // Red Orange - 50-65%
        const val EFFICIENCY_CRITICAL = 0xFFF44336 // Red - Below 50%

        // Sleep Phase Colors
        const val PHASE_DEEP_SLEEP = 0xFF1A237E     // Deep Blue - Deep sleep
        const val PHASE_REM_SLEEP = 0xFF7B1FA2      // Purple - REM sleep
        const val PHASE_LIGHT_SLEEP = 0xFF1976D2    // Light Blue - Light sleep
        const val PHASE_AWAKE = 0xFFE91E63          // Pink - Awake periods

        // Movement Analysis Colors
        const val MOVEMENT_MINIMAL = 0xFF4CAF50     // Green - Very still
        const val MOVEMENT_LOW = 0xFF8BC34A         // Light Green - Low movement
        const val MOVEMENT_MODERATE = 0xFFFF9800    // Orange - Moderate movement
        const val MOVEMENT_HIGH = 0xFFFF5722        // Red Orange - High movement
        const val MOVEMENT_EXCESSIVE = 0xFFF44336   // Red - Excessive movement

        // Noise Level Colors
        const val NOISE_SILENT = 0xFF4CAF50         // Green - Very quiet
        const val NOISE_QUIET = 0xFF8BC34A          // Light Green - Quiet
        const val NOISE_MODERATE = 0xFFFF9800       // Orange - Moderate noise
        const val NOISE_LOUD = 0xFFFF5722           // Red Orange - Loud
        const val NOISE_DISRUPTIVE = 0xFFF44336     // Red - Very disruptive

        // Background and Surface Colors
        const val BACKGROUND_PRIMARY = 0xFF121212    // Main dark background
        const val BACKGROUND_SECONDARY = 0xFF1E1E1E  // Card/surface background
        const val SURFACE_ELEVATED = 0xFF2C2C2C     // Elevated surfaces
        const val SURFACE_VARIANT = 0xFF404040      // Variant surfaces

        // Text and Content Colors
        const val TEXT_PRIMARY = 0xFFFFFFFF         // Primary text (white)
        const val TEXT_SECONDARY = 0xB3FFFFFF       // Secondary text (70% white)
        const val TEXT_TERTIARY = 0x99FFFFFF        // Tertiary text (60% white)
        const val TEXT_DISABLED = 0x66FFFFFF        // Disabled text (40% white)

        // Grid and Axis Colors
        const val GRID_PRIMARY = 0x33FFFFFF         // 20% white for major grids
        const val GRID_SECONDARY = 0x1AFFFFFF       // 10% white for minor grids
        const val AXIS_LINE = 0x66FFFFFF           // 40% white for axis lines
        const val AXIS_LABEL = 0x99FFFFFF          // 60% white for axis labels

        // Accent and Highlight Colors
        const val ACCENT_PRIMARY = 0xFF03DAC6       // Teal accent
        const val ACCENT_SECONDARY = 0xFF018786     // Dark teal
        const val HIGHLIGHT = 0xFFBB86FC            // Purple highlight
        const val WARNING = 0xFFFFB300              // Amber warning
        const val ERROR = 0xFFCF6679                // Pink error

        // Gradient Colors
        const val GRADIENT_START_QUALITY = 0xFF4CAF50
        const val GRADIENT_END_QUALITY = 0xFF81C784
        const val GRADIENT_START_EFFICIENCY = 0xFF2196F3
        const val GRADIENT_END_EFFICIENCY = 0xFF64B5F6
        const val GRADIENT_START_TREND_UP = 0xFF4CAF50
        const val GRADIENT_END_TREND_UP = 0xFF81C784
        const val GRADIENT_START_TREND_DOWN = 0xFFF44336
        const val GRADIENT_END_TREND_DOWN = 0xFFE57373
    }

    /**
     * Typography system for charts
     */
    object Typography {
        const val TITLE_SIZE = 18f
        const val SUBTITLE_SIZE = 14f
        const val AXIS_LABEL_SIZE = 12f
        const val VALUE_LABEL_SIZE = 10f
        const val LEGEND_SIZE = 11f
        const val ANNOTATION_SIZE = 9f

        // Font weights (approximate through Paint flags)
        const val FONT_NORMAL = 0
        const val FONT_MEDIUM = Paint.FAKE_BOLD_TEXT_FLAG
        const val FONT_BOLD = Paint.FAKE_BOLD_TEXT_FLAG
    }

    /**
     * Spacing and sizing constants
     */
    object Dimensions {
        const val CHART_PADDING = 16f
        const val LEGEND_MARGIN = 12f
        const val AXIS_MARGIN = 8f
        const val GRID_LINE_WIDTH = 1f
        const val AXIS_LINE_WIDTH = 2f
        const val DATA_LINE_WIDTH = 3f
        const val HIGHLIGHT_LINE_WIDTH = 4f
        const val MARKER_RADIUS = 4f
        const val BAR_CORNER_RADIUS = 4f
    }

    // ========== CHART STYLE DEFINITIONS ==========

    /**
     * Base chart styling that applies to all chart types
     */
    fun applyBaseChartStyle(chart: Chart<*>, context: Context) {
        chart.apply {
            // Background and general appearance
            setBackgroundColor(Colors.BACKGROUND_SECONDARY.toColorInt())
            description.isEnabled = false
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(false)
            setPinchZoom(false)
            setDrawGridBackground(false)
            setGridBackgroundColor(Colors.BACKGROUND_PRIMARY.toColorInt())

            // Animation
            animateXY(800, 800)

            // Legend styling
            legend.apply {
                isEnabled = true
                textColor = Colors.TEXT_SECONDARY.toColorInt()
                textSize = Typography.LEGEND_SIZE
                form = Legend.LegendForm.CIRCLE
                formSize = 8f
                xEntrySpace = 12f
                yEntrySpace = 4f
                verticalAlignment = Legend.LegendVerticalAlignment.BOTTOM
                horizontalAlignment = Legend.LegendHorizontalAlignment.CENTER
                orientation = Legend.LegendOrientation.HORIZONTAL
                setDrawInside(false)
            }

            // Extra offsets for better appearance
            setExtraOffsets(
                Dimensions.CHART_PADDING,
                Dimensions.CHART_PADDING,
                Dimensions.CHART_PADDING,
                Dimensions.CHART_PADDING + 40f // Extra bottom for legend
            )
        }
    }

    /**
     * Apply styling specific to line charts (trends, time series)
     */
    fun applyLineChartStyle(chart: LineChart, context: Context) {
        applyBaseChartStyle(chart, context)

        chart.apply {
            // X-Axis styling
            xAxis.apply {
                textColor = Colors.AXIS_LABEL.toColorInt()
                textSize = Typography.AXIS_LABEL_SIZE
                gridColor = Colors.GRID_SECONDARY.toColorInt()
                gridLineWidth = Dimensions.GRID_LINE_WIDTH
                axisLineColor = Colors.AXIS_LINE.toColorInt()
                axisLineWidth = Dimensions.AXIS_LINE_WIDTH
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(true)
                setDrawAxisLine(true)
                setDrawLabels(true)
                granularity = 1f
                setLabelCount(6, false)
                setAvoidFirstLastClipping(true)
            }

            // Y-Axis styling (left)
            axisLeft.apply {
                textColor = Colors.AXIS_LABEL.toColorInt()
                textSize = Typography.AXIS_LABEL_SIZE
                gridColor = Colors.GRID_PRIMARY.toColorInt()
                gridLineWidth = Dimensions.GRID_LINE_WIDTH
                axisLineColor = Colors.AXIS_LINE.toColorInt()
                axisLineWidth = Dimensions.AXIS_LINE_WIDTH
                setDrawGridLines(true)
                setDrawAxisLine(true)
                setDrawLabels(true)
                setDrawZeroLine(true)
                zeroLineColor = Colors.GRID_PRIMARY.toColorInt()
                zeroLineWidth = Dimensions.GRID_LINE_WIDTH
                setLabelCount(5, false)
                setPosition(YAxis.YAxisLabelPosition.OUTSIDE_CHART)
            }

            // Y-Axis styling (right) - disable
            axisRight.isEnabled = false

            // Enable marker view for data points
            setDrawMarkers(true)
        }
    }

    /**
     * Apply styling specific to bar charts (distributions, comparisons)
     */
    fun applyBarChartStyle(chart: BarChart, context: Context) {
        applyBaseChartStyle(chart, context)

        chart.apply {
            // X-Axis styling
            xAxis.apply {
                textColor = Colors.AXIS_LABEL.toColorInt()
                textSize = Typography.AXIS_LABEL_SIZE
                gridColor = Colors.GRID_SECONDARY.toColorInt()
                gridLineWidth = Dimensions.GRID_LINE_WIDTH
                axisLineColor = Colors.AXIS_LINE.toColorInt()
                axisLineWidth = Dimensions.AXIS_LINE_WIDTH
                position = XAxis.XAxisPosition.BOTTOM
                setDrawGridLines(false) // Cleaner look for bar charts
                setDrawAxisLine(true)
                setDrawLabels(true)
                granularity = 1f
                setAvoidFirstLastClipping(true)
            }

            // Y-Axis styling (left)
            axisLeft.apply {
                textColor = Colors.AXIS_LABEL.toColorInt()
                textSize = Typography.AXIS_LABEL_SIZE
                gridColor = Colors.GRID_PRIMARY.toColorInt()
                gridLineWidth = Dimensions.GRID_LINE_WIDTH
                axisLineColor = Colors.AXIS_LINE.toColorInt()
                axisLineWidth = Dimensions.AXIS_LINE_WIDTH
                setDrawGridLines(true)
                setDrawAxisLine(true)
                setDrawLabels(true)
                setDrawZeroLine(true)
                zeroLineColor = Colors.AXIS_LINE.toColorInt()
                zeroLineWidth = Dimensions.AXIS_LINE_WIDTH
                setLabelCount(5, false)
                axisMinimum = 0f
            }

            // Y-Axis styling (right) - disable
            axisRight.isEnabled = false

            // Bar-specific settings
            setFitBars(true)
            setDrawValueAboveBar(true)
            setDrawBarShadow(false)
        }
    }

    /**
     * Apply styling specific to pie charts (distributions, breakdowns)
     */
    fun applyPieChartStyle(chart: PieChart, context: Context) {
        applyBaseChartStyle(chart, context)

        chart.apply {
            // Pie-specific styling
            setUsePercentValues(true)
            isDrawHoleEnabled = true
            setHoleColor(Colors.BACKGROUND_SECONDARY.toColorInt())
            holeRadius = 35f
            transparentCircleRadius = 40f
            setTransparentCircleColor(Colors.BACKGROUND_PRIMARY.toColorInt())
            setTransparentCircleAlpha(50)

            // Center text styling
            setCenterTextTypeface(Typeface.DEFAULT_BOLD)
            setCenterTextSize(Typography.SUBTITLE_SIZE)
            setCenterTextColor(Colors.TEXT_PRIMARY.toColorInt())

            // Entry label styling
            setEntryLabelTypeface(Typeface.DEFAULT)
            setEntryLabelTextSize(Typography.VALUE_LABEL_SIZE)
            setEntryLabelColor(Colors.TEXT_PRIMARY.toColorInt())
            setDrawEntryLabels(true)

            // Rotation and interaction
            rotationAngle = 0f
            isRotationEnabled = true
            isHighlightPerTapEnabled = true

            // Animation
            animateY(1000)
        }
    }

    // ========== DATA SET STYLING ==========

    /**
     * Style a line dataset for quality trends
     */
    fun styleQualityTrendDataSet(dataSet: LineDataSet): LineDataSet {
        return dataSet.apply {
            color = Colors.QUALITY_GOOD.toColorInt()
            setCircleColor(Colors.QUALITY_EXCELLENT.toColorInt())
            lineWidth = Dimensions.DATA_LINE_WIDTH
            circleRadius = Dimensions.MARKER_RADIUS
            setDrawCircleHole(true)
            circleHoleRadius = Dimensions.MARKER_RADIUS * 0.6f
            circleHoleColor = Colors.BACKGROUND_SECONDARY.toColorInt()
            setDrawValues(false)
            setDrawHighlightIndicators(true)
            highlightLineWidth = Dimensions.HIGHLIGHT_LINE_WIDTH
            highLightColor = Colors.HIGHLIGHT.toColorInt()
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.2f

            // Gradient fill
            setDrawFilled(true)
            fillDrawable = createQualityGradient()
            fillAlpha = 180
        }
    }

    /**
     * Style a line dataset for efficiency trends
     */
    fun styleEfficiencyTrendDataSet(dataSet: LineDataSet): LineDataSet {
        return dataSet.apply {
            color = Colors.EFFICIENCY_EXCELLENT.toColorInt()
            setCircleColor(Colors.EFFICIENCY_OPTIMAL.toColorInt())
            lineWidth = Dimensions.DATA_LINE_WIDTH
            circleRadius = Dimensions.MARKER_RADIUS
            setDrawCircleHole(true)
            circleHoleRadius = Dimensions.MARKER_RADIUS * 0.6f
            circleHoleColor = Colors.BACKGROUND_SECONDARY.toColorInt()
            setDrawValues(false)
            setDrawHighlightIndicators(true)
            highlightLineWidth = Dimensions.HIGHLIGHT_LINE_WIDTH
            highLightColor = Colors.HIGHLIGHT.toColorInt()
            mode = LineDataSet.Mode.CUBIC_BEZIER
            cubicIntensity = 0.2f

            // Gradient fill
            setDrawFilled(true)
            fillDrawable = createEfficiencyGradient()
            fillAlpha = 180
        }
    }

    /**
     * Style a bar dataset for quality distribution
     */
    fun styleQualityDistributionDataSet(dataSet: BarDataSet): BarDataSet {
        return dataSet.apply {
            colors = getQualityGradientColors()
            setDrawValues(true)
            valueTextColor = Colors.TEXT_PRIMARY.toColorInt()
            valueTextSize = Typography.VALUE_LABEL_SIZE
            valueTypeface = Typeface.DEFAULT_BOLD
            highlightAlpha = 200
            setDrawIcons(false)
        }
    }

    /**
     * Style a pie dataset for sleep phase distribution
     */
    fun stylePhaseDistributionDataSet(dataSet: PieDataSet): PieDataSet {
        return dataSet.apply {
            colors = getSleepPhaseColors()
            setDrawValues(true)
            valueTextColor = Colors.TEXT_PRIMARY.toColorInt()
            valueTextSize = Typography.VALUE_LABEL_SIZE
            valueTypeface = Typeface.DEFAULT_BOLD
            valueLineColor = Colors.TEXT_SECONDARY.toColorInt()
            valueLinePart1OffsetPercentage = 80f
            valueLinePart1Length = 0.2f
            valueLinePart2Length = 0.4f
            isValueLineVariableLength = true
            sliceSpace = 2f
            selectionShift = 8f
        }
    }

    /**
     * Style a combined dataset for movement analysis
     */
    fun styleMovementAnalysisDataSet(dataSet: LineDataSet): LineDataSet {
        return dataSet.apply {
            color = Colors.MOVEMENT_MODERATE.toColorInt()
            setCircleColor(Colors.MOVEMENT_HIGH.toColorInt())
            lineWidth = Dimensions.DATA_LINE_WIDTH * 0.8f
            circleRadius = Dimensions.MARKER_RADIUS * 0.8f
            setDrawCircleHole(false)
            setDrawValues(false)
            setDrawHighlightIndicators(true)
            highlightLineWidth = Dimensions.HIGHLIGHT_LINE_WIDTH
            highLightColor = Colors.MOVEMENT_EXCESSIVE.toColorInt()
            mode = LineDataSet.Mode.LINEAR

            // Area fill with movement gradient
            setDrawFilled(true)
            fillDrawable = createMovementGradient()
            fillAlpha = 120
        }
    }

    /**
     * Style a dataset for noise level analysis
     */
    fun styleNoiseAnalysisDataSet(dataSet: LineDataSet): LineDataSet {
        return dataSet.apply {
            color = Colors.NOISE_MODERATE.toColorInt()
            setCircleColor(Colors.NOISE_LOUD.toColorInt())
            lineWidth = Dimensions.DATA_LINE_WIDTH * 0.8f
            circleRadius = Dimensions.MARKER_RADIUS * 0.8f
            setDrawCircleHole(false)
            setDrawValues(false)
            setDrawHighlightIndicators(true)
            highlightLineWidth = Dimensions.HIGHLIGHT_LINE_WIDTH
            highLightColor = Colors.NOISE_DISRUPTIVE.toColorInt()
            mode = LineDataSet.Mode.STEPPED

            // Area fill with noise gradient
            setDrawFilled(true)
            fillDrawable = createNoiseGradient()
            fillAlpha = 120
        }
    }

    // ========== COLOR UTILITIES ==========

    /**
     * Get quality-based color for a given score (0-10)
     */
    fun getQualityColor(score: Float): Int {
        return when {
            score >= 9f -> Colors.QUALITY_EXCELLENT.toColorInt()
            score >= 8f -> Colors.QUALITY_VERY_GOOD.toColorInt()
            score >= 7f -> Colors.QUALITY_GOOD.toColorInt()
            score >= 5f -> Colors.QUALITY_FAIR.toColorInt()
            score >= 3f -> Colors.QUALITY_POOR.toColorInt()
            else -> Colors.QUALITY_VERY_POOR.toColorInt()
        }
    }

    /**
     * Get efficiency-based color for a given percentage
     */
    fun getEfficiencyColor(efficiency: Float): Int {
        return when {
            efficiency >= 90f -> Colors.EFFICIENCY_OPTIMAL.toColorInt()
            efficiency >= 85f -> Colors.EFFICIENCY_EXCELLENT.toColorInt()
            efficiency >= 75f -> Colors.EFFICIENCY_GOOD.toColorInt()
            efficiency >= 65f -> Colors.EFFICIENCY_FAIR.toColorInt()
            efficiency >= 50f -> Colors.EFFICIENCY_POOR.toColorInt()
            else -> Colors.EFFICIENCY_CRITICAL.toColorInt()
        }
    }

    /**
     * Get movement intensity color
     */
    fun getMovementColor(intensity: Float): Int {
        return when {
            intensity <= 1.5f -> Colors.MOVEMENT_MINIMAL.toColorInt()
            intensity <= 2.5f -> Colors.MOVEMENT_LOW.toColorInt()
            intensity <= 4f -> Colors.MOVEMENT_MODERATE.toColorInt()
            intensity <= 6f -> Colors.MOVEMENT_HIGH.toColorInt()
            else -> Colors.MOVEMENT_EXCESSIVE.toColorInt()
        }
    }

    /**
     * Get noise level color
     */
    fun getNoiseColor(decibelLevel: Float): Int {
        return when {
            decibelLevel <= 30f -> Colors.NOISE_SILENT.toColorInt()
            decibelLevel <= 40f -> Colors.NOISE_QUIET.toColorInt()
            decibelLevel <= 50f -> Colors.NOISE_MODERATE.toColorInt()
            decibelLevel <= 60f -> Colors.NOISE_LOUD.toColorInt()
            else -> Colors.NOISE_DISRUPTIVE.toColorInt()
        }
    }

    /**
     * Get sleep phase color
     */
    fun getSleepPhaseColor(phase: SleepPhase): Int {
        return when (phase) {
            SleepPhase.DEEP_SLEEP -> Colors.PHASE_DEEP_SLEEP.toColorInt()
            SleepPhase.REM_SLEEP -> Colors.PHASE_REM_SLEEP.toColorInt()
            SleepPhase.LIGHT_SLEEP -> Colors.PHASE_LIGHT_SLEEP.toColorInt()
            SleepPhase.AWAKE -> Colors.PHASE_AWAKE.toColorInt()
            SleepPhase.UNKNOWN -> Colors.TEXT_DISABLED.toColorInt()
        }
    }

    /**
     * Get trend-based color (for improvement/decline indicators)
     */
    fun getTrendColor(trendDirection: TrendDirection): Int {
        return when (trendDirection) {
            TrendDirection.STRONGLY_IMPROVING, TrendDirection.IMPROVING -> Colors.QUALITY_EXCELLENT.toColorInt()
            TrendDirection.STABLE -> Colors.ACCENT_PRIMARY.toColorInt()
            TrendDirection.DECLINING, TrendDirection.STRONGLY_DECLINING -> Colors.ERROR.toColorInt()
            TrendDirection.INSUFFICIENT_DATA -> Colors.TEXT_DISABLED.toColorInt()
        }
    }

    // ========== GRADIENT CREATORS ==========

    private fun createQualityGradient(): GradientDrawable {
        return GradientDrawable().apply {
            orientation = GradientDrawable.Orientation.TOP_BOTTOM
            colors = intArrayOf(
                Colors.GRADIENT_START_QUALITY.toColorInt(),
                Colors.GRADIENT_END_QUALITY.toColorInt()
            )
        }
    }

    private fun createEfficiencyGradient(): GradientDrawable {
        return GradientDrawable().apply {
            orientation = GradientDrawable.Orientation.TOP_BOTTOM
            colors = intArrayOf(
                Colors.GRADIENT_START_EFFICIENCY.toColorInt(),
                Colors.GRADIENT_END_EFFICIENCY.toColorInt()
            )
        }
    }

    private fun createMovementGradient(): GradientDrawable {
        return GradientDrawable().apply {
            orientation = GradientDrawable.Orientation.TOP_BOTTOM
            colors = intArrayOf(
                Color.argb(100, 255, 152, 0), // Transparent orange
                Color.argb(50, 255, 152, 0)
            )
        }
    }

    private fun createNoiseGradient(): GradientDrawable {
        return GradientDrawable().apply {
            orientation = GradientDrawable.Orientation.TOP_BOTTOM
            colors = intArrayOf(
                Color.argb(100, 244, 67, 54), // Transparent red
                Color.argb(50, 244, 67, 54)
            )
        }
    }

    private fun createTrendGradient(isPositive: Boolean): GradientDrawable {
        return GradientDrawable().apply {
            orientation = GradientDrawable.Orientation.TOP_BOTTOM
            colors = if (isPositive) {
                intArrayOf(
                    Colors.GRADIENT_START_TREND_UP.toColorInt(),
                    Colors.GRADIENT_END_TREND_UP.toColorInt()
                )
            } else {
                intArrayOf(
                    Colors.GRADIENT_START_TREND_DOWN.toColorInt(),
                    Colors.GRADIENT_END_TREND_DOWN.toColorInt()
                )
            }
        }
    }

    // ========== COLOR ARRAYS ==========

    private fun getQualityGradientColors(): List<Int> {
        return listOf(
            Colors.QUALITY_VERY_POOR.toColorInt(),
            Colors.QUALITY_POOR.toColorInt(),
            Colors.QUALITY_FAIR.toColorInt(),
            Colors.QUALITY_GOOD.toColorInt(),
            Colors.QUALITY_VERY_GOOD.toColorInt(),
            Colors.QUALITY_EXCELLENT.toColorInt()
        )
    }

    private fun getSleepPhaseColors(): List<Int> {
        return listOf(
            Colors.PHASE_DEEP_SLEEP.toColorInt(),
            Colors.PHASE_REM_SLEEP.toColorInt(),
            Colors.PHASE_LIGHT_SLEEP.toColorInt(),
            Colors.PHASE_AWAKE.toColorInt()
        )
    }

    private fun getMovementColors(): List<Int> {
        return listOf(
            Colors.MOVEMENT_MINIMAL.toColorInt(),
            Colors.MOVEMENT_LOW.toColorInt(),
            Colors.MOVEMENT_MODERATE.toColorInt(),
            Colors.MOVEMENT_HIGH.toColorInt(),
            Colors.MOVEMENT_EXCESSIVE.toColorInt()
        )
    }

    private fun getNoiseColors(): List<Int> {
        return listOf(
            Colors.NOISE_SILENT.toColorInt(),
            Colors.NOISE_QUIET.toColorInt(),
            Colors.NOISE_MODERATE.toColorInt(),
            Colors.NOISE_LOUD.toColorInt(),
            Colors.NOISE_DISRUPTIVE.toColorInt()
        )
    }

    // ========== CUSTOM FORMATTERS ==========

    /**
     * Value formatter for sleep duration (hours and minutes)
     */
    class DurationFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            val hours = value.toInt()
            val minutes = ((value - hours) * 60).toInt()
            return if (hours > 0) "${hours}h ${minutes}m" else "${minutes}m"
        }
    }

    /**
     * Value formatter for quality scores (0-10 scale)
     */
    class QualityFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return String.format("%.1f", value)
        }
    }

    /**
     * Value formatter for efficiency percentages
     */
    class EfficiencyFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return "${value.toInt()}%"
        }
    }

    /**
     * Value formatter for movement intensity
     */
    class MovementFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return String.format("%.1f", value)
        }
    }

    /**
     * Value formatter for noise levels (decibels)
     */
    class NoiseFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return "${value.toInt()} dB"
        }
    }

    /**
     * Date formatter for X-axis time series
     */
    class DateFormatter : ValueFormatter() {
        private val dateFormat = SimpleDateFormat("MMM dd", Locale.getDefault())

        override fun getFormattedValue(value: Float): String {
            return dateFormat.format(Date(value.toLong()))
        }
    }

    /**
     * Time formatter for X-axis within a day
     */
    class TimeFormatter : ValueFormatter() {
        private val timeFormat = SimpleDateFormat("HH:mm", Locale.getDefault())

        override fun getFormattedValue(value: Float): String {
            return timeFormat.format(Date(value.toLong()))
        }
    }

    /**
     * Phase formatter for sleep phases
     */
    class PhaseFormatter : ValueFormatter() {
        override fun getFormattedValue(value: Float): String {
            return when (value.toInt()) {
                0 -> "Deep"
                1 -> "REM"
                2 -> "Light"
                3 -> "Awake"
                else -> "Unknown"
            }
        }
    }

    // ========== ACCESSIBILITY UTILITIES ==========

    /**
     * Ensure color has sufficient contrast for accessibility
     */
    fun ensureContrast(color: Int, backgroundColor: Int = Colors.BACKGROUND_SECONDARY.toColorInt()): Int {
        val contrast = calculateContrast(color, backgroundColor)
        return if (contrast < 4.5) { // WCAG AA standard
            adjustColorForContrast(color, backgroundColor)
        } else {
            color
        }
    }

    /**
     * Calculate contrast ratio between two colors
     */
    private fun calculateContrast(color1: Int, color2: Int): Double {
        val l1 = getLuminance(color1)
        val l2 = getLuminance(color2)
        val lighter = maxOf(l1, l2)
        val darker = minOf(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)
    }

    /**
     * Get relative luminance of a color
     */
    private fun getLuminance(color: Int): Double {
        val r = Color.red(color) / 255.0
        val g = Color.green(color) / 255.0
        val b = Color.blue(color) / 255.0

        fun adjustComponent(c: Double): Double {
            return if (c <= 0.03928) c / 12.92 else ((c + 0.055) / 1.055).pow(2.4)
        }

        return 0.2126 * adjustComponent(r) + 0.7152 * adjustComponent(g) + 0.0722 * adjustComponent(b)
    }

    /**
     * Adjust color to meet contrast requirements
     */
    private fun adjustColorForContrast(color: Int, backgroundColor: Int): Int {
        val hsv = FloatArray(3)
        Color.colorToHSV(color, hsv)

        // Adjust value (brightness) to improve contrast
        var adjustedColor = color
        var step = 0.1f

        repeat(10) { // Max 10 iterations to find suitable contrast
            if (calculateContrast(adjustedColor, backgroundColor) >= 4.5) {
                return adjustedColor
            }

            hsv[2] = (hsv[2] + step).coerceIn(0f, 1f)
            adjustedColor = Color.HSVToColor(hsv)

            if (hsv[2] >= 1f || hsv[2] <= 0f) {
                step = -step // Reverse direction if we hit limits
            }
        }

        return adjustedColor
    }

    // ========== CHART TYPE PRESETS ==========

    /**
     * Complete styling preset for sleep quality trend charts
     */
    fun applySleepQualityTrendStyle(chart: LineChart, context: Context) {
        applyLineChartStyle(chart, context)

        chart.apply {
            // Specific styling for quality trends
            axisLeft.apply {
                axisMinimum = 0f
                axisMaximum = 10f
                valueFormatter = QualityFormatter()
                setLabelCount(6, true)
            }

            xAxis.valueFormatter = DateFormatter()

            // Add quality zones as limit lines
            axisLeft.addLimitLine(LimitLine(8f, "Excellent").apply {
                lineColor = Colors.QUALITY_EXCELLENT.toColorInt()
                lineWidth = 2f
                textColor = Colors.QUALITY_EXCELLENT.toColorInt()
                textSize = Typography.ANNOTATION_SIZE
            })

            axisLeft.addLimitLine(LimitLine(6f, "Good").apply {
                lineColor = Colors.QUALITY_GOOD.toColorInt()
                lineWidth = 1f
                textColor = Colors.QUALITY_GOOD.toColorInt()
                textSize = Typography.ANNOTATION_SIZE
            })
        }
    }

    /**
     * Complete styling preset for sleep efficiency charts
     */
    fun applySleepEfficiencyStyle(chart: LineChart, context: Context) {
        applyLineChartStyle(chart, context)

        chart.apply {
            axisLeft.apply {
                axisMinimum = 0f
                axisMaximum = 100f
                valueFormatter = EfficiencyFormatter()
                setLabelCount(6, true)
            }

            xAxis.valueFormatter = DateFormatter()

            // Add efficiency benchmarks
            axisLeft.addLimitLine(LimitLine(85f, "Target").apply {
                lineColor = Colors.EFFICIENCY_EXCELLENT.toColorInt()
                lineWidth = 2f
                textColor = Colors.EFFICIENCY_EXCELLENT.toColorInt()
                textSize = Typography.ANNOTATION_SIZE
            })
        }
    }

    /**
     * Complete styling preset for movement analysis charts
     */
    fun applyMovementAnalysisStyle(chart: LineChart, context: Context) {
        applyLineChartStyle(chart, context)

        chart.apply {
            axisLeft.apply {
                axisMinimum = 0f
                valueFormatter = MovementFormatter()
            }

            xAxis.valueFormatter = TimeFormatter()
        }
    }

    /**
     * Complete styling preset for phase distribution pie charts
     */
    fun applyPhaseDistributionStyle(chart: PieChart, context: Context) {
        applyPieChartStyle(chart, context)

        chart.apply {
            setCenterText("Sleep\nPhases")
            setEntryLabelTextSize(Typography.VALUE_LABEL_SIZE)
        }
    }

    // ========== ANIMATION UTILITIES ==========

    /**
     * Custom animation for data changes
     */
    fun animateDataChange(chart: Chart<*>, duration: Int = 800) {
        chart.animateXY(duration, duration)
    }

    /**
     * Highlight animation for specific data points
     */
    fun animateHighlight(chart: Chart<*>, entry: Entry) {
        chart.highlightValue(entry.x, entry.y, 0)
    }

    // ========== EXTENSION FUNCTIONS ==========

    /**
     * Extension function to convert Long color values to Int safely
     */
    private fun Long.toColorInt(): Int = this.toInt()
}

// ========== ENUMS FOR DATA TYPES ==========

/**
 * Sleep phase enumeration
 */
enum class SleepPhase {
    DEEP_SLEEP,
    REM_SLEEP,
    LIGHT_SLEEP,
    AWAKE,
    UNKNOWN
}

/**
 * Trend direction enumeration
 */
enum class TrendDirection {
    STRONGLY_IMPROVING,
    IMPROVING,
    STABLE,
    DECLINING,
    STRONGLY_DECLINING,
    INSUFFICIENT_DATA
}

/**
 * Extension functions for easier chart theming
 */

/**
 * Apply sleep analytics theme to any line chart
 */
fun LineChart.applySleepTheme(context: Context, chartType: SleepChartType = SleepChartType.QUALITY_TREND) {
    when (chartType) {
        SleepChartType.QUALITY_TREND -> ChartTheme.applySleepQualityTrendStyle(this, context)
        SleepChartType.EFFICIENCY_TREND -> ChartTheme.applySleepEfficiencyStyle(this, context)
        SleepChartType.MOVEMENT_ANALYSIS -> ChartTheme.applyMovementAnalysisStyle(this, context)
        SleepChartType.GENERIC -> ChartTheme.applyLineChartStyle(this, context)
    }
}

/**
 * Apply sleep analytics theme to any bar chart
 */
fun BarChart.applySleepTheme(context: Context) {
    ChartTheme.applyBarChartStyle(this, context)
}

/**
 * Apply sleep analytics theme to any pie chart
 */
fun PieChart.applySleepTheme(context: Context, chartType: SleepPieChartType = SleepPieChartType.PHASE_DISTRIBUTION) {
    when (chartType) {
        SleepPieChartType.PHASE_DISTRIBUTION -> ChartTheme.applyPhaseDistributionStyle(this, context)
        SleepPieChartType.GENERIC -> ChartTheme.applyPieChartStyle(this, context)
    }
}

/**
 * Chart type enums for theme application
 */
enum class SleepChartType {
    QUALITY_TREND,
    EFFICIENCY_TREND,
    MOVEMENT_ANALYSIS,
    GENERIC
}

enum class SleepPieChartType {
    PHASE_DISTRIBUTION,
    GENERIC
}