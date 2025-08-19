package com.example.somniai.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import kotlin.math.sqrt

/**
 * Data class representing a movement event detected by the accelerometer
 */
data class MovementEvent(
    val timestamp: Long,
    val intensity: Float,
    val x: Float,
    val y: Float,
    val z: Float
)

/**
 * Accelerometer-based motion detector for sleep tracking
 * Monitors device movement and detects significant motion events during sleep
 */
class SleepMotionDetector(
    private val context: Context,
    private val onMovementDetected: (MovementEvent) -> Unit
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    // Movement detection configuration
    private var movementThreshold = DEFAULT_MOVEMENT_THRESHOLD
    private var lastMovementTime = 0L
    private val minimumTimeBetweenEvents = MIN_TIME_BETWEEN_EVENTS

    // Baseline calculation for filtering gravity/orientation changes
    private val recentValues = mutableListOf<Float>()
    private val maxRecentValues = BASELINE_WINDOW_SIZE
    private var baseline = GRAVITY_BASELINE

    // Monitoring state
    var isMonitoring = false
        private set

    // Statistics
    private var totalMovements = 0
    private var lastIntensity = 0f

    /**
     * Start accelerometer monitoring
     * @return true if monitoring started successfully, false otherwise
     */
    fun startMonitoring(): Boolean {
        return if (accelerometer != null) {
            val registered = sensorManager.registerListener(
                this,
                accelerometer,
                SensorManager.SENSOR_DELAY_NORMAL
            )

            if (registered) {
                isMonitoring = true
                totalMovements = 0
                recentValues.clear()
                baseline = GRAVITY_BASELINE
                Log.d(TAG, "Accelerometer monitoring started with threshold: $movementThreshold")
            } else {
                Log.e(TAG, "Failed to register accelerometer listener")
            }

            registered
        } else {
            Log.e(TAG, "Accelerometer sensor not available on this device")
            false
        }
    }

    /**
     * Stop accelerometer monitoring
     */
    fun stopMonitoring() {
        if (isMonitoring) {
            sensorManager.unregisterListener(this)
            isMonitoring = false
            recentValues.clear()
            Log.d(TAG, "Accelerometer monitoring stopped. Total movements detected: $totalMovements")
        }
    }

    /**
     * Set the movement detection threshold
     * @param threshold Sensitivity level (lower = more sensitive)
     */
    fun setMovementThreshold(threshold: Float) {
        movementThreshold = threshold.coerceIn(0.5f, 10.0f)
        Log.d(TAG, "Movement threshold updated to: $movementThreshold")
    }

    /**
     * Get current movement sensitivity level as human-readable string
     */
    fun getSensitivityLevel(): String {
        return when {
            movementThreshold <= 1.0f -> "Very High"
            movementThreshold <= 2.0f -> "High"
            movementThreshold <= 3.0f -> "Medium"
            movementThreshold <= 4.0f -> "Low"
            else -> "Very Low"
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER && isMonitoring) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]

            // Calculate total acceleration magnitude
            val magnitude = sqrt(x * x + y * y + z * z)

            // Update baseline using rolling average
            updateBaseline(magnitude)

            // Calculate deviation from baseline (filters out gravity/orientation)
            val deviation = kotlin.math.abs(magnitude - baseline)
            lastIntensity = deviation

            // Detect significant movement
            if (isSignificantMovement(deviation)) {
                val currentTime = System.currentTimeMillis()

                // Debounce - ensure minimum time between events
                if (currentTime - lastMovementTime >= minimumTimeBetweenEvents) {
                    lastMovementTime = currentTime
                    totalMovements++

                    val movementEvent = MovementEvent(
                        timestamp = currentTime,
                        intensity = deviation,
                        x = x,
                        y = y,
                        z = z
                    )

                    Log.d(TAG, "Movement #$totalMovements detected - Intensity: %.2f (threshold: %.2f)".format(deviation, movementThreshold))
                    onMovementDetected(movementEvent)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        when (accuracy) {
            SensorManager.SENSOR_STATUS_ACCURACY_HIGH ->
                Log.d(TAG, "Accelerometer accuracy: HIGH")
            SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM ->
                Log.d(TAG, "Accelerometer accuracy: MEDIUM")
            SensorManager.SENSOR_STATUS_ACCURACY_LOW ->
                Log.w(TAG, "Accelerometer accuracy: LOW - readings may be unreliable")
            SensorManager.SENSOR_STATUS_UNRELIABLE ->
                Log.e(TAG, "Accelerometer accuracy: UNRELIABLE")
        }
    }

    /**
     * Update the baseline gravity reading using a rolling average
     * This helps filter out device orientation changes from actual movement
     */
    private fun updateBaseline(magnitude: Float) {
        recentValues.add(magnitude)

        // Keep only recent values for baseline calculation
        if (recentValues.size > maxRecentValues) {
            recentValues.removeAt(0)
        }

        // Calculate baseline as moving average (more stable than instant values)
        if (recentValues.size >= MIN_VALUES_FOR_BASELINE) {
            // Use median of recent values for more robust baseline
            val sortedValues = recentValues.sorted()
            baseline = if (sortedValues.size % 2 == 0) {
                // Even number - average of middle two
                val mid = sortedValues.size / 2
                (sortedValues[mid - 1] + sortedValues[mid]) / 2f
            } else {
                // Odd number - middle value
                sortedValues[sortedValues.size / 2]
            }
        }
    }

    /**
     * Determine if detected change represents significant movement
     */
    private fun isSignificantMovement(deviation: Float): Boolean {
        return deviation > movementThreshold
    }

    /**
     * Get current movement intensity (real-time)
     */
    fun getCurrentMovementIntensity(): Float {
        return lastIntensity
    }

    /**
     * Get total number of movements detected in current session
     */
    fun getTotalMovements(): Int {
        return totalMovements
    }

    /**
     * Get current baseline gravity reading
     */
    fun getCurrentBaseline(): Float {
        return baseline
    }

    /**
     * Get monitoring status as human-readable string
     */
    fun getMonitoringStatus(): String {
        return when {
            !isMonitoring -> "Inactive"
            accelerometer == null -> "Sensor Unavailable"
            recentValues.size < MIN_VALUES_FOR_BASELINE -> "Calibrating..."
            else -> "Active (${getSensitivityLevel()} sensitivity)"
        }
    }

    /**
     * Check if accelerometer sensor is available on device
     */
    fun isSensorAvailable(): Boolean {
        return accelerometer != null
    }

    /**
     * Get sensor information for debugging
     */
    fun getSensorInfo(): String {
        return if (accelerometer != null) {
            "Accelerometer: ${accelerometer.name} (${accelerometer.vendor})"
        } else {
            "Accelerometer not available"
        }
    }

    companion object {
        private const val TAG = "SleepMotionDetector"

        // Configuration constants
        private const val DEFAULT_MOVEMENT_THRESHOLD = 2.0f
        private const val MIN_TIME_BETWEEN_EVENTS = 1000L // 1 second
        private const val BASELINE_WINDOW_SIZE = 20
        private const val MIN_VALUES_FOR_BASELINE = 5
        private const val GRAVITY_BASELINE = 9.8f // Standard gravity

        // Sensitivity presets
        const val SENSITIVITY_VERY_HIGH = 0.8f
        const val SENSITIVITY_HIGH = 1.5f
        const val SENSITIVITY_MEDIUM = 2.5f
        const val SENSITIVITY_LOW = 4.0f
        const val SENSITIVITY_VERY_LOW = 6.0f
    }
}