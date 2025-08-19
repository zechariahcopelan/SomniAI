package com.example.somniai.sensor

import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.*
import java.io.IOException

/**
 * Data class representing a noise event detected by the microphone
 */
data class NoiseEvent(
    val timestamp: Long,
    val decibelLevel: Float,
    val amplitude: Int
)

/**
 * Microphone amplitude monitor for sleep tracking
 * Monitors ambient noise levels without recording audio (privacy-safe)
 * Only measures volume amplitude, no audio data is stored
 */
class AudioLevelMonitor(
    private val onNoiseDetected: (NoiseEvent) -> Unit
) {

    private var mediaRecorder: MediaRecorder? = null
    private var monitoringJob: Job? = null
    private var isMonitoring = false

    // Noise detection configuration
    private var noiseThreshold = DEFAULT_NOISE_THRESHOLD
    private var samplingInterval = DEFAULT_SAMPLING_INTERVAL
    private var lastNoiseTime = 0L
    private val minimumTimeBetweenEvents = MIN_TIME_BETWEEN_EVENTS

    // Baseline calculation for ambient noise level
    private val recentAmplitudes = mutableListOf<Int>()
    private val maxRecentAmplitudes = BASELINE_WINDOW_SIZE
    private var baselineAmplitude = 0
    private var maxAmplitudeRecorded = 0

    // Statistics
    private var totalNoiseEvents = 0
    private var currentAmplitude = 0

    /**
     * Start monitoring microphone amplitude levels
     * @return true if monitoring started successfully, false otherwise
     */
    fun startMonitoring(): Boolean {
        if (isMonitoring) {
            Log.w(TAG, "Audio monitoring already active")
            return true
        }

        return try {
            initializeMediaRecorder()
            startAmplitudeMonitoring()
            isMonitoring = true
            totalNoiseEvents = 0
            maxAmplitudeRecorded = 0
            Log.d(TAG, "Audio level monitoring started (threshold: $noiseThreshold)")
            true
        } catch (e: SecurityException) {
            Log.e(TAG, "Missing microphone permission", e)
            cleanup()
            false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start audio monitoring", e)
            cleanup()
            false
        }
    }

    /**
     * Stop monitoring microphone amplitude
     */
    fun stopMonitoring() {
        if (!isMonitoring) return

        isMonitoring = false
        monitoringJob?.cancel()
        cleanup()
        recentAmplitudes.clear()
        Log.d(TAG, "Audio monitoring stopped. Events detected: $totalNoiseEvents, Max amplitude: $maxAmplitudeRecorded")
    }

    /**
     * Set noise detection threshold
     * @param threshold Amplitude threshold for noise detection
     */
    fun setNoiseThreshold(threshold: Int) {
        noiseThreshold = threshold.coerceIn(100, 10000)
        Log.d(TAG, "Noise threshold updated to: $noiseThreshold")
    }

    /**
     * Set sampling interval for amplitude checks
     * @param intervalMs Interval between amplitude readings in milliseconds
     */
    fun setSamplingInterval(intervalMs: Long) {
        samplingInterval = intervalMs.coerceIn(500L, 5000L)
        Log.d(TAG, "Sampling interval updated to: ${samplingInterval}ms")
    }

    /**
     * Get current noise sensitivity level as human-readable string
     */
    fun getSensitivityLevel(): String {
        return when {
            noiseThreshold <= 500 -> "Very High"
            noiseThreshold <= 1000 -> "High"
            noiseThreshold <= 1500 -> "Medium"
            noiseThreshold <= 2000 -> "Low"
            else -> "Very Low"
        }
    }

    /**
     * Initialize MediaRecorder for amplitude monitoring only
     * No audio data is recorded - output goes to /dev/null
     */
    private fun initializeMediaRecorder() {
        mediaRecorder = MediaRecorder().apply {
            try {
                // Configure for amplitude monitoring only
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)

                // CRITICAL: Output to /dev/null - no audio file is created
                // We only want amplitude readings, not recorded audio
                setOutputFile("/dev/null")

                prepare()
                start()

                Log.d(TAG, "MediaRecorder initialized for amplitude monitoring (no recording)")
            } catch (e: IOException) {
                Log.e(TAG, "MediaRecorder initialization failed", e)
                throw e
            } catch (e: SecurityException) {
                Log.e(TAG, "Microphone permission required", e)
                throw e
            }
        }
    }

    /**
     * Start continuous amplitude monitoring in background
     */
    private fun startAmplitudeMonitoring() {
        monitoringJob = CoroutineScope(Dispatchers.IO).launch {
            Log.d(TAG, "Starting amplitude monitoring loop")

            while (isActive && isMonitoring) {
                try {
                    val amplitude = getAmplitude()
                    currentAmplitude = amplitude

                    // Track maximum amplitude for statistics
                    if (amplitude > maxAmplitudeRecorded) {
                        maxAmplitudeRecorded = amplitude
                    }

                    updateBaseline(amplitude)
                    processAmplitude(amplitude)

                    delay(samplingInterval)
                } catch (e: Exception) {
                    Log.e(TAG, "Error during amplitude monitoring", e)
                    if (e is IllegalStateException) {
                        // MediaRecorder in invalid state - restart monitoring
                        Log.w(TAG, "MediaRecorder in invalid state, attempting restart")
                        break
                    }
                }
            }

            Log.d(TAG, "Amplitude monitoring loop ended")
        }
    }

    /**
     * Get current amplitude from MediaRecorder
     * @return Current amplitude level (0-32767)
     */
    private fun getAmplitude(): Int {
        return try {
            mediaRecorder?.maxAmplitude ?: 0
        } catch (e: IllegalStateException) {
            Log.w(TAG, "MediaRecorder not in valid state for amplitude reading")
            0
        }
    }

    /**
     * Update baseline ambient noise level
     */
    private fun updateBaseline(amplitude: Int) {
        recentAmplitudes.add(amplitude)

        // Keep only recent values for baseline calculation
        if (recentAmplitudes.size > maxRecentAmplitudes) {
            recentAmplitudes.removeAt(0)
        }

        // Calculate baseline as median of lower percentile (quiet periods)
        if (recentAmplitudes.size >= MIN_VALUES_FOR_BASELINE) {
            val sortedAmplitudes = recentAmplitudes.sorted()

            // Use 25th percentile as baseline (represents quiet ambient level)
            val percentile25Index = (sortedAmplitudes.size * 0.25).toInt()
            baselineAmplitude = sortedAmplitudes[percentile25Index]
        }
    }

    /**
     * Process amplitude reading and detect noise events
     */
    private fun processAmplitude(amplitude: Int) {
        val currentTime = System.currentTimeMillis()
        val deviationFromBaseline = amplitude - baselineAmplitude

        // Detect significant noise above threshold and baseline
        if (isSignificantNoise(amplitude, deviationFromBaseline)) {
            // Debounce - ensure minimum time between events
            if (currentTime - lastNoiseTime >= minimumTimeBetweenEvents) {
                lastNoiseTime = currentTime
                totalNoiseEvents++

                // Convert amplitude to approximate decibel level
                val decibelLevel = amplitudeToDecibels(amplitude)

                val noiseEvent = NoiseEvent(
                    timestamp = currentTime,
                    decibelLevel = decibelLevel,
                    amplitude = amplitude
                )

                Log.d(TAG, "Noise #$totalNoiseEvents detected - Amplitude: $amplitude, ~${decibelLevel.toInt()}dB (baseline: $baselineAmplitude)")
                onNoiseDetected(noiseEvent)
            }
        }
    }

    /**
     * Determine if amplitude represents significant noise
     */
    private fun isSignificantNoise(amplitude: Int, deviationFromBaseline: Int): Boolean {
        return amplitude > noiseThreshold &&
                deviationFromBaseline > (noiseThreshold * 0.3f).toInt() // Must be above baseline + 30% of threshold
    }

    /**
     * Convert amplitude to approximate decibel level
     * Note: This is a rough approximation and varies by device
     */
    private fun amplitudeToDecibels(amplitude: Int): Float {
        return if (amplitude > 0) {
            // Convert to dB scale (approximate)
            // Formula: 20 * log10(amplitude / reference) + offset
            val normalized = amplitude.toDouble() / 32767.0 // Normalize to 0-1
            val db = 20 * kotlin.math.log10(normalized.coerceAtLeast(0.001)) + 90
            db.toFloat().coerceAtLeast(0f)
        } else {
            0f
        }
    }

    /**
     * Clean up MediaRecorder resources
     */
    private fun cleanup() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error during MediaRecorder cleanup", e)
        } finally {
            mediaRecorder = null
        }
    }

    /**
     * Get current amplitude reading (real-time)
     */
    fun getCurrentAmplitude(): Int {
        return currentAmplitude
    }

    /**
     * Get current baseline ambient noise level
     */
    fun getBaselineAmplitude(): Int {
        return baselineAmplitude
    }

    /**
     * Get maximum amplitude recorded in current session
     */
    fun getMaxAmplitude(): Int {
        return maxAmplitudeRecorded
    }

    /**
     * Get total number of noise events detected
     */
    fun getTotalNoiseEvents(): Int {
        return totalNoiseEvents
    }

    /**
     * Get current amplitude as approximate decibel level
     */
    fun getCurrentDecibelLevel(): Float {
        return amplitudeToDecibels(currentAmplitude)
    }

    /**
     * Get monitoring status as human-readable string
     */
    fun getMonitoringStatus(): String {
        return when {
            !isMonitoring -> "Inactive"
            mediaRecorder == null -> "Microphone Unavailable"
            recentAmplitudes.size < MIN_VALUES_FOR_BASELINE -> "Calibrating..."
            else -> "Active (${getSensitivityLevel()} sensitivity)"
        }
    }

    /**
     * Get detailed status for debugging
     */
    fun getDetailedStatus(): String {
        return if (isMonitoring) {
            "Monitoring: threshold=$noiseThreshold, baseline=$baselineAmplitude, " +
                    "current=$currentAmplitude, events=$totalNoiseEvents, max=$maxAmplitudeRecorded"
        } else {
            "Not monitoring"
        }
    }

    /**
     * Check if monitoring is healthy (MediaRecorder responding)
     */
    fun isHealthy(): Boolean {
        return isMonitoring && mediaRecorder != null &&
                monitoringJob?.isActive == true
    }

    companion object {
        private const val TAG = "AudioLevelMonitor"

        // Configuration constants
        private const val DEFAULT_NOISE_THRESHOLD = 1000
        private const val DEFAULT_SAMPLING_INTERVAL = 1000L // 1 second
        private const val MIN_TIME_BETWEEN_EVENTS = 2000L // 2 seconds
        private const val BASELINE_WINDOW_SIZE = 30
        private const val MIN_VALUES_FOR_BASELINE = 10

        // Sensitivity presets
        const val SENSITIVITY_VERY_HIGH = 300
        const val SENSITIVITY_HIGH = 600
        const val SENSITIVITY_MEDIUM = 1000
        const val SENSITIVITY_LOW = 1500
        const val SENSITIVITY_VERY_LOW = 2500

        // Amplitude ranges (device-dependent)
        const val AMPLITUDE_QUIET = 0..500
        const val AMPLITUDE_LOW = 501..1500
        const val AMPLITUDE_MEDIUM = 1501..5000
        const val AMPLITUDE_HIGH = 5001..15000
        const val AMPLITUDE_VERY_HIGH = 15001..32767
    }
}