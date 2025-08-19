package com.example.somniai.service

import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.somniai.MainActivity
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.sensor.*
import com.example.somniai.repository.SleepRepository
import com.example.somniai.repository.SessionManager
import com.example.somniai.analytics.SessionAnalytics
import com.example.somniai.analytics.SleepAnalyzer
import com.example.somniai.database.SleepDatabase
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.max

/**
 * Enhanced Sleep Tracking Service with Database Integration and Advanced Analytics
 *
 * Major Features:
 * - Complete database integration via SleepRepository
 * - Session lifecycle management with SessionManager
 * - Periodic data saving every 30 seconds with batch processing
 * - Real-time analytics updates with SessionAnalytics
 * - Memory management for long tracking sessions
 * - Advanced background data processing with coroutines
 * - Comprehensive error handling and recovery
 * - Performance monitoring and optimization
 */
class SleepTrackingService : Service() {

    companion object {
        const val ACTION_START_TRACKING = "START_TRACKING"
        const val ACTION_STOP_TRACKING = "STOP_TRACKING"
        const val ACTION_GET_STATUS = "GET_STATUS"
        const val ACTION_UPDATE_SETTINGS = "UPDATE_SETTINGS"

        // Broadcast actions
        const val BROADCAST_SENSOR_UPDATE = "com.example.somniai.SENSOR_UPDATE"
        const val BROADCAST_SESSION_COMPLETE = "com.example.somniai.SESSION_COMPLETE"
        const val BROADCAST_PHASE_CHANGE = "com.example.somniai.PHASE_CHANGE"
        const val BROADCAST_DATA_SAVED = "com.example.somniai.DATA_SAVED"
        const val BROADCAST_PERFORMANCE_UPDATE = "com.example.somniai.PERFORMANCE_UPDATE"

        // Broadcast extras
        const val EXTRA_SENSOR_STATUS = "sensor_status"
        const val EXTRA_LIVE_METRICS = "live_metrics"
        const val EXTRA_COMPLETED_SESSION = "completed_session"
        const val EXTRA_PHASE_TRANSITION = "phase_transition"
        const val EXTRA_SENSOR_SETTINGS = "sensor_settings"
        const val EXTRA_SAVE_STATUS = "save_status"

        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "sleep_tracking_channel"
        private const val TAG = "SleepTrackingService"

        // Data persistence intervals
        private const val DATA_SAVE_INTERVAL = 30000L // Save data every 30 seconds
        private const val ANALYTICS_UPDATE_INTERVAL = 60000L // Update analytics every minute
        private const val MEMORY_CLEANUP_INTERVAL = 120000L // Cleanup memory every 2 minutes

        // Memory management thresholds
        private const val MAX_EVENTS_IN_MEMORY = 1000 // Max events before forced save
        private const val MEMORY_CLEANUP_THRESHOLD = 500 // Events to keep after cleanup

        fun isServiceRunning(context: Context): Boolean {
            val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            @Suppress("DEPRECATION")
            for (service in manager.getRunningServices(Integer.MAX_VALUE)) {
                if (SleepTrackingService::class.java.name == service.service.className) {
                    return true
                }
            }
            return false
        }
    }

    // Service binding
    private val binder = LocalBinder()

    // Database and analytics dependencies
    private lateinit var sleepRepository: SleepRepository
    private lateinit var sessionManager: SessionManager
    private lateinit var sessionAnalytics: SessionAnalytics
    private lateinit var sleepAnalyzer: SleepAnalyzer

    // Tracking state
    private var isTracking = false
    private var sessionStartTime: Long = 0
    private var currentSessionId: Long? = null

    // Sensor components
    private var motionDetector: SleepMotionDetector? = null
    private var audioMonitor: AudioLevelMonitor? = null
    private var dataProcessor: SensorDataProcessor? = null

    // Enhanced data collection with thread-safe queues
    private val movementEventQueue = ConcurrentLinkedQueue<MovementEvent>()
    private val noiseEventQueue = ConcurrentLinkedQueue<NoiseEvent>()
    private val phaseTransitionQueue = ConcurrentLinkedQueue<PhaseTransition>()

    // Data persistence tracking
    private var lastDataSaveTime = 0L
    private var unsavedMovementEvents = 0
    private var unsavedNoiseEvents = 0
    private var totalEventsProcessed = 0

    // Configuration
    private var currentSettings = SensorSettings()

    // Enhanced background jobs with coroutine scope
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var statusUpdateJob: Job? = null
    private var notificationUpdateJob: Job? = null
    private var dataPersistenceJob: Job? = null
    private var analyticsUpdateJob: Job? = null
    private var memoryManagementJob: Job? = null

    // Update intervals
    private val statusUpdateInterval = 2000L // Update UI every 2 seconds
    private val notificationUpdateInterval = 10000L // Update notification every 10 seconds

    // Service health monitoring
    private var lastMotionEventTime = 0L
    private var lastNoiseEventTime = 0L
    private var sensorHealthCheckInterval = 30000L // Check every 30 seconds

    // Advanced session analytics
    private var currentSleepPhase = SleepPhase.AWAKE
    private var lastPhaseTransition = 0L
    private var sessionQualityMetrics = mutableMapOf<String, Float>()

    // Performance monitoring
    private var lastMemoryCleanup = 0L
    private var totalDataSaves = 0
    private var averageSaveTime = 0L

    inner class LocalBinder : Binder() {
        fun getService(): SleepTrackingService = this@SleepTrackingService
    }

    override fun onCreate() {
        super.onCreate()

        // Initialize database and analytics dependencies
        initializeDependencies()

        createNotificationChannel()
        Log.d(TAG, "SleepTrackingService created with database integration")
    }

    /**
     * Initialize repository, analytics, and other dependencies
     */
    private fun initializeDependencies() {
        try {
            val database = SleepDatabase.getDatabase(this)
            sleepRepository = SleepRepository(database, this)
            sessionManager = SessionManager(sleepRepository)
            sessionAnalytics = SessionAnalytics()
            sleepAnalyzer = SleepAnalyzer()

            Log.d(TAG, "All dependencies initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize dependencies", e)
            throw e // Service cannot function without these dependencies
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_TRACKING -> {
                val settings = intent.getParcelableExtra<SensorSettings>(EXTRA_SENSOR_SETTINGS)
                startTracking(settings)
            }
            ACTION_STOP_TRACKING -> stopTracking()
            ACTION_GET_STATUS -> broadcastCurrentStatus()
            ACTION_UPDATE_SETTINGS -> {
                val settings = intent.getParcelableExtra<SensorSettings>(EXTRA_SENSOR_SETTINGS)
                settings?.let { updateSensorSettings(it) }
            }
        }

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    /**
     * Start sleep tracking with database session creation
     */
    private fun startTracking(settings: SensorSettings? = null) {
        if (isTracking) {
            Log.w(TAG, "Sleep tracking already active")
            return
        }

        Log.d(TAG, "Starting sleep tracking with database integration...")

        // Update settings if provided
        settings?.let { currentSettings = it }

        // Initialize tracking state
        sessionStartTime = System.currentTimeMillis()
        lastDataSaveTime = sessionStartTime
        lastMemoryCleanup = sessionStartTime

        // Clear previous session data
        clearEventQueues()
        resetSessionMetrics()

        // Create database session
        serviceScope.launch {
            try {
                val sessionResult = sleepRepository.createSession(sessionStartTime, currentSettings)
                sessionResult.onSuccess { sessionId ->
                    currentSessionId = sessionId
                    isTracking = true

                    // Start foreground service with notification
                    withContext(Dispatchers.Main) {
                        val notification = createNotification()
                        startForeground(NOTIFICATION_ID, notification)
                    }

                    // Initialize and start sensors
                    if (initializeSensors()) {
                        startSensors()
                        startAllBackgroundJobs()

                        Log.d(TAG, "Sleep tracking started successfully with session ID: $sessionId")
                        broadcastTrackingStarted()
                    } else {
                        Log.e(TAG, "Failed to initialize sensors")
                        stopTracking()
                    }
                }.onFailure { exception ->
                    Log.e(TAG, "Failed to create database session", exception)
                    broadcastError("Failed to start tracking: ${exception.message}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error starting tracking session", e)
                broadcastError("Failed to start tracking: ${e.message}")
            }
        }
    }

    /**
     * Stop sleep tracking and complete database session with analytics
     */
    private fun stopTracking() {
        if (!isTracking) {
            Log.w(TAG, "Sleep tracking not active")
            return
        }

        Log.d(TAG, "Stopping sleep tracking and finalizing session...")

        val sessionEndTime = System.currentTimeMillis()
        val finalSessionId = currentSessionId

        // Stop all background jobs first
        stopAllBackgroundJobs()

        // Stop sensors
        stopSensors()

        // Perform final data save and session completion
        serviceScope.launch {
            try {
                // Save any remaining data
                saveDataToDatabase(force = true)

                // Generate comprehensive session analytics
                val sessionAnalytics = generateFinalSessionAnalytics(sessionEndTime)

                // Complete the database session
                if (finalSessionId != null) {
                    val completionResult = sleepRepository.completeSession(sessionEndTime, sessionAnalytics)
                    completionResult.onSuccess { completedSession ->
                        Log.d(TAG, "Session completed successfully: ID $finalSessionId, " +
                                "Duration: ${completedSession.duration}ms, " +
                                "Quality: ${completedSession.sleepQualityScore}")

                        // Broadcast session completion
                        withContext(Dispatchers.Main) {
                            broadcastSessionComplete(sessionAnalytics, completedSession)
                        }

                        // Generate insights and recommendations
                        generatePostSessionInsights(completedSession)

                    }.onFailure { exception ->
                        Log.e(TAG, "Failed to complete session", exception)
                        broadcastError("Failed to save session: ${exception.message}")
                    }
                } else {
                    Log.w(TAG, "No session ID available for completion")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error during session completion", e)
                broadcastError("Error completing session: ${e.message}")
            } finally {
                // Reset service state
                withContext(Dispatchers.Main) {
                    isTracking = false
                    currentSessionId = null

                    // Stop foreground service
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            }
        }
    }

    /**
     * Initialize all sensor components
     */
    private fun initializeSensors(): Boolean {
        try {
            // Initialize data processor
            dataProcessor = SensorDataProcessor().apply {
                startSession(sessionStartTime)
            }

            // Initialize motion detector
            motionDetector = SleepMotionDetector(this) { motionEvent ->
                handleMotionEvent(motionEvent)
            }.apply {
                setMovementThreshold(currentSettings.movementThreshold)
            }

            // Initialize audio monitor
            audioMonitor = AudioLevelMonitor { noiseEvent ->
                handleNoiseEvent(noiseEvent)
            }.apply {
                setNoiseThreshold(currentSettings.noiseThreshold)
                setSamplingInterval(currentSettings.noiseSamplingInterval)
            }

            Log.d(TAG, "All sensors initialized successfully")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize sensors", e)
            return false
        }
    }

    /**
     * Start all sensor monitoring
     */
    private fun startSensors() {
        var sensorsStarted = 0
        var totalSensors = 0

        // Start motion detection if enabled
        if (currentSettings.enableMovementDetection) {
            totalSensors++
            if (motionDetector?.startMonitoring() == true) {
                sensorsStarted++
                Log.d(TAG, "Motion detector started")
            } else {
                Log.e(TAG, "Failed to start motion detector")
            }
        }

        // Start audio monitoring if enabled
        if (currentSettings.enableNoiseDetection) {
            totalSensors++
            if (audioMonitor?.startMonitoring() == true) {
                sensorsStarted++
                Log.d(TAG, "Audio monitor started")
            } else {
                Log.e(TAG, "Failed to start audio monitor")
            }
        }

        Log.d(TAG, "Sensors started: $sensorsStarted/$totalSensors")

        if (sensorsStarted == 0) {
            Log.e(TAG, "No sensors could be started")
        }
    }

    /**
     * Stop all sensor monitoring
     */
    private fun stopSensors() {
        motionDetector?.stopMonitoring()
        audioMonitor?.stopMonitoring()
        Log.d(TAG, "All sensors stopped")
    }

    /**
     * Handle detected movement event with queue-based processing
     */
    private fun handleMotionEvent(motionEvent: com.example.somniai.sensor.MovementEvent) {
        // Convert to data model
        val dataModelEvent = MovementEvent(
            timestamp = motionEvent.timestamp,
            intensity = motionEvent.intensity,
            x = motionEvent.x,
            y = motionEvent.y,
            z = motionEvent.z
        )

        // Add to queue for batch processing
        movementEventQueue.offer(dataModelEvent)
        unsavedMovementEvents++
        totalEventsProcessed++
        lastMotionEventTime = motionEvent.timestamp

        // Process for real-time analysis
        dataProcessor?.processMovementEvent(dataModelEvent)

        // Check for memory management
        checkMemoryThresholds()

        Log.d(TAG, "Movement event queued: intensity=${motionEvent.intensity}, queue size=${movementEventQueue.size}")
    }

    /**
     * Handle detected noise event with queue-based processing
     */
    private fun handleNoiseEvent(noiseEvent: com.example.somniai.sensor.NoiseEvent) {
        // Convert to data model
        val dataModelEvent = NoiseEvent(
            timestamp = noiseEvent.timestamp,
            decibelLevel = noiseEvent.decibelLevel,
            amplitude = noiseEvent.amplitude
        )

        // Add to queue for batch processing
        noiseEventQueue.offer(dataModelEvent)
        unsavedNoiseEvents++
        totalEventsProcessed++
        lastNoiseEventTime = noiseEvent.timestamp

        // Process for real-time analysis
        dataProcessor?.processNoiseEvent(dataModelEvent)

        // Check for memory management
        checkMemoryThresholds()

        Log.d(TAG, "Noise event queued: ${noiseEvent.decibelLevel}dB, queue size=${noiseEventQueue.size}")
    }

    /**
     * Handle sleep phase transitions with database persistence
     */
    private fun handlePhaseTransition(fromPhase: SleepPhase, toPhase: SleepPhase, confidence: Float) {
        val transition = PhaseTransition(
            timestamp = System.currentTimeMillis(),
            fromPhase = fromPhase,
            toPhase = toPhase,
            confidence = confidence
        )

        // Add to queue for batch processing
        phaseTransitionQueue.offer(transition)
        currentSleepPhase = toPhase
        lastPhaseTransition = transition.timestamp

        // Broadcast phase change immediately for UI updates
        broadcastPhaseChange(transition)

        Log.d(TAG, "Phase transition: ${fromPhase.name} -> ${toPhase.name} (${(confidence * 100).toInt()}%)")
    }

    /**
     * Start all background monitoring and data processing jobs
     */
    private fun startAllBackgroundJobs() {
        // Start status update broadcasts
        statusUpdateJob = serviceScope.launch {
            while (isActive && isTracking) {
                try {
                    broadcastCurrentStatus()
                    broadcastLiveMetrics()
                    checkSensorHealth()
                    delay(statusUpdateInterval)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in status update job", e)
                }
            }
        }

        // Start notification updates
        notificationUpdateJob = serviceScope.launch {
            while (isActive && isTracking) {
                try {
                    updateNotification()
                    delay(notificationUpdateInterval)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in notification update job", e)
                }
            }
        }

        // Start periodic data persistence
        dataPersistenceJob = serviceScope.launch {
            while (isActive && isTracking) {
                try {
                    saveDataToDatabase()
                    delay(DATA_SAVE_INTERVAL)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in data persistence job", e)
                }
            }
        }

        // Start analytics updates
        analyticsUpdateJob = serviceScope.launch {
            while (isActive && isTracking) {
                try {
                    updateSessionAnalytics()
                    delay(ANALYTICS_UPDATE_INTERVAL)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in analytics update job", e)
                }
            }
        }

        // Start memory management
        memoryManagementJob = serviceScope.launch {
            while (isActive && isTracking) {
                try {
                    performMemoryCleanup()
                    delay(MEMORY_CLEANUP_INTERVAL)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in memory management job", e)
                }
            }
        }

        Log.d(TAG, "All background jobs started")
    }

    /**
     * Stop all background jobs
     */
    private fun stopAllBackgroundJobs() {
        statusUpdateJob?.cancel()
        notificationUpdateJob?.cancel()
        dataPersistenceJob?.cancel()
        analyticsUpdateJob?.cancel()
        memoryManagementJob?.cancel()
        Log.d(TAG, "All background jobs stopped")
    }

    // ========== DATA PERSISTENCE METHODS ==========

    /**
     * Save queued data to database in batches
     */
    private suspend fun saveDataToDatabase(force: Boolean = false) {
        val currentTime = System.currentTimeMillis()

        // Check if save is needed
        if (!force && currentTime - lastDataSaveTime < DATA_SAVE_INTERVAL &&
            totalEventsInQueues() < MAX_EVENTS_IN_MEMORY) {
            return
        }

        val saveStartTime = System.currentTimeMillis()

        try {
            // Collect events from queues for batch processing
            val movementEvents = mutableListOf<MovementEvent>()
            val noiseEvents = mutableListOf<NoiseEvent>()
            val phaseTransitions = mutableListOf<PhaseTransition>()

            // Drain queues efficiently
            drainQueue(movementEventQueue, movementEvents, 100) // Batch size of 100
            drainQueue(noiseEventQueue, noiseEvents, 100)
            drainQueue(phaseTransitionQueue, phaseTransitions, 50)

            // Save to repository in batch
            if (movementEvents.isNotEmpty() || noiseEvents.isNotEmpty() || phaseTransitions.isNotEmpty()) {
                val result = sleepRepository.recordEventsBatch(
                    movements = movementEvents,
                    noises = noiseEvents,
                    phases = phaseTransitions
                )

                result.onSuccess {
                    lastDataSaveTime = currentTime
                    totalDataSaves++
                    unsavedMovementEvents = max(0, unsavedMovementEvents - movementEvents.size)
                    unsavedNoiseEvents = max(0, unsavedNoiseEvents - noiseEvents.size)

                    val saveTime = System.currentTimeMillis() - saveStartTime
                    updateAverageSaveTime(saveTime)

                    Log.d(TAG, "Data saved successfully: ${movementEvents.size} movements, " +
                            "${noiseEvents.size} noises, ${phaseTransitions.size} phases " +
                            "in ${saveTime}ms")

                    // Update session with current metrics
                    updateCurrentSession()

                    // Broadcast save status
                    broadcastDataSaved(movementEvents.size + noiseEvents.size + phaseTransitions.size)

                }.onFailure { exception ->
                    Log.e(TAG, "Failed to save data to database", exception)
                    // Re-add events to queues for retry (with size limit)
                    reQueueEventsWithLimit(movementEvents, noiseEvents, phaseTransitions)
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during data save operation", e)
        }
    }

    /**
     * Update current session with aggregated metrics
     */
    private suspend fun updateCurrentSession() {
        try {
            val currentTime = System.currentTimeMillis()
            val duration = currentTime - sessionStartTime
            val liveMetrics = dataProcessor?.getLiveMetrics()

            sleepRepository.updateCurrentSession(
                duration = duration,
                efficiency = liveMetrics?.sleepEfficiency,
                movementIntensity = liveMetrics?.averageMovementIntensity,
                noiseLevel = liveMetrics?.averageNoiseLevel,
                movementEventCount = totalEventsProcessed,
                noiseEventCount = noiseEventQueue.size + unsavedNoiseEvents,
                phaseTransitionCount = phaseTransitionQueue.size
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error updating current session", e)
        }
    }

    // ========== MEMORY MANAGEMENT METHODS ==========

    /**
     * Check memory thresholds and trigger save if needed
     */
    private fun checkMemoryThresholds() {
        val totalEvents = totalEventsInQueues()

        if (totalEvents >= MAX_EVENTS_IN_MEMORY) {
            Log.w(TAG, "Memory threshold reached ($totalEvents events), forcing data save")
            serviceScope.launch {
                saveDataToDatabase(force = true)
            }
        }
    }

    /**
     * Perform periodic memory cleanup
     */
    private suspend fun performMemoryCleanup() {
        val currentTime = System.currentTimeMillis()

        if (currentTime - lastMemoryCleanup < MEMORY_CLEANUP_INTERVAL) {
            return
        }

        try {
            // Force save current data
            saveDataToDatabase(force = true)

            // Clear processed data from sensor components
            dataProcessor?.clearOldData(currentTime - (5 * 60 * 1000L)) // Keep last 5 minutes

            lastMemoryCleanup = currentTime

            // Log memory status
            val totalEvents = totalEventsInQueues()
            Log.d(TAG, "Memory cleanup completed. Events in queues: $totalEvents")

        } catch (e: Exception) {
            Log.e(TAG, "Error during memory cleanup", e)
        }
    }

    /**
     * Clear all event queues
     */
    private fun clearEventQueues() {
        movementEventQueue.clear()
        noiseEventQueue.clear()
        phaseTransitionQueue.clear()
        unsavedMovementEvents = 0
        unsavedNoiseEvents = 0
        totalEventsProcessed = 0
    }

    /**
     * Reset session metrics
     */
    private fun resetSessionMetrics() {
        currentSleepPhase = SleepPhase.AWAKE
        lastPhaseTransition = 0L
        sessionQualityMetrics.clear()
        totalDataSaves = 0
        averageSaveTime = 0L
    }

    // ========== ANALYTICS UPDATE METHODS ==========

    /**
     * Update session analytics periodically
     */
    private suspend fun updateSessionAnalytics() {
        try {
            val currentTime = System.currentTimeMillis()
            val sessionDuration = currentTime - sessionStartTime

            // Get recent data for analysis
            val recentMovements = getRecentMovements(10 * 60 * 1000L) // Last 10 minutes
            val recentNoises = getRecentNoises(10 * 60 * 1000L)
            val phaseHistory = getPhaseHistory()

            // Perform real-time session analysis
            val realTimeAnalysis = sessionAnalytics.analyzeRealTimeSession(
                sessionStart = sessionStartTime,
                currentTime = currentTime,
                recentMovements = recentMovements,
                recentNoises = recentNoises,
                currentPhase = currentSleepPhase,
                phaseHistory = phaseHistory
            )

            // Update quality metrics
            updateQualityMetrics(realTimeAnalysis)

            // Check for phase transitions
            checkForPhaseTransitions(realTimeAnalysis, recentMovements, recentNoises)

            Log.d(TAG, "Session analytics updated: Phase=${currentSleepPhase.name}, " +
                    "Quality=${realTimeAnalysis.qualityIndicators}")

        } catch (e: Exception) {
            Log.e(TAG, "Error updating session analytics", e)
        }
    }

    /**
     * Generate final comprehensive session analytics
     */
    private suspend fun generateFinalSessionAnalytics(endTime: Long): SleepSessionAnalytics? {
        return try {
            // Get all session data
            val allMovements = getAllMovements()
            val allNoises = getAllNoises()
            val allPhaseTransitions = getAllPhaseTransitions()

            // Create complete session for analysis
            val completeSession = SleepSession(
                id = currentSessionId ?: 0L,
                startTime = sessionStartTime,
                endTime = endTime,
                totalDuration = endTime - sessionStartTime,
                movementEvents = allMovements,
                noiseEvents = allNoises,
                phaseTransitions = allPhaseTransitions
            )

            // Generate comprehensive analytics
            val qualityAnalysis = sessionAnalytics.analyzeSessionQuality(completeSession)
            val onsetAnalysis = sessionAnalytics.detectSleepOnset(sessionStartTime, allMovements, allNoises)
            val wakeAnalysis = sessionAnalytics.detectWakeEvents(allMovements, allNoises, endTime)
            val phaseAnalysis = sessionAnalytics.analyzePhaseTransitions(allPhaseTransitions, completeSession.totalDuration)

            // Create final analytics object
            SleepSessionAnalytics(
                sessionId = currentSessionId ?: 0L,
                totalDuration = completeSession.totalDuration,
                sleepEfficiency = qualityAnalysis.basicMetrics.efficiency,
                sleepLatency = onsetAnalysis.sleepLatency ?: 0L,
                qualityFactors = SleepQualityFactors(
                    movementScore = qualityAnalysis.movementQuality.restlessness,
                    noiseScore = qualityAnalysis.noiseQuality.quietness,
                    durationScore = qualityAnalysis.basicMetrics.duration,
                    consistencyScore = qualityAnalysis.timingQuality.consistency,
                    overallScore = qualityAnalysis.overallScore
                ),
                averageMovementIntensity = allMovements.map { it.intensity }.average().toFloat(),
                averageNoiseLevel = allNoises.map { it.decibelLevel }.average().toFloat(),
                movementFrequency = allMovements.size.toFloat() / (completeSession.totalDuration / (60 * 60 * 1000f))
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error generating final session analytics", e)
            null
        }
    }

    // ========== UTILITY METHODS ==========

    /**
     * Efficiently drain a queue into a list with size limit
     */
    private fun <T> drainQueue(queue: ConcurrentLinkedQueue<T>, destination: MutableList<T>, maxSize: Int) {
        var count = 0
        while (count < maxSize && !queue.isEmpty()) {
            queue.poll()?.let {
                destination.add(it)
                count++
            }
        }
    }

    /**
     * Get total events in all queues
     */
    private fun totalEventsInQueues(): Int {
        return movementEventQueue.size + noiseEventQueue.size + phaseTransitionQueue.size
    }

    /**
     * Re-queue events with size limit for retry
     */
    private fun reQueueEventsWithLimit(
        movements: List<MovementEvent>,
        noises: List<NoiseEvent>,
        phases: List<PhaseTransition>
    ) {
        val totalCurrentEvents = totalEventsInQueues()
        val availableSpace = MAX_EVENTS_IN_MEMORY - totalCurrentEvents

        if (availableSpace > 0) {
            // Add back only what fits
            movements.take(availableSpace / 3).forEach { movementEventQueue.offer(it) }
            noises.take(availableSpace / 3).forEach { noiseEventQueue.offer(it) }
            phases.take(availableSpace / 3).forEach { phaseTransitionQueue.offer(it) }
        }
    }

    /**
     * Update average save time metric
     */
    private fun updateAverageSaveTime(saveTime: Long) {
        averageSaveTime = if (totalDataSaves == 1) {
            saveTime
        } else {
            ((averageSaveTime * (totalDataSaves - 1)) + saveTime) / totalDataSaves
        }
    }

    /**
     * Get recent movement events for analysis
     */
    private fun getRecentMovements(timeWindow: Long): List<MovementEvent> {
        val cutoffTime = System.currentTimeMillis() - timeWindow
        return movementEventQueue.filter { it.timestamp >= cutoffTime }
    }

    /**
     * Get recent noise events for analysis
     */
    private fun getRecentNoises(timeWindow: Long): List<NoiseEvent> {
        val cutoffTime = System.currentTimeMillis() - timeWindow
        return noiseEventQueue.filter { it.timestamp >= cutoffTime }
    }

    /**
     * Get all movement events (for final analysis)
     */
    private fun getAllMovements(): List<MovementEvent> {
        return movementEventQueue.toList()
    }

    /**
     * Get all noise events (for final analysis)
     */
    private fun getAllNoises(): List<NoiseEvent> {
        return noiseEventQueue.toList()
    }

    /**
     * Get all phase transitions (for final analysis)
     */
    private fun getAllPhaseTransitions(): List<PhaseTransition> {
        return phaseTransitionQueue.toList()
    }

    /**
     * Get phase history for analysis
     */
    private fun getPhaseHistory(): List<PhaseTransition> {
        return phaseTransitionQueue.toList()
    }

    /**
     * Update quality metrics from real-time analysis
     */
    private fun updateQualityMetrics(analysis: SessionAnalytics.RealTimeSessionAnalysis) {
        sessionQualityMetrics["movement"] = analysis.movementAnalysis.averageIntensity
        sessionQualityMetrics["noise"] = analysis.noiseAnalysis.averageDecibel
        sessionQualityMetrics["efficiency"] = analysis.currentEfficiency
        sessionQualityMetrics["phase_stability"] = analysis.phaseAnalysis.stability
    }

    /**
     * Check for phase transitions based on analysis
     */
    private fun checkForPhaseTransitions(
        analysis: SessionAnalytics.RealTimeSessionAnalysis,
        recentMovements: List<MovementEvent>,
        recentNoises: List<NoiseEvent>
    ) {
        // Use existing data processor logic or implement phase detection
        dataProcessor?.let { processor ->
            val prediction = serviceScope.async {
                sessionAnalytics.predictNextPhaseTransition(
                    currentPhase = currentSleepPhase,
                    timeInCurrentPhase = System.currentTimeMillis() - lastPhaseTransition,
                    recentMovements = recentMovements,
                    phaseHistory = getPhaseHistory()
                )
            }

            // If high confidence prediction, trigger phase change
            serviceScope.launch {
                val phaseTransitionPrediction = prediction.await()
                if (phaseTransitionPrediction.confidence > 0.8f) {
                    handlePhaseTransition(
                        fromPhase = currentSleepPhase,
                        toPhase = phaseTransitionPrediction.predictedNextPhase,
                        confidence = phaseTransitionPrediction.confidence
                    )
                }
            }
        }
    }

    /**
     * Generate post-session insights and recommendations
     */
    private suspend fun generatePostSessionInsights(completedSession: SleepSession) {
        try {
            // Analyze session for insights
            val qualityAnalysis = sessionAnalytics.analyzeSessionQuality(completedSession)

            // Generate AI insights based on the session
            val insights = generateInsightsFromAnalysis(qualityAnalysis, completedSession)

            // Save insights to database
            insights.forEach { insight ->
                sleepRepository.addInsight(insight)
            }

            Log.d(TAG, "Generated ${insights.size} insights for session")

        } catch (e: Exception) {
            Log.e(TAG, "Error generating post-session insights", e)
        }
    }

    /**
     * Generate insights from session analysis
     */
    private fun generateInsightsFromAnalysis(
        qualityAnalysis: SessionAnalytics.SessionQualityAnalysis,
        session: SleepSession
    ): List<SleepInsight> {
        val insights = mutableListOf<SleepInsight>()
        val sessionId = session.id

        // Duration insights
        if (session.totalDuration < 6 * 60 * 60 * 1000L) { // Less than 6 hours
            insights.add(SleepInsight(
                sessionId = sessionId,
                category = InsightCategory.DURATION,
                title = "Short Sleep Duration",
                description = "This session was shorter than recommended",
                recommendation = "Try to get 7-9 hours of sleep for optimal recovery",
                priority = 2,
                timestamp = System.currentTimeMillis(),
                isAiGenerated = true
            ))
        }

        // Movement insights
        if (qualityAnalysis.movementQuality.restlessness > 7f) {
            insights.add(SleepInsight(
                sessionId = sessionId,
                category = InsightCategory.MOVEMENT,
                title = "Restful Sleep Achieved",
                description = "You had minimal movement during sleep",
                recommendation = "Continue current sleep practices",
                priority = 3,
                timestamp = System.currentTimeMillis(),
                isAiGenerated = true
            ))
        }

        // Add more insight generation logic based on quality factors...

        return insights
    }

    // ========== SENSOR HEALTH AND STATUS METHODS ==========

    /**
     * Check sensor health and restart if needed
     */
    private fun checkSensorHealth() {
        val currentTime = System.currentTimeMillis()

        // Check if sensors are responding
        val motionHealthy = !currentSettings.enableMovementDetection ||
                (motionDetector?.isMonitoring == true)

        val audioHealthy = !currentSettings.enableNoiseDetection ||
                (audioMonitor?.isHealthy() == true)

        if (!motionHealthy || !audioHealthy) {
            Log.w(TAG, "Sensor health check failed - Motion: $motionHealthy, Audio: $audioHealthy")

            // Attempt restart of unhealthy sensors
            if (!motionHealthy && currentSettings.enableMovementDetection) {
                Log.d(TAG, "Attempting to restart motion detector")
                motionDetector?.stopMonitoring()
                motionDetector?.startMonitoring()
            }

            if (!audioHealthy && currentSettings.enableNoiseDetection) {
                Log.d(TAG, "Attempting to restart audio monitor")
                audioMonitor?.stopMonitoring()
                audioMonitor?.startMonitoring()
            }
        }
    }

    /**
     * Update sensor settings during active tracking
     */
    private fun updateSensorSettings(newSettings: SensorSettings) {
        currentSettings = newSettings

        // Apply new settings to sensors
        motionDetector?.setMovementThreshold(newSettings.movementThreshold)
        audioMonitor?.apply {
            setNoiseThreshold(newSettings.noiseThreshold)
            setSamplingInterval(newSettings.noiseSamplingInterval)
        }

        Log.d(TAG, "Sensor settings updated")
        broadcastCurrentStatus()
    }

    // ========== BROADCAST METHODS ==========

    /**
     * Broadcast current sensor status
     */
    private fun broadcastCurrentStatus() {
        val status = getCurrentSensorStatus()
        val intent = Intent(BROADCAST_SENSOR_UPDATE).apply {
            putExtra(EXTRA_SENSOR_STATUS, status)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    /**
     * Broadcast live sleep metrics
     */
    private fun broadcastLiveMetrics() {
        val liveMetrics = dataProcessor?.getLiveMetrics()
        if (liveMetrics != null) {
            val intent = Intent(BROADCAST_SENSOR_UPDATE).apply {
                putExtra(EXTRA_LIVE_METRICS, liveMetrics)
            }
            LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
        }
    }

    /**
     * Broadcast tracking started event
     */
    private fun broadcastTrackingStarted() {
        val intent = Intent(BROADCAST_SENSOR_UPDATE).apply {
            putExtra("tracking_started", true)
            putExtra(EXTRA_SENSOR_SETTINGS, currentSettings)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    /**
     * Broadcast phase change for immediate UI updates
     */
    private fun broadcastPhaseChange(transition: PhaseTransition) {
        val intent = Intent(BROADCAST_PHASE_CHANGE).apply {
            putExtra(EXTRA_PHASE_TRANSITION, transition)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    /**
     * Broadcast data save status
     */
    private fun broadcastDataSaved(eventCount: Int) {
        val intent = Intent(BROADCAST_DATA_SAVED).apply {
            putExtra(EXTRA_SAVE_STATUS, "Saved $eventCount events")
            putExtra("total_saves", totalDataSaves)
            putExtra("average_save_time", averageSaveTime)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    /**
     * Enhanced broadcast session completion with detailed analytics
     */
    private fun broadcastSessionComplete(
        sessionAnalytics: SleepSessionAnalytics?,
        completedSession: SleepSession
    ) {
        val intent = Intent(BROADCAST_SESSION_COMPLETE).apply {
            putExtra(EXTRA_COMPLETED_SESSION, sessionAnalytics)
            putExtra("completed_sleep_session", completedSession)
            putExtra("session_summary", generateSessionSummary(completedSession, sessionAnalytics))
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    /**
     * Broadcast error messages
     */
    private fun broadcastError(message: String) {
        val intent = Intent(BROADCAST_SENSOR_UPDATE).apply {
            putExtra("error", true)
            putExtra("error_message", message)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    /**
     * Get current sensor status with enhanced metrics
     */
    private fun getCurrentSensorStatus(): SensorStatus {
        val liveMetrics = dataProcessor?.getLiveMetrics()

        return SensorStatus(
            isAccelerometerActive = motionDetector?.isMonitoring ?: false,
            isMicrophoneActive = audioMonitor?.isMonitoring ?: false,
            currentMovementIntensity = motionDetector?.getCurrentMovementIntensity() ?: 0f,
            currentNoiseLevel = audioMonitor?.getCurrentAmplitude()?.toFloat() ?: 0f,
            movementThreshold = currentSettings.movementThreshold,
            noiseThreshold = currentSettings.noiseThreshold,
            accelerometerStatus = motionDetector?.getMonitoringStatus() ?: "Not initialized",
            microphoneStatus = audioMonitor?.getMonitoringStatus() ?: "Not initialized",
            totalMovementEvents = movementEventQueue.size + unsavedMovementEvents,
            totalNoiseEvents = noiseEventQueue.size + unsavedNoiseEvents,
            sessionStartTime = sessionStartTime,
            currentPhase = liveMetrics?.currentPhase ?: currentSleepPhase,
            phaseConfidence = liveMetrics?.phaseConfidence ?: 0f
        )
    }

    /**
     * Calculate memory usage level as percentage
     */
    private fun calculateMemoryUsageLevel(): Float {
        val totalEvents = totalEventsInQueues()
        return (totalEvents.toFloat() / MAX_EVENTS_IN_MEMORY) * 100f
    }

    /**
     * Generate session summary for UI display
     */
    private suspend fun generateSessionSummary(
        session: SleepSession,
        analytics: SleepSessionAnalytics?
    ): SessionAnalytics.SessionSummary {
        return try {
            sessionAnalytics.generateSessionSummary(
                session = session,
                onsetAnalysis = null, // Could be enhanced to include onset analysis
                wakeAnalysis = null,
                phaseAnalysis = null,
                qualityAnalysis = null
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error generating session summary", e)
            SessionAnalytics.SessionSummary(sessionId = session.id)
        }
    }

    // ========== NOTIFICATION METHODS ==========

    /**
     * Create notification channel for Android 8.0+
     */
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_description)
                setShowBadge(false)
                enableVibration(false)
                setSound(null, null)
            }

            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.createNotificationChannel(channel)
        }
    }

    /**
     * Create foreground service notification
     */
    private fun createNotification(): Notification {
        val notificationIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, notificationIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.sleep_tracking_notification_title))
            .setContentText(getNotificationText())
            .setSmallIcon(R.drawable.ic_sleep)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    /**
     * Update notification with current status
     */
    private fun updateNotification() {
        if (isTracking) {
            val notification = createNotification()
            val notificationManager = getSystemService(NotificationManager::class.java)
            notificationManager.notify(NOTIFICATION_ID, notification)
        }
    }

    /**
     * Get dynamic notification text with enhanced information
     */
    private fun getNotificationText(): String {
        if (!isTracking) {
            return getString(R.string.sleep_tracking_notification_text)
        }

        val duration = System.currentTimeMillis() - sessionStartTime
        val hours = duration / (1000 * 60 * 60)
        val minutes = (duration % (1000 * 60 * 60)) / (1000 * 60)

        val liveMetrics = dataProcessor?.getLiveMetrics()
        val phase = liveMetrics?.currentPhase?.getDisplayName() ?: currentSleepPhase.getDisplayName()

        val totalMovements = movementEventQueue.size + unsavedMovementEvents
        val totalNoises = noiseEventQueue.size + unsavedNoiseEvents

        return String.format(
            "Tracking: %02d:%02d | %s | %d movements, %d sounds | Quality: %.1f",
            hours, minutes, phase, totalMovements, totalNoises,
            sessionQualityMetrics["efficiency"] ?: 0f
        )
    }

    // ========== ENHANCED PUBLIC API METHODS ==========

    /**
     * Get enhanced current status for UI with analytics
     */
    fun getEnhancedCurrentStatus(): EnhancedSensorStatus {
        val basicStatus = getCurrentSensorStatus()
        val liveMetrics = dataProcessor?.getLiveMetrics()

        return EnhancedSensorStatus(
            basicStatus = basicStatus,
            liveMetrics = liveMetrics,
            qualityMetrics = sessionQualityMetrics.toMap(),
            sessionAnalytics = getSessionAnalyticsSummary(),
            performanceMetrics = getPerformanceMetrics()
        )
    }

    /**
     * Get session analytics summary for UI
     */
    private fun getSessionAnalyticsSummary(): Map<String, Any> {
        return mapOf(
            "total_events_processed" to totalEventsProcessed,
            "current_phase" to currentSleepPhase.name,
            "phase_duration" to (System.currentTimeMillis() - lastPhaseTransition),
            "session_efficiency" to (sessionQualityMetrics["efficiency"] ?: 0f),
            "data_save_frequency" to if (totalDataSaves > 0)
                (System.currentTimeMillis() - sessionStartTime) / totalDataSaves else 0L
        )
    }

    /**
     * Get performance metrics for monitoring
     */
    private fun getPerformanceMetrics(): Map<String, Any> {
        return mapOf(
            "memory_usage_percent" to calculateMemoryUsageLevel(),
            "queue_sizes" to mapOf(
                "movements" to movementEventQueue.size,
                "noises" to noiseEventQueue.size,
                "phases" to phaseTransitionQueue.size
            ),
            "save_performance" to mapOf(
                "total_saves" to totalDataSaves,
                "average_save_time_ms" to averageSaveTime,
                "last_save_time" to lastDataSaveTime
            ),
            "sensor_health" to getSensorHealth()
        )
    }

    /**
     * Force immediate data save (for UI requests)
     */
    fun forceSaveData(): Boolean {
        return try {
            serviceScope.launch {
                saveDataToDatabase(force = true)
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error forcing data save", e)
            false
        }
    }

    /**
     * Get comprehensive session statistics
     */
    fun getSessionStatistics(): SessionStatistics {
        val duration = if (isTracking) System.currentTimeMillis() - sessionStartTime else 0L

        return SessionStatistics(
            sessionId = currentSessionId,
            duration = duration,
            currentPhase = currentSleepPhase,
            totalMovements = movementEventQueue.size + unsavedMovementEvents,
            totalNoises = noiseEventQueue.size + unsavedNoiseEvents,
            totalPhaseTransitions = phaseTransitionQueue.size,
            averageMovementIntensity = sessionQualityMetrics["movement"] ?: 0f,
            averageNoiseLevel = sessionQualityMetrics["noise"] ?: 0f,
            currentEfficiency = sessionQualityMetrics["efficiency"] ?: 0f,
            dataIntegrity = DataIntegrityMetrics(
                totalEventsSaved = totalDataSaves,
                pendingEvents = totalEventsInQueues(),
                lastSaveTime = lastDataSaveTime,
                saveSuccessRate = 1.0f // Could be enhanced with failure tracking
            )
        )
    }

    /**
     * Get enhanced sensor health information
     */
    fun getSensorHealth(): Map<String, Any> {
        return mapOf(
            "motion_detector_active" to (motionDetector?.isMonitoring ?: false),
            "audio_monitor_active" to (audioMonitor?.isMonitoring ?: false),
            "motion_detector_healthy" to (motionDetector?.isMonitoring ?: false),
            "audio_monitor_healthy" to (audioMonitor?.isHealthy() ?: false),
            "last_motion_event" to lastMotionEventTime,
            "last_noise_event" to lastNoiseEventTime,
            "total_movement_events" to (movementEventQueue.size + unsavedMovementEvents),
            "total_noise_events" to (noiseEventQueue.size + unsavedNoiseEvents),
            "queue_health" to mapOf(
                "movement_queue_size" to movementEventQueue.size,
                "noise_queue_size" to noiseEventQueue.size,
                "phase_queue_size" to phaseTransitionQueue.size,
                "memory_usage_percent" to calculateMemoryUsageLevel()
            ),
            "data_persistence_health" to mapOf(
                "total_saves" to totalDataSaves,
                "average_save_time" to averageSaveTime,
                "last_save_time" to lastDataSaveTime,
                "unsaved_events" to totalEventsInQueues()
            )
        )
    }

    // Public API methods for UI interaction

    /**
     * Get current status for UI
     */
    fun getCurrentStatus(): SensorStatus = getCurrentSensorStatus()

    /**
     * Get current live metrics
     */
    fun getLiveMetrics(): LiveSleepMetrics? = dataProcessor?.getLiveMetrics()

    /**
     * Get current sensor settings
     */
    fun getCurrentSettings(): SensorSettings = currentSettings

    /**
     * Check if tracking is active
     */
    fun isTracking(): Boolean = isTracking

    /**
     * Get session duration
     */
    fun getSessionDuration(): Long =
        if (isTracking) System.currentTimeMillis() - sessionStartTime else 0L

    /**
     * Adjust sensor sensitivity during tracking
     */
    fun adjustSensitivity(movementThreshold: Float, noiseThreshold: Int) {
        val newSettings = currentSettings.copy(
            movementThreshold = movementThreshold,
            noiseThreshold = noiseThreshold
        )
        updateSensorSettings(newSettings)
    }

    override fun onDestroy() {
        Log.d(TAG, "Service destroying...")

        // Cancel all coroutines
        serviceScope.cancel()

        if (isTracking) {
            // Force save any remaining data before destruction
            runBlocking {
                saveDataToDatabase(force = true)
            }
            stopTracking()
        }

        // Cleanup resources
        try {
            sleepRepository.cleanup()
        } catch (e: Exception) {
            Log.e(TAG, "Error during repository cleanup", e)
        }

        super.onDestroy()
    }
}

// ========== SUPPORTING DATA CLASSES ==========

/**
 * Enhanced sensor status with analytics
 */
data class EnhancedSensorStatus(
    val basicStatus: SensorStatus,
    val liveMetrics: LiveSleepMetrics?,
    val qualityMetrics: Map<String, Float>,
    val sessionAnalytics: Map<String, Any>,
    val performanceMetrics: Map<String, Any>
)

/**
 * Comprehensive session statistics
 */
data class SessionStatistics(
    val sessionId: Long?,
    val duration: Long,
    val currentPhase: SleepPhase,
    val totalMovements: Int,
    val totalNoises: Int,
    val totalPhaseTransitions: Int,
    val averageMovementIntensity: Float,
    val averageNoiseLevel: Float,
    val currentEfficiency: Float,
    val dataIntegrity: DataIntegrityMetrics
)

/**
 * Data integrity tracking
 */
data class DataIntegrityMetrics(
    val totalEventsSaved: Int,
    val pendingEvents: Int,
    val lastSaveTime: Long,
    val saveSuccessRate: Float
)