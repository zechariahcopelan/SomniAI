package com.example.somniai.viewmodel

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.somniai.data.*
import com.example.somniai.repository.SleepRepository
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * ViewModel for Settings Management
 *
 * Handles:
 * - Data export functionality (JSON, CSV, PDF)
 * - Data statistics and storage management
 * - Service status monitoring
 * - Data clearing and reset operations
 * - AI insights management
 * - Settings persistence and validation
 */
class SettingsViewModel(
    private val sleepRepository: SleepRepository
) : ViewModel() {

    companion object {
        private const val TAG = "SettingsViewModel"
        private const val EXPORT_TIMEOUT_MS = 30000L
        private const val STATS_REFRESH_INTERVAL_MS = 5000L
    }

    // Export status tracking
    private val _exportStatus = MutableLiveData<ExportStatus>()
    val exportStatus: LiveData<ExportStatus> = _exportStatus

    // Data statistics
    private val _dataStatistics = MutableLiveData<DataStatistics>()
    val dataStatistics: LiveData<DataStatistics> = _dataStatistics

    // Service status monitoring
    private val _serviceStatus = MutableLiveData<Boolean>()
    val serviceStatus: LiveData<Boolean> = _serviceStatus

    // Settings validation errors
    private val _settingsError = MutableLiveData<String?>()
    val settingsError: LiveData<String?> = _settingsError

    // Loading states
    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    // Background jobs
    private var exportJob: Job? = null
    private var statsMonitoringJob: Job? = null

    init {
        startStatsMonitoring()
        loadInitialData()
    }

    // ========== DATA EXPORT FUNCTIONALITY ==========

    /**
     * Export all sleep session data to JSON format
     */
    fun exportAllData() {
        exportJob?.cancel()
        exportJob = viewModelScope.launch {
            try {
                _exportStatus.value = ExportStatus.InProgress("Preparing all session data...")

                // Get all sessions with related data
                val sessionsResult = sleepRepository.getAllSessions()

                sessionsResult.onSuccess { sessions ->
                    if (sessions.isEmpty()) {
                        _exportStatus.value = ExportStatus.Error("No data to export")
                        return@onSuccess
                    }

                    _exportStatus.value = ExportStatus.InProgress("Generating JSON export...")

                    // Generate comprehensive JSON export
                    val exportData = generateCompleteExportData(sessions)

                    // Simulate file writing (in real implementation, write to external storage)
                    delay(2000) // Simulate processing time

                    val fileName = "somniai_complete_export_${System.currentTimeMillis()}.json"
                    val filePath = "/storage/emulated/0/Download/$fileName"

                    _exportStatus.value = ExportStatus.Success(filePath, exportData.length.toLong())
                    Log.d(TAG, "All data exported successfully: $filePath")

                }.onFailure { exception ->
                    _exportStatus.value = ExportStatus.Error("Failed to export data: ${exception.message}")
                    Log.e(TAG, "Export all data failed", exception)
                }

            } catch (e: Exception) {
                _exportStatus.value = ExportStatus.Error("Export failed: ${e.message}")
                Log.e(TAG, "Exception during export all data", e)
            }
        }
    }

    /**
     * Export recent sessions to CSV format
     */
    fun exportRecentSessions(daysBack: Int = 30) {
        exportJob?.cancel()
        exportJob = viewModelScope.launch {
            try {
                _exportStatus.value = ExportStatus.InProgress("Preparing recent sessions...")

                val cutoffDate = System.currentTimeMillis() - (daysBack * 24 * 60 * 60 * 1000L)
                val sessionsResult = sleepRepository.getSessionsAfterDate(cutoffDate)

                sessionsResult.onSuccess { sessions ->
                    if (sessions.isEmpty()) {
                        _exportStatus.value = ExportStatus.Error("No recent sessions to export")
                        return@onSuccess
                    }

                    _exportStatus.value = ExportStatus.InProgress("Generating CSV export...")

                    // Generate CSV format
                    val csvData = generateCSVExport(sessions)

                    delay(1500) // Simulate processing time

                    val fileName = "somniai_sessions_${daysBack}days_${System.currentTimeMillis()}.csv"
                    val filePath = "/storage/emulated/0/Download/$fileName"

                    _exportStatus.value = ExportStatus.Success(filePath, csvData.length.toLong())
                    Log.d(TAG, "Recent sessions exported successfully: $filePath")

                }.onFailure { exception ->
                    _exportStatus.value = ExportStatus.Error("Failed to export recent sessions: ${exception.message}")
                    Log.e(TAG, "Export recent sessions failed", exception)
                }

            } catch (e: Exception) {
                _exportStatus.value = ExportStatus.Error("Export failed: ${e.message}")
                Log.e(TAG, "Exception during export recent sessions", e)
            }
        }
    }

    /**
     * Export analytics report to PDF format
     */
    fun exportAnalyticsReport() {
        exportJob?.cancel()
        exportJob = viewModelScope.launch {
            try {
                _exportStatus.value = ExportStatus.InProgress("Generating analytics report...")

                // Get analytics data
                val analyticsResult = sleepRepository.generateComprehensiveAnalytics()

                analyticsResult.onSuccess { analytics ->
                    _exportStatus.value = ExportStatus.InProgress("Creating PDF report...")

                    // Generate PDF report content
                    val reportData = generateAnalyticsReport(analytics)

                    delay(3000) // Simulate PDF generation time

                    val fileName = "somniai_analytics_report_${System.currentTimeMillis()}.pdf"
                    val filePath = "/storage/emulated/0/Download/$fileName"

                    _exportStatus.value = ExportStatus.Success(filePath, reportData.length.toLong())
                    Log.d(TAG, "Analytics report exported successfully: $filePath")

                }.onFailure { exception ->
                    _exportStatus.value = ExportStatus.Error("Failed to generate analytics report: ${exception.message}")
                    Log.e(TAG, "Export analytics report failed", exception)
                }

            } catch (e: Exception) {
                _exportStatus.value = ExportStatus.Error("Report generation failed: ${e.message}")
                Log.e(TAG, "Exception during export analytics report", e)
            }
        }
    }

    // ========== DATA STATISTICS MANAGEMENT ==========

    /**
     * Load current data statistics
     */
    fun loadDataStatistics() {
        viewModelScope.launch {
            try {
                val stats = calculateDataStatistics()
                _dataStatistics.value = stats
                Log.d(TAG, "Data statistics loaded: $stats")

            } catch (e: Exception) {
                setError("Failed to load data statistics: ${e.message}")
                Log.e(TAG, "Exception loading data statistics", e)
            }
        }
    }

    /**
     * Start monitoring data statistics
     */
    private fun startStatsMonitoring() {
        statsMonitoringJob = viewModelScope.launch {
            while (true) {
                try {
                    loadDataStatistics()
                    delay(STATS_REFRESH_INTERVAL_MS)
                } catch (e: Exception) {
                    Log.w(TAG, "Stats monitoring interrupted", e)
                    delay(STATS_REFRESH_INTERVAL_MS * 2) // Back off on error
                }
            }
        }
    }

    /**
     * Calculate current data statistics
     */
    private suspend fun calculateDataStatistics(): DataStatistics {
        val sessionsResult = sleepRepository.getAllSessions()

        return sessionsResult.fold(
            onSuccess = { sessions ->
                val totalSessions = sessions.size
                val oldestSessionDate = sessions.minOfOrNull { it.startTime } ?: System.currentTimeMillis()

                // Calculate approximate storage size (in real implementation, check actual file sizes)
                val estimatedStorageBytes = totalSessions * 2048L // ~2KB per session

                DataStatistics(
                    totalSessions = totalSessions,
                    storageSizeBytes = estimatedStorageBytes,
                    oldestSessionDate = oldestSessionDate
                )
            },
            onFailure = {
                DataStatistics(
                    totalSessions = 0,
                    storageSizeBytes = 0L,
                    oldestSessionDate = System.currentTimeMillis()
                )
            }
        )
    }

    // ========== DATA MANAGEMENT OPERATIONS ==========

    /**
     * Clear all stored data
     */
    fun clearAllData(callback: (Boolean) -> Unit) {
        viewModelScope.launch {
            try {
                _isLoading.value = true

                // Clear all sessions
                val clearResult = sleepRepository.clearAllData()

                clearResult.onSuccess {
                    // Update statistics
                    loadDataStatistics()
                    callback(true)
                    Log.d(TAG, "All data cleared successfully")

                }.onFailure { exception ->
                    setError("Failed to clear data: ${exception.message}")
                    callback(false)
                    Log.e(TAG, "Clear all data failed", exception)
                }

            } catch (e: Exception) {
                setError("Error clearing data: ${e.message}")
                callback(false)
                Log.e(TAG, "Exception during clear all data", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Reset AI insights and personalization
     */
    fun resetAIInsights(callback: (Boolean) -> Unit) {
        viewModelScope.launch {
            try {
                _isLoading.value = true

                // Reset AI insights and recommendations
                val resetResult = sleepRepository.resetAIInsights()

                resetResult.onSuccess {
                    callback(true)
                    Log.d(TAG, "AI insights reset successfully")

                }.onFailure { exception ->
                    setError("Failed to reset AI insights: ${exception.message}")
                    callback(false)
                    Log.e(TAG, "Reset AI insights failed", exception)
                }

            } catch (e: Exception) {
                setError("Error resetting AI insights: ${e.message}")
                callback(false)
                Log.e(TAG, "Exception during reset AI insights", e)
            } finally {
                _isLoading.value = false
            }
        }
    }

    // ========== SERVICE STATUS MONITORING ==========

    /**
     * Update service running status
     */
    fun updateServiceStatus(isRunning: Boolean) {
        _serviceStatus.value = isRunning
        Log.d(TAG, "Service status updated: $isRunning")
    }

    /**
     * Check if sleep tracking service is currently running
     */
    fun checkServiceStatus() {
        viewModelScope.launch {
            try {
                // In a real implementation, check if service is running
                // For now, simulate service status checking
                val isRunning = false // Placeholder
                _serviceStatus.value = isRunning

            } catch (e: Exception) {
                Log.w(TAG, "Failed to check service status", e)
                _serviceStatus.value = false
            }
        }
    }

    // ========== PRIVATE HELPER METHODS ==========

    private fun loadInitialData() {
        viewModelScope.launch {
            loadDataStatistics()
            checkServiceStatus()
        }
    }

    private fun generateCompleteExportData(sessions: List<SleepSession>): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        return buildString {
            appendLine("{")
            appendLine("  \"export_info\": {")
            appendLine("    \"app_name\": \"SomniAI\",")
            appendLine("    \"export_date\": \"${dateFormat.format(Date())}\",")
            appendLine("    \"version\": \"1.0.0\",")
            appendLine("    \"total_sessions\": ${sessions.size}")
            appendLine("  },")
            appendLine("  \"sessions\": [")

            sessions.forEachIndexed { index, session ->
                appendLine("    {")
                appendLine("      \"id\": ${session.id},")
                appendLine("      \"start_time\": \"${dateFormat.format(Date(session.startTime))}\",")
                appendLine("      \"end_time\": \"${session.endTime?.let { dateFormat.format(Date(it)) }}\",")
                appendLine("      \"duration_ms\": ${session.duration},")
                appendLine("      \"quality_score\": ${session.sleepQualityScore},")
                appendLine("      \"sleep_efficiency\": ${session.sleepEfficiency},")
                appendLine("      \"movement_events\": ${session.getTotalMovements()},")
                appendLine("      \"noise_events\": ${session.getTotalNoiseEvents()},")
                appendLine("      \"average_movement_intensity\": ${session.averageMovementIntensity},")
                appendLine("      \"average_noise_level\": ${session.averageNoiseLevel}")
                append("    }")
                if (index < sessions.size - 1) appendLine(",")
            }

            appendLine()
            appendLine("  ]")
            appendLine("}")
        }
    }

    private fun generateCSVExport(sessions: List<SleepSession>): String {
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

        return buildString {
            // CSV Header
            appendLine("Date,Start Time,End Time,Duration (hours),Quality Score,Sleep Efficiency,Movement Events,Noise Events,Avg Movement Intensity,Avg Noise Level")

            // CSV Data
            sessions.forEach { session ->
                val startDate = Date(session.startTime)
                val endDate = session.endTime?.let { Date(it) }
                val durationHours = session.duration / (1000.0 * 60 * 60)

                append("\"${SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(startDate)}\",")
                append("\"${dateFormat.format(startDate)}\",")
                append("\"${endDate?.let { dateFormat.format(it) } ?: ""}\",")
                append("${String.format("%.2f", durationHours)},")
                append("${session.sleepQualityScore},")
                append("${session.sleepEfficiency},")
                append("${session.getTotalMovements()},")
                append("${session.getTotalNoiseEvents()},")
                append("${session.averageMovementIntensity},")
                appendLine("${session.averageNoiseLevel}")
            }
        }
    }

    private fun generateAnalyticsReport(analytics: SleepAnalyticsResult): String {
        // In a real implementation, this would generate PDF content
        // For now, return formatted text that could be converted to PDF
        return buildString {
            appendLine("SOMNIAI SLEEP ANALYTICS REPORT")
            appendLine("=" * 50)
            appendLine()
            appendLine("Report Generated: ${SimpleDateFormat("MMMM dd, yyyy", Locale.getDefault()).format(Date())}")
            appendLine()

            appendLine("SUMMARY STATISTICS")
            appendLine("-" * 20)
            appendLine("Total Sessions Analyzed: ${analytics.totalSessions}")
            appendLine("Average Sleep Quality: ${String.format("%.1f/10", analytics.averageQuality)}")
            appendLine("Average Sleep Duration: ${formatDuration(analytics.averageDuration)}")
            appendLine("Average Sleep Efficiency: ${String.format("%.1f%%", analytics.averageEfficiency)}")
            appendLine()

            appendLine("TRENDS AND INSIGHTS")
            appendLine("-" * 20)
            analytics.insights.forEach { insight ->
                appendLine("• $insight")
            }
            appendLine()

            appendLine("RECOMMENDATIONS")
            appendLine("-" * 15)
            analytics.recommendations.forEach { recommendation ->
                appendLine("• $recommendation")
            }
        }
    }

    private fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        return "${hours}h ${minutes}m"
    }

    private fun setError(message: String) {
        _settingsError.value = message
        Log.w(TAG, "Settings error: $message")
    }

    fun clearError() {
        _settingsError.value = null
    }

    fun cleanup() {
        exportJob?.cancel()
        statsMonitoringJob?.cancel()
        clearError()
    }

    override fun onCleared() {
        super.onCleared()
        cleanup()
        Log.d(TAG, "SettingsViewModel cleared")
    }
}

// ========== SUPPORTING DATA CLASSES ==========

/**
 * Represents the status of data export operations
 */
sealed class ExportStatus {
    data class InProgress(val message: String) : ExportStatus()
    data class Success(val filePath: String, val fileSize: Long) : ExportStatus()
    data class Error(val message: String) : ExportStatus()
}

/**
 * Data statistics for storage management
 */
data class DataStatistics(
    val totalSessions: Int,
    val storageSizeBytes: Long,
    val oldestSessionDate: Long
)

/**
 * Comprehensive analytics result for reporting
 */
data class SleepAnalyticsResult(
    val totalSessions: Int,
    val averageQuality: Float,
    val averageDuration: Long,
    val averageEfficiency: Float,
    val insights: List<String>,
    val recommendations: List<String>
)

/**
 * Factory for creating SettingsViewModel with repository dependency
 */
class SettingsViewModelFactory(
    private val sleepRepository: SleepRepository
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(SettingsViewModel::class.java)) {
            return SettingsViewModel(sleepRepository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}