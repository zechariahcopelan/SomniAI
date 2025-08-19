package com.example.somniai.activities

import android.Manifest
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.MenuItem
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.preference.PreferenceManager
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.database.SleepDatabase
import com.example.somniai.repository.SleepRepository
import com.example.somniai.service.SleepTrackingService
import com.example.somniai.viewmodel.SettingsViewModel
import com.example.somniai.databinding.ActivitySettingsBinding
import com.google.android.material.timepicker.MaterialTimePicker
import com.google.android.material.timepicker.TimeFormat
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

/**
 * Comprehensive Settings Activity for SomniAI
 *
 * Features:
 * - Sensor configuration with real-time preview
 * - Sleep goals and preferences
 * - Data management and export
 * - Notification settings
 * - AI insights configuration
 * - Privacy and permissions
 * - About and support information
 */
class SettingsActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "SettingsActivity"
        private const val REQUEST_PERMISSIONS = 1001
        private const val EXPORT_REQUEST_CODE = 1002

        // Preference keys
        const val PREF_MOVEMENT_SENSITIVITY = "movement_sensitivity"
        const val PREF_AUDIO_THRESHOLD = "audio_threshold"
        const val PREF_TARGET_SLEEP_HOURS = "target_sleep_hours"
        const val PREF_BEDTIME_HOUR = "bedtime_hour"
        const val PREF_BEDTIME_MINUTE = "bedtime_minute"
        const val PREF_WAKE_TIME_HOUR = "wake_time_hour"
        const val PREF_WAKE_TIME_MINUTE = "wake_time_minute"
        const val PREF_BEDTIME_REMINDER = "bedtime_reminder"
        const val PREF_WEEKLY_REPORTS = "weekly_reports"
        const val PREF_ACHIEVEMENT_NOTIFICATIONS = "achievement_notifications"
        const val PREF_INSIGHT_FREQUENCY = "insight_frequency"
        const val PREF_PERSONALIZATION_ENABLED = "personalization_enabled"
        const val PREF_DATA_RETENTION_DAYS = "data_retention_days"
    }

    private lateinit var binding: ActivitySettingsBinding
    private lateinit var settingsViewModel: SettingsViewModel
    private lateinit var sharedPreferences: SharedPreferences

    // Current settings state
    private var isServiceRunning = false
    private var currentMovementSensitivity = 2.5f
    private var currentAudioThreshold = 1000f
    private var targetSleepHours = 8f
    private var bedtimeHour = 22
    private var bedtimeMinute = 0
    private var wakeTimeHour = 6
    private var wakeTimeMinute = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        initializeViewModel()
        initializePreferences()
        setupSettingsInterface()
        setupObservers()
        loadCurrentSettings()
        checkPermissions()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setDisplayShowHomeEnabled(true)
            title = "Settings"
        }
    }

    private fun initializeViewModel() {
        val database = SleepDatabase.getDatabase(this)
        val repository = SleepRepository(database, this)
        val factory = SettingsViewModelFactory(repository)
        settingsViewModel = ViewModelProvider(this, factory)[SettingsViewModel::class.java]
    }

    private fun initializePreferences() {
        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this)
    }

    private fun setupSettingsInterface() {
        setupSensorSettings()
        setupSleepGoals()
        setupDataManagement()
        setupNotifications()
        setupAIInsights()
        setupPrivacySettings()
        setupAboutSection()
    }

    // ========== SENSOR SETTINGS ==========

    private fun setupSensorSettings() {
        // Movement Sensitivity
        binding.movementSensitivitySlider.apply {
            valueFrom = 1f
            valueTo = 5f
            stepSize = 0.5f
            value = currentMovementSensitivity

            addOnChangeListener { _, value, _ ->
                currentMovementSensitivity = value
                updateMovementSensitivityLabel(value)
                savePreference(PREF_MOVEMENT_SENSITIVITY, value)
                updateServiceSettings()
            }
        }

        // Audio Threshold
        binding.audioThresholdSlider.apply {
            valueFrom = 500f
            valueTo = 2000f
            stepSize = 100f
            value = currentAudioThreshold

            addOnChangeListener { _, value, _ ->
                currentAudioThreshold = value
                updateAudioThresholdLabel(value)
                savePreference(PREF_AUDIO_THRESHOLD, value)
                updateServiceSettings()
            }
        }

        // Calibrate Sensors Button
        binding.calibrateSensorsButton.setOnClickListener {
            showSensorCalibration()
        }

        // Test Sensors Button
        binding.testSensorsButton.setOnClickListener {
            testSensors()
        }
    }

    private fun updateMovementSensitivityLabel(value: Float) {
        val description = when {
            value <= 1.5f -> "Very Low (least sensitive)"
            value <= 2.5f -> "Low"
            value <= 3.5f -> "Medium (recommended)"
            value <= 4.5f -> "High"
            else -> "Very High (most sensitive)"
        }
        binding.movementSensitivityLabel.text = "Movement Sensitivity: $description"
    }

    private fun updateAudioThresholdLabel(value: Float) {
        val description = when {
            value <= 700f -> "Very Quiet Environment"
            value <= 1000f -> "Quiet Environment (recommended)"
            value <= 1500f -> "Normal Environment"
            else -> "Noisy Environment"
        }
        binding.audioThresholdLabel.text = "Audio Threshold: $description"
    }

    // ========== SLEEP GOALS ==========

    private fun setupSleepGoals() {
        // Target Sleep Duration
        binding.targetSleepSlider.apply {
            valueFrom = 6f
            valueTo = 10f
            stepSize = 0.25f
            value = targetSleepHours

            addOnChangeListener { _, value, _ ->
                targetSleepHours = value
                updateTargetSleepLabel(value)
                savePreference(PREF_TARGET_SLEEP_HOURS, value)
            }
        }

        // Bedtime Setting
        binding.setBedtimeButton.setOnClickListener {
            showTimePicker("Set Bedtime", bedtimeHour, bedtimeMinute) { hour, minute ->
                bedtimeHour = hour
                bedtimeMinute = minute
                updateBedtimeLabel()
                savePreference(PREF_BEDTIME_HOUR, hour)
                savePreference(PREF_BEDTIME_MINUTE, minute)
            }
        }

        // Wake Time Setting
        binding.setWakeTimeButton.setOnClickListener {
            showTimePicker("Set Wake Time", wakeTimeHour, wakeTimeMinute) { hour, minute ->
                wakeTimeHour = hour
                wakeTimeMinute = minute
                updateWakeTimeLabel()
                savePreference(PREF_WAKE_TIME_HOUR, hour)
                savePreference(PREF_WAKE_TIME_MINUTE, minute)
            }
        }

        // Sleep Goals Toggle
        binding.sleepGoalsSwitch.setOnCheckedChangeListener { _, isChecked ->
            binding.sleepGoalsOptions.visibility = if (isChecked) {
                android.view.View.VISIBLE
            } else {
                android.view.View.GONE
            }
        }
    }

    private fun updateTargetSleepLabel(hours: Float) {
        val hoursInt = hours.toInt()
        val minutes = ((hours - hoursInt) * 60).toInt()
        binding.targetSleepLabel.text = if (minutes == 0) {
            "Target: ${hoursInt}h"
        } else {
            "Target: ${hoursInt}h ${minutes}m"
        }
    }

    private fun updateBedtimeLabel() {
        val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
        val calendar = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, bedtimeHour)
            set(Calendar.MINUTE, bedtimeMinute)
        }
        binding.bedtimeLabel.text = "Bedtime: ${timeFormat.format(calendar.time)}"
    }

    private fun updateWakeTimeLabel() {
        val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())
        val calendar = Calendar.getInstance().apply {
            set(Calendar.HOUR_OF_DAY, wakeTimeHour)
            set(Calendar.MINUTE, wakeTimeMinute)
        }
        binding.wakeTimeLabel.text = "Wake Time: ${timeFormat.format(calendar.time)}"
    }

    // ========== DATA MANAGEMENT ==========

    private fun setupDataManagement() {
        // Export Data
        binding.exportDataButton.setOnClickListener {
            showExportOptions()
        }

        // Clear All Data
        binding.clearDataButton.setOnClickListener {
            confirmClearAllData()
        }

        // Data Retention
        binding.dataRetentionSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                val retentionDays = when (position) {
                    0 -> 30  // 1 month
                    1 -> 90  // 3 months
                    2 -> 180 // 6 months
                    3 -> 365 // 1 year
                    else -> -1 // Forever
                }
                savePreference(PREF_DATA_RETENTION_DAYS, retentionDays)
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        // Load data usage statistics
        loadDataStatistics()
    }

    // ========== NOTIFICATIONS ==========

    private fun setupNotifications() {
        // Bedtime Reminder
        binding.bedtimeReminderSwitch.setOnCheckedChangeListener { _, isChecked ->
            savePreference(PREF_BEDTIME_REMINDER, isChecked)
            if (isChecked) {
                scheduleNotifications()
            }
        }

        // Weekly Reports
        binding.weeklyReportsSwitch.setOnCheckedChangeListener { _, isChecked ->
            savePreference(PREF_WEEKLY_REPORTS, isChecked)
        }

        // Achievement Notifications
        binding.achievementNotificationsSwitch.setOnCheckedChangeListener { _, isChecked ->
            savePreference(PREF_ACHIEVEMENT_NOTIFICATIONS, isChecked)
        }

        // Notification Settings Button
        binding.notificationSettingsButton.setOnClickListener {
            openNotificationSettings()
        }
    }

    // ========== AI INSIGHTS ==========

    private fun setupAIInsights() {
        // Personalization Toggle
        binding.personalizationSwitch.setOnCheckedChangeListener { _, isChecked ->
            savePreference(PREF_PERSONALIZATION_ENABLED, isChecked)
            binding.personalizationOptions.visibility = if (isChecked) {
                android.view.View.VISIBLE
            } else {
                android.view.View.GONE
            }
        }

        // Insight Frequency
        binding.insightFrequencySpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                val frequency = when (position) {
                    0 -> "daily"
                    1 -> "weekly"
                    2 -> "monthly"
                    else -> "weekly"
                }
                savePreference(PREF_INSIGHT_FREQUENCY, frequency)
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
        }

        // Reset AI Insights
        binding.resetAiInsightsButton.setOnClickListener {
            confirmResetAIInsights()
        }
    }

    // ========== PRIVACY SETTINGS ==========

    private fun setupPrivacySettings() {
        // Permission Status
        updatePermissionStatus()

        // Request Permissions Button
        binding.requestPermissionsButton.setOnClickListener {
            requestAllPermissions()
        }

        // Privacy Policy
        binding.privacyPolicyButton.setOnClickListener {
            openPrivacyPolicy()
        }

        // Data Usage Info
        binding.dataUsageButton.setOnClickListener {
            showDataUsageInfo()
        }
    }

    // ========== ABOUT SECTION ==========

    private fun setupAboutSection() {
        // App Version
        try {
            val packageInfo = packageManager.getPackageInfo(packageName, 0)
            binding.appVersionText.text = "Version ${packageInfo.versionName} (${packageInfo.longVersionCode})"
        } catch (e: PackageManager.NameNotFoundException) {
            binding.appVersionText.text = "Version 1.0.0"
        }

        // Support Contact
        binding.supportButton.setOnClickListener {
            openSupportContact()
        }

        // GitHub Repository
        binding.githubButton.setOnClickListener {
            openGitHubRepository()
        }

        // Rate App
        binding.rateAppButton.setOnClickListener {
            openPlayStore()
        }

        // Share App
        binding.shareAppButton.setOnClickListener {
            shareApp()
        }
    }

    // ========== OBSERVERS ==========

    private fun setupObservers() {
        // Observe export status
        settingsViewModel.exportStatus.observe(this) { status ->
            when (status) {
                is ExportStatus.Success -> {
                    Toast.makeText(this, "Data exported successfully", Toast.LENGTH_SHORT).show()
                }
                is ExportStatus.Error -> {
                    Toast.makeText(this, "Export failed: ${status.message}", Toast.LENGTH_LONG).show()
                }
                is ExportStatus.InProgress -> {
                    // Show progress indicator
                }
            }
        }

        // Observe data statistics
        settingsViewModel.dataStatistics.observe(this) { stats ->
            updateDataStatisticsDisplay(stats)
        }

        // Observe service status
        settingsViewModel.serviceStatus.observe(this) { status ->
            isServiceRunning = status
            updateSensorTestAvailability()
        }
    }

    // ========== HELPER METHODS ==========

    private fun loadCurrentSettings() {
        currentMovementSensitivity = sharedPreferences.getFloat(PREF_MOVEMENT_SENSITIVITY, 2.5f)
        currentAudioThreshold = sharedPreferences.getFloat(PREF_AUDIO_THRESHOLD, 1000f)
        targetSleepHours = sharedPreferences.getFloat(PREF_TARGET_SLEEP_HOURS, 8f)
        bedtimeHour = sharedPreferences.getInt(PREF_BEDTIME_HOUR, 22)
        bedtimeMinute = sharedPreferences.getInt(PREF_BEDTIME_MINUTE, 0)
        wakeTimeHour = sharedPreferences.getInt(PREF_WAKE_TIME_HOUR, 6)
        wakeTimeMinute = sharedPreferences.getInt(PREF_WAKE_TIME_MINUTE, 0)

        // Update UI
        updateMovementSensitivityLabel(currentMovementSensitivity)
        updateAudioThresholdLabel(currentAudioThreshold)
        updateTargetSleepLabel(targetSleepHours)
        updateBedtimeLabel()
        updateWakeTimeLabel()

        // Set switch states
        binding.bedtimeReminderSwitch.isChecked = sharedPreferences.getBoolean(PREF_BEDTIME_REMINDER, true)
        binding.weeklyReportsSwitch.isChecked = sharedPreferences.getBoolean(PREF_WEEKLY_REPORTS, true)
        binding.achievementNotificationsSwitch.isChecked = sharedPreferences.getBoolean(PREF_ACHIEVEMENT_NOTIFICATIONS, true)
        binding.personalizationSwitch.isChecked = sharedPreferences.getBoolean(PREF_PERSONALIZATION_ENABLED, true)
    }

    private fun savePreference(key: String, value: Any) {
        with(sharedPreferences.edit()) {
            when (value) {
                is Boolean -> putBoolean(key, value)
                is Int -> putInt(key, value)
                is Float -> putFloat(key, value)
                is String -> putString(key, value)
            }
            apply()
        }
    }

    private fun updateServiceSettings() {
        // Update running service with new settings
        if (isServiceRunning) {
            val intent = Intent(this, SleepTrackingService::class.java).apply {
                putExtra("movement_sensitivity", currentMovementSensitivity)
                putExtra("audio_threshold", currentAudioThreshold)
            }
            startService(intent)
        }
    }

    private fun showTimePicker(title: String, hour: Int, minute: Int, callback: (Int, Int) -> Unit) {
        val timePicker = MaterialTimePicker.Builder()
            .setTimeFormat(TimeFormat.CLOCK_12H)
            .setHour(hour)
            .setMinute(minute)
            .setTitleText(title)
            .build()

        timePicker.addOnPositiveButtonClickListener {
            callback(timePicker.hour, timePicker.minute)
        }

        timePicker.show(supportFragmentManager, "TIME_PICKER")
    }

    private fun showSensorCalibration() {
        AlertDialog.Builder(this)
            .setTitle("Sensor Calibration")
            .setMessage("Place your phone on a stable surface (like your nightstand) and press 'Calibrate'. This will help improve movement detection accuracy.")
            .setPositiveButton("Calibrate") { _, _ ->
                startSensorCalibration()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startSensorCalibration() {
        // Implement sensor calibration logic
        Toast.makeText(this, "Calibrating sensors... Please keep phone still.", Toast.LENGTH_LONG).show()

        lifecycleScope.launch {
            // Simulate calibration process
            kotlinx.coroutines.delay(3000)
            Toast.makeText(this@SettingsActivity, "Sensor calibration complete!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun testSensors() {
        if (!isServiceRunning) {
            Toast.makeText(this, "Start sleep tracking to test sensors", Toast.LENGTH_SHORT).show()
            return
        }

        AlertDialog.Builder(this)
            .setTitle("Sensor Test")
            .setMessage("Tap lightly on your phone or make a small sound to test sensor responsiveness.")
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showExportOptions() {
        val options = arrayOf(
            "Export All Data (JSON)",
            "Export Recent Sessions (CSV)",
            "Export Analytics Report (PDF)"
        )

        AlertDialog.Builder(this)
            .setTitle("Export Data")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> exportAllData()
                    1 -> exportRecentSessions()
                    2 -> exportAnalyticsReport()
                }
            }
            .show()
    }

    private fun exportAllData() {
        settingsViewModel.exportAllData()
    }

    private fun exportRecentSessions() {
        settingsViewModel.exportRecentSessions(30) // Last 30 days
    }

    private fun exportAnalyticsReport() {
        settingsViewModel.exportAnalyticsReport()
    }

    private fun confirmClearAllData() {
        AlertDialog.Builder(this)
            .setTitle("Clear All Data")
            .setMessage("This will permanently delete all your sleep sessions, analytics, and settings. This action cannot be undone.")
            .setPositiveButton("Clear All") { _, _ ->
                clearAllData()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun clearAllData() {
        settingsViewModel.clearAllData {
            Toast.makeText(this, "All data cleared", Toast.LENGTH_SHORT).show()
            loadDataStatistics()
        }
    }

    private fun confirmResetAIInsights() {
        AlertDialog.Builder(this)
            .setTitle("Reset AI Insights")
            .setMessage("This will reset all AI personalization and start fresh recommendations. Your sleep data will remain intact.")
            .setPositiveButton("Reset") { _, _ ->
                resetAIInsights()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun resetAIInsights() {
        settingsViewModel.resetAIInsights {
            Toast.makeText(this, "AI insights reset", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadDataStatistics() {
        settingsViewModel.loadDataStatistics()
    }

    private fun updateDataStatisticsDisplay(stats: DataStatistics) {
        binding.totalSessionsCount.text = "${stats.totalSessions} sessions"
        binding.dataStorageSize.text = formatFileSize(stats.storageSizeBytes)
        binding.oldestSessionDate.text = "Since: ${formatDate(stats.oldestSessionDate)}"
    }

    private fun formatFileSize(bytes: Long): String {
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        return when {
            mb >= 1 -> String.format("%.1f MB", mb)
            kb >= 1 -> String.format("%.1f KB", kb)
            else -> "$bytes bytes"
        }
    }

    private fun formatDate(timestamp: Long): String {
        val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
        return dateFormat.format(Date(timestamp))
    }

    private fun checkPermissions() {
        val requiredPermissions = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )

        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            binding.permissionWarning.visibility = android.view.View.VISIBLE
        } else {
            binding.permissionWarning.visibility = android.view.View.GONE
        }

        updatePermissionStatus()
    }

    private fun updatePermissionStatus() {
        val audioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
        val storagePermission = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)

        binding.audioPermissionStatus.text = if (audioPermission == PackageManager.PERMISSION_GRANTED) "✓ Granted" else "✗ Required"
        binding.storagePermissionStatus.text = if (storagePermission == PackageManager.PERMISSION_GRANTED) "✓ Granted" else "✗ Required"
    }

    private fun requestAllPermissions() {
        val permissions = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )

        ActivityCompat.requestPermissions(this, permissions, REQUEST_PERMISSIONS)
    }

    private fun scheduleNotifications() {
        // Implement notification scheduling
        Toast.makeText(this, "Bedtime reminders scheduled", Toast.LENGTH_SHORT).show()
    }

    private fun openNotificationSettings() {
        val intent = Intent().apply {
            action = "android.settings.APP_NOTIFICATION_SETTINGS"
            putExtra("android.provider.extra.APP_PACKAGE", packageName)
        }
        startActivity(intent)
    }

    private fun openPrivacyPolicy() {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://somniai.app/privacy"))
        startActivity(intent)
    }

    private fun showDataUsageInfo() {
        AlertDialog.Builder(this)
            .setTitle("Data Usage")
            .setMessage("""
                SomniAI processes all data locally on your device:
                
                • Audio: Only volume levels are analyzed (no recording)
                • Movement: Accelerometer data for sleep phase detection
                • Storage: All data stays on your device
                • Analytics: Generated locally using AI algorithms
                
                No personal data is transmitted to external servers.
            """.trimIndent())
            .setPositiveButton("OK", null)
            .show()
    }

    private fun openSupportContact() {
        val intent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:support@somniai.app")
            putExtra(Intent.EXTRA_SUBJECT, "SomniAI Support")
        }
        startActivity(intent)
    }

    private fun openGitHubRepository() {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://github.com/your-username/somniai"))
        startActivity(intent)
    }

    private fun openPlayStore() {
        try {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=$packageName"))
            startActivity(intent)
        } catch (e: Exception) {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/apps/details?id=$packageName"))
            startActivity(intent)
        }
    }

    private fun shareApp() {
        val shareText = "Check out SomniAI - AI-powered sleep tracking! https://play.google.com/store/apps/details?id=$packageName"
        val shareIntent = Intent().apply {
            action = Intent.ACTION_SEND
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, shareText)
        }
        startActivity(Intent.createChooser(shareIntent, "Share SomniAI"))
    }

    private fun updateSensorTestAvailability() {
        binding.testSensorsButton.isEnabled = isServiceRunning
        binding.testSensorsButton.alpha = if (isServiceRunning) 1.0f else 0.5f
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS) {
            updatePermissionStatus()
            checkPermissions()
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

    override fun onDestroy() {
        super.onDestroy()
        settingsViewModel.cleanup()
    }
}