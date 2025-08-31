package com.example.somniai.utils

import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.math.*

/**
 * Time and Duration utilities for SomniAI sleep tracking
 *
 * Provides comprehensive time calculations, formatting, and sleep-specific
 * time analysis functions
 */
object TimeUtils {

    // ========== DURATION FORMATTING ==========

    /**
     * Format duration in milliseconds to readable string
     * @param durationMs Duration in milliseconds
     * @return Formatted string like "8h 45m" or "45m" or "2h 0m"
     */
    fun formatDuration(durationMs: Long): String {
        if (durationMs < 0) return "0m"

        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)

        return when {
            hours > 0 -> "${hours}h ${minutes}m"
            minutes > 0 -> "${minutes}m"
            else -> "0m"
        }
    }

    /**
     * Format duration with seconds precision
     */
    fun formatDurationWithSeconds(durationMs: Long): String {
        if (durationMs < 0) return "0s"

        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        val seconds = (durationMs % (1000 * 60)) / 1000

        return when {
            hours > 0 -> "${hours}h ${minutes}m ${seconds}s"
            minutes > 0 -> "${minutes}m ${seconds}s"
            else -> "${seconds}s"
        }
    }

    /**
     * Format duration in hours with decimal precision
     */
    fun formatDurationAsHours(durationMs: Long): String {
        val hours = durationMs / (1000.0 * 60 * 60)
        return String.format("%.1f hours", hours)
    }

    /**
     * Format duration for display in UI (shorter format)
     */
    fun formatDurationShort(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)

        return when {
            hours > 0 && minutes > 0 -> "${hours}:${String.format("%02d", minutes)}h"
            hours > 0 -> "${hours}h"
            else -> "${minutes}m"
        }
    }

    // ========== TIME FORMATTING ==========

    /**
     * Format timestamp to time string (HH:mm)
     */
    fun formatTime(timestamp: Long, use24Hour: Boolean = true): String {
        val format = if (use24Hour) "HH:mm" else "h:mm a"
        return SimpleDateFormat(format, Locale.getDefault()).format(Date(timestamp))
    }

    /**
     * Format timestamp to date string
     */
    fun formatDate(timestamp: Long, pattern: String = "MMM dd, yyyy"): String {
        return SimpleDateFormat(pattern, Locale.getDefault()).format(Date(timestamp))
    }

    /**
     * Format timestamp to date and time string
     */
    fun formatDateTime(timestamp: Long, use24Hour: Boolean = true): String {
        val timePattern = if (use24Hour) "HH:mm" else "h:mm a"
        val pattern = "MMM dd, yyyy '$timePattern'"
        return SimpleDateFormat(pattern, Locale.getDefault()).format(Date(timestamp))
    }

    /**
     * Format time range for display
     */
    fun formatTimeRange(startTime: Long, endTime: Long, use24Hour: Boolean = true): String {
        val startFormatted = formatTime(startTime, use24Hour)
        val endFormatted = formatTime(endTime, use24Hour)

        // Check if same day
        val startCal = Calendar.getInstance().apply { timeInMillis = startTime }
        val endCal = Calendar.getInstance().apply { timeInMillis = endTime }

        return if (startCal.get(Calendar.DAY_OF_YEAR) == endCal.get(Calendar.DAY_OF_YEAR)) {
            "$startFormatted - $endFormatted"
        } else {
            val startDate = formatDate(startTime, "MMM dd")
            val endDate = formatDate(endTime, "MMM dd")
            "$startDate $startFormatted - $endDate $endFormatted"
        }
    }

    // ========== SLEEP TIME CALCULATIONS ==========

    /**
     * Calculate sleep efficiency percentage
     * @param totalTimeInBed Total time from bed to wake up
     * @param actualSleepTime Time actually asleep
     * @return Sleep efficiency as percentage (0-100)
     */
    fun calculateSleepEfficiency(totalTimeInBed: Long, actualSleepTime: Long): Float {
        return if (totalTimeInBed > 0) {
            (actualSleepTime.toFloat() / totalTimeInBed * 100).coerceIn(0f, 100f)
        } else 0f
    }

    /**
     * Calculate sleep latency (time to fall asleep)
     */
    fun calculateSleepLatency(bedTime: Long, sleepOnsetTime: Long): Long {
        return maxOf(0L, sleepOnsetTime - bedTime)
    }

    /**
     * Calculate wake after sleep onset (WASO)
     */
    fun calculateWASO(sleepPeriods: List<Pair<Long, Long>>): Long {
        if (sleepPeriods.isEmpty()) return 0L

        val totalSleepPeriod = sleepPeriods.last().second - sleepPeriods.first().first
        val actualSleepTime = sleepPeriods.sumOf { it.second - it.first }

        return maxOf(0L, totalSleepPeriod - actualSleepTime)
    }

    /**
     * Determine if time falls within typical bedtime window
     */
    fun isTypicalBedtime(timestamp: Long): Boolean {
        val hour = Calendar.getInstance().apply { timeInMillis = timestamp }.get(Calendar.HOUR_OF_DAY)
        return hour >= 21 || hour <= 2 // 9 PM to 2 AM
    }

    /**
     * Determine if time falls within typical wake time window
     */
    fun isTypicalWakeTime(timestamp: Long): Boolean {
        val hour = Calendar.getInstance().apply { timeInMillis = timestamp }.get(Calendar.HOUR_OF_DAY)
        return hour in 5..10 // 5 AM to 10 AM
    }

    // ========== SLEEP CYCLE CALCULATIONS ==========

    /**
     * Calculate number of complete sleep cycles
     * Average sleep cycle is ~90 minutes
     */
    fun calculateSleepCycles(sleepDurationMs: Long): Int {
        val cycleLength = 90 * 60 * 1000L // 90 minutes in milliseconds
        return (sleepDurationMs / cycleLength).toInt()
    }

    /**
     * Get optimal wake time suggestions based on sleep cycles
     * Returns list of timestamps for optimal wake times
     */
    fun getOptimalWakeTimes(sleepStartTime: Long, maxCycles: Int = 6): List<Long> {
        val cycleLength = 90 * 60 * 1000L // 90 minutes
        val results = mutableListOf<Long>()

        // Start after minimum sleep (3 hours) and add cycle-based wake times
        val minSleep = 3 * 60 * 60 * 1000L // 3 hours
        var wakeTime = sleepStartTime + minSleep

        for (i in 1..maxCycles) {
            wakeTime = sleepStartTime + (i * cycleLength)
            if (wakeTime - sleepStartTime <= 10 * 60 * 60 * 1000L) { // Max 10 hours
                results.add(wakeTime)
            }
        }

        return results
    }

    /**
     * Calculate ideal bedtime for target wake time
     */
    fun calculateIdealBedtime(targetWakeTime: Long, desiredSleepHours: Float = 8f): Long {
        val sleepDurationMs = (desiredSleepHours * 60 * 60 * 1000).toLong()
        val sleepLatencyBuffer = 15 * 60 * 1000L // 15 minutes to fall asleep
        return targetWakeTime - sleepDurationMs - sleepLatencyBuffer
    }

    // ========== TIME ZONE HANDLING ==========

    /**
     * Convert timestamp to different timezone
     */
    fun convertToTimezone(timestamp: Long, targetTimezone: String): Long {
        val sourceTimezone = TimeZone.getDefault()
        val targetTz = TimeZone.getTimeZone(targetTimezone)

        val sourceOffset = sourceTimezone.getOffset(timestamp)
        val targetOffset = targetTz.getOffset(timestamp)

        return timestamp + (sourceOffset - targetOffset)
    }

    /**
     * Get timezone offset in hours
     */
    fun getTimezoneOffsetHours(timestamp: Long = System.currentTimeMillis()): Float {
        val offset = TimeZone.getDefault().getOffset(timestamp)
        return offset / (1000f * 60 * 60)
    }

    /**
     * Check if timestamp is in daylight saving time
     */
    fun isDaylightSavingTime(timestamp: Long): Boolean {
        return TimeZone.getDefault().inDaylightTime(Date(timestamp))
    }

    // ========== DATE CALCULATIONS ==========

    /**
     * Get start of day timestamp
     */
    fun getStartOfDay(timestamp: Long): Long {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = timestamp
        calendar.set(Calendar.HOUR_OF_DAY, 0)
        calendar.set(Calendar.MINUTE, 0)
        calendar.set(Calendar.SECOND, 0)
        calendar.set(Calendar.MILLISECOND, 0)
        return calendar.timeInMillis
    }

    /**
     * Get end of day timestamp
     */
    fun getEndOfDay(timestamp: Long): Long {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = timestamp
        calendar.set(Calendar.HOUR_OF_DAY, 23)
        calendar.set(Calendar.MINUTE, 59)
        calendar.set(Calendar.SECOND, 59)
        calendar.set(Calendar.MILLISECOND, 999)
        return calendar.timeInMillis
    }

    /**
     * Get days between two timestamps
     */
    fun getDaysBetween(startTime: Long, endTime: Long): Int {
        val diffMs = abs(endTime - startTime)
        return (diffMs / (24 * 60 * 60 * 1000L)).toInt()
    }

    /**
     * Check if two timestamps are on the same day
     */
    fun isSameDay(timestamp1: Long, timestamp2: Long): Boolean {
        val cal1 = Calendar.getInstance().apply { timeInMillis = timestamp1 }
        val cal2 = Calendar.getInstance().apply { timeInMillis = timestamp2 }

        return cal1.get(Calendar.YEAR) == cal2.get(Calendar.YEAR) &&
                cal1.get(Calendar.DAY_OF_YEAR) == cal2.get(Calendar.DAY_OF_YEAR)
    }

    /**
     * Get week number of year for timestamp
     */
    fun getWeekOfYear(timestamp: Long): Int {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = timestamp
        return calendar.get(Calendar.WEEK_OF_YEAR)
    }

    // ========== RELATIVE TIME ==========

    /**
     * Get relative time string (e.g., "2 hours ago", "in 30 minutes")
     */
    fun getRelativeTime(timestamp: Long): String {
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        val absDiff = abs(diff)

        val future = diff < 0
        val prefix = if (future) "in " else ""
        val suffix = if (future) "" else " ago"

        return when {
            absDiff < 60 * 1000 -> "just now"
            absDiff < 60 * 60 * 1000 -> {
                val minutes = absDiff / (60 * 1000)
                "$prefix$minutes minute${if (minutes != 1L) "s" else ""}$suffix"
            }
            absDiff < 24 * 60 * 60 * 1000 -> {
                val hours = absDiff / (60 * 60 * 1000)
                "$prefix$hours hour${if (hours != 1L) "s" else ""}$suffix"
            }
            absDiff < 7 * 24 * 60 * 60 * 1000 -> {
                val days = absDiff / (24 * 60 * 60 * 1000)
                "$prefix$days day${if (days != 1L) "s" else ""}$suffix"
            }
            absDiff < 30 * 24 * 60 * 60 * 1000 -> {
                val weeks = absDiff / (7 * 24 * 60 * 60 * 1000)
                "$prefix$weeks week${if (weeks != 1L) "s" else ""}$suffix"
            }
            else -> formatDate(timestamp)
        }
    }

    // ========== SLEEP ANALYSIS HELPERS ==========

    /**
     * Calculate sleep debt based on target vs actual sleep
     */
    fun calculateSleepDebt(actualSleepMs: Long, targetSleepMs: Long = 8 * 60 * 60 * 1000L): Long {
        return maxOf(0L, targetSleepMs - actualSleepMs)
    }

    /**
     * Determine sleep quality based on duration and efficiency
     */
    fun getSleepQualityCategory(durationMs: Long, efficiency: Float): String {
        val hours = durationMs / (1000f * 60 * 60)

        return when {
            hours >= 7 && efficiency >= 85f -> "Excellent"
            hours >= 6 && efficiency >= 80f -> "Good"
            hours >= 5 && efficiency >= 75f -> "Fair"
            else -> "Poor"
        }
    }

    /**
     * Check if sleep timing is consistent with previous patterns
     */
    fun isConsistentSleepTiming(
        currentBedtime: Long,
        previousBedtimes: List<Long>,
        toleranceMinutes: Int = 30
    ): Boolean {
        if (previousBedtimes.isEmpty()) return true

        val avgBedtime = previousBedtimes.average().toLong()
        val difference = abs(currentBedtime - avgBedtime)
        val toleranceMs = toleranceMinutes * 60 * 1000L

        return difference <= toleranceMs
    }

    /**
     * Get sleep timing consistency score (0-1)
     */
    fun calculateTimingConsistency(bedtimes: List<Long>): Float {
        if (bedtimes.size < 2) return 1.0f

        // Convert to hour of day for comparison
        val hours = bedtimes.map { timestamp ->
            Calendar.getInstance().apply { timeInMillis = timestamp }.get(Calendar.HOUR_OF_DAY)
        }

        val mean = hours.average()
        val variance = hours.map { (it - mean).pow(2) }.average()
        val stdDev = sqrt(variance)

        // Convert to 0-1 score (lower variance = higher consistency)
        return (1.0f / (1.0f + stdDev.toFloat())).coerceIn(0f, 1f)
    }

    // ========== UTILITY CONVERSIONS ==========

    /**
     * Convert hours to milliseconds
     */
    fun hoursToMs(hours: Float): Long = (hours * 60 * 60 * 1000).toLong()

    /**
     * Convert minutes to milliseconds
     */
    fun minutesToMs(minutes: Int): Long = minutes * 60 * 1000L

    /**
     * Convert milliseconds to hours
     */
    fun msToHours(ms: Long): Float = ms / (1000f * 60 * 60)

    /**
     * Convert milliseconds to minutes
     */
    fun msToMinutes(ms: Long): Int = (ms / (1000 * 60)).toInt()

    /**
     * Round timestamp to nearest minute
     */
    fun roundToNearestMinute(timestamp: Long): Long {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = timestamp
        calendar.set(Calendar.SECOND, 0)
        calendar.set(Calendar.MILLISECOND, 0)
        return calendar.timeInMillis
    }
}