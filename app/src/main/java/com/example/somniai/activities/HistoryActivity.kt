package com.example.somniai.activities

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SearchView
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.DividerItemDecoration
import com.example.somniai.R
import com.example.somniai.data.*
import com.example.somniai.database.SleepDatabase
import com.example.somniai.repository.SleepRepository
import com.example.somniai.viewmodel.HistoryViewModel
import com.example.somniai.databinding.ActivityHistoryBinding
import com.google.android.material.chip.Chip
import com.google.android.material.datepicker.MaterialDatePicker
import java.text.SimpleDateFormat
import java.util.*

/**
 * Session History Activity with advanced filtering and search capabilities
 *
 * Features:
 * - RecyclerView with session list using SessionSummaryDTO
 * - Advanced filtering by date range, quality, duration
 * - Search functionality
 * - Session detail expansion
 * - Export and sharing capabilities
 * - Pagination for large datasets
 * - Integration with SleepRepository
 */
class HistoryActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "HistoryActivity"
        private const val SESSIONS_PER_PAGE = 20
        private const val MIN_SEARCH_LENGTH = 2
    }

    private lateinit var binding: ActivityHistoryBinding
    private lateinit var historyViewModel: HistoryViewModel
    private lateinit var sessionAdapter: SessionHistoryAdapter

    // Current filter state
    private var currentFilterType = FilterType.ALL
    private var currentDateRange: Pair<Long, Long>? = null
    private var currentSearchQuery = ""
    private var isLoading = false

    // Date formatters
    private val dateFormat = SimpleDateFormat("MMM dd, yyyy", Locale.getDefault())
    private val timeFormat = SimpleDateFormat("h:mm a", Locale.getDefault())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        initializeViewModel()
        setupRecyclerView()
        setupFilterChips()
        setupSearchView()
        setupObservers()
        setupSwipeRefresh()

        // Load initial data
        loadSessionHistory()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.apply {
            setDisplayHomeAsUpEnabled(true)
            setDisplayShowHomeEnabled(true)
            title = "Sleep History"
        }
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                finish()
                true
            }
            R.id.action_export_all -> {
                exportAllSessions()
                true
            }
            R.id.action_clear_all -> {
                confirmClearAllSessions()
                true
            }
            R.id.action_sort -> {
                showSortOptions()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun initializeViewModel() {
        val database = SleepDatabase.getDatabase(this)
        val repository = SleepRepository(database, this)
        val factory = HistoryViewModelFactory(repository)
        historyViewModel = ViewModelProvider(this, factory)[HistoryViewModel::class.java]
    }

    private fun setupRecyclerView() {
        sessionAdapter = SessionHistoryAdapter(
            onSessionClick = { session -> showSessionDetails(session) },
            onSessionLongClick = { session -> showSessionContextMenu(session) }
        )

        binding.sessionsRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@HistoryActivity)
            adapter = sessionAdapter

            // Add dividers between items
            val divider = DividerItemDecoration(this@HistoryActivity, DividerItemDecoration.VERTICAL)
            addItemDecoration(divider)

            // Add scroll listener for pagination
            addOnScrollListener(PaginationScrollListener())
        }
    }

    private fun setupFilterChips() {
        // Quality filter chips
        val qualityFilters = listOf(
            "All" to FilterType.ALL,
            "Excellent (8+)" to FilterType.QUALITY_EXCELLENT,
            "Good (6-8)" to FilterType.QUALITY_GOOD,
            "Poor (<6)" to FilterType.QUALITY_POOR
        )

        qualityFilters.forEach { (text, filterType) ->
            val chip = Chip(this).apply {
                this.text = text
                isCheckable = true
                isChecked = filterType == FilterType.ALL

                setOnCheckedChangeListener { _, isChecked ->
                    if (isChecked) {
                        // Uncheck other chips
                        binding.filterChipGroup.children.forEach { child ->
                            if (child != this && child is Chip) {
                                child.isChecked = false
                            }
                        }
                        currentFilterType = filterType
                        applyFilters()
                    }
                }
            }
            binding.filterChipGroup.addView(chip)
        }

        // Date range filter
        binding.dateRangeChip.setOnClickListener {
            showDateRangePicker()
        }

        // Clear filters button
        binding.clearFiltersButton.setOnClickListener {
            clearAllFilters()
        }
    }

    private fun setupSearchView() {
        binding.searchView.setOnQueryTextListener(object : SearchView.OnQueryTextListener {
            override fun onQueryTextSubmit(query: String?): Boolean {
                query?.let {
                    currentSearchQuery = it
                    applyFilters()
                }
                binding.searchView.clearFocus()
                return true
            }

            override fun onQueryTextChange(newText: String?): Boolean {
                if (newText.isNullOrEmpty()) {
                    currentSearchQuery = ""
                    applyFilters()
                } else if (newText.length >= MIN_SEARCH_LENGTH) {
                    currentSearchQuery = newText
                    applyFilters()
                }
                return true
            }
        })
    }

    private fun setupObservers() {
        // Observe loading state
        historyViewModel.isLoading.observe(this) { loading ->
            isLoading = loading
            updateLoadingState(loading)
        }

        // Observe error messages
        historyViewModel.errorMessage.observe(this) { error ->
            error?.let {
                showError(it)
                historyViewModel.clearError()
            }
        }

        // Observe session data
        historyViewModel.sessionHistory.observe(this) { sessions ->
            updateSessionList(sessions)
        }

        // Observe filtered sessions count
        historyViewModel.filteredSessionsCount.observe(this) { count ->
            updateResultsInfo(count)
        }

        // Observe statistics
        historyViewModel.historyStatistics.observe(this) { stats ->
            updateStatistics(stats)
        }
    }

    private fun setupSwipeRefresh() {
        binding.swipeRefreshLayout.setOnRefreshListener {
            refreshSessionHistory()
        }

        // Set color scheme
        binding.swipeRefreshLayout.setColorSchemeColors(
            ContextCompat.getColor(this, R.color.accent_green),
            ContextCompat.getColor(this, R.color.accent_amber)
        )
    }

    // ========== DATA LOADING ==========

    private fun loadSessionHistory() {
        historyViewModel.loadSessionHistory(
            filterType = currentFilterType,
            dateRange = currentDateRange,
            searchQuery = currentSearchQuery,
            page = 0,
            pageSize = SESSIONS_PER_PAGE
        )
    }

    private fun refreshSessionHistory() {
        historyViewModel.refreshSessionHistory(
            filterType = currentFilterType,
            dateRange = currentDateRange,
            searchQuery = currentSearchQuery
        )
    }

    private fun loadMoreSessions() {
        if (!isLoading) {
            historyViewModel.loadMoreSessions()
        }
    }

    // ========== FILTERING ==========

    private fun applyFilters() {
        historyViewModel.applyFilters(
            filterType = currentFilterType,
            dateRange = currentDateRange,
            searchQuery = currentSearchQuery
        )
    }

    private fun clearAllFilters() {
        currentFilterType = FilterType.ALL
        currentDateRange = null
        currentSearchQuery = ""
        binding.searchView.setQuery("", false)
        binding.dateRangeChip.text = "Date Range"
        binding.dateRangeChip.isChecked = false

        // Reset quality filter chips
        binding.filterChipGroup.children.forEachIndexed { index, child ->
            if (child is Chip) {
                child.isChecked = index == 0 // Check "All" chip
            }
        }

        applyFilters()
    }

    private fun showDateRangePicker() {
        val dateRangePicker = MaterialDatePicker.Builder.dateRangePicker()
            .setTitleText("Select date range")
            .build()

        dateRangePicker.addOnPositiveButtonClickListener { selection ->
            val startDate = selection.first
            val endDate = selection.second

            if (startDate != null && endDate != null) {
                currentDateRange = Pair(startDate, endDate + 24 * 60 * 60 * 1000L) // Include end day

                val startDateStr = dateFormat.format(Date(startDate))
                val endDateStr = dateFormat.format(Date(endDate))
                binding.dateRangeChip.text = "$startDateStr - $endDateStr"
                binding.dateRangeChip.isChecked = true

                applyFilters()
            }
        }

        dateRangePicker.show(supportFragmentManager, "DATE_RANGE_PICKER")
    }

    // ========== UI UPDATES ==========

    private fun updateSessionList(sessions: List<SessionSummaryDTO>) {
        if (sessions.isEmpty()) {
            showEmptyState()
        } else {
            hideEmptyState()
            sessionAdapter.updateSessions(sessions)
        }
    }

    private fun updateResultsInfo(count: Int) {
        binding.resultsInfoText.text = when {
            count == 0 -> "No sessions found"
            count == 1 -> "1 session found"
            else -> "$count sessions found"
        }
        binding.resultsInfoText.visibility = if (count > 0) View.VISIBLE else View.GONE
    }

    private fun updateStatistics(stats: HistoryStatistics?) {
        stats?.let { statistics ->
            binding.statisticsLayout.visibility = View.VISIBLE

            binding.totalSessionsText.text = "${statistics.totalSessions} sessions"
            binding.totalSleepTimeText.text = formatDuration(statistics.totalSleepTime)
            binding.averageQualityText.text = String.format("%.1f/10", statistics.averageQuality)
            binding.bestStreakText.text = "${statistics.longestStreak} days"
        } ?: run {
            binding.statisticsLayout.visibility = View.GONE
        }
    }

    private fun updateLoadingState(loading: Boolean) {
        binding.swipeRefreshLayout.isRefreshing = loading
        binding.loadingIndicator.visibility = if (loading && sessionAdapter.itemCount == 0) {
            View.VISIBLE
        } else {
            View.GONE
        }
    }

    private fun showEmptyState() {
        binding.emptyStateLayout.visibility = View.VISIBLE
        binding.sessionsRecyclerView.visibility = View.GONE

        val message = when {
            currentSearchQuery.isNotEmpty() -> "No sessions match your search"
            currentFilterType != FilterType.ALL -> "No sessions match your filters"
            currentDateRange != null -> "No sessions in selected date range"
            else -> "No sleep sessions recorded yet"
        }

        binding.emptyStateText.text = message

        // Setup start tracking button
        binding.startTrackingButton.setOnClickListener {
            finish() // Go back to main activity to start tracking
        }
    }

    private fun hideEmptyState() {
        binding.emptyStateLayout.visibility = View.GONE
        binding.sessionsRecyclerView.visibility = View.VISIBLE
    }

    private fun showError(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    // ========== SESSION INTERACTIONS ==========

    private fun showSessionDetails(session: SessionSummaryDTO) {
        // Create session detail dialog
        val detailsText = buildString {
            appendLine("📅 ${dateFormat.format(Date(session.startTime))}")
            appendLine("⏱️ Duration: ${session.formattedDuration}")
            appendLine("⭐ Quality: ${session.qualityGrade} (${String.format("%.1f/10", session.qualityScore ?: 0f)})")
            appendLine("📊 Efficiency: ${session.efficiencyGrade} (${String.format("%.1f%%", session.sleepEfficiency)})")
            appendLine("🏃 Movement Events: ${session.totalMovementEvents}")
            appendLine("🔊 Noise Events: ${session.totalNoiseEvents}")
            appendLine("📈 Avg Movement: ${String.format("%.1f", session.averageMovementIntensity)}")
            appendLine("🔉 Avg Noise: ${String.format("%.0f", session.averageNoiseLevel)}")
        }

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Session Details")
            .setMessage(detailsText)
            .setPositiveButton("View Charts") { _, _ ->
                viewSessionCharts(session)
            }
            .setNeutralButton("Share") { _, _ ->
                shareSession(session)
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showSessionContextMenu(session: SessionSummaryDTO) {
        val options = arrayOf(
            "View Details",
            "View in Charts",
            "Share Session",
            "Export Data",
            "Delete Session"
        )

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Session Options")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showSessionDetails(session)
                    1 -> viewSessionCharts(session)
                    2 -> shareSession(session)
                    3 -> exportSession(session)
                    4 -> confirmDeleteSession(session)
                }
            }
            .show()
    }

    private fun viewSessionCharts(session: SessionSummaryDTO) {
        val intent = Intent(this, ChartsActivity::class.java).apply {
            putExtra("session_id", session.id)
            putExtra("focus_session", true)
        }
        startActivity(intent)
    }

    private fun shareSession(session: SessionSummaryDTO) {
        val shareText = buildString {
            appendLine("🌙 Sleep Session Summary")
            appendLine()
            appendLine("📅 Date: ${dateFormat.format(Date(session.startTime))}")
            appendLine("⏱️ Duration: ${session.formattedDuration}")
            appendLine("⭐ Quality: ${session.qualityGrade}")
            appendLine("📊 Efficiency: ${session.efficiencyGrade}")
            appendLine("🏃 Movement Events: ${session.totalMovementEvents}")
            appendLine("🔊 Noise Events: ${session.totalNoiseEvents}")
            appendLine()
            appendLine("Tracked with SomniAI")
        }

        val shareIntent = Intent().apply {
            action = Intent.ACTION_SEND
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, shareText)
            putExtra(Intent.EXTRA_SUBJECT, "My Sleep Session")
        }

        startActivity(Intent.createChooser(shareIntent, "Share Sleep Session"))
    }

    private fun exportSession(session: SessionSummaryDTO) {
        historyViewModel.exportSession(session) { success, filePath ->
            if (success && filePath != null) {
                Toast.makeText(this, "Session exported to: $filePath", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Failed to export session", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun confirmDeleteSession(session: SessionSummaryDTO) {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Delete Session")
            .setMessage("Are you sure you want to delete this sleep session? This action cannot be undone.")
            .setPositiveButton("Delete") { _, _ ->
                deleteSession(session)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun deleteSession(session: SessionSummaryDTO) {
        historyViewModel.deleteSession(session.id) { success ->
            if (success) {
                Toast.makeText(this, "Session deleted", Toast.LENGTH_SHORT).show()
                refreshSessionHistory()
            } else {
                Toast.makeText(this, "Failed to delete session", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ========== MENU ACTIONS ==========

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.history_menu, menu)
        return true
    }

    private fun exportAllSessions() {
        historyViewModel.exportAllSessions { success, filePath ->
            if (success && filePath != null) {
                Toast.makeText(this, "All sessions exported to: $filePath", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "Failed to export sessions", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun confirmClearAllSessions() {
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Clear All Sessions")
            .setMessage("Are you sure you want to delete ALL sleep sessions? This action cannot be undone.")
            .setPositiveButton("Clear All") { _, _ ->
                clearAllSessions()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun clearAllSessions() {
        historyViewModel.clearAllSessions { success ->
            if (success) {
                Toast.makeText(this, "All sessions cleared", Toast.LENGTH_SHORT).show()
                refreshSessionHistory()
            } else {
                Toast.makeText(this, "Failed to clear sessions", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun showSortOptions() {
        val sortOptions = arrayOf(
            "Date (Newest First)",
            "Date (Oldest First)",
            "Duration (Longest First)",
            "Duration (Shortest First)",
            "Quality (Best First)",
            "Quality (Worst First)"
        )

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Sort Sessions")
            .setItems(sortOptions) { _, which ->
                val sortType = when (which) {
                    0 -> SortType.DATE_DESC
                    1 -> SortType.DATE_ASC
                    2 -> SortType.DURATION_DESC
                    3 -> SortType.DURATION_ASC
                    4 -> SortType.QUALITY_DESC
                    5 -> SortType.QUALITY_ASC
                    else -> SortType.DATE_DESC
                }
                historyViewModel.setSortType(sortType)
            }
            .show()
    }

    // ========== PAGINATION ==========

    private inner class PaginationScrollListener : androidx.recyclerview.widget.RecyclerView.OnScrollListener() {
        override fun onScrolled(recyclerView: androidx.recyclerview.widget.RecyclerView, dx: Int, dy: Int) {
            super.onScrolled(recyclerView, dx, dy)

            val layoutManager = recyclerView.layoutManager as LinearLayoutManager
            val visibleItemCount = layoutManager.childCount
            val totalItemCount = layoutManager.itemCount
            val firstVisibleItemPosition = layoutManager.findFirstVisibleItemPosition()

            if (!isLoading && (visibleItemCount + firstVisibleItemPosition) >= totalItemCount - 5) {
                loadMoreSessions()
            }
        }
    }

    // ========== UTILITY METHODS ==========

    private fun formatDuration(durationMs: Long): String {
        val hours = durationMs / (1000 * 60 * 60)
        val minutes = (durationMs % (1000 * 60 * 60)) / (1000 * 60)
        return "${hours}h ${minutes}m"
    }

    override fun onDestroy() {
        super.onDestroy()
        historyViewModel.cleanup()
    }
}

// ========== SUPPORTING ENUMS ==========

enum class FilterType {
    ALL,
    QUALITY_EXCELLENT,
    QUALITY_GOOD,
    QUALITY_POOR,
    DURATION_SHORT,
    DURATION_NORMAL,
    DURATION_LONG,
    RECENT_WEEK,
    RECENT_MONTH
}

enum class SortType {
    DATE_DESC,
    DATE_ASC,
    DURATION_DESC,
    DURATION_ASC,
    QUALITY_DESC,
    QUALITY_ASC
}

// ========== DATA CLASSES ==========

data class HistoryStatistics(
    val totalSessions: Int,
    val totalSleepTime: Long,
    val averageQuality: Float,
    val averageEfficiency: Float,
    val longestStreak: Int,
    val currentStreak: Int
)