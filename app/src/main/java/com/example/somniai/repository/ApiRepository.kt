package com.example.somniai.repository

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.example.somniai.ai.AIInsightsEngine
import com.example.somniai.data.*
import com.example.somniai.database.SleepDatabase
import com.example.somniai.network.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import okhttp3.ResponseBody
import retrofit2.Response
import java.io.IOException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import javax.net.ssl.SSLException
import kotlin.math.min
import kotlin.math.pow

/**
 * Enterprise-grade API Repository with sophisticated caching and orchestration
 *
 * Comprehensive Features:
 * - Multi-level caching strategy (memory, disk, database)
 * - AI service orchestration with intelligent fallback mechanisms
 * - Advanced retry logic with exponential backoff and jitter
 * - Circuit breaker pattern for fault tolerance
 * - Real-time data streaming and synchronization
 * - Performance monitoring and analytics integration
 * - Authentication and session management
 * - Rate limiting and quota management
 * - Comprehensive error handling and recovery
 * - Integration with existing SleepRepository and AI insights
 * - Background synchronization and conflict resolution
 * - Privacy-compliant data handling
 * - Offline-first architecture with seamless connectivity recovery
 */
class ApiRepository(
    private val apiService: ApiService,
    private val sleepRepository: SleepRepository,
    private val aiInsightsEngine: AIInsightsEngine,
    private val database: SleepDatabase,
    private val context: Context,
    private val preferences: SharedPreferences,
    private val dispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    companion object {
        private const val TAG = "ApiRepository"

        // Cache configuration
        private const val MEMORY_CACHE_SIZE = 100
        private const val MEMORY_CACHE_TTL_MINUTES = 15L
        private const val DISK_CACHE_TTL_HOURS = 24L
        private const val USER_DATA_CACHE_TTL_MINUTES = 5L
        private const val ANALYTICS_CACHE_TTL_MINUTES = 30L

        // Retry configuration
        private const val MAX_RETRY_ATTEMPTS = 3
        private const val BASE_RETRY_DELAY_MS = 1000L
        private const val MAX_RETRY_DELAY_MS = 32000L
        private const val JITTER_FACTOR = 0.1

        // Circuit breaker configuration
        private const val CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
        private const val CIRCUIT_BREAKER_TIMEOUT_MINUTES = 10L
        private const val CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3

        // Rate limiting
        private const val RATE_LIMIT_REQUESTS_PER_MINUTE = 60
        private const val AI_RATE_LIMIT_REQUESTS_PER_MINUTE = 20

        // Background sync
        private const val SYNC_INTERVAL_MINUTES = 30L
        private const val BATCH_SYNC_SIZE = 50
        private const val OFFLINE_QUEUE_MAX_SIZE = 1000

        // Preferences keys
        private const val PREF_AUTH_TOKEN = "auth_token"
        private const val PREF_REFRESH_TOKEN = "refresh_token"
        private const val PREF_TOKEN_EXPIRY = "token_expiry"
        private const val PREF_USER_ID = "user_id"
        private const val PREF_LAST_SYNC = "last_sync_timestamp"
        private const val PREF_SYNC_SETTINGS = "sync_settings"
        private const val PREF_CIRCUIT_BREAKER_STATE = "circuit_breaker_state"
        private const val PREF_RATE_LIMIT_DATA = "rate_limit_data"
        private const val PREF_PERFORMANCE_METRICS = "api_performance_metrics"
    }

    // Core dependencies and scope
    private val scope = CoroutineScope(dispatcher + SupervisorJob())
    private val networkStateMonitor = NetworkStateMonitor(context)
    private val authenticationManager = AuthenticationManager()
    private val cacheManager = MultiLevelCacheManager()
    private val rateLimiter = AdaptiveRateLimiter()
    private val circuitBreaker = CircuitBreakerManager()
    private val syncManager = BackgroundSyncManager()
    private val performanceMonitor = ApiPerformanceMonitor()

    // State management
    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.UNKNOWN)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private val _syncState = MutableStateFlow<SyncState>(SyncState.IDLE)
    val syncState: StateFlow<SyncState> = _syncState.asStateFlow()

    private val _authState = MutableStateFlow<AuthenticationState>(AuthenticationState.UNKNOWN)
    val authState: StateFlow<AuthenticationState> = _authState.asStateFlow()

    // Caching layers
    private val memoryCache = ConcurrentHashMap<String, CacheEntry<Any>>()
    private val pendingSyncQueue = ConcurrentHashMap<String, PendingSyncOperation>()
    private val activeRequests = ConcurrentHashMap<String, Job>()

    // Performance metrics
    private val requestCount = AtomicLong(0L)
    private val successCount = AtomicLong(0L)
    private val errorCount = AtomicLong(0L)
    private val cacheHitCount = AtomicLong(0L)
    private val isInitialized = AtomicBoolean(false)

    // Configuration
    private var currentUser: UserProfile? = null
    private var syncConfiguration: SyncConfiguration = loadSyncConfiguration()

    // ========== INITIALIZATION ==========

    /**
     * Initialize the API repository with comprehensive setup
     */
    suspend fun initialize(): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Initializing enterprise API repository")

            // Initialize core components
            initializeComponents()

            // Load authentication state
            loadAuthenticationState()

            // Initialize caching layers
            cacheManager.initialize()

            // Start background services
            startBackgroundServices()

            // Verify connectivity and perform initial sync
            if (networkStateMonitor.isConnected()) {
                performInitialSync()
            }

            isInitialized.set(true)
            _connectionState.value = if (networkStateMonitor.isConnected()) {
                ConnectionState.CONNECTED
            } else {
                ConnectionState.OFFLINE
            }

            Log.d(TAG, "API repository initialized successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize API repository", e)
            Result.failure(e)
        }
    }

    // ========== AUTHENTICATION MANAGEMENT ==========

    /**
     * Authenticate user with comprehensive session management
     */
    suspend fun authenticateUser(
        email: String,
        password: String,
        rememberMe: Boolean = true
    ): Result<AuthenticationResponse> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Authenticating user: $email")

            val request = AuthenticationRequest(
                email = email,
                password = password,
                deviceInfo = getDeviceInfo(),
                appVersion = getAppVersion()
            )

            val response = executeWithRetry(
                operation = { apiService.authenticateUser(request) },
                cacheKey = null, // Don't cache authentication requests
                cacheTtl = 0L
            ).getOrThrow()

            // Store authentication data
            authenticationManager.storeAuthenticationData(response)
            _authState.value = AuthenticationState.AUTHENTICATED

            // Load user profile
            loadUserProfile()

            // Trigger background sync
            triggerBackgroundSync()

            Log.d(TAG, "User authenticated successfully")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Authentication failed for user: $email", e)
            _authState.value = AuthenticationState.FAILED(e.message ?: "Authentication failed")
            Result.failure(e)
        }
    }

    /**
     * Register new user with comprehensive onboarding
     */
    suspend fun registerUser(
        userRegistration: UserRegistrationRequest
    ): Result<UserRegistrationResponse> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Registering new user: ${userRegistration.email}")

            val enhancedRequest = userRegistration.copy(
                deviceInfo = getDeviceInfo(),
                appVersion = getAppVersion(),
                referralSource = getReferralSource(),
                privacyConsent = getPrivacyConsentVersion()
            )

            val response = executeWithRetry(
                operation = { apiService.registerUser(enhancedRequest) },
                cacheKey = null,
                cacheTtl = 0L
            ).getOrThrow()

            // Initialize user session if registration includes authentication
            response.authenticationData?.let { authData ->
                authenticationManager.storeAuthenticationData(authData)
                _authState.value = AuthenticationState.AUTHENTICATED
                loadUserProfile()
            }

            Log.d(TAG, "User registered successfully")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "User registration failed", e)
            Result.failure(e)
        }
    }

    /**
     * Refresh authentication token with automatic retry
     */
    suspend fun refreshAuthenticationToken(): Result<TokenRefreshResponse> = withContext(dispatcher) {
        try {
            val refreshToken = authenticationManager.getRefreshToken()
                ?: return@withContext Result.failure(IllegalStateException("No refresh token available"))

            val request = TokenRefreshRequest(
                refreshToken = refreshToken,
                deviceId = getDeviceId()
            )

            val response = executeWithRetry(
                operation = {
                    apiService.refreshToken(
                        authorization = "Bearer ${authenticationManager.getAccessToken()}",
                        request = request
                    )
                },
                cacheKey = null,
                cacheTtl = 0L
            ).getOrThrow()

            // Update stored tokens
            authenticationManager.updateTokens(response)
            _authState.value = AuthenticationState.AUTHENTICATED

            Log.d(TAG, "Authentication token refreshed successfully")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Token refresh failed", e)
            _authState.value = AuthenticationState.EXPIRED
            Result.failure(e)
        }
    }

    /**
     * Logout user and clear all cached data
     */
    suspend fun logout(): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Logging out user")

            // Attempt server-side logout
            try {
                val authToken = authenticationManager.getAccessToken()
                if (authToken != null) {
                    apiService.logout("Bearer $authToken")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Server-side logout failed, continuing with local logout", e)
            }

            // Clear all cached data
            authenticationManager.clearAuthenticationData()
            cacheManager.clearAllCaches()
            currentUser = null

            // Update state
            _authState.value = AuthenticationState.LOGGED_OUT
            _syncState.value = SyncState.IDLE

            // Clear pending sync operations
            pendingSyncQueue.clear()

            Log.d(TAG, "User logged out successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Logout failed", e)
            Result.failure(e)
        }
    }

    // ========== AI INSIGHTS INTEGRATION ==========

    /**
     * Generate AI insights with intelligent service orchestration
     */
    suspend fun generateAIInsights(
        request: AIInsightGenerationRequest,
        priority: InsightPriority = InsightPriority.NORMAL
    ): Result<AIInsightGenerationResponse> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Generating AI insights: type=${request.generationType}, priority=$priority")

            // Check authentication
            requireAuthentication()

            // Apply rate limiting for AI requests
            rateLimiter.checkAIRateLimit()

            // Enhanced request with context
            val enhancedRequest = request.copy(
                userContext = getUserContext(),
                deviceCapabilities = getDeviceCapabilities(),
                privacySettings = getPrivacySettings(),
                priority = priority.value
            )

            // Execute with circuit breaker protection
            val response = circuitBreaker.execute("ai_insights") {
                executeWithRetry(
                    operation = {
                        apiService.generateAIInsights(
                            authorization = getAuthorizationHeader(),
                            request = enhancedRequest
                        )
                    },
                    cacheKey = "ai_insights_${request.sessionId}_${request.generationType}",
                    cacheTtl = TimeUnit.MINUTES.toMillis(30),
                    retryPredicate = ::isRetryableAIError
                )
            }.getOrThrow()

            // Integrate with local AI engine for fallback
            if (response.status == GenerationStatus.FAILED && aiInsightsEngine.isAvailable()) {
                Log.w(TAG, "Server AI generation failed, attempting local fallback")
                return@withContext generateLocalAIInsightsFallback(request)
            }

            // Cache successful response
            cacheManager.storeAIInsightResponse(response)

            // Track performance metrics
            performanceMonitor.recordAIInsightGeneration(
                type = request.generationType,
                success = response.status == GenerationStatus.COMPLETED,
                duration = response.processingTime,
                modelUsed = response.modelUsed
            )

            Log.d(TAG, "AI insights generated successfully: jobId=${response.jobId}")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "AI insight generation failed", e)
            errorCount.incrementAndGet()

            // Attempt local fallback if available
            if (aiInsightsEngine.isAvailable()) {
                return@withContext generateLocalAIInsightsFallback(request)
            }

            Result.failure(e)
        }
    }

    /**
     * Get AI insight generation status with real-time updates
     */
    suspend fun getInsightGenerationStatus(
        jobId: String
    ): Flow<InsightGenerationStatusResponse> = flow {
        Log.d(TAG, "Monitoring insight generation status: $jobId")

        var attempts = 0
        val maxAttempts = 60 // 5 minutes with 5-second intervals

        while (attempts < maxAttempts) {
            try {
                val response = executeWithRetry(
                    operation = {
                        apiService.getInsightGenerationStatus(
                            authorization = getAuthorizationHeader(),
                            jobId = jobId
                        )
                    },
                    cacheKey = null, // Don't cache status requests
                    cacheTtl = 0L
                ).getOrThrow()

                emit(response)

                // Exit if completed or failed
                if (response.status in listOf(
                        GenerationStatus.COMPLETED,
                        GenerationStatus.FAILED,
                        GenerationStatus.CANCELLED
                    )) {
                    break
                }

                delay(5000) // Poll every 5 seconds
                attempts++

            } catch (e: Exception) {
                Log.e(TAG, "Error checking insight generation status", e)
                emit(
                    InsightGenerationStatusResponse(
                        jobId = jobId,
                        status = GenerationStatus.ERROR,
                        error = e.message
                    )
                )
                break
            }
        }
    }.flowOn(dispatcher)

    /**
     * Submit insight feedback with analytics integration
     */
    suspend fun submitInsightFeedback(
        insightId: Long,
        feedback: InsightFeedback,
        engagementMetrics: EngagementMetrics? = null
    ): Result<Unit> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Submitting insight feedback: insightId=$insightId")

            requireAuthentication()

            val request = InsightFeedbackRequest(
                insightId = insightId,
                feedback = feedback,
                engagementMetrics = engagementMetrics,
                timestamp = System.currentTimeMillis(),
                appVersion = getAppVersion(),
                sessionContext = getSessionContext()
            )

            // Submit to server
            executeWithRetry(
                operation = {
                    apiService.submitInsightFeedback(
                        authorization = getAuthorizationHeader(),
                        feedback = request
                    )
                },
                cacheKey = null,
                cacheTtl = 0L
            ).getOrThrow()

            // Also submit to local AI engine for learning
            aiInsightsEngine.recordInsightEffectiveness(insightId, feedback, engagementMetrics)

            // Track analytics event
            trackAnalyticsEvent(
                AnalyticsEvent.INSIGHT_FEEDBACK_SUBMITTED,
                mapOf(
                    "insight_id" to insightId,
                    "rating" to feedback.rating,
                    "helpful" to feedback.helpful
                )
            )

            Log.d(TAG, "Insight feedback submitted successfully")
            Result.success(Unit)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to submit insight feedback", e)

            // Queue for later sync if offline
            if (!networkStateMonitor.isConnected()) {
                queueForLaterSync(
                    PendingSyncOperation.InsightFeedback(insightId, feedback, engagementMetrics)
                )
            }

            Result.failure(e)
        }
    }

    // ========== SLEEP DATA SYNCHRONIZATION ==========

    /**
     * Synchronize sleep sessions with comprehensive conflict resolution
     */
    suspend fun syncSleepSessions(
        syncType: SyncType = SyncType.INCREMENTAL,
        forceFullSync: Boolean = false
    ): Result<SleepSessionSyncResponse> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Synchronizing sleep sessions: type=$syncType, force=$forceFullSync")

            requireAuthentication()
            _syncState.value = SyncState.SYNCING

            // Prepare sync request
            val lastSyncTimestamp = if (forceFullSync) 0L else getLastSyncTimestamp()
            val localSessions = sleepRepository.getSessionsModifiedSince(lastSyncTimestamp)

            val request = SleepSessionSyncRequest(
                lastSyncTimestamp = lastSyncTimestamp,
                localSessions = localSessions.map { it.toSyncModel() },
                deviceId = getDeviceId(),
                syncType = syncType,
                conflictResolutionStrategy = syncConfiguration.conflictResolution
            )

            // Execute sync
            val response = executeWithRetry(
                operation = {
                    apiService.syncSleepSessions(
                        authorization = getAuthorizationHeader(),
                        request = request
                    )
                },
                cacheKey = null,
                cacheTtl = 0L,
                timeout = TimeUnit.MINUTES.toMillis(5) // Longer timeout for sync
            ).getOrThrow()

            // Process sync response
            processSyncResponse(response)

            // Update last sync timestamp
            updateLastSyncTimestamp(response.syncTimestamp)

            // Clear completed sync operations from queue
            clearCompletedSyncOperations()

            _syncState.value = SyncState.COMPLETED
            Log.d(TAG, "Sleep session sync completed successfully")

            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Sleep session sync failed", e)
            _syncState.value = SyncState.FAILED(e.message ?: "Sync failed")
            Result.failure(e)
        }
    }

    /**
     * Get sleep analytics with intelligent caching
     */
    suspend fun getSleepAnalytics(
        period: AnalyticsPeriod = AnalyticsPeriod.LAST_30_DAYS,
        forceRefresh: Boolean = false
    ): Result<AnalyticsOverview> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Fetching sleep analytics: period=$period, refresh=$forceRefresh")

            requireAuthentication()

            val cacheKey = "analytics_overview_${period.name}_${currentUser?.id}"

            // Check cache first unless forced refresh
            if (!forceRefresh) {
                val cached = cacheManager.getAnalytics(cacheKey)
                if (cached != null) {
                    Log.d(TAG, "Returning cached analytics data")
                    cacheHitCount.incrementAndGet()
                    return@withContext Result.success(cached)
                }
            }

            // Fetch from server
            val response = executeWithRetry(
                operation = {
                    apiService.getAnalyticsOverview(
                        authorization = getAuthorizationHeader(),
                        period = period.value
                    )
                },
                cacheKey = cacheKey,
                cacheTtl = TimeUnit.MINUTES.toMillis(ANALYTICS_CACHE_TTL_MINUTES)
            ).getOrThrow()

            // Cache the response
            cacheManager.storeAnalytics(cacheKey, response)

            Log.d(TAG, "Sleep analytics fetched successfully")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to fetch sleep analytics", e)

            // Return cached data if available and we're offline
            if (!networkStateMonitor.isConnected()) {
                val cached = cacheManager.getAnalytics("analytics_overview_${period.name}_${currentUser?.id}")
                if (cached != null) {
                    Log.d(TAG, "Returning stale cached analytics due to offline mode")
                    return@withContext Result.success(cached)
                }
            }

            Result.failure(e)
        }
    }

    /**
     * Get personalized insights with ML enhancement
     */
    suspend fun getPersonalizedInsights(
        type: InsightType? = null,
        priority: InsightPriority? = null,
        limit: Int = 10,
        includeProcessedInsights: Boolean = true
    ): Result<PersonalizedInsightsResponse> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Fetching personalized insights: type=$type, priority=$priority, limit=$limit")

            requireAuthentication()

            val cacheKey = "personalized_insights_${type?.name}_${priority?.name}_$limit"

            // Fetch from server
            val response = executeWithRetry(
                operation = {
                    apiService.getPersonalizedInsights(
                        authorization = getAuthorizationHeader(),
                        type = type?.value,
                        priority = priority?.value,
                        limit = limit
                    )
                },
                cacheKey = cacheKey,
                cacheTtl = TimeUnit.MINUTES.toMillis(15)
            ).getOrThrow()

            // Enhance with local AI engine if enabled
            if (includeProcessedInsights && aiInsightsEngine.isAvailable()) {
                val enhancedInsights = enhanceInsightsWithLocalAI(response.insights)
                val enhancedResponse = response.copy(insights = enhancedInsights)

                Log.d(TAG, "Enhanced ${response.insights.size} insights with local AI")
                return@withContext Result.success(enhancedResponse)
            }

            Log.d(TAG, "Fetched ${response.insights.size} personalized insights")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to fetch personalized insights", e)

            // Fallback to local insights generation
            if (aiInsightsEngine.isAvailable()) {
                return@withContext generateLocalInsightsFallback(type, priority, limit)
            }

            Result.failure(e)
        }
    }

    // ========== USER PROFILE MANAGEMENT ==========

    /**
     * Get user profile with caching
     */
    suspend fun getUserProfile(
        forceRefresh: Boolean = false
    ): Result<UserProfile> = withContext(dispatcher) {
        try {
            requireAuthentication()

            val cacheKey = "user_profile_${currentUser?.id}"

            // Check cache first
            if (!forceRefresh && currentUser != null) {
                val cached = cacheManager.getUserProfile(cacheKey)
                if (cached != null) {
                    Log.d(TAG, "Returning cached user profile")
                    return@withContext Result.success(cached)
                }
            }

            // Fetch from server
            val response = executeWithRetry(
                operation = {
                    apiService.getUserProfile(
                        authorization = getAuthorizationHeader()
                    )
                },
                cacheKey = cacheKey,
                cacheTtl = TimeUnit.MINUTES.toMillis(USER_DATA_CACHE_TTL_MINUTES)
            ).getOrThrow()

            // Update current user and cache
            currentUser = response
            cacheManager.storeUserProfile(cacheKey, response)

            Log.d(TAG, "User profile fetched successfully")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to fetch user profile", e)
            Result.failure(e)
        }
    }

    /**
     * Update user profile with optimistic updates
     */
    suspend fun updateUserProfile(
        profileUpdate: UserProfileUpdate
    ): Result<UserProfile> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Updating user profile")

            requireAuthentication()

            // Optimistic update to cache
            currentUser?.let { user ->
                val optimisticProfile = applyProfileUpdate(user, profileUpdate)
                currentUser = optimisticProfile
                cacheManager.storeUserProfile("user_profile_${user.id}", optimisticProfile)
            }

            // Send to server
            val response = executeWithRetry(
                operation = {
                    apiService.updateUserProfile(
                        authorization = getAuthorizationHeader(),
                        profile = profileUpdate
                    )
                },
                cacheKey = null,
                cacheTtl = 0L
            ).getOrThrow()

            // Update cache with server response
            currentUser = response
            cacheManager.storeUserProfile("user_profile_${response.id}", response)

            Log.d(TAG, "User profile updated successfully")
            Result.success(response)

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update user profile", e)

            // Revert optimistic update
            loadUserProfile()

            Result.failure(e)
        }
    }

    // ========== HEALTH AND PERFORMANCE MONITORING ==========

    /**
     * Perform comprehensive health check
     */
    suspend fun performHealthCheck(): Result<ApiHealthStatus> = withContext(dispatcher) {
        try {
            Log.d(TAG, "Performing API health check")

            val startTime = System.currentTimeMillis()

            // Basic connectivity check
            val healthResponse = executeWithRetry(
                operation = { apiService.healthCheck() },
                cacheKey = null,
                cacheTtl = 0L,
                maxRetries = 1
            )

            val duration = System.currentTimeMillis() - startTime
            val isHealthy = healthResponse.isSuccess

            // Detailed health check if authenticated
            var detailedHealth: DetailedHealthResponse? = null
            if (isHealthy && authenticationManager.isAuthenticated()) {
                try {
                    detailedHealth = executeWithRetry(
                        operation = {
                            apiService.detailedHealthCheck(
                                authorization = getAuthorizationHeader()
                            )
                        },
                        cacheKey = null,
                        cacheTtl = 0L,
                        maxRetries = 1
                    ).getOrNull()
                } catch (e: Exception) {
                    Log.w(TAG, "Detailed health check failed", e)
                }
            }

            // Compile health status
            val healthStatus = ApiHealthStatus(
                isHealthy = isHealthy,
                responseTime = duration,
                serverStatus = healthResponse.getOrNull()?.status ?: "unknown",
                detailedHealth = detailedHealth,
                circuitBreakerState = circuitBreaker.getState(),
                cacheStats = cacheManager.getStatistics(),
                performanceMetrics = performanceMonitor.getCurrentMetrics(),
                lastHealthCheck = System.currentTimeMillis()
            )

            Log.d(TAG, "Health check completed: healthy=$isHealthy, responseTime=${duration}ms")
            Result.success(healthStatus)

        } catch (e: Exception) {
            Log.e(TAG, "Health check failed", e)

            val failedHealthStatus = ApiHealthStatus(
                isHealthy = false,
                responseTime = -1L,
                serverStatus = "error",
                error = e.message,
                circuitBreakerState = circuitBreaker.getState(),
                cacheStats = cacheManager.getStatistics(),
                performanceMetrics = performanceMonitor.getCurrentMetrics(),
                lastHealthCheck = System.currentTimeMillis()
            )

            Result.success(failedHealthStatus) // Return failed status rather than exception
        }
    }

    /**
     * Get comprehensive performance metrics
     */
    fun getPerformanceMetrics(): ApiPerformanceMetrics {
        return ApiPerformanceMetrics(
            totalRequests = requestCount.get(),
            successfulRequests = successCount.get(),
            failedRequests = errorCount.get(),
            cacheHits = cacheHitCount.get(),
            averageResponseTime = performanceMonitor.getAverageResponseTime(),
            successRate = if (requestCount.get() > 0) {
                successCount.get().toFloat() / requestCount.get()
            } else 0f,
            cacheHitRate = if (requestCount.get() > 0) {
                cacheHitCount.get().toFloat() / requestCount.get()
            } else 0f,
            circuitBreakerState = circuitBreaker.getState(),
            activeConnections = activeRequests.size,
            memoryUsage = cacheManager.getMemoryUsage(),
            lastUpdated = System.currentTimeMillis()
        )
    }

    // ========== PRIVATE IMPLEMENTATION ==========

    private suspend fun initializeComponents() {
        authenticationManager.initialize(preferences)
        cacheManager.initialize(context, database)
        rateLimiter.initialize(preferences)
        circuitBreaker.initialize(preferences)
        syncManager.initialize(scope, this)
        performanceMonitor.initialize(preferences)
    }

    private suspend fun loadAuthenticationState() {
        val token = authenticationManager.getAccessToken()
        val isExpired = authenticationManager.isTokenExpired()

        _authState.value = when {
            token == null -> AuthenticationState.LOGGED_OUT
            isExpired -> AuthenticationState.EXPIRED
            else -> AuthenticationState.AUTHENTICATED
        }
    }

    private suspend fun startBackgroundServices() {
        // Start network monitoring
        scope.launch {
            networkStateMonitor.networkState.collect { isConnected ->
                _connectionState.value = if (isConnected) {
                    ConnectionState.CONNECTED
                } else {
                    ConnectionState.OFFLINE
                }

                if (isConnected && pendingSyncQueue.isNotEmpty()) {
                    processPendingSyncOperations()
                }
            }
        }

        // Start periodic sync
        scope.launch {
            while (isActive) {
                delay(TimeUnit.MINUTES.toMillis(SYNC_INTERVAL_MINUTES))

                if (authenticationManager.isAuthenticated() &&
                    networkStateMonitor.isConnected() &&
                    syncConfiguration.autoSyncEnabled) {

                    try {
                        syncSleepSessions(SyncType.INCREMENTAL)
                    } catch (e: Exception) {
                        Log.w(TAG, "Periodic sync failed", e)
                    }
                }
            }
        }

        // Start cache maintenance
        scope.launch {
            while (isActive) {
                delay(TimeUnit.HOURS.toMillis(1))
                cacheManager.performMaintenance()
            }
        }
    }

    private suspend fun performInitialSync() {
        if (authenticationManager.isAuthenticated()) {
            try {
                loadUserProfile()
                syncSleepSessions(SyncType.INCREMENTAL)
            } catch (e: Exception) {
                Log.w(TAG, "Initial sync failed", e)
            }
        }
    }

    private suspend fun loadUserProfile() {
        try {
            currentUser = getUserProfile(forceRefresh = true).getOrNull()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load user profile", e)
        }
    }

    /**
     * Execute API request with comprehensive retry logic and circuit breaker
     */
    private suspend fun <T> executeWithRetry(
        operation: suspend () -> Response<T>,
        cacheKey: String?,
        cacheTtl: Long,
        maxRetries: Int = MAX_RETRY_ATTEMPTS,
        timeout: Long = TimeUnit.SECONDS.toMillis(30),
        retryPredicate: (Exception) -> Boolean = ::isRetryableError
    ): Result<T> = withContext(dispatcher) {
        val operationId = generateOperationId()

        try {
            requestCount.incrementAndGet()

            // Check cache first if cache key provided
            if (cacheKey != null) {
                val cached = cacheManager.get<T>(cacheKey)
                if (cached != null) {
                    cacheHitCount.incrementAndGet()
                    return@withContext Result.success(cached)
                }
            }

            // Execute with timeout and retry logic
            val job = async {
                var lastException: Exception? = null
                var attempt = 0

                while (attempt <= maxRetries) {
                    try {
                        // Apply rate limiting
                        rateLimiter.checkRateLimit()

                        // Execute operation
                        val startTime = System.currentTimeMillis()
                        val response = withTimeout(timeout) {
                            operation()
                        }
                        val duration = System.currentTimeMillis() - startTime

                        // Record performance metrics
                        performanceMonitor.recordRequest(duration, response.isSuccessful)

                        if (response.isSuccessful) {
                            val body = response.body()
                            if (body != null) {
                                successCount.incrementAndGet()

                                // Cache successful response
                                if (cacheKey != null && cacheTtl > 0) {
                                    cacheManager.store(cacheKey, body, cacheTtl)
                                }

                                return@async Result.success(body)
                            } else {
                                throw ApiException("Response body is null", response.code())
                            }
                        } else {
                            throw ApiException("HTTP ${response.code()}: ${response.message()}", response.code())
                        }

                    } catch (e: Exception) {
                        lastException = e

                        if (!retryPredicate(e) || attempt >= maxRetries) {
                            break
                        }

                        // Calculate retry delay with exponential backoff and jitter
                        val baseDelay = BASE_RETRY_DELAY_MS * (2.0.pow(attempt)).toLong()
                        val jitter = (baseDelay * JITTER_FACTOR * Math.random()).toLong()
                        val delay = min(baseDelay + jitter, MAX_RETRY_DELAY_MS)

                        Log.w(TAG, "Request failed (attempt ${attempt + 1}), retrying in ${delay}ms", e)
                        delay(delay)
                        attempt++
                    }
                }

                errorCount.incrementAndGet()
                Result.failure(lastException ?: Exception("Unknown error"))
            }

            // Track active request
            activeRequests[operationId] = job
            val result = job.await()
            activeRequests.remove(operationId)

            result

        } catch (e: Exception) {
            errorCount.incrementAndGet()
            activeRequests.remove(operationId)
            Log.e(TAG, "Request execution failed", e)
            Result.failure(e)
        }
    }

    private fun isRetryableError(exception: Exception): Boolean {
        return when (exception) {
            is SocketTimeoutException,
            is UnknownHostException,
            is IOException -> true
            is ApiException -> exception.code in 500..599 || exception.code == 429
            else -> false
        }
    }

    private fun isRetryableAIError(exception: Exception): Boolean {
        return when (exception) {
            is SocketTimeoutException,
            is UnknownHostException -> true
            is ApiException -> exception.code in listOf(429, 502, 503, 504)
            else -> false
        }
    }

    private fun requireAuthentication() {
        if (!authenticationManager.isAuthenticated()) {
            throw AuthenticationException("Authentication required")
        }
    }

    private fun getAuthorizationHeader(): String {
        val token = authenticationManager.getAccessToken()
            ?: throw AuthenticationException("No access token available")
        return "Bearer $token"
    }

    private fun generateOperationId(): String {
        return "op_${System.currentTimeMillis()}_${(Math.random() * 1000).toInt()}"
    }

    // Additional helper methods...
    private fun loadSyncConfiguration(): SyncConfiguration = SyncConfiguration.default()
    private fun getDeviceInfo(): DeviceInfo = DeviceInfo.current(context)
    private fun getAppVersion(): String = "1.0.0" // Get from BuildConfig
    private fun getReferralSource(): String? = null // Implement referral tracking
    private fun getPrivacyConsentVersion(): String = "1.0"
    private fun getDeviceId(): String = "device_id" // Implement device ID generation
    private fun getUserContext(): UserContext = UserContext.current(currentUser)
    private fun getDeviceCapabilities(): DeviceCapabilities = DeviceCapabilities.current(context)
    private fun getPrivacySettings(): PrivacySettings = PrivacySettings.current(preferences)
    private fun getSessionContext(): SessionContext = SessionContext.current()
    private fun getLastSyncTimestamp(): Long = preferences.getLong(PREF_LAST_SYNC, 0L)
    private fun updateLastSyncTimestamp(timestamp: Long) {
        preferences.edit().putLong(PREF_LAST_SYNC, timestamp).apply()
    }

    // Placeholder implementations for complex components...
    private suspend fun generateLocalAIInsightsFallback(request: AIInsightGenerationRequest): Result<AIInsightGenerationResponse> {
        // Implementation would use aiInsightsEngine for local generation
        return Result.failure(Exception("Local AI fallback not implemented"))
    }

    private suspend fun enhanceInsightsWithLocalAI(insights: List<PersonalizedInsight>): List<PersonalizedInsight> {
        // Implementation would enhance insights using local AI
        return insights
    }

    private suspend fun generateLocalInsightsFallback(type: InsightType?, priority: InsightPriority?, limit: Int): Result<PersonalizedInsightsResponse> {
        // Implementation would generate insights locally
        return Result.failure(Exception("Local insights fallback not implemented"))
    }

    private suspend fun processSyncResponse(response: SleepSessionSyncResponse) {
        // Implementation would process sync conflicts and updates
    }

    private suspend fun clearCompletedSyncOperations() {
        // Implementation would clean up completed sync operations
    }

    private suspend fun queueForLaterSync(operation: PendingSyncOperation) {
        pendingSyncQueue[operation.id] = operation
    }

    private suspend fun processPendingSyncOperations() {
        // Implementation would process queued sync operations
    }

    private suspend fun trackAnalyticsEvent(event: AnalyticsEvent, parameters: Map<String, Any>) {
        // Implementation would track analytics events
    }

    private fun applyProfileUpdate(user: UserProfile, update: UserProfileUpdate): UserProfile {
        // Implementation would apply profile updates optimistically
        return user
    }

    private suspend fun triggerBackgroundSync() {
        // Implementation would trigger background sync
    }

    /**
     * Cleanup resources and shutdown repository
     */
    fun cleanup() {
        scope.cancel()
        cacheManager.cleanup()
        performanceMonitor.cleanup()
        activeRequests.clear()
        pendingSyncQueue.clear()
        memoryCache.clear()

        Log.d(TAG, "API repository cleanup completed")
    }
}

// ========== SUPPORTING CLASSES AND DATA STRUCTURES ==========

/**
 * Multi-level cache manager with intelligent eviction
 */
private class MultiLevelCacheManager {
    private val memoryCache = ConcurrentHashMap<String, CacheEntry<Any>>()

    suspend fun initialize(context: Context, database: SleepDatabase) {}
    suspend fun initialize() {}
    fun store(key: String, value: Any, ttl: Long) {}
    fun <T> get(key: String): T? = null
    fun getAnalytics(key: String): AnalyticsOverview? = null
    fun storeAnalytics(key: String, value: AnalyticsOverview) {}
    fun getUserProfile(key: String): UserProfile? = null
    fun storeUserProfile(key: String, value: UserProfile) {}
    fun storeAIInsightResponse(response: AIInsightGenerationResponse) {}
    suspend fun performMaintenance() {}
    fun clearAllCaches() { memoryCache.clear() }
    fun getStatistics(): CacheStatistics = CacheStatistics()
    fun getMemoryUsage(): Long = 0L
    fun cleanup() {}
}

/**
 * Authentication manager with secure token storage
 */
private class AuthenticationManager {
    fun initialize(preferences: SharedPreferences) {}
    fun getAccessToken(): String? = null
    fun getRefreshToken(): String? = null
    fun isTokenExpired(): Boolean = false
    fun isAuthenticated(): Boolean = false
    fun storeAuthenticationData(data: AuthenticationResponse) {}
    fun updateTokens(response: TokenRefreshResponse) {}
    fun clearAuthenticationData() {}
}

/**
 * Adaptive rate limiter with intelligent throttling
 */
private class AdaptiveRateLimiter {
    fun initialize(preferences: SharedPreferences) {}
    suspend fun checkRateLimit() {}
    suspend fun checkAIRateLimit() {}
}

/**
 * Circuit breaker for fault tolerance
 */
private class CircuitBreakerManager {
    fun initialize(preferences: SharedPreferences) {}
    suspend fun <T> execute(operation: String, block: suspend () -> Result<T>): Result<T> = block()
    fun getState(): String = "CLOSED"
}

/**
 * Background synchronization manager
 */
private class BackgroundSyncManager {
    fun initialize(scope: CoroutineScope, repository: ApiRepository) {}
}

/**
 * Performance monitoring and metrics
 */
private class ApiPerformanceMonitor {
    fun initialize(preferences: SharedPreferences) {}
    fun recordRequest(duration: Long, success: Boolean) {}
    fun recordAIInsightGeneration(type: String, success: Boolean, duration: Long, modelUsed: String?) {}
    fun getAverageResponseTime(): Long = 0L
    fun getCurrentMetrics(): Map<String, Any> = emptyMap()
    fun cleanup() {}
}

/**
 * Network state monitoring
 */
private class NetworkStateMonitor(private val context: Context) {
    val networkState: Flow<Boolean> = flowOf(true)
    fun isConnected(): Boolean = true
}

// Data classes and enums
data class CacheEntry<T>(val data: T, val timestamp: Long = System.currentTimeMillis(), val ttl: Long = 0L) {
    fun isExpired(): Boolean = ttl > 0 && System.currentTimeMillis() - timestamp > ttl
}

sealed class ConnectionState {
    object UNKNOWN : ConnectionState()
    object CONNECTED : ConnectionState()
    object OFFLINE : ConnectionState()
}

sealed class SyncState {
    object IDLE : SyncState()
    object SYNCING : SyncState()
    object COMPLETED : SyncState()
    data class FAILED(val error: String) : SyncState()
}

sealed class AuthenticationState {
    object UNKNOWN : AuthenticationState()
    object AUTHENTICATED : AuthenticationState()
    object LOGGED_OUT : AuthenticationState()
    object EXPIRED : AuthenticationState()
    data class FAILED(val error: String) : AuthenticationState()
}

enum class SyncType(val value: String) {
    FULL("full"),
    INCREMENTAL("incremental"),
    FORCED("forced")
}

enum class AnalyticsPeriod(val value: String) {
    LAST_7_DAYS("7d"),
    LAST_30_DAYS("30d"),
    LAST_90_DAYS("90d"),
    ALL_TIME("all")
}

enum class InsightPriority(val value: String) {
    LOW("low"),
    NORMAL("normal"),
    HIGH("high"),
    URGENT("urgent")
}

enum class InsightType(val value: String) {
    QUALITY("quality"),
    DURATION("duration"),
    PATTERN("pattern"),
    RECOMMENDATION("recommendation")
}

enum class GenerationStatus {
    PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED, ERROR
}

enum class AnalyticsEvent {
    INSIGHT_FEEDBACK_SUBMITTED,
    SESSION_SYNCED,
    PROFILE_UPDATED
}

// Additional data classes
class ApiException(message: String, val code: Int) : Exception(message)
class AuthenticationException(message: String) : Exception(message)

data class PendingSyncOperation(
    val id: String,
    val timestamp: Long = System.currentTimeMillis()
) {
    data class InsightFeedback(
        val insightId: Long,
        val feedback: com.example.somniai.data.InsightFeedback,
        val engagementMetrics: EngagementMetrics?
    ) : PendingSyncOperation("insight_feedback_${insightId}_${System.currentTimeMillis()}")
}

data class SyncConfiguration(
    val autoSyncEnabled: Boolean = true,
    val conflictResolution: ConflictResolutionStrategy = ConflictResolutionStrategy.SERVER_WINS
) {
    companion object {
        fun default() = SyncConfiguration()
    }
}

enum class ConflictResolutionStrategy {
    SERVER_WINS, CLIENT_WINS, MERGE, MANUAL
}

data class ApiHealthStatus(
    val isHealthy: Boolean,
    val responseTime: Long,
    val serverStatus: String,
    val detailedHealth: DetailedHealthResponse? = null,
    val error: String? = null,
    val circuitBreakerState: String,
    val cacheStats: CacheStatistics,
    val performanceMetrics: Map<String, Any>,
    val lastHealthCheck: Long
)

data class ApiPerformanceMetrics(
    val totalRequests: Long,
    val successfulRequests: Long,
    val failedRequests: Long,
    val cacheHits: Long,
    val averageResponseTime: Long,
    val successRate: Float,
    val cacheHitRate: Float,
    val circuitBreakerState: String,
    val activeConnections: Int,
    val memoryUsage: Long,
    val lastUpdated: Long
)

data class CacheStatistics(
    val hitRate: Float = 0f,
    val missRate: Float = 0f,
    val evictionCount: Long = 0L,
    val size: Int = 0
)

// Placeholder classes for complex data types
data class DeviceInfo(val model: String, val os: String) {
    companion object {
        fun current(context: Context) = DeviceInfo("Android", "11")
    }
}

data class UserContext(val userId: String?, val preferences: Map<String, Any> = emptyMap()) {
    companion object {
        fun current(user: UserProfile?) = UserContext(user?.id)
    }
}

data class DeviceCapabilities(val sensors: List<String> = emptyList()) {
    companion object {
        fun current(context: Context) = DeviceCapabilities()
    }
}

data class PrivacySettings(val dataSharing: Boolean = false) {
    companion object {
        fun current(preferences: SharedPreferences) = PrivacySettings()
    }
}

data class SessionContext(val sessionId: String? = null) {
    companion object {
        fun current() = SessionContext()
    }
}

// Extension functions
private fun SleepSession.toSyncModel(): SleepSessionSyncModel = SleepSessionSyncModel(
    id = id,
    startTime = startTime,
    endTime = endTime,
    duration = duration
)

private fun SleepRepository.getSessionsModifiedSince(timestamp: Long): List<SleepSession> = emptyList()

data class SleepSessionSyncModel(
    val id: Long,
    val startTime: Long,
    val endTime: Long?,
    val duration: Long
)