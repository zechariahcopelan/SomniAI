package com.example.somniai.network

import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*
import java.util.concurrent.TimeUnit

/**
 * Enterprise-grade API service interface for SomniAI
 *
 * Comprehensive Retrofit interface supporting:
 * - Multiple AI service integrations (OpenAI, Anthropic, Google)
 * - Sleep data synchronization and backup
 * - Advanced analytics and insights endpoints
 * - User management and authentication
 * - Performance monitoring and health checks
 * - Rate limiting and timeout management
 * - Comprehensive error handling and fallback strategies
 * - Real-time data streaming capabilities
 * - A/B testing and experiment management
 * - Privacy-compliant data handling
 */
interface ApiService {

    companion object {
        // Base URLs for different services
        const val OPENAI_BASE_URL = "https://api.openai.com/v1/"
        const val ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
        const val GOOGLE_AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
        const val SOMNIAI_BACKEND_URL = "https://api.somniai.com/v1/"

        // Timeout configurations
        const val DEFAULT_TIMEOUT_SECONDS = 30L
        const val AI_GENERATION_TIMEOUT_SECONDS = 60L
        const val UPLOAD_TIMEOUT_SECONDS = 120L
        const val STREAMING_TIMEOUT_SECONDS = 300L

        // Rate limiting
        const val MAX_REQUESTS_PER_MINUTE = 60
        const val AI_MAX_REQUESTS_PER_MINUTE = 20

        // API versions
        const val API_VERSION = "v1"
        const val AI_API_VERSION = "v1"
    }

    // ========== OPENAI INTEGRATION ==========

    /**
     * Generate insights using OpenAI GPT models
     */
    @POST("chat/completions")
    @Headers(
        "Content-Type: application/json",
        "User-Agent: SomniAI/1.0"
    )
    suspend fun generateOpenAIInsights(
        @Header("Authorization") authorization: String,
        @Body request: OpenAICompletionRequest
    ): Response<OpenAICompletionResponse>

    /**
     * Generate embeddings for sleep data analysis
     */
    @POST("embeddings")
    @Headers("Content-Type: application/json")
    suspend fun generateEmbeddings(
        @Header("Authorization") authorization: String,
        @Body request: OpenAIEmbeddingRequest
    ): Response<OpenAIEmbeddingResponse>

    /**
     * Analyze sleep data using OpenAI Assistant API
     */
    @POST("assistants/{assistant_id}/threads/{thread_id}/runs")
    @Headers("Content-Type: application/json")
    suspend fun runOpenAIAssistant(
        @Header("Authorization") authorization: String,
        @Path("assistant_id") assistantId: String,
        @Path("thread_id") threadId: String,
        @Body request: OpenAIRunRequest
    ): Response<OpenAIRunResponse>

    /**
     * Get OpenAI assistant run status
     */
    @GET("assistants/{assistant_id}/threads/{thread_id}/runs/{run_id}")
    suspend fun getOpenAIRunStatus(
        @Header("Authorization") authorization: String,
        @Path("assistant_id") assistantId: String,
        @Path("thread_id") threadId: String,
        @Path("run_id") runId: String
    ): Response<OpenAIRunResponse>

    // ========== ANTHROPIC CLAUDE INTEGRATION ==========

    /**
     * Generate insights using Anthropic Claude
     */
    @POST("messages")
    @Headers(
        "Content-Type: application/json",
        "anthropic-version: 2023-06-01"
    )
    suspend fun generateClaudeInsights(
        @Header("x-api-key") apiKey: String,
        @Body request: ClaudeMessageRequest
    ): Response<ClaudeMessageResponse>

    /**
     * Stream Claude responses for real-time insights
     */
    @POST("messages")
    @Streaming
    @Headers(
        "Content-Type: application/json",
        "anthropic-version: 2023-06-01"
    )
    suspend fun streamClaudeInsights(
        @Header("x-api-key") apiKey: String,
        @Body request: ClaudeStreamRequest
    ): Response<ResponseBody>

    // ========== GOOGLE GEMINI INTEGRATION ==========

    /**
     * Generate insights using Google Gemini
     */
    @POST("models/{model}/generateContent")
    @Headers("Content-Type: application/json")
    suspend fun generateGeminiInsights(
        @Header("Authorization") authorization: String,
        @Path("model") model: String,
        @Query("key") apiKey: String,
        @Body request: GeminiGenerateRequest
    ): Response<GeminiGenerateResponse>

    /**
     * Stream Gemini responses
     */
    @POST("models/{model}/streamGenerateContent")
    @Streaming
    @Headers("Content-Type: application/json")
    suspend fun streamGeminiInsights(
        @Header("Authorization") authorization: String,
        @Path("model") model: String,
        @Query("key") apiKey: String,
        @Body request: GeminiGenerateRequest
    ): Response<ResponseBody>

    /**
     * Get Gemini model information
     */
    @GET("models/{model}")
    suspend fun getGeminiModelInfo(
        @Header("Authorization") authorization: String,
        @Path("model") model: String,
        @Query("key") apiKey: String
    ): Response<GeminiModelInfo>

    // ========== SOMNIAI BACKEND SERVICES ==========

    /**
     * User authentication and registration
     */
    @POST("auth/login")
    @Headers("Content-Type: application/json")
    suspend fun authenticateUser(
        @Body request: AuthenticationRequest
    ): Response<AuthenticationResponse>

    @POST("auth/register")
    @Headers("Content-Type: application/json")
    suspend fun registerUser(
        @Body request: UserRegistrationRequest
    ): Response<UserRegistrationResponse>

    @POST("auth/refresh")
    @Headers("Content-Type: application/json")
    suspend fun refreshToken(
        @Header("Authorization") authorization: String,
        @Body request: TokenRefreshRequest
    ): Response<TokenRefreshResponse>

    @POST("auth/logout")
    suspend fun logout(
        @Header("Authorization") authorization: String
    ): Response<ApiResponse<Unit>>

    /**
     * User profile and preferences
     */
    @GET("user/profile")
    suspend fun getUserProfile(
        @Header("Authorization") authorization: String
    ): Response<UserProfile>

    @PUT("user/profile")
    @Headers("Content-Type: application/json")
    suspend fun updateUserProfile(
        @Header("Authorization") authorization: String,
        @Body profile: UserProfileUpdate
    ): Response<UserProfile>

    @GET("user/preferences")
    suspend fun getUserPreferences(
        @Header("Authorization") authorization: String
    ): Response<UserPreferences>

    @PUT("user/preferences")
    @Headers("Content-Type: application/json")
    suspend fun updateUserPreferences(
        @Header("Authorization") authorization: String,
        @Body preferences: UserPreferencesUpdate
    ): Response<UserPreferences>

    /**
     * Sleep data synchronization
     */
    @POST("sleep/sessions/sync")
    @Headers("Content-Type: application/json")
    suspend fun syncSleepSessions(
        @Header("Authorization") authorization: String,
        @Body request: SleepSessionSyncRequest
    ): Response<SleepSessionSyncResponse>

    @GET("sleep/sessions")
    suspend fun getSleepSessions(
        @Header("Authorization") authorization: String,
        @Query("start_date") startDate: Long? = null,
        @Query("end_date") endDate: Long? = null,
        @Query("limit") limit: Int? = null,
        @Query("offset") offset: Int? = null
    ): Response<SleepSessionsResponse>

    @GET("sleep/sessions/{session_id}")
    suspend fun getSleepSession(
        @Header("Authorization") authorization: String,
        @Path("session_id") sessionId: Long
    ): Response<SleepSessionDetailResponse>

    @DELETE("sleep/sessions/{session_id}")
    suspend fun deleteSleepSession(
        @Header("Authorization") authorization: String,
        @Path("session_id") sessionId: Long
    ): Response<ApiResponse<Unit>>

    /**
     * Analytics and insights
     */
    @GET("analytics/overview")
    suspend fun getAnalyticsOverview(
        @Header("Authorization") authorization: String,
        @Query("period") period: String = "30d"
    ): Response<AnalyticsOverview>

    @GET("analytics/trends")
    suspend fun getSleepTrends(
        @Header("Authorization") authorization: String,
        @Query("period") period: String = "30d",
        @Query("metrics") metrics: List<String>? = null
    ): Response<SleepTrendsResponse>

    @GET("analytics/insights")
    suspend fun getPersonalizedInsights(
        @Header("Authorization") authorization: String,
        @Query("type") type: String? = null,
        @Query("priority") priority: String? = null,
        @Query("limit") limit: Int? = null
    ): Response<PersonalizedInsightsResponse>

    @POST("analytics/insights/feedback")
    @Headers("Content-Type: application/json")
    suspend fun submitInsightFeedback(
        @Header("Authorization") authorization: String,
        @Body feedback: InsightFeedbackRequest
    ): Response<ApiResponse<Unit>>

    /**
     * AI insights generation
     */
    @POST("ai/insights/generate")
    @Headers("Content-Type: application/json")
    suspend fun generateAIInsights(
        @Header("Authorization") authorization: String,
        @Body request: AIInsightGenerationRequest
    ): Response<AIInsightGenerationResponse>

    @GET("ai/insights/status/{job_id}")
    suspend fun getInsightGenerationStatus(
        @Header("Authorization") authorization: String,
        @Path("job_id") jobId: String
    ): Response<InsightGenerationStatusResponse>

    @POST("ai/insights/batch")
    @Headers("Content-Type: application/json")
    suspend fun generateBatchInsights(
        @Header("Authorization") authorization: String,
        @Body request: BatchInsightGenerationRequest
    ): Response<BatchInsightGenerationResponse>

    /**
     * Data backup and export
     */
    @POST("data/backup")
    @Headers("Content-Type: application/json")
    suspend fun createDataBackup(
        @Header("Authorization") authorization: String,
        @Body request: DataBackupRequest
    ): Response<DataBackupResponse>

    @GET("data/backup/{backup_id}")
    suspend fun getDataBackup(
        @Header("Authorization") authorization: String,
        @Path("backup_id") backupId: String
    ): Response<ResponseBody>

    @POST("data/export")
    @Headers("Content-Type: application/json")
    suspend fun exportUserData(
        @Header("Authorization") authorization: String,
        @Body request: DataExportRequest
    ): Response<DataExportResponse>

    @GET("data/export/{export_id}/download")
    suspend fun downloadExportedData(
        @Header("Authorization") authorization: String,
        @Path("export_id") exportId: String
    ): Response<ResponseBody>

    /**
     * Real-time data streaming
     */
    @GET("stream/sleep/live")
    @Streaming
    suspend fun streamLiveSleepData(
        @Header("Authorization") authorization: String,
        @Query("session_id") sessionId: Long
    ): Response<ResponseBody>

    @GET("stream/insights/live")
    @Streaming
    suspend fun streamLiveInsights(
        @Header("Authorization") authorization: String
    ): Response<ResponseBody>

    /**
     * Performance monitoring and health checks
     */
    @GET("health/check")
    suspend fun healthCheck(): Response<HealthCheckResponse>

    @GET("health/detailed")
    suspend fun detailedHealthCheck(
        @Header("Authorization") authorization: String
    ): Response<DetailedHealthResponse>

    @POST("monitoring/events")
    @Headers("Content-Type: application/json")
    suspend fun submitMonitoringEvent(
        @Header("Authorization") authorization: String,
        @Body event: MonitoringEventRequest
    ): Response<ApiResponse<Unit>>

    @GET("monitoring/metrics")
    suspend fun getPerformanceMetrics(
        @Header("Authorization") authorization: String,
        @Query("start_time") startTime: Long? = null,
        @Query("end_time") endTime: Long? = null
    ): Response<PerformanceMetricsResponse>

    /**
     * A/B testing and experiments
     */
    @GET("experiments/config")
    suspend fun getExperimentConfig(
        @Header("Authorization") authorization: String,
        @Query("version") version: String? = null
    ): Response<ExperimentConfigResponse>

    @POST("experiments/events")
    @Headers("Content-Type: application/json")
    suspend fun trackExperimentEvent(
        @Header("Authorization") authorization: String,
        @Body event: ExperimentEventRequest
    ): Response<ApiResponse<Unit>>

    /**
     * Push notifications
     */
    @POST("notifications/register")
    @Headers("Content-Type: application/json")
    suspend fun registerForNotifications(
        @Header("Authorization") authorization: String,
        @Body request: NotificationRegistrationRequest
    ): Response<ApiResponse<Unit>>

    @POST("notifications/preferences")
    @Headers("Content-Type: application/json")
    suspend fun updateNotificationPreferences(
        @Header("Authorization") authorization: String,
        @Body preferences: NotificationPreferencesUpdate
    ): Response<NotificationPreferences>

    @GET("notifications/history")
    suspend fun getNotificationHistory(
        @Header("Authorization") authorization: String,
        @Query("limit") limit: Int? = null,
        @Query("offset") offset: Int? = null
    ): Response<NotificationHistoryResponse>

    /**
     * Social features and sharing
     */
    @POST("social/share")
    @Headers("Content-Type: application/json")
    suspend fun shareInsight(
        @Header("Authorization") authorization: String,
        @Body request: InsightSharingRequest
    ): Response<InsightSharingResponse>

    @GET("social/leaderboard")
    suspend fun getLeaderboard(
        @Header("Authorization") authorization: String,
        @Query("type") type: String? = null,
        @Query("period") period: String? = null
    ): Response<LeaderboardResponse>

    @POST("social/challenges/join")
    @Headers("Content-Type: application/json")
    suspend fun joinChallenge(
        @Header("Authorization") authorization: String,
        @Body request: ChallengJoinRequest
    ): Response<ChallengeJoinResponse>

    /**
     * Research and data contribution
     */
    @POST("research/consent")
    @Headers("Content-Type: application/json")
    suspend fun updateResearchConsent(
        @Header("Authorization") authorization: String,
        @Body consent: ResearchConsentRequest
    ): Response<ApiResponse<Unit>>

    @POST("research/contribute")
    @Headers("Content-Type: application/json")
    suspend fun contributeToResearch(
        @Header("Authorization") authorization: String,
        @Body contribution: ResearchContributionRequest
    ): Response<ResearchContributionResponse>

    /**
     * Third-party integrations
     */
    @POST("integrations/connect")
    @Headers("Content-Type: application/json")
    suspend fun connectIntegration(
        @Header("Authorization") authorization: String,
        @Body request: IntegrationConnectionRequest
    ): Response<IntegrationConnectionResponse>

    @GET("integrations/status")
    suspend fun getIntegrationStatus(
        @Header("Authorization") authorization: String
    ): Response<IntegrationStatusResponse>

    @POST("integrations/sync/{integration_id}")
    suspend fun syncIntegration(
        @Header("Authorization") authorization: String,
        @Path("integration_id") integrationId: String
    ): Response<IntegrationSyncResponse>

    @DELETE("integrations/{integration_id}")
    suspend fun disconnectIntegration(
        @Header("Authorization") authorization: String,
        @Path("integration_id") integrationId: String
    ): Response<ApiResponse<Unit>>

    /**
     * Machine learning model management
     */
    @GET("ml/models/status")
    suspend fun getMLModelStatus(
        @Header("Authorization") authorization: String
    ): Response<MLModelStatusResponse>

    @POST("ml/models/train")
    @Headers("Content-Type: application/json")
    suspend fun triggerModelTraining(
        @Header("Authorization") authorization: String,
        @Body request: MLModelTrainingRequest
    ): Response<MLModelTrainingResponse>

    @POST("ml/predictions")
    @Headers("Content-Type: application/json")
    suspend fun getPredictions(
        @Header("Authorization") authorization: String,
        @Body request: MLPredictionRequest
    ): Response<MLPredictionResponse>

    /**
     * Content and recommendations
     */
    @GET("content/articles")
    suspend fun getEducationalContent(
        @Header("Authorization") authorization: String,
        @Query("category") category: String? = null,
        @Query("personalized") personalized: Boolean = true
    ): Response<EducationalContentResponse>

    @GET("content/recommendations")
    suspend fun getPersonalizedRecommendations(
        @Header("Authorization") authorization: String,
        @Query("type") type: String? = null
    ): Response<PersonalizedRecommendationsResponse>

    @POST("content/feedback")
    @Headers("Content-Type: application/json")
    suspend fun submitContentFeedback(
        @Header("Authorization") authorization: String,
        @Body feedback: ContentFeedbackRequest
    ): Response<ApiResponse<Unit>>

    /**
     * Support and feedback
     */
    @POST("support/ticket")
    @Headers("Content-Type: application/json")
    suspend fun createSupportTicket(
        @Header("Authorization") authorization: String,
        @Body ticket: SupportTicketRequest
    ): Response<SupportTicketResponse>

    @GET("support/tickets")
    suspend fun getSupportTickets(
        @Header("Authorization") authorization: String
    ): Response<SupportTicketsResponse>

    @POST("support/feedback")
    @Headers("Content-Type: application/json")
    suspend fun submitGeneralFeedback(
        @Header("Authorization") authorization: String,
        @Body feedback: GeneralFeedbackRequest
    ): Response<ApiResponse<Unit>>

    /**
     * Administrative endpoints
     */
    @GET("admin/statistics")
    suspend fun getAdminStatistics(
        @Header("Authorization") authorization: String,
        @Query("start_date") startDate: Long? = null,
        @Query("end_date") endDate: Long? = null
    ): Response<AdminStatisticsResponse>

    @POST("admin/broadcast")
    @Headers("Content-Type: application/json")
    suspend fun sendBroadcastMessage(
        @Header("Authorization") authorization: String,
        @Body message: BroadcastMessageRequest
    ): Response<ApiResponse<Unit>>

    @GET("admin/users")
    suspend fun getUsers(
        @Header("Authorization") authorization: String,
        @Query("page") page: Int? = null,
        @Query("limit") limit: Int? = null,
        @Query("search") search: String? = null
    ): Response<UsersResponse>

    /**
     * Rate limiting and quota management
     */
    @GET("quota/status")
    suspend fun getQuotaStatus(
        @Header("Authorization") authorization: String
    ): Response<QuotaStatusResponse>

    @POST("quota/upgrade")
    @Headers("Content-Type: application/json")
    suspend fun upgradeQuota(
        @Header("Authorization") authorization: String,
        @Body request: QuotaUpgradeRequest
    ): Response<QuotaUpgradeResponse>

    /**
     * Webhook management
     */
    @POST("webhooks/register")
    @Headers("Content-Type: application/json")
    suspend fun registerWebhook(
        @Header("Authorization") authorization: String,
        @Body webhook: WebhookRegistrationRequest
    ): Response<WebhookRegistrationResponse>

    @GET("webhooks")
    suspend fun getWebhooks(
        @Header("Authorization") authorization: String
    ): Response<WebhooksResponse>

    @DELETE("webhooks/{webhook_id}")
    suspend fun deleteWebhook(
        @Header("Authorization") authorization: String,
        @Path("webhook_id") webhookId: String
    ): Response<ApiResponse<Unit>>

    /**
     * Cache management
     */
    @POST("cache/invalidate")
    @Headers("Content-Type: application/json")
    suspend fun invalidateCache(
        @Header("Authorization") authorization: String,
        @Body request: CacheInvalidationRequest
    ): Response<ApiResponse<Unit>>

    @GET("cache/stats")
    suspend fun getCacheStatistics(
        @Header("Authorization") authorization: String
    ): Response<CacheStatisticsResponse>

    /**
     * Maintenance and system status
     */
    @GET("system/status")
    suspend fun getSystemStatus(): Response<SystemStatusResponse>

    @GET("system/version")
    suspend fun getSystemVersion(): Response<SystemVersionResponse>

    @GET("system/maintenance")
    suspend fun getMaintenanceSchedule(): Response<MaintenanceScheduleResponse>

    /**
     * Feature flags and configuration
     */
    @GET("config/features")
    suspend fun getFeatureFlags(
        @Header("Authorization") authorization: String,
        @Query("version") version: String? = null
    ): Response<FeatureFlagsResponse>

    @GET("config/app")
    suspend fun getAppConfiguration(
        @Header("Authorization") authorization: String,
        @Query("platform") platform: String = "android",
        @Query("version") version: String? = null
    ): Response<AppConfigurationResponse>

    /**
     * Emergency and crisis support
     */
    @POST("emergency/alert")
    @Headers("Content-Type: application/json")
    suspend fun submitEmergencyAlert(
        @Header("Authorization") authorization: String,
        @Body alert: EmergencyAlertRequest
    ): Response<EmergencyAlertResponse>

    @GET("emergency/resources")
    suspend fun getEmergencyResources(
        @Header("Authorization") authorization: String,
        @Query("location") location: String? = null
    ): Response<EmergencyResourcesResponse>

    /**
     * Batch operations
     */
    @POST("batch/operations")
    @Headers("Content-Type: application/json")
    suspend fun executeBatchOperation(
        @Header("Authorization") authorization: String,
        @Body operations: BatchOperationRequest
    ): Response<BatchOperationResponse>

    @GET("batch/operations/{operation_id}/status")
    suspend fun getBatchOperationStatus(
        @Header("Authorization") authorization: String,
        @Path("operation_id") operationId: String
    ): Response<BatchOperationStatusResponse>

    /**
     * Data privacy and compliance
     */
    @POST("privacy/data-request")
    @Headers("Content-Type: application/json")
    suspend fun submitDataRequest(
        @Header("Authorization") authorization: String,
        @Body request: DataPrivacyRequest
    ): Response<DataPrivacyResponse>

    @POST("privacy/consent")
    @Headers("Content-Type: application/json")
    suspend fun updatePrivacyConsent(
        @Header("Authorization") authorization: String,
        @Body consent: PrivacyConsentUpdate
    ): Response<PrivacyConsentResponse>

    @DELETE("privacy/account")
    suspend fun deleteAccount(
        @Header("Authorization") authorization: String,
        @Body confirmation: AccountDeletionRequest
    ): Response<ApiResponse<Unit>>

    /**
     * File upload and management
     */
    @Multipart
    @POST("files/upload")
    suspend fun uploadFile(
        @Header("Authorization") authorization: String,
        @Part("file") file: okhttp3.MultipartBody.Part,
        @Part("type") type: okhttp3.RequestBody,
        @Part("metadata") metadata: okhttp3.RequestBody? = null
    ): Response<FileUploadResponse>

    @GET("files/{file_id}")
    suspend fun downloadFile(
        @Header("Authorization") authorization: String,
        @Path("file_id") fileId: String
    ): Response<ResponseBody>

    @DELETE("files/{file_id}")
    suspend fun deleteFile(
        @Header("Authorization") authorization: String,
        @Path("file_id") fileId: String
    ): Response<ApiResponse<Unit>>

    /**
     * Analytics event tracking
     */
    @POST("analytics/events")
    @Headers("Content-Type: application/json")
    suspend fun trackAnalyticsEvent(
        @Header("Authorization") authorization: String,
        @Body event: AnalyticsEventRequest
    ): Response<ApiResponse<Unit>>

    @POST("analytics/events/batch")
    @Headers("Content-Type: application/json")
    suspend fun trackAnalyticsEventsBatch(
        @Header("Authorization") authorization: String,
        @Body events: AnalyticsEventsBatchRequest
    ): Response<ApiResponse<Unit>>

    /**
     * Custom AI model endpoints
     */
    @POST("ai/custom/train")
    @Headers("Content-Type: application/json")
    suspend fun trainCustomModel(
        @Header("Authorization") authorization: String,
        @Body request: CustomModelTrainingRequest
    ): Response<CustomModelTrainingResponse>

    @POST("ai/custom/predict")
    @Headers("Content-Type: application/json")
    suspend fun getCustomModelPrediction(
        @Header("Authorization") authorization: String,
        @Body request: CustomModelPredictionRequest
    ): Response<CustomModelPredictionResponse>

    @GET("ai/custom/models")
    suspend fun getCustomModels(
        @Header("Authorization") authorization: String
    ): Response<CustomModelsResponse>

    /**
     * Community and social features
     */
    @GET("community/posts")
    suspend fun getCommunityPosts(
        @Header("Authorization") authorization: String,
        @Query("category") category: String? = null,
        @Query("limit") limit: Int? = null,
        @Query("offset") offset: Int? = null
    ): Response<CommunityPostsResponse>

    @POST("community/posts")
    @Headers("Content-Type: application/json")
    suspend fun createCommunityPost(
        @Header("Authorization") authorization: String,
        @Body post: CommunityPostRequest
    ): Response<CommunityPostResponse>

    @POST("community/posts/{post_id}/like")
    suspend fun likeCommunityPost(
        @Header("Authorization") authorization: String,
        @Path("post_id") postId: String
    ): Response<ApiResponse<Unit>>

    @POST("community/posts/{post_id}/comment")
    @Headers("Content-Type: application/json")
    suspend fun commentOnCommunityPost(
        @Header("Authorization") authorization: String,
        @Path("post_id") postId: String,
        @Body comment: CommunityCommentRequest
    ): Response<CommunityCommentResponse>

    /**
     * Sleep coaching and guidance
     */
    @GET("coaching/plan")
    suspend fun getCoachingPlan(
        @Header("Authorization") authorization: String
    ): Response<CoachingPlanResponse>

    @POST("coaching/session")
    @Headers("Content-Type: application/json")
    suspend fun scheduleCoachingSession(
        @Header("Authorization") authorization: String,
        @Body request: CoachingSessionRequest
    ): Response<CoachingSessionResponse>

    @POST("coaching/feedback")
    @Headers("Content-Type: application/json")
    suspend fun submitCoachingFeedback(
        @Header("Authorization") authorization: String,
        @Body feedback: CoachingFeedbackRequest
    ): Response<ApiResponse<Unit>>

    /**
     * Weather and environment data
     */
    @GET("environment/weather")
    suspend fun getWeatherData(
        @Header("Authorization") authorization: String,
        @Query("lat") latitude: Double,
        @Query("lng") longitude: Double,
        @Query("date") date: Long? = null
    ): Response<WeatherDataResponse>

    @GET("environment/air-quality")
    suspend fun getAirQualityData(
        @Header("Authorization") authorization: String,
        @Query("lat") latitude: Double,
        @Query("lng") longitude: Double
    ): Response<AirQualityResponse>

    /**
     * Sleep device integrations
     */
    @POST("devices/pair")
    @Headers("Content-Type: application/json")
    suspend fun pairDevice(
        @Header("Authorization") authorization: String,
        @Body request: DevicePairingRequest
    ): Response<DevicePairingResponse>

    @GET("devices")
    suspend fun getConnectedDevices(
        @Header("Authorization") authorization: String
    ): Response<ConnectedDevicesResponse>

    @POST("devices/{device_id}/sync")
    suspend fun syncDevice(
        @Header("Authorization") authorization: String,
        @Path("device_id") deviceId: String
    ): Response<DeviceSyncResponse>

    @DELETE("devices/{device_id}")
    suspend fun unpairDevice(
        @Header("Authorization") authorization: String,
        @Path("device_id") deviceId: String
    ): Response<ApiResponse<Unit>>
}