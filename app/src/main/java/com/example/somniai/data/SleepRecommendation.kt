package com.example.somniai.data

/**
 * Data class representing a sleep recommendation
 */
data class SleepRecommendation(
    val id: Long = 0L,
    val title: String,
    val description: String,
    val type: RecommendationType,
    val priority: Int = 2, // 1 = High, 2 = Medium, 3 = Low
    val actionText: String? = null,
    val isAiGenerated: Boolean = false,
    val confidence: Float = 0.8f,
    val effectivenessScore: Float? = null,
    val createdAt: Long = System.currentTimeMillis(),
    val isCompleted: Boolean = false,
    val tags: List<String> = emptyList()
)