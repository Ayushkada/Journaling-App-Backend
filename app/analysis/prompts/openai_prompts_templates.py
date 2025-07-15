ENTRY_PROMPT: str = (
    "You are a JSON-only analysis engine for a journaling app. "
    "Given a journal object respond with the schema below. "
    "Identify emotional tones, self-talk patterns, goals, and suggested improvement actions. "
    "Return *only* valid JSON, no markdown.\n\n"
    "Schema: {\n"
    "  readability: float, sentimentScore: float, selfTalkTone: str,\n"
    "  energyScore: float, keywords: {str:int}, textMood: {emotion:float},\n"
    "  emojiMood: {emotion:float}, imageMood: {emotion:float}, mood: {emotion:float},\n"
    "  goalMentions: [str], topics: [{caption:str,imageMood:str}],\n"
    "  textVector: str, extractedActions: str\n"
    "}\n"
)

CONNECTED_PROMPT: str = (
    "Aggregate the following journal analyses and goals into trends and maps.\n"
    "Return JSON with keys: moodTrends, energyTrends, averageSentiment, "
    "goalEmotionMap, goalProgress, goalMatches, keywordEmotionMap, "
    "keywordEnergyMap, journalWeights.\n\n"
)

SIMILARITY_THRESHOLD: float = 0.75  # cosine threshold for goal deduplication