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

# System prompts
CONNECTED_SYSTEM_PROMPT: str = "Respond only with JSON."
FEEDBACK_SYSTEM_PROMPT: str = "Return JSON with feedback, reflectiveQuestion, motivation."
PROMPTS_SYSTEM_PROMPT: str = "Return JSON array only."

# User prompt templates
FEEDBACK_USER_TEMPLATE: str = (
    "Write {tone_style} feedback, reflectiveQuestion and motivation as JSON based on: {connected_json}"
)

PROMPTS_USER_TEMPLATE: str = (
    "Suggest 3 {tone_style} journaling prompts tailored to this data. Return JSON array of strings.\n"
    "Data: {connected_json}"
)

SIMILARITY_THRESHOLD: float = 0.75  # cosine threshold for goal deduplication