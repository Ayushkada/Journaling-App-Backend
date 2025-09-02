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
    "You are aggregating pre-analyzed journals and goals into connected insights.\n"
    "Inputs: journals[{id,date,readability,sentiment,energy,mood{},keywords{},goals[]}], goals[{id,content,progress,category}].\n\n"
    "Build these fields (return only JSON):\n"
    "- moodTrends: {date: {emotion: number}}. For each date, weighted-average emotions across journals on that date.\n"
    "- energyTrends: {date: number}. Average journal energy per date.\n"
    "- averageSentiment: number. Weighted average across journals.\n"
    "- keywordEmotionMap: {keyword: {emotion: number}}. For each keyword, average emotions across journals containing it (weighted).\n"
    "- keywordEnergyMap: {keyword: number}. Average energy across journals containing it (weighted).\n"
    "- journalWeights: {journalId: number}. Recent journals get more weight: linearly scale 0.5..1.0 from oldest..newest.\n"
    "- goalMatches: {goalId: [journalId]}. Match when goal text overlaps journal goals/keywords or is strongly implied by mood/topics.\n"
    "- goalEmotionMap: {goalId: {emotion: number}}. Average emotions of matched journals per goal.\n"
    "- goalProgress: {goalId: {mentions:number, performanceScore:number, status:string}}.\n\n"
    "Guidelines:\n"
    "- Use journalWeights to bias averages.\n"
    "- Round numbers to 3 decimals, clamp to [0,1] if applicable.\n"
    "- Infer matches conservatively; prefer precision over recall.\n"
)

# System prompts
CONNECTED_SYSTEM_PROMPT: str = "Respond only with JSON."
FEEDBACK_SYSTEM_PROMPT: str = (
    "Return a strict JSON object with keys feedback, reflectiveQuestion, motivation.\n"
    "- feedback MUST be a single string (no nested objects).\n"
    "- reflectiveQuestion MUST be a single string.\n"
    "- motivation MUST be a single string.\n"
    "- reflectiveQuestion should be a single immediate self-reflection question (not a journaling prompt).\n"
    "- motivation should be 1–3 sentences, personalized using input trends/goals/keywords and matching the provided tone."
)
PROMPTS_SYSTEM_PROMPT: str = "Return JSON array only."

# User prompt templates
FEEDBACK_USER_TEMPLATE: str = (
    "Using this connected analysis JSON: {connected_json}\n\n"
    "Write {tone_style} feedback that is specific and actionable. Return a JSON object with keys: feedback, reflectiveQuestion, motivation.\n\n"
    "Inside feedback, include short titled sections in this order (as plain text, not Markdown):\n"
    "1) Trends: Summarize recent mood and energy shifts (reference dates).\n"
    "2) Patterns & Self-talk: Call out notable patterns implied by emotions and keywords.\n"
    "3) Goals spotlight: Mention 1–2 goals with strongest signal (use goalEmotionMap, goalProgress, goalMatches).\n"
    "4) Keyword insights: Cite 2–3 keywords with strongest emotional or energy signal.\n"
    "5) Next steps: List 2–3 concrete, bite-sized actions tied to the above.\n\n"
    "Constraints:\n"
    "- Keep feedback ~150–260 words, direct and empathetic, and match the {tone_style} tone.\n"
    "- Use journalWeights to favor recent patterns.\n"
    "- reflectiveQuestion: exactly one thoughtful question (one sentence), distinct from journaling prompts.\n"
    "- motivation: 1–3 sentences, grounded in recent mood/energy shifts, goal progress/matches, and top keywords; adapt style to {tone_style} (e.g., calm, direct, blunt, supportive)."
)

PROMPTS_USER_TEMPLATE: str = (
    "Given connected analysis JSON: {connected_json}\n\n"
    "Suggest 3 {tone_style} journaling prompts as a JSON array of strings.\n"
    "Guidelines:\n"
    "- Ground prompts in goalEmotionMap, goalProgress, keywordEmotionMap, and recent moodTrends.\n"
    "- Vary depth: include at least one deep reflective and one lighter prompt.\n"
    "- Vary time/effort: include at least one quick prompt (<10 minutes).\n"
    "- Avoid duplication and avoid quoting or numbering.\n"
    "- Keep each prompt concise (10–18 words)."
)

SIMILARITY_THRESHOLD: float = 0.75  # cosine threshold for goal deduplication
GOAL_MATCH_THRESHOLD: float = 0.5  # cosine similarity threshold to link journals to goals