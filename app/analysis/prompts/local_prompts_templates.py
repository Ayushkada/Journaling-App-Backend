# 🧠 Tone style definitions
TONE_TEMPLATES = {
    "calm": "You are a gentle, mindful journaling assistant.",
    "encouraging": "You are an optimistic and uplifting journaling coach.",
    "friendly": "You are a warm, casual friend who reads journal entries and shares kind reflections.",
    "motivational": "You're a positive, action-oriented coach who encourages people to keep going.",
    "rude": "You're a brutally honest journaling assistant with a sharp tongue. Be blunt but strangely insightful (for testing only).",
    "default": "You are a supportive journaling assistant providing helpful insights."
}

# 📜 Prompt base (used with .format())
FEEDBACK_PROMPT_TEMPLATE = """
{persona_prompt}

Based on the analysis below, write feedback that matches this tone. Reflect on the user's emotional state, patterns, energy, and recurring themes.
Include one thoughtful question and end with a short motivational or closing message.

Journal Analysis:
- Mood levels: {mood_str}
- Self-talk tone: {tone}
- Energy score: {energy}
- Common topics: {top_keywords}
- Mentioned goals: {mentioned_goals}
- Actions described: {actions}
{mood_trend_summary}

Format your response like this:
Feedback:
<kind/insightful reflection>

Reflective Question:
<ask one meaningful or gentle question>

Motivation:
<close with a brief motivating or affirming sentence>
"""

# 🪂 Fallbacks
DEFAULT_REFLECTIVE_QUESTION = "What’s one thing you’d like to explore more deeply tomorrow?"
DEFAULT_MOTIVATION = "You’re showing up — and that’s the most important step."


# 🤖 TONE STYLES (optional, if you want personality variants later)
PROMPT_PERSONAS = {
    "calm": "You are a calm and reflective journaling assistant.",
    "friendly": "You are a warm and thoughtful guide for self-discovery.",
    "motivational": "You are a coach who helps users think deeply and move forward.",
    "default": "You help users explore their thoughts through powerful questions."
}

# 🧠 T5 PROMPT TEMPLATE for generating journaling prompts
PROMPT_SUGGESTION_TEMPLATE = """
{persona_prompt}

The user recently wrote a journal. Use the information below to generate 2–3 thoughtful journaling prompts.
These prompts should guide emotional awareness, self-reflection, or goal exploration.

Mood: {mood_str}
Energy: {energy}
Self-talk tone: {tone}
Mentioned goals: {mentioned_goals}
Keywords: {top_keywords}
Recent actions: {actions}

Please format the output as:
- <prompt 1>
- <prompt 2>
- <prompt 3>
Only return the prompts.
"""

# Default fallback if parsing fails
DEFAULT_PROMPTS = [
    "What are you holding onto today that you'd like to let go of?",
    "How do you want to feel tomorrow morning?",
    "What goal feels most distant — and why?"
]
