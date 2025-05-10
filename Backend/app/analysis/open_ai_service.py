from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, List, Optional, Sequence, Union
from uuid import UUID

from app.goals.schemas import GoalBase, GoalCreate
from openai import OpenAI

from app.journals.schemas import (
    ConnectedAnalysisCreate,
    JournalAnalysisBase,
    JournalAnalysisCreate,
    JournalEntryBase,
)
import app.analysis.openai_prompts as prompts
from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")

# OpenAI client helpers
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _chat_json(messages: List[dict[str, Any]], *, max_tokens: int = 512) -> dict[str, Any]:
    """Run a chat completion and parse the JSON from the first choice."""
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def _embed(text: Union[str, Sequence[str]]) -> Union[List[float], List[List[float]]]:
    """Return embedding(s) for one string or a batch."""
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    if isinstance(text, str):
        return resp.data[0].embedding
    return [d.embedding for d in resp.data]


class OpenAIAIService:
    """Facade around OpenAI endpoints that speaks Pydantic schemas."""

    model_tag = "chatgpt"

    # 1. Single-entry analysis
    @staticmethod
    def analyze_entry(journal: JournalEntryBase) -> JournalAnalysisCreate:
        payload: dict[str, Any] = {
            "id": str(journal.id),
            "content": journal.content.strip(),
            "emojis": journal.emojis or [],
            "photos": journal.images or [],
            "analyze_images": journal.analyze_images,
        }

        vision_messages: list[dict[str, Any]] = []
        if payload["photos"] and payload["analyze_images"]:
            vision_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                        for img in payload["photos"][:3]
                    ],
                }
            )

        messages = [
            {"role": "system", "content": prompts.ENTRY_PROMPT},
            *vision_messages,
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        raw = _chat_json(messages)

        bag_of_words = " ".join(list(raw.get("keywords", {}).keys()) + raw.get("goalMentions", []))
        raw["textEmbedding"] = _embed(bag_of_words)

        return JournalAnalysisCreate(
            journal_id=journal.id,
            readability=raw["readability"],
            sentiment_score=raw["sentimentScore"],
            self_talk_tone=raw["selfTalkTone"],
            energy_score=raw["energyScore"],
            keywords=raw["keywords"],
            text_mood=raw["textMood"],
            emoji_mood=raw["emojiMood"],
            image_mood=raw["imageMood"],
            combined_mood=raw["mood"],
            goal_mentions=raw["goalMentions"],
            topics=raw["topics"],
            text_vector=raw["textVector"],
            text_embedding=raw["textEmbedding"],
            extracted_actions=raw["extractedActions"],
            date=journal.date.isoformat(),
            model=OpenAIAIService.model_tag
        )

    # 2. Connected analysis
    @staticmethod
    def analyze_connected(
        analyzed: List[JournalAnalysisBase],
        goals: List[GoalBase],
    ) -> ConnectedAnalysisCreate:
        prompt = (
            f"{prompts.CONNECTED_PROMPT}\n"
            f"journals = {json.dumps([a.dict() for a in analyzed])}\n\n"
            f"goals = {json.dumps([g.dict() for g in goals])}"
        )
        raw = _chat_json(
            [
                {"role": "system", "content": "Respond only with JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )

        return ConnectedAnalysisCreate(
            user_id=goals[0].user_id if goals else analyzed[0].journal_id,
            mood_trends=raw["moodTrends"],
            energy_trends=raw["energyTrends"],
            average_sentiment=raw["averageSentiment"],
            goal_emotion_map=raw["goalEmotionMap"],
            goal_progress=raw["goalProgress"],
            goal_matches=raw["goalMatches"],
            keyword_emotion_map=raw["keywordEmotionMap"],
            keyword_energy_map=raw["keywordEnergyMap"],
            journal_weights=raw["journalWeights"],
            model=OpenAIAIService.model_tag
        )

    # 3. Feedback & journaling prompt helpers
    @staticmethod
    def generate_feedback(connected: ConnectedAnalysisCreate, *, tone_style: str = "calm") -> dict[str, str]:
        prompt = (
            f"Write {tone_style} feedback, reflectiveQuestion and motivation as JSON "
            f"based on: {json.dumps(connected.dict())}"
        )
        return _chat_json(
            [
                {"role": "system", "content": "Return JSON with feedback, reflectiveQuestion, motivation."},
                {"role": "user", "content": prompt},
            ]
        )

    @staticmethod
    def recommend_journaling_prompts(connected: ConnectedAnalysisCreate, *, tone_style: str = "friendly") -> list[str]:
        prompt = (
            f"Suggest 3 {tone_style} journaling prompts tailored to this data. "
            f"Return JSON array of strings.\nData: {json.dumps(connected.dict())}"
        )
        return _chat_json(
            [
                {"role": "system", "content": "Return JSON array only."},
                {"role": "user", "content": prompt},
            ]
        )

    # 4. Goal deduplication
    @staticmethod
    def get_unique_goal_texts(
        existing_goal_texts: list[str],
        entry_id_to_goals: dict[UUID, list[str]],
        *,
        similarity_threshold: float = prompts.SIMILARITY_THRESHOLD,
    ) -> dict[UUID, list[GoalCreate]]:
        existing_emb: Optional[list[list[float]]] = None
        if existing_goal_texts:
            existing_emb = _embed([g.lower().strip() for g in existing_goal_texts])  # type: ignore

        out: dict[UUID, list[GoalCreate]] = {}
        for eid, goals in entry_id_to_goals.items():
            for goal_text in goals:
                emb = _embed(goal_text)  # type: ignore
                if existing_emb and max(sum(a*b for a, b in zip(emb, vec)) for vec in existing_emb) > similarity_threshold:
                    continue
                new_goal = GoalCreate(
                    content=goal_text,
                    aiGenerated=True,
                    category="AI Suggested",
                    created_at=_dt.datetime.now(_dt.timezone.utc),
                    progress_score=0.0,
                )
                out.setdefault(eid, []).append(new_goal)
                if existing_emb:
                    existing_emb.append(emb)
                else:
                    existing_emb = [emb]
        return out
