from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, List, Optional, Sequence, Union
from uuid import UUID
import logging

from app.goals.schemas import GoalBase
from openai import OpenAI

from app.journals.schemas import (
    JournalEntryBase,
)
from app.analysis.schemas import (
    ConnectedAnalysisCreate,
    JournalAnalysisBase,
    JournalAnalysisCreate
)
import app.analysis.prompts.openai_prompts_templates as prompts
from dotenv import load_dotenv
from app.analysis.schemas import EntryLLMResponse as _RawEntryAnalysis, ConnectedLLMResponse as _RawConnectedAnalysis
import textstat
import re

load_dotenv()
logger = logging.getLogger(__name__)

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not CHAT_MODEL:
    raise RuntimeError("Missing OPENAI_CHAT_MODEL in environment")
if not EMBED_MODEL:
    raise RuntimeError("Missing OPENAI_EMBED_MODEL in environment")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

_EMBED_CACHE: dict[str, List[float]] = {}

ENTRY_JSON_SCHEMA: dict[str, Any] = {
    "name": "entry_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "readability": {"type": "number"},
            "sentimentScore": {"type": "number"},
            "selfTalkTone": {"type": "string"},
            "energyScore": {"type": "number"},
            "keywords": {"type": "object", "additionalProperties": {"type": "integer"}},
            "textMood": {"type": "object", "additionalProperties": {"type": "number"}},
            "emojiMood": {"type": "object", "additionalProperties": {"type": "number"}},
            "imageMood": {"type": "object", "additionalProperties": {"type": "number"}},
            "mood": {"type": "object", "additionalProperties": {"type": "number"}},
            "goalMentions": {"type": "array", "items": {"type": "string"}},
            "topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "imageMood": {"type": "string"}
                    },
                    "additionalProperties": True
                }
            },
            "textVector": {"type": ["string", "null"]},
            "extractedActions": {"type": "string"}
        },
        "additionalProperties": True
    }
}

CONNECTED_JSON_SCHEMA: dict[str, Any] = {
    "name": "connected_analysis",
    "schema": {
        "type": "object",
        "properties": {
            "moodTrends": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                }
            },
            "energyTrends": {"type": "object", "additionalProperties": {"type": "number"}},
            "averageSentiment": {"type": "number"},
            "goalEmotionMap": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                }
            },
            "goalProgress": {"type": "object"},
            "goalMatches": {"type": "object"},
            "keywordEmotionMap": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                }
            },
            "keywordEnergyMap": {"type": "object", "additionalProperties": {"type": "number"}},
            "journalWeights": {"type": "object", "additionalProperties": {"type": "number"}}
        },
        "additionalProperties": True
    }
}

PROMPTS_ARRAY_SCHEMA: dict[str, Any] = {
    "name": "prompt_array",
    "schema": {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["prompts"],
        "additionalProperties": False
    }
}



# Validation models are imported from analysis.schemas for separation of concerns

# Strict schema for feedback JSON to keep fields as strings
FEEDBACK_JSON_SCHEMA: dict[str, Any] = {
    "name": "feedback_object",
    "schema": {
        "type": "object",
        "properties": {
            "feedback": {"type": "string"},
            "reflectiveQuestion": {"type": "string"},
            "motivation": {"type": "string"}
        },
        "required": ["feedback", "reflectiveQuestion", "motivation"],
        "additionalProperties": False
    }
}

def _chat_json(
    messages: List[dict[str, Any]], *, max_tokens: int = 512, response_format_type: Optional[str] = "json_object", response_format_schema: Optional[dict[str, Any]] = None
) -> Any:
    """Run a chat completion and parse JSON (object or array) from the first choice.

    response_format_type: set to "json_object" for strict objects; set to None to allow arrays.
    """
    kwargs: dict[str, Any] = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if response_format_schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": response_format_schema}
    elif response_format_type:
        kwargs["response_format"] = {"type": response_format_type}
    last_error: Optional[Exception] = None
    augmented_messages = list(messages)
    def _try_repair_parse(raw: Optional[str]) -> Any:
        if not raw:
            raise ValueError("Empty content")
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`\n ")
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            sub = s[start : end + 1]
            return json.loads(sub)
        return json.loads(s)

    for attempt in range(2):
        try:
            resp = openai_client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except Exception:
                return _try_repair_parse(content)
        except Exception as e:
            last_error = e
            logger.warning(f"OpenAI chat JSON parse failed (attempt {attempt+1}): {e}")
            augmented_messages = [
                {"role": "system", "content": "Return strictly valid JSON only, no prose."},
                *messages,
            ]
            kwargs["messages"] = augmented_messages
    raise RuntimeError(f"Failed to parse JSON from Chat Completions: {last_error}")

def _embed(text: Union[str, Sequence[str]]) -> Union[List[float], List[List[float]]]:
    """Return embedding(s) for one string or a batch."""
    if isinstance(text, str):
        cached = _EMBED_CACHE.get(text)
        if cached is not None:
            return cached
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
        emb = resp.data[0].embedding
        _EMBED_CACHE[text] = emb
        return emb
    results: List[List[float]] = []
    to_query: List[str] = []
    positions: List[int] = []
    for i, s in enumerate(text):  # type: ignore
        if s in _EMBED_CACHE:
            results.append(_EMBED_CACHE[s])
        else:
            results.append([]) 
            to_query.append(s)
            positions.append(i)
    if to_query:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=to_query)
        for j, d in enumerate(resp.data):
            emb = d.embedding
            idx = positions[j]
            results[idx] = emb
            _EMBED_CACHE[to_query[j]] = emb
    return results


class OpenAIAIService:
    """Facade around OpenAI endpoints that speaks Pydantic schemas."""

    model_tag = "chatgpt"

    # 1. Single-entry analysis
    @staticmethod
    def analyze_entry(journal: JournalEntryBase) -> JournalAnalysisCreate:
        """Analyze a single journal entry and return a JournalAnalysisCreate."""
        content = (journal.content or "").strip()
        if len(content) > 4000:
            content = content[:4000]
        readability = round(textstat.flesch_reading_ease(content), 2) if content else 0.0
        payload: dict[str, Any] = {
            "id": str(journal.id),
            "content": content,
            "emojis": journal.emojis or [],
            "signal": {
                "readability": readability,
                "wordCount": len(content.split()),
                "exclamations": content.count("!"),
                "capsWords": len([w for w in content.split() if len(w) > 2 and w.isupper()]),
            },
            # Images deferred for later implementation
            "photos": [],
            "analyze_images": False,
        }

        messages = [
            {"role": "system", "content": prompts.ENTRY_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        try:
            raw_obj = _chat_json(messages, response_format_type=None, response_format_schema=ENTRY_JSON_SCHEMA)
            raw = _RawEntryAnalysis(**raw_obj).model_dump()
        except Exception as e:
            logger.warning(f"Falling back to heuristic entry analysis due to JSON error: {e}")
            # Heuristic fallback: derive simple keywords and defaults
            words_clean = [w.strip(".,!?;:\"'()") for w in content.lower().split()]
            kw_counts: dict[str, int] = {}
            for w in words_clean:
                if len(w) > 3:
                    kw_counts[w] = kw_counts.get(w, 0) + 1
            raw = {
                "readability": readability,
                "sentimentScore": 0.0,
                "selfTalkTone": "NEUTRAL",
                "energyScore": 0.0,
                "keywords": dict(sorted(kw_counts.items(), key=lambda x: -x[1])[:15]),
                "textMood": {},
                "emojiMood": {},
                "imageMood": {},
                "mood": {},
                "goalMentions": [],
                "topics": [],
                "textVector": None,
                "extractedActions": "",
            }

        keywords = raw.get("keywords", {}) or {}
        goal_mentions = raw.get("goalMentions", []) or []
        bag_of_words = " ".join(list(keywords.keys()) + goal_mentions).strip()
        if not bag_of_words:
            bag_of_words = (payload["content"][:256]).strip() or "journal"
        text_embedding: List[float] = _embed(bag_of_words)  # type: ignore

        text_vector = raw.get("textVector") or " ".join(list(keywords.keys()))

        def _compute_energy_score(text: str) -> float:
            words = text.split()
            word_count = max(len(words), 1)
            exclamations = text.count("!")
            caps_ratio = sum(1 for w in words if len(w) > 2 and w.isupper()) / word_count
            sentences = [s for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
            lens = [len(s.split()) for s in sentences] if sentences else [word_count]
            mean_len = sum(lens) / len(lens)
            var_len = sum((l - mean_len) ** 2 for l in lens) / max(len(lens), 1)
            base = (exclamations / word_count) + caps_ratio + (var_len / 100.0)
            return round(min(max(base, 0.0), 1.0), 3)

        return JournalAnalysisCreate(
            journal_id=journal.id,
            readability=float(raw.get("readability", 0.0)),
            sentiment_score=float(raw.get("sentimentScore", 0.0)),
            self_talk_tone=str(raw.get("selfTalkTone", "NEUTRAL")),
            energy_score=float(raw.get("energyScore", _compute_energy_score(payload["content"]))),
            keywords=keywords,
            text_mood=raw.get("textMood", {}) or {},
            emoji_mood=raw.get("emojiMood", {}) or {},
            image_mood={},
            combined_mood=raw.get("mood", {}) or {},
            goal_mentions=goal_mentions,
            topics=raw.get("topics", []) or [],
            text_vector=text_vector,
            text_embedding=text_embedding,
            extracted_actions=str(raw.get("extractedActions", "")),
            date=journal.date.isoformat(),
            model=OpenAIAIService.model_tag
        )

    # 2. Connected analysis
    @staticmethod
    def analyze_connected(
        analyzed: List[JournalAnalysisBase],
        goals: List[GoalBase],
    ) -> ConnectedAnalysisCreate:
        """Build compact summaries and produce a ConnectedAnalysisCreate."""
        def _top_k(d: dict[str, float], k: int = 5) -> dict[str, float]:
            try:
                items = sorted(d.items(), key=lambda x: abs(x[1]), reverse=True)
                return {k_: float(v) for k_, v in items[:k]}
            except Exception:
                return {}

        def _top_k_int(d: dict[str, int], k: int = 10) -> dict[str, int]:
            try:
                items = sorted(d.items(), key=lambda x: x[1], reverse=True)
                return {k_: int(v) for k_, v in items[:k]}
            except Exception:
                return {}

        def _date_to_str(val: Any) -> str:
            try:
                if isinstance(val, str):
                    return val[:10]
                return val.isoformat()[:10]
            except Exception:
                return ""

        analyzed_slim: list[dict[str, Any]] = []
        for a in analyzed[:30]:  # hard cap
            analyzed_slim.append(
                {
                    "id": str(getattr(a, "id", "")),
                    "date": _date_to_str(getattr(a, "date", None) or getattr(a, "analysis_date", None)),
                    "readability": float(a.readability),
                    "sentiment": float(a.sentiment_score),
                    "energy": float(a.energy_score),
                    "mood": _top_k(getattr(a, "combined_mood", {}) or {}),
                    "keywords": _top_k_int(getattr(a, "keywords", {}) or {}),
                    "goals": (getattr(a, "goal_mentions", []) or [])[:3],
                }
            )

        goals_slim: list[dict[str, Any]] = []
        for g in goals[:50]:
            goals_slim.append(
                {
                    "id": str(getattr(g, "id", "")),
                    "content": getattr(g, "content", ""),
                    "progress": float(getattr(g, "progress_score", 0.0) or 0.0),
                    "category": getattr(g, "category", None),
                }
            )

        prompt = (
            f"{prompts.CONNECTED_PROMPT}\n"
            f"journals = {json.dumps(analyzed_slim, default=str)}\n\n"
            f"goals = {json.dumps(goals_slim, default=str)}"
        )
        try:
            raw_obj = _chat_json(
                [
                    {"role": "system", "content": prompts.CONNECTED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                response_format_type=None,
                response_format_schema=CONNECTED_JSON_SCHEMA,
            )
            raw = _RawConnectedAnalysis(**raw_obj).model_dump()
        except Exception as e:
            logger.warning(f"Falling back to heuristic connected analysis due to JSON error: {e}")
            # Heuristic fallback from provided analyzed data
            mood_trends: dict[str, dict[str, float]] = {}
            energy_trends: dict[str, float] = {}
            sentiments: list[float] = []
            for a in analyzed:
                _dt_val = getattr(a, "date", None) or getattr(a, "analysis_date", None)
                if isinstance(_dt_val, str):
                    date_key = _dt_val[:10]
                else:
                    try:
                        date_key = _dt_val.isoformat()[:10]
                    except Exception:
                        date_key = ""
                energy_trends[date_key] = a.energy_score
                sentiments.append(a.sentiment_score)
                cm = a.combined_mood or {}
                if cm:
                    mt = mood_trends.setdefault(date_key, {})
                    for emotion, score in cm.items():
                        prev = mt.get(emotion)
                        mt[emotion] = float(score) if prev is None else (prev + float(score)) / 2.0
            raw = {
                "moodTrends": mood_trends,
                "energyTrends": energy_trends,
                "averageSentiment": round(sum(sentiments) / max(len(sentiments), 1), 3) if sentiments else 0.0,
                "goalEmotionMap": {},
                "goalProgress": {},
                "goalMatches": {},
                "keywordEmotionMap": {},
                "keywordEnergyMap": {},
                "journalWeights": {},
            }

        # Deterministic backfill for empty maps using available analyzed data
        try:
            # 1) Entry weights by recency (0.5 .. 1.0)
            def _date_val(a: JournalAnalysisBase) -> str:
                d = getattr(a, "date", None) or getattr(a, "analysis_date", None)
                try:
                    return d[:10] if isinstance(d, str) else d.isoformat()[:10]
                except Exception:
                    return ""

            sorted_entries = sorted(analyzed, key=lambda x: _date_val(x))
            total = max(len(sorted_entries) - 1, 1)
            entry_weights: dict[str, float] = {
                str(e.id): round(0.5 + 0.5 * (i / total), 3) for i, e in enumerate(sorted_entries)
            }

            # 2) Keyword emotion/energy maps
            kw_emotion_acc: dict[str, dict[str, list[float]]] = {}
            kw_energy_acc: dict[str, list[float]] = {}
            for a in analyzed:
                weight = entry_weights.get(str(a.id), 1.0)
                cm = getattr(a, "combined_mood", {}) or {}
                kws = (getattr(a, "keywords", {}) or {}).keys()
                for kw in kws:
                    em = kw_emotion_acc.setdefault(kw, {})
                    for emotion, score in cm.items():
                        em.setdefault(emotion, []).append(float(score) * weight)
                    kw_energy_acc.setdefault(kw, []).append(float(getattr(a, "energy_score", 0.0)) * weight)

            def _avg_list(vals: list[float]) -> float:
                return round(sum(vals) / max(len(vals), 1), 3) if vals else 0.0

            kw_emotion_final: dict[str, dict[str, float]] = {
                kw: {emo: _avg_list(vals) for emo, vals in emo_map.items()} for kw, emo_map in kw_emotion_acc.items()
            }
            kw_energy_final: dict[str, float] = {kw: _avg_list(vals) for kw, vals in kw_energy_acc.items()}

            # 3) Goal matches and goal emotion map
            # Prepare journal embedding matrix and ids
            journal_ids: list[str] = [str(a.id) for a in analyzed]
            journal_embs: list[list[float]] = [list(getattr(a, "text_embedding", []) or []) for a in analyzed]

            def _norm(vec: list[float]) -> float:
                return (sum(v * v for v in vec)) ** 0.5

            def _cosine(a: list[float], b: list[float]) -> float:
                da = _norm(a)
                db = _norm(b)
                if da == 0.0 or db == 0.0:
                    return 0.0
                return sum(x * y for x, y in zip(a, b)) / (da * db)

            goal_emotion_map: dict[str, dict[str, float]] = {}
            goal_matches: dict[str, list[str]] = {}
            goal_progress: dict[str, dict] = {}

            for g in goals[:50]:
                gtext = f"{getattr(g, 'content', '')}"
                try:
                    gemb = _embed(gtext)  # type: ignore
                except Exception:
                    gemb = []
                matched_idxs: list[int] = []
                thr = float(getattr(prompts, "GOAL_MATCH_THRESHOLD", 0.5))
                if gemb and journal_embs:
                    for i, emb in enumerate(journal_embs):
                        if not emb:
                            continue
                        if _cosine(gemb, emb) >= thr:  # type: ignore[arg-type]
                            matched_idxs.append(i)

                # Aggregate
                related_moods: dict[str, list[float]] = {}
                matched_ids: list[str] = []
                for i in matched_idxs:
                    a = analyzed[i]
                    matched_ids.append(journal_ids[i])
                    cm = getattr(a, "combined_mood", {}) or {}
                    for emotion, score in cm.items():
                        related_moods.setdefault(emotion, []).append(float(score))

                gid = str(getattr(g, "id", ""))
                if matched_ids:
                    goal_emotion_map[gid] = {e: _avg_list(vals) for e, vals in related_moods.items()}
                goal_matches[gid] = matched_ids
                perf = round(len(matched_ids) / max(len(journal_ids), 1), 3)
                status = "on_track" if (float(getattr(g, "progress_score", 0.0) or 0.0) >= 0.8) else "behind"
                goal_progress[gid] = {"mentions": len(matched_ids), "performanceScore": perf, "status": status}

            # 4) Merge into raw only if empty/missing
            if not raw.get("keywordEmotionMap"):
                raw["keywordEmotionMap"] = kw_emotion_final
            if not raw.get("keywordEnergyMap"):
                raw["keywordEnergyMap"] = kw_energy_final
            if not raw.get("journalWeights"):
                raw["journalWeights"] = entry_weights
            if not raw.get("goalEmotionMap"):
                raw["goalEmotionMap"] = goal_emotion_map
            if not raw.get("goalMatches"):
                raw["goalMatches"] = goal_matches
            if not raw.get("goalProgress"):
                raw["goalProgress"] = goal_progress
        except Exception as _e:
            # Keep best-effort raw
            logger.warning(f"Connected deterministic backfill failed: {_e}")
        # Post-process: fill missing dates and smooth trends (EMA)
        def _parse_date(s: str) -> _dt.date:
            try:
                return _dt.date.fromisoformat(s[:10])
            except Exception:
                return _dt.date.today()

        def _daterange(d0: _dt.date, d1: _dt.date):
            cur = d0
            while cur <= d1:
                yield cur
                cur = cur + _dt.timedelta(days=1)

        def _ema(values: List[float], alpha: float = 0.3) -> List[float]:
            if not values:
                return values
            out = [values[0]]
            for v in values[1:]:
                out.append(alpha * v + (1 - alpha) * out[-1])
            return out

        mood_trends_raw = raw.get("moodTrends", {}) or {}
        energy_trends_raw = raw.get("energyTrends", {}) or {}

        # Normalize date keys and fill gaps
        if mood_trends_raw:
            dates = sorted(_parse_date(d) for d in mood_trends_raw.keys())
            if dates:
                start, end = dates[0], dates[-1]
                all_days = [d.isoformat() for d in _daterange(start, end)]
                # Collect all emotions
                emotions = set()
                for m in mood_trends_raw.values():
                    emotions.update(m.keys())
                mood_trends_filled: dict[str, dict[str, float]] = {}
                for day in all_days:
                    if day in mood_trends_raw:
                        mood_trends_filled[day] = {e: float(mood_trends_raw[day].get(e, 0.0)) for e in emotions}
                    else:
                        mood_trends_filled[day] = {e: 0.0 for e in emotions}
                # Smooth each emotion series via EMA
                smoothed: dict[str, dict[str, float]] = {d: {} for d in all_days}
                for e in emotions:
                    series = [mood_trends_filled[d][e] for d in all_days]
                    s = _ema(series)
                    for i, d in enumerate(all_days):
                        smoothed[d][e] = round(s[i], 3)
                mood_trends_out = smoothed
            else:
                mood_trends_out = {}
        else:
            mood_trends_out = {}

        if energy_trends_raw:
            dates_e = sorted(_parse_date(d) for d in energy_trends_raw.keys())
            if dates_e:
                start, end = dates_e[0], dates_e[-1]
                all_days = [d.isoformat() for d in _daterange(start, end)]
                series = [float(energy_trends_raw.get(d, 0.0)) for d in all_days]
                s = _ema(series)
                energy_trends_out = {d: round(s[i], 3) for i, d in enumerate(all_days)}
            else:
                energy_trends_out = {}
        else:
            energy_trends_out = {}

        return ConnectedAnalysisCreate(
            mood_trends=mood_trends_out,
            energy_trends=energy_trends_out,
            average_sentiment=float(raw.get("averageSentiment", 0.0)),
            goal_emotion_map=raw.get("goalEmotionMap", {}) or {},
            goal_progress=raw.get("goalProgress", {}) or {},
            goal_matches=raw.get("goalMatches", {}) or {},
            keyword_emotion_map=raw.get("keywordEmotionMap", {}) or {},
            keyword_energy_map=raw.get("keywordEnergyMap", {}) or {},
            journal_weights=raw.get("journalWeights", {}) or {},
            model=OpenAIAIService.model_tag
        )

    # 3. Feedback & journaling prompt helpers
    @staticmethod
    def generate_feedback(connected: ConnectedAnalysisCreate, *, tone_style: str = "calm") -> dict[str, str]:
        """Return dict with feedback, reflectiveQuestion, and motivation."""
        prompt = prompts.FEEDBACK_USER_TEMPLATE.format(
            tone_style=tone_style, connected_json=json.dumps(connected.model_dump(), default=str)
        )
        raw = _chat_json(
            [
                {"role": "system", "content": prompts.FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format_type=None,
            response_format_schema=FEEDBACK_JSON_SCHEMA,
        )
        # Normalize to strings defensively
        def _to_str(val: Any) -> str:
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                # Flatten dict sections into a readable paragraph
                parts = []
                for k, v in val.items():
                    try:
                        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    except Exception:
                        text = str(v)
                    parts.append(f"{k}: {text}")
                return "\n".join(parts)
            if isinstance(val, list):
                return "\n".join(_to_str(x) for x in val)
            try:
                return str(val)
            except Exception:
                return ""

        return {
            "feedback": _to_str(raw.get("feedback", "")),
            "reflectiveQuestion": _to_str(raw.get("reflectiveQuestion", "")),
            "motivation": _to_str(raw.get("motivation", "")),
        }

    @staticmethod
    def recommend_journaling_prompts(connected: ConnectedAnalysisCreate, *, tone_style: str = "friendly") -> list[str]:
        """Return list of journaling prompts based on a connected analysis."""
        prompt = prompts.PROMPTS_USER_TEMPLATE.format(
            tone_style=tone_style, connected_json=json.dumps(connected.model_dump(), default=str)
        )
        raw = _chat_json(
            [
                {"role": "system", "content": prompts.PROMPTS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format_type=None,
            response_format_schema=PROMPTS_ARRAY_SCHEMA,
        )
        if isinstance(raw, dict) and "prompts" in raw and isinstance(raw["prompts"], list):
            return [str(x) for x in raw["prompts"]]
        return []

    # 4. Goal deduplication
    @staticmethod
    def get_unique_goal_texts(
        existing_goal_texts: list[str],
        entry_id_to_goals: dict[UUID, list[str]],
        *,
        similarity_threshold: float = prompts.SIMILARITY_THRESHOLD,
    ) -> dict[UUID, list[dict[str, Any]]]:
        CATEGORY_KEYWORDS: dict[str, list[str]] = {
            "Fitness": ["workout", "run", "jog", "lift", "bench", "gym", "exercise"],
            "Nutrition": ["calorie", "protein", "eat", "meal", "diet", "water"],
            "Career": ["resume", "interview", "job", "apply", "portfolio", "network"],
            "Learning": ["study", "course", "read", "learn", "exam", "flashcard"],
            "Productivity": ["ship", "focus", "plan", "habit", "track", "organize"],
            "Mindfulness": ["meditate", "journal", "reflect", "gratitude", "breath"],
            "Social": ["friend", "conversation", "ask", "call", "text", "meet"],
            "Sleep": ["sleep", "bedtime", "wake", "alarm", "rest"],
            "Finance": ["budget", "save", "invest", "spend"],
        }

        def _infer_categories(goal_text: str) -> list[str]:
            text = goal_text.lower()
            hits: list[str] = []
            for cat, kws in CATEGORY_KEYWORDS.items():
                if any(kw in text for kw in kws):
                    hits.append(cat)
            return hits or ["Other"]

        DURATION_RE = re.compile(r"(?P<num>\d+)\s*(?P<unit>day|days|week|weeks|month|months|year|years)", re.I)

        def _parse_time_limit(goal_text: str) -> Optional[_dt.datetime]:
            m = DURATION_RE.search(goal_text)
            if not m:
                return None
            num = int(m.group("num"))
            unit = m.group("unit").lower()
            now = _dt.datetime.now(_dt.timezone.utc)
            if unit.startswith("day"):
                return now + _dt.timedelta(days=num)
            if unit.startswith("week"):
                return now + _dt.timedelta(weeks=num)
            if unit.startswith("month"):
                # approximate month as 30 days for simplicity
                return now + _dt.timedelta(days=30 * num)
            if unit.startswith("year"):
                return now + _dt.timedelta(days=365 * num)
            return None

        # Build reverse index: cleaned goal text -> all entry IDs mentioning it
        cleaned_to_eids: dict[str, list[UUID]] = {}
        for eid, texts in entry_id_to_goals.items():
            for t in texts:
                cleaned = t.strip().lower()
                if not cleaned:
                    continue
                cleaned_to_eids.setdefault(cleaned, []).append(eid)
        def _norm(vec: List[float]) -> float:
            return sum(v * v for v in vec) ** 0.5

        def _cosine(a: List[float], b: List[float]) -> float:
            denom = _norm(a) * _norm(b)
            return (sum(x * y for x, y in zip(a, b)) / denom) if denom else 0.0

        existing_emb: Optional[list[list[float]]] = None
        if existing_goal_texts:
            existing_emb = _embed([g.lower().strip() for g in existing_goal_texts])  # type: ignore

        out: dict[UUID, list[dict[str, Any]]] = {}
        for eid, goals in entry_id_to_goals.items():
            for goal_text in goals:
                cleaned = goal_text.strip()
                if not cleaned:
                    continue
                emb = _embed(cleaned)  # type: ignore
                is_dup = False
                if existing_emb:
                    if max(_cosine(emb, vec) for vec in existing_emb) > similarity_threshold:  # type: ignore
                        is_dup = True
                if is_dup:
                    continue
                related_ids_full = [str(x) for x in cleaned_to_eids.get(cleaned.lower(), [])]
                categories = _infer_categories(cleaned)
                time_limit = _parse_time_limit(cleaned)
                new_goal_dict: dict[str, Any] = {
                    "content": cleaned,
                    "ai_generated": True,
                    "category": categories,
                    "created_at": _dt.datetime.now(_dt.timezone.utc),
                    "progress_score": 0.0,
                    "related_entry_ids": related_ids_full or [str(eid)],
                }
                if time_limit:
                    new_goal_dict["time_limit"] = time_limit
                out.setdefault(eid, []).append(new_goal_dict)
                if existing_emb:
                    existing_emb.append(emb)  # type: ignore
                else:
                    existing_emb = [emb]  # type: ignore
        return out
