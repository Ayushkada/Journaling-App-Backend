import datetime
import base64
import io
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
from uuid import UUID

import nltk
import spacy
import torch
import emoji
import textstat
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from app.journals.schemas import ConnectedAnalysisBase, JournalAnalysisBase
from app.goals.schemas import GoalBase
from app.journals.service import get_journal_entry_date
import app.analysis.local_prompts as local_prompts
from abc import ABC

class AIService(ABC):
    nltk.download("punkt")
    nlp = spacy.load("en_core_web_sm")
    model_tag: str

    def __init__(self):
        self.emotion_model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
        )
        self.t5_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.image_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.image_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

    def replace_emojis_and_extract(
        self, text: str, extra_emojis: List[str] = []
    ) -> Tuple[str, List[str]]:
        """Replace emojis in text and return cleaned text and emoji names."""
        new_content = emoji.demojize(text)
        content_emojis = [char for char in text if char in emoji.EMOJI_DATA]
        emoji_text_list = []

        for e in extra_emojis:
            if e in emoji.EMOJI_DATA:
                emoji_name = emoji.demojize(e).strip(":")
                emoji_text_list.extend([emoji_name] * 2)

        for e in content_emojis:
            emoji_name = emoji.demojize(e).strip(":")
            emoji_text_list.append(emoji_name)

        return new_content.strip(), emoji_text_list

    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """Analyze a base64-encoded image to return a caption and mood."""
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
            inputs = self.image_processor(image, return_tensors="pt")
            out = self.image_model.generate(**inputs)
            caption = self.image_processor.decode(out[0], skip_special_tokens=True)
            emotion_scores = self.emotion_model(caption)[0]
            image_mood = max(emotion_scores, key=lambda x: x["score"])["label"]
            return {"caption": caption, "imageMood": image_mood}
        except Exception:
            return {"caption": "Unable to process image", "imageMood": "unknown"}

    def analyze_entry(self, journal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single journal entry for sentiment, emotions, keywords, goals, etc."""
        # -- Parsing journal fields
        content = emoji.demojize(journal.get("content", "")).strip()
        emojis = journal.get("emojis", [])
        photos = journal.get("photos", [])
        analyze_images = journal.get("analyze_images", False)
        date = get_journal_entry_date(journal)

        # -- Sentiment and Self-talk
        sentiment_result = self.sentiment_model(content)[0]
        sentiment_score = (
            sentiment_result["score"]
            if sentiment_result["label"] == "POSITIVE"
            else -sentiment_result["score"]
        )

        sentences = [s for s in nltk.sent_tokenize(content) if "I " in s or "I'm" in s]
        self_talk_tone = "NEUTRAL"
        if sentences:
            combined = " ".join(sentences)
            self_talk_tone = self.sentiment_model(combined)[0]["label"]

        # -- Emotions
        text_emotion_scores = self.emotion_model(content)[0]
        text_mood = {e["label"]: round(e["score"], 3) for e in text_emotion_scores}

        # -- Emoji Mood
        emoji_text_str = " ".join(
            emoji.demojize(e).strip(":") for e in emojis if e in emoji.EMOJI_DATA
        )
        emoji_mood = {}
        if emoji_text_str:
            emoji_scores = self.emotion_model(emoji_text_str.strip())[0]
            emoji_mood = {e["label"]: round(e["score"], 3) for e in emoji_scores}

        # -- Combined Mood
        combined_mood = defaultdict(float)
        for mood in set(text_mood) | set(emoji_mood):
            t = text_mood.get(mood, 0)
            e = emoji_mood.get(mood, 0)
            combined_mood[mood] = round(min(1.0, 0.7 * t + 0.3 * e), 3)

        # -- Readability and Energy
        readability = round(textstat.flesch_reading_ease(content), 2)
        exclamations = content.count("!")
        long_words = len([w for w in content.split() if len(w) > 8])
        capital_words = len([w for w in content.split() if w.isupper() and len(w) > 2])
        word_count = len(content.split())
        energy_score = (exclamations + long_words + capital_words) / max(word_count, 1)

        # -- Keywords
        doc = nlp(content.lower())
        keywords = [
            token.lemma_ for token in doc if token.is_alpha and not token.is_stop
        ]
        top_keywords = dict(Counter(keywords).most_common(15))

        # -- Goal Extraction
        goal_prompt = f"Extract goals and timeframes from:\n{content}"
        goal_output = self.t5_model(goal_prompt, max_length=128, do_sample=False)[0][
            "generated_text"
        ].strip()
        goal_lines = [
            line.strip("-• ")
            for line in goal_output.split("\n")
            if len(line.strip()) > 4
        ]

        # -- Actions
        action_prompt = f"List any actions described:\n{content}"
        actions = self.t5_model(action_prompt, max_length=64, do_sample=False)[0][
            "generated_text"
        ].strip()

        # -- Image Analysis (optional)
        topics = []
        image_mood_scores = defaultdict(float)
        if analyze_images:
            for photo in photos:
                result = self.analyze_image(photo)
                topics.append(result)
                if result.get("imageMood") and result["imageMood"] != "unknown":
                    image_mood_scores[result["imageMood"]] += 0.1

        image_mood_scores = {k: round(v, 3) for k, v in image_mood_scores.items()}
        for mood, score in image_mood_scores.items():
            current = combined_mood.get(mood, 0.0)
            combined_mood[mood] = round(min(1.0, current + score), 3)

        # -- Semantic Embeddings
        text_vector_input = " ".join(top_keywords.keys()) + " " + goal_output
        text_embedding = self.embedding_model.encode(
            text_vector_input, convert_to_tensor=False
        ).tolist()

        return {
            "id": journal.get("id"),
            "analysisComplete": True,
            "analysisDate": datetime.datetime.now().isoformat(),
            "readability": readability,
            "sentimentScore": round(sentiment_score, 3),
            "selfTalkTone": self_talk_tone,
            "energyScore": round(energy_score, 3),
            "keywords": top_keywords,
            "textMood": text_mood,
            "emojiMood": emoji_mood,
            "imageMood": image_mood_scores,
            "mood": dict(combined_mood),
            "goalMentions": goal_lines,
            "topics": topics,
            "textVector": text_vector_input,
            "textEmbedding": text_embedding,
            "extractedActions": actions,
            "date": date,
            "model": self.model_tag
        }

    def analyze_connected(
        self, analyzed_journals: List[JournalAnalysisBase], goals: List[GoalBase]
    ) -> Dict[str, Any]:
        """Analyze mood trends, energy, goals, and keywords based on all journal analyses."""
        mood_trends: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        energy_trends: Dict[str, float] = {}
        sentiment_scores: List[float] = []
        journal_embeddings: List[List[float]] = []
        journal_ids: List[UUID] = []

        keyword_emotion_map: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        keyword_energy_map: Dict[str, List[float]] = defaultdict(list)

        # Weight recent journals more heavily
        sorted_entries = sorted(
            analyzed_journals, key=lambda x: (x.date or x.analysis_date).isoformat()
        )
        total = len(sorted_entries)
        entry_weights = {
            e.id: 0.5 + 0.5 * (i / max(total - 1, 1))
            for i, e in enumerate(sorted_entries)
        }

        for entry in analyzed_journals:
            date_str = (entry.date or entry.analysis_date).isoformat()[:10]
            weight = entry_weights.get(entry.id, 1.0)
            journal_ids.append(entry.id)

            # Mood and energy
            for emotion, score in entry.mood.items():
                mood_trends[date_str][emotion].append(score * weight)

            energy_trends[date_str] = entry.energy_score
            sentiment_scores.append(entry.sentiment_score * weight)

            # Keyword analysis
            for keyword in entry.keywords.keys():
                for emotion, score in entry.mood.items():
                    keyword_emotion_map[keyword][emotion].append(score * weight)
                keyword_energy_map[keyword].append(entry.energy_score * weight)

            journal_embeddings.append(entry.text_embedding)

        avg_sentiment = round(sum(sentiment_scores) / max(len(sentiment_scores), 1), 3)
        journal_embeddings_tensor = torch.tensor(journal_embeddings)

        # Goal Analysis
        goal_emotion_map: Dict[str, Dict[str, float]] = {}
        progress_report: Dict[str, Any] = {}
        goal_matches: Dict[str, List[UUID]] = {}

        for goal in goals:
            goal_text = f"{goal.content} {goal.notes or ''}".strip()
            goal_emb = self.embedding_model.encode(goal_text, convert_to_tensor=True)
            similarities = util.cos_sim(goal_emb, journal_embeddings_tensor)[0]

            related_moods = defaultdict(list)
            matched_journals = []

            for i, sim in enumerate(similarities):
                if sim.item() > 0.5:
                    matched_journals.append(journal_ids[i])
                    for emotion, score in analyzed_journals[i].mood.items():
                        related_moods[emotion].append(score)

            if matched_journals:
                goal_emotion_map[str(goal.id)] = {
                    e: round(sum(scores) / len(scores), 3)
                    for e, scores in related_moods.items()
                }

            goal_matches[str(goal.id)] = matched_journals
            performance_score = len(matched_journals) / max(len(journal_embeddings), 1)

            progress_report[str(goal.id)] = {
                "mentions": len(matched_journals),
                "performanceScore": round(performance_score, 3),
                "status": "on_track" if goal.progress_score >= 0.8 else "behind",
            }

        # Final aggregation
        mood_trend_summary = {
            date: {
                emotion: round(sum(scores) / len(scores), 3)
                for emotion, scores in emotions.items()
            }
            for date, emotions in mood_trends.items()
        }

        keyword_emotion_summary = {
            kw: {
                e: round(sum(scores) / len(scores), 3) for e, scores in emotions.items()
            }
            for kw, emotions in keyword_emotion_map.items()
        }

        keyword_energy_summary = {
            kw: round(sum(scores) / len(scores), 3)
            for kw, scores in keyword_energy_map.items()
        }

        return {
            "moodTrends": mood_trend_summary,
            "energyTrends": energy_trends,
            "averageSentiment": avg_sentiment,
            "goalEmotionMap": goal_emotion_map,
            "goalProgress": progress_report,
            "goalMatches": goal_matches,
            "keywordEmotionMap": keyword_emotion_summary,
            "keywordEnergyMap": keyword_energy_summary,
            "journalWeights": entry_weights,
            "model": self.model_tag
        }

    def generate_feedback(
        self, connected: ConnectedAnalysisBase, tone_style: str = "calm"
    ) -> Dict[str, str]:
        """Generate feedback summary from connected analysis trends, goal progress, and keyword insights."""
        recent_trends = list(connected.mood_trends.items())[-3:]
        mood_trend_summary = (
            "\n".join(
                f"{date}: "
                + ", ".join(
                    f"{emotion} {score:.2f}" for emotion, score in moods.items()
                )
                for date, moods in recent_trends
            )
            if recent_trends
            else ""
        )

        goal_progress_summary = (
            "\n".join(
                f"- Goal {goal_id} is {progress['status']} (Score {progress['performanceScore']})"
                for goal_id, progress in connected.goal_progress.items()
            )
            if connected.goal_progress
            else ""
        )

        keyword_flags = []
        for kw, energy in connected.keyword_energy_map.items():
            if energy < 0.3:
                keyword_flags.append(f"{kw} (low energy)")

        keyword_summary = "\n".join(keyword_flags)

        persona_prompt = local_prompts.TONE_TEMPLATES.get(
            tone_style.lower(), local_prompts.TONE_TEMPLATES["default"]
        )

        full_prompt = local_prompts.FEEDBACK_PROMPT_TEMPLATE.format(
            persona_prompt=persona_prompt,
            mood_str=mood_trend_summary,
            energy=connected.average_sentiment,
            tone="",
            top_keywords=", ".join(connected.keyword_emotion_map.keys()),
            mentioned_goals=", ".join(connected.goal_matches.keys()),
            actions="",
            mood_trend_summary=mood_trend_summary
            + "\n"
            + goal_progress_summary
            + "\n"
            + keyword_summary,
        )

        output = self.t5_model(full_prompt, max_length=384, do_sample=False)[0][
            "generated_text"
        ].strip()

        parts = output.split("Reflective Question:")
        feedback = (
            parts[0].replace("Feedback:", "").strip() if parts else output.strip()
        )
        question = local_prompts.DEFAULT_REFLECTIVE_QUESTION
        motivation = local_prompts.DEFAULT_MOTIVATION

        if len(parts) > 1:
            rest = parts[1].split("Motivation:")
            question = rest[0].strip()
            if len(rest) > 1:
                motivation = rest[1].strip()

        return {
            "feedback": feedback,
            "reflectiveQuestion": question,
            "motivation": motivation,
        }

    def recommend_journaling_prompts(
        self, connected: ConnectedAnalysisBase, tone_style: str = "friendly"
    ) -> List[str]:
        """Generate 2–3 custom journaling prompts from connected mood, goals, and keywords."""
        recent_trends = list(connected.mood_trends.items())[-3:]
        mood_str = (
            "; ".join(
                f"{date}: "
                + ", ".join(
                    f"{emotion} {score:.2f}" for emotion, score in moods.items()
                )
                for date, moods in recent_trends
            )
            if recent_trends
            else "No mood trends available"
        )

        top_keywords = ", ".join(connected.keyword_emotion_map.keys()) or "None"
        mentioned_goals = ", ".join(connected.goal_matches.keys()) or "None"
        average_sentiment = round(connected.average_sentiment, 3)

        persona_prompt = local_prompts.PROMPT_PERSONAS.get(
            tone_style.lower(), local_prompts.PROMPT_PERSONAS["default"]
        )

        full_prompt = local_prompts.PROMPT_SUGGESTION_TEMPLATE.format(
            persona_prompt=persona_prompt,
            mood_str=mood_str,
            energy=average_sentiment,
            tone="",
            mentioned_goals=mentioned_goals,
            top_keywords=top_keywords,
            actions="",
        )

        try:
            output = self.t5_model(full_prompt, max_length=128, do_sample=False)[0][
                "generated_text"
            ].strip()
            prompts = [
                line.strip("-• ")
                for line in output.split("\n")
                if len(line.strip()) > 3
            ]
            return prompts if prompts else local_prompts.DEFAULT_PROMPTS
        except Exception:
            return local_prompts.DEFAULT_PROMPTS

    def get_unique_goal_texts(
        self,
        existing_goal_texts: List[str],
        entry_id_to_goals: Dict[UUID, List[str]],
        entry_id_to_data: Optional[Dict[UUID, Dict[str, Any]]] = None,
        similarity_threshold: float = 0.75,
    ) -> Dict[UUID, List[Dict[str, Any]]]:
        """Return only new, non-duplicate goals with metadata enrichment."""
        existing_cleaned = [g.strip().lower() for g in existing_goal_texts]
        existing_embeddings = (
            self.embedding_model.encode(existing_cleaned, convert_to_tensor=True)
            if existing_cleaned
            else None
        )

        new_unique_goals: Dict[UUID, List[Dict[str, Any]]] = {}

        for entry_id, goal_texts in entry_id_to_goals.items():
            unique_goals: List[Dict[str, Any]] = []

            for goal_text in goal_texts:
                cleaned = goal_text.strip().lower()
                new_emb = self.embedding_model.encode(cleaned, convert_to_tensor=True)

                is_duplicate = False
                if existing_embeddings is not None:
                    sims = util.cos_sim(new_emb, existing_embeddings)[0]
                    if sims.max().item() > similarity_threshold:
                        is_duplicate = True

                if not is_duplicate:
                    entry_data = (
                        entry_id_to_data.get(entry_id) if entry_id_to_data else None
                    )

                    goal_metadata = {
                        "content": goal_text,
                        "aiGenerated": True,
                        "category": "AI Suggested",
                        "created_at": datetime.datetime.utcnow(),
                        "progress_score": None,
                        "emotion_trend": None,
                        "related_entry_ids": [entry_id],
                        "time_limit": None,
                        "verified": False,
                    }

                    if entry_data:
                        # (Optional) further enrich here based on journal analysis if needed
                        pass

                    unique_goals.append(goal_metadata)

                    # Expand similarity check memory
                    if existing_embeddings is not None:
                        existing_embeddings = torch.cat(
                            [existing_embeddings, new_emb.unsqueeze(0)], dim=0
                        )
                    else:
                        existing_embeddings = new_emb.unsqueeze(0)
                    existing_cleaned.append(cleaned)

            if unique_goals:
                new_unique_goals[entry_id] = unique_goals

        return new_unique_goals
