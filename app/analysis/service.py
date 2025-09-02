import datetime
from uuid import UUID
from typing import List, Dict, Set, Tuple, Any, Optional
from app.goals.models import Goal
from app.goals.schemas import GoalBase
from app.analysis.ai_providers.openai import OpenAIAIService
from sqlalchemy.orm import Session

from app.analysis.db import (
    bulk_upsert_journal_analyses,
    delete_journal_analyses_by_ids,
    get_connected_analysis,
    get_journal_analyses_by_ids,
    upsert_feedback,
    upsert_connected_analysis
)
from app.journals.db import get_user_journals, get_journal
from app.goals.db import create_goal, get_user_goals, update_goal
from app.journals.schemas import JournalEntryBase
from app.analysis.schemas import ConnectedAnalysisBase, FeedbackCreate, JournalAnalysisBase, JournalAnalysisCreate
from app.journals.models import JournalEntry
from app.analysis.models import ConnectedAnalysis, JournalAnalysis
from app.goals.schemas import GoalCreate, GoalUpdate
import re
from app.analysis.ai_providers.base import AIService

DEFAULT_FEEDBACK_TONE = "calm"
DEFAULT_PROMPT_TONE = "friendly"


def pick_ai_service(model: str) -> AIService:
    """
    Selects the appropriate AI service implementation based on the input model string.

    Args:
        model (str): The model identifier (e.g., "chatgpt", "local").

    Returns:
        AIService: An instance of the selected AI service.
    """
    # Force OpenAI provider for all inputs
    return OpenAIAIService()


def analyze_and_upsert_journals(
    db: Session,
    user_id: UUID,
    journals: List[JournalEntry],
    ai_service: AIService
) -> Tuple[List[JournalAnalysis], List[JournalAnalysis]]:
    """
    Analyzes journal entries using the provided AI service and stores new analysis results.

    Args:
        db (Session): SQLAlchemy session object.
        user_id (UUID): ID of the current user.
        journals (List[JournalEntry]): List of journal entries to process.
        ai_service (AIService): AI service used to analyze entries.

    Returns:
        Tuple containing:
            - All valid JournalAnalysis entries after processing.
            - Only the newly analyzed JournalAnalysis entries.
    """
    journal_ids = [j.id for j in journals]
    existing = get_journal_analyses_by_ids(db, user_id, journal_ids)

    newly_analyzed_schemas, matching, not_matching_ids = analyze_journals_helper(journals, existing, ai_service)

    if not_matching_ids:
        delete_journal_analyses_by_ids(db, user_id, not_matching_ids)
    if newly_analyzed_schemas:
        bulk_upsert_journal_analyses(db, user_id, newly_analyzed_schemas)

    newly_ids = [ja.journalId for ja in newly_analyzed_schemas]
    matching_ids = [ja.journal_id for ja in matching]
    all_valid_ids = matching_ids + newly_ids

    full_results = get_journal_analyses_by_ids(db, user_id, all_valid_ids)
    full_by_id = {j.journal_id: j for j in full_results}
    newly_analyzed = [full_by_id[jid] for jid in newly_ids if jid in full_by_id]

    return full_results, newly_analyzed


def analyze_journals_helper(
    journals: List[JournalEntry],
    existing: List[JournalAnalysis],
    ai_service: AIService,
) -> Tuple[List[JournalAnalysisCreate], List[JournalAnalysis], List[UUID]]:
    """
    Determines which journals require fresh analysis and returns the appropriate collections.

    Args:
        journals (List[JournalEntry]): All journals to assess.
        existing (List[JournalAnalysis]): Previously analyzed journals.
        ai_service (AIService): AI engine in use.

    Returns:
        Tuple containing:
            - JournalAnalysisCreate objects for journals requiring analysis.
            - Existing JournalAnalysis entries that match the current model.
            - Journal IDs needing reprocessing.
    """
    matching = []
    not_matching_ids = []

    updated_by_id: Dict[UUID, datetime.datetime] = {}
    for je in journals:
        if je.updated_date:
            updated_by_id[je.id] = je.updated_date

    def _as_aware(dt: datetime.datetime) -> datetime.datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    for j in existing:
        same_model = (j.model == ai_service.model_tag)
        is_stale = False
        upd = updated_by_id.get(j.journal_id)
        if upd is not None and j.analysis_date is not None:
            try:
                if _as_aware(upd) > _as_aware(j.analysis_date):
                    is_stale = True
            except Exception:
                is_stale = True

        if same_model and not is_stale:
            matching.append(j)
        else:
            not_matching_ids.append(j.journal_id)

    existing_ids = {j.journal_id for j in matching}
    newly_analyzed = []

    for journal in journals:
        if journal.id not in existing_ids:
            result = ai_service.analyze_entry(JournalEntryBase.model_validate(journal))
            newly_analyzed.append(result)

    return newly_analyzed, matching, not_matching_ids


def generate_connected_analysis(
    journal_analyses: List[JournalAnalysis],
    goals: List[Goal],
    ai_service: AIService,
) -> ConnectedAnalysis:
    """
    Produces a ConnectedAnalysis object from journal analyses and goals.

    Args:
        journal_analyses (List[JournalAnalysis]): Individual journal AI outputs.
        goals (List[Goal]): User's active goals.
        ai_service (AIService): AI service responsible for analysis.

    Returns:
        ConnectedAnalysis: Synthesized AI interpretation across all data.
    """
    analysis_data = [JournalAnalysisBase.model_validate(j) for j in journal_analyses]
    goals_data = [GoalBase.model_validate(g) for g in goals]
    return ai_service.analyze_connected(analysis_data, goals_data)


def give_feedback(
    connected_analysis: ConnectedAnalysis,
    ai_service: AIService,
    tone_style: str = DEFAULT_FEEDBACK_TONE,
) -> FeedbackCreate:
    """
    Generates AI feedback from the connected analysis.

    Args:
        connected_analysis (ConnectedAnalysis): Result from prior AI summarization.
        ai_service (AIService): AI system to generate feedback.
        tone_style (str): Tone to apply to feedback.

    Returns:
        FeedbackCreate: Feedback, reflective question, and motivational message.
    """
    feedback_content = ai_service.generate_feedback(
        ConnectedAnalysisBase.model_validate(connected_analysis), tone_style=tone_style
    )

    return FeedbackCreate(
        connected_analysis_id=connected_analysis.id,
        tone=tone_style,
        feedback=feedback_content["feedback"],
        reflective_question=feedback_content["reflectiveQuestion"],
        motivation=feedback_content["motivation"],
    )


def give_recommendations(
    connected_analysis: ConnectedAnalysis,
    ai_service: AIService,
    tone_style: str = DEFAULT_PROMPT_TONE,
) -> List[str]:
    """
    Generates journaling prompt suggestions from connected analysis.

    Args:
        connected_analysis (ConnectedAnalysis): Input data for AI prompt generation.
        ai_service (AIService): AI service to use.
        tone_style (str): Desired tone of the generated prompts.

    Returns:
        List[str]: A list of journaling prompts.
    """
    return ai_service.recommend_journaling_prompts(
        ConnectedAnalysisBase.model_validate(connected_analysis), tone_style=tone_style
    )


def create_goals_with_ai(
    goals: List[Goal],
    newly_analyzed_entries: List[JournalAnalysis],
    ai_service: AIService,
) -> List[GoalCreate]:
    """
    Generates new goals based on AI insights from journal entries.

    Args:
        goals (List[Goal]): Existing user goals.
        newly_analyzed_entries (List[JournalAnalysis]): Newly processed journals.
        ai_service (AIService): AI provider to extract goal suggestions.

    Returns:
        List[GoalCreate]: Valid AI-generated goals not yet in the system.
    """
    entry_goal_map = {
        entry.journal_id: (entry.goal_mentions or [])
        for entry in newly_analyzed_entries
    }

    existing_goal_texts = [g.content for g in goals]

    raw_goal_suggestions = ai_service.get_unique_goal_texts(
        existing_goal_texts, entry_goal_map
    )

    all_generated_goals = []
    for goal_list in raw_goal_suggestions.values():
        for goal_dict in goal_list:
            all_generated_goals.append(GoalCreate(**goal_dict))

    return all_generated_goals


def update_existing_goals_with_ai(
    db: Session,
    user_id: UUID,
    goals: List[Goal],
    analyses: List[JournalAnalysis],
) -> None:
    """Lightweight backend updates for existing goals based on analyzed journals.

    - Merge related_entry_ids with all journals where the goal text appears
    - Parse simple timeframes ("in 2 weeks/months/years") from goal content; latest mention wins
    - Update notes with mention counts and simple adherence signal
    """
    if not goals or not analyses:
        return

    # Map normalized goal text to Goal
    text_to_goal: dict[str, Goal] = {g.content.strip().lower(): g for g in goals}

    # Fuzzy match helper between goal text and mention text
    def _is_match(goal_text: str, mention_text: str) -> bool:
        gt = goal_text.strip().lower()
        mt = mention_text.strip().lower()
        if not gt or not mt:
            return False
        if gt == mt:
            return True
        if gt in mt or mt in gt:
            return True
        gt_tokens = {w for w in gt.replace("/", " ").split() if len(w) > 2}
        mt_tokens = {w for w in mt.replace("/", " ").split() if len(w) > 2}
        if not gt_tokens or not mt_tokens:
            return False
        overlap = len(gt_tokens & mt_tokens) / max(len(gt_tokens), 1)
        return overlap >= 0.5


    mentions: dict[str, List[tuple[UUID, str]]] = {}
    for a in analyses:
        gm = a.goal_mentions or []
        a_date_iso = a.date or (a.analysis_date.isoformat() if a.analysis_date else "")
        for t in gm:
            for g_text, g in text_to_goal.items():
                if _is_match(g_text, t):
                    mentions.setdefault(g_text, []).append((a.journal_id, a_date_iso))
                    break

    if not mentions:
        return

    duration_re = re.compile(r"(?P<num>\d+)\s*(?P<unit>day|days|week|weeks|month|months|year|years)", re.I)

    def parse_time_limit(text: str, ref_iso: str) -> Optional[datetime.datetime]:
        m = duration_re.search(text)
        if not m:
            return None
        num = int(m.group("num"))
        unit = m.group("unit").lower()
        try:
            base = datetime.datetime.fromisoformat(ref_iso.replace("Z", "+00:00")) if ref_iso else None
        except Exception:
            base = None
        if base is None:
            base = datetime.datetime.now(datetime.timezone.utc)
        if unit.startswith("day"):
            return base + datetime.timedelta(days=num)
        if unit.startswith("week"):
            return base + datetime.timedelta(weeks=num)
        if unit.startswith("month"):
            return base + datetime.timedelta(days=30 * num)
        if unit.startswith("year"):
            return base + datetime.timedelta(days=365 * num)
        return None


    ja_by_id: Dict[UUID, JournalAnalysis] = {a.journal_id: a for a in analyses}


    positive_keys = {"hope", "hopeful", "contentment", "calm", "trust", "determination", "positive"}
    negative_keys = {"anxiety", "anxious", "negative", "frustrated", "loneliness", "heaviness"}

    def mood_score(ja: JournalAnalysis) -> float:
        try:
            cm = ja.combined_mood or {}
            if cm:
                pos = sum(float(cm.get(k, 0.0)) for k in positive_keys)
                neg = sum(float(cm.get(k, 0.0)) for k in negative_keys)
                return round(pos - neg, 3)
            return float(ja.sentiment_score or 0.0)
        except Exception:
            return float(ja.sentiment_score or 0.0)

    for key, pairs in mentions.items():
        goal = text_to_goal[key]

        existing_ids = set(goal.related_entry_ids or [])
        new_ids = {str(jid) for jid, _ in pairs}
        combined_ids = list(existing_ids.union(new_ids))

        existing_notes = goal.notes or ""
        appended_lines: List[str] = []

        latest_iso_for_tl = None
        for jid, d in pairs:
            try:
                dtp = datetime.datetime.fromisoformat((d or "").replace("Z", "+00:00"))
                latest_iso_for_tl = dtp.isoformat()
                day = dtp.date().isoformat()
            except Exception:
                day = ""
            j = get_journal(db, jid, user_id)
            if j and j.content:
                content_text = j.content
                key_phrase = goal.content.strip()
                idx = content_text.lower().find(key_phrase.lower()) if key_phrase else -1
                if idx != -1:
                    start = max(0, idx - 60)
                    end = min(len(content_text), idx + len(key_phrase) + 60)
                    snippet = content_text[start:end].strip()
                else:
                    snippet = content_text.split(".")[:1][0].strip()
                line = f"- {day}: \"{snippet}\""
                if line not in existing_notes and line not in appended_lines:
                    appended_lines.append(line)

        tl = parse_time_limit(goal.content, latest_iso_for_tl or "")

        trend_samples: List[tuple[datetime.datetime, float]] = []
        for jid, d in pairs:
            ja = ja_by_id.get(jid)
            if not ja:
                continue
            try:
                dtp = datetime.datetime.fromisoformat((d or "").replace("Z", "+00:00")) if d else None
            except Exception:
                dtp = None
            if dtp is None:
                dtp = ja.analysis_date or datetime.datetime.now(datetime.timezone.utc)
            trend_samples.append((dtp, mood_score(ja)))
        trend_samples.sort(key=lambda x: x[0])
        emotion_trend_vals = [v for _, v in trend_samples] if trend_samples else None
        new_notes = (existing_notes + ("\n" if existing_notes and appended_lines else "") + "\n".join(appended_lines)).strip()

        payload = GoalUpdate(
            notes=new_notes if new_notes else None,
            related_entry_ids=combined_ids,
            time_limit=tl if tl else None,
            emotion_trend=emotion_trend_vals if emotion_trend_vals else None,
            updated_at=datetime.datetime.now(datetime.timezone.utc),
        )
        update_goal(db, goal.id, payload, user_id)


def full_analysis_pipeline(
    db: Session,
    user_id: UUID,
    ai_service: AIService,
    feedback_tone: str = DEFAULT_FEEDBACK_TONE,
    prompt_tone: str = DEFAULT_PROMPT_TONE,
) -> Dict[str, Any]:
    """
    Executes the full journaling analysis workflow: 
    analyzes journals, generates feedback, prompts, and AI goals.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): Current user ID.
        ai_service (AIService): Selected AI engine.
        feedback_tone (str): Desired tone for feedback message.
        prompt_tone (str): Desired tone for journaling prompts.

    Returns:
        Dict[str, Any]: Feedback object, journaling prompts, and new goals.
    """
    journals = get_user_journals(db, user_id, limit=30)
    goals = get_user_goals(db, user_id)

    all_journal_analyses, newly_analyzed_journals = analyze_and_upsert_journals(db, user_id, journals, ai_service)

    if len(newly_analyzed_journals) != 30:
        connected = generate_connected_analysis(all_journal_analyses, goals, ai_service)
        upsert_connected_analysis(db, connected, user_id)
    else:
        connected = get_connected_analysis(db, user_id)

    feedback = give_feedback(connected, ai_service, feedback_tone)
    upsert_feedback(db, feedback, user_id)
    prompts = give_recommendations(connected, ai_service, prompt_tone)

    new_goals = create_goals_with_ai(goals, newly_analyzed_journals, ai_service)
    current_goal_texts = {g.content for g in goals}
    saved_goals = []

    for goal in new_goals:
        if goal.content not in current_goal_texts:
            create_goal(db, goal, user_id)
            saved_goals.append(goal)

    return {
        "feedback": feedback,
        "prompts": prompts,
        "new_goals": saved_goals,
    }