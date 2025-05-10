import datetime
from uuid import UUID
from typing import List, Dict, Tuple, Any
from app.analysis.local_ai_service import LocalAIService
from app.analysis.open_ai_service import OpenAIAIService
from sqlalchemy.orm import Session

from app.journals.service import (
    get_user_journals,
    upsert_journal_analysis,
    upsert_connected_analysis,
)
from app.goals.service import create_goal, get_user_goals
from app.journals.schemas import JournalAnalysisBase, JournalAnalysisCreate
from app.goals.service import GoalCreate
from app.analysis.ai_service import AIService

DEFAULT_FEEDBACK_TONE = "calm"
DEFAULT_PROMPT_TONE = "friendly"


def pick_ai_service(model: str | None) -> AIService:
    match model:
        case "chatgpt":
            return OpenAIAIService()
        case "local-large":
            return LocalAIService(large=True)
        case _:
            return LocalAIService()

def analyze_pending_journals(
    journals: List[Any], ai_service: AIService
) -> Tuple[List[JournalAnalysisBase], List[Tuple[UUID, JournalAnalysisCreate]]]:
    analyzed: List[JournalAnalysisBase] = []
    newly_analyzed: List[Tuple[UUID, JournalAnalysisCreate]] = []

    for journal in journals:
        if journal.analysis_status == "pending" or journal.model != ai_service.model_tag:
            result = ai_service.analyze_entry(journal.dict())
            analysis_data = JournalAnalysisCreate(**result)
            newly_analyzed.append((journal.id, analysis_data))
        else:
            analyzed.append(journal)

    return analyzed, newly_analyzed


def generate_connected_analysis(
    newly_analyzed: List[Tuple[UUID, Dict[str, Any]]],
    existing_analyses: Dict[UUID, Dict[str, Any]],
    goals: List[Any],
    ai_service: AIService,
) -> Dict[str, Any]:
    analyzed_journals = list(existing_analyses.values())
    all_analyzed = [r for _, r in newly_analyzed] + analyzed_journals
    return ai_service.analyze_connected(all_analyzed, goals)


def give_feedback(
    connected_analysis: Dict[str, Any],
    ai_service: AIService,
    tone_style: str = DEFAULT_FEEDBACK_TONE,
) -> Dict[str, str]:
    return ai_service.generate_feedback(connected_analysis, tone_style)


def give_recommendations(
    connected_analysis: Dict[str, Any],
    ai_service: AIService,
    tone_style: str = DEFAULT_PROMPT_TONE,
) -> List[str]:
    return ai_service.recommend_journaling_prompts(connected_analysis, tone_style)


def create_goals_with_ai(
    current_goal_texts: List[str],
    newly_analyzed_entries: Dict[UUID, Dict[str, Any]],
    ai_service: AIService,
) -> Dict[UUID, List[Dict[str, Any]]]:
    entry_goal_map = {entry_id: entry.get("goalMentions", []) for entry_id, entry in newly_analyzed_entries.items()}

    raw_goal_suggestions = ai_service.get_unique_goal_texts(current_goal_texts, entry_goal_map)

    generated_goals: Dict[UUID, List[Dict[str, Any]]] = {}
    for entry_id, goal_texts in raw_goal_suggestions.items():
        generated_goals[entry_id] = [
            {
                "content": goal_text,
                "aiGenerated": True,
                "category": "AI Suggested",
                "created_at": datetime.datetime.now(datetime.timezone.utc),
                "progress_score": None,
                "emotion_trend": None,
                "related_entry_ids": [entry_id],
                "time_limit": None,
                "verified": False,
            }
            for goal_text in goal_texts
        ]

    return generated_goals


def full_analysis_pipeline(
    db: Session,
    user_id: UUID,
    ai_service: AIService,
    feedback_tone: str = DEFAULT_FEEDBACK_TONE,
    prompt_tone: str = DEFAULT_PROMPT_TONE,
) -> Dict[str, Any]:
    journals = get_user_journals(db, user_id, limit=30)
    goals = get_user_goals(db, user_id)

    existing_analyses, newly_analyzed = analyze_pending_journals(journals, ai_service)
    for journal_id, result in newly_analyzed:
        upsert_journal_analysis(db, journal_id, result, user_id)

    if newly_analyzed:
        connected = generate_connected_analysis(newly_analyzed, existing_analyses, goals, ai_service)
        upsert_connected_analysis(db, connected, user_id)
    else:
        connected = existing_analyses

    feedback = give_feedback(connected, ai_service, feedback_tone)
    prompts = give_recommendations(connected, ai_service, prompt_tone)

    current_goal_texts = {goal.content for goal in goals if goal.content}
    new_goal_suggestions = create_goals_with_ai(list(current_goal_texts), dict(newly_analyzed), ai_service)

    for entry_id, goal_objs in new_goal_suggestions.items():
        for goal_data in goal_objs:
            if goal_data["content"] not in current_goal_texts:
                goal_obj = GoalCreate(**goal_data)
                create_goal(db, goal=goal_obj, user_id=user_id)

    return {"feedback": feedback, "prompts": prompts, "new_goals": new_goal_suggestions}
