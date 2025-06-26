import datetime
from uuid import UUID
from typing import List, Dict, Set, Tuple, Any
from app.goals.models import Goal
from app.goals.schemas import GoalBase
from app.analysis.ai_providers.local import LocalAIService
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
from app.journals.db import get_user_journals
from app.goals.db import create_goal, get_user_goals
from app.journals.schemas import JournalEntryBase
from app.analysis.schemas import ConnectedAnalysisBase, FeedbackCreate, JournalAnalysisBase, JournalAnalysisCreate
from app.journals.models import JournalEntry
from app.analysis.models import ConnectedAnalysis, JournalAnalysis
from app.goals.db import GoalCreate
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
    match model:
        case "chatgpt":
            return OpenAIAIService()
        case _:
            return LocalAIService()


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

    for j in existing:
        if j.model == ai_service.model_tag:
            matching.append(j)
        else:
            not_matching_ids.append(j.journal_id)

    existing_ids = {j.journal_id for j in matching}
    newly_analyzed = []

    for journal in journals:
        if journal.id not in existing_ids:
            result = ai_service.analyze_entry(JournalEntryBase.from_orm(journal))
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
    analysis_data = [JournalAnalysisBase.from_orm(j) for j in journal_analyses]
    goals_data = [GoalBase.from_orm(g) for g in goals]
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
        ConnectedAnalysisBase.from_orm(connected_analysis), tone_style
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
        ConnectedAnalysisBase.from_orm(connected_analysis), tone_style
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
        entry.journal_id: entry.goalMentions or []
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