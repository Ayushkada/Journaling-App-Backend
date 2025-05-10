from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Dict, Any

from app.core.database import get_db
from app.auth.service import get_current_user_id
from app.analysis.analysis_service import (
    full_analysis_pipeline,
    analyze_pending_journals,
    generate_connected_analysis,
    give_feedback,
    give_recommendations,
    create_goals_with_ai,
    pick_ai_service,
)
from app.journals.service import (
    get_connected_analysis,
    get_user_journals,
    get_journal_analysis,
    upsert_connected_analysis,
    upsert_journal_analysis,
)
from app.goals.service import get_user_goals, create_goal
from app.goals.schemas import GoalCreate

router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.post("/run", response_model=Dict[str, Any])
def full_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    feedback_tone: str = Query("calm"),
    prompt_tone: str = Query("friendly"),
    model: str | None = Query(
        None,
        description="Engine: local-small | local-large | chatgpt"
    ),
):
    ai_service = pick_ai_service(model)
    return full_analysis_pipeline(db, user_id, ai_service, feedback_tone, prompt_tone)


@router.post("/journals", response_model=Dict[str, Any])
def analyze_pending_journals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    model: str | None = Query(None),
):
    ai_service = pick_ai_service(model)
    journals = get_user_journals(db, user_id, limit=30)
    existing, new = analyze_pending_journals(journals, ai_service)

    for journal_id, result in new:
        upsert_journal_analysis(db, journal_id, result, user_id)

    return {"newly_analyzed": dict(new), "existing_analyses": dict(existing)}


@router.post("/connected", response_model=Dict[str, Any])
def generate_connected_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    model: str | None = Query(None),
):
    ai_service = pick_ai_service(model)
    journals = get_user_journals(db, user_id, limit=30)
    goals = get_user_goals(db, user_id)

    existing = {
        journal.id: get_journal_analysis(db, journal.id, user_id)
        for journal in journals
        if get_journal_analysis(db, journal.id, user_id)
    }

    newly = analyze_pending_journals(journals, ai_service)
    connected = generate_connected_analysis(newly, existing, goals, ai_service)
    upsert_connected_analysis(db, connected, user_id)

    return {"connected": connected}


@router.get("/feedback", response_model=Dict[str, Any])
def get_feedback_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    tone: str = Query("calm"),
    model: str | None = Query(None),
):
    ai_service = pick_ai_service(model)
    connected = get_connected_analysis(db, user_id)
    if not connected:
        raise HTTPException(404, "No connected analysis found")
    return give_feedback(connected.analysis, ai_service, tone)


@router.get("/prompts", response_model=Dict[str, Any])
def get_prompts_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    tone: str = Query("friendly"),
    model: str | None = Query(None),
):
    ai_service = pick_ai_service(model)
    connected = get_connected_analysis(db, user_id)
    if not connected:
        raise HTTPException(404, "No connected analysis found")
    return give_recommendations(connected.analysis, ai_service, tone)


@router.post("/goals", response_model=Dict[str, Any])
def generate_ai_goals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
    model: str | None = Query(None),
):
    ai_service = pick_ai_service(model)
    journals = get_user_journals(db, user_id, limit=30)
    goals = get_user_goals(db, user_id)

    newly = analyze_pending_journals(journals, ai_service)
    current = [g.content for g in goals if g.content]
    suggestions = create_goals_with_ai(current, dict(newly), ai_service)

    for _, goal_objs in suggestions.items():
        for goal_data in goal_objs:
            if goal_data["content"] not in current:
                create_goal(db, GoalCreate(**goal_data), user_id)

    return {"new_goals": suggestions}
