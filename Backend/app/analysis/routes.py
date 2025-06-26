from app.analysis.schemas import ConnectedAnalysisBase, FeedbackBase, JournalAnalysisBase
from fastapi import APIRouter, Depends, Query, HTTPException, Security, status
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Dict, Any, List, Literal

import logging

logger = logging.getLogger(__name__)

from app.core.database import get_db
from app.auth.service import get_current_user_id
from app.analysis.service import (
    analyze_and_upsert_journals,
    full_analysis_pipeline,
    generate_connected_analysis,
    give_feedback,
    give_recommendations,
    create_goals_with_ai,
    pick_ai_service
)
from app.journals.db import get_user_journals
from app.analysis.db import (
    delete_connected_analysis,
    delete_journal_analysis,
    get_all_user_journal_analyses,
    get_connected_analysis,
    get_feedback,
    get_journal_analyses_by_ids,
    get_journal_analysis,
    upsert_connected_analysis,
    upsert_feedback
)
from app.goals.db import get_user_goals, create_goal

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post(
    "/run",
    response_model=Dict[str, Any],
    tags=["Analysis"],
    summary="Run full journal analysis pipeline",
    description="""
                Trigger a full analysis of the user's most recent journal entries using the selected AI engine. 
                The system will generate personalized feedback, journaling prompts, and goal suggestions.
                """,
    responses={
        200: {"description": "Analysis completed successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Internal server error - AI processing failed."},
    },
)
def full_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
    feedback_tone: str = Query("calm", description="Tone for AI-generated feedback."),
    prompt_tone: str = Query("friendly", description="Tone for journaling prompts."),
    model: Literal["local-small", "local-large", "chatgpt"] = Query(
        "local-small", description="Engine: local-small | local-large | chatgpt"
    ),
) -> Dict[str, Any]:
    try:
        logger.info(f"Running full analysis for user {user_id} with model {model}")
        ai_service = pick_ai_service(model)
        return full_analysis_pipeline(
            db, user_id, ai_service, feedback_tone, prompt_tone
        )
    except Exception as e:
        logger.error(f"Failed during full analysis for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to complete full analysis")




@router.post(
    "/journals",
    response_model=Dict[str, Any],
    tags=["Analysis"],
    summary="Analyze and store recent user journals",
    description="""
                Analyze the user's most recent journal entries using the selected AI engine.
                This endpoint returns newly generated analyses as well as existing ones for reuse.
                """,
    responses={
        200: {"description": "Journals analyzed and stored successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Internal server error during journal analysis."},
    },
)
def analyze_journals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
    model: Literal["local-small", "local-large", "chatgpt"] = Query(
        "local-small", description="Engine: local-small | local-large | chatgpt"
    ),
) -> Dict[str, Any]:
    logger.info(
        f"Analyzing journals for user {user_id} using model: {model or 'default'}"
    )

    try:
        ai_service = pick_ai_service(model)
        journals = get_user_journals(db, user_id, limit=30)
    except Exception as e:
        logger.error(
            f"Failed to fetch journals or initialize AI service for user {user_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to prepare journal analysis"
        )

    try:
        analyzed, _ = analyze_and_upsert_journals(db, user_id, journals, ai_service)
    except Exception as e:
        logger.error(f"Journal analysis pipeline failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze journals")

    return analyzed


@router.get(
    "/journals/all",
    response_model=List[JournalAnalysisBase],
    tags=["Analysis"],
    summary="Fetch all journal analyses",
    description="Return a paginated list of journal analyses for the authenticated user.",
    responses={
        200: {"description": "List of journal analyses returned successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Failed to retrieve journal analyses."},
    },
)
def get_journals_route(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> List[JournalAnalysisBase]:
    return get_all_user_journal_analyses(db, user_id, skip, limit)


@router.get(
    "/journals/{journal_id}",
    response_model=JournalAnalysisBase,
    tags=["Analysis"],
    summary="Fetch specific journal analysis",
    description="Retrieve a specific journal analysis by ID.",
    responses={
        200: {"description": "Journal analysis retrieved successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        404: {"description": "Journal analysis not found."},
        500: {"description": "Failed to retrieve journal analysis."},
    },
)
def read_journal_analysis_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> JournalAnalysisBase:
    result = get_journal_analysis(db, journal_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Journal analysis not found")
    return result


@router.delete(
    "/journals/{journal_id}",
    response_model=Dict[str, str],
    tags=["Analysis"],
    summary="Delete specific journal analysis",
    description="Delete a journal analysis for a given ID.",
    responses={
        200: {"description": "Journal analysis deleted successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        404: {"description": "Journal analysis not found."},
        500: {"description": "Failed to delete journal analysis."},
    },
)
def delete_journal_analysis_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> Dict[str, str]:
    deleted_analysis = delete_journal_analysis(db, journal_id, user_id)
    if deleted_analysis is None:
        raise HTTPException(status_code=404, detail="Journal analysis not found")
    return {"detail": f"Journal analysis deleted successfully. ID: {journal_id}"}




@router.post(
    "/connected",
    response_model=Dict[str, Any],
    tags=["Analysis"],
    summary="Generate connected analysis from existing journal analyses",
    description="""
                Generate a high-level connected analysis by examining relationships across the user's existing journal entries and goals.

                This endpoint **does not perform any new AI journal analysis.**  
                It only uses previously stored journal analyses to extract patterns, insights, and trends — provided that at least 3 analyzed entries exist.

                The system will also update the connected analysis record in the database for future use (e.g., in feedback or journaling prompts).
                """,
    responses={
        200: {"description": "Connected analysis generated successfully."},
        400: {"description": "Not enough journal analyses to perform connected analysis (minimum: 3)."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Internal server error during connected analysis."},
    },
)
def generate_connected_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
    model: Literal["local-small", "local-large", "chatgpt"] = Query(
        "local-small", description="Engine: local-small | local-large | chatgpt"
    ),
) -> Dict[str, Any]:
    try:
        ai_service = pick_ai_service(model)
        journals = get_user_journals(db, user_id, limit=30)

        journal_ids = [j.id for j in journals]
        existing = get_journal_analyses_by_ids(db, user_id, journal_ids)
        if len(existing) < 3:
            raise HTTPException(
                status_code=400,
                detail="At least 3 journal analyses are required for connected analysis.",
            )

        goals = get_user_goals(db, user_id)

        connected = generate_connected_analysis(existing, goals, ai_service)
        upsert_connected_analysis(db, connected, user_id)

        return {"connected_analysis": connected}

    except Exception as e:
        logger.error(
            f"Failed during connected analysis pipeline for user {user_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail="Failed to generate connected analysis."
        )


@router.get(
    "/connected",
    response_model=ConnectedAnalysisBase,
    tags=["Analysis"],
    summary="Retrieve connected analysis",
    description="Fetch the most recent connected analysis for the authenticated user.",
    responses={
        200: {"description": "Connected analysis retrieved successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        404: {"description": "Connected analysis not found."},
        500: {"description": "Failed to retrieve connected analysis."},
    },
)
def read_connected_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> ConnectedAnalysisBase:
    result = get_connected_analysis(db, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Connected analysis not found")
    return result


@router.delete(
    "/connected",
    response_model=Dict[str, str],
    tags=["Analysis"],
    summary="Delete connected analysis",
    description="Delete the current connected analysis for the authenticated user.",
    responses={
        200: {"description": "Connected analysis deleted successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        404: {"description": "Connected analysis not found."},
        500: {"description": "Failed to delete connected analysis."},
    },
)
def delete_connected_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> Dict[str, str]:
    deleted_analysis = delete_connected_analysis(db, user_id)
    if deleted_analysis is None:
        raise HTTPException(status_code=404, detail="Connected analysis not found")
    return {"detail": "Connected analysis deleted successfully."}




@router.post(
    "/feedback",
    response_model=Dict[str, Any],
    tags=["Analysis"],
    summary="Generate personalized feedback from connected analysis",
    description="""
                Use the most recent connected analysis to generate AI-based personal feedback for the user.

                This endpoint does **not perform any new journal or connected analysis.**  
                It simply extracts reflective feedback based on the stored connected analysis, which must already exist in the database.

                You can control the tone of the feedback via the `tone` query parameter (e.g., calm, supportive, direct).

                > Note: This endpoint requires a connected analysis to exist. You can generate one by calling `/analysis/connected`.
                """,
    responses={
        200: {"description": "Feedback generated successfully."},
        404: {"description": "No connected analysis found for this user."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Internal server error while generating feedback."},
    },
)
def create_feedback_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
    tone: str = Query(
        "calm", description="Tone for feedback (e.g., calm, supportive, direct)"
    ),
    model: Literal["local-small", "local-large", "chatgpt"] = Query(
        "local-small", description="Engine to use for feedback generation"
    ),
) -> Dict[str, Any]:
    try:
        ai_service = pick_ai_service(model)
        connected = get_connected_analysis(db, user_id)
        if not connected:
            raise HTTPException(status_code=404, detail="No connected analysis found")
        feedback = give_feedback(connected.analysis, ai_service, tone)
        upsert_feedback(db, feedback, user_id)
        return {
            "feedback": feedback.feedback,
            "reflective_question": feedback.reflective_question,
            "motivation": feedback.motivation,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback generation failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feedback")


@router.get(
    "/feedback",
    response_model=FeedbackBase,
    tags=["Analysis"],
    summary="Retrieve most recent feedback",
    description="""
                Fetch the latest AI-generated feedback for the authenticated user.

                This endpoint retrieves stored feedback that was previously generated from a connected analysis.  
                If no feedback exists, a 404 error will be returned.

                > To generate feedback, use the `/analysis/feedback/create` endpoint.
                """,
    responses={
        200: {"description": "Feedback retrieved successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        404: {"description": "No feedback found for this user."},
        500: {"description": "Failed to retrieve feedback from database."},
    },
)
def get_feedback_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> FeedbackBase:
    try:
        feedback = get_feedback(db, user_id)
        if not feedback:
            raise HTTPException(status_code=404, detail="No feedback found")
        return FeedbackBase.from_orm(feedback)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve feedback for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve feedback")




@router.post(
    "/prompts",
    response_model=Dict[str, Any],
    tags=["Analysis"],
    summary="Generate journaling prompts from connected analysis",
    description="""
                Use the latest connected analysis to generate personalized journaling prompts for the user.

                This endpoint does **not perform any new journal analysis or connected analysis.**  
                It extracts thoughtful and tone-specific prompts based on the previously stored connected analysis.

                The `tone` query parameter allows customization of the prompt style (e.g., friendly, reflective, motivational).

                > Note: This endpoint requires a connected analysis to exist. You can generate one by calling `/analysis/connected`.
                """,
    responses={
        200: {"description": "Journaling prompts generated successfully."},
        404: {"description": "No connected analysis found for this user."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Internal server error while generating prompts."},
    },
)
def create_prompts_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
    tone: str = Query(
        "friendly",
        description="Tone for prompts (e.g., friendly, reflective, motivational)",
    ),
    model: Literal["local-small", "local-large", "chatgpt"] = Query(
        "local-small", description="Engine to use for prompt generation"
    ),
) -> Dict[str, Any]:
    try:
        ai_service = pick_ai_service(model)
        connected = get_connected_analysis(db, user_id)
        if not connected:
            raise HTTPException(status_code=404, detail="No connected analysis found")
        return give_recommendations(connected.analysis, ai_service, tone)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prompt generation failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate prompts")


@router.post(
    "/goals",
    response_model=Dict[str, Any],
    tags=["Analysis"],
    summary="Generate personalized goals from recent journal entries",
    description="""
                Use the user's recent journal entries and current goals to generate **AI-suggested goals**.

                This endpoint analyzes journals that have not yet been processed by the selected AI engine and identifies potential goals mentioned within them.  
                It then compares the suggested goals with the user's existing goals to avoid duplication and stores only **new, unique goals** in the database.

                > Note: This endpoint performs **new journal analysis**, but only for entries that haven't yet been processed by the selected model.
                """,
    responses={
        200: {"description": "Goals generated and saved successfully."},
        401: {"description": "Unauthorized - invalid or missing credentials."},
        500: {"description": "Internal server error while generating goals."},
    },
)
def generate_ai_goals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
    model: Literal["local-small", "local-large", "chatgpt"] = Query(
        "local-small", description="Engine: local-small | local-large | chatgpt"
    ),
) -> Dict[str, Any]:
    try:
        ai_service = pick_ai_service(model)
        journals = get_user_journals(db, user_id, limit=30)
        goals = get_user_goals(db, user_id)
        _, newly_analyzed = analyze_and_upsert_journals(
            db, user_id, journals, ai_service
        )
        new_goals = create_goals_with_ai(goals, newly_analyzed, ai_service)

        current_goal_texts = {g.content for g in goals}
        saved_goals = []

        for goal in new_goals:
            if goal.content not in current_goal_texts:
                create_goal(db, goal, user_id)
                saved_goals.append(goal)

        return {"new_goals": saved_goals}
    except Exception as e:
        logger.error(f"Goal generation failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate goals")
