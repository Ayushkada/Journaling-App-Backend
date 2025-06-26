from uuid import UUID
from typing import List, Dict
import logging

from fastapi import APIRouter, Depends, HTTPException, Security, status
from sqlalchemy.orm import Session

from app.auth.service import get_current_user_id
from app.goals.schemas import GoalCreate, GoalUpdate, GoalResponse
from app.goals.db import (
    create_goal,
    get_goal,
    update_goal,
    delete_goal,
    get_user_goals,
    delete_all_goals,
)
from app.core.database import get_db

router = APIRouter(prefix="/goals", tags=["Goals"])
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=List[GoalResponse],
    summary="Get all user goals",
    description="Retrieve all goals associated with the authenticated user. Supports pagination.",
    responses={
        200: {"description": "Goals retrieved successfully."},
        401: {"description": "Unauthorized."},
        500: {"description": "Failed to retrieve goals."},
    },
)
def read_user_goals_route(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> List[GoalResponse]:
    try:
        return get_user_goals(db, user_id, skip, limit)
    except Exception as e:
        logger.error(f"Failed to fetch goals for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve goals")


@router.get(
    "/{goal_id}",
    response_model=GoalResponse,
    summary="Get a specific goal",
    description="Retrieve a specific goal by its unique ID.",
    responses={
        200: {"description": "Goal retrieved successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "Goal not found."},
    },
)
def read_goal_route(
    goal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> GoalResponse:
    goal = get_goal(db, goal_id, user_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal


@router.post(
    "",
    response_model=GoalResponse,
    summary="Create a new goal",
    description="Create a new goal for the authenticated user.",
    responses={
        200: {"description": "Goal created successfully."},
        401: {"description": "Unauthorized."},
        500: {"description": "Goal creation failed."},
    },
)
def create_goal_route(
    goal: GoalCreate,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> GoalResponse:
    try:
        return create_goal(db, goal, user_id)
    except Exception as e:
        logger.error(f"Failed to create goal for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create goal")


@router.put(
    "/{goal_id}",
    response_model=GoalResponse,
    summary="Update an existing goal",
    description="Update the content or metadata of a goal by its ID.",
    responses={
        200: {"description": "Goal updated successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "Goal not found."},
        500: {"description": "Failed to update goal."},
    },
)
def update_goal_route(
    goal_id: UUID,
    goal: GoalUpdate,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> GoalResponse:
    try:
        updated = update_goal(db, goal_id, goal, user_id)
        if updated is None:
            raise HTTPException(status_code=404, detail="Goal not found")
        return updated
    except Exception as e:
        logger.error(f"Failed to update goal {goal_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update goal")


@router.delete(
    "/{goal_id}",
    response_model=Dict[str, str],
    summary="Delete a goal",
    description="Delete a specific goal by its ID.",
    responses={
        200: {"description": "Goal deleted successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "Goal not found."},
        500: {"description": "Failed to delete goal."},
    },
)
def delete_goal_route(
    goal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> Dict[str, str]:
    try:
        deleted = delete_goal(db, goal_id, user_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="Goal not found")
        return {"detail": "Goal deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete goal {goal_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete goal")


@router.delete(
    "/all",
    response_model=Dict[str, str],
    summary="Delete all user goals",
    description="Permanently delete all goals associated with the authenticated user.",
    responses={
        200: {"description": "All goals deleted successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "No goals found."},
        500: {"description": "Failed to delete goals."},
    },
)
def delete_all_goals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> Dict[str, str]:
    try:
        deleted = delete_all_goals(db, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="No goals found to delete")
        return {"detail": "All goals deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete all goals for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete goals")
