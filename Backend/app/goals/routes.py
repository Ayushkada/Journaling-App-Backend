from uuid import UUID
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth.service import get_current_user_id
from app.goals.schemas import GoalCreate, GoalUpdate, GoalResponse
from app.goals.service import (
    create_goal,
    get_goal,
    update_goal,
    delete_goal,
    get_user_goals,
    delete_all_goals,
)
from app.core.database import get_db

router = APIRouter(
    prefix="/goals",
    tags=["Goals"]
)

@router.get("", response_model=List[GoalResponse])
def read_user_goals_route(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), user_id: UUID = Depends(get_current_user_id)) -> List[GoalResponse]:
    return get_user_goals(db, user_id, skip, limit)

@router.get("/{goal_id}", response_model=GoalResponse)
def read_goal_route(goal_id: UUID, db: Session = Depends(get_db), user_id: UUID = Depends(get_current_user_id)) -> GoalResponse:
    goal = get_goal(db, goal_id, user_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal

@router.post("", response_model=GoalResponse)
def create_new_goal_route(goal: GoalCreate, db: Session = Depends(get_db), user_id: UUID = Depends(get_current_user_id)) -> GoalResponse:
    return create_goal(db, goal, user_id)

@router.put("/{goal_id}", response_model=GoalResponse)
def update_goal_entry_route(goal_id: UUID, goal: GoalUpdate, db: Session = Depends(get_db), user_id: UUID = Depends(get_current_user_id)) -> GoalResponse:
    updated_goal = update_goal(db, goal_id, goal, user_id)
    if updated_goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return updated_goal

@router.delete("/{goal_id}", response_model=dict)
def delete_goal_entry_route(goal_id: UUID, db: Session = Depends(get_db), user_id: UUID = Depends(get_current_user_id)) -> dict:
    deleted_goal = delete_goal(db, goal_id, user_id)
    if deleted_goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return {"detail": "Goal deleted successfully."}

@router.delete("/all", response_model=dict)
def delete_all_goals_route(db: Session = Depends(get_db), user_id: UUID = Depends(get_current_user_id)) -> dict:
    deleted_goals = delete_all_goals(db, user_id)
    if deleted_goals is None:
        raise HTTPException(status_code=404, detail="No goals found")
    return {"detail": "All goals deleted successfully."}
