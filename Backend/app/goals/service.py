from sqlalchemy.orm import Session
from app.goals.models import Goal
from app.goals.schemas import GoalCreate, GoalUpdate
from typing import List, Optional
from uuid import UUID, uuid4

def create_goal(db: Session, goal: GoalCreate, user_id: UUID) -> Goal:
    new_goal = Goal(
        id=uuid4(),
        user_id=user_id,
        content=goal.content,
        aiGenerated=goal.aiGenerated,
        category=goal.category,
        created_at=goal.created_at,
        completed_at=goal.completed_at,
        progress_score=goal.progress_score,
        emotion_trend=goal.emotion_trend,
        related_entry_ids=goal.related_entry_ids,
        time_limit=goal.time_limit,
        verified=goal.verified,
        notes=goal.notes,
        first_mensioned_at=goal.first_mensioned_at,
        last_mensioned_at=goal.last_mensioned_at,
    )
    db.add(new_goal)
    db.commit()
    db.refresh(new_goal)
    return new_goal

def get_goal(db: Session, goal_id: UUID, user_id: UUID) -> Optional[Goal]:
    return db.query(Goal).filter(
        Goal.id == goal_id,
        Goal.user_id == user_id
    ).first()

def get_user_goals(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[Goal]:
    return db.query(Goal).filter(Goal.user_id == user_id).offset(skip).limit(limit).all()

def update_goal(db: Session, goal_id: UUID, updated_goal: GoalUpdate, user_id: UUID) -> Optional[Goal]:
    goal = get_goal(db, goal_id, user_id)
    if goal:
        update_data = updated_goal.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(goal, field, value)
        db.commit()
        db.refresh(goal)
        return goal
    return None

def delete_goal(db: Session, goal_id: UUID, user_id: UUID) -> Optional[Goal]:
    goal = get_goal(db, goal_id, user_id)
    if goal:
        db.delete(goal)
        db.commit()
        return goal
    return None

def delete_all_goals(db: Session, user_id: UUID) -> List[Goal]:
    goals_to_delete = db.query(Goal).filter(Goal.user_id == user_id).all()
    db.query(Goal).filter(Goal.user_id == user_id).delete()
    db.commit()
    return goals_to_delete
