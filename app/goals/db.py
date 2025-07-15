from uuid import UUID, uuid4
from typing import List, Optional

from sqlalchemy.orm import Session
from app.goals.models import Goal
from app.goals.schemas import GoalCreate, GoalUpdate


def create_goal(db: Session, goal: GoalCreate, user_id: UUID) -> Goal:
    """
    Creates a new goal for the user.

    Args:
        db (Session): SQLAlchemy session.
        goal (GoalCreate): Input data for the goal.
        user_id (UUID): ID of the user.

    Returns:
        Goal: The created goal object.
    """
    new_goal = Goal(
        id=uuid4(),
        user_id=user_id,
        content=goal.content,
        ai_generated=goal.ai_generated,
        category=goal.category,
        created_at=goal.created_at,
        completed_at=goal.completed_at,
        progress_score=goal.progress_score,
        emotion_trend=goal.emotion_trend,
        related_entry_ids=goal.related_entry_ids,
        time_limit=goal.time_limit,
        verified=goal.verified,
        notes=goal.notes,
        first_mentioned_at=goal.first_mentioned_at,
        last_mentioned_at=goal.last_mentioned_at,
    )
    db.add(new_goal)
    db.commit()
    db.refresh(new_goal)
    return new_goal


def get_goal(db: Session, goal_id: UUID, user_id: UUID) -> Optional[Goal]:
    """
    Retrieves a goal by ID for the user.

    Args:
        db (Session): SQLAlchemy session.
        goal_id (UUID): ID of the goal.
        user_id (UUID): ID of the user.

    Returns:
        Optional[Goal]: The goal if found, else None.
    """
    return db.query(Goal).filter(
        Goal.id == goal_id,
        Goal.user_id == user_id
    ).first()


def get_user_goals(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[Goal]:
    """
    Retrieves all goals for a given user (paginated).

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): ID of the user.
        skip (int): Pagination offset.
        limit (int): Pagination limit.

    Returns:
        List[Goal]: List of goals.
    """
    return db.query(Goal).filter(
        Goal.user_id == user_id
    ).offset(skip).limit(limit).all()


def update_goal(db: Session, goal_id: UUID, updated_goal: GoalUpdate, user_id: UUID) -> Optional[Goal]:
    """
    Updates a specific goal for a user.

    Args:
        db (Session): SQLAlchemy session.
        goal_id (UUID): ID of the goal to update.
        updated_goal (GoalUpdate): Update payload.
        user_id (UUID): ID of the user.

    Returns:
        Optional[Goal]: Updated goal if successful, else None.
    """
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
    """
    Deletes a single goal by ID for the user.

    Args:
        db (Session): SQLAlchemy session.
        goal_id (UUID): ID of the goal.
        user_id (UUID): ID of the user.

    Returns:
        Optional[Goal]: Deleted goal or None.
    """
    goal = get_goal(db, goal_id, user_id)
    if goal:
        db.delete(goal)
        db.commit()
        return goal
    return None


def delete_all_goals(db: Session, user_id: UUID) -> List[Goal]:
    """
    Deletes all goals associated with the user.

    ⚠️ Irreversible operation.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): ID of the user.

    Returns:
        List[Goal]: List of deleted goal objects (from before deletion).
    """
    goals_to_delete = db.query(Goal).filter(
        Goal.user_id == user_id
    ).all()

    db.query(Goal).filter(
        Goal.user_id == user_id
    ).delete(synchronize_session=False)

    db.commit()
    return goals_to_delete
