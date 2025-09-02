import datetime
from uuid import UUID
from typing import Dict, Optional, List, Set, Iterable, Tuple

from sqlalchemy.orm import Session
from app.journals.models import JournalEntry
from app.analysis.models import ConnectedAnalysis, Feedback, JournalAnalysis, PromptCatalog, UserPrompt, PromptInteraction
from app.analysis.schemas import ConnectedAnalysisCreate, FeedbackCreate, JournalAnalysisCreate, PromptCatalogCreate, UserPromptCreate, PromptInteractionCreate


def get_all_user_journal_analyses(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[JournalAnalysis]:
    """
    Retrieves all journal analyses for a user, paginated.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): ID of the user.
        skip (int): Offset for pagination.
        limit (int): Max number of results.

    Returns:
        List[JournalAnalysis]: The user's journal analyses.
    """
    return (
        db.query(JournalAnalysis)
        .join(JournalEntry, JournalEntry.id == JournalAnalysis.journal_id)
        .filter(JournalEntry.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_journal_analysis(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalAnalysis]:
    """
    Retrieves a specific journal analysis by journal ID and user ID.

    Args:
        db (Session): SQLAlchemy session.
        journal_id (UUID): Journal ID.
        user_id (UUID): ID of the user.

    Returns:
        Optional[JournalAnalysis]: The journal analysis or None.
    """
    return db.query(JournalAnalysis).filter(
        JournalAnalysis.journal_id == journal_id,
        JournalAnalysis.user_id == user_id
    ).first()


def get_journal_analyses_by_ids(db: Session, user_id: UUID, journal_ids: List[UUID]) -> List[JournalAnalysis]:
    """
    Retrieves multiple journal analyses by journal IDs for a specific user.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): ID of the user.
        journal_ids (List[UUID]): List of journal IDs.

    Returns:
        List[JournalAnalysis]: Matching journal analyses.
    """
    return (
        db.query(JournalAnalysis)
        .filter(
            JournalAnalysis.user_id == user_id,
            JournalAnalysis.journal_id.in_(journal_ids)
        )
        .all()
    )


def upsert_journal_analysis(db: Session, journal_id: UUID, data: JournalAnalysisCreate, user_id: UUID) -> JournalAnalysis:
    """
    Inserts or updates a single journal analysis.

    Args:
        db (Session): SQLAlchemy session.
        journal_id (UUID): Journal ID to upsert.
        data (JournalAnalysisCreate): New or updated data.
        user_id (UUID): ID of the user.

    Returns:
        JournalAnalysis: The updated or newly created object.
    """
    existing = get_journal_analysis(db, journal_id, user_id)
    update_data = data.dict(exclude_unset=True)

    if existing:
        for field, value in update_data.items():
            setattr(existing, field, value)
    else:
        update_data["journal_id"] = journal_id
        update_data["user_id"] = user_id
        existing = JournalAnalysis(**update_data)
        db.add(existing)

    db.commit()
    db.refresh(existing)
    return existing


def bulk_upsert_journal_analyses(db: Session, user_id: UUID, entries: List[JournalAnalysisCreate]):
    """
    Performs a bulk insert or update of journal analyses.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): User ID for ownership.
        entries (List[JournalAnalysisCreate]): List of analysis entries.
    """
    if not entries:
        return

    journal_ids = [entry.journal_id for entry in entries]

    existing_analyses = {
        a.journal_id: a
        for a in db.query(JournalAnalysis)
        .filter(
            JournalAnalysis.user_id == user_id,
            JournalAnalysis.journal_id.in_(journal_ids)
        )
        .all()
    }

    for schema in entries:
        journal_id = schema.journal_id
        update_data = schema.dict(exclude_unset=True)
        analysis = existing_analyses.get(journal_id)

        if analysis:
            for key, value in update_data.items():
                setattr(analysis, key, value)
        else:
            update_data["user_id"] = user_id
            db.add(JournalAnalysis(**update_data))

    db.commit()


def delete_journal_analysis(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalAnalysis]:
    """
    Deletes a journal analysis for a specific user and journal ID.

    Returns:
        Optional[JournalAnalysis]: Deleted object or None.
    """
    analysis = get_journal_analysis(db, journal_id, user_id)
    if analysis:
        db.delete(analysis)
        db.commit()
        return analysis
    return None


def delete_journal_analyses_by_ids(db: Session, user_id: UUID, journal_ids: List[UUID]) -> None:
    """
    Deletes multiple journal analyses by ID for a given user.

    Notes:
        Uses `synchronize_session=False` for performance.
    """
    db.query(JournalAnalysis).filter(
        JournalAnalysis.user_id == user_id,
        JournalAnalysis.journal_id.in_(journal_ids),
    ).delete(synchronize_session=False)
    db.commit()


def get_connected_analysis(db: Session, user_id: UUID) -> Optional[ConnectedAnalysis]:
    """
    Retrieves the user's most recent connected analysis.

    Returns:
        Optional[ConnectedAnalysis]: The connected analysis or None.
    """
    return db.query(ConnectedAnalysis).filter(
        ConnectedAnalysis.user_id == user_id
    ).first()


def upsert_connected_analysis(db: Session, data: ConnectedAnalysisCreate, user_id: UUID) -> ConnectedAnalysis:
    """
    Inserts or updates the user's connected analysis.
    """
    existing = get_connected_analysis(db, user_id)
    update_data = data.dict(exclude_unset=True)

    if existing:
        for field, value in update_data.items():
            setattr(existing, field, value)
        existing.created_at = datetime.datetime.now(datetime.timezone.utc)
    else:
        existing = ConnectedAnalysis(**update_data, user_id=user_id)
        db.add(existing)

    db.commit()
    db.refresh(existing)
    return existing


def delete_connected_analysis(db: Session, user_id: UUID) -> Optional[ConnectedAnalysis]:
    """
    Deletes the user's connected analysis if it exists.
    """
    analysis = get_connected_analysis(db, user_id)
    if analysis:
        db.delete(analysis)
        db.commit()
        return analysis
    return None


def delete_old_journal_analyses(db: Session, user_id: UUID, keep_limit: int = 30) -> Dict[str, int]:
    """
    Deletes the oldest journal analyses beyond the specified keep limit.

    Returns:
        Dict[str, int]: Count of deleted analyses.
    """
    analyses = (
        db.query(JournalAnalysis)
        .filter(JournalAnalysis.user_id == user_id)
        .order_by(JournalAnalysis.analysis_date.desc())
        .offset(keep_limit)
        .all()
    )
    for analysis in analyses:
        db.delete(analysis)
    db.commit()

    return {"deleted_count": len(analyses)}


def delete_old_connected_analyses(db: Session, user_id: UUID, keep_limit: int = 30) -> Dict[str, int]:
    """
    Deletes older connected analyses beyond the keep limit.

    Returns:
        Dict[str, int]: Count of deleted items.
    """
    analyses = (
        db.query(ConnectedAnalysis)
        .filter(ConnectedAnalysis.user_id == user_id)
        .order_by(ConnectedAnalysis.created_at.desc())
        .offset(keep_limit)
        .all()
    )
    for analysis in analyses:
        db.delete(analysis)
    db.commit()

    return {"deleted_count": len(analyses)}


def get_feedback(db: Session, user_id: UUID) -> Optional[Feedback]:
    """
    Retrieves the most recent feedback for the user.

    Returns:
        Optional[Feedback]: Feedback object or None.
    """
    return db.query(Feedback).filter(Feedback.user_id == user_id).first()


def upsert_feedback(db: Session, data: FeedbackCreate, user_id: UUID) -> Feedback:
    """
    Inserts or updates feedback for a connected analysis.
    """
    existing = db.query(Feedback).filter(
        Feedback.user_id == user_id,
        Feedback.connected_analysis_id == data.connected_analysis_id
    ).first()

    update_data = data.dict(exclude_unset=True)

    if existing:
        for field, value in update_data.items():
            setattr(existing, field, value)
        existing.created_at = datetime.datetime.now(datetime.timezone.utc)
    else:
        existing = Feedback(**update_data, user_id=user_id)
        db.add(existing)

    db.commit()
    db.refresh(existing)
    return existing


def delete_feedback(db: Session, user_id: UUID) -> Optional[Feedback]:
    """
    Deletes the user's feedback if present.
    """
    feedback = get_feedback(db, user_id)
    if feedback:
        db.delete(feedback)
        db.commit()
        return feedback
    return None


# -------------------- Prompt Catalog & User Prompts --------------------

def list_prompt_catalog(
    db: Session,
    *,
    limit: int = 10,
    tone: Optional[str] = None,
    exclude_texts: Optional[Set[str]] = None,
    topics: Optional[List[str]] = None,
    time_budget: Optional[float] = None,
) -> List[PromptCatalog]:
    q = db.query(PromptCatalog)
    if tone:
        q = q.filter(PromptCatalog.tone == tone)
    results = q.order_by(PromptCatalog.created_at.desc()).limit(limit * 4).all()
    # Client-side filter for topics/time as tags/time_estimate are JSON/float fields
    def _ok(p: PromptCatalog) -> bool:
        if exclude_texts and (p.text.strip().lower() in exclude_texts):
            return False
        if topics:
            tags = p.tags or []
            if not any(t.lower() in {tag.lower() for tag in tags} for t in topics):
                return False
        if time_budget is not None and p.time_estimate is not None:
            if p.time_estimate > time_budget:
                return False
        return True
    filtered = [p for p in results if _ok(p)]
    return filtered[:limit]


def create_prompt_catalog(db: Session, data: PromptCatalogCreate) -> PromptCatalog:
    obj = PromptCatalog(
        text=data.text,
        tone=data.tone,
        tags=data.tags,
        time_estimate=data.time_estimate,
        source=data.source,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def list_user_favorites(db: Session, user_id: UUID, skip: int = 0, limit: int = 50) -> List[UserPrompt]:
    return (
        db.query(UserPrompt)
        .filter(UserPrompt.user_id == user_id, UserPrompt.is_favorite == True)
        .order_by(UserPrompt.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_user_favorite(db: Session, user_id: UUID, data: UserPromptCreate) -> UserPrompt:
    # Deduplicate by text for user
    existing = (
        db.query(UserPrompt)
        .filter(UserPrompt.user_id == user_id, UserPrompt.text == data.text, UserPrompt.is_favorite == True)
        .first()
    )
    if existing:
        return existing
    obj = UserPrompt(
        user_id=user_id,
        catalog_id=data.catalog_id,
        text=data.text,
        tone=data.tone,
        tags=data.tags,
        source=data.source,
        is_favorite=True,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def delete_user_favorite(db: Session, user_id: UUID, favorite_id: UUID) -> Optional[UserPrompt]:
    fav = db.query(UserPrompt).filter(UserPrompt.id == favorite_id, UserPrompt.user_id == user_id, UserPrompt.is_favorite == True).first()
    if fav:
        db.delete(fav)
        db.commit()
        return fav
    return None


def record_prompt_interaction(db: Session, user_id: UUID, data: PromptInteractionCreate) -> PromptInteraction:
    obj = PromptInteraction(
        user_id=user_id,
        catalog_id=data.catalog_id,
        prompt_text=data.prompt_text,
        event=data.event,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_prompt_interaction_scores(
    db: Session,
    user_id: UUID,
    texts: Iterable[str],
) -> Dict[str, Dict[str, int]]:
    """Aggregate simple counts by event per prompt_text for a user."""
    text_list = [t.strip() for t in texts if t and t.strip()]
    if not text_list:
        return {}
    rows = (
        db.query(PromptInteraction.prompt_text, PromptInteraction.event)
        .filter(
            PromptInteraction.user_id == user_id,
            PromptInteraction.prompt_text.in_(text_list),
        )
        .all()
    )
    out: Dict[str, Dict[str, int]] = {}
    for prompt_text, event in rows:
        d = out.setdefault(prompt_text, {})
        d[event] = d.get(event, 0) + 1
    return out
