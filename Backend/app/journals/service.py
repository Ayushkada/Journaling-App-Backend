import datetime
from uuid import UUID, uuid4
from typing import Dict, Optional, List

from sqlalchemy.orm import Session
from app.journals.models import ConnectedAnalysis, JournalAnalysis, JournalEntry
from app.journals.schemas import ConnectedAnalysisCreate, JournalEntryCreate, JournalEntryUpdate, JournalAnalysisCreate


# Helper
def get_journal_entry_date(journal_entry) -> str:
    return journal_entry.get("date") or str(datetime.date.today())


# Journal CRUD
def get_journal(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalEntry]:
    return db.query(JournalEntry).filter(
        JournalEntry.id == journal_id,
        JournalEntry.user_id == user_id
    ).first()

def get_user_journals(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[JournalEntry]:
    return db.query(JournalEntry).filter(
        JournalEntry.user_id == user_id
    ).offset(skip).limit(limit).all()

def create_journal(db: Session, journal: JournalEntryCreate, user_id: UUID) -> JournalEntry:
    new_journal = JournalEntry(
        id=uuid4(),
        user_id=user_id,
        title=journal.title,
        content=journal.content,
        date=journal.date,
        emojis=journal.emojis,
        images=journal.images,
        analyze_images=journal.analyze_images,
        source=journal.source,
        analysis_status="pending",
    )
    db.add(new_journal)
    db.commit()
    db.refresh(new_journal)
    return new_journal

def update_journal(db: Session, journal_id: UUID, updated_journal: JournalEntryUpdate, user_id: UUID) -> Optional[JournalEntry]:
    journal = get_journal(db, journal_id, user_id)
    if journal:
        update_data = updated_journal.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(journal, field, value)
        db.commit()
        db.refresh(journal)
        return journal
    return None

def delete_journal(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalEntry]:
    journal = get_journal(db, journal_id, user_id)
    if journal:
        db.delete(journal)
        db.commit()
        return journal
    return None

def delete_all_journals(db: Session, user_id: UUID) -> List[JournalEntry]:
    deleted_journals = db.query(JournalEntry).filter(
        JournalEntry.user_id == user_id
    ).all()
    db.query(JournalEntry).filter(
        JournalEntry.user_id == user_id
    ).delete()
    db.commit()
    return deleted_journals


# Single Journal Analysis CRUD
def get_journal_analysis(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalAnalysis]:
    return db.query(JournalAnalysis).filter(
        JournalAnalysis.journal_id == journal_id,
        JournalAnalysis.user_id == user_id
    ).first()

def upsert_journal_analysis(db: Session, journal_id: UUID, data: JournalAnalysisCreate, user_id: UUID) -> JournalAnalysis:
    existing = db.query(JournalAnalysis).filter(
        JournalAnalysis.journal_id == journal_id,
        JournalAnalysis.user_id == user_id
    ).first()

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

def delete_journal_analysis(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalAnalysis]:
    analysis = get_journal_analysis(db, journal_id, user_id)
    if analysis:
        db.delete(analysis)
        db.commit()
        return analysis
    return None


# Connected Analysis CRUD
def get_connected_analysis(db: Session, user_id: UUID) -> Optional[ConnectedAnalysis]:
    return db.query(ConnectedAnalysis).filter(
        ConnectedAnalysis.user_id == user_id
    ).first()

def upsert_connected_analysis(db: Session, data: ConnectedAnalysisCreate, user_id: UUID) -> ConnectedAnalysis:
    existing = db.query(ConnectedAnalysis).filter(
        ConnectedAnalysis.user_id == user_id
    ).first()

    update_data = data.dict(exclude_unset=True)

    if existing:
        for field, value in update_data.items():
            setattr(existing, field, value)
        existing.created_at = datetime.datetime.utcnow()
    else:
        existing = ConnectedAnalysis(**update_data)
        db.add(existing)

    db.commit()
    db.refresh(existing)
    return existing

def delete_connected_analysis(db: Session, user_id: UUID) -> Optional[ConnectedAnalysis]:
    analysis = get_connected_analysis(db, user_id)
    if analysis:
        db.delete(analysis)
        db.commit()
        return analysis
    return None


# Delete old analyses
def delete_old_journal_analyses(db: Session, user_id: UUID, keep_limit: int = 30) -> Dict[str, int]:
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
