import datetime
from uuid import UUID, uuid4
from typing import Optional, List

from sqlalchemy.orm import Session
from app.journals.models import JournalEntry
from app.journals.schemas import JournalEntryCreate, JournalEntryUpdate


# Helper
def get_journal_entry_date(journal_entry) -> str:
    """
    Returns the date string for a journal entry. Defaults to today's date.

    Args:
        journal_entry (dict): Dictionary-like journal object (e.g. from JSON).

    Returns:
        str: ISO-formatted date.
    """
    return journal_entry.get("date") or str(datetime.date.today())


# Journal Entry CRUD
def get_journal(db: Session, journal_id: UUID, user_id: UUID) -> Optional[JournalEntry]:
    """
    Retrieves a journal entry by its ID for a given user.

    Args:
        db (Session): SQLAlchemy session.
        journal_id (UUID): ID of the journal.
        user_id (UUID): ID of the owner.

    Returns:
        Optional[JournalEntry]: The journal if found, else None.
    """
    return db.query(JournalEntry).filter(
        JournalEntry.id == journal_id,
        JournalEntry.user_id == user_id
    ).first()


def get_user_journals(db: Session, user_id: UUID, skip: int = 0, limit: int = 100) -> List[JournalEntry]:
    """
    Retrieves a paginated list of journal entries for a user, sorted by date descending.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): ID of the user.
        skip (int): Pagination offset.
        limit (int): Pagination limit.

    Returns:
        List[JournalEntry]: List of journal entries.
    """
    return (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == user_id)
        .order_by(JournalEntry.date.desc())  # ✅ consistent ordering
        .offset(skip)
        .limit(limit)
        .all()
    )



def create_journal(db: Session, journal: JournalEntryCreate, user_id: UUID) -> JournalEntry:
    """
    Creates a new journal entry for a user.

    Args:
        db (Session): SQLAlchemy session.
        journal (JournalEntryCreate): Pydantic journal input.
        user_id (UUID): ID of the user.

    Returns:
        JournalEntry: The created journal.
    """
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
        analysis_status="pending",  # Initial state
    )
    db.add(new_journal)
    db.commit()
    db.refresh(new_journal)
    return new_journal


def update_journal(db: Session, journal_id: UUID, updated_journal: JournalEntryUpdate, user_id: UUID) -> Optional[JournalEntry]:
    """
    Updates an existing journal entry for a user.

    Args:
        db (Session): SQLAlchemy session.
        journal_id (UUID): ID of the journal to update.
        updated_journal (JournalEntryUpdate): Pydantic update input.
        user_id (UUID): ID of the user.

    Returns:
        Optional[JournalEntry]: The updated journal or None if not found.
    """
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
    """
    Deletes a journal entry by ID for a user.

    Args:
        db (Session): SQLAlchemy session.
        journal_id (UUID): ID of the journal to delete.
        user_id (UUID): ID of the user.

    Returns:
        Optional[JournalEntry]: The deleted journal or None.
    """
    journal = get_journal(db, journal_id, user_id)
    if journal:
        db.delete(journal)
        db.commit()
        return journal
    return None


def delete_all_journals(db: Session, user_id: UUID) -> List[JournalEntry]:
    """
    Deletes all journal entries for a user.

    ⚠️ This operation is irreversible and deletes all content for the user.

    Args:
        db (Session): SQLAlchemy session.
        user_id (UUID): ID of the user.

    Returns:
        List[JournalEntry]: List of deleted journals (pre-deletion state).
    """
    deleted_journals = db.query(JournalEntry).filter(
        JournalEntry.user_id == user_id
    ).all()

    db.query(JournalEntry).filter(
        JournalEntry.user_id == user_id
    ).delete(synchronize_session=False)

    db.commit()
    return deleted_journals
