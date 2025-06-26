from uuid import UUID
from typing import List, Dict
import logging

from fastapi import APIRouter, Depends, HTTPException, Security, status
from sqlalchemy.orm import Session

from app.auth.service import get_current_user_id
from app.core.database import get_db
from app.journals.schemas import (
    JournalEntryCreate,
    JournalEntryUpdate,
    JournalEntryBase,
)
from app.journals.db import (
    create_journal,
    delete_all_journals,
    get_journal,
    update_journal,
    delete_journal,
    get_user_journals,
)

router = APIRouter(prefix="/journals", tags=["Journals"])
logger = logging.getLogger(__name__)


@router.get(
    "/all",
    response_model=List[JournalEntryBase],
    summary="Get all journal entries",
    description="Retrieve a paginated list of all journal entries for the authenticated user.",
    responses={
        200: {"description": "Journal entries retrieved successfully."},
        401: {"description": "Unauthorized."},
        500: {"description": "Failed to retrieve journal entries."},
    },
)
def get_journals_route(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> List[JournalEntryBase]:
    try:
        return get_user_journals(db, user_id, skip, limit)
    except Exception as e:
        logger.error(f"Error fetching journals for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch journal entries")


@router.get(
    "/{journal_id}",
    response_model=JournalEntryBase,
    summary="Get a journal by ID",
    description="Retrieve a specific journal entry by its unique identifier.",
    responses={
        200: {"description": "Journal retrieved successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "Journal not found."},
        500: {"description": "Failed to retrieve journal."},
    },
)
def read_journal_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> JournalEntryBase:
    try:
        journal = get_journal(db, journal_id, user_id)
        if journal is None:
            raise HTTPException(status_code=404, detail="Journal not found")
        return journal
    except Exception as e:
        logger.error(f"Error retrieving journal {journal_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve journal")


@router.post(
    "",
    response_model=JournalEntryBase,
    summary="Create a new journal",
    description="Create a new journal entry for the authenticated user.",
    responses={
        200: {"description": "Journal created successfully."},
        401: {"description": "Unauthorized."},
        500: {"description": "Failed to create journal."},
    },
)
def create_journal_route(
    journal: JournalEntryCreate,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> JournalEntryBase:
    try:
        return create_journal(db, journal, user_id)
    except Exception as e:
        logger.error(f"Error creating journal for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create journal")


@router.put(
    "/{journal_id}",
    response_model=JournalEntryBase,
    summary="Update a journal entry",
    description="Update an existing journal entry by its ID.",
    responses={
        200: {"description": "Journal updated successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "Journal not found."},
        500: {"description": "Failed to update journal."},
    },
)
def update_journal_route(
    journal_id: UUID,
    journal: JournalEntryUpdate,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> JournalEntryBase:
    try:
        updated = update_journal(db, journal_id, journal, user_id)
        if updated is None:
            raise HTTPException(status_code=404, detail="Journal not found")
        return updated
    except Exception as e:
        logger.error(f"Error updating journal {journal_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update journal")


@router.delete(
    "/{journal_id}",
    response_model=Dict[str, str],
    summary="Delete a journal by ID",
    description="Delete a specific journal entry for the authenticated user.",
    responses={
        200: {"description": "Journal deleted successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "Journal not found."},
        500: {"description": "Failed to delete journal."},
    },
)
def delete_journal_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> Dict[str, str]:
    try:
        deleted = delete_journal(db, journal_id, user_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="Journal not found")
        return {"detail": "Journal deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting journal {journal_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete journal")


@router.delete(
    "/all",
    response_model=Dict[str, str],
    summary="Delete all journals",
    description="Permanently delete all journal entries for the authenticated user.",
    responses={
        200: {"description": "All journals deleted successfully."},
        401: {"description": "Unauthorized."},
        404: {"description": "No journals found."},
        500: {"description": "Failed to delete journals."},
    },
)
def delete_all_journals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Security(get_current_user_id),
) -> Dict[str, str]:
    try:
        deleted = delete_all_journals(db, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="No journals found to delete")
        return {"detail": "All journals deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting all journals for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete journals")
