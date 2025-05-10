from uuid import UUID
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth.service import get_current_user_id
from app.journals.schemas import (
    ConnectedAnalysisCreate,
    ConnectedAnalysisResponse,
    JournalAnalysisCreate,
    JournalAnalysisResponse,
    JournalEntryCreate,
    JournalEntryUpdate,
    JournalEntryResponse,
)
from app.journals.service import (
    create_journal,
    delete_all_journals,
    delete_connected_analysis,
    delete_journal_analysis,
    get_connected_analysis,
    get_journal,
    get_journal_analysis,
    update_journal,
    delete_journal,
    get_user_journals,
    upsert_connected_analysis,
    upsert_journal_analysis,
)
from app.core.database import get_db

router = APIRouter(prefix="/journals", tags=["Journals"])

# Journal Entries
@router.get("", response_model=List[JournalEntryResponse])
def get_journals_route(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> List[JournalEntryResponse]:
    return get_user_journals(db, user_id, skip, limit)

@router.get("/{journal_id}", response_model=JournalEntryResponse)
def read_journal_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> JournalEntryResponse:
    journal = get_journal(db, journal_id, user_id)
    if journal is None:
        raise HTTPException(status_code=404, detail="Journal not found")
    return journal

@router.post("", response_model=JournalEntryResponse)
def create_new_journal_route(
    journal: JournalEntryCreate,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> JournalEntryResponse:
    return create_journal(db, journal, user_id)

@router.put("/{journal_id}", response_model=JournalEntryResponse)
def update_journal_entry_route(
    journal_id: UUID,
    journal: JournalEntryUpdate,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> JournalEntryResponse:
    updated_journal = update_journal(db, journal_id, journal, user_id)
    if updated_journal is None:
        raise HTTPException(status_code=404, detail="Journal not found")
    return updated_journal

@router.delete("/{journal_id}", response_model=dict)
def delete_journal_entry_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> dict:
    deleted_journal = delete_journal(db, journal_id, user_id)
    if deleted_journal is None:
        raise HTTPException(status_code=404, detail="Journal not found")
    return {"detail": "Journal deleted successfully."}

@router.delete("/all", response_model=dict)
def delete_all_journals_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> dict:
    deleted_journals = delete_all_journals(db, user_id)
    if deleted_journals is None:
        raise HTTPException(status_code=404, detail="No journals found")
    return {"detail": "All journals deleted successfully."}


# Single Journal Analysis
@router.get("/journal-analysis/{journal_id}", response_model=JournalAnalysisResponse)
def read_journal_analysis_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> JournalAnalysisResponse:
    result = get_journal_analysis(db, journal_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Journal analysis not found")
    return result

@router.post("/journal-analysis/{journal_id}", response_model=JournalAnalysisResponse)
def upsert_journal_analysis_route(
    journal_id: UUID,
    data: JournalAnalysisCreate,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> JournalAnalysisResponse:
    result = upsert_journal_analysis(db, journal_id, data, user_id)
    return result

@router.delete("/journal-analysis/{journal_id}", response_model=dict)
def delete_journal_analysis_route(
    journal_id: UUID,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> dict:
    deleted_analysis = delete_journal_analysis(db, journal_id, user_id)
    if deleted_analysis is None:
        raise HTTPException(status_code=404, detail="Journal analysis not found")
    return {"detail": "Journal analysis deleted successfully."}


# Connected Analysis
@router.get("/connected-analysis", response_model=ConnectedAnalysisResponse)
def read_connected_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> ConnectedAnalysisResponse:
    result = get_connected_analysis(db, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Connected analysis not found")
    return result

@router.post("/connected-analysis", response_model=ConnectedAnalysisResponse)
def upsert_connected_analysis_route(
    data: ConnectedAnalysisCreate,
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> ConnectedAnalysisResponse:
    result = upsert_connected_analysis(db, data, user_id)
    return result

@router.delete("/connected-analysis", response_model=dict)
def delete_connected_analysis_route(
    db: Session = Depends(get_db),
    user_id: UUID = Depends(get_current_user_id),
) -> dict:
    deleted_analysis = delete_connected_analysis(db, user_id)
    if deleted_analysis is None:
        raise HTTPException(status_code=404, detail="Connected analysis not found")
    return {"detail": "Connected analysis deleted successfully."}
