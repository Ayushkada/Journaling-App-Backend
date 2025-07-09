import { useEffect, useState, useMemo, useCallback } from "react";
import {
  useParams,
  useLocation,
  useNavigate,
  Outlet,
  Link,
} from "react-router-dom";
import JournalSidebarList from "@/pages/Journals/JournalSidebar";
import { getAllJournals } from "@/lib/journalService";
import { Button } from "@/components/ui/Button";
import type { JournalEntryBase } from "@/types/journal";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";

const JournalLayout: React.FC = () => {
  const { id } = useParams<{ id?: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const [journals, setJournals] = useState<JournalEntryBase[]>([]);
  const [journalTitle, setJournalTitle] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [pendingNav, setPendingNav] = useState<string | null>(null);

  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const isAddPage = useMemo(
    () => location.pathname === "/journals/add",
    [location.pathname]
  );

  const exists = useMemo(() => {
    return !id || journals.some((j) => j.id === id);
  }, [id, journals]);

  // Fetch journals on mount
  const fetchJournals = useCallback(async () => {
    setLoading(true);
    try {
      const all = await getAllJournals(0, 100);
      setJournals(all);
    } catch (e) {
      console.error("Failed to load journals", e);
      setError("Unable to load journals.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchJournals();
  }, [fetchJournals]);

  // Automatically navigate if no unsaved changes
  useEffect(() => {
    if (pendingNav && !hasUnsavedChanges) {
      navigate(pendingNav);
      setJournalTitle("");
      setPendingNav(null);
    }
  }, [pendingNav, hasUnsavedChanges, navigate]);

  const outletContext = useMemo(
    () => ({
      journals,
      setJournals,
      journalTitle,
      setJournalTitle,
      hasUnsavedChanges,
      setHasUnsavedChanges,
      setPendingNav,
    }),
    [journals, journalTitle, hasUnsavedChanges]
  );

  return (
    <>
      <JournalSidebarList
        journals={journals}
        isAddPage={isAddPage}
        journalTitle={journalTitle}
        hasUnsavedChanges={hasUnsavedChanges}
        setPendingNav={setPendingNav}
      />

      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <p>Loading journals...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : !exists ? (
          <div className="text-center text-muted-foreground text-lg py-12">
            <p>No journal found.</p>
            <Button asChild className="mt-4">
              <Link to="/journals/add">Create New Journal</Link>
            </Button>
          </div>
        ) : (
          <Outlet context={outletContext} />
        )}
      </div>

      <ConfirmDialog
        open={!!pendingNav && hasUnsavedChanges}
        onClose={() => {
          setPendingNav(null);
        }}
        onConfirm={() => {
          if (pendingNav) {
            navigate(pendingNav);
            setPendingNav(null);
            setHasUnsavedChanges(false);
            setJournalTitle("");
          }
        }}
        title="Discard Changes?"
        message="You have unsaved changes. Are you sure you want to leave this page? Changes will be lost."
        confirmMessage="Leave"
      />
    </>
  );
};

export default JournalLayout;
