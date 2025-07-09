import { useEffect, useState, useMemo, useCallback } from "react";
import {
  useParams,
  useLocation,
  useNavigate,
  Outlet,
  Link,
} from "react-router-dom";
import GoalSidebarList from "@/pages/Goals/GoalSidebar";
import { Button } from "@/components/ui/Button";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { GoalBase } from "@/types/goal";
import { getAllGoals } from "@/lib/goalService";
import { getAllJournals } from "@/lib/journalService";
import { JournalEntryBase } from "@/types/journal";

const GoalLayout: React.FC = () => {
  const { id } = useParams<{ id?: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const [goals, setGoals] = useState<GoalBase[]>([]);
  const [journals, setJournals] = useState<JournalEntryBase[]>([]);
  const [goalTitle, setGoalTitle] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [pendingNav, setPendingNav] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  const isAddPage = useMemo(
    () => location.pathname === "/goals/add",
    [location.pathname]
  );

  const exists = useMemo(() => {
    return !id || goals.some((j) => j.id === id);
  }, [id, goals]);

  const fetchGoals = useCallback(async () => {
    setLoading(true);
    try {
      const all = await getAllGoals();
      const allJournals = await getAllJournals();
      setGoals(all);
      setJournals(allJournals);
    } catch (e) {
      console.error("Failed to load goals or journals", e);
      setError("Unable to load goals.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGoals();
  }, [fetchGoals]);

  // Auto-navigate if no unsaved changes
  useEffect(() => {
    if (pendingNav && !hasUnsavedChanges) {
      navigate(pendingNav);
      setGoalTitle("");
      setPendingNav(null);
    }
  }, [pendingNav, hasUnsavedChanges, navigate]);

  const outletContext = useMemo(
    () => ({
      goals,
      setGoals,
      goalTitle,
      setGoalTitle,
      hasUnsavedChanges,
      setHasUnsavedChanges,
      setPendingNav,
      journals,
    }),
    [goals, goalTitle, hasUnsavedChanges, journals]
  );

  return (
    <>
      <GoalSidebarList
        goals={goals}
        isAddPage={isAddPage}
        goalTitle={goalTitle}
        hasUnsavedChanges={hasUnsavedChanges}
        setPendingNav={setPendingNav}
      />

      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <p>Loading goals...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : !exists ? (
          <div className="text-center text-muted-foreground text-lg py-12">
            <p>No goal found.</p>
            <Button asChild className="mt-4">
              <Link to="/goals/add">Create New Goal</Link>
            </Button>
          </div>
        ) : (
          <Outlet context={outletContext} />
        )}
      </div>

      <ConfirmDialog
        open={!!pendingNav && hasUnsavedChanges}
        onClose={() => setPendingNav(null)}
        onConfirm={() => {
          if (pendingNav) {
            navigate(pendingNav);
            setPendingNav(null);
            setHasUnsavedChanges(false);
            setGoalTitle("");
          }
        }}
        title="Discard Changes?"
        message="You have unsaved changes. Are you sure you want to leave this page? Changes will be lost."
        confirmMessage="Leave"
      />
    </>
  );
};

export default GoalLayout;
