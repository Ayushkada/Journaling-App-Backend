import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { Search } from "lucide-react";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

import { getAllGoals, deleteGoal } from "@/lib/goalService";
import { GoalBase } from "@/types/goal";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { GoalSection } from "../../components/ui/GoalSection";

const Goals = () => {
  const [goals, setGoals] = useState<GoalBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [goalToDelete, setGoalToDelete] = useState<string | null>(null);

  const navigate = useNavigate();
  const [query, setQuery] = useState("");

  useEffect(() => {
    const fetchGoals = async () => {
      try {
        const data = await getAllGoals(0, 100);
        setGoals(data);
      } catch (err) {
        console.error("Failed to load goals:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchGoals();
  }, [navigate]);

  const handleDelete = async () => {
    if (!goalToDelete) return;
    try {
      await deleteGoal(goalToDelete);
      setGoals((prev) => prev.filter((g) => g.id !== goalToDelete));
    } catch (err) {
      console.error("Error deleting goal:", err);
    } finally {
      setGoalToDelete(null);
    }
  };

  const lowerQuery = query.toLowerCase();

  const aiSuggested = goals.filter(
    (g) =>
      g.ai_generated &&
      !g.verified &&
      (g.content.toLowerCase().includes(lowerQuery) ||
        g.notes?.toLowerCase().includes(lowerQuery))
  );

  const activeGoals = goals.filter(
    (g) =>
      g.verified &&
      g.progress_score < 100 &&
      !g.completed_at &&
      (g.content.toLowerCase().includes(lowerQuery) ||
        g.notes?.toLowerCase().includes(lowerQuery))
  );

  const completedGoals = goals.filter(
    (g) =>
      g.verified &&
      g.progress_score === 100 &&
      g.completed_at &&
      (g.content.toLowerCase().includes(lowerQuery) ||
        g.notes?.toLowerCase().includes(lowerQuery))
  );

  return (
    <div className="min-h-screen bg-background flex">
      <main className="flex-1 p-8" role="main">
        <div className="max-w-7xl mx-auto w-full">
          {loading ? (
            <div className="min-h-screen flex items-center justify-center px-4">
              <p>Loading…</p>
            </div>
          ) : (
            <>
              <header className="flex justify-between items-center mb-8">
                <div>
                  <h1 className="text-4xl font-bold text-foreground mb-1">
                    My Goals
                  </h1>
                  <p className="text-muted-foreground text-lg">
                    A personal space to set intentions and track progress
                  </p>
                </div>
                <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
                  <div className="relative w-full sm:w-48 md:w-48 lg:w-56 xl:w-80 2xl:w-80">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground w-4 h-4" />
                    <Input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search goals..."
                      className="pl-10"
                      aria-label="Search your goals"
                    />
                  </div>
                  <Button asChild className="w-full sm:w-auto">
                    <Link to="/goals/add" aria-label="Create new goal">
                      Create New Goal
                    </Link>
                  </Button>
                </div>
              </header>

              {aiSuggested.length > 0 && (
                <GoalSection
                  title="AI Suggested Goals"
                  goals={aiSuggested}
                  onDelete={(id) => setGoalToDelete(id)}
                />
              )}

              {activeGoals.length > 0 && (
                <GoalSection
                  title="Active Goals"
                  goals={activeGoals}
                  onDelete={(id) => setGoalToDelete(id)}
                />
              )}

              {completedGoals.length > 0 && (
                <GoalSection
                  title="Completed Goals"
                  goals={completedGoals}
                  onDelete={(id) => setGoalToDelete(id)}
                />
              )}

              {query &&
                aiSuggested.length +
                  activeGoals.length +
                  completedGoals.length ===
                  0 && (
                  <div className="text-center text-muted-foreground py-12 text-lg">
                    <p>No goals match your search.</p>
                  </div>
                )}

              {goals.length === 0 && !loading && (
                <div className="text-center py-16" role="alert">
                  <div className="text-6xl mb-4" aria-hidden="true">
                    🎯
                  </div>
                  <h3 className="text-2xl font-semibold text-foreground mb-2">
                    No goals yet
                  </h3>
                  <p className="text-muted-foreground mb-6">
                    Start setting meaningful goals to guide your personal growth
                  </p>
                  <Button asChild className="mt-4">
                    <Link to="/goals/add" aria-label="Start your first goal">
                      Start your first goal
                    </Link>
                  </Button>
                </div>
              )}

              <ConfirmDialog
                open={!!goalToDelete}
                onClose={() => setGoalToDelete(null)}
                onConfirm={handleDelete}
                title="Delete Goal?"
                message="Are you sure you want to delete this goal? This action cannot be undone."
                confirmMessage="Delete"
              />
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default Goals;
