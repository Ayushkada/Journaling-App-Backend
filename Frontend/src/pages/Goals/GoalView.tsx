import React, { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate, useOutletContext } from "react-router-dom";
import { Edit, Trash2 } from "lucide-react";
import { useIsMobile } from "@/hooks/useMobile";
import { Button } from "@/components/ui/Button";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { GoalBase } from "@/types/goal";
import type { JournalEntryBase } from "@/types/journal";
import { deleteGoal, getAllGoals, getGoalById } from "@/lib/goalService";
import { getAllJournals } from "@/lib/journalService";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { format, formatDistanceToNowStrict } from "date-fns";
import { getCategoryColor } from "@/utils/helpers";
import {
  baseCategoryColors,
  fallbackCategoryColors,
} from "@/config/categoryColors";

type OutletCtx = {
  goals: GoalBase[];
  setGoals: React.Dispatch<React.SetStateAction<GoalBase[]>>;
};

const GoalView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isMobile = useIsMobile();

  const [goal, setGoal] = useState<GoalBase | null>(null);
  const [journals, setJournals] = useState<JournalEntryBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [toDelete, setToDelete] = useState<string | null>(null);
  const [fallbackMap, setFallbackMap] = useState<Record<string, string>>({});

  const { goals, setGoals } = useOutletContext<OutletCtx>();

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [allGoals, single, allJournals] = await Promise.all([
        getAllGoals(),
        id ? getGoalById(id) : Promise.resolve(null),
        getAllJournals(),
      ]);
      setGoals(allGoals);
      setJournals(allJournals);
      if (single) setGoal(single);
    } catch (e) {
      console.error("Fetch error", e);
    } finally {
      setLoading(false);
    }
  }, [id, setGoals]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const confirmDelete = useCallback(async () => {
    if (!toDelete) return;
    try {
      await deleteGoal(toDelete);
      const updated = goals.filter((g) => g.id !== toDelete);
      setGoals(updated);
      const next = updated
        .slice()
        .sort(
          (a, b) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        )[0];
      navigate(next ? `/goals/${next.id}` : "/goals");
    } catch (e) {
      console.error("Delete error", e);
    } finally {
      setToDelete(null);
    }
  }, [toDelete, goals, navigate, setGoals]);

  if (loading || !goal) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <p>Loading…</p>
      </div>
    );
  }

  // find related journals
  const related = (goal.related_entry_ids || [])
    .map((jid) => journals.find((j) => j.id === jid))
    .filter(Boolean) as JournalEntryBase[];

  const getUpdatedText = (created: string, updated?: string): string | null => {
    if (!updated || updated === created) return null;

    const updatedTime = new Date(updated);
    const createdTime = new Date(created);
    const diffMs = updatedTime.getTime() - createdTime.getTime();
    const diffHours = diffMs / 36e5;

    if (diffHours < 24) {
      return `Updated ${formatDistanceToNowStrict(updatedTime, {
        addSuffix: true,
      })}`;
    } else {
      return `Updated: ${format(updatedTime, "MMMM d, yyyy 'at' hh:mm a")}`;
    }
  };

  return (
    <main className="bg-background flex flex-1 p-8" role="main">
      <article className="max-w-7xl mx-auto space-y-8 w-full">
        {loading ? (
          <div className="min-h-screen flex items-center justify-center px-4">
            <p>Loading…</p>
          </div>
        ) : (
          <>
            {/* Header */}
            <header className="space-y-2">
              <div className="space-y-0">
                <div className="flex flex-col sm:flex-row justify-between items-center">
                  <div className="flex flex-col space-y-2">
                    {goal.ai_generated || goal.category?.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {goal.ai_generated && (
                          <Badge
                            variant="outline"
                            className="px-2 py-1 text-xs"
                          >
                            AI-Generated
                          </Badge>
                        )}
                        {goal.category?.map((cat) => {
                          const colorClass = getCategoryColor(cat, {
                            fallbackMap,
                            setFallbackMap,
                          });

                          return (
                            <Badge
                              key={cat}
                              className={`px-2 py-1 text-xs ${colorClass}`}
                            >
                              {cat}
                            </Badge>
                          );
                        })}
                      </div>
                    ) : (
                      <h1 className="text-3xl font-bold">
                        {goal.content || "Untitled"}
                      </h1>
                    )}
                  </div>

                  <div className="flex items-center">
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => navigate(`/goals/${goal.id}/edit`)}
                      aria-label="Edit goal"
                    >
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      onClick={() => setToDelete(goal.id)}
                      aria-label="Delete goal"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>

                {(goal.ai_generated || goal.category?.length > 0) && (
                  <h1 className="text-3xl font-bold">
                    {goal.content || "Untitled"}
                  </h1>
                )}
              </div>

              <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
                <div className="text-muted-foreground text-sm">
                  <div>
                    Created:{" "}
                    {format(
                      new Date(goal.created_at),
                      "MMMM d, yyyy 'at' hh:mm a"
                    )}
                  </div>
                  {goal.completed_at ? (
                    <div>
                      Completed:{" "}
                      {format(goal.completed_at, "MMMM d, yyyy 'at' hh:mm a")}
                    </div>
                  ) : (
                    getUpdatedText(goal.created_at, goal.updated_at) && (
                      <div>
                        {getUpdatedText(goal.created_at, goal.updated_at)}
                      </div>
                    )
                  )}
                </div>
                <div className="text-muted-foreground text-sm">
                  {goal.time_limit && (
                    <div>
                      Due:{" "}
                      {format(
                        new Date(goal.time_limit),
                        "MMMM d, yyyy 'at' hh:mm a"
                      )}
                    </div>
                  )}
                </div>
              </div>
            </header>

            {/* Progress bar */}
            <section aria-label="Progress">
              <h2 className="font-semibold mb-2">Progress</h2>
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <Progress value={goal.progress_score} />
                  <p className="text-sm mt-1">
                    {goal.progress_score.toFixed(0)}%
                  </p>
                </div>
              </div>
            </section>

            {/* Notes */}
            {goal.notes && (
              <section>
                <h2 className="font-semibold mb-2">Notes</h2>
                <div
                  aria-label="Notes"
                  className="bg-card border rounded-xl p-4 whitespace-pre-wrap break-words"
                >
                  {goal.notes.split("\n").map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                </div>
              </section>
            )}

            {/* Related Journals */}
            {related.length > 0 && (
              <section aria-label="Related Journals">
                <h2 className="font-semibold mb-2">Related Journals</h2>
                <div className="flex flex-wrap gap-2 mb-2">
                  {related.map((j) => {
                    const title =
                      j.title?.trim() ||
                      format(new Date(j.date), "MMMM d, yyyy");
                    return (
                      <span
                        key={j.id}
                        className="inline-flex items-center text-xs font-medium px-2 py-1 rounded-full bg-primary text-primary-foreground max-w-[12rem]"
                      >
                        <a
                          href={`/journals/${j.id}`}
                          title={title}
                          className="truncate underline inline-block max-w-[8rem]"
                        >
                          {title.length > 50 ? `${title.slice(0, 50)}…` : title}
                        </a>
                      </span>
                    );
                  })}
                </div>
              </section>
            )}
          </>
        )}
      </article>

      <ConfirmDialog
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        onConfirm={confirmDelete}
        title="Delete Goal?"
        message="Are you sure you want to delete this goal? This action cannot be undone."
        confirmMessage="Delete"
      />
    </main>
  );
};

export default GoalView;
