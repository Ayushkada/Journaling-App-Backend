import React, { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate, useOutletContext } from "react-router-dom";
import { Edit, Trash2 } from "lucide-react";
import {
  getJournalById,
  deleteJournal,
  getAllJournals,
} from "@/lib/journalService";
import type { JournalEntryBase } from "@/types/journal";
import { Button } from "@/components/ui/Button";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { format } from "date-fns";
import { getUpdatedText } from "@/utils/helpers";

type OutletCtx = {
  journals: JournalEntryBase[];
  setJournals: React.Dispatch<React.SetStateAction<JournalEntryBase[]>>;
};

const JournalView: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [journal, setJournal] = useState<JournalEntryBase | null>(null);
  const [loading, setLoading] = useState(true);
  const [toDelete, setToDelete] = useState<string | null>(null);

  const { journals, setJournals } = useOutletContext<OutletCtx>();

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [all, single] = await Promise.all([
        getAllJournals(),
        id ? getJournalById(id) : Promise.resolve(null),
      ]);
      setJournals(all);
      if (single) setJournal(single);
    } catch (e) {
      console.error("Fetch error", e);
    } finally {
      setLoading(false);
    }
  }, [id, setJournals]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const confirmDelete = useCallback(async () => {
    if (!toDelete) return;
    try {
      await deleteJournal(toDelete);
      const updated = journals.filter((j) => j.id !== toDelete);
      setJournals(updated);
      const next = updated
        .slice()
        .sort(
          (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
        )[0];
      navigate(next ? `/journals/${next.id}` : "/journals");
    } catch (e) {
      console.error("Delete error", e);
    } finally {
      setToDelete(null);
    }
  }, [toDelete, journals, navigate, setJournals]);

  return (
    <main className="bg-background flex flex-1 p-8" role="main">
      <article className="max-w-7xl mx-auto space-y-6 w-full">
        {loading && !journal ? (
          <div className="min-h-screen flex items-center justify-center px-4">
            <p>Loading…</p>
          </div>
        ) : (
          <>
            {/* Header */}
            <header className="flex flex-col sm:flex-row justify-between items-start gap-4">
              <div className="space-y-2">
                <h1 className="text-4xl font-bold">
                  {journal.title || "Untitled"}
                </h1>
                <div className="flex flex-wrap gap-4 text-muted-foreground text-sm">
                  <span>
                    Created:{" "}
                    {format(
                      new Date(journal.date),
                      "MMMM d, yyyy 'at' hh:mm a"
                    )}
                  </span>
                  {getUpdatedText(journal.date, journal.updated_date) && (
                    <span>
                      {getUpdatedText(journal.date, journal.updated_date)}
                    </span>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2 pt-1">
                <Button
                  size="icon"
                  variant="ghost"
                  onClick={() => navigate(`/journals/${journal.id}/edit`)}
                  aria-label="Edit journal entry"
                >
                  <Edit className="w-4 h-4" />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  className="text-destructive hover:text-destructive"
                  onClick={() => setToDelete(journal.id)}
                  aria-label="Delete journal entry"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            </header>

            {/* Emojis */}
            {journal.emojis?.length > 0 && (
              <section
                aria-label="Journal emojis"
                className="flex gap-2 text-2xl"
              >
                {journal.emojis.map((e, i) => (
                  <span key={i}>{e}</span>
                ))}
              </section>
            )}

            {/* Images */}
            {journal.images?.length > 0 && (
              <section
                aria-label="Journal images"
                className="flex overflow-x-auto space-x-4 py-2"
              >
                {journal.images.map((url, i) => (
                  <img
                    key={i}
                    src={url}
                    alt={`img-${i}`}
                    className="h-48 w-auto flex-shrink-0 rounded-lg object-cover"
                  />
                ))}
              </section>
            )}

            {/* Content */}
            <section
              aria-label="Journal content"
              className="bg-card border rounded-xl p-6 whitespace-pre-wrap break-words hyphens-auto"
            >
              {journal.content}
            </section>
          </>
        )}
      </article>

      <ConfirmDialog
        open={!!toDelete}
        onClose={() => setToDelete(null)}
        onConfirm={confirmDelete}
        title="Delete Journal Entry?"
        message="Are you sure you want to delete this journal entry? This action cannot be undone."
        confirmMessage="Delete"
      />
    </main>
  );
};

export default JournalView;
