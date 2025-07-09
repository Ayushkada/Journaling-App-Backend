import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { Link } from "react-router-dom";
import { Search, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { JournalCard } from "@/components/ui/JournalCard";
import { deleteJournal, getAllJournals } from "@/lib/journalService";
import type { JournalEntryBase } from "@/types/journal";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { format } from "date-fns";

const LIMIT = 20;

const Journals: React.FC = () => {
  const [journals, setJournals] = useState<JournalEntryBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [query, setQuery] = useState("");

  const observerRef = useRef<HTMLDivElement | null>(null);

  const fetchPage = useCallback(async () => {
    setLoading(true);
    try {
      const page = await getAllJournals(offset, LIMIT);
      setJournals((prev) => {
        const existingIds = new Set(prev.map((j) => j.id));
        return [...prev, ...page.filter((j) => !existingIds.has(j.id))];
      });
      setHasMore(page.length === LIMIT);
    } catch (e) {
      console.error("Load error", e);
    } finally {
      setLoading(false);
    }
  }, [offset]);

  useEffect(() => {
    fetchPage();
  }, [fetchPage]);

  // Infinite scroll
  useEffect(() => {
    if (loading || !hasMore) return;
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) setOffset((prev) => prev + LIMIT);
      },
      { threshold: 1.0 }
    );

    const el = observerRef.current;
    if (el) obs.observe(el);
    return () => {
      if (el) obs.unobserve(el);
    };
  }, [loading, hasMore]);

  const sorted = useMemo(
    () =>
      [...journals].sort(
        (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
      ),
    [journals]
  );

  const confirmDelete = useCallback(async () => {
    if (!deletingId) return;
    try {
      await deleteJournal(deletingId);
      setJournals((prev) => prev.filter((j) => j.id !== deletingId));
    } catch (e) {
      console.error("Delete error", e);
    } finally {
      setDeletingId(null);
    }
  }, [deletingId]);

  const filteredJournals = useMemo(() => {
    const lowerQuery = query.toLowerCase();
    return sorted.filter(
      (j) =>
        j.title?.toLowerCase().includes(lowerQuery) ||
        j.content.toLowerCase().includes(lowerQuery)
    );
  }, [query, sorted]);

  return (
    <main className="flex-1 p-8" role="main">
      <section className="max-w-7xl mx-auto w-full">
        {loading && journals.length === 0 ? (
          <div className="min-h-screen flex items-center justify-center px-4">
            <p>Loading…</p>
          </div>
        ) : (
          <>
            <header className="flex flex-col md:flex-row md:justify-between md:items-center gap-4 mb-8">
              <div>
                <h1 className="text-4xl font-bold text-foreground mb-1">
                  My Journals
                </h1>
                <p className="text-muted-foreground text-lg">
                  A personal space for reflection and growth
                </p>
              </div>
              <div className="flex flex-col sm:flex-row items-center gap-3 w-full sm:w-auto">
                <div className="relative w-full sm:w-48 md:w-48 lg:w-56 xl:w-64">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search journals..."
                    className="pl-10"
                    aria-label="Search entries"
                  />
                </div>
                <Button asChild className="w-full sm:w-auto">
                  <Link
                    to="/journals/add"
                    aria-label="Create new journal entry"
                  >
                    New Journal Entry
                  </Link>
                </Button>
              </div>
            </header>

            {journals.length === 0 && !loading ? (
              <div className="col-span-full text-center text-muted-foreground text-lg py-12">
                <p>No journal entries yet.</p>
                <Button asChild className="mt-4">
                  <Link
                    to="/journals/add"
                    aria-label="Start your first journal entry"
                  >
                    Start your first journal
                  </Link>
                </Button>
              </div>
            ) : filteredJournals.length === 0 ? (
              <div className="col-span-full text-center text-muted-foreground text-lg py-12">
                <p>No journals match your search.</p>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-2 gap-6 auto-rows-fr">
                  {filteredJournals.map((j) => (
                    <JournalCard
                      key={j.id}
                      id={j.id}
                      title={j.title || "Untitled"}
                      preview={`${j.content}`}
                      date={format(j.date, "MMMM d, yyyy")}
                      image={j.images?.[0] || null}
                      emojis={j.emojis || []}
                      viewUrl={`/journals/${j.id}`}
                      editUrl={`/journals/${j.id}/edit`}
                      onDelete={() => setDeletingId(j.id)}
                    />
                  ))}
                </div>

                {hasMore && (
                  <div
                    ref={observerRef}
                    className="w-full flex justify-center mt-6"
                  >
                    <Loader2 className="w-5 h-5 text-muted-foreground animate-spin" />
                  </div>
                )}
              </>
            )}

            <ConfirmDialog
              open={!!deletingId}
              onClose={() => setDeletingId(null)}
              onConfirm={confirmDelete}
              title="Delete Journal Entry?"
              message="Are you sure you want to delete this journal entry? This action cannot be undone."
              confirmMessage="Delete"
            />
          </>
        )}
      </section>
    </main>
  );
};

export default Journals;
