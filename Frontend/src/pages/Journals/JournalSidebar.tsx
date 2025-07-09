import React, { useEffect, useRef, useCallback } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Sidebar from "../../components/layout/Sidebar";
import type { JournalEntryBase } from "@/types/journal";
import { Plus } from "lucide-react";
import { format } from "date-fns";

interface Props {
  journals: JournalEntryBase[];
  journalTitle?: string;
  isAddPage?: boolean;
  setPendingNav: (link: string) => void;
  hasUnsavedChanges: boolean;
}

const JournalSidebar: React.FC<Props> = ({
  journals,
  journalTitle,
  isAddPage = false,
  setPendingNav,
  hasUnsavedChanges,
}) => {
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Scroll to active item
  useEffect(() => {
    if (id && itemRefs.current[id]) {
      itemRefs.current[id]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [id, journals]);

  const handleClick = useCallback(
    (path: string) => {
      if (hasUnsavedChanges) setPendingNav(path);
      else navigate(path);
    },
    [hasUnsavedChanges, navigate, setPendingNav]
  );

  const sorted = React.useMemo(
    () =>
      [...journals].sort(
        (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
      ),
    [journals]
  );

  return (
    <Sidebar>
      <nav
        className="flex flex-col h-full"
        role="navigation"
        aria-label="Journal Navigation"
      >
        <header className="shrink-0 sticky top-0 bg-background z-10 px-4 pt-4 pb-2">
          <button
            onClick={() => handleClick("/journals")}
            className="text-sm text-primary hover:underline block mb-3"
          >
            ← Back
          </button>
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold uppercase tracking-wide text-muted-foreground">
              Journals
            </h2>
            {!isAddPage && (
              <button
                onClick={() => handleClick("/journals/add")}
                className="text-sm text-primary hover:underline flex items-center gap-1"
              >
                <Plus className="w-4 h-4" /> New
              </button>
            )}
          </div>
        </header>

        <section className="flex-1 overflow-y-auto px-4 pt-2 pb-4 space-y-2">
          {isAddPage && (
            <div className="px-4 py-3 rounded-xl border bg-background text-left shadow-inner border-dashed border-accent">
              <p className="font-medium text-sm text-muted-foreground italic truncate">
                {journalTitle?.trim() || "Untitled"}
              </p>
              <p className="text-xs mt-1 text-muted-foreground">
                {format(new Date().toISOString(), "MMMM d, yyyy")}
              </p>
            </div>
          )}

          {sorted.map((item) => {
            const active = item.id === id;
            const titleText =
              active && journalTitle?.trim()
                ? journalTitle.trim()
                : item.title || "Untitled";

            return (
              <div
                key={item.id}
                ref={(el) => (itemRefs.current[item.id] = el)}
                onClick={() => handleClick(`/journals/${item.id}`)}
                className={`cursor-pointer px-4 py-3 rounded-xl border shadow-sm transition-colors duration-200 ${
                  active
                    ? "bg-primary text-white border-primary"
                    : "bg-white text-foreground hover:bg-accent"
                }`}
              >
                <p
                  className={`text-sm font-medium truncate ${
                    active ? "text-white" : ""
                  }`}
                >
                  {titleText}
                </p>
                <p
                  className={`text-xs mt-1 ${
                    active ? "text-white/80" : "text-muted-foreground"
                  }`}
                >
                  {format(item.date, "MMMM d, yyyy")}
                </p>
              </div>
            );
          })}
        </section>
      </nav>
    </Sidebar>
  );
};

export default JournalSidebar;
