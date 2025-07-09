import React, { useRef, useEffect, useState, useCallback } from "react";
import { useParams, useNavigate, useOutletContext } from "react-router-dom";
import { Edit, Save, Trash2 } from "lucide-react";
import Picker from "@emoji-mart/react";
import data from "@emoji-mart/data";
import type {
  JournalEntryCreate,
  JournalEntryBase,
  JournalEntryUpdate,
} from "@/types/journal";
import { useIsMobile } from "@/hooks/useMobile";
import {
  createJournal,
  deleteJournal,
  getJournalById,
  updateJournal,
} from "@/lib/journalService";
import { Button } from "@/components/ui/Button";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { format, formatDistanceToNowStrict } from "date-fns";
import { getUpdatedText, navigateWithUnsavedCheck } from "@/utils/helpers";

interface OutletContext {
  journals: JournalEntryBase[];
  setJournals: React.Dispatch<React.SetStateAction<JournalEntryBase[]>>;
  setJournalTitle: (title: string) => void;
  setHasUnsavedChanges: React.Dispatch<React.SetStateAction<boolean>>;
  setPendingNav: (link: string) => void;
  hasUnsavedChanges: boolean;
}

const JournalForm: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const isEditing = Boolean(id);
  const navigate = useNavigate();
  const isMobile = useIsMobile();

  const [loading, setLoading] = useState(isEditing);
  const [journal, setJournal] = useState<JournalEntryBase | null>(null);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [mood, setMood] = useState<string[]>([]);
  const [images, setImages] = useState<string[]>([]);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [journalToDelete, setJournalToDelete] = useState<string | null>(null);

  const contentRef = useRef<HTMLTextAreaElement>(null);
  const emojiPickerRef = useRef<HTMLDivElement>(null);

  const {
    journals,
    setJournals,
    setJournalTitle,
    setHasUnsavedChanges,
    setPendingNav,
    hasUnsavedChanges,
  } = useOutletContext<OutletContext>();

  // Auth & initial load
  useEffect(() => {
    if (isEditing && id) {
      getJournalById(id)
        .then((j) => {
          setJournal(j);
          setTitle(j.title || "");
          setContent(j.content);
          setMood(j.emojis || []);
          setImages(j.images || []);
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [isEditing, id]);

  // Sync title & dirty flag
  useEffect(() => {
    setJournalTitle(title);
    const isDirty =
      isEditing && journal
        ? title.trim() !== (journal.title || "").trim() ||
          content.trim() !== (journal.content || "").trim() ||
          JSON.stringify(mood) !== JSON.stringify(journal.emojis || []) ||
          JSON.stringify(images) !== JSON.stringify(journal.images || [])
        : !!title.trim() ||
          !!content.trim() ||
          mood.length > 0 ||
          images.length > 0;

    setHasUnsavedChanges(isDirty);
  }, [
    title,
    content,
    mood,
    images,
    isEditing,
    journal,
    setHasUnsavedChanges,
    setJournalTitle,
  ]);

  // Auto-expand textarea
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.style.height = "auto";
      contentRef.current.style.height = `${contentRef.current.scrollHeight}px`;
    }
  }, [content]);

  // Close emoji picker on outside click
  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (
        emojiPickerRef.current &&
        !emojiPickerRef.current.contains(e.target as Node)
      ) {
        setShowEmojiPicker(false);
      }
    };
    if (showEmojiPicker) document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [showEmojiPicker]);

  const handleImageUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      Promise.all(
        files.map(
          (file) =>
            new Promise<string>((resolve) => {
              const reader = new FileReader();
              reader.onload = () => resolve(reader.result as string);
              reader.readAsDataURL(file);
            })
        )
      ).then((newImgs) => setImages((prev) => [...prev, ...newImgs]));
      e.target.value = "";
    },
    []
  );

  const handleCancel = () => {
    setJournalTitle("");
    const fallback = [...journals].sort(
      (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
    )[0];
    navigateWithUnsavedCheck(
      isEditing
        ? `/journals/${id}`
        : fallback
        ? `/journals/${fallback.id}`
        : "/journals",
      hasUnsavedChanges,
      navigate,
      setPendingNav
    );
  };

  const handleDelete = async () => {
    if (!journalToDelete) return;
    try {
      await deleteJournal(journalToDelete);
      const updated = journals.filter((j) => j.id !== journalToDelete);
      setJournals(updated);
      const next = updated.sort(
        (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
      )[0];
      navigate(next ? `/journals/${next.id}` : "/journals");
    } catch (e) {
      console.error(e);
    } finally {
      setJournalToDelete(null);
      setJournalTitle("");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;

    const payload = {
      title: title || null,
      content,
      emojis: mood.length ? mood : null,
      images: images.length ? images : null,
    };

    try {
      if (isEditing && id) {
        const updatePayload: JournalEntryUpdate = {
          ...payload,
          updated_date: new Date().toISOString(),
        };
        const updated = await updateJournal(id, updatePayload);
        setJournals((prev) =>
          prev.map((j) => (j.id === id ? { ...j, ...updated } : j))
        );
        navigate(`/journals/${id}`);
      } else {
        const createPayload: JournalEntryCreate = {
          ...payload,
          date: new Date().toISOString(),
          source: "web",
        };
        const created = await createJournal(createPayload);
        setJournals((prev) => [created, ...prev]);
        navigate(`/journals/${created.id}`);
      }
    } catch (e) {
      console.error("Failed to save journal:", e);
    } finally {
      setJournalTitle("");
      setHasUnsavedChanges(false);
    }
  };

  return (
    <main className="min-h-screen bg-background flex" role="main">
      <section className="flex-1 p-8 w-full max-w-7xl mx-auto">
        {loading ? (
          <div className="min-h-screen flex items-center justify-center px-4">
            <p>Loading…</p>
          </div>
        ) : (
          <>
            <header className="flex flex-col sm:flex-row justify-between items-start gap-4">
              <div className="space-y-2">
                <h1 className="text-4xl font-bold">
                  {isEditing ? "Edit" : "New"} Journal Entry
                </h1>
                {isEditing && (
                  <div className="text-muted-foreground text-sm flex flex-wrap gap-x-4">
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
                )}
              </div>
              {isEditing && (
                <Button
                  size="icon"
                  variant="ghost"
                  className="text-destructive hover:text-destructive"
                  onClick={() => journal?.id && setJournalToDelete(journal.id)}
                  aria-label="Delete journal entry"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              )}
            </header>

            <form
              onSubmit={handleSubmit}
              className="bg-card rounded-xl shadow-lg border p-8 space-y-6 mt-4"
            >
              <section>
                <label className="block text-sm font-medium mb-2">Title</label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full px-4 py-3 border rounded-lg bg-background focus:ring-2 focus:ring-primary"
                  placeholder="Give your entry a title…"
                />
              </section>

              <section>
                <label className="block text-sm font-medium mb-2">
                  Your Thoughts *
                </label>
                <textarea
                  ref={contentRef}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onInput={() => {
                    if (contentRef.current) {
                      contentRef.current.style.height = "auto";
                      contentRef.current.style.height = `${contentRef.current.scrollHeight}px`;
                    }
                  }}
                  placeholder="Express yourself…"
                  className="w-full px-4 py-3 border rounded-lg bg-background resize-none overflow-y-auto max-h-[70vh] focus:ring-2 focus:ring-primary"
                />
              </section>

              <section className="relative" ref={emojiPickerRef}>
                <label className="block text-sm font-medium mb-2">
                  How are you feeling?
                </label>
                <input
                  type="text"
                  value={mood.join("")}
                  readOnly
                  onClick={() => setShowEmojiPicker(true)}
                  className="w-full px-4 py-3 border rounded-lg bg-background cursor-text focus:ring-2 focus:ring-primary"
                  placeholder="Select emojis"
                />
                {showEmojiPicker && (
                  <div className="absolute z-50 mt-2">
                    <Picker
                      data={data}
                      onEmojiSelect={(e) =>
                        setMood((prev) => [...prev, e.native])
                      }
                      theme="light"
                      previewPosition="none"
                    />
                  </div>
                )}
              </section>

              <section>
                <label className="block text-sm font-medium mb-2">
                  Add Images
                </label>
                <div className="border-2 border-dashed rounded-lg p-6 space-y-4 text-center hover:border-primary transition-colors">
                  {images.length ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {images.map((src, idx) => (
                        <div key={idx} className="relative">
                          <img
                            src={src}
                            alt="upload"
                            className="w-full h-48 object-cover rounded-lg"
                          />
                          <button
                            type="button"
                            className="absolute top-2 right-2 bg-destructive rounded-full w-6 h-6 flex items-center justify-center text-white text-sm"
                            onClick={() =>
                              setImages((imgs) =>
                                imgs.filter((_, i) => i !== idx)
                              )
                            }
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-muted-foreground">
                      No images uploaded.
                    </div>
                  )}
                  <div>
                    <p className="mb-4">Drag/drop or select images</p>
                    <input
                      id="upload"
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={handleImageUpload}
                    />
                    <label
                      htmlFor="upload"
                      className="bg-primary hover:bg-primary/90 text-primary-foreground px-4 py-2 rounded-lg cursor-pointer"
                    >
                      Choose Images
                    </label>
                  </div>
                </div>
              </section>

              <footer className="flex justify-between gap-4 pt-6">
                <button
                  type="button"
                  onClick={handleCancel}
                  className="px-4 py-2 border rounded"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!content.trim()}
                  className={`min-w-[128px] px-6 py-3 rounded-lg font-semibold flex items-center justify-center transition ${
                    !content.trim()
                      ? "bg-muted text-muted-foreground cursor-not-allowed opacity-60"
                      : "bg-primary hover:bg-primary/90 text-primary-foreground"
                  }`}
                >
                  <Save className="w-4 h-4 mr-2" />
                  {isEditing ? "Update Entry" : "Save Entry"}
                </button>
              </footer>
            </form>

            <ConfirmDialog
              open={!!journalToDelete}
              onClose={() => setJournalToDelete(null)}
              onConfirm={handleDelete}
              title="Delete Journal Entry?"
              message="Are you sure you want to delete this journal entry? This action
                      cannot be undone."
              confirmMessage="Delete"
            />
          </>
        )}
      </section>
    </main>
  );
};

export default JournalForm;
