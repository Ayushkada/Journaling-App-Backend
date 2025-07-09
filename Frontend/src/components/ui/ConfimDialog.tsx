import { Button } from "@/components/ui/Button";
import { Dialog, DialogPanel, DialogTitle } from "@headlessui/react";
import { useEffect, useState } from "react";

interface ConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmMessage?: string;
  requiresInputConfirmation?: boolean;
  requiredText?: string;
}

export const ConfirmDialog = ({
  open,
  onClose,
  onConfirm,
  title,
  message,
  confirmMessage = "Confirm",
  requiresInputConfirmation = false,
  requiredText = "DELETE",
}: ConfirmDialogProps) => {
  const [input, setInput] = useState("");

  useEffect(() => {
    if (!open) setInput("");
  }, [open]);

  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div
        className="fixed inset-0 bg-black/30 backdrop-blur-sm"
        aria-hidden="true"
      />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="bg-white dark:bg-zinc-900 p-6 rounded-xl shadow-xl max-w-md w-full">
          <DialogTitle className="text-lg font-semibold text-foreground mb-2">
            {title}
          </DialogTitle>

          <p className="text-sm text-muted-foreground mb-4">{message}</p>

          {requiresInputConfirmation && (
            <div className="mb-4">
              <input
                type="text"
                placeholder={`Type "${requiredText}" to confirm`}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="w-full px-3 py-2 border border-muted-foreground/30 rounded-lg text-sm bg-background text-foreground"
              />
            </div>
          )}

          <div className="flex justify-end space-x-3">
            <Button onClick={onClose}>Cancel</Button>
            <Button
              variant="destructive"
              onClick={onConfirm}
              disabled={
                requiresInputConfirmation && input.trim() !== requiredText
              }
            >
              {confirmMessage}
            </Button>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
};
