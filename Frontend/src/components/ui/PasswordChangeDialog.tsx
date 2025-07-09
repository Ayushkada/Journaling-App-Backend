import { useState } from "react";
import { Dialog, DialogPanel, DialogTitle } from "@headlessui/react";
import { Button } from "@/components/ui/Button";
import { changePassword } from "@/lib/authService";
import { Eye, EyeOff } from "lucide-react";

export const PasswordChangeDialog = ({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) => {
  const [oldPass, setOldPass] = useState("");
  const [newPass, setNewPass] = useState("");
  const [confirmPass, setConfirmPass] = useState("");
  const [step, setStep] = useState<"form" | "confirm" | "error" | "success">(
    "form"
  );
  const [error, setError] = useState("");
  const [showOld, setShowOld] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const handleNext = () => {
    if (!oldPass || !newPass || !confirmPass) {
      setError("All fields are required.");
      setStep("error");
      return;
    }

    if (newPass.length < 8) {
      setError("New password must be at least 8 characters.");
      setStep("error");
      return;
    }

    if (oldPass === newPass) {
      setError("New password must be different from the old password.");
      setStep("error");
      return;
    }

    if (newPass !== confirmPass) {
      setError("New passwords do not match.");
      setStep("error");
      return;
    }

    setError("");
    setStep("confirm");
  };

  const handleChange = async () => {
    try {
      await changePassword(oldPass, newPass);
      setStep("success");
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Password change failed.");
      setStep("error");
    }
  };

  const reset = () => {
    setOldPass("");
    setNewPass("");
    setConfirmPass("");
    setError("");
    setStep("form");
    onClose();
  };

  return (
    <Dialog open={open} onClose={reset} className="relative z-50">
      <div
        className="fixed inset-0 bg-black/30 backdrop-blur-sm"
        aria-hidden="true"
      />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="bg-white dark:bg-zinc-900 p-6 rounded-xl shadow-xl max-w-md w-full">
          <DialogTitle className="text-lg font-semibold mb-4">
            Change Password
          </DialogTitle>

          {step === "form" && (
            <>
              {/** Old Password */}
              <div className="relative mb-3">
                <input
                  type={showOld ? "text" : "password"}
                  placeholder="Old password"
                  className="w-full px-4 py-2 border rounded pr-10"
                  value={oldPass}
                  onChange={(e) => setOldPass(e.target.value)}
                />
                {oldPass && (
                  <button
                    type="button"
                    onClick={() => setShowOld((prev) => !prev)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
                    aria-label={showOld ? "Hide password" : "Show password"}
                  >
                    {showOld ? <Eye size={18} /> : <EyeOff size={18} />}
                  </button>
                )}
              </div>

              {/** New Password */}
              <div className="relative mb-3">
                <input
                  type={showNew ? "text" : "password"}
                  placeholder="New password"
                  className="w-full px-4 py-2 border rounded pr-10"
                  value={newPass}
                  onChange={(e) => setNewPass(e.target.value)}
                />
                {newPass && (
                  <button
                    type="button"
                    onClick={() => setShowNew((prev) => !prev)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
                    aria-label={showNew ? "Hide password" : "Show password"}
                  >
                    {showNew ? <Eye size={18} /> : <EyeOff size={18} />}
                  </button>
                )}
              </div>

              {/** Confirm Password */}
              <div className="relative mb-6">
                <input
                  type={showConfirm ? "text" : "password"}
                  placeholder="Confirm new password"
                  className="w-full px-4 py-2 border rounded pr-10"
                  value={confirmPass}
                  onChange={(e) => setConfirmPass(e.target.value)}
                />
                {confirmPass && (
                  <button
                    type="button"
                    onClick={() => setShowConfirm((prev) => !prev)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
                    aria-label={showConfirm ? "Hide password" : "Show password"}
                  >
                    {showConfirm ? <Eye size={18} /> : <EyeOff size={18} />}
                  </button>
                )}
              </div>

              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={reset}>
                  Cancel
                </Button>
                <Button onClick={handleNext}>Next</Button>
              </div>
            </>
          )}

          {step === "confirm" && (
            <>
              <p className="text-sm text-muted-foreground mb-6">
                Are you sure you want to change your password?
              </p>
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={reset}>
                  Cancel
                </Button>
                <Button variant="destructive" onClick={handleChange}>
                  Confirm
                </Button>
              </div>
            </>
          )}

          {step === "error" && (
            <>
              <p className="text-sm text-red-600 mb-6">{error}</p>
              <div className="flex justify-end space-x-2">
                <Button onClick={() => setStep("form")}>Try Again</Button>
              </div>
            </>
          )}

          {step === "success" && (
            <>
              <p className="text-sm text-green-600 mb-6">
                Password changed successfully!
              </p>
              <div className="flex justify-end">
                <Button onClick={reset}>Close</Button>
              </div>
            </>
          )}
        </DialogPanel>
      </div>
    </Dialog>
  );
};
