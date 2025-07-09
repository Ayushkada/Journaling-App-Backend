import { useEffect, useState } from "react";
import { User, Trash2, LogOut } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/provider";
import { ConfirmDialog } from "@/components/ui/ConfimDialog";
import { Button } from "@/components/ui/Button";
import { PasswordChangeDialog } from "@/components/ui/PasswordChangeDialog";
import {
  changeEmail,
  changeUsername,
  deleteAccount,
  getCurrentUser,
} from "@/lib/authService";
import { toast } from "sonner";
import { deleteAllGoals } from "@/lib/goalService";
import { deleteAllJournals } from "@/lib/journalService";

const Profile = () => {
  const [name, setName] = useState("");
  const [originalName, setOriginalName] = useState("");
  const [email, setEmail] = useState("");
  const [originalEmail, setOriginalEmail] = useState("");
  const [deleteType, setDeleteType] = useState<
    "account" | "goals" | "journals" | null
  >(null);
  const [isEditingName, setIsEditingName] = useState(false);
  const [isEditingEmail, setIsEditingEmail] = useState(false);
  const [useLogout, setUseLogout] = useState(false);
  const [showPasswordDialog, setShowPasswordDialog] = useState(false);
  const [confirmEmail, setConfirmEmail] = useState(false);

  const { logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const loadUser = async () => {
      try {
        const user = await getCurrentUser();
        setName(user.name);
        setOriginalName(user.name);
        setEmail(user.email);
        setOriginalEmail(user.email);
      } catch (err) {
        console.error("Failed to load user profile", err);
      }
    };
    loadUser();
  }, []);

  const handleLogout = async () => {
    await logout();
    navigate("/auth");
  };

  const handleDelete = async () => {
    try {
      if (deleteType === "account") {
        await deleteAllGoals();
        await deleteAllJournals();
        await deleteAccount();
        toast.success("Account deleted");
        await handleLogout();
      } else if (deleteType === "goals") {
        await deleteAllGoals();
        toast.success("All goals deleted");
      } else if (deleteType === "journals") {
        await deleteAllJournals();
        toast.success("All journals deleted");
      }
    } catch (err) {
      console.error("Deletion failed:", err);
      toast.error("Failed to delete " + deleteType);
    } finally {
      setDeleteType(null);
    }
  };

  const handleUsernameUpdate = async () => {
    try {
      if (name !== originalName) {
        await changeUsername(name);
        setOriginalName(name);
        toast.success("Username updated");
      }
      setIsEditingName(false);
    } catch (err) {
      console.error("Username update failed:", err);
      toast.error("Could not update username");
    }
  };

  const handleUsernameCancel = async () => {
    setName(originalName);
    setIsEditingName(false);
  };

  const handleEmailUpdate = async () => {
    setConfirmEmail(false);
    try {
      if (email !== originalEmail) {
        await changeEmail(email);
        setOriginalEmail(email);
        toast.success("Email updated.");
      }
      setIsEditingEmail(false);
    } catch (err) {
      console.error("Email update failed:", err);
      toast.error("Could not update email");
    }
  };

  const handleEmailCancel = () => {
    setEmail(originalEmail);
    setIsEditingEmail(false);
  };

  const deleteOptions = [
    {
      title: "Delete Account",
      desc: "This will delete all your journals, goals, and data permanently.",
      type: "account",
    },
    {
      title: "Delete Goals",
      desc: "This will delete all your goal data permanently.",
      type: "goals",
    },
    {
      title: "Delete Journals",
      desc: "This will delete all your journal entries permanently.",
      type: "journals",
    },
  ] as const;

  return (
    <div className="min-h-screen bg-background flex">
      <div className="flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-foreground mb-2">
              Profile Settings
            </h1>
            <p className="text-muted-foreground text-lg">
              Manage your account and preferences
            </p>
          </div>

          <div className="grid gap-8">
            {/* Personal Info */}
            <div className="bg-card rounded-xl shadow-lg border p-6">
              <h2 className="text-xl font-semibold text-card-foreground mb-6 flex items-center">
                <User className="w-5 h-5 mr-2" />
                Personal Information
              </h2>

              <div className="space-y-4">
                {/* Username */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Username
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      value={name}
                      disabled={!isEditingName}
                      onChange={(e) => setName(e.target.value)}
                      className={`flex-1 h-10 px-4 border rounded-lg text-sm transition-all ${
                        isEditingName
                          ? "bg-background focus:ring-2 focus:ring-primary"
                          : "bg-muted text-muted-foreground cursor-not-allowed"
                      }`}
                    />
                    <Button
                      className="h-10 w-20 text-sm"
                      onClick={
                        isEditingName
                          ? name !== originalName
                            ? handleUsernameUpdate
                            : () => setIsEditingName(false) // just close edit if no change
                          : () => setIsEditingName(true)
                      }
                    >
                      {isEditingName ? "Save" : "Edit"}
                    </Button>
                    {isEditingName && (
                      <Button
                        className="h-10 w-20 text-sm"
                        variant="outline"
                        onClick={handleUsernameCancel}
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>

                {/* Email */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Email Address
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="email"
                      value={email}
                      disabled={!isEditingEmail}
                      onChange={(e) => setEmail(e.target.value)}
                      className={`flex-1 h-10 px-4 border rounded-lg text-sm transition-all ${
                        isEditingEmail
                          ? "bg-background focus:ring-2 focus:ring-primary"
                          : "bg-muted text-muted-foreground cursor-not-allowed"
                      }`}
                    />
                    <Button
                      className="h-10 w-20 text-sm"
                      onClick={
                        isEditingEmail
                          ? email !== originalEmail
                            ? () => setConfirmEmail(true)
                            : () => setIsEditingEmail(false)
                          : () => setIsEditingEmail(true)
                      }
                    >
                      {isEditingEmail ? "Save" : "Edit"}
                    </Button>
                    {isEditingEmail && (
                      <Button
                        className="h-10 w-20 text-sm"
                        variant="outline"
                        onClick={handleEmailCancel}
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                  <ConfirmDialog
                    open={!!confirmEmail}
                    onClose={() => setConfirmEmail(false)}
                    onConfirm={handleEmailUpdate}
                    title="Comfirm Email Change"
                    message={`Are you sure you want to change your email from ${originalEmail} to ${email}`}
                    confirmMessage="Confirm"
                  />
                </div>

                {/* Password */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Password
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="password"
                      value="********"
                      disabled
                      className="flex-1 h-10 px-4 border rounded-lg bg-muted text-muted-foreground cursor-not-allowed text-sm"
                    />
                    <Button
                      className="h-10 w-20 text-sm"
                      onClick={() => setShowPasswordDialog(true)}
                    >
                      Edit
                    </Button>
                  </div>
                </div>

                <PasswordChangeDialog
                  open={showPasswordDialog}
                  onClose={() => setShowPasswordDialog(false)}
                />
              </div>
            </div>

            {/* Logout */}
            <div className="bg-card rounded-xl shadow-lg border p-6">
              <h2 className="text-xl font-semibold text-card-foreground mb-2 flex items-center">
                <LogOut className="w-5 h-5 mr-2" />
                Logout
              </h2>

              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <p className="text-muted-foreground">
                  Sign out of your account on this device.
                </p>
                <Button
                  onClick={() => setUseLogout(true)}
                  className="h-10 w-20 text-sm self-start sm:self-auto"
                >
                  Logout
                </Button>
              </div>

              <ConfirmDialog
                open={!!useLogout}
                onClose={() => setUseLogout(false)}
                onConfirm={handleLogout}
                title="Logout"
                message="Are you sure you want to logout on this device?"
                confirmMessage="Logout"
              />
            </div>

            {/* Danger Zone */}
            <div className="bg-card rounded-xl shadow-md border border-destructive/30 p-6">
              <h2 className="text-xl font-semibold text-destructive mb-6 flex items-center">
                <Trash2 className="w-5 h-5 mr-2" />
                Danger Zone
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {deleteOptions.map(({ title, desc, type }) => (
                  <div
                    key={type}
                    className="bg-destructive/10 rounded-xl p-5 border border-destructive/30 shadow-sm flex flex-col"
                  >
                    <h3 className="font-semibold text-destructive text-lg mb-2">
                      {title}
                    </h3>
                    <p className="text-sm text-destructive/80 mb-4">{desc}</p>
                    <div className="mt-auto">
                      <Button
                        onClick={() => setDeleteType(type)}
                        className="w-full"
                        variant="destructive"
                      >
                        {title}
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Confirm Delete Dialog */}
          <ConfirmDialog
            open={!!deleteType}
            onClose={() => setDeleteType(null)}
            onConfirm={handleDelete}
            title={`Delete ${
              deleteType?.charAt(0).toUpperCase() + deleteType?.slice(1)
            }?`}
            message={`Are you sure you want to delete your ${
              deleteType === "account"
                ? "account and all associated data"
                : deleteType
            }? This action cannot be undone.`}
            confirmMessage={`Delete ${
              deleteType?.charAt(0).toUpperCase() + deleteType?.slice(1)
            }`}
          />
        </div>
      </div>
    </div>
  );
};

export default Profile;
