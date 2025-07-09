import { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { User } from "lucide-react";

const Sidebar = ({ children }: { children: ReactNode }) => {
  const location = useLocation();
  const isProfileActive = location.pathname === "/profile";

  return (
    <aside
      className="w-64 h-screen bg-white shadow-sm flex flex-col border-r"
      aria-label="Sidebar Navigation"
    >
      <header className="flex items-center justify-between px-6 py-4 border-b">
        <h2 className="text-xl font-semibold text-primary">Mindful Journey</h2>
        <Link
          to="/profile"
          title="Profile"
          aria-label="Go to profile"
          className={`p-2 rounded-full transition ${
            isProfileActive
              ? "bg-primary text-white"
              : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
          }`}
        >
          <User className="w-6 h-6" />
        </Link>
      </header>

      <nav className="flex-1 overflow-hidden">{children}</nav>
    </aside>
  );
};

export default Sidebar;
