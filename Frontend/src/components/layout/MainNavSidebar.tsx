import { Link, useLocation } from "react-router-dom";
import { BookOpen, Target, BarChart3, User } from "lucide-react";
import Sidebar from "./Sidebar";

const navItems = [
  { path: "/journals", icon: BookOpen, label: "Journals" },
  { path: "/goals", icon: Target, label: "Goals" },
  { path: "/analysis", icon: BarChart3, label: "Analysis" },
];

const MainNavSidebar = () => {
  const location = useLocation();

  return (
    <Sidebar>
      <nav
        className="flex flex-col h-full"
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="shrink-0 sticky top-0 bg-background z-10 px-4 pt-4 pb-5 space-y-2">
          {navItems.map(({ path, icon: Icon, label }) => {
            const isActive = location.pathname === path;
            return (
              <Link
                key={path}
                to={path}
                className={`flex items-center px-4 py-3 text-md font-medium rounded-lg transition-colors ${
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                }`}
                aria-current={isActive ? "page" : undefined}
              >
                <Icon className="w-5 h-5 mr-3" aria-hidden="true" />
                {label}
              </Link>
            );
          })}
        </div>
      </nav>
    </Sidebar>
  );
};

export default MainNavSidebar;
