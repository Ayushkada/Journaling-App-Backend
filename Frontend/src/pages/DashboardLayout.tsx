import { Outlet } from "react-router-dom";
import { useIsMobile } from "@/hooks/useMobile";

const DashboardLayout = () => {
  const isMobile = useIsMobile();

  if (isMobile) {
    return (
      <div className="min-h-screen flex items-center justify-center text-center px-4">
        <p className="text-lg text-white bg-black/70 rounded-lg p-6">
          📱 Mobile login view is coming soon. Please use a desktop for now.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Outlet />
    </div>
  );
};

export default DashboardLayout;
