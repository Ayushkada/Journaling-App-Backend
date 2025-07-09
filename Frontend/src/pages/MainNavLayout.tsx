import { Outlet } from "react-router-dom";
import MainNavSidebar from "@/components/layout/MainNavSidebar";

const MainNavLayout = () => {
  return (
    <>
      <MainNavSidebar />
      <div className="flex-1 overflow-y-auto p-6">
        <Outlet />
      </div>
    </>
  );
};

export default MainNavLayout;
