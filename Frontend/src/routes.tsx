import React from "react";
import { RouteObject } from "react-router-dom";

import Landing from "@/pages/Landing";
import Auth from "./pages/Auth";
import Journals from "./pages/Journals/Journals";
import Goals from "@/pages/Goals/Goals";
import Analysis from "@/pages/Analysis";
import Profile from "@/pages/Profile";
import NotFound from "@/pages/NotFound";
import DashboardLayout from "@/pages/DashboardLayout";

import AuthGuard from "@/guards/AuthGuard";
import GuestGuard from "@/guards/GuestGuard";
import JournalForm from "./pages/Journals/JournalForm";
import JournalView from "@/pages/Journals/JournalView";
import GoalForm from "@/pages/Goals/GoalForm";
import GoalLayout from "./pages/Goals/GoalLayout";
import JournalLayout from "./pages/Journals/JournalLayout";
import MainNavLayout from "./pages/MainNavLayout";
import GoalView from "./pages/Goals/GoalView";

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Landing />,
  },
  {
    path: "/auth",
    element: (
      <GuestGuard>
        <Auth />
      </GuestGuard>
    ),
  },
  {
    path: "/",
    element: (
      <AuthGuard>
        <DashboardLayout />
      </AuthGuard>
    ),
    children: [
      {
        path: "/",
        element: <MainNavLayout />,
        children: [
          { path: "journals", element: <Journals /> },
          { path: "goals", element: <Goals /> },
          { path: "analysis", element: <Analysis /> },
          { path: "profile", element: <Profile /> },
        ],
      },
      {
        path: "journals",
        element: <JournalLayout />,
        children: [
          { path: "add", element: <JournalForm /> },
          { path: ":id/edit", element: <JournalForm /> },
          { path: ":id", element: <JournalView /> },
        ],
      },
      {
        path: "goals",
        element: <GoalLayout />,
        children: [
          { path: "add", element: <GoalForm /> },
          { path: ":id/edit", element: <GoalForm /> },
          { path: ":id", element: <GoalView /> },
        ],
      },
    ],
  },
  {
    path: "*",
    element: <NotFound />,
  },
];
