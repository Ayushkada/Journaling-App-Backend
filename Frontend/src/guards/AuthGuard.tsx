import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../provider";

const AuthGuard = ({ children }: { children: JSX.Element }) => {
  const { authStatus, isAuthenticated } = useAuth();
  if (authStatus === "loading") return null;
  return isAuthenticated ? children : <Navigate to="/auth" replace />;
};
export default AuthGuard;
