import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import { useLocation } from "react-router-dom";
import {
  login as loginService,
  logoutUser,
  refreshToken,
  getCurrentUser,
  signupAndLogin,
} from "@/lib/authService";
import type { LoginRequest, UserCreate, UserOut } from "@/types/auth";
import { jwtDecode } from "jwt-decode";

type AuthStatus = "loading" | "authenticated" | "unauthenticated";

interface AuthContextType {
  user: UserOut | null;
  authStatus: AuthStatus;
  isAuthenticated: boolean;
  login: (data: LoginRequest, signupData?: UserCreate) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

function tokenExpired(token: string): boolean {
  try {
    const { exp } = jwtDecode<{ exp: number }>(token);
    return Date.now() >= exp * 1000;
  } catch {
    return true;
  }
}

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<UserOut | null>(null);
  const [authStatus, setAuthStatus] = useState<AuthStatus>("loading");
  const location = useLocation();

  const handleLogin = async (data: LoginRequest, signupData?: UserCreate) => {
    try {
      // signupAndLogin / loginService should set an HTTP‐only refresh cookie
      const { access_token } = signupData
        ? await signupAndLogin(signupData)
        : await loginService(data);

      localStorage.setItem("access_token", access_token);
      const u = await getCurrentUser();
      setUser(u);
      setAuthStatus("authenticated");
      localStorage.setItem("user", JSON.stringify(u));
    } catch (err) {
      setUser(null);
      setAuthStatus("unauthenticated");
      throw err;
    }
  };

  const handleLogout = async () => {
    try {
      await logoutUser();
    } finally {
      setUser(null);
      setAuthStatus("unauthenticated");
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");
    }
  };

  const handleRefresh = async () => {
    try {
      const { access_token } = await refreshToken();
      if (!access_token) throw new Error("Refresh failed");
      localStorage.setItem("access_token", access_token);

      const u = await getCurrentUser();
      setUser(u);
      setAuthStatus("authenticated");
      localStorage.setItem("user", JSON.stringify(u));
    } catch {
      setUser(null);
      setAuthStatus("unauthenticated");
      localStorage.removeItem("access_token");
      localStorage.removeItem("user");
      throw new Error("Refresh failed");
    }
  };

  // Initialize on mount or when the route changes
  useEffect(() => {
    const initAuth = async () => {
      const at = localStorage.getItem("access_token");

      // 1) If no access_token or it’s expired → try a refresh
      if (!at || tokenExpired(at)) {
        try {
          await handleRefresh();
          return;
        } catch {
          setAuthStatus("unauthenticated");
          return;
        }
      }

      // 2) access_token valid → hydrate saved user
      const saved = localStorage.getItem("user");
      if (saved) {
        setUser(JSON.parse(saved));
        setAuthStatus("authenticated");
        return;
      }

      // 3) no saved user → fetch profile, on 401 try one more refresh
      try {
        const u = await getCurrentUser();
        setUser(u);
        setAuthStatus("authenticated");
        localStorage.setItem("user", JSON.stringify(u));
      } catch (err: any) {
        if (err.response?.status === 401) {
          try {
            await handleRefresh();
          } catch {
            setAuthStatus("unauthenticated");
          }
        } else {
          setAuthStatus("unauthenticated");
        }
      }
    };

    initAuth();
  }, [location.pathname]);

  // Optional: keep session alive by refreshing every 55 minutes
  useEffect(() => {
    if (authStatus !== "authenticated") return;
    const id = setInterval(() => {
      handleRefresh().catch(() => setAuthStatus("unauthenticated"));
    }, 55 * 60 * 1000);
    return () => clearInterval(id);
  }, [authStatus]);

  return (
    <AuthContext.Provider
      value={{
        user,
        authStatus,
        isAuthenticated: authStatus === "authenticated",
        login: handleLogin,
        logout: handleLogout,
        refresh: handleRefresh,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
