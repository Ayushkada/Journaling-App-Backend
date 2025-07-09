import axios, { AxiosError, AxiosRequestConfig } from "axios";
import { refreshToken } from "@/lib/authService";

let accessToken: string | null = localStorage.getItem("access_token");

/**
 * Updates the in-memory and persisted access token.
 */
export const setAccessToken = (token: string | null) => {
  accessToken = token;
  if (token) {
    localStorage.setItem("access_token", token);
  } else {
    localStorage.removeItem("access_token");
  }
};

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  withCredentials: true, // send refresh-token cookie
});

// Attach Bearer token on each request
api.interceptors.request.use((config) => {
  if (accessToken && config.headers) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});

// Handle 401 responses: attempt one refresh then retry original
api.interceptors.response.use(
  (response) => response,
  async (
    error: AxiosError & { config: AxiosRequestConfig & { _retry?: boolean } }
  ) => {
    const originalRequest = error.config;

    // Only retry once, and skip if we're already calling refresh
    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      originalRequest.url !== "/auth/refresh"
    ) {
      originalRequest._retry = true;
      try {
        // Try refreshing the access token
        const { access_token } = await refreshToken();
        setAccessToken(access_token);

        // Update the Authorization header for the retry
        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
        }

        // Retry the original request with new token
        return api(originalRequest);
      } catch (_err) {
        // Refresh failed: clear token and redirect to login
        setAccessToken(null);
        window.location.href = "/auth";
      }
    }

    return Promise.reject(error);
  }
);

export default api;
