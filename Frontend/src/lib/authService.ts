import { LoginRequest, TokenResponse, UserCreate, UserOut } from "@/types/auth";
import api, { setAccessToken } from "./api";

export const login = async (data: LoginRequest): Promise<TokenResponse> => {
  const response = await api.post("/auth/login", data);
  setAccessToken(response.data.access_token); // ✅ store token
  return response.data;
};

export const signup = async (data: UserCreate): Promise<UserOut> => {
  const response = await api.post("/auth/signup", data);
  return response.data;
};

export const signupAndLogin = async (
  data: UserCreate
): Promise<TokenResponse> => {
  await signup(data);
  return await login({ email: data.email, password: data.password });
};

export const loginWithGoogle = async (
  data: LoginRequest
): Promise<TokenResponse> => {
  const response = await api.post("/auth/google-login", data);
  setAccessToken(response.data.access_token); // ✅ store token
  return response.data;
};

export const loginWithApple = async (data: any): Promise<TokenResponse> => {
  const response = await api.post("/auth/apple-login", data);
  setAccessToken(response.data.access_token); // ✅ store token
  return response.data;
};

export const getGoogleOAuthUrl = async (): Promise<{ url: string }> => {
  const response = await api.get("/auth/google/url");
  return response.data;
};

export const getCurrentUser = async (): Promise<UserOut> => {
  const response = await api.get("/auth/me");
  return response.data;
};

export const refreshToken = async (): Promise<TokenResponse> => {
  const response = await api.post("/auth/refresh");
  setAccessToken(response.data.access_token);
  return response.data;
};

export const logout = async (): Promise<{ detail: string }> => {
  const response = await api.post("/auth/logout");
  return response.data;
};

export const logoutUser = async () => {
  await logout();
  setAccessToken(null);
  localStorage.removeItem("access_token");
};

export const changeEmail = async (newEmail: string) => {
  return api.put("/auth/change-email", { new_email: newEmail });
};

export const changeUsername = async (newUsername: string) => {
  return api.put("/auth/change-username", { new_username: newUsername });
};

export const changePassword = async (
  oldPassword: string,
  newPassword: string
) => {
  return api.put("/auth/change-password", {
    old_password: oldPassword,
    new_password: newPassword,
  });
};

export const deleteAccount = async (): Promise<{ detail: string }> => {
  const response = await api.delete("/auth/delete");
  return response.data;
};
