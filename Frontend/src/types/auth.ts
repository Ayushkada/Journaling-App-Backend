export interface UserBase {
  email: string;
  name: string;
  type: 'normal' | string;
  auth_methods: string[];
  storage_type: string;
  storage_path?: string | null;
}

export interface UserCreate extends UserBase {
  password: string;
}

export interface UserOut extends UserBase {
  id: string; // UUIDs are strings in TypeScript
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  user: UserOut;
}

export interface GoogleLoginRequest {
  code: string;
}
