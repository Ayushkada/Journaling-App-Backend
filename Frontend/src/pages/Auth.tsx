import { useEffect, useState } from "react";
import { useIsMobile } from "@/hooks/useMobile";
import { GoogleIcon } from "@/components/icons/GoogleIcon";
import { AppleIcon } from "@/components/icons/AppleIcon";
import { Eye, EyeOff } from "lucide-react";

import type { UserCreate } from "@/types/auth";
import { useAuth } from "@/provider";

import { useNavigate } from "react-router-dom";

const Auth = () => {
  const isMobile = useIsMobile();

  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [fieldErrors, setFieldErrors] = useState<{ [key: string]: string }>({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const { login: loginUser } = useAuth();

  const BACKGROUND_IMAGE_URL = "login-page.jpg";
  const headingText = isLogin ? "Welcome Back" : "Join Us";
  const subText = isLogin
    ? "Continue your mindful journey"
    : "Begin your transformation today";
  const actionText = isLogin ? "Sign In" : "Create Account";
  const toggleText = isLogin
    ? "Don't have an account? Sign up"
    : "Already have an account? Sign in";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);
    setFieldErrors({});

    const newErrors: { [key: string]: string } = {};

    if (!email) newErrors.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
      newErrors.email = "Invalid email address.";

    if (!password) newErrors.password = "Password is required.";
    else if (!isLogin && password.length < 8)
      newErrors.password = "Password must be at least 8 characters.";

    if (!isLogin) {
      if (!username) newErrors.username = "Username is required.";
      if (!confirmPassword)
        newErrors.confirmPassword = "Please confirm your password.";
      else if (password !== confirmPassword)
        newErrors.confirmPassword = "Passwords do not match.";
    }

    if (Object.keys(newErrors).length > 0) {
      setFieldErrors(newErrors);
      setLoading(false);
      return;
    }

    try {
      if (isLogin) {
        await loginUser({ email, password });
        setSuccess("Logged in successfully!");
      } else {
        const newUser: UserCreate = {
          email,
          name: username,
          password,
          type: "normal",
          auth_methods: ["email"],
          storage_type: "db",
        };
        await loginUser({ email, password }, newUser);
        setSuccess("Account created and logged in!");
      }
    } catch (err: any) {
      console.log("login failed:", err);
      const msg =
        err?.response?.status === 401
          ? "Invalid email or password."
          : err?.response?.data?.detail ||
            err.message ||
            "Something went wrong.";

      setError(msg);

      if (isLogin) {
        setFieldErrors({
          email: " ",
          password: " ",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const getInputClass = (field: string) =>
    `w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent transition-all bg-background text-foreground ${
      fieldErrors[field] ? "border-red-500" : "border-input"
    }`;

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
    <div
      className="min-h-screen flex flex-col md:flex-row bg-cover bg-center md:bg-none relative"
      style={{ backgroundImage: `url('${BACKGROUND_IMAGE_URL}')` }}
    >
      <div className="absolute inset-0 bg-black/30 md:hidden z-0" />

      <div
        className="hidden md:flex flex-1 bg-cover bg-center relative"
        style={{
          backgroundImage: `linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)), url('${BACKGROUND_IMAGE_URL}')`,
        }}
      >
        <div className="absolute inset-0 flex items-center justify-center text-white p-8">
          <div className="text-center max-w-lg">
            <h2 className="text-4xl font-bold mb-6">
              "The journey of a thousand miles begins with one step."
            </h2>
            <p className="text-2xl opacity-90">
              Start your mindful journey today and discover the power of
              self-reflection.
            </p>
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex items-center justify-center p-4 md:p-8 md:bg-white">
        <div className="w-full max-w-md">
          <div className="bg-card p-4 md:p-8 rounded-2xl shadow-xl border bg-opacity-90">
            <div className="text-center mb-8">
              <h1 className="text-2xl md:text-3xl font-bold text-card-foreground mb-2">
                {headingText}
              </h1>
              <p className="text-muted-foreground">{subText}</p>
            </div>

            {error && (
              <div className="text-sm text-red-500 mb-4 text-center">
                {error}
              </div>
            )}
            {success && (
              <div className="text-sm text-green-600 mb-4 text-center">
                {success}
              </div>
            )}

            <form onSubmit={handleSubmit} noValidate className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-card-foreground mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value);
                    setFieldErrors((prev) => ({ ...prev, email: "" }));
                  }}
                  className={getInputClass("email")}
                  placeholder="your@email.com"
                />
                {fieldErrors.email && (
                  <p className="text-sm text-red-500 mt-1">
                    {fieldErrors.email}
                  </p>
                )}
              </div>

              {!isLogin && (
                <div>
                  <label className="block text-sm font-medium text-card-foreground mb-2">
                    Username
                  </label>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => {
                      setUsername(e.target.value);
                      setFieldErrors((prev) => ({ ...prev, username: "" }));
                    }}
                    className={getInputClass("username")}
                    placeholder="Your username"
                  />
                  {fieldErrors.username && (
                    <p className="text-sm text-red-500 mt-1">
                      {fieldErrors.username}
                    </p>
                  )}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-card-foreground mb-2">
                  Password
                  {!isLogin && (
                    <span className="text-muted-foreground text-xs">
                      {" "}
                      (Must be at least 8 characters.)
                    </span>
                  )}
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    autoComplete={isLogin ? "current-password" : "new-password"}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      setFieldErrors((prev) => ({ ...prev, password: "" }));
                    }}
                    className={`${getInputClass("password")} pr-10`}
                    placeholder="••••••••"
                  />
                  {password && (
                    <button
                      type="button"
                      onClick={() => setShowPassword((prev) => !prev)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
                      aria-label={
                        showPassword ? "Hide password" : "Show password"
                      }
                    >
                      {showPassword ? <Eye size={18} /> : <EyeOff size={18} />}
                    </button>
                  )}
                </div>
                {fieldErrors.password && (
                  <p className="text-sm text-red-500 mt-1">
                    {fieldErrors.password}
                  </p>
                )}
              </div>

              {!isLogin && (
                <div>
                  <label className="block text-sm font-medium text-card-foreground mb-2">
                    Confirm Password
                  </label>
                  <div className="relative">
                    <input
                      type={showConfirmPassword ? "text" : "password"}
                      value={confirmPassword}
                      autoComplete="new-password"
                      onChange={(e) => {
                        setConfirmPassword(e.target.value);
                        setFieldErrors((prev) => ({
                          ...prev,
                          confirmPassword: "",
                        }));
                      }}
                      className={`${getInputClass("confirmPassword")} pr-10`}
                      placeholder="••••••••"
                    />
                    {confirmPassword && (
                      <button
                        type="button"
                        onClick={() => setShowConfirmPassword((prev) => !prev)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground"
                        aria-label={
                          showConfirmPassword
                            ? "Hide password"
                            : "Show password"
                        }
                      >
                        {showConfirmPassword ? (
                          <Eye size={18} />
                        ) : (
                          <EyeOff size={18} />
                        )}
                      </button>
                    )}
                  </div>
                  {fieldErrors.confirmPassword && (
                    <p className="text-sm text-red-500 mt-1">
                      {fieldErrors.confirmPassword}
                    </p>
                  )}
                </div>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-3 rounded-lg font-semibold transition-all transform hover:scale-105 block text-center"
              >
                {loading ? "Please wait..." : actionText}
              </button>
            </form>

            <div className="mt-8 text-center">
              <button
                type="button"
                onClick={() => {
                  setIsLogin(!isLogin);
                  setEmail("");
                  setUsername("");
                  setPassword("");
                  setConfirmPassword("");
                  setError("");
                  setSuccess("");
                  setFieldErrors({});
                  setShowPassword(false);
                  setShowConfirmPassword(false);
                }}
                className="text-primary hover:text-primary/80 font-medium transition-colors"
              >
                {toggleText}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Auth;
