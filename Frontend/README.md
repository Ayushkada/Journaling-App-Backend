## Mindfull Journey - Frontend

A modern journaling and goal-tracking web app focused on mental wellness, built with **React**, **TypeScript**, **Vite**, **TailwindCSS**, and **ShadCN UI**. This frontend integrates with a FastAPI backend to deliver AI-powered insights on journal entries and goals.

## 🌐 Live Demo

Coming soon via [Vercel](https://vercel.com)

---

## 📦 Tech Stack

- **Framework**: [React](https://reactjs.org/) + [Vite](https://vitejs.dev/)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **UI**: [TailwindCSS](https://tailwindcss.com/) + [ShadCN](https://ui.shadcn.com/)
- **Routing**: [React Router v6](https://reactrouter.com/)
- **Form Handling**: [React Hook Form](https://react-hook-form.com/)
- **Auth**: Custom context + JWT + HTTP-only cookies
- **API**: Axios with token refresh handling
- **Charts**: [Recharts](https://recharts.org/en-US/)
- **Animation**: Framer Motion, Headless UI, Radix UI

---

## 📁 Folder Structure

```
src/
│
├── components/            # Shared UI + layout components
│   ├── layout/            # Navbar, Sidebar, etc.
│   ├── ui/                # Reusable UI like Button, Form, Calendar, etc.
│
├── config/                # Static config like category color maps
├── guards/                # Route guards (auth/guest)
├── hooks/                 # Custom hooks (e.g., mobile detection, combobox)
├── lib/                   # API services (auth, journals, goals, analysis)
├── pages/                 # Route-level components (Journals, Goals, Profile)
├── types/                 # Shared TypeScript types
├── utils/                 # Utility functions (classnames, helpers, etc.)
├── provider.tsx          # Auth context provider
├── routes.tsx            # Route declarations with guards
├── App.tsx               # Root app shell
└── main.tsx              # Entry point
```

---

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+)
- pnpm / npm / yarn
- Backend running locally or hosted with HTTPS + CORS support

---

### 1. Install Dependencies

```bash
pnpm install
# or
npm install
```

---

### 2. Setup Environment Variables

Create a `.env` file:

```
VITE_API_BASE_URL=https://your-backend-domain.com
```

---

### 3. Run the App Locally

```bash
pnpm dev
# or
npm run dev
```

This runs the app on [http://localhost:8080](http://localhost:8080)

---

### 4. Build for Production

```bash
pnpm build
# or
npm run build
```

---

## 🔐 Authentication Flow

- Auth tokens are stored in `localStorage` (access) + HTTP-only cookie (refresh)
- Automatically refreshes token if expired (via `lib/api.ts`)
- Route protection using:

  - `AuthGuard` → for authenticated users
  - `GuestGuard` → for unauthenticated users

---

## 🧠 Features

- 📝 Journals with sentiment & emotion analysis
- 🌟 Goals with progress tracking
- 📊 Analysis coming soon
- 🧠 AI insights from backend (e.g., Hugging Face)
- 🧩 Modular UI components with consistent design system
- 🌙 Light/dark theme support (future)

---

> Built with ❤️ for mindful growth and reflection.
