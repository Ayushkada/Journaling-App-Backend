# 🧠 Journaling App – Full Stack

This repository contains the **full-stack codebase** for a journaling application that helps users reflect, track emotional growth, and make progress on goals using AI insights.

---

## 📂 Project Structure

```
/Journaling-App
├── backend/        # FastAPI backend for authentication, journals, goals, and AI analysis
├── frontend/       # Vite + React + TypeScript frontend with shadcn/ui and Tailwind
├── README.md       # Root overview (this file)
```

---

## 🔧 Setup

Each folder contains its own `README.md` with setup instructions.

Start by navigating to either folder:

```bash
cd frontend/   # for the frontend app
cd backend/    # for the backend API
```

Then follow the README in each directory to run locally or deploy.

---

## 📦 Tech Summary

### Backend

* **Framework**: FastAPI
* **Database**: PostgreSQL
* **AI Models**: Hugging Face (T5, RoBERTa, spaCy), OpenAI (optional)
* **Auth**: JWT with refresh tokens (HTTP-only cookie)
* **DevOps**: Docker, Cloudflare, optional CI/CD

### Frontend

* **Framework**: React + TypeScript (Vite)
* **UI**: shadcn/ui, Tailwind CSS, Headless UI
* **Routing**: React Router v6
* **State/Auth**: Context API, JWT
* **Deployment**: Vercel

---

## ✨ Key Features

* Secure auth with Apple login (coming soon)
* AI-powered journal and goal analysis
* Emotion, tone, energy, and sentiment insights
* Goal progress tracking with journal integration
* Real-time UI feedback, modals, and confirmation dialogs
* Responsive layout with mobile-first design
* Local + cloud-ready architecture

---

## 👥 Author

**Ayush Kadakia**
[LinkedIn](https://linkedin.com/in/ayush-kadakia1/) • [GitHub](https://github.com/AyushKada)

---

For detailed setup, API routes, and deployment instructions:

* See [`backend/README.md`](./backend/README.md)
* See [`frontend/README.md`](./frontend/README.md)
