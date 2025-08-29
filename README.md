## 📓 Journaling App Backend

## 🌐 [Live Demo](https://journaling-app-frontend-ecru.vercel.app/) (Currently Offline)

## 🚀 Project Overview  
This is the backend for a journaling application built with FastAPI. It exposes secure, RESTful API endpoints for user authentication, journal management, goal tracking, and AI-powered analysis. The system helps users reflect on their mental and emotional health while tracking personal progress through data insights.

## 🔑 Features  
- **JWT Authentication**: Secure user login, signup, Apple Sign-In, and Google OAuth support, backed by PostgreSQL.  
- **Goal Management**: Create, update, and track personal goals with automatic analysis and progress tracking.  
- **Journaling System**: Add, edit, and delete journal entries, stored securely with encrypted insights.  
- **AI-Driven Analysis**: NLP models from Hugging Face power sentiment detection, goal extraction, and self-talk feedback.  
- **CORS Configured**: Supports integration with any frontend via customizable CORS settings.

## 🧰 Tech Stack  
- **Backend Framework**: FastAPI  
- **Database**: PostgreSQL  
- **ORM**: SQLAlchemy  
- **AI Models**: Hugging Face Transformers (RoBERTa, T5, etc.)  
- **Testing**: Pytest  
- **Deployment**: Docker, Uvicorn  
- **Security**: JWT Auth, Environment Variables
- **OAuth**: Google OAuth 2.0, Apple Sign-In

## 📦 Key Libraries & Tools  
- `FastAPI`, `SQLAlchemy`, `psycopg2`, `PyJWT`  
- `transformers`, `torch`, `spacy`, `nltk`, `pydantic`, `python-dotenv`  
- `Uvicorn`, `pytest`, `httpx`, `requests`, `loguru`  
- `Docker`, `docker-compose`  
- `google-auth`, `requests` (for Google OAuth)
- Optional: `cloudflared` (for temporary public tunnel)

## ⚙️ Setup Instructions

### Option 1: Run with `setup.sh` (Mac/Linux)
```bash
./setup.sh
```

### Option 2: Manual Setup
1. **Clone the repo**  
   ```bash
   git clone https://github.com/AyushKada/Journaling-App.git
   cd Journaling-App
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**  
  You'll need to create a `.env` file with the following keys. See `.env.example` for format.

- Get your OpenAI API key at [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- For Apple login, register a service at [Apple Developer Portal](https://developer.apple.com/account/resources/identifiers/)
- For Google OAuth, set up credentials at [Google Cloud Console](https://console.cloud.google.com/) (see `GOOGLE_OAUTH_SETUP.md` for detailed instructions)
- Generate a secure `SECRET_KEY` using `openssl rand -hex 32` or similar.

5. **Run database migration** (if adding Google OAuth to existing database)
   ```bash
   python migrate_add_google_id.py
   ```

6. **Ensure PostgreSQL is running**, and the DB in `.env` is created.

7. **Run the app locally**  
   ```bash
   uvicorn app.main:app --reload
   ```

## Option 3: 🐳 Docker Deployment  
Make sure Docker and Docker Compose are installed:

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8000`.

## 🌐 Temporary Public URL (via Cloudflare Tunnel)  
To expose your local app to the internet:

```bash
cloudflared tunnel --url http://localhost:8000
```

A `.trycloudflare.com` URL will be generated — share this for testing or demos.

## 📚 API Endpoints  
(*See full list in `docs/api_endpoints.md` or FastAPI Swagger UI at `/docs` after running the server.*)

- **Auth Routes**
  - **POST /auth/login**: Login with username and password.
  - **POST /auth/signup**: Register a new user.
  - **POST /auth/apple-login**: Login using Apple credentials.
  - **POST /auth/google-login**: Login using Google OAuth.
  - **GET /auth/google/url**: Get Google OAuth URL for frontend redirect.
  - **GET /auth/me**: Retrieve the current user's profile.

- **Analysis Routes**
  - **POST /analysis/run**: Run a full analysis pipeline.
  - **POST /analysis/journals**: Analyze pending journals.
  - **POST /analysis/connected**: Generate connected analysis.
  - **GET /analysis/feedback**: Get feedback based on analysis.
  - **GET /analysis/prompts**: Get prompts for recommendations.
  - **POST /analysis/goals**: Generate AI-based goals.

- **Goals Routes**
  - **GET /goals**: Retrieve all user goals.
  - **GET /goals/{goal_id}**: Retrieve a specific goal.
  - **POST /goals**: Create a new goal.
  - **PUT /goals/{goal_id}**: Update an existing goal.
  - **DELETE /goals/{goal_id}**: Delete a specific goal.
  - **DELETE /goals/all**: Delete all goals.

- **Journals Routes**
  - **GET /journals**: Retrieve all journal entries.
  - **GET /journals/{journal_id}**: Retrieve a specific journal entry.
  - **POST /journals**: Create a new journal entry.
  - **PUT /journals/{journal_id}**: Update an existing journal entry.
  - **DELETE /journals/{journal_id}**: Delete a specific journal entry.
  - **DELETE /journals/all**: Delete all journal entries.
  - **GET /journals/journal-analysis/{journal_id}**: Retrieve analysis for a specific journal.
  - **POST /journals/journal-analysis/{journal_id}**: Upsert analysis for a specific journal.
  - **DELETE /journals/journal-analysis/{journal_id}**: Delete analysis for a specific journal.
  - **GET /journals/connected-analysis**: Retrieve connected analysis.
  - **POST /journals/connected-analysis**: Upsert connected analysis.
  - **DELETE /journals/connected-analysis**: Delete connected analysis.

- **System Routes**
  - **POST /system/test-login**: Developer login for testing.
  - **GET /system/debug/users**: Retrieve all users for debugging.
  - **GET /system/debug/journals**: Retrieve all journals for debugging.
  - **GET /system/debug/goals**: Retrieve all goals for debugging.

## 🔐 OAuth Setup

### Google OAuth
For detailed Google OAuth setup instructions, see [`GOOGLE_OAUTH_SETUP.md`](GOOGLE_OAUTH_SETUP.md).

Quick setup:
1. Create a Google Cloud Project
2. Enable OAuth 2.0 API
3. Create OAuth 2.0 credentials
4. Add environment variables:
   ```env
   GOOGLE_CLIENT_ID=your_client_id
   GOOGLE_CLIENT_SECRET=your_client_secret
   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
   ```

### Apple Sign-In
1. Register your app in Apple Developer Portal
2. Add environment variables:
   ```env
   APPLE_AUDIENCE=your_bundle_id
   APPLE_ISSUER=https://appleid.apple.com
   APPLE_KEYS_URL=https://appleid.apple.com/auth/keys
   ```

## ☁️ Cloud Deployment Notes  
Can be deployed to:
- **Render**, **Heroku**, or **Fly.io** for quick container hosting  
- **AWS EC2**, **ECS**, or **Lightsail** for custom infrastructure

Be sure to:
- Secure secrets in environment variables
- Use HTTPS in production
- Set up PostgreSQL access
- Use a reverse proxy like Nginx if needed
- Update OAuth redirect URIs for production domains
