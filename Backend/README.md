## 📓 Journaling App Backend

## 🚀 Project Overview  
This is the backend for a journaling application built with FastAPI. It exposes secure, RESTful API endpoints for user authentication, journal management, goal tracking, and AI-powered analysis. The system helps users reflect on their mental and emotional health while tracking personal progress through data insights.

## 🔑 Features  
- **JWT Authentication**: Secure user login, signup, and Apple Sign-In support, backed by PostgreSQL.  
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

## 📦 Key Libraries & Tools  
- `FastAPI`, `SQLAlchemy`, `psycopg2`, `PyJWT`  
- `transformers`, `torch`, `spacy`, `nltk`, `pydantic`, `python-dotenv`  
- `Uvicorn`, `pytest`, `httpx`, `requests`, `loguru`  
- `Docker`, `docker-compose`  
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
   - Copy `.env.example` to `.env` and configure

5. **Ensure PostgreSQL is running**, and the DB in `.env` is created.

6. **Run the app locally**  
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

## ☁️ Cloud Deployment Notes  
Can be deployed to:
- **Render**, **Heroku**, or **Fly.io** for quick container hosting  
- **AWS EC2**, **ECS**, or **Lightsail** for custom infrastructure

Be sure to:
- Secure secrets in environment variables
- Use HTTPS in production
- Set up PostgreSQL access
- Use a reverse proxy like Nginx if needed