from app.auth import routes as auth_router
from app.goals import routes as goals_router
from app.journals import routes as journals_router
from app.analysis import routes as analysis_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import Base, engine

app = FastAPI(
    title="Journal API",
    version="1.0.0",
    description="Backend for JRL — journaling, feedback, and AI-driven analysis.",
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://journaling-app-frontend-ecru.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router.router)
app.include_router(goals_router.router)
app.include_router(journals_router.router)
app.include_router(analysis_router.router)


# DB Tables
@app.on_event("startup")
def create_tables():
    Base.metadata.create_all(bind=engine)
