from app.auth import routes as auth_router
from app.goals import routes as goals_router
from app.journals import routes as journals_router
from app.analysis import routes as analysis_router
from app.system import routes as system_router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import Base, engine

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router.router)
app.include_router(goals_router.router)
app.include_router(journals_router.router)
app.include_router(analysis_router.router)
app.include_router(system_router.router)

# DB Tables
Base.metadata.create_all(bind=engine)