import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

DATABASE_URL = os.getenv("DATABASE_URL") 

# Token & Auth
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Apple Sign-In
APPLE_KEYS_URL = os.getenv("APPLE_KEYS_URL")
APPLE_AUDIENCE = os.getenv("APPLE_AUDIENCE")
APPLE_ISSUER = os.getenv("APPLE_ISSUER")

# Google Sign-In
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")