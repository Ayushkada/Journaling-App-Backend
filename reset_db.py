# app/system/reset_db.py
from app.core.database import Base, engine
from sqlalchemy import text

def reset_database():
    with engine.connect() as conn:
        print("⚠️ Dropping all tables with CASCADE...")
        conn.execute(text("DROP SCHEMA public CASCADE;"))
        conn.execute(text("CREATE SCHEMA public;"))
        conn.commit()

    print("🚀 Recreating tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables recreated successfully.")

if __name__ == "__main__":
    reset_database()
