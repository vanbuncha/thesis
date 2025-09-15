import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL is not set!"

# Example: postgresql+psycopg2://user:password@database:5432/dbname
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # recycle dead connections automatically
    future=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except SQLAlchemyError:
            pass
