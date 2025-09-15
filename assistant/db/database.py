import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DBSession
from fastapi import Depends
from .models import Base

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL is not None, "DATABASE_URL is not set!"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
