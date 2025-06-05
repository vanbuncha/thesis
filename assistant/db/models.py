from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    identifier = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.now())

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    started_at = Column(DateTime, default=datetime.now())
    ended_at = Column(DateTime)

    user = relationship("User", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    user_input = Column(Text)
    llm_response = Column(Text)
    created_at = Column(DateTime, default=datetime.now())

    session = relationship("Session", back_populates="interactions")
