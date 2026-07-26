"""
SQLite database for platform features that don't belong in MLflow: user
accounts and a log of who queried the diagnostic model with what result.
MLflow stays the source of truth for training runs/metrics; this is for
runtime/serving-side state instead.
"""
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = "medsync.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="clinician")  # clinician | admin
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    predictions = relationship("PredictionLog", back_populates="user")


class PredictionLog(Base):
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    top_finding = Column(String, nullable=False)
    top_probability = Column(Float, nullable=False)
    inference_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="predictions")


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
