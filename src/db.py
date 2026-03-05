import os
from dotenv import load_dotenv
from sqlmodel import SQLModel, Session, create_engine

# LOAD .env FILE
load_dotenv()


def _require_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure your real database URL."
        )
    return database_url


DATABASE_URL = _require_database_url()

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)