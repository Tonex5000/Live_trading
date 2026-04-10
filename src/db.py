import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlmodel import SQLModel, Session, create_engine

load_dotenv()


def _resolve_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        return database_url
    # Safe fallback for local paper trading/dev mode.
    return "sqlite:///./paper_trading.db"


DATABASE_URL = _resolve_database_url()

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope():
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    return Session(engine)