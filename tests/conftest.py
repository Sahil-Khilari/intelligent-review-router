"""
Shared fixtures.

ALLOW_CPU is set here, before any app import, because CI runners have no GPU.
This is the only place CPU execution is enabled; the application never sets it
(see app/services/device.py).
"""

import os

os.environ.setdefault("ALLOW_CPU", "1")

import tempfile  # noqa: E402

import pytest  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db.database import Base, get_db  # noqa: E402
from app.services.aspect_extraction import AspectExtractor  # noqa: E402
from app.services.routing import SupportRouter  # noqa: E402
from app.services.sentiment import SentimentAnalyzer  # noqa: E402


@pytest.fixture(scope="session")
def analyzer():
    """Loading a model is expensive; share one across the session."""
    return SentimentAnalyzer()


@pytest.fixture
def extractor():
    return AspectExtractor()


@pytest.fixture
def router():
    return SupportRouter()


@pytest.fixture
def client():
    """
    TestClient bound to a throwaway SQLite file.

    The suite previously wrote into app/ecommerce_router.db, so running tests
    mutated real application data and /analytics assertions depended on
    whatever had run before.
    """
    from main import app

    handle, path = tempfile.mkstemp(suffix=".db")
    os.close(handle)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    engine.dispose()
    try:
        os.unlink(path)
    except PermissionError:
        pass
