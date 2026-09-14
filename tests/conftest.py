import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base, get_db
from app.services.aspect_extraction import AspectExtractor
from app.services.routing import SupportRouter
from app.services.sentiment import SentimentAnalyzer


@pytest.fixture(scope="session")
def analyzer():
    """Loading the model is expensive; share one across the session."""
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

    The suite used to write into app/ecommerce_router.db, so running the
    tests polluted real application data and /analytics assertions depended
    on whatever had run before.
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
