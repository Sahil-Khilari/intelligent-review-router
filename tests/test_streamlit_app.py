"""
Headless tests for the Streamlit dashboard.

Checking that port 8501 answers only proves the server started; the pages are
rendered over a websocket, so a page that raises still returns HTTP 200. These
run the script with Streamlit's AppTest and inspect what it actually renders.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from streamlit.testing.v1 import AppTest

import app.db.database as database

APP = "../streamlit_app.py"
TIMEOUT = 180  # the first run loads models

PAGES = [
    "🔍 Live Review Analyzer",
    "📁 Batch Review Inspector",
    "🎫 Support Ticket Desk",
    "📊 Analytics & Insights",
]


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Keep the dashboard off the real application database."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'ui.db'}", connect_args={"check_same_thread": False}
    )
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(
        database, "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    yield
    engine.dispose()


def fresh_app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(isolated_db, page):
    at = fresh_app()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    assert at.title, f"{page} rendered no title"


def test_sidebar_reports_the_models_actually_loaded(isolated_db):
    """A fallback that looks like the real model is how a stale one went unnoticed."""
    at = fresh_app()
    panels = [e.value for e in at.sidebar.info] + [e.value for e in at.sidebar.warning]
    text = " ".join(panels)
    assert "Models in use" in text
    assert "Sentiment:" in text and "Aspects:" in text


def test_analyze_renders_a_routing_summary(isolated_db):
    at = fresh_app()
    at.text_area[0].input(
        "Charged twice for my order and the refund never came."
    ).run()
    analyze = next(b for b in at.button if "Analyze" in b.label)
    analyze.click().run()

    assert not at.exception, [e.value for e in at.exception]
    subheaders = [s.value for s in at.subheader]
    assert any("Analysis & Routing Summary" in s for s in subheaders), subheaders
    assert at.dataframe, "the routing matrix did not render"


def test_batch_tab_explains_itself_when_there_is_no_data(isolated_db, monkeypatch, tmp_path):
    """With no corpus built, the tab must say how to get one rather than go blank."""
    import streamlit as st

    import app.services.artifacts as artifacts

    monkeypatch.setattr(artifacts, "DATA_DIR", tmp_path / "no-data")
    monkeypatch.setattr(artifacts, "REPO_ROOT", tmp_path)
    # load_sample_reviews is st.cache_data, so an earlier test's sample would
    # otherwise be returned without the patched paths ever being read.
    st.cache_data.clear()

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("📁 Batch Review Inspector").run()

    assert not at.exception, [e.value for e in at.exception]
    messages = " ".join(i.value for i in at.info)
    assert "fetch_real_datasets.py" in messages, messages
