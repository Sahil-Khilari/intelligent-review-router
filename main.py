from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import Base, engine
from app.api.routes import router as api_router

# Create database tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Intelligent E-Commerce Customer Review Analyzer & Support Router",
    description="NLP-based e-commerce review analyzer with sentiment classification, aspect-based extraction, priority calculation, and support department routing.",
    version="1.0.0"
)

# Enable CORS for Streamlit / external frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/health", summary="Liveness and model readiness")
def health():
    """
    Reports which models are actually answering, and on what device.

    The service stays up on fallbacks when artifacts are missing, so "is it
    running" and "is it running the models I think" are different questions.
    This answers the second one.
    """
    from app.api.routes import aspect_extractor, sentiment_analyzer
    from app.services.device import describe_device

    sentiment = sentiment_analyzer.describe()
    absa = (aspect_extractor.describe()
            if hasattr(aspect_extractor, "describe")
            else {"loaded": False, "artifacts_present": False,
                  "note": "keyword lexicon fallback"})

    degraded = sentiment["model_type"] != "transformer" or not absa.get("artifacts_present")
    return {
        "status": "ok",
        "degraded": degraded,
        "device": describe_device(),
        "models": {"sentiment": sentiment, "absa": absa},
    }


@app.get("/", summary="Root Health Check")
def root():
    return {
        "status": "online",
        "service": "Intelligent Review Router API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
