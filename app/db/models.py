from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from app.db.database import Base

class ReviewAnalysis(Base):
    __tablename__ = "review_analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(50), unique=True, index=True)
    customer_id = Column(String(50), default="CUST-GUEST")
    product_id = Column(String(50), default="PROD-GENERIC")
    review_text = Column(Text, nullable=False)
    overall_sentiment = Column(String(20), index=True)
    overall_confidence = Column(Float)
    model_type = Column(String(30))
    issues = Column(JSON)
    primary_department = Column(String(50), index=True)
    overall_priority = Column(String(20), index=True)
    routing = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(String(50), unique=True, index=True)
    customer_id = Column(String(50))
    product_id = Column(String(50))
    department = Column(String(50), index=True)
    priority = Column(String(20), index=True)
    status = Column(String(20), default="OPEN", index=True)  # OPEN, IN_PROGRESS, RESOLVED, CLOSED
    review_text = Column(Text)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
