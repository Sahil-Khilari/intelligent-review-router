from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ReviewAnalysis, SupportTicket
from app.services.aspect_extraction import AspectExtractor
from app.services.routing import SupportRouter
from app.services.sentiment import SentimentAnalyzer
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/api/v1", tags=["Review Router Engine"])

MAX_REVIEW_CHARS = 20_000
VALID_STATUSES = ("OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED")

# Initialize engine services singleton
sentiment_analyzer = SentimentAnalyzer()
aspect_extractor = AspectExtractor(sentiment_analyzer=sentiment_analyzer)
support_router = SupportRouter()
ticket_service = TicketService(sentiment_analyzer, aspect_extractor, support_router)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReviewRequest(BaseModel):
    review_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_REVIEW_CHARS,
        json_schema_extra={"example": "The phone is excellent, but the delivery was very late and my refund has not arrived."},
    )
    customer_id: Optional[str] = Field("CUST-GUEST", max_length=50, json_schema_extra={"example": "CUST-10492"})
    product_id: Optional[str] = Field("PROD-GENERIC", max_length=50, json_schema_extra={"example": "PROD-SMARTPHONE"})

    @field_validator("review_text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review_text cannot be empty or whitespace only")
        return value


class TicketStatusUpdate(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "RESOLVED"})

    @field_validator("status")
    @classmethod
    def known_status(cls, value: str) -> str:
        upper = value.upper()
        if upper not in VALID_STATUSES:
            raise ValueError(f"status must be one of {list(VALID_STATUSES)}")
        return upper


class IssueOut(BaseModel):
    aspect: str
    opinion: str
    sentiment: str
    confidence: float
    department: str
    source: Optional[str] = None


class RoutingOut(BaseModel):
    department: str
    aspect: str
    priority: str
    reason: str


class AnalysisOut(BaseModel):
    ticket_id: str
    customer_id: str
    product_id: str
    review_text: str
    overall_sentiment: str
    overall_confidence: float
    model_type: str
    issues: list[IssueOut]
    primary_department: str
    overall_priority: str
    routing: list[RoutingOut]
    created_at: str


class TicketOut(BaseModel):
    id: int
    ticket_id: str
    customer_id: Optional[str] = None
    product_id: Optional[str] = None
    department: Optional[str] = None
    priority: Optional[str] = None
    status: str
    review_text: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class AnalyticsOut(BaseModel):
    total_reviews_analyzed: int
    total_tickets_created: int
    sentiments: dict[str, int]
    departments: dict[str, int]
    priorities: dict[str, int]
    statuses: dict[str, int]


def _summary(analysis: dict[str, Any], prefix: str) -> str:
    joined = "; ".join(i["opinion"] for i in analysis["issues"])
    return f"{prefix}: {joined}"[:300]


def _persist(db: Session, analysis: dict[str, Any], *, always_ticket: bool) -> Optional[SupportTicket]:
    """
    Writes the analysis and, when warranted, its ticket in one transaction.

    These used to be two separate commits, so a failure in between left an
    analysis on record with no ticket behind it.
    """
    db.add(ReviewAnalysis(
        ticket_id=analysis["ticket_id"],
        customer_id=analysis["customer_id"],
        product_id=analysis["product_id"],
        review_text=analysis["review_text"],
        overall_sentiment=analysis["overall_sentiment"],
        overall_confidence=analysis["overall_confidence"],
        model_type=analysis["model_type"],
        issues=analysis["issues"],
        primary_department=analysis["primary_department"],
        overall_priority=analysis["overall_priority"],
        routing=analysis["routing"],
    ))

    ticket = None
    warranted = (
        analysis["overall_sentiment"] == "Negative"
        or analysis["overall_priority"] in ("HIGH", "URGENT")
    )
    if always_ticket or warranted:
        prefix = (
            f"{analysis['overall_priority']} Priority Ticket" if always_ticket
            else f"{analysis['overall_priority']} Issue for {analysis['primary_department']}"
        )
        ticket = SupportTicket(
            ticket_id=analysis["ticket_id"],
            customer_id=analysis["customer_id"],
            product_id=analysis["product_id"],
            department=analysis["primary_department"],
            priority=analysis["overall_priority"],
            status="OPEN",
            review_text=analysis["review_text"],
            summary=_summary(analysis, prefix),
        )
        db.add(ticket)

    db.commit()
    if ticket is not None:
        db.refresh(ticket)
    return ticket


@router.post("/analyze", response_model=AnalysisOut, summary="Analyze Customer Review")
def analyze_review(req: ReviewRequest, db: Session = Depends(get_db)):
    analysis = ticket_service.analyze_and_route(
        review_text=req.review_text,
        customer_id=req.customer_id,
        product_id=req.product_id,
    )
    _persist(db, analysis, always_ticket=False)
    return analysis


@router.post("/tickets", response_model=TicketOut, summary="Manually Create Support Ticket")
def create_ticket(req: ReviewRequest, db: Session = Depends(get_db)):
    analysis = ticket_service.analyze_and_route(req.review_text, req.customer_id, req.product_id)
    # always_ticket: a manual create is an explicit request, and the analysis
    # is recorded too so it is counted by /analytics like any other review.
    return _persist(db, analysis, always_ticket=True)


@router.get("/tickets", response_model=list[TicketOut], summary="List Support Tickets")
def list_tickets(
    department: Optional[str] = Query(None, max_length=50),
    priority: Optional[Literal["LOW", "MEDIUM", "HIGH", "URGENT"]] = Query(None),
    status: Optional[Literal["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"]] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    query = db.query(SupportTicket)
    if department:
        query = query.filter(SupportTicket.department == department)
    if priority:
        query = query.filter(SupportTicket.priority == priority)
    if status:
        query = query.filter(SupportTicket.status == status)

    return (
        query.order_by(SupportTicket.created_at.desc(), SupportTicket.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.patch("/tickets/{ticket_id}", response_model=TicketOut, summary="Update Ticket Status")
def update_ticket_status(ticket_id: str, update: TicketStatusUpdate, db: Session = Depends(get_db)):
    ticket = db.query(SupportTicket).filter(SupportTicket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    # `updated_at` has onupdate=utcnow on the column; setting it here too only
    # introduced a second clock.
    ticket.status = update.status
    db.commit()
    db.refresh(ticket)
    return ticket


def _counts(db: Session, column) -> dict[str, int]:
    """One GROUP BY instead of loading every row and counting in Python."""
    rows = db.query(column, func.count()).group_by(column).all()
    return {(value or "Unknown"): count for value, count in rows}


@router.get("/analytics", response_model=AnalyticsOut, summary="Support Router Analytics Metrics")
def get_analytics(db: Session = Depends(get_db)):
    return AnalyticsOut(
        total_reviews_analyzed=db.query(func.count(ReviewAnalysis.id)).scalar() or 0,
        total_tickets_created=db.query(func.count(SupportTicket.id)).scalar() or 0,
        sentiments=_counts(db, ReviewAnalysis.overall_sentiment),
        departments=_counts(db, SupportTicket.department),
        priorities=_counts(db, SupportTicket.priority),
        statuses=_counts(db, SupportTicket.status),
    )
