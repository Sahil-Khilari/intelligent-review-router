from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from app.db.database import get_db
from app.db.models import ReviewAnalysis, SupportTicket
from app.services.sentiment import SentimentAnalyzer
from app.services.absa import build_extractor
from app.services.routing import SupportRouter
from app.services.ticket_service import TicketService

router = APIRouter(prefix="/api/v1", tags=["Review Router Engine"])

# Initialize engine services singleton
sentiment_analyzer = SentimentAnalyzer()
aspect_extractor = build_extractor(sentiment_analyzer=sentiment_analyzer)
support_router = SupportRouter()
ticket_service = TicketService(sentiment_analyzer, aspect_extractor, support_router)

class ReviewRequest(BaseModel):
    review_text: str = Field(..., json_schema_extra={"example": "The phone is excellent, but the delivery was very late and my refund has not arrived."})
    customer_id: Optional[str] = Field("CUST-GUEST", json_schema_extra={"example": "CUST-10492"})
    product_id: Optional[str] = Field("PROD-GENERIC", json_schema_extra={"example": "PROD-SMARTPHONE"})

class TicketStatusUpdate(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "RESOLVED"})

@router.post("/analyze", summary="Analyze Customer Review")
def analyze_review(req: ReviewRequest, db: Session = Depends(get_db)):
    if not req.review_text.strip():
        raise HTTPException(status_code=400, detail="review_text cannot be empty")
        
    analysis = ticket_service.analyze_and_route(
        review_text=req.review_text,
        customer_id=req.customer_id,
        product_id=req.product_id
    )
    
    # Save analysis record to DB
    record = ReviewAnalysis(
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
        routing=analysis["routing"]
    )
    db.add(record)
    db.commit()
    
    # Auto-create ticket if sentiment is Negative or priority is URGENT/HIGH
    if analysis["overall_sentiment"] == "Negative" or analysis["overall_priority"] in ["HIGH", "URGENT"]:
        summary_txt = f"{analysis['overall_priority']} Issue for {analysis['primary_department']}: " + "; ".join([i['opinion'] for i in analysis['issues']])
        ticket = SupportTicket(
            ticket_id=analysis["ticket_id"],
            customer_id=analysis["customer_id"],
            product_id=analysis["product_id"],
            department=analysis["primary_department"],
            priority=analysis["overall_priority"],
            status="OPEN",
            review_text=analysis["review_text"],
            summary=summary_txt[:300]
        )
        db.add(ticket)
        db.commit()

    return analysis

@router.post("/tickets", summary="Manually Create Support Ticket")
def create_ticket(req: ReviewRequest, db: Session = Depends(get_db)):
    analysis = ticket_service.analyze_and_route(req.review_text, req.customer_id, req.product_id)
    summary_txt = f"{analysis['overall_priority']} Priority Ticket: " + "; ".join([i['opinion'] for i in analysis['issues']])
    
    ticket = SupportTicket(
        ticket_id=analysis["ticket_id"],
        customer_id=analysis["customer_id"],
        product_id=analysis["product_id"],
        department=analysis["primary_department"],
        priority=analysis["overall_priority"],
        status="OPEN",
        review_text=analysis["review_text"],
        summary=summary_txt[:300]
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket

@router.get("/tickets", summary="List Support Tickets")
def list_tickets(
    department: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(SupportTicket)
    if department:
        query = query.filter(SupportTicket.department == department)
    if priority:
        query = query.filter(SupportTicket.priority == priority)
    if status:
        query = query.filter(SupportTicket.status == status)
    
    tickets = query.order_by(SupportTicket.created_at.desc()).limit(limit).all()
    return tickets

@router.patch("/tickets/{ticket_id}", summary="Update Ticket Status")
def update_ticket_status(ticket_id: str, update: TicketStatusUpdate, db: Session = Depends(get_db)):
    ticket = db.query(SupportTicket).filter(SupportTicket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
        
    valid_statuses = ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"]
    if update.status.upper() not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")
        
    ticket.status = update.status.upper()
    ticket.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(ticket)
    return ticket

@router.get("/analytics", summary="Support Router Analytics Metrics")
def get_analytics(db: Session = Depends(get_db)):
    total_reviews = db.query(ReviewAnalysis).count()
    total_tickets = db.query(SupportTicket).count()
    
    sentiments = {}
    for row in db.query(ReviewAnalysis.overall_sentiment).all():
        s = row[0] or "Unknown"
        sentiments[s] = sentiments.get(s, 0) + 1
        
    departments = {}
    for row in db.query(SupportTicket.department).all():
        d = row[0] or "Unknown"
        departments[d] = departments.get(d, 0) + 1
        
    priorities = {}
    for row in db.query(SupportTicket.priority).all():
        p = row[0] or "Unknown"
        priorities[p] = priorities.get(p, 0) + 1

    statuses = {}
    for row in db.query(SupportTicket.status).all():
        st = row[0] or "Unknown"
        statuses[st] = statuses.get(st, 0) + 1

    return {
        "total_reviews_analyzed": total_reviews,
        "total_tickets_created": total_tickets,
        "sentiments": sentiments,
        "departments": departments,
        "priorities": priorities,
        "statuses": statuses
    }
