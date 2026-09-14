import os
import json
import html
import pandas as pd
import streamlit as st
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Review Router AI - Support Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .badge-positive {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-neutral {
        background-color: #9e6a03;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-negative {
        background-color: #da3633;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-urgent {
        background-color: #bd561d;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-high {
        background-color: #d97706;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-medium {
        background-color: #2563eb;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-low {
        background-color: #4b5563;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
    }
    .issue-box {
        background: rgba(22, 27, 34, 0.8);
        border-left: 4px solid #58a6ff;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Cache loaded services
@st.cache_resource
def load_services():
    from app.services.sentiment import SentimentAnalyzer
    from app.services.aspect_extraction import AspectExtractor
    from app.services.routing import SupportRouter
    from app.services.ticket_service import TicketService
    from app.db.database import SessionLocal, Base, engine
    from app.db.models import SupportTicket, ReviewAnalysis
    
    Base.metadata.create_all(bind=engine)
    
    sentiment_analyzer = SentimentAnalyzer()
    aspect_extractor = AspectExtractor(sentiment_analyzer=sentiment_analyzer)
    support_router = SupportRouter()
    ticket_service = TicketService(sentiment_analyzer, aspect_extractor, support_router)
    
    return ticket_service, SessionLocal

try:
    ticket_service, SessionLocal = load_services()
except Exception as e:
    st.error(f"Error loading system services: {e}")
    st.stop()

# Sidebar
st.sidebar.title("⚡ Review Router AI")
st.sidebar.caption("Intelligent E-Commerce Customer Review Analyzer & Router")

nav_choice = st.sidebar.radio(
    "Navigation Desk",
    ["🔍 Live Review Analyzer", "📁 Batch Review Inspector", "🎫 Support Ticket Desk", "📊 Analytics & Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**System Architecture:**
- **Sentiment Model:** DistilBERT / TF-IDF
- **Aspect Engine:** Rule & Context Guided ATE
- **Department Routing:** Automated Rule Engine
- **Backend:** FastAPI & SQLAlchemy
""")

# TAB 1: LIVE REVIEW ANALYZER
if nav_choice == "🔍 Live Review Analyzer":
    st.title("🔍 Live Review Analyzer & Router")
    st.caption("Paste customer feedback to analyze multi-aspect sentiment, assign priority, and route to the appropriate support team.")
    
    col_presets, _ = st.columns([3, 1])
    with col_presets:
        preset_choice = st.selectbox(
            "Quick Demo Examples:",
            [
                "Custom Text Input",
                "The phone is excellent, but the delivery was very late and my refund has not arrived.",
                "Charged twice for my order! Payment failed but money was deducted immediately.",
                "Top notch quality! Arrived earlier than expected and packaging was pristine.",
                "Support agent was disconnected and useless when I asked about return pickup."
            ]
        )
    
    default_text = "" if preset_choice == "Custom Text Input" else preset_choice
    
    review_input = st.text_area(
        "Customer Review Text:",
        value=default_text,
        height=130,
        placeholder="Enter customer review here..."
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        cust_id = st.text_input("Customer ID", value="CUST-98214")
    with col2:
        prod_id = st.text_input("Product ID", value="PROD-PHONE-X")
        
    if st.button("🚀 Analyze & Route Review", type="primary"):
        if not review_input.strip():
            st.warning("Please enter a valid review text.")
        else:
            with st.spinner("Processing NLP models & routing logic..."):
                st.session_state["analysis"] = ticket_service.analyze_and_route(
                    review_input, cust_id, prod_id
                )
                st.session_state.pop("ticket_saved", None)

    # Rendered outside the button block: st.button is only True on the run
    # immediately after its own click, so anything nested inside another
    # button's block disappears the moment it is pressed.
    if st.session_state.get("analysis"):
            res = st.session_state["analysis"]
            st.markdown("---")
            st.subheader("📋 Analysis & Routing Summary")
            
            m1, m2, m3, m4 = st.columns(4)
            
            # Sentiment Badge
            sent = res["overall_sentiment"]
            sent_color = "badge-positive" if sent == "Positive" else ("badge-negative" if sent == "Negative" else "badge-neutral")
            m1.markdown(f"**Overall Sentiment**<br><span class='{sent_color}'>{sent}</span> ({res['overall_confidence']*100:.1f}%)", unsafe_allow_html=True)
            
            # Department Badge
            m2.markdown(f"**Primary Department**<br>📌 **{res['primary_department']}**", unsafe_allow_html=True)
            
            # Priority Badge
            prio = res["overall_priority"]
            prio_color = f"badge-{prio.lower()}"
            m3.markdown(f"**Calculated Priority**<br><span class='{prio_color}'>{prio}</span>", unsafe_allow_html=True)
            
            # Model Engine Badge
            m4.markdown(f"**NLP Engine**<br>🤖 `{res['model_type'].upper()}`", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Aspect & Opinion Breakdown
            st.markdown("### 🧩 Detected Aspects & Opinion Clauses")
            for issue in res["issues"]:
                st.markdown(f"""
                <div class='issue-box'>
                    <strong>Aspect:</strong> <code>{issue['aspect'].upper()}</code> &nbsp;|&nbsp;
                    <strong>Department:</strong> {issue['department']} &nbsp;|&nbsp;
                    <strong>Sentiment:</strong> <b>{issue['sentiment']}</b> ({issue['confidence']*100:.0f}% conf)<br>
                    <em>"{html.escape(str(issue['opinion']))}"</em>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### 🔀 Routing Matrix")
            df_route = pd.DataFrame(res["routing"])
            st.dataframe(df_route, width='stretch')
            
            # Create Ticket Option
            st.markdown("---")
            if st.session_state.get("ticket_saved"):
                st.success(
                    f"Support Ticket `{st.session_state['ticket_saved']}` dispatched to "
                    f"{res['primary_department']}!"
                )
            elif st.button("➕ Create Support Ticket in Database"):
                db = SessionLocal()
                try:
                    from app.db.models import SupportTicket
                    summary_txt = f"{res['overall_priority']} Issue for {res['primary_department']}: " + "; ".join([i['opinion'] for i in res['issues']])
                    ticket = SupportTicket(
                        ticket_id=res["ticket_id"],
                        customer_id=res["customer_id"],
                        product_id=res["product_id"],
                        department=res["primary_department"],
                        priority=res["overall_priority"],
                        status="OPEN",
                        review_text=res["review_text"],
                        summary=summary_txt[:300]
                    )
                    db.add(ticket)
                    db.commit()
                    st.session_state["ticket_saved"] = res["ticket_id"]
                finally:
                    db.close()
                st.rerun()
                    
            with st.expander("🛠️ Raw JSON Response Payload"):
                st.json(res)

# TAB 2: BATCH REVIEW INSPECTOR
elif nav_choice == "📁 Batch Review Inspector":
    st.title("📁 Batch Review Inspector")
    st.caption("Upload a dataset CSV or process sample dataset to evaluate bulk routing performance.")
    
    data_file = st.file_uploader("Upload Reviews CSV (Must contain 'review' column)", type=["csv"])
    
    if data_file is not None:
        batch_df = pd.read_csv(data_file)
    else:
        st.info("No CSV uploaded. Showing sample records from dataset.")
        sample_path = "data/processed/ecommerce_reviews.csv"
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path).head(50)
        else:
            batch_df = pd.DataFrame()
            
    if not batch_df.empty and 'review' not in batch_df.columns:
        st.error(
            "This CSV has no `review` column. Found: "
            + ", ".join(f"`{c}`" for c in batch_df.columns)
        )
        batch_df = pd.DataFrame()

    if not batch_df.empty:
        st.write(f"Loaded **{len(batch_df)}** reviews for batch processing.")
        preview_cols = [c for c in ['review_id', 'review', 'sentiment', 'priority'] if c in batch_df.columns]
        st.dataframe(batch_df[preview_cols].head(10), width='stretch')
        
        if st.button("⚡ Run Batch Analysis"):
            results = []
            progress_bar = st.progress(0)
            
            for position, (_, row) in enumerate(batch_df.iterrows(), start=1):
                rev = str(row['review'])
                res = ticket_service.analyze_and_route(rev)
                results.append({
                    "review_text": rev,
                    "predicted_sentiment": res["overall_sentiment"],
                    "confidence": res["overall_confidence"],
                    "primary_department": res["primary_department"],
                    "overall_priority": res["overall_priority"],
                    "detected_aspects": ", ".join([i['aspect'] for i in res['issues']])
                })
                progress_bar.progress(position / len(batch_df))
                
            res_df = pd.DataFrame(results)
            st.success("Batch analysis complete!")
            st.dataframe(res_df, width='stretch')
            
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Routing Report (CSV)",
                data=csv_data,
                file_name="batch_review_routing_results.csv",
                mime="text/csv"
            )

# TAB 3: SUPPORT TICKET DESK
elif nav_choice == "🎫 Support Ticket Desk":
    st.title("🎫 Support Ticket Desk Queue")
    st.caption("Manage, filter, and resolve routed customer support tickets.")
    
    db = SessionLocal()
    try:
        from app.db.models import SupportTicket
        
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            dept_filter = st.selectbox("Department Filter", ["All", "Logistics", "Product Support", "Payments", "Refunds", "Returns / Replacement", "Customer Support"])
        with f_col2:
            prio_filter = st.selectbox("Priority Filter", ["All", "LOW", "MEDIUM", "HIGH", "URGENT"])
        with f_col3:
            status_filter = st.selectbox("Status Filter", ["All", "OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"])
            
        query = db.query(SupportTicket)
        if dept_filter != "All":
            query = query.filter(SupportTicket.department == dept_filter)
        if prio_filter != "All":
            query = query.filter(SupportTicket.priority == prio_filter)
        if status_filter != "All":
            query = query.filter(SupportTicket.status == status_filter)
            
        tickets = query.order_by(SupportTicket.created_at.desc()).all()
        
        st.write(f"Total Tickets Matching Filters: **{len(tickets)}**")
        
        if tickets:
            ticket_data = []
            for t in tickets:
                ticket_data.append({
                    "Ticket ID": t.ticket_id,
                    "Customer": t.customer_id,
                    "Department": t.department,
                    "Priority": t.priority,
                    "Status": t.status,
                    "Summary": t.summary,
                    "Created At": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else ""
                })
            st.dataframe(pd.DataFrame(ticket_data), width='stretch')
            
            st.markdown("---")
            st.subheader("✏️ Quick Update Ticket Status")
            u_col1, u_col2, u_col3 = st.columns([2, 2, 1])
            
            with u_col1:
                t_ids = [t.ticket_id for t in tickets]
                selected_t_id = st.selectbox("Select Ticket ID", t_ids)
            with u_col2:
                new_status = st.selectbox("New Status", ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"])
            with u_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Update Status"):
                    t_obj = db.query(SupportTicket).filter(SupportTicket.ticket_id == selected_t_id).first()
                    if t_obj:
                        t_obj.status = new_status
                        db.commit()
                        st.success(f"Ticket {selected_t_id} updated to {new_status}")
                        st.rerun()
        else:
            st.info("No tickets found matching current filters.")
    finally:
        db.close()

# TAB 4: ANALYTICS & INSIGHTS
elif nav_choice == "📊 Analytics & Insights":
    st.title("📊 Analytics & Department Insights")
    st.caption("Aggregated analytics on customer review sentiment distribution and department workload.")
    
    db = SessionLocal()
    try:
        from app.db.models import SupportTicket, ReviewAnalysis
        
        total_revs = db.query(ReviewAnalysis).count()
        total_ticks = db.query(SupportTicket).count()
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Reviews Processed", total_revs)
        k2.metric("Total Tickets Created", total_ticks)
        k3.metric("Ticket Conversion Rate", f"{(total_ticks/total_revs*100):.1f}%" if total_revs > 0 else "0%")
        
        st.markdown("---")
        
        # Load processed CSV for visualizations
        st.subheader("Live routing decisions")
        st.caption("Aggregated from reviews this system has actually analysed.")

        live = pd.DataFrame(
            db.query(ReviewAnalysis.overall_sentiment, ReviewAnalysis.primary_department,
                     ReviewAnalysis.overall_priority).all(),
            columns=["sentiment", "department", "priority"],
        )

        if live.empty:
            st.info("No reviews analysed yet. Use the Live Review Analyzer to populate this.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sentiment breakdown**")
                st.bar_chart(live['sentiment'].value_counts())
            with c2:
                st.markdown("**Department workload**")
                st.bar_chart(live['department'].value_counts())

            st.markdown("**Priority distribution by department**")
            st.dataframe(pd.crosstab(live['department'], live['priority']), width='stretch')

        # The training corpus is shown separately and labelled as such. It used
        # to be rendered under the same heading as the live metrics, so the
        # dashboard presented generated data as system output.
        sample_path = "data/processed/ecommerce_reviews.csv"
        if os.path.exists(sample_path):
            with st.expander("Reference: synthetic training corpus (not live data)"):
                df_analytics = pd.read_csv(sample_path)
                st.caption(
                    f"{len(df_analytics)} generated reviews used to train the model. "
                    "Shown for comparison only."
                )
                st.bar_chart(df_analytics['sentiment'].value_counts())
                st.dataframe(
                    pd.crosstab(df_analytics['department'], df_analytics['priority']),
                    width='stretch',
                )
    finally:
        db.close()
