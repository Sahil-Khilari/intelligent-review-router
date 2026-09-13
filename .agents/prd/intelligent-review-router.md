Intelligent E-Commerce Customer Review Analyzer & Support Router

Complete End-to-End Implementation Guide

Audience: Beginner who has used Python/ML libraries but has never
fine-tuned a Transformer model.

Goal: Build the entire project from scratch: dataset →
preprocessing → sentiment model → aspect/opinion extraction → aspect
sentiment → priority → department routing → database → FastAPI →
Streamlit → testing → deployment.

0. First Understand What We Are Building

The project takes a customer review and converts it into a structured
support decision.

Example input

"The phone is excellent, but the delivery was very late and my refund
has not arrived."

Desired output

{
"overall_sentiment": "Negative",
"overall_confidence": 0.95,
"issues": [
{
"aspect": "delivery",
"opinion": "very late",
"sentiment": "Negative",
"confidence": 0.93
},
{
"aspect": "refund_return",
"opinion": "has not arrived",
"sentiment": "Negative",
"confidence": 0.91
}
],
"routing": [
{
"department": "Logistics",
"priority": "HIGH"
},
{
"department": "Refunds",
"priority": "HIGH"
}
]
}

The important distinction is:

ML/NLP
↓
"What is the customer saying?"

Routing/business logic
↓
"Which department should handle it?"

Priority logic
↓
"How urgently should it be handled?"

Routing is not another ML model by default. It is a business-rule
layer that consumes the ML output.

1. Project Scope

The project has four major intelligence tasks:

Overall Sentiment Analysis

Positive

Neutral

Negative

Aspect Extraction

Find the problem/topic mentioned in the review.

Examples: delivery, payment, refund, product quality.

Aspect-Level Sentiment / Opinion

Identify what is said about each aspect.

Example:

delivery → "very late" → Negative

product_quality → "excellent" → Positive

Smart Support Routing

Map detected issues to departments.

Assign priority.

Optionally create a support ticket.

2. Research Papers: Why They Matter

The project is based on three research directions.

Paper 1 --- Aspect-Based Sentiment Classification

The first paper focuses on aspect-level sentiment classification in
e-commerce reviews.

The basic idea:

"The phone is good but delivery is late."

phone → Positive
delivery → Negative

This is more informative than only saying:

Overall → Negative

The paper evaluates ML/Transformer approaches for e-commerce
aspect-based sentiment classification and reports strong performance
from RoBERTa.

What we take from Paper 1

Use aspect-level sentiment instead of relying only on one overall
sentiment label.

Paper 2 --- Aspect Extraction + Sentiment as Related Tasks

The second paper separates ABSA into two related tasks:

ATE --- Aspect Term Extraction

APC --- Aspect Polarity Classification

Example:

"The delivery was extremely late."

ATE:
delivery

APC:
delivery → Negative

The paper studies multitask/joint learning, where related tasks can help
each other.

What we take from Paper 2

Understand that:

Find the aspect

- Determine its sentiment

are related NLP tasks.

For this student project, do not try to reproduce the complete
research architecture immediately. Start with a simpler implementation
and improve it only if the data and time support it.

Paper 3 --- Long / Multiple-Issue E-Commerce Reviews

The third paper focuses on long reviews containing multiple aspects and
introduces structured annotations such as:

Target
Aspect
Opinion
Sentiment

Example:

"The phone is excellent, but the screen is damaged and my refund hasn't arrived."

phone → excellent → Positive
screen → damaged → Negative
refund → hasn't arrived → Negative

What we take from Paper 3

A real review may contain multiple issues, so the system should
preserve multiple aspect-level results instead of forcing the whole
review into one label.

3. Important Beginner Decision: Do Not Start With a Huge Model

You said you have never trained a big model before.

That is completely fine.

Do not begin by trying to build a large Transformer architecture
from scratch.

We will use:

Baseline first
↓
Understand the data
↓
Train simple model
↓
Fine-tune a pretrained Transformer
↓
Add aspect extraction

You are not training BERT/DistilBERT from zero.

You start with a pretrained model and fine-tune it on your task.

4. Recommended Technology Stack

Machine Learning

Python
Pandas
NumPy
Scikit-learn
PyTorch
Hugging Face Transformers
Hugging Face Datasets

NLP

Hugging Face Tokenizers
Transformer model

Recommended first Transformer:

DistilBERT

Why?

Easier and lighter than a large BERT-style model.

Fast enough for a student project.

Uses the same general Transformer fine-tuning workflow.

Good starting point for learning.

Backend

FastAPI
Uvicorn
Pydantic

Database

Start with:

SQLite
SQLAlchemy

Move to PostgreSQL only if needed.

Frontend

Streamlit

Testing

Pytest

Version control

Git
GitHub

5. Recommended Repository

Repository name

ecommerce-review-router

GitHub description

NLP-based e-commerce customer review analyzer that performs sentiment analysis, aspect extraction, priority detection, and intelligent support department routing.

6. Final Repository Structure

Create this gradually. You do not need to create every file on day
one.

ecommerce-review-router/
│
├── app/
│ ├── main.py
│ │
│ ├── api/
│ │ └── routes.py
│ │
│ ├── services/
│ │ ├── preprocessing.py
│ │ ├── sentiment.py
│ │ ├── aspect_extraction.py
│ │ ├── routing.py
│ │ └── ticket_service.py
│ │
│ ├── models/
│ │ └── trained_model/
│ │
│ └── db/
│ ├── database.py
│ └── models.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_sentiment_baseline.ipynb
│ ├── 03_transformer_sentiment.ipynb
│ └── 04_aspect_extraction.ipynb
│
├── tests/
│ ├── test_sentiment.py
│ ├── test_aspects.py
│ ├── test_routing.py
│ └── test_api.py
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md

7. PHASE 1 --- Set Up the Project

Step 1: Create the GitHub repository

Create:

ecommerce-review-router

Then clone it.

git clone <your-repository-url>
cd ecommerce-review-router

Step 2: Create a virtual environment

Windows:

python -m venv .venv

Activate:

.venv\Scripts\activate

Check:

python --version

Step 3: Install initial packages

Start with:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Later install Transformer dependencies:

pip install torch transformers datasets accelerate

Backend:

pip install fastapi uvicorn pydantic

Database:

pip install sqlalchemy

Testing:

pip install pytest

Save dependencies:

pip freeze > requirements.txt

8. PHASE 2 --- Dataset

This is where the real ML work starts.

Do not train the model yet.

First find a dataset that supports your project.

You need at minimum:

review text
sentiment label

For example:

review sentiment

"The product is amazing" Positive
"Delivery was terrible" Negative
"The product is okay" Neutral

For aspect-based work, you ideally need additional annotations:

review aspect opinion sentiment

"Delivery was delivery extremely late Negative
extremely late"

"Camera quality product_quality excellent Positive
is excellent"

Critical point

A normal sentiment dataset is not automatically enough to train an
aspect extraction model.

Before choosing the final dataset, inspect its labels.

9. PHASE 3 --- Exploratory Data Analysis

Create:

notebooks/01_eda.ipynb

Load the dataset.

import pandas as pd

df = pd.read_csv("path/to/dataset.csv")

df.head()

Inspect:

df.shape
df.info()
df.columns
df.isnull().sum()
df.duplicated().sum()

Check sentiment distribution:

df["sentiment"].value_counts()

Visualize:

df["sentiment"].value_counts().plot(kind="bar")

Check review length:

df["review_length"] = df["review"].astype(str).str.len()

df["review_length"].describe()

Inspect actual examples:

df[["review", "sentiment"]].sample(20)

Questions you must answer after EDA

How many reviews?

Which sentiment classes exist?

Is the dataset balanced?

Are there missing reviews?

Are there duplicate reviews?

Are labels clean?

Are the reviews actually e-commerce reviews?

Does the dataset contain aspect information?

Are aspect labels compatible with our department categories?

Do not skip this phase.

10. PHASE 4 --- Decide Your Label Taxonomy

Before training aspect extraction, define the project's supported
categories.

Start with:

delivery
product_quality
payment
refund_return
packaging
customer_support

Example mapping:

delivery
↓
Logistics

product_quality
↓
Product Support

payment
↓
Payments

refund_return
↓
Refunds

packaging
↓
Returns / Replacement

customer_support
↓
Customer Support

Keep the number of categories small initially.

You can add categories later.

11. PHASE 5 --- Sentiment Baseline

Before using Transformers, build a simple baseline.

Why?

Because you need to know whether the more advanced model actually
improves performance.

The baseline:

Review
↓
TF-IDF
↓
Logistic Regression
↓
Sentiment

Step 1: Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
df["review"],
df["sentiment"],
test_size=0.2,
random_state=42,
stratify=df["sentiment"]
)

Step 2: TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
max_features=10000,
ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

Step 3: Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

Predict:

predictions = model.predict(X_test_tfidf)

Evaluate:

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

Record:

Accuracy
Precision
Recall
F1-score

This becomes your baseline.

12. PHASE 6 --- Understand What a Transformer Actually Does

You do not need to understand every mathematical detail before using it.

Think of the model as:

Raw sentence
↓
Tokenizer
↓
Numbers / token IDs
↓
Pretrained Transformer
↓
Text representation
↓
Classification layer
↓
Positive / Neutral / Negative

Example:

"I love this phone"
↓
Tokenizer
↓
[101, ..., ...]
↓
DistilBERT
↓
Classifier
↓
Positive

Pretrained vs training from scratch

You are doing:

Pretrained DistilBERT +
Your review dataset
↓
Fine-tuning

You are NOT doing:

Random neural network
↓
Learn English from zero

That would require enormous data and compute.

13. PHASE 7 --- Fine-Tune DistilBERT for Overall Sentiment

Create:

notebooks/03_transformer_sentiment.ipynb

First convert labels:

Negative → 0
Neutral → 1
Positive → 2

Use a tokenizer.

Conceptually:

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

Tokenize:

def tokenize(batch):
return tokenizer(
batch["review"],
padding="max_length",
truncation=True,
max_length=128
)

Then load the classification model:

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
model_name,
num_labels=3
)

Then train using the Hugging Face training workflow.

Do not try to write the entire training loop manually at first.

Use:

Dataset
↓
Tokenizer
↓
DistilBERT
↓
Trainer
↓
Fine-tuned model

14. PHASE 8 --- Evaluate the Transformer

Do not just say:

"BERT worked."

Compare:

TF-IDF + Logistic Regression
vs
DistilBERT

Example report format:

Model Accuracy F1

TF-IDF + Logistic Regression X X
DistilBERT Y Y

Also inspect:

Confusion Matrix
Classification Report

If DistilBERT does not improve substantially, investigate the data
instead of blindly making the model bigger.

15. PHASE 9 --- Save the Sentiment Model

Once the sentiment model works:

app/models/trained_model/sentiment/

Save:

model
tokenizer
label mapping

Your API should later load this model rather than retraining it every
time.

The production flow should be:

Application starts
↓
Load trained model
↓
Wait for review

NOT:

Every API request
↓
Train model

Training and inference are separate operations.

16. PHASE 10 --- Aspect Extraction

This is the most important part to understand before implementing it.

Overall sentiment

Question:

"How does the customer feel about the review overall?"

Example:

Review → Negative

Aspect extraction

Question:

"What is the customer talking about?"

Example:

"The delivery was late"

Aspect → delivery

Opinion extraction

Question:

"What words express the customer's opinion about that aspect?"

delivery → "late"

So:

Aspect = delivery
Opinion = late

17. Three Possible Approaches for Aspect Extraction

Do them in this order.

Approach A --- Keyword baseline

Create a simple dictionary.

aspect_keywords = {
"delivery": [
"delivery", "delayed", "late", "shipment", "shipping"
],
"payment": [
"payment", "charged", "transaction"
],
"refund_return": [
"refund", "return", "money back"
],
"product_quality": [
"broken", "damaged", "quality", "defective"
]
}

This is NOT your final intelligent model.

It is a baseline to understand the problem.

Approach B --- Text classification

If your dataset provides one aspect label per example, you can train an
aspect classifier.

Review
↓
Transformer
↓
Aspect category

Example:

"Money was deducted but payment failed"
↓
payment

If reviews can have multiple aspects, you need a multi-label
classification setup rather than ordinary single-label classification.

Approach C --- Token-level aspect extraction

For true ATE, the model identifies which words are aspect terms.

Example:

"The delivery was extremely late"

The → O
delivery → B-ASPECT
was → O
extremely → O
late → O

This is usually formulated as token classification / sequence
labeling.

A BIO scheme is common:

B-ASPECT = beginning of an aspect
I-ASPECT = continuation of an aspect
O = not an aspect

Example:

"poor customer service"

poor → O
customer → B-ASPECT
service → I-ASPECT

This is the more faithful implementation of ATE.

18. What Should You Implement?

Since you are learning Transformer training for the first time:

Version 1

Build:

Overall sentiment → Transformer
Aspect category → simple supervised classifier
Routing → rules
Priority → rules

Version 2

Improve aspect extraction:

Transformer token classification
↓
Aspect terms

Do not start with the most complex research architecture.

19. PHASE 11 --- Aspect Sentiment

Once aspects are available, determine sentiment for each aspect.

Example:

"Camera is amazing but delivery is terrible."

Output:

camera
opinion = amazing
sentiment = Positive

delivery
opinion = terrible
sentiment = Negative

You can initially implement this with a sentiment classifier applied to
the relevant sentence/aspect context.

For example:

"delivery is terrible"
↓
sentiment model
↓
Negative

Later, if the dataset supports proper aspect-level sentiment labels,
fine-tune a dedicated aspect sentiment model.

20. PHASE 12 --- Create One Unified Analysis Object

At this stage, make one Python function responsible for combining model
outputs.

Conceptually:

def analyze_review(text):
overall_sentiment = sentiment_model(text)

    aspects = aspect_model(text)

    # determine aspect-level sentiment
    issues = ...

    return {
        "overall_sentiment": overall_sentiment,
        "issues": issues
    }

The important thing is that your ML modules should return structured
data, not random strings.

Example:

{
"aspect": "delivery",
"opinion": "very late",
"sentiment": "Negative",
"confidence": 0.93
}

21. PHASE 13 --- Routing Engine

Now the ML part gives you:

aspect = delivery
sentiment = Negative

The routing engine decides:

delivery
↓
Logistics

This is ordinary application logic.

Example:

DEPARTMENT_MAP = {
"delivery": "Logistics",
"product_quality": "Product Support",
"payment": "Payments",
"refund_return": "Refunds",
"packaging": "Returns / Replacement",
"customer_support": "Customer Support"
}

Then:

department = DEPARTMENT_MAP.get(
aspect,
"Manual Review"
)

22. PHASE 14 --- Priority Engine

Create another independent function.

Example rules:

Strong negative / angry +
critical issue
↓
HIGH

Possible basic policy:

HIGH:

- severe negative sentiment
- payment charged but failed
- refund not received
- damaged/broken product
- explicit urgent language

MEDIUM:

- normal negative complaint
- repeated delay
- unresolved support problem

LOW:

- minor complaint
- neutral inquiry
- mild inconvenience

Do not pretend this is an ML model unless you actually train a priority
model.

It is a rule-based business layer.

23. PHASE 15 --- Multiple Issues

This is where Paper 3 becomes important.

Input:

"The product is great, but the package was damaged and my refund is
still pending."

The system should not return only:

Negative

Instead:

issues = [
{
"aspect": "packaging",
"opinion": "damaged",
"sentiment": "Negative"
},
{
"aspect": "refund_return",
"opinion": "still pending",
"sentiment": "Negative"
}
]

Then routing can create multiple department assignments:

packaging
↓
Returns / Replacement

refund_return
↓
Refunds

24. PHASE 16 --- Database

Start simple with SQLite.

Use three logical entities.

reviews

id
text
overall_sentiment
overall_confidence
created_at

review_issues

id
review_id
aspect
opinion
sentiment
confidence

tickets

id
review_id
department
priority
routing_reason
status
created_at

Relationships:

Review
│
├── Issue
├── Issue
└── Issue
│
└── Ticket / Routing

25. PHASE 17 --- FastAPI Backend

Now wrap the system in an API.

Recommended endpoint:

POST /reviews/analyze

Request:

{
"text": "Payment failed and my money was deducted."
}

Backend flow:

POST /reviews/analyze
↓
Validate request
↓
Preprocess
↓
Overall sentiment model
↓
Aspect extraction
↓
Aspect sentiment
↓
Priority engine
↓
Routing engine
↓
Save to DB
↓
Return JSON

Response:

{
"review_id": 101,
"overall_sentiment": "Negative",
"overall_confidence": 0.96,
"issues": [
{
"aspect": "payment",
"opinion": "failed",
"sentiment": "Negative",
"confidence": 0.95
}
],
"routing": [
{
"department": "Payments",
"priority": "HIGH",
"reason": "Payment issue with money deduction"
}
]
}

26. API Endpoints

Start with only these:

Analyze review

POST /reviews/analyze

Get ticket

GET /tickets/{ticket_id}

List tickets

GET /tickets

Health check

GET /health

Do not build 20 endpoints for a mini-project.

27. PHASE 18 --- Streamlit Dashboard

Only after the API works.

Basic UI:

---

       E-Commerce Review Analyzer

---

Enter customer review:

[ The delivery was very late and I want a refund ]

                [ ANALYZE ]

---

Overall Sentiment
NEGATIVE
Confidence: 95%

Detected Issues

Delivery
Opinion: very late
Sentiment: Negative
Confidence: 93%

Refund
Opinion: want a refund
Sentiment: Negative
Confidence: 91%

---

Routing

Department Priority
Logistics HIGH
Refunds HIGH

---

The UI should demonstrate the whole pipeline, not just display one
sentiment label.

28. PHASE 19 --- Testing

Test each layer independently.

Sentiment

Input:

"I love this product."

Expected:

Positive

Aspect

Input:

"Delivery was extremely late."

Expected:

delivery

Routing

Input:

aspect = delivery

Expected:

Logistics

Priority

Input:

sentiment = Negative
critical_issue = True

Expected:

HIGH

API

Send:

{
"text": "My payment failed and money was deducted."
}

Verify that the response contains:

sentiment
issues
department
priority

29. PHASE 20 --- End-to-End Test

Now test the complete pipeline with real examples.

Example 1

"The product is excellent."

Expected:

Overall: Positive
Issue: product_quality
Aspect sentiment: Positive
Priority: LOW

Example 2

"Delivery was five days late."

Expected:

Overall: Negative
Aspect: delivery
Department: Logistics
Priority: MEDIUM

Example 3

"Payment failed twice and money was deducted. This is extremely frustrating!"

Expected:

Overall: Negative
Aspect: payment
Department: Payments
Priority: HIGH

Example 4

"The phone is good but the screen is broken and my refund hasn't arrived."

Expected:

phone → Positive

screen → Negative
→ Returns / Replacement

refund → Negative
→ Refunds

30. PHASE 21 --- Error Handling

You need to handle:

Empty review

""

Return:

400 Bad Request

Extremely long review

Truncate or chunk it.

Unknown aspect

Example:

aspect = unknown

Route to:

Manual Review

Low model confidence

Example:

confidence < threshold

Instead of confidently routing it:

Manual Review

This is important because an automated support router should have a safe
fallback.

31. PHASE 22 --- Confidence Threshold

Suppose model says:

payment
confidence = 0.97

Good.

But:

payment
confidence = 0.51

Do not blindly route it.

Possible rule:

confidence >= 0.80
↓
automatic routing

confidence < 0.80
↓
Manual Review

The exact threshold should be selected based on validation results
rather than invented as a scientific fact.

32. PHASE 23 --- Model Evaluation

Your report should show more than accuracy.

For sentiment:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

For aspect extraction, depending on your formulation:

Precision
Recall
F1-score

For routing:

Routing Accuracy

Example:

## Correctly routed tickets

Total routed tickets

Also report:

Manual Review Rate

This tells you how often the system was uncertain.

33. PHASE 24 --- Compare Baseline vs Transformer

This is important for the AML component.

Your report should have something like:

Baseline:
TF-IDF + Logistic Regression

Advanced:
DistilBERT fine-tuned on review sentiment

Then compare their results.

Do not make up numbers.

Use your actual test results.

34. PHASE 25 --- Deployment Architecture

Final system:

                 User
                   │
                   ▼
             Streamlit UI
                   │
                   ▼
               FastAPI
                   │
                   ▼
           Analysis Service
          ┌────────┴─────────┐
          ▼                  ▼
    Sentiment Model     Aspect Model
          │                  │
          └────────┬─────────┘
                   ▼
          Aspect + Sentiment
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Priority           Routing
          │                 │
          └────────┬────────┘
                   ▼
                SQLite
                   │
                   ▼
             Ticket Records

35. Exact Order You Should Follow

This is the most important section.

DO NOT jump around.

Follow this order:

1.  Create GitHub repo
    ↓
2.  Create Python environment
    ↓
3.  Install packages
    ↓
4.  Get dataset
    ↓
5.  Inspect dataset
    ↓
6.  EDA
    ↓
7.  Decide labels
    ↓
8.  Train/test split
    ↓
9.  TF-IDF + Logistic Regression
    ↓
10. Evaluate baseline
    ↓
11. Learn tokenizer + Transformer basics
    ↓
12. Fine-tune DistilBERT sentiment model
    ↓
13. Evaluate Transformer
    ↓
14. Save sentiment model
    ↓
15. Build aspect baseline
    ↓
16. Build/improve aspect extraction model
    ↓
17. Add aspect-level sentiment
    ↓
18. Create structured analysis output
    ↓
19. Create department routing rules
    ↓
20. Create priority rules
    ↓
21. Create database
    ↓
22. Connect analysis to DB
    ↓
23. Build FastAPI
    ↓
24. Test API
    ↓
25. Build Streamlit
    ↓
26. End-to-end testing
    ↓
27. README + documentation
    ↓
28. Deploy

29. What You Should Do TODAY

Do not try to complete the entire project today.

Your first milestone is only:

GitHub repo +
Python environment +
Dataset +
EDA

Then stop and inspect the data.

Your immediate commands

git clone <repo-url>
cd ecommerce-review-router

python -m venv .venv

.venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Create:

data/raw/
data/processed/
notebooks/
app/
tests/

Then:

jupyter notebook

Create:

01_eda.ipynb

37. What NOT To Do

❌ Don't train BERT from scratch

You will waste time and compute.

❌ Don't immediately build FastAPI

First make the ML pipeline work in a notebook.

❌ Don't start with Streamlit

UI comes after the backend/ML pipeline.

❌ Don't make routing an ML model

For the current project, routing is a business-rule layer.

❌ Don't use only keywords and call it AI

Keyword routing is fine as a baseline/fallback, but the project should
demonstrate ML/NLP.

❌ Don't create a huge architecture

This is a student AML project. Keep the system understandable.

❌ Don't invent evaluation numbers

Every accuracy/F1 number in the report should come from your actual
experiment.

38. The Mental Model You Need

You do not need to think about the entire project at once.

Think in four boxes:

BOX 1 — SENTIMENT

"What is the overall mood?"

        ↓

BOX 2 — ASPECT

"What is the customer talking about?"

        ↓

BOX 3 — ROUTING

"Which department handles that issue?"

        ↓

BOX 4 — PRIORITY

"How urgently should they handle it?"

Example:

"My payment failed and money was deducted.
I'm extremely angry."

        ↓

SENTIMENT
Negative

        ↓

ASPECT
Payment

        ↓

ROUTING
Payments Department

        ↓

PRIORITY
HIGH

That is the complete idea.

39. Final MVP Definition

Your project is successful when this works:

Input Review
↓
Overall Sentiment
↓
Aspect(s)
↓
Opinion(s)
↓
Aspect Sentiment
↓
Department
↓
Priority
↓
Database
↓
API Response
↓
Dashboard

Minimum viable demo

Customer enters review
↓
Clicks Analyze
↓
System analyzes it
↓
Shows:
Overall sentiment
Detected issue
Opinion
Aspect sentiment
Department
Priority
↓
Stores ticket in database

40. Your First Goal

For now, forget:

FastAPI
Streamlit
Docker
Deployment
Database

Your first goal is simply:

Get a real e-commerce review dataset, understand its columns and
labels, and successfully train your first sentiment baseline.

Once that works, the Transformer training will make much more sense.

Development strategy

Build one layer at a time:

Dataset
✓
EDA
↓
Baseline
↓
Transformer
↓
Aspect
↓
Routing
↓
Backend
↓
UI

Never move to the next major layer while you don't understand the
current layer.Intelligent E-Commerce Customer Review Analyzer & Support Router

Complete End-to-End Implementation Guide

Audience: Beginner who has used Python/ML libraries but has never
fine-tuned a Transformer model.

Goal: Build the entire project from scratch: dataset →
preprocessing → sentiment model → aspect/opinion extraction → aspect
sentiment → priority → department routing → database → FastAPI →
Streamlit → testing → deployment.

0. First Understand What We Are Building

The project takes a customer review and converts it into a structured
support decision.

Example input

"The phone is excellent, but the delivery was very late and my refund
has not arrived."

Desired output

{
"overall_sentiment": "Negative",
"overall_confidence": 0.95,
"issues": [
{
"aspect": "delivery",
"opinion": "very late",
"sentiment": "Negative",
"confidence": 0.93
},
{
"aspect": "refund_return",
"opinion": "has not arrived",
"sentiment": "Negative",
"confidence": 0.91
}
],
"routing": [
{
"department": "Logistics",
"priority": "HIGH"
},
{
"department": "Refunds",
"priority": "HIGH"
}
]
}

The important distinction is:

ML/NLP
↓
"What is the customer saying?"

Routing/business logic
↓
"Which department should handle it?"

Priority logic
↓
"How urgently should it be handled?"

Routing is not another ML model by default. It is a business-rule
layer that consumes the ML output.

1. Project Scope

The project has four major intelligence tasks:

Overall Sentiment Analysis

Positive

Neutral

Negative

Aspect Extraction

Find the problem/topic mentioned in the review.

Examples: delivery, payment, refund, product quality.

Aspect-Level Sentiment / Opinion

Identify what is said about each aspect.

Example:

delivery → "very late" → Negative

product_quality → "excellent" → Positive

Smart Support Routing

Map detected issues to departments.

Assign priority.

Optionally create a support ticket.

2. Research Papers: Why They Matter

The project is based on three research directions.

Paper 1 --- Aspect-Based Sentiment Classification

The first paper focuses on aspect-level sentiment classification in
e-commerce reviews.

The basic idea:

"The phone is good but delivery is late."

phone → Positive
delivery → Negative

This is more informative than only saying:

Overall → Negative

The paper evaluates ML/Transformer approaches for e-commerce
aspect-based sentiment classification and reports strong performance
from RoBERTa.

What we take from Paper 1

Use aspect-level sentiment instead of relying only on one overall
sentiment label.

Paper 2 --- Aspect Extraction + Sentiment as Related Tasks

The second paper separates ABSA into two related tasks:

ATE --- Aspect Term Extraction

APC --- Aspect Polarity Classification

Example:

"The delivery was extremely late."

ATE:
delivery

APC:
delivery → Negative

The paper studies multitask/joint learning, where related tasks can help
each other.

What we take from Paper 2

Understand that:

Find the aspect

- Determine its sentiment

are related NLP tasks.

For this student project, do not try to reproduce the complete
research architecture immediately. Start with a simpler implementation
and improve it only if the data and time support it.

Paper 3 --- Long / Multiple-Issue E-Commerce Reviews

The third paper focuses on long reviews containing multiple aspects and
introduces structured annotations such as:

Target
Aspect
Opinion
Sentiment

Example:

"The phone is excellent, but the screen is damaged and my refund hasn't arrived."

phone → excellent → Positive
screen → damaged → Negative
refund → hasn't arrived → Negative

What we take from Paper 3

A real review may contain multiple issues, so the system should
preserve multiple aspect-level results instead of forcing the whole
review into one label.

3. Important Beginner Decision: Do Not Start With a Huge Model

You said you have never trained a big model before.

That is completely fine.

Do not begin by trying to build a large Transformer architecture
from scratch.

We will use:

Baseline first
↓
Understand the data
↓
Train simple model
↓
Fine-tune a pretrained Transformer
↓
Add aspect extraction

You are not training BERT/DistilBERT from zero.

You start with a pretrained model and fine-tune it on your task.

4. Recommended Technology Stack

Machine Learning

Python
Pandas
NumPy
Scikit-learn
PyTorch
Hugging Face Transformers
Hugging Face Datasets

NLP

Hugging Face Tokenizers
Transformer model

Recommended first Transformer:

DistilBERT

Why?

Easier and lighter than a large BERT-style model.

Fast enough for a student project.

Uses the same general Transformer fine-tuning workflow.

Good starting point for learning.

Backend

FastAPI
Uvicorn
Pydantic

Database

Start with:

SQLite
SQLAlchemy

Move to PostgreSQL only if needed.

Frontend

Streamlit

Testing

Pytest

Version control

Git
GitHub

5. Recommended Repository

Repository name

ecommerce-review-router

GitHub description

NLP-based e-commerce customer review analyzer that performs sentiment analysis, aspect extraction, priority detection, and intelligent support department routing.

6. Final Repository Structure

Create this gradually. You do not need to create every file on day
one.

ecommerce-review-router/
│
├── app/
│ ├── main.py
│ │
│ ├── api/
│ │ └── routes.py
│ │
│ ├── services/
│ │ ├── preprocessing.py
│ │ ├── sentiment.py
│ │ ├── aspect_extraction.py
│ │ ├── routing.py
│ │ └── ticket_service.py
│ │
│ ├── models/
│ │ └── trained_model/
│ │
│ └── db/
│ ├── database.py
│ └── models.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_sentiment_baseline.ipynb
│ ├── 03_transformer_sentiment.ipynb
│ └── 04_aspect_extraction.ipynb
│
├── tests/
│ ├── test_sentiment.py
│ ├── test_aspects.py
│ ├── test_routing.py
│ └── test_api.py
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md

7. PHASE 1 --- Set Up the Project

Step 1: Create the GitHub repository

Create:

ecommerce-review-router

Then clone it.

git clone <your-repository-url>
cd ecommerce-review-router

Step 2: Create a virtual environment

Windows:

python -m venv .venv

Activate:

.venv\Scripts\activate

Check:

python --version

Step 3: Install initial packages

Start with:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Later install Transformer dependencies:

pip install torch transformers datasets accelerate

Backend:

pip install fastapi uvicorn pydantic

Database:

pip install sqlalchemy

Testing:

pip install pytest

Save dependencies:

pip freeze > requirements.txt

8. PHASE 2 --- Dataset

This is where the real ML work starts.

Do not train the model yet.

First find a dataset that supports your project.

You need at minimum:

review text
sentiment label

For example:

review sentiment

"The product is amazing" Positive
"Delivery was terrible" Negative
"The product is okay" Neutral

For aspect-based work, you ideally need additional annotations:

review aspect opinion sentiment

"Delivery was delivery extremely late Negative
extremely late"

"Camera quality product_quality excellent Positive
is excellent"

Critical point

A normal sentiment dataset is not automatically enough to train an
aspect extraction model.

Before choosing the final dataset, inspect its labels.

9. PHASE 3 --- Exploratory Data Analysis

Create:

notebooks/01_eda.ipynb

Load the dataset.

import pandas as pd

df = pd.read_csv("path/to/dataset.csv")

df.head()

Inspect:

df.shape
df.info()
df.columns
df.isnull().sum()
df.duplicated().sum()

Check sentiment distribution:

df["sentiment"].value_counts()

Visualize:

df["sentiment"].value_counts().plot(kind="bar")

Check review length:

df["review_length"] = df["review"].astype(str).str.len()

df["review_length"].describe()

Inspect actual examples:

df[["review", "sentiment"]].sample(20)

Questions you must answer after EDA

How many reviews?

Which sentiment classes exist?

Is the dataset balanced?

Are there missing reviews?

Are there duplicate reviews?

Are labels clean?

Are the reviews actually e-commerce reviews?

Does the dataset contain aspect information?

Are aspect labels compatible with our department categories?

Do not skip this phase.

10. PHASE 4 --- Decide Your Label Taxonomy

Before training aspect extraction, define the project's supported
categories.

Start with:

delivery
product_quality
payment
refund_return
packaging
customer_support

Example mapping:

delivery
↓
Logistics

product_quality
↓
Product Support

payment
↓
Payments

refund_return
↓
Refunds

packaging
↓
Returns / Replacement

customer_support
↓
Customer Support

Keep the number of categories small initially.

You can add categories later.

11. PHASE 5 --- Sentiment Baseline

Before using Transformers, build a simple baseline.

Why?

Because you need to know whether the more advanced model actually
improves performance.

The baseline:

Review
↓
TF-IDF
↓
Logistic Regression
↓
Sentiment

Step 1: Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
df["review"],
df["sentiment"],
test_size=0.2,
random_state=42,
stratify=df["sentiment"]
)

Step 2: TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
max_features=10000,
ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

Step 3: Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

Predict:

predictions = model.predict(X_test_tfidf)

Evaluate:

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

Record:

Accuracy
Precision
Recall
F1-score

This becomes your baseline.

12. PHASE 6 --- Understand What a Transformer Actually Does

You do not need to understand every mathematical detail before using it.

Think of the model as:

Raw sentence
↓
Tokenizer
↓
Numbers / token IDs
↓
Pretrained Transformer
↓
Text representation
↓
Classification layer
↓
Positive / Neutral / Negative

Example:

"I love this phone"
↓
Tokenizer
↓
[101, ..., ...]
↓
DistilBERT
↓
Classifier
↓
Positive

Pretrained vs training from scratch

You are doing:

Pretrained DistilBERT +
Your review dataset
↓
Fine-tuning

You are NOT doing:

Random neural network
↓
Learn English from zero

That would require enormous data and compute.

13. PHASE 7 --- Fine-Tune DistilBERT for Overall Sentiment

Create:

notebooks/03_transformer_sentiment.ipynb

First convert labels:

Negative → 0
Neutral → 1
Positive → 2

Use a tokenizer.

Conceptually:

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

Tokenize:

def tokenize(batch):
return tokenizer(
batch["review"],
padding="max_length",
truncation=True,
max_length=128
)

Then load the classification model:

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
model_name,
num_labels=3
)

Then train using the Hugging Face training workflow.

Do not try to write the entire training loop manually at first.

Use:

Dataset
↓
Tokenizer
↓
DistilBERT
↓
Trainer
↓
Fine-tuned model

14. PHASE 8 --- Evaluate the Transformer

Do not just say:

"BERT worked."

Compare:

TF-IDF + Logistic Regression
vs
DistilBERT

Example report format:

Model Accuracy F1

TF-IDF + Logistic Regression X X
DistilBERT Y Y

Also inspect:

Confusion Matrix
Classification Report

If DistilBERT does not improve substantially, investigate the data
instead of blindly making the model bigger.

15. PHASE 9 --- Save the Sentiment Model

Once the sentiment model works:

app/models/trained_model/sentiment/

Save:

model
tokenizer
label mapping

Your API should later load this model rather than retraining it every
time.

The production flow should be:

Application starts
↓
Load trained model
↓
Wait for review

NOT:

Every API request
↓
Train model

Training and inference are separate operations.

16. PHASE 10 --- Aspect Extraction

This is the most important part to understand before implementing it.

Overall sentiment

Question:

"How does the customer feel about the review overall?"

Example:

Review → Negative

Aspect extraction

Question:

"What is the customer talking about?"

Example:

"The delivery was late"

Aspect → delivery

Opinion extraction

Question:

"What words express the customer's opinion about that aspect?"

delivery → "late"

So:

Aspect = delivery
Opinion = late

17. Three Possible Approaches for Aspect Extraction

Do them in this order.

Approach A --- Keyword baseline

Create a simple dictionary.

aspect_keywords = {
"delivery": [
"delivery", "delayed", "late", "shipment", "shipping"
],
"payment": [
"payment", "charged", "transaction"
],
"refund_return": [
"refund", "return", "money back"
],
"product_quality": [
"broken", "damaged", "quality", "defective"
]
}

This is NOT your final intelligent model.

It is a baseline to understand the problem.

Approach B --- Text classification

If your dataset provides one aspect label per example, you can train an
aspect classifier.

Review
↓
Transformer
↓
Aspect category

Example:

"Money was deducted but payment failed"
↓
payment

If reviews can have multiple aspects, you need a multi-label
classification setup rather than ordinary single-label classification.

Approach C --- Token-level aspect extraction

For true ATE, the model identifies which words are aspect terms.

Example:

"The delivery was extremely late"

The → O
delivery → B-ASPECT
was → O
extremely → O
late → O

This is usually formulated as token classification / sequence
labeling.

A BIO scheme is common:

B-ASPECT = beginning of an aspect
I-ASPECT = continuation of an aspect
O = not an aspect

Example:

"poor customer service"

poor → O
customer → B-ASPECT
service → I-ASPECT

This is the more faithful implementation of ATE.

18. What Should You Implement?

Since you are learning Transformer training for the first time:

Version 1

Build:

Overall sentiment → Transformer
Aspect category → simple supervised classifier
Routing → rules
Priority → rules

Version 2

Improve aspect extraction:

Transformer token classification
↓
Aspect terms

Do not start with the most complex research architecture.

19. PHASE 11 --- Aspect Sentiment

Once aspects are available, determine sentiment for each aspect.

Example:

"Camera is amazing but delivery is terrible."

Output:

camera
opinion = amazing
sentiment = Positive

delivery
opinion = terrible
sentiment = Negative

You can initially implement this with a sentiment classifier applied to
the relevant sentence/aspect context.

For example:

"delivery is terrible"
↓
sentiment model
↓
Negative

Later, if the dataset supports proper aspect-level sentiment labels,
fine-tune a dedicated aspect sentiment model.

20. PHASE 12 --- Create One Unified Analysis Object

At this stage, make one Python function responsible for combining model
outputs.

Conceptually:

def analyze_review(text):
overall_sentiment = sentiment_model(text)

    aspects = aspect_model(text)

    # determine aspect-level sentiment
    issues = ...

    return {
        "overall_sentiment": overall_sentiment,
        "issues": issues
    }

The important thing is that your ML modules should return structured
data, not random strings.

Example:

{
"aspect": "delivery",
"opinion": "very late",
"sentiment": "Negative",
"confidence": 0.93
}

21. PHASE 13 --- Routing Engine

Now the ML part gives you:

aspect = delivery
sentiment = Negative

The routing engine decides:

delivery
↓
Logistics

This is ordinary application logic.

Example:

DEPARTMENT_MAP = {
"delivery": "Logistics",
"product_quality": "Product Support",
"payment": "Payments",
"refund_return": "Refunds",
"packaging": "Returns / Replacement",
"customer_support": "Customer Support"
}

Then:

department = DEPARTMENT_MAP.get(
aspect,
"Manual Review"
)

22. PHASE 14 --- Priority Engine

Create another independent function.

Example rules:

Strong negative / angry +
critical issue
↓
HIGH

Possible basic policy:

HIGH:

- severe negative sentiment
- payment charged but failed
- refund not received
- damaged/broken product
- explicit urgent language

MEDIUM:

- normal negative complaint
- repeated delay
- unresolved support problem

LOW:

- minor complaint
- neutral inquiry
- mild inconvenience

Do not pretend this is an ML model unless you actually train a priority
model.

It is a rule-based business layer.

23. PHASE 15 --- Multiple Issues

This is where Paper 3 becomes important.

Input:

"The product is great, but the package was damaged and my refund is
still pending."

The system should not return only:

Negative

Instead:

issues = [
{
"aspect": "packaging",
"opinion": "damaged",
"sentiment": "Negative"
},
{
"aspect": "refund_return",
"opinion": "still pending",
"sentiment": "Negative"
}
]

Then routing can create multiple department assignments:

packaging
↓
Returns / Replacement

refund_return
↓
Refunds

24. PHASE 16 --- Database

Start simple with SQLite.

Use three logical entities.

reviews

id
text
overall_sentiment
overall_confidence
created_at

review_issues

id
review_id
aspect
opinion
sentiment
confidence

tickets

id
review_id
department
priority
routing_reason
status
created_at

Relationships:

Review
│
├── Issue
├── Issue
└── Issue
│
└── Ticket / Routing

25. PHASE 17 --- FastAPI Backend

Now wrap the system in an API.

Recommended endpoint:

POST /reviews/analyze

Request:

{
"text": "Payment failed and my money was deducted."
}

Backend flow:

POST /reviews/analyze
↓
Validate request
↓
Preprocess
↓
Overall sentiment model
↓
Aspect extraction
↓
Aspect sentiment
↓
Priority engine
↓
Routing engine
↓
Save to DB
↓
Return JSON

Response:

{
"review_id": 101,
"overall_sentiment": "Negative",
"overall_confidence": 0.96,
"issues": [
{
"aspect": "payment",
"opinion": "failed",
"sentiment": "Negative",
"confidence": 0.95
}
],
"routing": [
{
"department": "Payments",
"priority": "HIGH",
"reason": "Payment issue with money deduction"
}
]
}

26. API Endpoints

Start with only these:

Analyze review

POST /reviews/analyze

Get ticket

GET /tickets/{ticket_id}

List tickets

GET /tickets

Health check

GET /health

Do not build 20 endpoints for a mini-project.

27. PHASE 18 --- Streamlit Dashboard

Only after the API works.

Basic UI:

---

       E-Commerce Review Analyzer

---

Enter customer review:

[ The delivery was very late and I want a refund ]

                [ ANALYZE ]

---

Overall Sentiment
NEGATIVE
Confidence: 95%

Detected Issues

Delivery
Opinion: very late
Sentiment: Negative
Confidence: 93%

Refund
Opinion: want a refund
Sentiment: Negative
Confidence: 91%

---

Routing

Department Priority
Logistics HIGH
Refunds HIGH

---

The UI should demonstrate the whole pipeline, not just display one
sentiment label.

28. PHASE 19 --- Testing

Test each layer independently.

Sentiment

Input:

"I love this product."

Expected:

Positive

Aspect

Input:

"Delivery was extremely late."

Expected:

delivery

Routing

Input:

aspect = delivery

Expected:

Logistics

Priority

Input:

sentiment = Negative
critical_issue = True

Expected:

HIGH

API

Send:

{
"text": "My payment failed and money was deducted."
}

Verify that the response contains:

sentiment
issues
department
priority

29. PHASE 20 --- End-to-End Test

Now test the complete pipeline with real examples.

Example 1

"The product is excellent."

Expected:

Overall: Positive
Issue: product_quality
Aspect sentiment: Positive
Priority: LOW

Example 2

"Delivery was five days late."

Expected:

Overall: Negative
Aspect: delivery
Department: Logistics
Priority: MEDIUM

Example 3

"Payment failed twice and money was deducted. This is extremely frustrating!"

Expected:

Overall: Negative
Aspect: payment
Department: Payments
Priority: HIGH

Example 4

"The phone is good but the screen is broken and my refund hasn't arrived."

Expected:

phone → Positive

screen → Negative
→ Returns / Replacement

refund → Negative
→ Refunds

30. PHASE 21 --- Error Handling

You need to handle:

Empty review

""

Return:

400 Bad Request

Extremely long review

Truncate or chunk it.

Unknown aspect

Example:

aspect = unknown

Route to:

Manual Review

Low model confidence

Example:

confidence < threshold

Instead of confidently routing it:

Manual Review

This is important because an automated support router should have a safe
fallback.

31. PHASE 22 --- Confidence Threshold

Suppose model says:

payment
confidence = 0.97

Good.

But:

payment
confidence = 0.51

Do not blindly route it.

Possible rule:

confidence >= 0.80
↓
automatic routing

confidence < 0.80
↓
Manual Review

The exact threshold should be selected based on validation results
rather than invented as a scientific fact.

32. PHASE 23 --- Model Evaluation

Your report should show more than accuracy.

For sentiment:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

For aspect extraction, depending on your formulation:

Precision
Recall
F1-score

For routing:

Routing Accuracy

Example:

## Correctly routed tickets

Total routed tickets

Also report:

Manual Review Rate

This tells you how often the system was uncertain.

33. PHASE 24 --- Compare Baseline vs Transformer

This is important for the AML component.

Your report should have something like:

Baseline:
TF-IDF + Logistic Regression

Advanced:
DistilBERT fine-tuned on review sentiment

Then compare their results.

Do not make up numbers.

Use your actual test results.

34. PHASE 25 --- Deployment Architecture

Final system:

                 User
                   │
                   ▼
             Streamlit UI
                   │
                   ▼
               FastAPI
                   │
                   ▼
           Analysis Service
          ┌────────┴─────────┐
          ▼                  ▼
    Sentiment Model     Aspect Model
          │                  │
          └────────┬─────────┘
                   ▼
          Aspect + Sentiment
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Priority           Routing
          │                 │
          └────────┬────────┘
                   ▼
                SQLite
                   │
                   ▼
             Ticket Records

35. Exact Order You Should Follow

This is the most important section.

DO NOT jump around.

Follow this order:

1.  Create GitHub repo
    ↓
2.  Create Python environment
    ↓
3.  Install packages
    ↓
4.  Get dataset
    ↓
5.  Inspect dataset
    ↓
6.  EDA
    ↓
7.  Decide labels
    ↓
8.  Train/test split
    ↓
9.  TF-IDF + Logistic Regression
    ↓
10. Evaluate baseline
    ↓
11. Learn tokenizer + Transformer basics
    ↓
12. Fine-tune DistilBERT sentiment model
    ↓
13. Evaluate Transformer
    ↓
14. Save sentiment model
    ↓
15. Build aspect baseline
    ↓
16. Build/improve aspect extraction model
    ↓
17. Add aspect-level sentiment
    ↓
18. Create structured analysis output
    ↓
19. Create department routing rules
    ↓
20. Create priority rules
    ↓
21. Create database
    ↓
22. Connect analysis to DB
    ↓
23. Build FastAPI
    ↓
24. Test API
    ↓
25. Build Streamlit
    ↓
26. End-to-end testing
    ↓
27. README + documentation
    ↓
28. Deploy

29. What You Should Do TODAY

Do not try to complete the entire project today.

Your first milestone is only:

GitHub repo +
Python environment +
Dataset +
EDA

Then stop and inspect the data.

Your immediate commands

git clone <repo-url>
cd ecommerce-review-router

python -m venv .venv

.venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Create:

data/raw/
data/processed/
notebooks/
app/
tests/

Then:

jupyter notebook

Create:

01_eda.ipynb

37. What NOT To Do

❌ Don't train BERT from scratch

You will waste time and compute.

❌ Don't immediately build FastAPI

First make the ML pipeline work in a notebook.

❌ Don't start with Streamlit

UI comes after the backend/ML pipeline.

❌ Don't make routing an ML model

For the current project, routing is a business-rule layer.

❌ Don't use only keywords and call it AI

Keyword routing is fine as a baseline/fallback, but the project should
demonstrate ML/NLP.

❌ Don't create a huge architecture

This is a student AML project. Keep the system understandable.

❌ Don't invent evaluation numbers

Every accuracy/F1 number in the report should come from your actual
experiment.

38. The Mental Model You Need

You do not need to think about the entire project at once.

Think in four boxes:

BOX 1 — SENTIMENT

"What is the overall mood?"

        ↓

BOX 2 — ASPECT

"What is the customer talking about?"

        ↓

BOX 3 — ROUTING

"Which department handles that issue?"

        ↓

BOX 4 — PRIORITY

"How urgently should they handle it?"

Example:

"My payment failed and money was deducted.
I'm extremely angry."

        ↓

SENTIMENT
Negative

        ↓

ASPECT
Payment

        ↓

ROUTING
Payments Department

        ↓

PRIORITY
HIGH

That is the complete idea.

39. Final MVP Definition

Your project is successful when this works:

Input Review
↓
Overall Sentiment
↓
Aspect(s)
↓
Opinion(s)
↓
Aspect Sentiment
↓
Department
↓
Priority
↓
Database
↓
API Response
↓
Dashboard

Minimum viable demo

Customer enters review
↓
Clicks Analyze
↓
System analyzes it
↓
Shows:
Overall sentiment
Detected issue
Opinion
Aspect sentiment
Department
Priority
↓
Stores ticket in database

40. Your First Goal

For now, forget:

FastAPI
Streamlit
Docker
Deployment
Database

Your first goal is simply:

Get a real e-commerce review dataset, understand its columns and
labels, and successfully train your first sentiment baseline.

Once that works, the Transformer training will make much more sense.

Development strategy

Build one layer at a time:

Dataset
✓
EDA
↓
Baseline
↓
Transformer
↓
Aspect
↓
Routing
↓
Backend
↓
UI

Never move to the next major layer while you don't understand the
current layer.
