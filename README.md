
---
title: Jira Automation
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# Jira Intelligence Automation Tool

An AI-driven automation engine designed to streamline Jira ticket management. This tool leverages transformer-based models to automatically classify ticket priority, identify historical context via semantic similarity, and assess developer availability through sentiment analysis.

## 🚀 Features

### 1. Automated Priority Prediction
- **Model:** DistilBERT
- **Dataset:** Apache Jira Dataset
- **Functionality:** Analyzes the ticket `Summary` and `Description` to predict the priority level, reducing manual triaging effort.

### 2. Semantic Ticket Matching
- **Model:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Mechanism:** Generates vector embeddings for the new ticket and compares them against historical data using **Cosine Similarity**.
- **Output:** Fetches the most relevant previously closed tickets to provide context for the current issue.

### 3. Developer Sentiment & Workload Analysis
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Logic:** Identifies the developer assigned to the most similar historical tickets and analyzes their 10 most recent comments.
- **Goal:** Evaluates sentiment to gauge if the developer is currently overloaded or stressed before auto-assignment recommendations.

## Tech Stack

- **Backend Framework:** FastAPI (Asynchronous API endpoints)
- **Machine Learning:** Hugging Face Transformers, Sentence-Transformers
- **Model Hosting:** Hugging Face Hub (Versioned, ready-to-use models)

## Deployment
- Application containerized using Docker
- CI/CD pipeline triggers on every GitHub push
- Automatic deployment to Hugging Face Spaces
- Models uploaded and versioned on Hugging Face Hub
- Datasets loaded directly from Hugging Face Datasets

## Project Structure

```text
├── src/ui/app/             # Streamlit UI
├── src/api/                # FastAPI application endpoints
├── src/models/             # Scripts for model loading and inference
├── src/data/               # Scripts for data loading and preprocessing
├── src/storage/            # Uploading and downloading models from HF
├── scripts/                # Data preprocessing and training logic
├── pyproject.toml          # Project dependencies
└── README.md

```
HuggingFace Spaces: https://huggingface.co/spaces/ZaraQureshi/JiraAutomation
HuggingFace Hub: https://huggingface.co/ZaraQureshi/jira-ml-models
```
