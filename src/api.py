from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import get_settings
from src.service import get_service


settings=get_settings()
app=FastAPI(title="Jira ML Automation", version="1.0.0")

class AnalyzeRequest(BaseModel):
    summary:str
    description:str
    ticket_key: str=None

class AnalyzeResponse(BaseModel):
    predicted_priority: dict
    recommended_dev: str
    dev_mood: dict
    similar_tickets: List[dict]

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        service=get_service()
        result=service.analyze(req.summary,req.description)
        return AnalyzeResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/v1/models")
async def list_models():
    return {"models": ["priority_model", "embeddings"]}