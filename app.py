from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import uvicorn
from typing import List, Dict, Any
import os
import sys

# Add current directory to path for importing
sys.path.append(".")

from hybrid_recommender import HybridRecommender

# ========================= CONFIG =========================
MODEL_PATH = 'processed_data/final_hybrid_recommender.pkl'

app = FastAPI(
    title="Handmade Marketplace AI Recommender API",
    description="AI-Based Product Recommendation System for Handmade Marketplace - COM4901 Project",
    version="1.0"
)

# Load the hybrid recommender
print("Loading Hybrid Recommender Model...")
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        recommender = pickle.load(f)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run hybrid_recommender.py first.")

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    message: str

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Handmade Marketplace AI Recommender API",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        recs = recommender.get_recommendations(request.user_id, request.top_n)
        return {
            "user_id": request.user_id,
            "recommendations": recs,
            "message": f"Successfully generated {len(recs)} recommendations for user {request.user_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/users/{user_id}/recommendations")
async def get_recommendations_get(user_id: int, top_n: int = 10):
    try:
        recs = recommender.get_recommendations(user_id, top_n)
        return {
            "user_id": user_id,
            "recommendations": recs,
            "message": f"Successfully generated {len(recs)} recommendations"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)