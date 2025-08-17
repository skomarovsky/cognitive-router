"""
FastAPI server example for Cognitive Router
Demonstrates production deployment with Poetry
"""

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("❌ FastAPI dependencies not installed")
    print("   Install with: poetry install --extras 'api'")
    exit(1)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_router import create_similarity_router, create_keyword_router

app = FastAPI(
    title="Cognitive Router API",
    description="Intelligent natural language understanding and routing",
    version="1.0.0"
)

# Initialize cognitive routers
cognitive_similarity = None
cognitive_keyword = None

@app.on_event("startup")
async def startup_event():
    global cognitive_similarity, cognitive_keyword
    try:
        cognitive_similarity = create_similarity_router()
        cognitive_keyword = create_keyword_router()
        print("🧠 Cognitive routers initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize cognitive routers: {e}")

class QueryRequest(BaseModel):
    query: str
    router_type: str = "similarity"  # similarity, keyword, or ensemble

class RoutingResponse(BaseModel):
    intent: str
    confidence: float
    router_type: str
    processing_time_ms: float

@app.post("/classify", response_model=RoutingResponse)
async def classify_query(request: QueryRequest):
    import time
    
    start_time = time.time()
    
    try:
        if request.router_type == "similarity":
            if cognitive_similarity is None:
                raise HTTPException(status_code=503, detail="Similarity router not available")
            intent, confidence = cognitive_similarity.classify(request.query)
        elif request.router_type == "keyword":
            if cognitive_keyword is None:
                raise HTTPException(status_code=503, detail="Keyword router not available")
            intent, confidence = cognitive_keyword.classify(request.query)
        else:
            raise HTTPException(status_code=400, detail="Invalid router_type")
        
        processing_time = (time.time() - start_time) * 1000
        
        return RoutingResponse(
            intent=intent,
            confidence=confidence,
            router_type=request.router_type,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "cognitive_status": "operational"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
