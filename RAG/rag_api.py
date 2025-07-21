from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import rag_builder
import asyncio

app = FastAPI()

class IngestRequest(BaseModel):
    pdf_urls: List[str]
    user_id: str
    reset: bool = False

@app.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        await asyncio.to_thread(
            rag_builder.main_from_api, 
            request.pdf_urls, 
            request.user_id, 
            request.reset
        )
        return {"status": "ok", "message": "Embeddings successfully created."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
