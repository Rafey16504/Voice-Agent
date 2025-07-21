from fastapi import FastAPI, Query
from typing import List
import rag_builder
import asyncio

app = FastAPI()

@app.post("/ingest")
async def ingest(
    pdf_urls: List[str] = Query(...),
    user_id: str = Query(...),
    reset: bool = Query(False)
):
    try:
        await asyncio.to_thread(rag_builder.main_from_api, pdf_urls, user_id, reset)
        return {"status": "ok", "message": "Embeddings successfully created."}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
