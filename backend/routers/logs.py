from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import json
import aiofiles
import os

router = APIRouter()

LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "events.log")

@router.post("/log-event/")
async def log_event(request: Request):
    try:
        data = await request.json()
        async with aiofiles.open(LOG_FILE, "a") as f:
            await f.write(json.dumps(data) + "\n")
        return {"status": "logged"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

