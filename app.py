from fastapi import FastAPI
from dotenv import load_dotenv
from api import router as api_router
import os

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 7860))

# ✅ Enable docs explicitly
app = FastAPI(
    title="Email Classification API",
    description="Mask PII and classify emails using OpenAI few-shot.",
    version="1.0.0",
    docs_url="/docs",         # Swagger UI
    redoc_url="/redoc",       # ReDoc UI
)

app.include_router(api_router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Email Classification API!",
        "docs_url": "/docs",
        "classify_endpoint": "/classify (POST)"
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return b"", 204

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
