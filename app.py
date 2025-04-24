import os
from fastapi import FastAPI
from dotenv import load_dotenv
from api import router as api_router

load_dotenv()

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(
    title="Email Classification API",
    description="Mask PII and classify emails using OpenAI few-shot.",
    version="1.0.0",
)

app.include_router(api_router)

@app.on_event("startup")
async def _print_docs_url():
    print(f"\n🚀 Swagger UI available at → http://{HOST}:{PORT}/docs")

@app.get("/", tags=["Root"])
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
