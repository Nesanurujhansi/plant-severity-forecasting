import sys
import uvicorn

if __name__ == "__main__":
    print("Initializing crop-prognosis server...", flush=True)
    from app.main import app
    print("App loaded. Starting uvicorn on http://0.0.0.0:8000 ...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
