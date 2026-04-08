"""
server/app.py — OpenEnv multi-mode deployment entry point.
Required by the OpenEnv validator spec.
This file re-exports the FastAPI app from api.py.
"""
import uvicorn
from api import app

def main():
    """Entry point for the server script."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
