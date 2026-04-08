"""
server/app.py — OpenEnv multi-mode deployment entry point.
Required by the OpenEnv validator spec.
This file re-exports the FastAPI app from api.py.
"""
from api import app

__all__ = ["app"]
