"""Vercel entrypoint: Vercel looks for a top-level `app` in index.py.

Importing it from the `app` package keeps the package's relative imports working.
Run locally with:  uvicorn index:app --port 7860   (or uvicorn app.main:app)
"""
from app.main import app  # noqa: F401
