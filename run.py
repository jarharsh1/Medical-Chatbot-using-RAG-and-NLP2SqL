"""
Entry point for the Medical AI Backend.

Usage:
    python run.py
"""

import logging
import uvicorn

from backend.config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"\nMedical AI Backend starting at http://localhost:{SERVER_PORT}")
    uvicorn.run("backend.app:app", host=SERVER_HOST, port=SERVER_PORT)
