"""Start the UI.

    python run.py

Reads env/.env for TRADER_SUBMITTER_KEY and the coordinator address.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env", ".env"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("trader.web.app:app", host="127.0.0.1", port=8600, reload=False)
