#!/bin/bash

# 1. Start the FastAPI backend in the background on port 8000
# We use & to make sure the script continues to the next command
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# 2. Start the Streamlit frontend in the foreground
# HF Spaces REQUIRES port 7860
streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0