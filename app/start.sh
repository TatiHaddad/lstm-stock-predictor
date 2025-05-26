#!/bin/bash
echo "Starting FastAPI with Uvicorn..."
uvicorn app.main:app --host=0.0.0.0 --port=$PORT
