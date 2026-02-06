# LangChain FastAPI + Streamlit Demo

This project provides a simple FastAPI backend and Streamlit frontend wired together with a LangChain runnable. The backend exposes a `/chat` endpoint, and the Streamlit app sends messages to it.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the backend

```bash
uvicorn backend.main:app --reload
```

## Run the frontend

```bash
streamlit run frontend/app.py
```

## Configuration

Set the backend URL for Streamlit using `API_BASE_URL`:

```bash
export API_BASE_URL=http://localhost:8000
```
