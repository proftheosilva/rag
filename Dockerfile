FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# As dependências pertencem ao próprio projeto RAG.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
