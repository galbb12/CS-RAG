ARG INCLUDE_FAISS=false

# Stage 1: Build frontend
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python backend + built frontend (base)
FROM python:3.12-slim AS base
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY *.json .
COPY --from=frontend /app/frontend/dist frontend/dist/

EXPOSE 8000
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}

# Stage 3a: Without FAISS (faiss_index created at runtime)
FROM base AS final-false

# Stage 3b: With FAISS baked in
FROM base AS final-true
COPY faiss_index/ faiss_index/

# Select final stage
FROM final-${INCLUDE_FAISS}
