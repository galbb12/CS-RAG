# HAIFA-RAG

**[Hosted Demo](https://cs-rag.onrender.com/)**

A Retrieval-Augmented Generation (RAG) chatbot that helps Computer Science students at the University of Haifa with course selection and academic planning. It answers questions about courses, lecturers, prerequisites, exam difficulty, and grade distributions based on real student reviews and academic data.

## Architecture

```
User ──► React Frontend ──► FastAPI Server ──► RAG Engine (LangChain)
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                            search_course     kdams_tree      course_grades
                              _reviews
                                │               │               │
                            FAISS Index      SQL dump        SQL dump
                          (student reviews)  (prerequisites)  (grades)
```

The system is a **tool-augmented RAG agent** - an LLM orchestrates three specialized tools to answer student questions in Hebrew, combining vector similarity search over unstructured text (student reviews) with structured lookups for prerequisites and grade distributions.

## Screenshots

### Prerequisite Graph Visualization
Interactive DAG showing course prerequisites and dependent courses, rendered as a custom SVG flowchart with zoom/pan and hover highlighting.

![Prerequisite Graph](assets/prerequisite-graph.png)

### Grade Distribution Histograms
Historical grade distributions with color-coded bars, average line overlay, and filtering by lecturer/semester/moed.

![Grade Histograms](assets/grade-histogram.png)

### Follow-up Suggestions
After each answer, the bot suggests 3 related questions as clickable buttons so you can keep exploring without typing.

## Tools

| Tool | Purpose | Data Source |
|---|---|---|
| `search_course_reviews` | Semantic search over student reviews with metadata pre-filtering (course, lecturer, type) | FAISS vector index (Google Gemini embeddings) |
| `kdams_tree` | Prerequisite graph traversal - shows what courses are needed and what they unlock | SQL dump (parsed at runtime) |
| `course_grades` | Historical grade distributions with filtering by lecturer, year, semester, moed | Grade data (parsed from SQL) |

## Tech Stack

**Backend:** Python 3.12, FastAPI, LangChain, FAISS, Google Gemini Embeddings, OpenAI-compatible LLM (NVIDIA NIM)

**Frontend:** React 19, Vite, custom SVG graph renderer, grade histogram visualization

**Deployment:** Docker (multi-stage build), Render.com

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- API keys for the LLM provider and Google Gemini embeddings

### Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=<your-nvidia-nim-api-key>
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=openai/gpt-oss-120b
GOOGLE_API_KEY=<your-google-api-key>
```

### Running Locally

```bash
# Backend
pip install -r requirements.txt
uvicorn server:app --port 8000

# Frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker build -t haifa-rag .
docker run -p 8000:8000 --env-file .env haifa-rag
```

## API

The server exposes an **OpenAI-compatible API**, so it works with any OpenAI client library or tool:

```
POST /v1/chat/completions   # Chat (streaming & non-streaming)
GET  /v1/models              # List models
GET  /health                 # Health check
```

## Evaluation

The project includes an LLM-as-a-judge evaluation pipeline (`evaluate.py`) that scores responses on:

- **Faithfulness** - factual correctness against the database
- **Relevance** - whether the answer addresses the question
- **Completeness** - coverage of all asked aspects

Results (n=40, judge: Claude Opus 4.6): Faithfulness **8.5**/10, Relevance **9.1**/10, Completeness **8.2**/10.

![Evaluation Results](assets/eval-results.png)

## Project Structure

```
├── server.py               # FastAPI server (OpenAI-compatible API)
├── rag_engine.py            # Core RAG pipeline & agent loop
├── comment_match_prompt.py  # Semantic search tool (FAISS + embeddings)
├── kdams_tool.py            # Prerequisite graph tool
├── grades_tool.py           # Grade distribution tool
├── format_data.py           # SQL dump parser & data processing
├── evaluate.py              # LLM-as-judge evaluation pipeline
├── documents.json           # Parsed student reviews
├── faiss_index/             # Pre-built FAISS vector index
├── frontend/                # React chat interface
├── Dockerfile               # Multi-stage build (Node + Python)
└── render.yaml              # Render.com deployment config
```

## Authors

Gal Bareket & Ariel Yucovich
