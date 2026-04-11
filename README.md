# HAIFA-RAG

**[Hosted Demo](https://cs-rag.onrender.com/)**

A Retrieval-Augmented Generation (RAG) chatbot that helps Computer Science students at the University of Haifa with course selection and academic planning. It answers questions about courses, lecturers, prerequisites, exam difficulty, and grade distributions based on real student reviews and academic data.

## Architecture

```
User ──► React Frontend ──► FastAPI Server ──► RAG Engine (LangChain)
                                                      │
                              ┌────────────────────────┼────────────────────────┐
                              ▼                        ▼                        ▼
                    search_course_reviews          kdams_tree             course_grades
                              │                        │                        │
                              ▼                        └────────┬───────────────┘
                         FAISS Index                            ▼
                                                        MySQL / SQL dump
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
| `search_course_reviews` | Semantic search over student reviews with metadata pre-filtering (course, lecturer, type) | FAISS vector index |
| `kdams_tree` | Prerequisite graph traversal - shows what courses are needed and what they unlock | MySQL or SQL dump |
| `course_grades` | Historical grade distributions with filtering by lecturer, year, semester, moed | MySQL or SQL dump |

## Tech Stack

**Backend:** Python 3.12, FastAPI, LangChain, FAISS, OpenAI-compatible LLM (NVIDIA NIM)

**Frontend:** React 19, Vite, custom SVG graph renderer, grade histogram visualization

**Embeddings:** Google Gemini or NVIDIA NV-Embed (configurable)

**Database:** MySQL/MariaDB (live connection) or SQL dump file

**Deployment:** Docker (multi-stage build), Render.com

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- API key for the LLM provider

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=openai/gpt-oss-120b

# Embeddings
EMBEDDING_MODEL=nvidia/nv-embed-v1
# GEMINI_EMBEDDING_API_KEY=<your-google-api-key>  # If not set, base provider is used for embeddings

# Data source (pick one)
DB_URL=mysql://user:password@host:3306/dbname   # Live MySQL/MariaDB connection
# DB_SQL_PATH=/path/to/dump.sql                  # Or use a SQL dump file

# Optional
FAISS_INDEX_PATH=faiss_index
# DB_POLL_INTERVAL=300                            # Seconds between polling for new comments (default 300)
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

### Connecting to an Existing Database

The app can connect directly to a MySQL/MariaDB database. Set `DB_URL` in `.env`:

```env
DB_URL=mysql://user:password@host:3306/dbname
```

The database should have the following tables: `Tcomments`, `Tquestions`, `Tkdams`, `Tgrades`, `TgradesSemesters`.

When `DB_URL` is set, the app will:
- Load all data from the database on startup
- Poll for new comments every `DB_POLL_INTERVAL` seconds (default 300)
- Incrementally embed only new comments without rebuilding the full index

If `DB_URL` is not set, the app falls back to reading from a SQL dump file via `DB_SQL_PATH`.

To test locally with a MariaDB container:

```bash
podman run -d --name cs-rag-db \
  -e MYSQL_ROOT_PASSWORD=root \
  -e MYSQL_DATABASE=ragdb \
  -p 3306:3306 \
  -v cs-rag-db-data:/var/lib/mysql \
  mariadb:11

# Import the SQL dump
podman exec -i cs-rag-db mariadb -uroot -proot ragdb < your-dump.sql
```

### Docker

```bash
# Without FAISS index (built on first run)
docker build -t haifa-rag .

# With FAISS index baked in (for deployment)
docker build --build-arg INCLUDE_FAISS=true -t haifa-rag .

# Run
docker run --env-file .env --network=host haifa-rag
```

## API

The server exposes an **OpenAI-compatible API**, so it works with any OpenAI client library or tool:

```
POST /v1/chat/completions   # Chat (streaming & non-streaming)
GET  /v1/models              # List models
GET  /health                 # Health check
```

## Embedding as a Widget

The chatbot can be embedded in any website as a floating chat button. Add one line before `</body>`:

```html
<script src="https://YOUR_HOST/widget.js" data-host="https://YOUR_HOST"></script>
```

Replace `YOUR_HOST` with your server URL (e.g. `cs-rag.onrender.com` or `localhost:8000`).

The widget adds a chat button to the bottom-right corner. Clicking it opens the chatbot in a side panel with an option to expand to full screen.

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
├── format_data.py           # Data loader (MySQL or SQL dump)
├── evaluate.py              # LLM-as-judge evaluation pipeline
├── frontend/                # React chat interface
├── Dockerfile               # Multi-stage build (Node + Python)
└── render.yaml              # Render.com deployment config
```

## Authors

Gal Bareket & Ariel Yucovich
