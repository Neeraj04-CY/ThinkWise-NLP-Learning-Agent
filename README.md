# ThinkWise – AI Study Assistant

ThinkWise is a full-stack prototype of an AI-driven learning companion that uses long-form online video content (e.g. YouTube lectures, talks, podcasts) to generate structured Stoic-style lessons, actionable insights, and study materials.

It is developed with a research mindset, focusing on NLP, embeddings, retrieval, and full-stack architecture rather than a quick demo.

---

## 🚀 Overview

- Ingests video links or audio uploads (YouTube and audio link support in the MVP).
- Transcribes to text, cleans and segments the content.
- Generates embeddings for semantic retrieval (FAISS).
- Runs a retrieval-augmented generation (RAG) pipeline to extract lessons:
  - Titles
  - Explanations
  - Actionable steps
  - Quotes and tags
- Frontend dashboard built in React + TypeScript for submitting content and viewing lessons.
- Backend APIs built with FastAPI, orchestrating processing jobs and job tracking.
- Dockerized for local development and future deployment.

---

## 🔧 Architecture

**High-level flow**

`[YouTube / Audio] → Transcript → Chunking → Embedding + Indexing (FAISS) → Retrieval → Lesson Generation (LLM) → Structured Output`

**Key components**

- `transcription.py` – video/audio → text (currently stubbed, designed for WhisperX / faster-whisper).
- `chunker.py` – semantic segmentation of transcript into context windows.
- `embedder.py` – vector embeddings (currently stubbed; interface matches sentence-transformers / OpenAI).
- `retriever.py` – FAISS-based top‑k retrieval logic.
- `generator.py` – lesson generation, designed as a RAG-style LLM wrapper.
- Backend: FastAPI job pipelines (`/api/jobs`, `/api/results`, `/api/feedback`).
- Frontend: React/TS dashboard interface for job creation, monitoring, and lesson display.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Docker
- **Frontend:** React, Vite, TypeScript, (TailwindCSS – planned)
- **ML / NLP:** Transformers, sentence-transformers (planned), vector search (FAISS)
- **Database:** PostgreSQL (planned), vector index on disk
- **Deployment:** Docker Compose (development)
- **Testing / CI:** pytest (backend), Playwright (frontend) — planned

---

## 📊 Current Status

> Prototype / MVP — Architecture in place, some parts stubbed, not yet fully stable.

- ✅ Architecture defined & folder structure implemented.
- ✅ Backend API scaffolding for:
  - Creating processing jobs
  - Tracking job status
  - Fetching structured results
  - Submitting feedback
- ✅ Frontend prototype dashboard for:
  - Submitting YouTube links
  - Polling job status
  - Rendering structured “lessons” (titles, actions, timestamps, confidence, tags)
- 🔄 Core ML/NLP pipeline is **MVP-stage**:
  - Uses stubbed models and placeholder logic for transcription, embeddings, and generation.
- ⚠️ The codebase is **not yet fully functional**:
  - Contains some implementation errors and incomplete components.
  - Intended as an active in-progress research prototype.

---

## 📈 Preliminary Results & Research Direction

*(Based on internal experiments and target behavior; subject to change as the pipeline matures.)*

- Reduces 15–20 minute videos to ~150–200 word condensed lessons.
- Extracts ~5–8 key philosophical insights per video.
- Embedding-based clustering with cosine similarity > 0.70 on a small internal test set (≈10 samples).
- User feedback loop prototype implemented via `/api/feedback` for rating lesson quality.

---

## 🎯 Next Steps & Future Scope

- Integrate fine-tuned summarization or LLMs (e.g. T5/LongT5, GPT-family) for lesson generation.
- Replace stubs with:
  - WhisperX / faster-whisper transcription
  - Real embedding models and RAG pipeline
- Expand dataset to 100+ videos; build evaluation framework:
  - ROUGE/BERTScore + human evaluation.
- Develop user-facing study features:
  - Flashcards with difficulty levels
  - Quiz generation
  - Knowledge graph visualization of topics and sources
- Add basic user system:
  - Role-based login
  - Persisting results per user
  - Export/sharing of study notes
- Optimize vector database for scale (e.g. Milvus / Weaviate).
- Move from prototype to deployment:
  - Cloud hosting
  - Continuous integration & updates.

---

## 🧑‍💻 How to Use / Setup

> Note: This is **work in progress**. Some components are stubbed and parts of the pipeline may not run end-to-end without modification.

### 1. Clone the repository

```bash
git clone https://github.com/Neeraj04-CY/ThinkWise-NLP-Learning-Agent.git
cd ThinkWise-NLP-Learning-Agent
```

### 2. Backend

From the project root:

```bash
cd src/backend
pip install -r requirements.txt
# Option A: run via Docker Compose for backend + frontend
docker-compose up --build
# Option B: run backend only (if you have Python env ready)
uvicorn app.main:app --reload
```

Backend will be available at `http://localhost:8000`.

### 3. Frontend

In another terminal:

```bash
cd src/frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173`.

---

## 📂 Folder Structure

```text
thinkwise/
  docker-compose.yml
  requirements.txt
  src/
    backend/
      app/
        main.py
        core/
          config.py
          logging.py
        api/
          jobs.py
          results.py
          feedback.py
        db/
          schemas.py
          crud.py
        models/
          transcription.py
          chunker.py
          embedder.py
          retriever.py
          generator.py
    frontend/
      package.json
      vite.config.ts
      tsconfig.json
      src/
        main.tsx
        App.tsx
```

---

## 📛 License

MIT © Neeraj Patil

---

## 📞 Contact

If you’d like to collaborate, explore the code, or provide feedback:

- Email: [neerajpatil0402@gmail.com](mailto:neerajpatil0402@gmail.com)  
- GitHub: [Neeraj04-CY](https://github.com/Neeraj04-CY/ThinkWise-NLP-Learning-Agent)
