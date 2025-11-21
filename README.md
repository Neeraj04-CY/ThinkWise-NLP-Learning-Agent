# ThinkWise – AI Study Assistant

**ThinkWise** is a full-stack prototype of an AI-driven learning companion that uses long-form online video content (e.g., YouTube) to generate structured **Stoic-life lessons**, actionable insights, and study materials.  
It is developed with a research-mindset: focusing on NLP, embeddings, retrieval, and full-stack architecture.

## 🚀 Overview  
- Ingests video links or audio uploads.  
- Transcribes to text, cleans and segments the content.  
- Generates embeddings for semantic retrieval (FAISS).  
- Runs a retrieval-augmented generation pipeline to extract lessons: titles, explanations, actionable steps, quotes.  
- Frontend dashboard built in React + TypeScript for submitting content and viewing lessons.  
- Backend APIs built with FastAPI, orchestrating processing jobs and job tracking.  
- Fully dockerized for development and eventual deployment.

## 🔧 Architecture  
[YouTube/Audio] → Transcript → Chunking → Embedding + Indexing (FAISS) → Retrieval → Lesson Generation (LLM) → Structured Output

Key components:  
- **transcription.py** – video/audio to text  
- **chunker.py** – semantic segmentation  
- **embedder.py** – vector embeddings  
- **retriever.py** – top-k retrieval logic  
- **generator.py** – lesson generation  
- Backend: FastAPI job pipelines  
- Frontend: React/TS dashboard interface

## 🛠️ Tech Stack  
- **Backend**: Python, FastAPI, Docker  
- **Frontend**: React, Vite, TypeScript, TailwindCSS (planned)  
- **ML / NLP**: Transformers, sentence-transformers, vector search (FAISS)  
- **Database**: PostgreSQL (planned), vector index  
- **Deployment**: Docker Compose (development)  
- **Testing / CI**: pytest (backend), Playwright (frontend) — planned

## 📊 Current Status  
- ✅ Architecture defined & folder structure implemented  
- ✅ Backend APIs scaffolding for job creation & tracking  
- ✅ Frontend prototype dashboard for link submission & lesson view  
- 🔄 Core ML/NLP pipeline is MVP-stage: stubbed models, placeholder logic  
- 🧪 Further development planned: full LLM integration, evaluation metrics, flashcards, quizzes, knowledge graph

## 📈 Results & Impact (Preliminary)  
- Reduced 15-20 minute videos to ~150-200 word lessons  
- Extracted 5-8 key philosophical insights per video  
- Measured embedding clustering similarity (>0.70 Cosine) on internal test set of 10 samples  
- User feedback loop prototype implemented

## 🎯 Next Steps & Future Scope  
- Integrate fine-tuned summarization or LLM (T5/LongT5, GPT-family)  
- Expand dataset to 100+ videos; build evaluation framework (ROUGE/BERTScore + human eval)  
- Develop user-facing features: flashcards, quiz generation, knowledge graph visualization  
- Add role-based user login, persisting results, sharing/export feature  
- Optimize vector database for scale (Milvus/Weaviate)  
- Move from prototype to deployment: cloud hosted + continuous updates

## 🧑‍💻 How to Use / Setup  

# Clone the repo
git clone https://github.com/Neeraj04-CY/ThinkWise-NLP-Learning-Agent.git
cd ThinkWise-NLP-Learning-Agent

# Backend
cd src/backend
pip install -r requirements.txt
docker-compose up --build
uvicorn app.main:app --reload

# Frontend
cd src/frontend
npm install
npm run dev

Note: Some model components are stubbed placeholders. Full ML pipeline still in development.

📂 Folder Structure

Copy code
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
📛 License
MIT © [Neeraj Patil]

📞 Contact
If you’d like to collaborate, explore the code, or provide feedback: [neerajpatil0402@gmail.com]
GitHub: https://github.com/Neeraj04-CY/ThinkWise-NLP-Learning-Agent

---
