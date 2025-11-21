# ThinkWise Differentiated Learning Roadmap

## High-Level Principle
Move from one-off definitions to an active, personalized learning experience. Every topic should deliver layered explanations, interactive practice, source-backed evidence, and clear next steps.

## Phase 1 (Days 1-7): Progressive Depth & Practice
1. **Three-Level Explanations**  
   - Backend returns `short`, `medium`, `deep` sections per topic.  
   - Frontend toggle (TL;DR / Learn Quickly / Deep Dive).
2. **Worked Examples Generator**  
   - Generate 3 numeric examples with step-by-step reasoning (easy → hard).  
   - Provide structured JSON so frontend can render each step.
3. **Misconceptions & Self-Check**  
   - Add `common_misconceptions` + "How to know you understand" checklist.
4. **Mini Practice Quiz**  
   - Auto-create 3 questions (MCQ + short answer) with explanations.

## Phase 2 (Weeks 2-3): Personalization & Guidance
1. **Concept Map / Dependency Graph**  
   - LLM outputs nodes + edges; render with D3/vis.js.
2. **Adaptive Practice & SRS Export**  
   - Generate question bank, track correctness, schedule next review (SM-2 style).  
   - Provide Anki export.
3. **Micro-Projects & Application Tasks**  
   - Each topic returns 1–2 real-world mini projects (inputs, deliverables, rubric).
4. **Persona & Modality Modes**  
   - API flag (`persona`) to switch between intuitive/technical/child-friendly explanations.

## Phase 3 (Weeks 4-6): Trust & Rich Media
1. **RAG + Citations Layer**  
   - Index curated sources; answers include inline citations and evidence snippets.
2. **Video Summaries & Timestamped Quizzes**  
   - Accept YouTube link → transcript → key takeaways + timestamped questions.
3. **Interactive Step-Reveal & Code Sandboxes**  
   - For derivations/code, reveal one step at a time; runnable code blocks for Python/JS topics.
4. **Confidence + Provenance UI**  
   - Display model confidence, list sources, show retrieved snippets.

## Phase 4 (Weeks 7+): Intelligent Coaching & Curriculum Fit
1. **Personal Learning Profile & Heatmap**  
   - Aggregate quiz history, show strengths/weaknesses, recommended daily plan.
2. **Curriculum Mapping**  
   - Align topics to IB/AP/A-level/college syllabi; provide exam-style practice.
3. **Misconception Detector**  
   - Let users submit their explanation; system highlights gaps + fixes.
4. **Live Collaborative & Teacher Tools**  
   - Class dashboards, assignments, AI-generated feedback on submissions.

## Implementation Enablers
- **Structured JSON contracts** for every AI response (validated via Pydantic).  
- **Caching layer** (Redis + DB fallback) for repeated topic requests.  
- **Async processing** (Celery tasks) for heavy operations (RAG, video transcription).  
- **Source ingestion pipeline** (PDF/HTML chunking + embeddings).  
- **Metrics**: engagement, quiz accuracy, retention, feedback ratio.

## Quick Wins to Tackle Next
1. Extend `/api/v1/module/generate` schema to include `examples`, `quiz`, `projects`, `sources` fields.  
2. Build a lightweight frontend view to showcase the depth toggles + worked examples.  
3. Add caching + analytics logging around module generation to capture usage.
