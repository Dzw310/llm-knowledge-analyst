# LLM Knowledge Analyst — Project Plan

**Goal:** Build a full-stack RAG + Agent system to learn and evaluate LLM applications.
**Timeline:** 8 weeks (June 2026 – August 2026), ~20 hrs/week
**Approach:** Local-first, build bottom-up, evaluate continuously

---

## Week 1 — Local LLM Setup + Prompting Lab

**Learning goals:** Local model serving, prompt engineering comparison

- Install Ollama, pull a small model (e.g., `qwen2.5:3b` or `phi3:mini`)
- Build a CLI script that sends prompts to local model via Ollama's REST API
- Prompting comparison: zero-shot vs few-shot vs chain-of-thought on same task set
- Log and compare results — this becomes the first evaluation baseline

**Note:** Decorators and async are useful but not urgent here. Learn them later when tool calling (Week 5) and parallel workflows (Week 8) demand them.

**Deliverable:** CLI chat tool + prompt comparison report

---

## Week 2 — Retrieval Systems Lab

**Learning goals:** Embedding models, vector similarity search, keyword search, hybrid retrieval, reranking, query transformation, diversity retrieval, retrieval evaluation metrics, failure analysis

- Pull local models: qwen3-embedding:0.6b (embedding), nomic-embed-text (comparison), Qwen3-Reranker-0.6B (reranking)
- Set up ChromaDB (lightweight, no server needed)
- Build pipeline: embed text chunks → store in ChromaDB → query by similarity
- Experiment with different embedding models, distance metrics, flexible dimensions, and instruction-aware embeddings
- Visualize embedding clusters with UMAP to build intuition for semantic space
- Compare BM25 vs vector search vs hybrid (RRF) on same dataset
- Measure chunking impact on embedding quality + local embedding throughput
- Implement two-stage retrieval: first-stage recall (top-20) → cross-encoder reranking (top-5)
- Build formal retrieval evaluation framework: Hit@K, Precision@K, Recall@K, MRR, nDCG@K
- Experiment with query transformation: rewriting, expansion, HyDE (Hypothetical Document Embeddings)
- Implement MMR (Maximal Marginal Relevance) for diversity-aware retrieval
- Build hard-negatives benchmark with failure analysis across all retrieval methods

**Detailed plan:** See [docs/week2/learning-plan.md](week2/learning-plan.md)

**Deliverable:** Reusable retrieval lab comparing embedding models, vector search, BM25, hybrid search, metadata filtering, query transformation, reranking, MMR, chunking strategies — evaluated with formal metrics on a hard-negatives benchmark

---

## Week 3 — Basic RAG Pipeline + Simple Evaluation

**Learning goals:** Document ingestion, basic chunking, retrieval-augmented generation, evaluation basics

- Build document loader (PDF or markdown)
- Implement basic chunking (fixed-size with overlap)
- Connect retrieval to local LLM: query → retrieve context → generate answer
- Create a small test Q&A dataset with ground-truth answers
- Simple evaluation: base model (no RAG) vs base model + RAG

**Deliverable:** End-to-end RAG + base-vs-RAG comparison results

---

## Week 4 — RAG Optimization + Advanced Evaluation

**Learning goals:** Advanced chunking for production, end-to-end RAG evaluation, answer quality scoring, prompt template refinement

Note: Basic retrieval techniques (reranking, query transformation, hybrid search, retrieval metrics) are now covered in Week 2's Retrieval Systems Lab. Week 4 focuses on applying and extending those techniques within the full RAG pipeline.

- Experiment with advanced chunking strategies (recursive, semantic, hierarchical) and systematically tune chunk size/overlap/top-k using the Week 2 evaluation framework
- Implement hierarchical chunking: embed small child chunks for retrieval, return larger parent chunks to the LLM
- Explore contextual/late chunking: embed chunks with surrounding context for better retrieval
- Extend evaluation from retrieval-only (Week 2) to end-to-end RAG: add answer quality scoring (faithfulness, relevance, completeness) on top of retrieval metrics
- Integrate the best retrieval stack from Week 2 (hybrid + reranker + query transformation) into the RAG pipeline and measure end-to-end improvement
- Run the hard-negatives benchmark (from Week 2) through the full RAG pipeline — does the LLM compensate for retrieval failures, or amplify them?
- Refine prompt templates for context injection: test different context formats, ordering, and instruction styles

**Deliverable:** Optimized RAG pipeline with eval-backed improvement metrics across both retrieval quality and answer quality

---

## Week 5 — Tool Calling & Basic Agent Planning

**Learning goals:** Function/tool calling, ReAct pattern, agent loop design, agentic RAG, decorators and async (as needed)

- Understand tool use: what it means for an LLM to "call a function"
- Learn decorators and async here — they're natural prerequisites for tool registration and concurrent calls
- Build a minimal agent from scratch (no framework): think → act → observe loop
- Give it 2-3 tools: RAG search, calculator, web lookup
- Understand how the LLM decides which tool to use (ReAct, function calling formats)
- Implement **agentic RAG**: the agent decides *when* and *how* to retrieve, not just what — e.g., deciding whether a query needs retrieval at all, choosing between search strategies, or iterating on retrieval if first results are insufficient

**Deliverable:** Hand-built agent that reasons about and selects tools, including intelligent RAG retrieval

---

## Week 6 — Memory, Reflection & Workflow Orchestration

**Learning goals:** Conversation memory, agent self-reflection, multi-step workflows

- Add memory to your agent: short-term (conversation history) and long-term (persistent)
- Implement reflection: agent reviews its own output and self-corrects
- Build a multi-step workflow: e.g., research → summarize → fact-check pipeline
- Pick one framework (LangGraph or LlamaIndex) and rebuild to compare with from-scratch

**Deliverable:** Agent with memory + reflection + orchestrated workflow

---

## Week 7 — Fine-Tuning Comparison Experiment

**Learning goals:** LoRA/QLoRA basics, training data preparation, before/after evaluation

This is a focused experiment, not a full training milestone. Scope it to what's achievable with your hardware and time.

- Prepare a small domain-specific training dataset from your documents
- Fine-tune local model using LoRA (e.g., `unsloth` or `transformers + peft`)
- Run the 4-way comparison: base → base+RAG → fine-tuned → fine-tuned+RAG
- Use existing eval framework to quantify improvement at each stage
- If fine-tuning hits hardware/debugging walls, document what you learned and focus on the comparison with whatever model you have

**Deliverable:** Fine-tuning experiment report + comparison matrix with metrics

---

## Week 8 — App Integration, Evaluation Dashboard & Portfolio

**Learning goals:** UI development, system integration, project packaging

**Required:**
- Build Streamlit/Gradio UI: chat interface, source citations, eval dashboard
- Integrate all components: RAG + agents + fine-tuned model (if successful)
- Write project documentation: architecture, decisions, evaluation results
- Package for portfolio: README, demo screenshots/video, key findings

**Bonus (if time allows):**
- Parallel agents: multiple agents working concurrently on sub-tasks

**Deliverable:** Demo-ready app + evaluation dashboard + portfolio write-up

---

## Evaluation Strategy (Continuous)

Evaluation is not a phase — it runs throughout:

| Week | What to evaluate |
|------|-----------------|
| 1 | Prompt strategy comparison (zero/few/CoT) + chat parameter effects |
| 2 | Embedding models, retrieval methods (BM25/vector/hybrid/reranker/HyDE/MMR), formal metrics on hard-negatives benchmark |
| 3 | Base model vs base+RAG |
| 4 | Advanced chunking impact, end-to-end RAG answer quality (faithfulness, relevance), retrieval-to-generation error propagation |
| 5 | Agent tool selection accuracy, agentic RAG decision quality |
| 6 | Reflection quality, workflow correctness |
| 7 | 4-way comparison (base/RAG/fine-tuned/fine-tuned+RAG) |
| 8 | End-to-end system evaluation |

## Portfolio Story

> "I deployed a local LLM, built a RAG pipeline, designed an agent system with tool calling, memory, and reflection, ran a fine-tuning comparison experiment, and evaluated every stage with metrics — showing when each technique adds value and when it doesn't."
