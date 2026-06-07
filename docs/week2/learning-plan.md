# Week 2 — Retrieval Systems Lab: Learning Plan

**Goal:** Build a comprehensive understanding of modern retrieval systems — from raw embeddings through vector search, keyword search, hybrid retrieval, reranking, query transformation, and diversity-aware retrieval. The output is a reusable retrieval lab with formal evaluation metrics and a hard-negatives benchmark.

**Model:** qwen3-embedding:0.6b (primary), nomic-embed-text (comparison)
**Reranker:** dengcao/Qwen3-Reranker-0.6B (local via Ollama)
**Vector DB:** ChromaDB (lightweight, no server needed)

---

## Setup

```bash
# Pull embedding models
ollama pull qwen3-embedding:0.6b
ollama pull nomic-embed-text

# Pull reranker model
ollama pull dengcao/Qwen3-Reranker-0.6B

# Install Python dependencies
pip install chromadb rank-bm25 numpy umap-learn matplotlib scikit-learn
```

---

## Notebook 04 — Embeddings Intro

**Goal:** Understand what embeddings are, generate them locally, and build intuition through hands-on comparison.

### What to do

1. Use `ollama.embed()` to embed a few sentences:
   ```python
   import ollama

   response = ollama.embed(
       model='qwen3-embedding:0.6b',
       input='The sky is blue because of Rayleigh scattering'
   )
   print(len(response.embeddings[0]))  # check dimensions
   ```

2. Embed 5-6 sentence pairs — some semantically similar, some different:
   - "The cat sat on the mat" vs "A kitten rested on the rug" (similar meaning, different words)
   - "The cat sat on the mat" vs "Stock prices fell sharply today" (different meaning)
   - "How to fix a memory leak" vs "Debugging RAM issues in Python" (similar meaning, technical)

3. Compute cosine similarity by hand (numpy dot product of normalized vectors) to verify that similar sentences score higher.

4. Experiment with qwen3-embedding's flexible dimensions — embed the same text at 256, 512, and 1024 dimensions. Does similarity ranking stay consistent?

5. Compare qwen3-embedding:0.6b vs nomic-embed-text on the same sentence pairs. Do they agree on which pairs are most similar?

### Key concepts to understand

- **Embedding model vs chat model:** chat model takes text in, produces text out. Embedding model takes text in, produces a fixed-size numeric vector out. The vector captures *meaning*, not words.
- **Cosine similarity:** measures the angle between two vectors. 1.0 = identical direction (same meaning), 0.0 = orthogonal (unrelated), -1.0 = opposite.
- **Dimensions:** higher dimensions capture more nuance but use more storage. The tradeoff is worth measuring.

### Reference tutorials

- [Ollama Embeddings Docs](https://docs.ollama.com/capabilities/embeddings) — official API reference for `ollama.embed()`
- [Apidog: Qwen3-Embedding + Ollama Step-by-Step](https://apidog.com/blog/qwen-3-embedding-reranker-ollama/) — end-to-end RAG workflow with Qwen3
- [Glukhov: Qwen3-Embedding on Ollama](https://www.glukhov.org/rag/embeddings/qwen3-embedding-qwen3-reranker-on-ollama/) — embedding + reranker setup

---

## Notebook 05 — Embedding Visualization

**Goal:** Build visual intuition for how embeddings organize meaning in space.

### What to do

1. Prepare 30-40 sentences from 3-4 distinct topics (e.g., cooking, machine learning, sports, music). Include 8-10 sentences per topic.

2. Embed all sentences with qwen3-embedding:0.6b.

3. Reduce to 2D using UMAP and plot with matplotlib:
   ```python
   import umap
   import matplotlib.pyplot as plt
   import numpy as np

   reducer = umap.UMAP(n_components=2, random_state=42)
   embeddings_2d = reducer.fit_transform(np.array(embeddings))

   # Color by topic
   for topic, color in zip(topics, ['red', 'blue', 'green', 'orange']):
       mask = [t == topic for t in labels]
       plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=color, label=topic, alpha=0.7)
   plt.legend()
   plt.title("Sentence Embeddings by Topic")
   plt.show()
   ```

4. Observe: do sentences about the same topic cluster together? Are any topics closer to each other than expected?

5. Try the same visualization with nomic-embed-text — do the clusters look different?

6. Experiment: add a few ambiguous sentences (e.g., "Python is great for data science" — is it tech or animals?). Where do they land?

### Key concepts to understand

- **UMAP vs t-SNE:** Both reduce high-dimensional data to 2D for visualization. UMAP is faster, preserves both local and global structure, and is preferred for embedding visualization. t-SNE preserves local neighborhoods but distorts global distances.
- **Why visualize:** Numbers (cosine similarity = 0.83) are abstract. Seeing clusters on a plot makes the concept of "semantic space" concrete and memorable.

### Reference tutorials

- [Visualizing High-Dimensional BERT Embeddings (2026)](https://www.technetexperts.com/visualize-bert-sentence-embeddings/) — UMAP vs t-SNE comparison with Python code
- [Dimensionality Reduction in NLP: UMAP and t-SNE](https://exfai.xyz/docs/groups/dimensionality-reduction-in-nlp-visualizing-sentence-embeddings-with-umap-and-t-sne/) — roberta-base embeddings visualized with Plotly
- [How to Visualize Embeddings (Nomic Docs)](https://docs.nomic.ai/atlas/embeddings-and-retrieval/guides/how-to-visualize-embeddings) — t-SNE, UMAP, and Nomic Atlas guide
- [Visualizing Text and Image Embeddings: A Beginner's Guide](https://www.simiokunowo.me/blog/visualizing-text-and-image-embeddings) — UMAP + altair, beginner-friendly

---

## Notebook 06 — Instruction-Aware Embeddings

**Goal:** Test qwen3-embedding's instruction prefix feature and measure its impact on retrieval quality.

### What to do

1. Understand the instruction format. Qwen3-embedding accepts task-specific prefixes on queries:
   ```python
   # Without instruction
   response = ollama.embed(
       model='qwen3-embedding:0.6b',
       input='Python memory management'
   )

   # With instruction prefix
   response = ollama.embed(
       model='qwen3-embedding:0.6b',
       input='Instruct: Given a technical question, retrieve relevant programming tutorials\nQuery: Python memory management'
   )
   ```

2. Important: only **queries** get instructions. **Documents** are embedded without instructions.

3. Prepare a small test set:
   - 15-20 documents on a technical topic
   - 5 queries with known relevant documents

4. Embed documents (no instruction). Embed each query twice: once plain, once with a task-specific instruction. Compare which version retrieves the correct document in top-3.

5. Try different instruction styles and measure impact:
   - `"Given a web search query, retrieve relevant passages that answer the query"`
   - `"Given a technical question, retrieve code examples that solve the problem"`
   - `"Given a question, retrieve passages that answer the question"`

6. The Qwen3 team reports 1-5% improvement with instructions. Does your experiment confirm this?

### Key concepts to understand

- **Instruction-aware embeddings:** The model adjusts its embedding space based on what task you're performing. A "retrieve code" instruction emphasizes different features than a "classify sentiment" instruction.
- **Why this matters:** In production RAG, you can tune retrieval quality without changing models or retraining — just by writing better instructions. This is a zero-cost optimization.
- **Asymmetric embedding:** Queries and documents are treated differently. Queries are short and intent-laden; documents are long and information-dense. Instruction-aware models encode this asymmetry.

### Reference tutorials

- [How Qwen3 Instruction Prompting Improves Embedding Quality (Milvus)](https://milvus.io/ai-quick-reference/how-does-qwen3-instruction-prompting-improve-embedding-quality) — explains the mechanism
- [Qwen3-Embedding-0.6B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — official usage examples with instruction format
- [Mastering Text Embedding and Reranker with Qwen3 (Medium)](https://k-farruh.medium.com/mastering-text-embedding-and-reranker-with-qwen3-3284e7ae14c7) — domain-specific instruction examples (legal, e-commerce)
- [Hands-on RAG with Qwen3 Embedding using Milvus](https://milvus.io/blog/hands-on-rag-with-qwen3-embedding-and-reranking-models-using-milvus.md) — full RAG pipeline with instruction-aware embeddings

---

## Notebook 07 — ChromaDB Basics

**Goal:** Store embeddings in a vector database and retrieve by similarity.

### What to do

1. Create a ChromaDB collection with a custom Ollama embedding function:
   ```python
   import chromadb

   client = chromadb.Client()  # in-memory, no server needed
   collection = client.create_collection(name="week2_test")
   ```

2. Add 20-30 text passages to the collection. Use a topic you know well (ML concepts, cooking, etc.). ChromaDB will auto-embed them using its default model, or you can pass your own embeddings from Ollama.

3. Query the collection with natural language and inspect results:
   ```python
   results = collection.query(
       query_texts=["how does gradient descent work?"],
       n_results=3
   )
   ```

4. Experiment with:
   - **Distance metrics:** ChromaDB supports `l2`, `ip` (inner product), `cosine`. Create separate collections with each and compare retrieval order.
   - **Metadata filtering:** Add metadata (e.g., category, difficulty, source, section) to documents and combine similarity search with metadata filters. Compare pure semantic search against metadata-filtered semantic search — does filtering improve precision for scoped queries?
   - **Persistence:** Switch from in-memory to persistent storage (`chromadb.PersistentClient(path="./chroma_db")`) and verify data survives restart.

### Key concepts to understand

- **Vector database vs regular database:** A regular DB matches on exact values (WHERE category = 'ML'). A vector DB matches on *meaning* via nearest-neighbor search in embedding space.
- **Distance metrics:** Cosine focuses on direction (meaning similarity), L2 focuses on magnitude + direction, inner product is cosine without normalization. For normalized embeddings (which Ollama returns), cosine and inner product behave identically.
- **Collections:** ChromaDB's equivalent of a table. Each collection holds documents, their embeddings, and optional metadata.
- **Metadata filtering / faceted retrieval:** Combining vector similarity with metadata constraints narrows the search space. In production RAG, this is how you scope retrieval to a specific document type, time range, or category without changing the embedding model.

### Reference tutorials

- [DataCamp: ChromaDB Step-by-Step Guide](https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide) — beginner-friendly walkthrough of collections, adding, querying
- [Kalyna: Build Your First Vector Database (2026)](https://kalyna.pro/chroma-db-tutorial/) — covers custom embedding functions and persistence
- [Real Python: Vector Databases and Embeddings with ChromaDB](https://realpython.com/chromadb-vector-database/) — in-depth course with 17 lessons
- [Chroma Docs: Embedding Functions](https://docs.trychroma.com/docs/embeddings/embedding-functions) — how to plug in custom models

---

## Notebook 08 — BM25 vs Vector vs Hybrid Search

**Goal:** Compare keyword-based retrieval (BM25) with semantic retrieval (vector search) and hybrid, and understand when each wins.

### What to do

1. Prepare a test dataset of 20-30 passages. Intentionally include:
   - Passages that share keywords but differ in meaning (e.g., "Python the snake" vs "Python the language")
   - Passages that share meaning but differ in keywords (e.g., "fix a memory leak" vs "resolve RAM overflow issues")
   - Passages with rare/specific terms (e.g., error codes, product IDs)

2. Build a BM25 index:
   ```python
   from rank_bm25 import BM25Okapi

   tokenized_docs = [doc.lower().split() for doc in documents]
   bm25 = BM25Okapi(tokenized_docs)

   query = "how to debug memory issues"
   scores = bm25.get_scores(query.lower().split())
   ```

3. Build a ChromaDB vector index over the same documents.

4. Run 10 test queries against both and compare results side by side. Record:
   - Which retriever returned the correct passage in top-3?
   - Where did each method fail?

5. Categorize your findings:

   | Query type | BM25 | Vector | Why |
   |-----------|------|--------|-----|
   | Exact keyword match | Wins | — | BM25 matches tokens directly |
   | Synonym/paraphrase | — | Wins | Embeddings capture meaning |
   | Rare/specific terms | Wins | — | Rare tokens get high BM25 weight |
   | Typos/informal language | — | Wins | Embeddings are fuzzy |

6. **Hybrid search:** Combine both scores using Reciprocal Rank Fusion (RRF):
   ```python
   def rrf_score(bm25_rank, vector_rank, k=60):
       return 1 / (k + bm25_rank) + 1 / (k + vector_rank)
   ```
   Compare hybrid results against each method alone.

### Key concepts to understand

- **BM25:** A statistical ranking function based on term frequency and inverse document frequency. It asks "how often does this word appear in this document vs all documents?" No understanding of meaning — purely lexical.
- **Vector search:** Encodes query and documents into embedding space, finds nearest neighbors. Understands meaning but can miss exact keyword matches.
- **Hybrid search:** Runs both retrievers in parallel, fuses results via rank combination (RRF). The 2026 consensus is that hybrid outperforms either method alone for RAG.
- **Reciprocal Rank Fusion (RRF):** A rank-based fusion method that avoids the score normalization problem (BM25 scores and cosine scores are on completely different scales). It only cares about *rank position*, not raw scores.

### Reference tutorials

- [BM25 vs Vector Search: Choosing the Right Strategy](https://dev.to/aloknecessary/bm25-vs-vector-search-choosing-the-right-retrieval-strategy-for-production-systems-599n) — deep dive on when each wins, includes hybrid + reranking
- [Hybrid Search: BM25, Vector & Reranking (2026)](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026) — production reference with RRF explanation
- [Better RAG with Hybrid BM25 + Dense Vector Search](https://medium.com/@pbronck/better-rag-accuracy-with-hybrid-bm25-dense-vector-search-ea99d48cba93) — case study: 62% to 91% retrieval accuracy with hybrid
- [Hybrid Search in RAG: Combining Keyword and Vector Retrieval](https://teachmeidea.com/hybrid-search-bm25-vector-rag/) — practical tutorial with code

---

## Notebook 09 — Chunking & Embedding Throughput

**Goal:** Understand how text chunking affects embedding quality and measure local embedding speed.

### What to do

#### Part A — Chunking impact on retrieval

1. Take a long document (1-2 pages of text on a topic you know).

2. Chunk it 3 different ways:
   - **Full document** — embed the whole thing as one vector
   - **Fixed-size chunks** — 256 tokens with 50-token overlap
   - **Sentence-level** — each sentence as its own chunk

3. Store each chunking variant in a separate ChromaDB collection.

4. Run 5 queries and compare: which chunking strategy returns the most relevant result in top-3?

5. Observe the tradeoff: full-document embeddings capture broad context but lose detail. Sentence embeddings are precise but lose surrounding context. Fixed-size chunks are a middle ground.

#### Part B — Embedding throughput benchmark

1. Prepare 100-500 text passages (can reuse or generate).

2. Time how long it takes to embed them all:
   ```python
   import time

   start = time.time()
   for doc in documents:
       ollama.embed(model='qwen3-embedding:0.6b', input=doc)
   elapsed = time.time() - start
   print(f"{len(documents)} docs in {elapsed:.1f}s = {len(documents)/elapsed:.1f} docs/sec")
   ```

3. Also test batch embedding (pass multiple inputs at once if supported).

4. Record your M5's throughput — this tells you how fast you can index documents in Week 3.

### Key concepts to understand

- **Chunking is a first-class optimization lever.** A 2026 benchmark study found chunking strategy impacts retrieval quality as much as the embedding model itself — up to 9% recall difference between best and worst chunking on the same corpus.
- **The precision-context tradeoff:** Small chunks are precise to find but lack context for the LLM. Large chunks carry context but dilute the specific answer. The benchmark-validated default for 2026 is 512 tokens with 10-20% overlap, but optimal size depends on query type.
- **Hierarchical chunking** (production pattern): embed small child chunks for retrieval, but return the larger parent chunk to the LLM. You won't implement this yet, but knowing it exists prepares you for Week 4.

### Reference tutorials

- [RAG Chunking Strategies: A 2026 Retrieval Playbook](https://www.digitalapplied.com/blog/rag-chunking-strategies-2026-retrieval-quality-playbook) — comprehensive strategies with benchmarks
- [Chunking Strategies for RAG: A Complete Guide (2026)](https://atlan.com/know/chunking-strategies-rag/) — covers fixed, recursive, semantic, and hierarchical
- [RAG Chunking Strategies & Embeddings Optimization Benchmark Guide](https://nandigamharikrishna.substack.com/p/rag-chunking-strategies-and-embeddings) — data-backed defaults (512 tokens, 10-20% overlap)
- [RAG Architecture Guide 2026: Chunking, Embedding & Retrieval](https://pecollective.com/blog/rag-architecture-guide/) — end-to-end architecture overview

---

## Notebook 10 — Reranking Experiment

**Goal:** Understand the two-stage retrieval pattern: fast recall-oriented first stage, then precision-oriented reranking.

### What to do

1. Pull the Qwen3 reranker (already in setup):
   ```bash
   ollama pull dengcao/Qwen3-Reranker-0.6B
   ```

2. Build a first-stage retriever that gets top-20 candidates using vector search, BM25, or hybrid (reuse code from Notebook 08).

3. Implement reranking using the Qwen3 reranker. Since Ollama doesn't have a native rerank API, use the chat API with a yes/no relevance scoring approach:
   ```python
   from ollama import chat

   def rerank(query, documents):
       scores = []
       for doc in documents:
           prompt = f"""Judge whether the Document meets the requirements based on the Query.
   Answer only 'yes' or 'no'.

   Query: {query}
   Document: {doc}"""
           response = chat(
               model='dengcao/Qwen3-Reranker-0.6B',
               messages=[{"role": "user", "content": prompt}],
               options={"temperature": 0}
           )
           # Score based on whether response starts with "yes"
           score = 1.0 if response['message']['content'].strip().lower().startswith('yes') else 0.0
           scores.append(score)
       return scores
   ```

4. Compare retrieval quality at each stage:
   - First stage only (top-20 → return top-5)
   - First stage + reranker (top-20 → rerank → return top-5)
   - Measure how many of the final top-5 are actually relevant

5. Test across different first-stage retrievers:
   - Vector search top-20 → rerank → top-5
   - BM25 top-20 → rerank → top-5
   - Hybrid top-20 → rerank → top-5

6. Measure latency: how long does reranking 20 documents take on your M5? This determines whether reranking is viable for interactive use.

### Key concepts to understand

- **Bi-encoder vs cross-encoder:** A bi-encoder (embedding model) encodes query and documents independently — fast but approximate. A cross-encoder (reranker) processes query and document together as a single input — slow but much more accurate. The two-stage pattern combines the speed of bi-encoders with the precision of cross-encoders.
- **Recall vs precision tradeoff:** The first stage optimizes for recall (don't miss any relevant documents, even if some irrelevant ones sneak in). The reranker optimizes for precision (from the candidates, pick the truly relevant ones).
- **Why not just use the cross-encoder for everything?** A cross-encoder must score every (query, document) pair individually. For 10,000 documents, that's 10,000 forward passes. A bi-encoder embeds documents once, then retrieval is a fast nearest-neighbor search. The two-stage pattern: bi-encoder narrows 10,000 → 20, then cross-encoder scores 20 pairs.

### Reference tutorials

- [Apidog: Qwen3-Embedding + Reranker on Ollama](https://apidog.com/blog/qwen-3-embedding-reranker-ollama/) — step-by-step local setup with Python
- [Glukhov: Qwen3-Embedding + Reranker on Ollama](https://www.glukhov.org/rag/embeddings/qwen3-embedding-qwen3-reranker-on-ollama/) — detailed reranker integration guide
- [Reranking in RAG: Cross-Encoders, Cohere Rerank & FlashRank (Medium)](https://medium.com/@vaibhav-p-dixit/reranking-in-rag-cross-encoders-cohere-rerank-flashrank-c7d40c685f6a) — compares different reranking approaches
- [Advanced RAG: Cross-Encoders & Reranking (Towards Data Science)](https://towardsdatascience.com/advanced-rag-retrieval-cross-encoders-reranking/) — deep dive with fine-tuning guide
- [Build BGE Reranker for RAG (2026)](https://markaicode.com/bge-reranker-cross-encoder-reranking-rag/) — alternative reranker implementation

---

## Notebook 11 — Retrieval Metrics

**Goal:** Build a reusable evaluation framework with formal retrieval metrics to make all experiments rigorous and comparable.

### What to do

1. Create a small benchmark dataset: 10-15 queries with known relevant document IDs (ground truth). This is the dataset you'll reuse across all retrieval experiments.

2. Implement the standard retrieval metrics from scratch (numpy only, no frameworks):

   ```python
   def hit_at_k(retrieved_ids, relevant_ids, k):
       """Did at least one relevant doc appear in top-k?"""
       return int(any(doc in relevant_ids for doc in retrieved_ids[:k]))

   def precision_at_k(retrieved_ids, relevant_ids, k):
       """What fraction of top-k results are relevant?"""
       return sum(1 for doc in retrieved_ids[:k] if doc in relevant_ids) / k

   def recall_at_k(retrieved_ids, relevant_ids, k):
       """What fraction of all relevant docs appear in top-k?"""
       if not relevant_ids:
           return 0.0
       return sum(1 for doc in retrieved_ids[:k] if doc in relevant_ids) / len(relevant_ids)

   def mrr(retrieved_ids, relevant_ids):
       """Reciprocal rank of the first relevant result."""
       for i, doc in enumerate(retrieved_ids):
           if doc in relevant_ids:
               return 1.0 / (i + 1)
       return 0.0

   def ndcg_at_k(retrieved_ids, relevance_scores, k):
       """Normalized Discounted Cumulative Gain with graded relevance."""
       import math
       dcg = sum(relevance_scores.get(doc, 0) / math.log2(i + 2)
                 for i, doc in enumerate(retrieved_ids[:k]))
       ideal = sorted(relevance_scores.values(), reverse=True)[:k]
       idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal))
       return dcg / idcg if idcg > 0 else 0.0
   ```

3. Run all previous retrieval methods through the evaluation framework and create a comparison table:

   | Method | Hit@3 | Precision@3 | Recall@3 | MRR | nDCG@5 |
   |--------|-------|-------------|----------|-----|--------|
   | BM25 | | | | | |
   | Vector search | | | | | |
   | Hybrid (RRF) | | | | | |
   | Hybrid + reranker | | | | | |

4. Understand what each metric reveals:
   - **Hit@3** — "Did we find *anything* relevant?" (binary, forgiving)
   - **Precision@3** — "How clean are the top-3 results?" (penalizes noise)
   - **Recall@3** — "What fraction of relevant docs did we find?" (penalizes misses)
   - **MRR** — "How high is the first relevant result?" (cares about rank position)
   - **nDCG@5** — "How good is the overall ranking?" (graded relevance, rewards putting better results higher)

### Key concepts to understand

- **Order-unaware metrics** (Hit@K, Precision@K, Recall@K): treat top-K as a set — doesn't matter if the relevant doc is at position 1 or K.
- **Order-aware metrics** (MRR, nDCG): rank position matters — returning a relevant doc at position 1 scores higher than at position 5. These are more informative for RAG because LLMs often give more weight to earlier context.
- **Binary vs graded relevance:** Hit/Precision/Recall use binary relevance (relevant or not). nDCG supports graded relevance (a score of 3 = "highly relevant" vs 1 = "somewhat relevant"). Graded relevance is more realistic but requires more annotation effort.

### Reference tutorials

- [RAG Evaluation Metrics: Recall@K, MRR, Faithfulness & RAGAS (2026)](https://langcopilot.com/posts/2025-09-17-rag-evaluation-101-from-recall-k-to-answer-faithfulness) — comprehensive overview with examples
- [How to Evaluate Retrieval Quality in RAG: DCG@k and NDCG@k (TDS)](https://towardsdatascience.com/how-to-evaluate-retrieval-quality-in-rag-pipelines-part-3-dcgk-and-ndcgk/) — part of a 3-part series covering all metric families
- [Evaluation Measures in Information Retrieval (Pinecone)](https://www.pinecone.io/learn/offline-evaluation/) — clear explanations with visual examples
- [Evaluation Metrics for Search and Recommendation (Weaviate)](https://weaviate.io/blog/retrieval-evaluation-metrics) — includes pytrec_eval Python examples
- [Metrics for Retrieval in RAG Systems (Deconvolute)](https://deconvoluteai.com/blog/rag/metrics-retrieval) — categorized by binary/graded and order-aware/unaware

---

## Notebook 12 — Query Transformation

**Goal:** Understand how reformulating the query can improve retrieval quality independently of the retriever or embedding model.

### What to do

1. Pick 5 queries that performed poorly in previous notebooks (low MRR or Recall@3). These are your "hard queries."

2. Implement three query transformation strategies:

   **a) Query Rewriting** — Use the chat model to rephrase the query:
   ```python
   def rewrite_query(original_query):
       prompt = f"""Rewrite the following search query to be more specific and detailed.
   Return only the rewritten query, nothing else.

   Original: {original_query}"""
       response = chat_with_llm([{"role": "user", "content": prompt}], temperature=0)
       return response
   ```

   **b) Query Expansion** — Generate additional search terms:
   ```python
   def expand_query(original_query):
       prompt = f"""Generate 3 alternative phrasings of this search query.
   Return one per line, nothing else.

   Query: {original_query}"""
       response = chat_with_llm([{"role": "user", "content": prompt}], temperature=0.3)
       variants = response.strip().split('\n')
       return [original_query] + variants
   ```
   Retrieve for each variant, merge results with RRF.

   **c) HyDE (Hypothetical Document Embeddings)** — Generate a hypothetical answer, embed that instead of the query:
   ```python
   def hyde_query(original_query):
       prompt = f"""Write a short paragraph that would be the perfect answer to this question.
   Do not say you are generating an answer. Just write the answer directly.

   Question: {original_query}"""
       hypothetical_doc = chat_with_llm([{"role": "user", "content": prompt}], temperature=0)
       # Embed the hypothetical document, not the original query
       embedding = ollama.embed(model='qwen3-embedding:0.6b', input=hypothetical_doc)
       return embedding
   ```

3. For each hard query, compare retrieval results:
   - Original query → retrieve
   - Rewritten query → retrieve
   - Expanded query (multi-query + RRF) → retrieve
   - HyDE → retrieve

4. Measure with your Notebook 11 metrics. Create a comparison table:

   | Query | Original MRR | Rewritten MRR | Expanded MRR | HyDE MRR |
   |-------|-------------|---------------|-------------|----------|
   | ... | | | | |

5. Important caveat for HyDE: pass the **original query** (not the hypothetical doc) to the reranker if you chain it after HyDE. The reranker should judge relevance to what the user actually asked.

### Key concepts to understand

- **Query-document mismatch:** Users write short, informal queries. Documents are long and formal. This gap means even a perfect embedding model can fail — the query embedding may land far from the relevant document embedding in vector space.
- **HyDE bridges the gap:** By generating a hypothetical answer and embedding *that*, the search vector is now in "document space" rather than "query space." The embedding of a perfect answer is closer to real relevant documents than the embedding of a short question.
- **Multi-query increases recall:** Different phrasings of the same question hit different parts of the embedding space. Merging results via RRF compensates for any single phrasing being suboptimal.
- **When HyDE fails:** If the LLM hallucinates a bad hypothetical answer, the embedding will be wrong and retrieval will be worse. HyDE works best when the LLM has reasonable knowledge of the topic. For out-of-domain questions, simple query rewriting may be safer.

### Reference tutorials

- [Better RAG with HyDE: Hypothetical Document Embeddings](https://www.rohan-paul.com/p/better-rag-with-hyde-hypothetical) — explains the theory and provides implementation
- [How Query Expansion (HyDE) Boosts RAG Accuracy](https://www.chitika.com/hyde-query-expansion-rag/) — practical benchmarks
- [HyDE: A Query Transformation Technique for Advanced RAG](https://ragsimplified.hashnode.dev/hypothetical-document-embeddings-hyde-a-query-transformation-technique-for-advanced-rag) — step-by-step walkthrough
- [NirDiamant/RAG_Techniques — HyDE Notebook (GitHub)](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb) — complete Jupyter notebook
- [RAG Is Not Dead: Advanced Retrieval Patterns (2026)](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo) — covers HyDE, hybrid search, reranking, query decomposition
- [Introduction to HyDE (GeeksforGeeks)](https://www.geeksforgeeks.org/data-science/hypothetical-document-embeddings-hyde-hyde/) — beginner-friendly explanation

---

## Notebook 13 — MMR / Diversity Retrieval

**Goal:** Understand why vanilla top-k similarity search often returns redundant results, and how Maximal Marginal Relevance (MMR) improves context coverage.

### What to do

1. Create a scenario where redundancy is a problem: take a document with multiple sections, chunk it, and store in ChromaDB. Run a query — observe that top-5 results often come from the same section or contain near-duplicate content.

2. Implement MMR from scratch:
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   import numpy as np

   def mmr(query_embedding, candidate_embeddings, candidate_docs,
           lambda_param=0.5, top_k=5):
       """Select documents that balance relevance and diversity."""
       query_sim = cosine_similarity([query_embedding], candidate_embeddings)[0]
       selected = []
       remaining = list(range(len(candidate_docs)))

       for _ in range(top_k):
           best_score = -float('inf')
           best_idx = None
           for idx in remaining:
               relevance = query_sim[idx]
               # Max similarity to any already-selected document
               if selected:
                   selected_embs = candidate_embeddings[selected]
                   diversity_penalty = max(cosine_similarity(
                       [candidate_embeddings[idx]], selected_embs)[0])
               else:
                   diversity_penalty = 0
               score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
               if score > best_score:
                   best_score = score
                   best_idx = idx
           selected.append(best_idx)
           remaining.remove(best_idx)

       return [candidate_docs[i] for i in selected]
   ```

3. Compare three retrieval modes on the same queries:
   - **Pure top-k:** top-5 by cosine similarity (may be redundant)
   - **MMR lambda=0.7:** mostly relevant, slight diversity push
   - **MMR lambda=0.3:** strong diversity, may sacrifice some relevance

4. For each query, visually inspect: do the MMR results cover more distinct aspects of the answer? Does the LLM generate a better answer when given MMR-diverse context vs redundant top-k context?

5. Experiment with the lambda parameter: plot a curve of relevance score vs diversity score as lambda varies from 0 to 1.

### Key concepts to understand

- **The redundancy problem:** Top-k similarity search returns the K most similar documents. If your corpus has 3 near-duplicate paragraphs about the same subtopic, they'll dominate the top-5 — crowding out other relevant content.
- **MMR formula:** `score = lambda * similarity(doc, query) - (1-lambda) * max_similarity(doc, already_selected)`. It greedily selects documents that are relevant to the query AND dissimilar from documents already selected.
- **Lambda controls the tradeoff:** lambda=1.0 is pure relevance (identical to top-k). lambda=0.0 is pure diversity (ignores relevance). lambda=0.5-0.7 is typical for RAG.
- **Why this matters for RAG:** The LLM's answer quality depends on context coverage. Five redundant chunks give the LLM one perspective repeated five times. Five diverse chunks give it five different pieces of evidence — leading to more comprehensive and accurate answers.

### Reference tutorials

- [MMR with LangChain + ChromaDB (Medium)](https://kaustavmukherjee-66179.medium.com/improve-retrieval-of-documents-from-vectordb-using-maximum-marginal-relevance-mmr-for-balancing-f6ae56fb9512) — practical tutorial with ChromaDB integration
- [Mastering MMR: A Beginner's Guide (Medium)](https://medium.com/@ankitgeotek/mastering-maximal-marginal-relevance-mmr-a-beginners-guide-0f383035a985) — standalone Python implementation
- [Maximum Marginal Relevance (Full Stack Retrieval)](https://community.fullstackretrieval.com/retrieval-methods/maximum-marginal-relevance) — clear explanation of the algorithm

---

## Notebook 14 — Hard Negatives & Failure Analysis

**Goal:** Build a small diagnostic benchmark with intentionally difficult retrieval cases to stress-test your retrieval pipeline and understand its failure modes.

### What to do

1. Create a curated benchmark of 30-40 passages and 15-20 queries designed to expose specific failure modes:

   | Category | Example | Why it's hard |
   |----------|---------|---------------|
   | **Same keyword, different meaning** | "Java island" vs "Java programming" | Keyword match misleads BM25 and shallow embeddings |
   | **Same meaning, different words** | "terminate the process" vs "kill the running program" | Tests true semantic understanding |
   | **Ambiguous query** | "apple" (fruit? company? records?) | No single correct answer without context |
   | **Rare/specific terms** | "CUDA error: out of memory" | Embedding models may not handle error codes well |
   | **Cross-lingual** | Query in English, relevant doc contains Chinese terms | Tests multilingual embedding quality |
   | **Misleading keyword overlap** | "The model didn't perform well" vs "performing well on the model dataset" | Surface-level match but opposite meaning |
   | **Negation** | "not recommended for production" vs "recommended for production" | Tests whether embeddings distinguish negation |
   | **Long query vs short doc** | A detailed question vs a one-line answer | Tests asymmetric retrieval |

2. Run every retrieval method from the previous notebooks against this benchmark:
   - BM25
   - Vector search
   - Hybrid (RRF)
   - Hybrid + reranker
   - HyDE + vector search
   - MMR

3. Use Notebook 11 metrics (Hit@3, MRR, nDCG@5) to score each method on each failure category.

4. Create a failure analysis matrix:

   | Failure category | BM25 | Vector | Hybrid | +Reranker | +HyDE | +MMR |
   |-----------------|------|--------|--------|-----------|-------|------|
   | Same keyword, diff meaning | | | | | | |
   | Same meaning, diff words | | | | | | |
   | Ambiguous query | | | | | | |
   | ... | | | | | | |

5. For each failure category, write a 1-2 sentence explanation of *why* that method fails. This is the most valuable part — understanding failure modes lets you pick the right retrieval strategy for your use case.

6. Identify the "best stack" — which combination of techniques gives the highest overall score? Is there a single method that dominates, or does the best approach vary by query type?

### Key concepts to understand

- **Hard negatives:** Documents that look relevant (high keyword overlap or semantic similarity) but are actually irrelevant. They're the hardest cases for any retrieval system, and the cases that matter most — a retriever that fails on hard negatives will inject bad context into the LLM, causing hallucinations.
- **Failure analysis > benchmark scores:** A single MRR number hides what's actually going wrong. Breaking results down by failure category reveals actionable patterns: "Our vector search fails on negation" is something you can fix; "our MRR is 0.72" is not.
- **This benchmark is reusable.** You'll run new retrieval methods against it in Weeks 3-4. Every improvement can be measured against these hard cases.

### Reference tutorials

- [Hard Negative Mining for Domain-Specific Retrieval (arXiv 2025)](https://arxiv.org/html/2505.18366v1) — 15% MRR improvement via hard negative training
- [Negative Sampling Techniques in Information Retrieval: A Survey (arXiv 2026)](https://arxiv.org/pdf/2603.18005) — comprehensive taxonomy of negative sampling methods
- [RAG Is Not Dead: Advanced Retrieval Patterns (2026)](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo) — practical failure modes and solutions

---

## Week 2 Deliverable

A reusable **Retrieval Systems Lab** that covers:

1. Embedding generation with two models (qwen3-embedding, nomic-embed-text)
2. Embedding visualization (UMAP clusters)
3. Instruction-aware embedding experiments
4. ChromaDB vector store with metadata filtering
5. BM25 vs vector vs hybrid search comparison
6. Chunking impact analysis and throughput benchmarks
7. Two-stage retrieval with cross-encoder reranking
8. Formal retrieval metrics (Hit@K, Precision@K, Recall@K, MRR, nDCG@K)
9. Query transformation experiments (rewrite, expand, HyDE)
10. MMR diversity-aware retrieval
11. Hard negatives benchmark with failure analysis matrix

The benchmark dataset and evaluation functions carry forward into Weeks 3-4 as the retrieval evaluation foundation for your RAG pipeline.

---

## Dependencies to Add

```
chromadb
rank-bm25
numpy
umap-learn
matplotlib
scikit-learn
```
