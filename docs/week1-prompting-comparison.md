# Week 1: Prompting Comparison Report

**Model:** Qwen 3.5 4B (local via Ollama)
**Task:** Sentiment classification of slang sentence: *"The outfit ate, no crumbs."*
**Ground truth:** Positive (Gen-Z slang meaning "flawless, perfect")

---

## Strategies Tested

### 1. Zero-Shot (think=False)

**Prompt:** Direct classification instruction with no examples.

**Result:** Positive (correct) — but the model hedged heavily, didn't recognize the slang, and nearly misclassified it. It arrived at the right answer through a shaky interpretation ("lack of crumbs implies cleanliness").

**Takeaway:** Got lucky. The reasoning path was fragile and wouldn't generalize to other slang inputs.

### 2. Few-Shot with Generic Examples (think=False)

**Prompt:** 3 examples using standard, formal English (e.g., "I love this product", "The delivery arrived late").

**Result:** Neutral (incorrect). The model learned the output format but matched the sentence's unconventional tone to the "nothing special" neutral example rather than recognizing positive intent.

**Takeaway:** Few-shot examples teach the model what the task *looks like*. If all examples use formal language, the model assumes formal language is the domain — slang falls outside.

### 3. Few-Shot with Slang-Aware Examples (think=False)

**Prompt:** 3 examples using informal/tricky language — sarcasm ("Love waiting 40 minutes for cold fries"), slang ("She really did her thing"), and misleading phrasing ("This game is stupidly good"). Each example included a "Why" explanation.

**Result:** Positive (correct). The model recognized "ate" as slang praise and "no crumbs" as emphasis of perfection.

**Takeaway:** Domain-matched examples dramatically improve accuracy. Adding brief "Why" explanations to examples helped the model learn the *reasoning pattern*, not just the label pattern.

### 4. Chain-of-Thought with Generic Examples (think=True)

**Prompt:** Same generic few-shot examples as Strategy 2, with Qwen's built-in thinking mode enabled.

**Result:** No answer produced. The model spent its entire output budget on thinking, going in circles trying to parse "outfit ate" literally. Never reached a conclusion in the `content` field.

**Takeaway:** On a 4B model, thinking mode with an unfamiliar input can cause the model to spiral. The reasoning budget was consumed by confusion rather than productive analysis.

### 5. Chain-of-Thought via Prompt (think=False, "think step by step" in prompt)

**Prompt:** Same generic few-shot examples, with "Please think step by step" appended to the prompt.

**Result:** Neutral (incorrect). The model wrote extensive reasoning but talked itself into the wrong answer — it interpreted "outfit ate" literally, concluded the sentence was nonsensical, and defaulted to Neutral.

**Takeaway:** More reasoning doesn't help when the model lacks the right knowledge. CoT gave the model more rope to hang itself with.

---

## Summary Table

| Strategy | Answer | Correct? | Key Observation |
|----------|--------|----------|-----------------|
| Zero-shot | Positive | Yes | Right answer, wrong reasoning — fragile |
| Few-shot (generic) | Neutral | No | Formal examples misled the model |
| Few-shot (slang-aware) | Positive | Yes | Best result — domain-matched examples + explanations |
| CoT (think=True) | (none) | N/A | Model spiraled, exhausted output on thinking |
| CoT (in-prompt) | Neutral | No | More reasoning amplified confusion |

---

## Key Findings

1. **Example quality > example quantity.** Generic few-shot examples can actively hurt performance by anchoring the model to the wrong domain. Slang-aware examples with "Why" explanations were the only few-shot strategy that worked.

2. **Chain-of-thought is not universally helpful.** On a small model (4B) with unfamiliar input, CoT provides more room to go wrong rather than more room to reason correctly. The model doesn't know what "ate, no crumbs" means — thinking longer about it doesn't fix a knowledge gap.

3. **Small models are sensitive to prompt design.** The same model gave correct or incorrect answers depending entirely on how the prompt was structured. For a 4B model, carefully chosen examples matter more than sophisticated reasoning strategies.

4. **Qwen 3.5's think mode has a failure mode.** When the model is confused, enabling `think=True` can cause it to consume all output tokens on reasoning without producing a final answer. This is worth watching for in future experiments.

---

# Week 1: Chat Parameter Experiments

**Model:** Qwen 3.5 9B (local via Ollama)
**Task:** Movie idea generation — creative task chosen to make parameter effects visible
**Method:** Each parameter tested by varying one value at a time, running 3 generations per setting

---

## Parameters Tested

### 1. Seed — Reproducibility

**Setup:** 5 runs with `seed=42`, default temperature.

**Result:** All 5 outputs were identical, confirming deterministic generation.

**Takeaway:** Fixed seed enables reproducible output. Essential for evaluation and debugging — isolates prompt changes from sampling randomness. Combine with `temperature=0` for full determinism.

### 2. Temperature — Creativity vs Coherence

**Setup:** 3 runs each at temperature 0.0, 0.5, 1.0, 1.5.

| Temperature | Behavior |
|-------------|----------|
| 0.0 | Fully deterministic — 3 identical outputs. Always picks the highest-probability token. |
| 0.5 | Mild variation — outputs stay in the same theme ("memory trading") but differ in details. |
| 1.0 | Real diversity — one output broke entirely from the dominant theme (lighthouse/gravity idea). |
| 1.5 | Maximum variety — each run produces a genuinely different concept. But grammar starts degrading. |

**Takeaway:** Temperature is a creativity-coherence tradeoff. Higher values produce more novel outputs but sacrifice polish. For factual tasks (RAG), use 0.0–0.3. For creative generation, 0.7–1.0.

### 3. top_p — Nucleus Sampling

**Setup:** 3 runs each at top_p 0.0, 0.5, 1.0 with temperature fixed at 0.5.

| top_p | Behavior |
|-------|----------|
| 0.0 | Acts like temperature=0 — fully deterministic, 3 identical outputs. |
| 0.5 | Some variation but all stay in "memory/archivist" territory. |
| 1.0 | Wider range — similar to unconstrained sampling. |

**Takeaway:** `top_p` is a softer control than temperature. It restricts the model to tokens whose cumulative probability reaches P%. In practice, `top_p=0.9` is a common "safety net" paired with moderate temperature — caps wildness while allowing variety.

### 4. top_k — Hard Token Cutoff

**Setup:** 3 runs each at top_k 0, 1, 10, 50 with temperature fixed at 0.5.

| top_k | Behavior |
|-------|----------|
| 0 | No restriction — most diverse outputs (AI dreaming/poetry idea appeared). |
| 1 | Greedy decoding — fully deterministic, 3 identical outputs. |
| 10 | Slight variation, all stay close to "memory archivist" theme. |
| 50 | Moderate variety, still thematically similar. |

**Takeaway:** `top_k` is a blunter instrument than `top_p`. It hard-caps the number of candidate tokens regardless of their probability distribution. In practice, `top_p` is preferred because probability mass is more meaningful than an arbitrary count.

### 5. repeat_penalty — Repetition Control

**Setup:** 3 runs each at repeat_penalty 1.0, 1.5, 2.0 with temperature fixed at 0.5.

| Penalty | Behavior |
|---------|----------|
| 1.0 | No penalty — fluent output but repetitive structure across runs ("In a near-future where memories can be..."). |
| 1.5 | Forces vocabulary diversity — model breaks out of the dominant pattern (jazz pianist, AI architect). Sentences get longer. |
| 2.0 | Very aggressive — outputs become run-on sentences with awkward phrasing. The model avoids repeating *any* word so hard it distorts grammar. |

**Takeaway:** Default (1.0–1.2) is fine for normal use. Higher values force novelty but degrade coherence. At 2.0, the model is visibly struggling to avoid word reuse at the cost of readability.

---

## Recommended Settings by Use Case

| Use case | Recommended params |
|----------|-------------------|
| RAG factual answers (Week 3–4) | `temperature=0, top_p=0.9` |
| Agent tool selection (Week 5) | `temperature=0` — deterministic tool picks |
| Creative generation | `temperature=0.7–1.0, top_p=0.9, repeat_penalty=1.1` |
| Evaluation / benchmarks | `temperature=0` or fixed `seed` — reproducibility is essential |

---

## Key Findings (Chat Parameters)

1. **temperature=0 and top_k=1 and top_p=0.0 all produce identical behavior** — greedy decoding. Three different knobs, same effect. Understanding this equivalence prevents confusion when reading other people's configs.

2. **top_p is strictly better than top_k for most use cases.** top_p adapts to the shape of the probability distribution; top_k applies an arbitrary cutoff. A flat distribution with top_k=10 throws away viable tokens; a spiky distribution with top_k=50 includes garbage tokens.

3. **repeat_penalty has a narrow useful range.** 1.0–1.2 for normal text, up to 1.5 for tasks where diversity matters. Beyond 1.5, grammar deterioration outweighs the benefit.

4. **Parameter choice depends on the downstream task.** This is the most important lesson: there is no universally "best" setting. RAG answers need determinism; creative tasks need randomness; evaluation needs reproducibility. Choosing params is a design decision, not a tuning problem.
