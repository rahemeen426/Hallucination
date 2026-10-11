"""
Groundedness check -- Method 2: SelfCheckGPT-style self-consistency.

Generates the same answer multiple times at higher temperature, then
measures how semantically consistent the samples are with the original
answer using embedding similarity. Low consistency = higher hallucination
risk, independent of whether the answer looks fluent or confident. This is
a second, independent signal from groundedness.py's LLM-as-judge method --
they can (and sometimes will) disagree, which is expected and useful to
observe.

Usage (standalone test):
    python src/selfcheck.py
"""

import ollama

LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def generate_samples(prompt: str, n_samples: int = 4, temperature: float = 0.9) -> list[str]:
    samples = []
    for _ in range(n_samples):
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": temperature},
        )
        samples.append(response["response"])
    return samples


def selfcheck_consistency(prompt: str, main_answer: str, n_samples: int = 4) -> dict:
    """Returns a consistency score (0-1, higher = more consistent/trustworthy)
    by comparing embedding similarity between the main answer and resampled
    versions generated at higher temperature."""
    samples = generate_samples(prompt, n_samples=n_samples)

    main_vec = _embed(main_answer)
    similarities = [_cosine_similarity(main_vec, _embed(s)) for s in samples]

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    if avg_similarity >= 0.85:
        verdict = "consistent"
    elif avg_similarity >= 0.65:
        verdict = "somewhat inconsistent"
    else:
        verdict = "inconsistent"

    return {
        "consistency_score": round(avg_similarity, 3),
        "verdict": verdict,
        "n_samples": n_samples,
        "sample_similarities": [round(s, 3) for s in similarities],
    }


if __name__ == "__main__":
    sample_prompt = "In one sentence, what was Apple's total net sales in fiscal 2023?"
    sample_answer = "Apple's total net sales in fiscal 2023 were $383.3 billion."
    result = selfcheck_consistency(sample_prompt, sample_answer, n_samples=3)
    import json
    print(json.dumps(result, indent=2))
