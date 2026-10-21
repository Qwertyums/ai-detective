import json
from rank_bm25 import BM25Okapi
from typing import List
from src.schemas import Claim, Document


def load_jsonl(filepath: str, model_class):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [model_class(**json.loads(line)) for line in f]


def bm25_retrieve(query: str, claims: List[Claim], top_k: int = 15) -> List[Claim]:
    tokenized_corpus = [claim.text.split() for claim in claims]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    # Получаем индексы топ-K
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [claims[i] for i in top_indices]


def build_context(mode: str, question: str, claims: List[Claim], documents: List[Document],
                  stress_remove_claim_id: str = None, stress_add_contradiction: str = None) -> str:
    # Применение стресс-теста 2: Удаление посылки
    filtered_claims = [c for c in claims if c.claim_id != stress_remove_claim_id]

    if mode == "A":
        context = "ДОКУМЕНТЫ:\n" + "\n".join([f"[{d.document_id}] {d.title}: {d.text}" for d in documents])
        return context

    elif mode == "B":
        retrieved = bm25_retrieve(question, filtered_claims, top_k=15)
        context = "РЕЛЕВАНТНЫЕ УТВЕРЖДЕНИЯ:\n" + "\n".join([f"[{c.claim_id}] {c.text}" for c in retrieved])
        if stress_add_contradiction:
            context += f"\n[ДОБАВЛЕНО] {stress_add_contradiction}"
        return context

    elif mode == "C":
        from src.padic import group_by_padic_branches
        retrieved = bm25_retrieve(question, filtered_claims, top_k=15)  # Тот же набор, что и в B
        branches = group_by_padic_branches(retrieved)

        context = "P-АДИЧЕСКИЕ ВЕТВИ УТВЕРЖДЕНИЙ:\n"
        for branch_name, branch_claims in branches.items():
            context += f"\nВЕТВЬ {branch_name}:\n"
            for c in branch_claims:
                context += f"  {c.claim_id}: {c.text}\n"
        if stress_add_contradiction:
            context += f"\n[ДОБАВЛЕНО] {stress_add_contradiction}"
        return context
    return ""
