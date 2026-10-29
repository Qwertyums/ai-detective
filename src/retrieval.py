"""Deterministic retrieval shared by both context packagings."""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

def load_jsonl(filepath, model_class):
    with Path(filepath).open(encoding='utf-8-sig') as stream:
        return [model_class.model_validate(json.loads(line)) for line in stream if line.strip()]

def tokenize(text):
    return re.findall(r'\w+', text.casefold())

def bm25_retrieve(query, claims, top_k=15):
    if top_k < 0:
        raise ValueError('top_k must be nonnegative')
    if not claims:
        return []
    corpus = [tokenize(c.text + ' ' + c.evidence) for c in claims]
    if not any(corpus):
        return sorted(claims, key=lambda c: c.claim_id)[:min(15, top_k)]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    indices = sorted(range(len(claims)), key=lambda i: (-scores[i], claims[i].claim_id))[:min(15, top_k)]
    return [claims[i] for i in indices]

def build_context(mode, question, claims, documents):
    from src.context_builder import prepare_context
    return prepare_context(mode, question, claims, documents).text
