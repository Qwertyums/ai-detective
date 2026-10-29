"""Cross-file checks run before experimental requests."""
from collections import Counter
from pathlib import Path
from src.retrieval import load_jsonl
from src.schemas import Claim, Document, Question

ROOT = Path(__file__).resolve().parents[1]


def validate_dataset(documents, claims, questions, enforce_size=True):
    for name, items, key in [('documents', documents, 'document_id'), ('claims', claims, 'claim_id'), ('questions', questions, 'question_id')]:
        ids = [getattr(x, key) for x in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate {name} IDs')
    docs = {d.document_id: d for d in documents}
    claim_ids = {c.claim_id for c in claims}
    for claim in claims:
        if claim.document_id not in docs or not claim.evidence or claim.evidence not in docs[claim.document_id].text:
            raise ValueError(f'{claim.claim_id}: evidence must be a verbatim source excerpt')
    for q in questions:
        if not set(q.gold_evidence_ids + q.gold_counterevidence_ids) <= claim_ids:
            raise ValueError(f'{q.question_id}: unknown gold ID')
        if q.gold_status == 'both' and (not q.gold_evidence_ids or not q.gold_counterevidence_ids):
            raise ValueError(f'{q.question_id}: incomplete both reference')
    words = sum(len(d.text.split()) for d in documents)
    if enforce_size:
        if not 8 <= len(documents) <= 10 or not 5000 <= words <= 8000 or not 30 <= len(claims) <= 40:
            raise ValueError(f'Invalid dataset size: {len(documents)} documents, {words} words, {len(claims)} claims')
        if len(questions) != 12 or Counter(q.question_type for q in questions) != Counter(dict.fromkeys(['direct', 'multi_hop', 'temporal', 'rule_exception', 'contradiction', 'insufficient'], 2)):
            raise ValueError('Expected two questions of each of six types')
    return {'documents': len(documents), 'words': words, 'claims': len(claims), 'questions': len(questions)}


def load_dataset(directory=ROOT / 'data'):
    directory = Path(directory)
    docs = load_jsonl(directory / 'documents.jsonl', Document)
    claims = load_jsonl(directory / 'claims.jsonl', Claim)
    questions = load_jsonl(directory / 'questions.jsonl', Question)
    validate_dataset(docs, claims, questions)
    return docs, claims, questions


if __name__ == '__main__':
    print(validate_dataset(*load_dataset()))
