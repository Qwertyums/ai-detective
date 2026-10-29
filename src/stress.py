"""One-factor mutations and explicit, content-bound human review."""
import hashlib
import json
import random
from src.schemas import Question


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def snapshot(documents, claims, questions):
    return {name: [x.model_dump(mode='json') for x in values] for name, values in
            [('documents', documents), ('claims', claims), ('questions', questions)]}


def apply_case(case, documents, claims):
    documents = [d.model_copy(deep=True) for d in documents]
    claims = [c.model_copy(deep=True) for c in claims]
    if case.kind == 'permutation':
        random.Random(42).shuffle(documents)
    elif case.kind == 'remove_premise':
        target = next(c for c in claims if c.claim_id == case.remove_claim_id)
        # Remove the source excerpt too: full-context A must lose the premise.
        for doc in documents:
            doc.text = doc.text.replace(target.evidence, '')
        remaining = [c for c in claims if c.claim_id != target.claim_id]
        for claim in remaining:
            doc = next(d for d in documents if d.document_id == claim.document_id)
            if claim.evidence not in doc.text:
                raise ValueError('Removal affects another claim; prepare a separate atomic excerpt')
        claims = remaining
    else:
        if case.added_document.document_id in {d.document_id for d in documents} or case.added_claim.claim_id in {c.claim_id for c in claims}:
            raise ValueError('Added IDs must be unique')
        if case.added_claim.document_id != case.added_document.document_id or case.added_claim.evidence not in case.added_document.text:
            raise ValueError('Added evidence must refer to its source')
        documents.append(case.added_document.model_copy(deep=True))
        claims.append(case.added_claim.model_copy(deep=True))
    return documents, claims


def review_digest(case, documents, claims, question):
    payload = case.model_dump(mode='json', exclude={'reviewed_by', 'reviewed_at', 'review_digest'})
    modified_docs, modified_claims = apply_case(case, documents, claims)
    return digest({'case': payload, 'before': snapshot(documents, claims, [question]),
                   'after': snapshot(modified_docs, modified_claims, [question])})


def is_reviewed(case, documents, claims, question):
    return bool(case.reviewed_by and case.reviewed_at and
                case.review_digest == review_digest(case, documents, claims, question))


def expected_question(question, case):
    return Question(**{**question.model_dump(), 'gold_status': case.expected_status,
                       'gold_answer': case.rationale, 'gold_evidence_ids': case.expected_evidence_ids,
                       'gold_counterevidence_ids': case.expected_counterevidence_ids})
