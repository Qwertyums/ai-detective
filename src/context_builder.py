"""Only packaging changes between B and C; claim bodies are identical."""
from dataclasses import dataclass
import json
from src.retrieval import bm25_retrieve
from src.padic import group_by_padic_branches

@dataclass
class Context:
    text: str
    claim_ids: list[str]
    document_order: list[str]

def render_claim(claim, documents):
    source = next(d for d in documents if d.document_id == claim.document_id)
    payload = {**claim.model_dump(mode='json'), 'source_type': source.source_type}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)

def prepare_context(mode, question, claims, documents):
    if mode not in ('A', 'B', 'C'):
        raise ValueError('Unknown context mode')
    if mode == 'A':
        parts = []
        for doc in documents:
            parts.append(json.dumps(doc.model_dump(mode='json'), ensure_ascii=False))
            parts.append('Индекс утверждений документа:\n' + '\n'.join(render_claim(c, documents) for c in claims if c.document_id == doc.document_id))
        return Context('\n\n'.join(parts), [c.claim_id for c in claims], [d.document_id for d in documents])
    selected = bm25_retrieve(question, claims)
    if mode == 'B':
        text = '\n'.join(render_claim(c, documents) for c in selected)
    else:
        text = '\n\n'.join('ВЕТВЬ ' + key + '\n' + '\n'.join(render_claim(c, documents) for c in branch)
                            for key, branch in group_by_padic_branches(selected, documents).items())
    return Context(text, [c.claim_id for c in selected], [d.document_id for d in documents])
