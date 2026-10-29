"""Durable experiment records, paired comparisons and replayable provenance."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
from src.context_builder import prepare_context
from src.data_validation import ROOT
from src.evaluation import evaluate_single_question
from src.llm import infer, LLMError, configuration, system_prompt
from src.stress import apply_case, digest, expected_question, is_reviewed, snapshot


def result_filename(record, full_id=False):
    """Readable Windows-safe name; timestamps are displayed in Moscow time."""
    def part(value):
        return re.sub(r'[^\w-]+', '-', str(value), flags=re.UNICODE).strip('-_')[:40] or 'unknown'
    created = datetime.fromisoformat(record['created_at'])
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    stamp = created.astimezone(timezone(timedelta(hours=3))).strftime('%Y-%m-%d_%H-%M-%S_MSK')
    mode = {'A': 'A-full', 'B': 'B-bm25', 'C': 'C-padic'}.get(record['mode'], record['mode'])
    kind = record.get('kind', 'baseline')
    scenario = {'baseline': 'baseline', 'permutation': 'shuffle', 'add_contradiction': 'contradiction',
                'remove_premise': 'remove-' + str((record.get('case') or {}).get('remove_claim_id') or 'premise')}.get(kind, record.get('case_id', kind))
    provider = record.get('settings', {}).get('provider', 'local')
    provider = 'api' if provider == 'openai' else provider
    identifier = part(record['run_id'])
    suffix = identifier if full_id else identifier[:8]
    return '_'.join([stamp, part(record['question_id']), part(mode), part(scenario), part(provider), suffix]) + '.json'


def run_one(question, mode, documents, claims, case=None, output_dir=ROOT/'results/runs', provider=None, model=None):
    requested_model, settings = configuration(provider, model)
    provider = 'chatgpt' if settings['provider'] == 'chatgpt' else 'api_key'
    original = snapshot(documents, claims, [question])
    reviewed = case is None or is_reviewed(case, documents, claims, question)
    if case:
        documents, claims = apply_case(case, documents, claims)
    context = prepare_context(mode, question.question, claims, documents)
    expected = expected_question(question, case) if case else question
    record = {
        'run_id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'question_id': question.question_id, 'mode': mode, 'case_id': case.case_id if case else 'baseline',
        'kind': case.kind if case else 'baseline', 'model': requested_model, 'settings': settings,
        'prompt': system_prompt(), 'prompt_sha256': hashlib.sha256(system_prompt().encode()).hexdigest(),
        'original': original, 'modified': snapshot(documents, claims, [question]),
        'case': case.model_dump(mode='json') if case else None,
        'context': context.text, 'claim_ids': context.claim_ids, 'document_order': context.document_order,
        'expected': expected.model_dump(mode='json'), 'reviewed': reviewed,
        'dataset_digest': digest(original),
        'evaluation': None, 'output': None, 'error': None, 'usage': {}, 'raw_response': None, 'attempts': [],
    }
    # Paired modes share a fingerprint; a changed prompt/settings/source cannot
    # accidentally be averaged with an earlier experiment.
    record['experiment_digest'] = digest({'original': original, 'case': record['case'],
                                         'model': requested_model, 'settings': settings, 'prompt': record['prompt']})
    record['study_digest'] = digest({'documents': original['documents'], 'claims': original['claims'],
                                    'model': requested_model, 'settings': settings, 'prompt': record['prompt']})
    try:
        result = infer(question, context.text, context.claim_ids, provider=provider, model=requested_model,
                       auth_profile=settings.get('auth_profile'))
        record.update(output=result.output.model_dump(mode='json'), raw_response=result.raw_response,
                      usage=result.usage, model_revision=result.model_revision, attempts=result.attempts,
                      api_metadata=result.metadata)
        if reviewed:
            record['evaluation'] = evaluate_single_question(expected, result.output)
            record['evaluation']['answer'] = result.output.answer
    except LLMError as exc:
        record.update(error=str(exc), raw_response=exc.raw_response, usage=exc.usage, attempts=exc.attempts,
                      api_metadata=exc.metadata)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    record['filename'] = result_filename(record)
    try:
        stream = (folder / record['filename']).open('x', encoding='utf-8')
    except FileExistsError:
        record['filename'] = result_filename(record, full_id=True)
        stream = (folder / record['filename']).open('x', encoding='utf-8')
    with stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    return record


def load_records(folder=ROOT/'results/runs'):
    return sorted([json.loads(p.read_text(encoding='utf-8')) for p in Path(folder).glob('*.json')], key=lambda r: (r['created_at'], r['run_id']))


def paired_records(records):
    """Latest attempt per question+mode; only complete comparable A/B/C triples."""
    groups = {}
    for record in sorted(records, key=lambda r: r['created_at']):
        groups.setdefault(record['experiment_digest'], {})[record['mode']] = record
    return [r for group in groups.values() if set(group) == {'A','B','C'} for r in group.values()]
