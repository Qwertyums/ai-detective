import itertools
import json
from unittest.mock import patch
import pytest
from pydantic import ValidationError
from src.schemas import ModelOutput, StressCase
from src.data_validation import ROOT, load_dataset, validate_dataset
from src.retrieval import bm25_retrieve, load_jsonl
from src.padic import calculate_padic_distance, encode_path, get_padic_code, group_by_padic_branches
from src.context_builder import prepare_context, render_claim
from src.llm import LLMError, parse_output
from src.stress import apply_case, expected_question, is_reviewed, review_digest
from src.evaluation import aggregate_metrics, analyze_stability, evaluate_single_question
from src.experiments import load_records, run_one, paired_records

@pytest.fixture
def data():
    return load_dataset()

@pytest.fixture
def cases():
    # Tests must not depend on human approvals recorded in the working dataset.
    return [c.model_copy(update={'reviewed_by': None, 'reviewed_at': None, 'review_digest': None})
            for c in load_jsonl(ROOT/'data/stress_cases.jsonl', StressCase)]

def response(**updates):
    return {'question_id':'Q1', 'status':'entailed', 'answer':'10:10', 'evidence_ids':['C21'],
            'counterevidence_ids':[], 'confidence':.7, 'explanation':'Счёт указан в C21.', **updates}

@pytest.mark.parametrize('y,d', [((0,1,2),0), ((1,1,2),1), ((0,2,2),1/3), ((0,1,0),1/9)])
def test_distance_levels(y,d):
    assert calculate_padic_distance((0,1,2), y) == pytest.approx(d)

def test_all_ternary_ultrametrics():
    paths = list(itertools.product(range(3), repeat=3))
    assert {encode_path(x) for x in paths} == set(range(27))
    for x,y,z in itertools.product(paths, repeat=3):
        assert calculate_padic_distance(x,z) <= max(calculate_padic_distance(x,y), calculate_padic_distance(y,z))

def test_rules_and_time_are_metadata_based(data):
    docs, claims, _ = data
    rule = next(c for c in claims if c.claim_id == 'C17')
    exception = next(c for c in claims if c.claim_id == 'C18')
    assert get_padic_code(rule, docs) == (2,0,0)
    assert get_padic_code(exception, docs) == (2,2,0)
    sample = [next(c for c in claims if c.claim_id == key).model_copy(update={'document_id':'D05', 'status':'direct'}) for key in ['C16','C25']]
    branch = next(iter(group_by_padic_branches(sample, docs).values()))
    assert [c.claim_id for c in branch] == ['C25','C16']

def test_dataset_integrity_and_size(data):
    result = validate_dataset(*data)
    assert result['documents'] == 10 and result['claims'] == 40 and result['questions'] == 12
    assert 5000 <= result['words'] <= 8000
    docs, claims, qs = data
    claims[0].evidence = 'несуществующая цитата'
    with pytest.raises(ValueError, match='verbatim'):
        validate_dataset(docs, claims, qs)

@pytest.mark.parametrize('change', [
    {'confidence':1.1}, {'status':'both'}, {'evidence_ids':['C999']},
    {'status':'insufficient','answer':'Виноват Иванов','missing_information':'Нет видео'},
    {'status':'insufficient','answer':None}, {'question_id':'Q2'},
    {'status':'both','counterevidence_ids':['C21']},
])
def test_invalid_model_contracts(change):
    with pytest.raises(ValidationError):
        ModelOutput.model_validate(response(**change), context={'allowed_ids':['C21'], 'question_id':'Q1'})

def test_insufficient_abstains():
    out = ModelOutput.model_validate(response(status='insufficient', answer=None, missing_information='Нет идентификации пользователя.'))
    assert out.answer is None

@pytest.mark.parametrize('raw', ['broken JSON', '[]', 'null', '{"status":"entailed"}', '```'])
def test_bad_json_is_system_error(raw):
    with pytest.raises(LLMError):
        parse_output(raw, 'Q1', ['C21'])

def test_missing_request_metadata_can_be_filled():
    payload = response()
    payload.pop('question_id')
    assert parse_output(json.dumps(payload), 'Q1', ['C21']).question_id == 'Q1'

def test_missing_semantic_fields_are_not_invented():
    with pytest.raises(LLMError, match='status, confidence, explanation'):
        parse_output(json.dumps({'answer':'10:10', 'missing_information':None}), 'Q1', ['C21'])

def test_unquoted_known_key_can_be_normalized():
    from src.llm import normalize_response
    raw = json.dumps(response(), ensure_ascii=False)[:-1] + ', missing_information: null}'
    normalized, changes = normalize_response(raw)
    assert changes == ['quoted_property_names']
    assert json.loads(normalized)['missing_information'] is None
    assert parse_output(raw, 'Q1', ['C21']).confidence == .7

def test_key_normalization_preserves_quoted_string_contents():
    explanation = 'Фраза: , missing_information: текст; кавычки: "status".'
    raw = json.dumps(response(explanation=explanation), ensure_ascii=False)[:-1] + ', missing_information: null}'
    assert parse_output(raw, 'Q1', ['C21']).explanation == explanation

def test_normalization_does_not_accept_unknown_keys():
    raw = json.dumps(response())[:-1] + ', invented_field: 123}'
    with pytest.raises(LLMError):
        parse_output(raw, 'Q1', ['C21'])

def test_incomplete_response_gets_one_repair_request(data):
    from types import SimpleNamespace
    from src.llm import infer
    incomplete = json.dumps({'answer':'10:10', 'missing_information':None})
    valid = json.dumps(response())
    usage = {'input_tokens':100, 'output_tokens':20, 'response_seconds':1}
    client = SimpleNamespace(close=lambda:None)
    with patch('src.llm.create_client', return_value=client), patch('src.llm._generate', side_effect=[(incomplete,usage,{'model':'test'}),(valid,usage,{'model':'test'})]) as generate:
        result = infer(data[2][0], 'Доступная цитата C21', ['C21'])
    assert result.output.status == 'entailed'
    assert result.usage == {'input_tokens':200,'output_tokens':40,'response_seconds':2,'attempt_count':2,'retry_count':1}
    assert result.attempts[0]['raw_response'] == incomplete and result.attempts[0]['error']
    assert result.attempts[1]['error'] is None
    retry_messages = generate.call_args_list[1].args[1]
    assert retry_messages[-2] == {'role':'assistant','content':incomplete}
    assert 'status, confidence, explanation' in retry_messages[-1]['content']
    assert 'C21' in retry_messages[-1]['content']

def test_repeated_invalid_response_remains_an_error(data):
    from types import SimpleNamespace
    from src.llm import infer
    raw = json.dumps({'answer':'10:10'})
    with patch('src.llm.create_client', return_value=SimpleNamespace(close=lambda:None)), patch('src.llm._generate', return_value=(raw,{'input_tokens':10},{'model':'test'})) as generate:
        with pytest.raises(LLMError) as failure:
            infer(data[2][0], 'context', ['C21'])
    assert generate.call_count == 2
    assert failure.value.raw_response == raw
    assert len(failure.value.attempts) == 2 and failure.value.usage['input_tokens'] == 20

def test_unavailable_model_does_not_trigger_format_retries(data):
    from src.llm import infer
    with patch('src.llm.create_client', side_effect=LLMError('offline')), patch('src.llm._generate') as generate:
        with pytest.raises(LLMError, match='offline'):
            infer(data[2][0], 'context', ['C21'])
    generate.assert_not_called()

def test_bm25_and_fair_contexts_for_every_question(data):
    docs, claims, questions = data
    assert bm25_retrieve('empty', []) == []
    assert len(bm25_retrieve('боты', claims, 100)) == 15
    for q in questions:
        b = prepare_context('B', q.question, claims, docs)
        c = prepare_context('C', q.question, claims, docs)
        assert b.claim_ids == c.claim_ids
        assert len(b.claim_ids) == 15
        for claim in claims:
            if claim.claim_id in b.claim_ids:
                assert render_claim(claim, docs) in b.text and render_claim(claim, docs) in c.text

def test_stress_cases_have_valid_changed_sources(data, cases):
    docs, claims, questions = data
    assert len(cases) == 36
    assert {(c.question_id, c.kind) for c in cases} == {
        (q.question_id, kind) for q in questions
        for kind in ('permutation', 'remove_premise', 'add_contradiction')}
    for case in cases:
        ds, cs = apply_case(case, docs, claims)
        q = next(q for q in questions if q.question_id == case.question_id)
        validate_dataset(ds, cs, [expected_question(q,case)], enforce_size=False)
        before = prepare_context('A', q.question, claims, docs)
        after = prepare_context('A', q.question, cs, ds)
        assert before.text != after.text
        if case.kind == 'remove_premise':
            target = next(c for c in claims if c.claim_id == case.remove_claim_id)
            assert target.evidence not in after.text
            assert len(cs) == len(claims)-1
        elif case.kind == 'permutation':
            assert sorted(d.model_dump_json() for d in ds) == sorted(d.model_dump_json() for d in docs)
            assert before.document_order != after.document_order
        else:
            assert len(ds) == len(docs)+1 and len(cs) == len(claims)+1

def test_review_expires_on_content_change(data, cases):
    from datetime import datetime, timezone
    docs, claims, qs = data
    case = cases[0]
    q = next(q for q in qs if q.question_id == case.question_id)
    assert not is_reviewed(case, docs, claims, q)
    case.reviewed_by = 'Test reviewer'
    case.reviewed_at = datetime.now(timezone.utc)
    case.review_digest = review_digest(case, docs, claims, q)
    assert is_reviewed(case, docs, claims, q)
    docs[0].text += ' Изменение.'
    assert not is_reviewed(case, docs, claims, q)

def test_cannot_combine_stress_factors(cases):
    obj = cases[0].model_dump()
    obj['remove_claim_id'] = 'C25'
    with pytest.raises(ValidationError):
        StressCase.model_validate(obj)

def test_recognition_denominators(data):
    _,_,qs = data
    evaluations = []
    for q in qs:
        out = ModelOutput(question_id=q.question_id, status=q.gold_status,
                          answer=None if q.gold_status == 'insufficient' else q.gold_answer,
                          evidence_ids=q.gold_evidence_ids, counterevidence_ids=q.gold_counterevidence_ids,
                          confidence=.9, explanation='Проверочный ответ',
                          missing_information='Не хватает сведений' if q.gold_status == 'insufficient' else None)
        evaluations.append(evaluate_single_question(q,out))
    metrics = aggregate_metrics(evaluations).iloc[0]
    assert metrics['Both Recognition Rate'] == 1
    assert metrics['Insufficient Recognition Rate'] == 1
    assert metrics['Confident Unsupported Rate'] == 0

def test_no_information_high_confidence_rate(data):
    q = data[2][-1]
    out = ModelOutput(**response(question_id=q.question_id, confidence=.9))
    assert aggregate_metrics([evaluate_single_question(q,out)]).iloc[0]['Confident Unsupported Rate'] == 1

def test_llm_unavailable_is_saved_error_not_answer(data, tmp_path):
    docs, claims, qs = data
    with patch('src.experiments.infer', side_effect=LLMError('offline')):
        record = run_one(qs[0],'B',docs,claims,output_dir=tmp_path,provider='api_key')
    assert record['output'] is None and record['evaluation'] is None
    assert record['error'] == 'offline'
    assert load_records(tmp_path)[0]['run_id'] == record['run_id']

def test_unreviewed_stress_is_not_scored(data, cases, tmp_path):
    from src.llm import Inference
    docs, claims, qs = data
    output = ModelOutput(**response())
    with patch('src.experiments.infer', return_value=Inference(output, json.dumps(response()), {}, 'test')):
        record = run_one(qs[0],'B',docs,claims,cases[0],tmp_path,provider='api_key')
    assert record['output'] and record['evaluation'] is None and not record['reviewed']

def test_pairing_deduplicates_and_rejects_incomplete():
    rows = [{'experiment_digest':'same','mode':m,'created_at':str(i)} for i,m in enumerate('ABCB')]
    rows.append({'experiment_digest':'other','mode':'A','created_at':'9'})
    paired = paired_records(rows)
    assert len(paired) == 3
    assert next(x for x in paired if x['mode']=='B')['created_at'] == '3'

def test_permutation_vs_targeted_reaction():
    base = [{'question_id':'Q1','predicted_status':'entailed','status_accuracy':1,'evidence_ids':['C01'],'answer':'Да'}]
    stress = [{'question_id':'Q1','predicted_status':'insufficient','status_accuracy':1,'evidence_ids':[],'answer':None}]
    assert analyze_stability(base, stress, 'permutation')['stability_score'] == 0
    result = analyze_stability(base, stress, 'remove_premise')
    assert result['stability_score'] is None and result['expected_status_accuracy'] == 1

def test_streamlit_boots():
    from streamlit.testing.v1 import AppTest
    with patch('src.chatgpt_auth.ChatGPTAuth.profiles',return_value={'active':None,'profiles':[]}):
        app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception

def test_saved_comparison_and_calibration(data, tmp_path):
    from src.llm import Inference
    from scripts.summarize_results import summarize
    docs, claims, qs = data
    for mode in 'ABC':
        out = ModelOutput(**response())
        with patch('src.experiments.infer', return_value=Inference(out, json.dumps(response()),
                              {'input_tokens':100, 'output_tokens':20, 'response_seconds':1}, 'test')):
            run_one(qs[0],mode,docs,claims,output_dir=tmp_path,provider='api_key')
    summary = summarize(load_records(tmp_path))
    assert summary['paired_records'] == 3
    assert len(summary['calibration']) == 3
    assert all(row['Status Accuracy'] == 1 for row in summary['metrics'])
    json.dumps(summary, allow_nan=False)


def test_summary_shows_partial_stress_without_claiming_complete_comparison(data):
    from copy import deepcopy
    from scripts.summarize_results import summarize
    evaluation = evaluate_single_question(data[2][0], ModelOutput(**response()))
    evaluation['answer'] = '10:10'
    base = {'run_id':'base', 'created_at':'1', 'experiment_digest':'base', 'study_digest':'study',
            'question_id':'Q1', 'case_id':'baseline', 'kind':'baseline', 'mode':'C',
            'evaluation':evaluation, 'model':'test', 'settings':{}, 'attempts':[],
            'error':None, 'reviewed':True, 'usage':{}}
    records = [{**deepcopy(base), 'mode':m, 'run_id':m} for m in 'ABC']
    stress = {**deepcopy(base), 'run_id':'stress', 'created_at':'2',
              'experiment_digest':'stress', 'case_id':'P_Q1', 'kind':'permutation'}
    records += [stress, {**deepcopy(stress), 'run_id':'retry', 'created_at':'3'}]
    result = summarize(records)
    assert result['paired_records'] == 3 and result['available_records'] == 4
    assert {r['kind'] for r in result['metrics']} == {'baseline'}
    partial = [r for r in result['available_metrics'] if r['kind'] == 'permutation']
    assert len(partial) == 1 and partial[0]['mode'] == 'C' and partial[0]['attempts'] == 1
    assert partial[0]['Status Accuracy'] == 1
    coverage = next(r for r in result['coverage'] if r['kind'] == 'permutation')
    assert coverage['missing_modes'] == ['A','B']
    assert len(result['available_stability']) == 1
    assert result['available_stability'][0]['paired_questions'] == 1
    records[-1]['evaluation'] = None
    records[-1]['reviewed'] = False
    result = summarize(records)
    row = next(r for r in result['available_metrics'] if r['kind'] == 'permutation')
    assert row['valid_scored'] == 0 and row['unreviewed'] == 1
    assert 'Status Accuracy' not in row
    json.dumps(result, allow_nan=False)
