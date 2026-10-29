"""Streamlit research bench. Run from the repository root."""
from datetime import datetime, timezone
from io import StringIO
import json
import pandas as pd
import streamlit as st
from src.schemas import Document, Claim, Question, StressCase
from src.data_validation import ROOT, load_dataset, validate_dataset
from src.retrieval import load_jsonl
from src.context_builder import prepare_context
from src.experiments import load_records, run_one, result_filename
from src.auth_ui import connection_controls
from src.stress import apply_case, review_digest, is_reviewed, snapshot
from src.padic import get_padic_code, encode_path
from scripts.prepare_dataset import write_jsonl
from scripts.summarize_results import summarize

st.set_page_config(page_title='AI Detective', layout='wide')
st.title('AI Detective')
st.caption('Полностью вымышленное досье. Сравнение полного контекста, BM25 и p-adic группировки на одной модели.')

def uploaded_records(file, model):
    return [model.model_validate(json.loads(line)) for line in StringIO(file.getvalue().decode('utf-8-sig')) if line.strip()]

with st.sidebar.expander('Загрузка своего досье'):
    files = [st.file_uploader(label, type=['jsonl'], key=label) for label in ['documents', 'claims', 'questions']]
    if any(files) and not all(files):
        st.info('Выберите все три файла.')
try:
    if all(files):
        documents, claims, questions = [uploaded_records(f, m) for f,m in zip(files, [Document, Claim, Question])]
        validate_dataset(documents, claims, questions)
    else:
        documents, claims, questions = load_dataset()
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.write(validate_dataset(documents, claims, questions))
provider, selected_model, connection_ready = connection_controls()
mode = st.sidebar.radio('Режим', ['A','B','C'], format_func=lambda x: {'A':'A — всё досье','B':'B — BM25 top-15','C':'C — те же top-15 по ветвям'}[x])
question_id = st.selectbox('Вопрос', [q.question_id for q in questions], format_func=lambda key: next(q.question for q in questions if q.question_id == key))
question = next(q for q in questions if q.question_id == question_id)
cases = load_jsonl(ROOT/'data/stress_cases.jsonl', StressCase)
case_options = [c for c in cases if c.question_id == question_id] if not all(files) else []
case_id = st.sidebar.selectbox('Один стресс-фактор', ['baseline'] + [c.case_id for c in case_options])
case = next((c for c in case_options if c.case_id == case_id), None)
if case:
    changed_docs, changed_claims = apply_case(case, documents, claims)
else:
    changed_docs, changed_claims = documents, claims
context = prepare_context(mode, question.question, changed_claims, changed_docs)

with st.expander('Контекст запроса'):
    st.text(context.text)
    st.write('claim_id', context.claim_ids)
    st.write('Порядок документов', context.document_order)
with st.expander('p-adic ветви и коды'):
    st.dataframe(pd.DataFrame([{'claim_id': c.claim_id, 'path': str(get_padic_code(c, changed_docs)),
                                'code': encode_path(get_padic_code(c, changed_docs)), 'event_at': str(c.event_at),
                                'text': c.text} for c in changed_claims if c.claim_id in context.claim_ids]), hide_index=True)
if case:
    st.write('Ожидаемый статус:', case.expected_status)
    st.write(case.rationale)
    with st.expander('Ручная проверка сценария'):
        st.write('Проверьте исходный и изменённый состав, отсутствие утечки удалённой посылки и правильность нового эталона.')
        st.download_button('Исходные данные', json.dumps(snapshot(documents, claims, [question]), ensure_ascii=False, indent=2), 'baseline.json')
        st.download_button('Изменённые данные', json.dumps(snapshot(changed_docs, changed_claims, [question]), ensure_ascii=False, indent=2), case.case_id+'.json')
        st.json(case.model_dump(mode='json'))
        reviewer = st.text_input('Имя проверившего участника')
        confirmed = st.checkbox('Я вручную проверил изменение и ожидаемый ответ')
        if st.button('Сохранить мою проверку', disabled=not (reviewer.strip() and confirmed)):
            case.reviewed_by = reviewer.strip()
            case.reviewed_at = datetime.now(timezone.utc)
            case.review_digest = review_digest(case, documents, claims, question)
            cases = [case if c.case_id == case.case_id else c for c in cases]
            write_jsonl(ROOT/'data/stress_cases.jsonl', [c.model_dump(mode='json') for c in cases])
            st.success('Проверка сохранена. Последующие прогоны будут оцениваться.')
    if not is_reviewed(case, documents, claims, question):
        st.warning('Сценарий не проверен человеком. Ответ сохранится, но не попадёт в метрики.')

if st.button('Запустить модель', type='primary', disabled=not connection_ready):
    with st.spinner('Ожидание ответа модели...'):
        st.session_state['last_run'] = run_one(question, mode, documents, claims, case,
                                              provider=provider, model=selected_model)
record = st.session_state.get('last_run')
if record:
    st.subheader(f"Последний запуск в текущей сессии: {record['question_id']}, {record['mode']}, {record['case_id']}")
    st.caption('Здесь показан один последний ответ. Общая сводка всех сохранённых вопросов находится ниже.')
    if len(record.get('attempts', [])) > 1:
        st.caption('После ошибки формата выполнен один повторный запрос. Токены и время включают обе попытки.')
    if record.get('attempts'):
        with st.expander('Исходные ответы и ошибки проверки формата'):
            st.json(record['attempts'])
    if record['error']:
        st.error(record['error'])
        if record.get('api_metadata'):
            with st.expander('Диагностика API'):
                st.json(record['api_metadata'])
        st.text(record['raw_response'] or '')
    else:
        output = record['output']
        cols = st.columns(4)
        for col, title, value in zip(cols, ['Статус','Confidence','Токены входа','Время, с'],
                                     [output['status'], output['confidence'], record['usage'].get('input_tokens'), round(record['usage'].get('response_seconds',0),2)]):
            col.metric(title, value)
        st.write(output['answer'] or output.get('missing_information'))
        st.write(output['explanation'])
        available = {c['claim_id']: c for c in record['modified']['claims']}
        for label, ids in [('Доказательства', output['evidence_ids']), ('Контрдоказательства', output['counterevidence_ids'])]:
            st.write(label)
            for key in ids:
                st.write(key, available[key]['text'])
                st.caption(available[key]['document_id'] + ': ' + available[key]['evidence'])
        st.json(record['evaluation'] or {'оценка':'Не рассчитана: требуется ручная проверка сценария'})
    st.download_button('Скачать полный результат', json.dumps(record, ensure_ascii=False, indent=2),
                       record.get('filename') or result_filename(record))

st.subheader('Сохранённые результаты')
records = load_records()
if records:
    study = st.selectbox('Серия (единые данные, промпт и параметры)', sorted({r['study_digest'] for r in records}))
    selected = [r for r in records if r['study_digest'] == study]
    kind = st.selectbox('Условия сравнения', sorted({r['kind'] for r in selected}))
    summary = summarize(selected)
    st.write('Метрики доступных запусков')
    st.caption('Все сохранённые вопросы выбранной серии и условия. Выбор вопроса вверху страницы не ограничивает эту сводку. Для каждого эксперимента и режима взята последняя попытка.')
    st.dataframe(pd.DataFrame([r for r in summary['available_metrics'] if r['kind'] == kind]), hide_index=True)
    missing = [r for r in summary['coverage'] if r['kind'] == kind and r['missing_modes']]
    if missing:
        st.info('Для сравнения A/B/C не хватает запусков. Доступные метрики показаны выше; список недостающих режимов — ниже.')
        st.dataframe(pd.DataFrame([{'Вопрос':r['question_id'], 'Сценарий':r['case_id'],
                                   'Есть режимы':', '.join(r['modes']), 'Не хватает':', '.join(r['missing_modes'])}
                                  for r in missing]), hide_index=True)
    with st.expander('Сравнение полных троек A/B/C'):
        compared = [r for r in summary['metrics'] if r['kind'] == kind]
        if compared:
            st.dataframe(pd.DataFrame(compared), hide_index=True)
        else:
            st.info('Нет полной сопоставимой тройки A/B/C для этого условия.')
    st.caption('Ошибки и непроверенные стресс-сценарии не получают оценок качества. Базовые эталоны требуют проверки всей командой.')
    with st.expander('Устойчивость и калибровка confidence'):
        stability = [r for r in summary['available_stability'] if r['stress_type'] == kind]
        if stability:
            st.caption('Устойчивость рассчитана для доступных пар базового и стрессового ответа одного режима.')
            st.dataframe(pd.DataFrame(stability), hide_index=True)
        else:
            st.info('Для устойчивости выберите стресс-условие и выполните базовый и стрессовый запуск в одном режиме.')
        if summary['available_calibration']:
            st.caption('Калибровка confidence на базовых ответах выбранной серии.')
            st.dataframe(pd.DataFrame(summary['available_calibration']), hide_index=True)
        else:
            st.info('Для калибровки пока нет оценённых базовых ответов.')
    st.dataframe(pd.DataFrame([{'question': r['question_id'], 'mode':r['mode'], 'case':r['case_id'], 'status':r['output']['status'] if r['output'] else None,
                               'error':r['error'], 'reviewed':r['reviewed']} for r in selected if r['kind']==kind]), hide_index=True)
    st.download_button('Скачать сводку', json.dumps(summary, ensure_ascii=False, indent=2), 'summary.json')
else:
    st.info('Результатов пока нет. Каждый запуск сохраняется на диск автоматически.')
