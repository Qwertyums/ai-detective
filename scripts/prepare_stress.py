"""Prepare reviewable scenarios. No human approvals are generated here."""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_dataset import write_jsonl
from src.data_validation import ROOT, load_dataset
from src.stress import apply_case, snapshot
from src.schemas import StressCase
from src.retrieval import load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preserve-reviews', action='store_true',
                        help='Keep existing reviews only for exactly unchanged scenario definitions')
    args = parser.parse_args()
    docs, claims, questions = load_dataset()
    cases = []
    removals = {'Q1': 'C21', 'Q2': 'C25', 'Q3': 'C17', 'Q4': 'C27',
                'Q5': 'C06', 'Q6': 'C25', 'Q7': 'C18', 'Q8': 'C19',
                'Q9': 'C40', 'Q10': 'C10', 'Q11': 'C25', 'Q12': 'C30'}
    removal_rationales = {
        'Q7': 'Удалён текст исключения C18. Базовое правило и упоминания специальной нормы не устанавливают её применимость при заданном условии; ожидается insufficient. Проверить отсутствие эквивалентного пересказа.',
        'Q9': 'Удалена реплика C40 с написанием Azef Clan. Из конкретных написаний остаётся Ezaf Clan в C02: ожидается entailed. Проверить, не позволяют ли оставшиеся редакторские упоминания конфликта восстановить вторую версию.',
        'Q10': 'Удалено отрицательное свидетельство C10. Остаётся положительное свидетельство C11: ожидается entailed относительно представленных свидетельств. Проверить, не восстанавливается ли отрицательная позиция из редакторского пересказа конфликта.',
        'Q11': 'Удалена запись команды C25. Идентификация физического пользователя не появляется: insufficient сохраняется. Это контроль сохранения недостатка данных, а не переход из определённого ответа.',
        'Q12': 'Удалено описание февральского исправления C30. Данных о векторе мартовского инцидента по-прежнему нет: insufficient сохраняется. Это контроль сохранения недостатка данных.',
    }
    # Each addition is one incompatible factual/normative assertion. On the
    # insufficient questions a conflict in a premise does not identify a person
    # or establish an exploit: expected status therefore stays insufficient.
    additions = [
        ('В момент приостановки на решающей карте счёт был 9:10.', 'C21'),
        ('Журнал RCON зафиксировал bot_add только с IP 192.168.1.11, не с IP 192.168.1.10.', 'C25'),
        ('Базовый максимальный срок технической паузы составляет 180 минут.', 'C17'),
        ('Независимый журнал контроля доступа зарегистрировал Иванова за своим рабочим местом в момент команды RCON.', 'C27'),
        ('Официальное расследование открыто в 17:07 по московскому времени, не в 17:15.', 'C06'),
        ('Лимит ботов достигнут в 17:03:01 по московскому времени, не в 17:03:15.', 'C16'),
        ('Даже если вредоносный скрипт инициирован не участниками, базовое ограничение длительности паузы применяется.', 'C18'),
        ('При применении исключения вопрос о переносе требуется только после трёх часов паузы, а не после полутора.', 'C19'),
        ('В нашей регистрационной записи соперник указан как Azef Clan, не как Ezaf Clan.', 'C02'),
        ('По нашему независимому наблюдению боты мешали стрельбе.', 'C10'),
        ('Независимая выгрузка сообщает, что команды bot_add с IP 192.168.1.10 не выполнялись.', 'C25'),
        ('По независимому февральскому акту уязвимость спавна ботов в Deathmatch не исправлялась.', 'C30'),
    ]
    for i, q in enumerate(questions):
        common = {'question_id': q.question_id, 'expected_evidence_ids': q.gold_evidence_ids,
                  'expected_counterevidence_ids': q.gold_counterevidence_ids}
        cases.append(StressCase(case_id='P_' + q.question_id, kind='permutation', expected_status=q.gold_status,
                                rationale='Меняется только порядок документов с seed=42; факты и эталон сохраняются.', **common))
        if q.question_id in removals:
            removed = removals[q.question_id]
            cases.append(StressCase(case_id='R_' + q.question_id, kind='remove_premise', remove_claim_id=removed,
                                    expected_status='entailed' if q.question_id in ('Q9', 'Q10') else 'insufficient',
                                    expected_evidence_ids=[x for x in q.gold_evidence_ids if x != removed],
                                    expected_counterevidence_ids=[x for x in q.gold_counterevidence_ids if x != removed],
                                    question_id=q.question_id,
                                    rationale=removal_rationales.get(q.question_id, f'Удалено необходимое основание {removed} вместе с исходной цитатой. Требуется проверить отсутствие эквивалентного пересказа.')))
        text, conflict = additions[i]
        anchor = next(c for c in claims if c.claim_id == conflict)
        source = next(d for d in docs if d.document_id == anchor.document_id)
        new_id = f'C_X{i+1:02}'
        new_doc_id = f'D_X{i+1:02}'
        expected = 'insufficient' if q.gold_status == 'insufficient' else 'both'
        evidence = q.gold_evidence_ids if expected == 'insufficient' else [conflict]
        counter = [] if expected == 'insufficient' else [new_id]
        if q.question_id == 'Q9':
            evidence, counter = q.gold_evidence_ids, q.gold_counterevidence_ids + [new_id]
        elif q.question_id == 'Q10':
            evidence, counter = q.gold_evidence_ids + [new_id], q.gold_counterevidence_ids
        cases.append(StressCase(case_id='X_' + q.question_id, kind='add_contradiction', question_id=q.question_id,
                                added_document={'document_id': new_doc_id, 'title': 'Независимая конфликтующая запись',
                                                'source_type': source.source_type, 'created_at': source.created_at,
                                                'author': 'Независимый источник той же категории',
                                                'text': text + '\nЭто независимый источник той же доказательной силы. Оснований приоритета одной из конфликтующих записей нет.'},
                                added_claim={'claim_id': new_id, 'document_id': new_doc_id, 'text': text, 'evidence': text,
                                             'status': 'disputed', 'event_at': anchor.event_at},
                                expected_status=expected, expected_evidence_ids=evidence,
                                expected_counterevidence_ids=counter,
                                rationale=('Противоречие не устанавливает неизвестную личность или вектор атаки: недостаток данных сохраняется.'
                                           if expected == 'insufficient' else 'Источник той же категории даёт несовместимое основание; приоритет не задан, ожидается both.')))
    if args.preserve_reviews and (ROOT/'data/stress_cases.jsonl').exists():
        old = {c.case_id: c for c in load_jsonl(ROOT/'data/stress_cases.jsonl', StressCase)}
        review_fields = {'reviewed_by', 'reviewed_at', 'review_digest'}
        cases = [old[c.case_id] if c.case_id in old and
                 old[c.case_id].model_dump(exclude=review_fields) == c.model_dump(exclude=review_fields)
                 else c for c in cases]
    write_jsonl(ROOT/'data/stress_cases.jsonl', [c.model_dump(mode='json') for c in cases])
    destination = ROOT/'results/scenarios'
    destination.mkdir(parents=True, exist_ok=True)
    baseline = snapshot(docs, claims, questions)
    (destination/'baseline.json').write_text(__import__('json').dumps(baseline, ensure_ascii=False, indent=2), encoding='utf-8')
    for case in cases:
        changed_docs, changed_claims = apply_case(case, docs, claims)
        payload = {'case': case.model_dump(mode='json'), 'modified': snapshot(changed_docs, changed_claims, questions)}
        (destination/(case.case_id+'.json')).write_text(__import__('json').dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {len(cases)} cases (new or changed definitions require human review); snapshots: {destination}')


if __name__ == '__main__':
    main()
