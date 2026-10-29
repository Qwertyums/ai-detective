"""Build a readable checklist from the actual prepared scenarios."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_validation import ROOT, load_dataset
from src.retrieval import load_jsonl
from src.schemas import StressCase


def main():
    _, claims, questions = load_dataset()
    cases = load_jsonl(ROOT/'data/stress_cases.jsonl', StressCase)
    lines = ['# Ручная проверка стресс-сценариев', '',
             'Сценарии подготовлены ИИ. Для каждого проверьте вопрос, точную мутацию, ожидаемый статус и отсутствие оставшегося эквивалентного основания. Полные копии находятся в results/scenarios. Этот файл не является подписью участника.', '',
             'Перестановки P_Q1–P_Q12: тот же состав документов и утверждений, меняется только порядок с seed=42. Эталон сохраняется. В B/C детерминированный retrieval может полностью сохранить текст.', '']
    for case in cases:
        if case.kind == 'permutation':
            continue
        q = next(q for q in questions if q.question_id == case.question_id)
        lines += ['## '+case.case_id, '', '**Вопрос:** '+q.question, '', '**Исходный эталон:** '+q.gold_status+' — '+q.gold_answer, '']
        if case.remove_claim_id:
            target = next(c for c in claims if c.claim_id == case.remove_claim_id)
            lines += ['**Удаляется:** '+target.claim_id+' из '+target.document_id+': «'+target.evidence+'»', '']
        else:
            lines += ['**Добавляется:** '+case.added_claim.claim_id+' из нового '+case.added_document.document_id+': «'+case.added_claim.evidence+'»', '']
        lines += ['**Новый эталон:** '+case.expected_status+'. '+case.rationale, '',
                  'Evidence: '+', '.join(case.expected_evidence_ids)+'. Counterevidence: '+', '.join(case.expected_counterevidence_ids)+'.', '']
    lines += ['## Подтверждение', '', 'После фактической проверки сохраните подтверждение в приложении либо scripts/review_cases.py. Укажите действительное имя. Для всей команды отдельно требуется проверка исходных эталонов. Изменение содержания аннулирует прежнее подтверждение.']
    (ROOT/'data/STRESS_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
