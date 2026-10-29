"""Human review only: inspect scenarios first, then record your own approval."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_dataset import write_jsonl
from src.data_validation import ROOT, load_dataset
from src.retrieval import load_jsonl
from src.schemas import StressCase
from src.stress import review_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, help='A case ID or all')
    parser.add_argument('--reviewer', required=True, help='Name of the person who actually checked the scenario')
    parser.add_argument('--approve', action='store_true', help='Attest that you manually checked the scenario and expected answer')
    args = parser.parse_args()
    docs, claims, questions = load_dataset()
    cases = load_jsonl(ROOT/'data/stress_cases.jsonl', StressCase)
    chosen = [c for c in cases if args.case in (c.case_id, 'all')]
    if not chosen:
        parser.error('Unknown case ID')
    for case in chosen:
        print(case.model_dump_json(indent=2))
        if args.approve:
            q = next(q for q in questions if q.question_id == case.question_id)
            case.reviewed_by = args.reviewer.strip()
            if not case.reviewed_by:
                parser.error('Reviewer cannot be empty')
            case.reviewed_at = datetime.now(timezone.utc)
            case.review_digest = review_digest(case, docs, claims, q)
    if args.approve:
        write_jsonl(ROOT/'data/stress_cases.jsonl', [c.model_dump(mode='json') for c in cases])


if __name__ == '__main__':
    main()
