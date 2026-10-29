"""python scripts/run_experiments.py --suite all [--exploratory]"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_validation import ROOT, load_dataset
from src.experiments import load_records, run_one
from src.retrieval import load_jsonl
from src.schemas import StressCase
from src.stress import is_reviewed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['baseline', 'all'], default='baseline')
    parser.add_argument('--exploratory', action='store_true', help='Run unreviewed stress cases without scoring')
    parser.add_argument('--question', help='One question, e.g. Q1')
    parser.add_argument('--mode', choices=['A','B','C'])
    parser.add_argument('--resume', action='store_true', help='Skip already saved attempts for this exact experiment')
    parser.add_argument('--provider', choices=['chatgpt','api_key'], help='Default: OPENAI_AUTH_MODE or chatgpt')
    parser.add_argument('--model', help='Model available to the selected account')
    args = parser.parse_args()
    from src.llm import create_client, LLMError, configuration
    from src.chatgpt_auth import AuthError
    try:
        requested_model, settings = configuration(args.provider, args.model)
        if settings['provider'] == 'chatgpt':
            from src.chatgpt_inference import list_models
            if requested_model not in {m['slug'] for m in list_models()}:
                parser.exit(1, 'Модель недоступна. Посмотрите python scripts/chatgpt_login.py models и укажите --model.\n')
        else:
            create_client().close()
    except (LLMError, AuthError) as exc:
        parser.exit(1, str(exc) + '\n')
    docs, claims, questions = load_dataset()
    cases = load_jsonl(ROOT/'data/stress_cases.jsonl', StressCase) if args.suite == 'all' else []
    old = load_records() if args.resume else []
    from src.stress import digest, snapshot
    from src.llm import system_prompt
    for q in questions:
        if args.question and q.question_id != args.question:
            continue
        variants = [None] + [case for case in cases if case.question_id == q.question_id]
        for case in variants:
            if case and not args.exploratory and not is_reviewed(case, docs, claims, q):
                print('SKIP unreviewed', case.case_id, flush=True)
                continue
            fingerprint = digest({'original': snapshot(docs, claims, [q]), 'case': case.model_dump(mode='json') if case else None,
                                  'model': requested_model, 'settings': settings, 'prompt': system_prompt()})
            for mode in ([args.mode] if args.mode else ['A','B','C']):
                if any(r['experiment_digest'] == fingerprint and r['mode'] == mode for r in old):
                    continue
                print('RUN', q.question_id, mode, case.case_id if case else 'baseline', flush=True)
                result = run_one(q, mode, docs, claims, case, provider=args.provider, model=requested_model)
                print('ERROR '+result['error'] if result['error'] else 'OK '+result['output']['status'], result['usage'], flush=True)
                if result['error'] and result['raw_response'] is None:
                    raise SystemExit('Model/environment failure; stopping to avoid repeated invalid runs.')


if __name__ == '__main__':
    main()
