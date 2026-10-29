"""Print dossier claims as readable text without extra dependencies."""
import argparse
import json
from pathlib import Path
import sys

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / 'data' / 'claims.jsonl'


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Вывести утверждения досье.')
    parser.add_argument('--file', type=Path, default=DEFAULT_INPUT,
                        help='Путь к claims.jsonl (по умолчанию файл проекта).')
    parser.add_argument('--id', nargs='+', dest='claim_ids',
                        help='Вывести выбранные утверждения, например C01 C03.')
    parser.add_argument('--document', nargs='+', dest='document_ids',
                        help='Вывести утверждения из документов, например D01 D03.')
    parser.add_argument('--output', type=Path,
                        help='Сохранить текст в UTF-8 вместо вывода в консоль.')
    args = parser.parse_args()

    try:
        claims = []
        with args.file.open(encoding='utf-8-sig') as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    claim = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Строка {line_number}: некорректный JSON ({exc.msg}).') from exc
                required = ('claim_id', 'text', 'document_id', 'evidence', 'status')
                if not isinstance(claim, dict) or any(not isinstance(claim.get(key), str) for key in required):
                    raise ValueError(f'Строка {line_number}: нужны строковые поля {", ".join(required)}.')
                claims.append(claim)

        for requested, key, label in [
            (args.claim_ids, 'claim_id', 'Утверждения'),
            (args.document_ids, 'document_id', 'Документы'),
        ]:
            if requested:
                missing = set(requested) - {c[key] for c in claims}
                if missing:
                    raise ValueError(label + ' не найдены: ' + ', '.join(sorted(missing)))
        claims = [c for c in claims
                  if (not args.claim_ids or c['claim_id'] in args.claim_ids)
                  and (not args.document_ids or c['document_id'] in args.document_ids)]
        blocks = []
        for claim in claims:
            blocks.append(
                f"{'=' * 72}\n{claim['claim_id']} | Документ: {claim['document_id']} | Статус: {claim['status']}\n"
                f"{'=' * 72}\n\n{claim['text']}\n\n"
                f"Цитата: {claim['evidence']}\nВремя события: {claim.get('event_at') or 'не указано'}"
            )
        text = '\n\n'.join(blocks) + ('\n' if blocks else '')
        if args.output:
            args.output.write_text(text, encoding='utf-8')
        else:
            sys.stdout.write(text)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f'Ошибка: {exc}\n')


if __name__ == '__main__':
    main()
