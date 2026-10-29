"""Print dossier documents as readable text without extra dependencies."""
import argparse
import json
from pathlib import Path
import sys

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / 'data' / 'documents.jsonl'


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Вывести тексты документов досье.')
    parser.add_argument('--file', type=Path, default=DEFAULT_INPUT,
                        help='Путь к documents.jsonl (по умолчанию файл проекта).')
    parser.add_argument('--id', nargs='+', dest='document_ids',
                        help='Вывести только указанные документы, например D01 D03.')
    parser.add_argument('--output', type=Path,
                        help='Сохранить текст в UTF-8 вместо вывода в консоль.')
    args = parser.parse_args()

    try:
        documents = []
        with args.file.open(encoding='utf-8-sig') as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Строка {line_number}: некорректный JSON ({exc.msg}).') from exc
                if not isinstance(document, dict) or any(
                    not isinstance(document.get(key), str)
                    for key in ('document_id', 'title', 'text')
                ):
                    raise ValueError(f'Строка {line_number}: нужны строковые поля document_id, title и text.')
                documents.append(document)

        if args.document_ids:
            missing = set(args.document_ids) - {d['document_id'] for d in documents}
            if missing:
                raise ValueError('Документы не найдены: ' + ', '.join(sorted(missing)))
            documents = [d for d in documents if d['document_id'] in args.document_ids]

        blocks = [f"{'=' * 72}\n{d['document_id']} — {d['title']}\n{'=' * 72}\n\n{d['text']}"
                  for d in documents]
        text = '\n\n'.join(blocks) + ('\n' if blocks else '')
        if args.output:
            args.output.write_text(text, encoding='utf-8')
        else:
            sys.stdout.write(text)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f'Ошибка: {exc}\n')


if __name__ == '__main__':
    main()
