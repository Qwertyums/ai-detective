"""Rename legacy UUID result files without changing their JSON contents."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_validation import ROOT
from src.experiments import result_filename


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Show proposed names without renaming')
    args = parser.parse_args()
    folder = (ROOT/'results/runs').resolve()
    count = 0
    for old in sorted(folder.glob('*.json')):
        if not re.fullmatch(r'[0-9a-f]{32}\.json', old.name) or old.is_symlink():
            continue
        try:
            record = json.loads(old.read_text(encoding='utf-8'))
            new = folder / result_filename(record)
            if new.exists():
                new = folder / result_filename(record, full_id=True)
            if old.resolve().parent != folder or new.resolve().parent != folder:
                raise ValueError('Unexpected result path')
            if new.exists():
                print('Пропущен: имя уже занято — '+old.name)
                continue
            if not args.dry_run:
                old.rename(new)
            print(old.name+' → '+new.name)
            count += 1
        except (OSError, ValueError, KeyError, TypeError):
            print('Не удалось переименовать: '+old.name)
    print(('Будет переименовано' if args.dry_run else 'Переименовано')+f': {count}')


if __name__ == '__main__':
    main()
