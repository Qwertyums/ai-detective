"""Manage a local ChatGPT plan connection without displaying credentials."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chatgpt_auth import ChatGPTAuth, AuthError


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['login','status','models','select','logout'])
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--profile', help='Saved profile_id from status')
    choice.add_argument('--new', action='store_true', help='Add another ChatGPT account')
    parser.add_argument('--consent', action='store_true', help='Explicitly request plan permission again')
    args = parser.parse_args()
    if args.new and args.action != 'login':
        parser.error('--new is only available for login')
    auth = ChatGPTAuth()
    flow = None
    try:
        if args.action == 'status':
            print(json.dumps(auth.profiles(),ensure_ascii=False,indent=2))
        elif args.action == 'select':
            if not args.profile:
                parser.error('--profile is required')
            auth.select(args.profile)
            print('Аккаунт выбран.')
        elif args.action == 'models':
            from src.chatgpt_inference import list_models
            print(json.dumps(list_models(auth,refresh=True),ensure_ascii=False,indent=2))
        elif args.action == 'logout':
            confirmed = auth.logout(args.profile)
            print('Выход выполнен.' if confirmed else 'Локальные токены удалены; отзыв на сервере не подтверждён. Отключите приложение в настройках ChatGPT.')
        else:
            profiles = auth.profiles()
            profile = None if args.new else args.profile or profiles['active']
            flow = auth.begin(profile,consent=args.consent)
            print('Откройте ссылку в системном браузере и разрешите Use your ChatGPT plan. Ссылка действует 10 минут:',flush=True)
            print(flow.url,flush=True)
            while time.monotonic() < flow.deadline and not flow.received.wait(timeout=.5):
                pass
            result = flow.finish()
            print('Аккаунт:',result['label'])
            print('Использование тарифа разрешено.' if result['sharing'] else 'Разрешение на использование тарифа не предоставлено. Повторите login --consent.')
    except AuthError as exc:
        parser.exit(1,str(exc)+'\n')
    except KeyboardInterrupt:
        parser.exit(130,'Вход отменён.\n')
    finally:
        if flow:
            flow.close()


if __name__ == '__main__':
    main()
