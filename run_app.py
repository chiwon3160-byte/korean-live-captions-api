"""Visible startup errors even when launched using pythonw / hidden launcher."""
from pathlib import Path
import ctypes
import sys
import traceback

try:
    from app import main
    raise SystemExit(main())
except SystemExit:
    raise
except Exception as exc:
    root = Path(__file__).resolve().parent
    (root / '_local').mkdir(exist_ok=True)
    (root / '_local' / 'startup-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    message = ('자막창을 시작하지 못했습니다.\n\n' + str(exc) +
               '\n\n오류 기록: ' + str(root / '_local' / 'startup-error.txt'))
    if hasattr(ctypes, 'windll'):
        ctypes.windll.user32.MessageBoxW(None, message, '한국어 자막 시작 오류', 0x10)
    else:
        print(message, file=sys.stderr)  # macOS: the .command launcher keeps Terminal open.
