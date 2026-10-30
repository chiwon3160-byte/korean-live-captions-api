"""Global reset/stop keys only; does not log or intercept typed text."""
import ctypes
from ctypes import wintypes
from PySide6.QtCore import QAbstractNativeEventFilter


class GlobalHotkeys(QAbstractNativeEventFilter):
    def __init__(self, app, reset, stop):
        super().__init__()
        self.app = app
        self.actions = {0x4A01: reset, 0x4A02: stop}
        self.registered = []

    def register(self):
        for ident, key in [(0x4A01, ord('R')), (0x4A02, ord('S'))]:
            # MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT
            if ctypes.windll.user32.RegisterHotKey(None, ident, 0x4006, key):
                self.registered.append(ident)
            else:
                self.unregister()
                return False
        self.app.installNativeEventFilter(self)
        return True

    def unregister(self):
        for ident in self.registered:
            ctypes.windll.user32.UnregisterHotKey(None, ident)
        self.registered.clear()
        self.app.removeNativeEventFilter(self)

    def nativeEventFilter(self, event_type, message):
        msg = wintypes.MSG.from_address(int(message))
        if msg.message == 0x0312 and msg.wParam in self.actions:
            self.actions[msg.wParam]()
            return True, 0
        return False, 0
