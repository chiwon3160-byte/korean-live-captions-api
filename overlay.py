"""Movable, local-only subtitles. Never stores caption content."""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame


class SubtitleOverlay(QWidget):
    def __init__(self, settings, parent=None):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.settings = settings
        self.locked = False
        self._drag = None
        self._full_current = "번역을 시작하면 여기에 자막이 표시됩니다."
        self._full_previous = ""
        self._pages = [self._full_current]
        self._page_index = 0
        self._page_timer = QTimer(self)
        self._page_timer.timeout.connect(self._advance_page)
        self.setWindowTitle("영상 위 한국어 자막")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.panel = QFrame()
        self.panel.setObjectName("subtitlePanel")
        outer.addWidget(self.panel)
        box = QVBoxLayout(self.panel)
        box.setContentsMargins(16, 9, 16, 14)
        box.setSpacing(8)
        self.handle = QLabel("여기를 잡고 이동 · 고정은 전체 자막창에서")
        self.handle.setCursor(Qt.CursorShape.SizeAllCursor)
        self.handle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        box.addWidget(self.handle)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        flow = QVBoxLayout(self.content)
        flow.setContentsMargins(0, 0, 0, 0)
        flow.setSpacing(10)
        self.previous = QLabel()
        self.current = QLabel("번역을 시작하면 여기에 자막이 표시됩니다.")
        for label in (self.previous, self.current):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            flow.addWidget(label)
        self.scroll.setWidget(self.content)
        box.addWidget(self.scroll)
        self.page_hint = QLabel()
        self.page_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_hint.setStyleSheet("color: #a8c5b8; font-size: 11px;")
        self.page_hint.hide()
        box.addWidget(self.page_hint)
        self.setStyleSheet("""
            QWidget { background: transparent; color: white; font-family: 'Malgun Gothic'; }
            QFrame#subtitlePanel { background: rgba(12,16,23,225); border: 1px solid #60726f; border-radius: 12px; }
            QLabel { border: 0; }
            QScrollBar:vertical { width: 6px; background: #26322f; }
            QScrollBar::handle:vertical { background: #8dab9e; min-height: 20px; }
        """)
        self.handle.setStyleSheet("color: #a8c5b8; font-size: 11px;")
        self.previous.setStyleSheet("color: #bac6cf; font-size: 16px;")
        self.current.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: 600;")
        self.previous.hide()
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(470, screen.width() - 32), 150)
        self.move(screen.x() + round(screen.width() * .307), screen.y() + round(screen.height() * .70))
        saved = settings.value("overlay_geometry")
        if saved:
            self.restoreGeometry(saved)
        self._fit()

    def set_captions(self, captions):
        items = list(captions)[-2:]
        current = items[-1]["korean"] if items else "번역을 시작하면 여기에 자막이 표시됩니다."
        if current != self._full_current:
            self._page_index = 0
            self._page_timer.stop()
        self._full_current = current
        self._full_previous = items[-2]["korean"] if len(items) > 1 else ""
        self._fit()

    def _fit(self):
        screen = self.screen().availableGeometry()
        width = max(280, min(self.width(), screen.width() - 24))
        self.setFixedWidth(width)
        inner_width = width - 42
        # Keep every displayed page readable even while mouse input passes through.
        # Long current text takes priority over history; never hide it below a scroll.
        budget = max(80, min(360, screen.height() - 140))
        self.current.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: 600;")
        self.current.ensurePolished()
        self.current.setText(self._full_current)
        if self.current.heightForWidth(inner_width) > budget:
            self.current.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: 600;")
            self.current.ensurePolished()
        remaining = self._full_current
        pages = []
        while remaining:
            low, high = 1, len(remaining)
            while low < high:
                mid = (low + high + 1) // 2
                self.current.setText(remaining[:mid])
                if self.current.heightForWidth(inner_width) <= budget:
                    low = mid
                else:
                    high = mid - 1
            # Prefer a word boundary but retain every original character.
            end = low
            if end < len(remaining):
                boundary = remaining.rfind(" ", max(1, end // 2), end)
                if boundary > 0:
                    end = boundary + 1
            pages.append(remaining[:end])
            remaining = remaining[end:]
        self._pages = pages or [""]
        self._page_index = min(self._page_index, len(self._pages)-1)
        self.current.setText(self._pages[self._page_index])
        height = self.current.heightForWidth(inner_width)
        self.previous.setText(self._full_previous)
        previous_height = self.previous.heightForWidth(inner_width) if self._full_previous else 0
        show_previous = bool(self._full_previous) and len(self._pages) == 1 and height + previous_height + 10 <= budget
        self.previous.setVisible(show_previous)
        if show_previous:
            height += previous_height + 10
        paged = len(self._pages) > 1
        self.page_hint.setVisible(paged)
        self.page_hint.setText(f"긴 자막 {self._page_index+1}/{len(self._pages)} · 자동 넘김 · 전체 내용은 기록창에서")
        self.setFixedHeight(max(100, height + (45 if self.locked else 72) + (25 if paged else 0)))
        if paged and not self._page_timer.isActive():
            self._page_timer.start(max(4000, min(16000, len(self.current.text()) * 95)))
        elif not paged:
            self._page_timer.stop()
        self.scroll.verticalScrollBar().setValue(0)
        self.move(max(screen.left(), min(self.x(), screen.right()-self.width()+1)),
                  max(screen.top(), min(self.y(), screen.bottom()-self.height()+1)))

    def _advance_page(self):
        self._page_index = (self._page_index + 1) % len(self._pages)
        self._page_timer.stop()
        self._fit()

    def set_locked(self, locked):
        visible = self.isVisible()
        self.locked = bool(locked)
        self.handle.setVisible(not self.locked)
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self.locked)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, self.locked)
        self._fit()
        if visible:
            self.show()

    def mousePressEvent(self, event):
        if not self.locked and event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self.move(event.globalPosition().toPoint() - self._drag)
            event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._drag = None
            self._fit()
            self.save_position()

    def save_position(self):
        self.settings.setValue("overlay_geometry", self.saveGeometry())

