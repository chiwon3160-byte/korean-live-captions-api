"""Small, local-only Korean caption window. Runtime work belongs to app.py."""

from __future__ import annotations

from collections import deque
from overlay import SubtitleOverlay
import math
import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QCloseEvent, QFont, QFontDatabase, QKeySequence, QShortcut, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QVBoxLayout, QWidget, QLayout, QDialog, QLineEdit, QApplication,
)


def _label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(wrap)
    if name:
        label.setObjectName(name)
    return label


class ModelSelector(QComboBox):
    textChanged = Signal(str)

    def __init__(self):
        super().__init__()
        self.setEditable(False)
        self.setMinimumWidth(0)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(18)
        self.currentTextChanged.connect(self.textChanged.emit)

    def text(self):
        return self.currentData() or ''

    def setText(self, text):
        self.setCurrentIndex(self.findData(text))


class CaptionWindow(QMainWindow):
    start_requested = Signal(str, str)
    stop_requested = Signal()
    reset_requested = Signal()
    closing = Signal()
    refresh_requested = Signal()
    comment_setup_requested = Signal()
    translation_settings_requested = Signal(dict)

    def __init__(self, settings=None) -> None:
        super().__init__()
        # Qt's offscreen platform may not enumerate Windows fonts. Load the OS
        # font explicitly only in that case; native Windows uses its own catalog.
        if "Malgun Gothic" not in QFontDatabase.families():
            font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
            if font_path.is_file():
                QFontDatabase.addApplicationFont(str(font_path))
        self.setWindowTitle("한국어 실시간 자막")
        self.setMinimumSize(480, 540)
        self.resize(580, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self._settings = settings if settings is not None else QSettings("JamesTools", "LocalKoreanCaptions")
        if self._int_setting("layout_version", 1, 1, 2) < 2:
            if self._int_setting("font_size", 25, 14, 32) == 25:
                self._settings.setValue("font_size", 20)
            self._settings.setValue("layout_version", 2)
        self.overlay = SubtitleOverlay(self._settings, self)
        self._captions: deque[dict] = deque(maxlen=20)
        self._running = False
        self._sources: list[dict] = []
        self._saved_source = str(self._settings.value("source", ""))
        self._status_text = "소리 장치를 선택한 뒤 시작해 주세요."
        self._comment_setup_dialog = None
        self._translation_mode = "local"
        self._translation_description = "무료 · PC 안에서 처리"

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        title = _label("한국어 자막", "title")
        heading.addWidget(title)
        heading.addStretch()
        self.pin_check = QCheckBox("항상 위")
        self.pin_check.setChecked(self._bool_setting("always_on_top", True))
        self.pin_check.setToolTip("다른 프로그램 위에 자막창을 유지합니다.")
        heading.addWidget(self.pin_check)
        layout.addLayout(heading)
        self.selected_source_label = _label("소리 장치 확인 중…", "localNote", True)
        layout.addWidget(self.selected_source_label)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self.source_combo = QComboBox()
        self.source_combo.setAccessibleName("번역할 소리 장치")
        self.source_combo.setMinimumWidth(0)
        self.source_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.source_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source_combo.setMinimumContentsLength(18)
        self.refresh_button = QPushButton("새로고침")
        self.refresh_button.setToolTip("소리 장치를 다시 찾습니다. 장치를 바꾼 뒤 눌러 주세요.")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        source_row.addWidget(self.source_combo, 1)
        source_row.addWidget(self.refresh_button)
        self.settings_panel = QWidget()
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addLayout(source_row)
        from api_engine import PROVIDERS
        self.provider_combo = QComboBox()
        self.provider_combo.setAccessibleName("번역 방식")
        self.provider_combo.addItem("무료 로컬 · PC 안에서 처리", "local")
        for key, spec in PROVIDERS.items():
            self.provider_combo.addItem(spec["label"] + " · 유료 API", key)
        self.model_field = ModelSelector()
        self.model_label = _label("번역 모델 · 아래 메뉴를 눌러 선택하세요 ▼", "scope")
        self.model_field.setAccessibleName("API 모델")
        self.model_field.setPlaceholderText("모델 이름")
        self.reasoning_combo = QComboBox()
        self.reasoning_combo.setAccessibleName("OpenAI 추론 설정")
        self.reasoning_combo.addItem("추론 none · 빠른 응답 우선", "none")
        self.reasoning_combo.addItem("추론 low · 비교 사용", "low")
        self.model_field.textChanged.connect(self._model_changed)
        self.api_key_field = QLineEdit()
        self.api_key_field.setAccessibleName("API 키 · 이번 실행에서만 사용")
        self.api_key_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_field.setPlaceholderText("API 키 · 비우면 환경변수 또는 .env.local 사용")
        self.api_comment_check = QCheckBox("댓글도 유료 API로 전송 · 음성 중지 중에도 과금")
        self.api_comment_check.setChecked(False)
        self.apply_translation_button = QPushButton("번역 설정 적용")
        self.translation_note = _label("현재: 무료 로컬 · PC 안에서 처리", "scope", True)
        self.api_notice = _label("API를 적용하면 음성 원문과 최근 두 구간 문맥을 선택한 제공사에 보내며 비용이 발생합니다. 음성 파일은 보내지 않습니다. 키는 이번 실행에서만 사용합니다. 댓글 전송은 별도로 켜야 합니다.", "scope", True)
        self.api_notice.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        for widget in (self.provider_combo, self.model_label, self.model_field, self.reasoning_combo, self.api_key_field,
                       self.api_comment_check, self.api_notice,
                       self.apply_translation_button):
            settings_layout.addWidget(widget)
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self.apply_translation_button.clicked.connect(self._apply_translation_settings)
        self._provider_changed()

        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("소리·표시 설정")
        self.settings_dialog.resize(560, 600)
        self.settings_dialog.setMinimumSize(440, 360)
        dialog_layout = QVBoxLayout(self.settings_dialog)
        dialog_layout.addWidget(_label("번역 중에는 모델·소리 장치를 바꿀 수 없어요. 먼저 자막창의 ‘중지’를 눌러 주세요.", "scope", True))
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setWidget(self.settings_panel)
        dialog_layout.addWidget(settings_scroll)
        close_settings = QPushButton("닫기")
        close_settings.clicked.connect(self.settings_dialog.accept)
        dialog_layout.addWidget(close_settings)
        self.settings_button = QPushButton("소리·표시 설정")
        self.settings_button.setCheckable(True)
        self.settings_button.toggled.connect(self.settings_dialog.setVisible)
        self.settings_dialog.finished.connect(lambda _: self.settings_button.setChecked(False))
        utility_row = QHBoxLayout()
        utility_row.addWidget(self.settings_button, 1)
        self.comment_setup_button = QPushButton("틱톡 댓글 번역")
        self.comment_setup_button.setToolTip("Chrome에서 댓글 아래에 한국어 번역을 표시하도록 연결합니다.")
        self.comment_setup_button.clicked.connect(self.comment_setup_requested.emit)
        utility_row.addWidget(self.comment_setup_button)
        layout.addLayout(utility_row)
        layout.addWidget(self.translation_note)
        self.usage_label = _label("이번 실행 API 예상 비용: $0.000000 · 사용 전", "scope", True)
        self.usage_label.setAccessibleName("이번 실행 API 누적 예상 비용")
        layout.addWidget(self.usage_label)
        self.scope_label = _label("장치 확인 중…", "scope", True)
        layout.addWidget(self.scope_label)

        actions = QHBoxLayout()
        self.language_combo = QComboBox()
        self.language_combo.setAccessibleName("영상의 원래 언어")
        for name, code in (("언어 자동", "auto"), ("영어", "en"), ("스페인어", "es")):
            self.language_combo.addItem(name, code)
        previous_language = str(self._settings.value("language", "auto"))
        language_index = self.language_combo.findData(previous_language)
        self.language_combo.setCurrentIndex(max(0, language_index))
        self.start_button = QPushButton("번역 시작")
        self.start_button.setObjectName("startButton")
        self.start_button.setEnabled(False)
        self.stop_button = QPushButton("중지")
        self.stop_button.setEnabled(False)
        self.reset_button = QPushButton("새 영상")
        self.reset_button.setToolTip("이전 영상의 자막과 대기 결과를 비웁니다. 창 안 단축키: Ctrl+R")
        actions.addWidget(self.language_combo)
        actions.addStretch(1)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.reset_button)
        layout.addLayout(actions)

        meter_row = QHBoxLayout()
        meter_row.addWidget(_label("입력 소리", "muted"))
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(6)
        self.level_bar.setAccessibleName("수신 소리 크기")
        meter_row.addWidget(self.level_bar, 1)
        layout.addLayout(meter_row)
        self.status_label = _label(self._status_text, "status", True)
        layout.addWidget(self.status_label)

        self.scroll = QScrollArea()
        self.scroll.setMinimumHeight(120)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_holder = QWidget()
        self.caption_layout = QVBoxLayout(self.caption_holder)
        self.caption_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.caption_layout.setContentsMargins(0, 0, 2, 0)
        self.caption_layout.setSpacing(10)
        self.scroll.setWidget(self.caption_holder)
        self.scroll.verticalScrollBar().rangeChanged.connect(lambda *_: QTimer.singleShot(0, self._center_latest))
        layout.addWidget(self.scroll, 1)

        options = QHBoxLayout()
        self.original_check = QCheckBox("원문 보기")
        self.original_check.setChecked(self._bool_setting("show_original", True))
        settings_layout.addWidget(self.original_check)
        options.addWidget(_label("글자", "muted"))
        self.font_spin = QSpinBox()
        self.font_spin.setRange(14, 28)
        self.font_spin.setSuffix(" px")
        self.font_spin.setValue(self._int_setting("font_size", 20, 14, 28))
        self.font_spin.setAccessibleName("한국어 자막 글자 크기")
        options.addWidget(self.font_spin)
        options.addWidget(_label("불투명도", "muted"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(60, 100)
        self.opacity_slider.setValue(self._int_setting("opacity", 100, 60, 100))
        self.opacity_slider.setFixedWidth(70)
        self.opacity_slider.setAccessibleName("창 불투명도")
        options.addWidget(self.opacity_slider)
        options.addStretch()
        settings_layout.addLayout(options)
        overlay_row = QHBoxLayout()
        self.overlay_check = QCheckBox("영상 위 자막")
        self.overlay_check.setChecked(self._bool_setting("overlay_visible", True))
        self.overlay_lock = QCheckBox("위치 고정 · 클릭 통과")
        self.overlay_lock.setChecked(self._bool_setting("overlay_locked", False))
        self.overlay_check.toggled.connect(self.overlay.setVisible)
        self.overlay_lock.toggled.connect(self.overlay.set_locked)
        overlay_row.addWidget(self.overlay_check)
        overlay_row.addStretch()
        overlay_row.addWidget(self.overlay_lock)
        layout.addLayout(overlay_row)
        self.overlay.set_locked(self.overlay_lock.isChecked())
        self.overlay.setVisible(self.overlay_check.isChecked())
        footer = _label("새 영상으로 넘겼다면 ‘새 영상’ · 종료하면 자막 기록은 사라집니다.", "footer", True)
        layout.addWidget(footer)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.start_button.clicked.connect(self._request_start)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.reset_button.clicked.connect(self._request_reset)
        self.original_check.toggled.connect(self._redraw_captions)
        self.font_spin.valueChanged.connect(self._redraw_captions)
        self.pin_check.toggled.connect(self._apply_pin)
        self.opacity_slider.valueChanged.connect(lambda value: self.setWindowOpacity(value / 100.0))
        self._reset_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self._reset_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._reset_shortcut.activated.connect(self._request_reset)
        self._stop_shortcut = QShortcut(QKeySequence("Escape"), self)
        self._stop_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._stop_shortcut.activated.connect(self._request_stop)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #171b22; color: #e9eef5;
                font-family: 'Malgun Gothic', 'Segoe UI'; font-size: 12px; }
            QLabel#title { font-size: 21px; font-weight: 700; }
            QLabel#localNote { color: #84d8b7; font-size: 11px; }
            QLabel#scope { color: #b5c0d1; font-size: 11px; }
            QLabel#muted, QLabel#footer { color: #94a0b3; font-size: 11px; }
            QLabel#status { color: #c1cad8; padding: 3px 0; }
            QLabel#status[error="true"] { color: #ffb4ab; }
            QPushButton { background: #2b3340; border: 1px solid #414d60;
                border-radius: 6px; padding: 7px 11px; font-weight: 600; }
            QPushButton:hover { background: #354052; }
            QPushButton:pressed { background: #202835; }
            QPushButton:disabled { color: #738097; background: #222933; border-color: #303947; }
            QPushButton#startButton { background: #9ce7c5; color: #102c22; border-color: #9ce7c5; }
            QPushButton#startButton:hover { background: #b9f1d8; }
            QPushButton#startButton:disabled { background: #29463d; color: #8aada0; border-color: #29463d; }
            QComboBox, QSpinBox, QLineEdit { background: #222a36; border: 1px solid #414d60;
                border-radius: 5px; padding: 6px; min-height: 18px; }
            QComboBox::drop-down { border: 0; width: 22px; }
            QComboBox QAbstractItemView { background: #222a36; selection-background-color: #385247; }
            QCheckBox { spacing: 6px; }
            QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #65758c; border-radius: 3px; }
            QCheckBox::indicator:checked { background: #88d9b6; border-color: #88d9b6; }
            QProgressBar { background: #303b4c; border: none; border-radius: 3px; }
            QProgressBar::chunk { background: #88d9b6; border-radius: 3px; }
            QFrame#captionCard { background: #202733; border: 1px solid #354052; border-radius: 8px; }
            QFrame#latestCard { background: #222d38; border: 1px solid #668b7d; border-radius: 8px; }
            QFrame#captionCard QLabel, QFrame#latestCard QLabel { background: transparent; border: none; }
            QLabel#captionMeta { color: #a4b4c8; font-size: 11px; }
            QLabel#originalText { color: #adbacd; font-size: 13px; }
            QLabel#emptyTitle { color: #dae4ee; font-size: 20px; }
            QLabel#emptyBody { color: #a0afc2; font-size: 13px; }
            QScrollBar:vertical { background: #171b22; width: 8px; margin: 0; }
            QScrollBar::handle:vertical { background: #46566b; border-radius: 4px; min-height: 25px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QSlider::groove:horizontal { height: 4px; background: #3d495b; border-radius: 2px; }
            QSlider::handle:horizontal { background: #a6b9ce; width: 12px; margin: -4px 0; border-radius: 6px; }
        """)
        self._apply_pin(self.pin_check.isChecked())
        self.setWindowOpacity(self.opacity_slider.value() / 100.0)
        self._redraw_captions()
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            screen = self.screen().availableGeometry()
            self.move(screen.right() - self.width() - 24, screen.top() + 50)
        self._keep_on_screen()

    def _provider_changed(self):
        from api_engine import PROVIDERS
        provider = self.provider_combo.currentData()
        remote = provider != "local"
        self.model_field.clear()
        if provider == "openai":
            for label, model in (("Luna 6 · 기본 / 저렴한 비용", "gpt-6-luna"),
                                 ("Terra 6.1", "gpt-6.1-terra"),
                                 ("SOL 6.1", "gpt-6.1-sol")):
                self.model_field.addItem(label + "  ▾", model)
        elif remote:
            self.model_field.addItem(PROVIDERS[provider]["default_model"] + "  ▾", PROVIDERS[provider]["default_model"])
        self.model_field.setText(PROVIDERS[provider]["default_model"] if remote else "")
        self.api_key_field.clear()
        self.api_comment_check.setChecked(False)
        self.api_notice.setText("API 적용 시 전사 원문과 최근 문맥을 외부로 전송하며 과금됩니다. 음성 파일은 보내지 않습니다. 키는 저장하지 않습니다." + (" Gemini는 유료 Standard 계정용입니다. 무료/데이터 기여 설정은 데이터 사용 정책이 다르므로 제공사 콘솔을 확인하세요." if provider == "gemini" else ""))
        for widget in (self.model_label, self.model_field, self.api_key_field, self.api_comment_check, self.api_notice):
            widget.setVisible(remote)
        self.reasoning_combo.setVisible(provider == "openai")

    def _model_changed(self):
        sol = self.model_field.text().strip() == "gpt-6.1-sol"
        self.reasoning_combo.model().item(0).setEnabled(not sol)
        self.reasoning_combo.setCurrentIndex(1 if sol else 0)

    def set_usage_summary(self, text):
        self.usage_label.setText(text)

    def _apply_translation_settings(self):
        if self._running:
            return
        self.translation_settings_requested.emit({
            "provider": self.provider_combo.currentData(),
            "model": self.model_field.text().strip(),
            "reasoning_effort": self.reasoning_combo.currentData(),
            "api_key": self.api_key_field.text().strip(),
            "comments": self.api_comment_check.isChecked(),
        })

    def set_translation_mode(self, provider, description):
        self._translation_mode = provider
        self._translation_description = description
        self.translation_note.setText("현재: " + description)
        self.api_key_field.clear()
        if self._comment_setup_dialog is not None:
            self._comment_setup_dialog.close()

    def _bool_setting(self, key: str, default: bool) -> bool:
        value = self._settings.value(key, default)
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")

    def show_comment_setup(self, token: str, extension_path: str, status: str = "") -> None:
        """Display pairing material only in a transient dialog; never persist it."""
        if self._comment_setup_dialog is not None:
            self._comment_setup_dialog.close()
        dialog = QDialog(self)
        self._comment_setup_dialog = dialog
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("틱톡 댓글 번역 연결")
        dialog.setMinimumWidth(480)
        dialog.resize(540, 540)
        box = QVBoxLayout(dialog)
        box.setContentsMargins(22, 20, 22, 20)
        box.setSpacing(14)
        box.addWidget(_label("Chrome에서 댓글 바로 아래에 한국어로", "title", True))
        box.addWidget(_label("댓글 처리: " + self._translation_description + ". API 모드에서는 설정의 댓글 전송을 별도로 켜야 합니다. 음성 자막 처리를 우선합니다.", "scope", True))
        if status:
            box.addWidget(_label(str(status), "status", True))
        box.addWidget(_label("1. Chrome 주소창에 chrome://extensions 를 입력하세요.\n2. 오른쪽 위 ‘개발자 모드’를 켜고 ‘압축해제된 확장 프로그램을 로드합니다’를 누르세요. 아래 폴더를 선택하면 됩니다.", wrap=True))
        folder_row = QHBoxLayout()
        folder_label = _label(str(extension_path), "scope", True)
        folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        folder_row.addWidget(folder_label, 1)
        folder_button = QPushButton("확장 폴더 열기")
        folder_button.setObjectName("commentFolderButton")
        folder_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(extension_path))))
        folder_row.addWidget(folder_button)
        box.addLayout(folder_row)
        box.addWidget(_label("3. Chrome 도구 모음에서 확장 프로그램을 열고, 아래 연결 코드를 한 번 붙여 넣어 연결하세요.", wrap=True))
        token_row = QHBoxLayout()
        token_field = QLineEdit(str(token))
        token_field.setObjectName("commentPairingToken")
        token_field.setReadOnly(True)
        token_field.setAccessibleName("댓글 번역 연결 코드")
        token_field.setStyleSheet("background: #222a36; border: 1px solid #414d60; border-radius: 5px; padding: 8px;")
        token_row.addWidget(token_field, 1)
        copy_button = QPushButton("연결 코드 복사")
        copy_button.setObjectName("commentCopyButton")
        copy_button.setEnabled(bool(token))
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(token_field.text()))
        token_row.addWidget(copy_button)
        box.addLayout(token_row)
        box.addWidget(_label("4. 확장 프로그램에서 댓글 번역을 ON으로 켜고, 틱톡 웹에서 댓글을 펼치세요.\n\n댓글 번역은 확장 프로그램에서 따로 켜고 끕니다. 이 창의 ‘번역 시작’은 영상 소리용이며, 댓글만 볼 때는 누르지 않아도 됩니다. 이 앱은 켜 두세요.", wrap=True))
        box.addWidget(_label("연결 코드는 이 PC의 확장 프로그램 연결에만 사용하세요. 이 안내창은 코드를 설정이나 로그에 저장하지 않습니다.", "footer", True))
        close_button = QPushButton("닫기")
        close_button.clicked.connect(dialog.close)
        box.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.finished.connect(lambda _result: setattr(self, "_comment_setup_dialog", None))
        dialog.show()

    def _int_setting(self, key: str, default: int, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(self._settings.value(key, default))))
        except (ValueError, TypeError):
            return default

    def _keep_on_screen(self) -> None:
        """Recover a saved position if its former monitor is no longer present."""
        from PySide6.QtWidgets import QApplication
        if not any(screen.availableGeometry().intersects(self.frameGeometry()) for screen in QApplication.screens()):
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(screen.right() - self.width() - 24, screen.top() + 40)

    def _apply_pin(self, enabled: bool) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
        if visible:
            self.show()

    def _source_changed(self) -> None:
        source = self.source_combo.currentData()
        if not source:
            self.selected_source_label.setText("선택한 소리 장치 없음")
            self.scope_label.setText("재생 소리 장치가 없습니다. 이어폰·스피커 연결 후 새로고침해 주세요.")
            self.start_button.setEnabled(False)
            return
        scope = source.get("scope", "system")
        self.selected_source_label.setText("입력: " + str(source.get("label", source.get("id", ""))))
        if scope == "process":
            self.scope_label.setText("선택 앱의 소리를 번역합니다. 같은 앱의 다른 탭 소리도 들어올 수 있습니다.")
        else:
            self.scope_label.setText("이 장치의 PC 전체 소리: 영상 외 알림·통화도 포함됩니다. 다른 소리는 꺼 주세요.")
        self.start_button.setEnabled(not self._running)

    def set_sources(self, sources: list[dict]) -> None:
        current = self.source_combo.currentData()
        selected_id = current.get("id") if current else self._saved_source
        self._sources = [dict(source) for source in sources]
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for source in self._sources:
            self.source_combo.addItem(str(source.get("label", source["id"])), source)
        selected_index = next((i for i, source in enumerate(self._sources) if source["id"] == selected_id), 0)
        if self._sources:
            self.source_combo.setCurrentIndex(selected_index)
        self.source_combo.blockSignals(False)
        self._source_changed()

    def set_status(self, text: str) -> None:
        self._status_text = str(text)
        self.status_label.setProperty("error", False)
        self.status_label.setText(self._status_text)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_error(self, text: str) -> None:
        self.status_label.setProperty("error", True)
        self.status_label.setText(str(text))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_level(self, value: float) -> None:
        level = float(value)
        self.level_bar.setValue(round(max(0.0, min(1.0, level)) * 100) if math.isfinite(level) else 0)

    def set_running(self, value: bool) -> None:
        self._running = bool(value)
        self.start_button.setText("번역 중" if self._running else "번역 시작")
        self.start_button.setToolTip("이미 실행 중입니다. 설정 변경은 먼저 중지를 눌러 주세요." if self._running else "선택한 장치의 소리 번역을 시작합니다.")
        self.start_button.setEnabled(not self._running and self.source_combo.count() > 0)
        self.stop_button.setEnabled(self._running)
        self.source_combo.setEnabled(not self._running)
        self.language_combo.setEnabled(not self._running)
        self.refresh_button.setEnabled(not self._running)
        for widget in (self.provider_combo, self.model_field, self.reasoning_combo, self.api_key_field, self.api_comment_check, self.apply_translation_button):
            widget.setEnabled(not self._running)
        if not self._running:
            self.set_level(0)

    def _request_start(self) -> None:
        source = self.source_combo.currentData()
        if not source or self._running:
            return
        self.start_requested.emit(str(source["id"]), str(self.language_combo.currentData()))

    def _request_stop(self) -> None:
        if self._running:
            self.stop_requested.emit()

    def _request_reset(self) -> None:
        self.clear_captions()
        self.reset_requested.emit()

    def show_caption(self, original: str, korean: str, language: str, elapsed: float,
                     uncertain: bool = False, segment_id=None, provisional: bool = True) -> None:
        self._captions.append({"original": str(original), "korean": str(korean),
                               "language": str(language), "elapsed": float(elapsed),
                               "uncertain": bool(uncertain), "segment_id": segment_id,
                               "provisional": bool(provisional), "revised": False})
        for caption in list(self._captions)[:-2]:
            caption["provisional"] = False
        self._redraw_captions()
        QTimer.singleShot(0, self._center_latest)

    def upsert_caption(self, original: str, korean: str, language: str, elapsed: float,
                       uncertain: bool = False, segment_id=None, provisional: bool = True) -> None:
        """Continue a semantic unit in place without creating duplicate cards."""
        if segment_id is not None:
            for caption in self._captions:
                if caption.get('segment_id') == segment_id:
                    caption.update(original=str(original), korean=str(korean),
                                   language=str(language), elapsed=float(elapsed),
                                   uncertain=bool(uncertain), provisional=bool(provisional))
                    self._redraw_captions()
                    return
        self.show_caption(original, korean, language, elapsed, uncertain,
                          segment_id=segment_id, provisional=provisional)

    def revise_caption(self, segment_id, korean: str) -> None:
        if segment_id is None:
            return
        for caption in self._captions:
            if caption.get("segment_id") == segment_id:
                caption.update(korean=str(korean), provisional=False, revised=True)
                self._redraw_captions()
                return

    def _center_latest(self):
        if getattr(self, "_latest_card", None) is not None:
            card = self._latest_card
            viewport_height = self.scroll.viewport().height()
            offset = max(0, (viewport_height - card.height()) // 2)
            self.scroll.verticalScrollBar().setValue(max(0, card.y() - offset))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "scroll"):
            QTimer.singleShot(0, self._resize_captions)

    def _resize_captions(self):
        if hasattr(self, "font_spin"):
            self._redraw_captions()
            QTimer.singleShot(0, self._center_latest)

    def clear_captions(self) -> None:
        self._captions.clear()
        self._redraw_captions()

    def _redraw_captions(self, *_args) -> None:
        self._latest_card = None
        self.overlay.set_captions(self._captions)
        while self.caption_layout.count():
            item = self.caption_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        if not self._captions:
            empty = QWidget()
            empty_layout = QVBoxLayout(empty)
            empty_layout.setContentsMargins(14, 22, 14, 14)
            empty_layout.addWidget(_label("영상 소리가 한국어로 표시됩니다", "emptyTitle", True))
            empty_layout.addWidget(_label("브라우저나 MuMu에서 영상을 재생하세요.\n번역 시작을 누르면 선택한 장치의 소리를 받습니다.", "emptyBody", True))
            empty_layout.addStretch()
            self.caption_layout.addWidget(empty, 1)
            return
        for index, caption in enumerate(self._captions):
            card = QFrame()
            card.setObjectName("latestCard" if index == len(self._captions) - 1 else "captionCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(13, 10, 13, 12)
            card_layout.setSpacing(7)
            language = {"en": "영어", "es": "스페인어", "auto": "자동 감지", "ko": "한국어"}.get(caption["language"], caption["language"])
            elapsed = caption["elapsed"]
            timing = f" · 표시 지연 약 {elapsed:.1f}초" if math.isfinite(elapsed) and elapsed >= 0 else ""
            meta = ("확인 필요 · " if caption["uncertain"] else "") + language + " → 한국어" + timing
            if caption.get("revised"):
                meta += " · 문맥 반영"
            elif caption.get("provisional"):
                meta += " · 문맥 반영 중"
            card_layout.addWidget(_label(meta, "captionMeta", True))
            korean = _label(caption["korean"] or "인식한 말을 확인할 수 없습니다.", wrap=True)
            font = QFont("Malgun Gothic")
            font_size = self.font_spin.value() if index == len(self._captions)-1 else max(14, self.font_spin.value()-3)
            font.setPixelSize(font_size)
            font.setWeight(QFont.Weight.DemiBold if index == len(self._captions) - 1 else QFont.Weight.Normal)
            korean.setFont(font)
            weight = 600 if index == len(self._captions) - 1 else 400
            korean.setStyleSheet(f"font-size: {font_size}px; font-weight: {weight};")
            korean.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            card_layout.addWidget(korean)
            if self.original_check.isChecked() and caption["original"]:
                original = _label(caption["original"], "originalText", True)
                original.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                card_layout.addWidget(original)
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            card.ensurePolished()
            card.setFixedHeight(card_layout.totalHeightForWidth(max(200, self.scroll.viewport().width()-10)) + 8)
            self.caption_layout.addWidget(card)
            if index == len(self._captions)-1:
                self._latest_card = card
        self.caption_layout.addSpacing(max(80, self.scroll.viewport().height() // 2 - 55))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.overlay.save_position()
        self._settings.setValue("overlay_visible", self.overlay_check.isChecked())
        self._settings.setValue("overlay_locked", self.overlay_lock.isChecked())
        self.overlay.close()
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("always_on_top", self.pin_check.isChecked())
        self._settings.setValue("show_original", self.original_check.isChecked())
        self._settings.setValue("font_size", self.font_spin.value())
        self._settings.setValue("opacity", self.opacity_slider.value())
        self._settings.setValue("language", self.language_combo.currentData())
        source = self.source_combo.currentData()
        if source:
            self._settings.setValue("source", source["id"])
        self._settings.sync()
        self.closing.emit()
        super().closeEvent(event)
