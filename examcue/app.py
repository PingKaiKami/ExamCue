from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    AppTk = TkinterDnD.Tk
    DND_AVAILABLE = True
except ImportError:  # The app still starts and keeps the file-picker fallback.
    DND_FILES = "DND_Files"
    AppTk = tk.Tk
    DND_AVAILABLE = False

from .analysis import LLMStatus, OpenAICompatibleClient
from .domain import AnalysisReport, CueKind, ExamCue
from .media import MediaPreparationError, expand_youtube_urls
from .pipeline import ExamCuePipeline, PipelineCancelled, PipelineConfig
from .reporting import ANALYSIS_METHOD_LABELS, KIND_LABELS
from .transcription import cuda_diagnostics, test_cuda_transcription
from .util import seconds_to_timestamp

LLM_PROVIDERS = {
    "本機自動偵測（LM Studio／Ollama）": "",
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "自訂 OpenAI-compatible API": "custom",
}

TRANSCRIPTION_SOURCE_OPTIONS = {
    "自動（人工／自動字幕優先）": "auto",
    "強制 Whisper 語音辨識": "whisper",
}

TRANSCRIPTION_MODE_OPTIONS = {
    "快速（Beam 1／批次 8）": "fast",
    "標準（Beam 3／批次 4）": "standard",
    "精準（Beam 5／單批）": "accurate",
}

CHUNKING_MODE_OPTIONS = {
    "精細（60 秒／2,000 字元）": (60, 2000),
    "標準（120 秒／4,000 字元）": (120, 4000),
    "長篇（240 秒／8,000 字元）": (240, 8000),
}

ANALYSIS_SCAN_OPTIONS = {
    "完整掃描（建議，品質優先）": "full",
    "提示優先（較快，可能漏掉隱含考點）": "focused",
}

THEME_OPTIONS = {
    "淺色": "light",
    "深色": "dark",
}

DEFAULT_APP_SETTINGS = {
    "theme": "light",
    "chunking_mode": "標準（120 秒／4,000 字元）",
    "analysis_scan_mode": "完整掃描（建議，品質優先）",
    "auto_show_results": True,
    "show_completion_warnings": True,
}

REFERENCE_SUFFIXES = {".pdf", ".pptx", ".txt", ".md"}
MEDIA_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mp3",
    ".m4a",
    ".wav",
    ".aac",
    ".flac",
    ".ogg",
}


def _filter_reference_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    return _filter_existing_paths(paths, REFERENCE_SUFFIXES)


def _filter_media_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    return _filter_existing_paths(paths, MEDIA_SUFFIXES)


def _filter_existing_paths(
    paths: list[str], supported_suffixes: set[str]
) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    for value in paths:
        path = Path(value).expanduser()
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            rejected.append(path.name or str(path))
            continue
        resolved = str(path.resolve())
        if resolved not in accepted:
            accepted.append(resolved)
    return accepted, rejected


THEMES = {
    "light": {
        "background": "#E7E9F3",
        "surface": "#FFFFFF",
        "card": "#E1DFF4",
        "card_hover": "#D3D0EE",
        "text": "#15182F",
        "muted": "#62677D",
        "accent": "#5B4CE6",
        "accent_hover": "#4938D0",
        "accent_text": "#FFFFFF",
        "border": "#B8BED3",
        "input": "#FFFFFF",
        "selected": "#D2CCFF",
        "danger": "#B44355",
    },
    "dark": {
        "background": "#111425",
        "surface": "#191D33",
        "card": "#242944",
        "card_hover": "#303655",
        "text": "#F4F3FF",
        "muted": "#ADB3CB",
        "accent": "#887BFF",
        "accent_hover": "#A096FF",
        "accent_text": "#111425",
        "border": "#363C5E",
        "input": "#20253D",
        "selected": "#3A356B",
        "danger": "#FF8797",
    },
}

RESULT_GROUP_COLORS = {
    "light": {
        CueKind.EXPLICIT: ("#0F766E", "#E8F7F4", "#F4FBF9"),
        CueKind.INFERRED: ("#5B4CE6", "#EEEAFE", "#F7F5FF"),
        CueKind.WARNING: ("#B77900", "#FFF1C7", "#FFF9E8"),
        CueKind.EXCLUSION: ("#C9435B", "#FADDE3", "#FFF1F4"),
    },
    "dark": {
        CueKind.EXPLICIT: ("#218C82", "#173733", "#1B413C"),
        CueKind.INFERRED: ("#7165E8", "#29264B", "#312D59"),
        CueKind.WARNING: ("#B98216", "#3A3018", "#46391B"),
        CueKind.EXCLUSION: ("#C84F65", "#3B2029", "#482630"),
    },
}

RESULT_KIND_ORDER = (
    CueKind.EXPLICIT,
    CueKind.INFERRED,
    CueKind.WARNING,
    CueKind.EXCLUSION,
)

RESULT_KIND_ICONS = {
    CueKind.EXPLICIT: "✓",
    CueKind.INFERRED: "◆",
    CueKind.WARNING: "⚠",
    CueKind.EXCLUSION: "−",
}

RESULT_COLUMN_MIN_WIDTHS = (125, 105, 72, 170, 70, 280)
RESULT_COLUMN_EXTRA_WEIGHTS = (0, 0, 0, 2, 0, 4)


def _result_column_widths(available_width: int) -> tuple[int, ...]:
    """Return one shared pixel layout for the result header and every group.

    Letting each Tk grid negotiate its own requested width makes wrapped labels alter
    the source/time/cue boundaries from one category group to another. Compute the
    flexible columns once from the canvas width so every independent grid uses the
    exact same geometry.
    """
    minimum_total = sum(RESULT_COLUMN_MIN_WIDTHS)
    extra = max(available_width - minimum_total, 0)
    total_weight = sum(RESULT_COLUMN_EXTRA_WEIGHTS)
    widths = list(RESULT_COLUMN_MIN_WIDTHS)
    remaining = extra
    weighted_columns = [index for index, weight in enumerate(RESULT_COLUMN_EXTRA_WEIGHTS) if weight]
    for index in weighted_columns[:-1]:
        addition = extra * RESULT_COLUMN_EXTRA_WEIGHTS[index] // total_weight
        widths[index] += addition
        remaining -= addition
    if weighted_columns:
        widths[weighted_columns[-1]] += remaining
    return tuple(widths)


def _set_native_title_bar(window: tk.Misc, dark: bool) -> None:
    """Ask Windows to render Tk's native title bar in the selected theme."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        enabled = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                attribute,
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
            if result == 0:
                break
        caption = "#111425" if dark else "#E7E9F3"
        text = "#F4F3FF" if dark else "#15182F"
        for attribute, color in ((34, caption), (35, caption), (36, text)):
            rgb = color.lstrip("#")
            colorref = ctypes.c_int(
                int(rgb[0:2], 16) | (int(rgb[2:4], 16) << 8) | (int(rgb[4:6], 16) << 16)
            )
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                attribute,
                ctypes.byref(colorref),
                ctypes.sizeof(colorref),
            )
    except (AttributeError, OSError, tk.TclError):
        pass


class ExamCueApp(AppTk):
    def __init__(self):
        super().__init__()
        self.title("ExamCue — 課堂考點分析器")
        self.geometry("1160x800")
        self.minsize(960, 680)
        self.app_settings = self._load_app_settings()
        self.theme_name = str(self.app_settings["theme"])
        self.chunking_mode_var = tk.StringVar(value=str(self.app_settings["chunking_mode"]))
        self.analysis_scan_mode_var = tk.StringVar(
            value=str(self.app_settings["analysis_scan_mode"])
        )
        self.auto_show_results_var = tk.BooleanVar(
            value=bool(self.app_settings["auto_show_results"])
        )
        self.show_completion_warnings_var = tk.BooleanVar(
            value=bool(self.app_settings["show_completion_warnings"])
        )
        self.sources: list[str] = []
        self.reference_sources: list[str] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.llm_test_worker: threading.Thread | None = None
        self.url_expand_worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.is_paused = False
        self.report: AnalysisReport | None = None
        self.embedded_page: ttk.Frame | None = None
        self.style = ttk.Style(self)
        self._build_styles()
        self._build_ui()
        self._apply_theme()
        self.after(150, self._poll_events)

    @property
    def colors(self) -> dict[str, str]:
        return THEMES[self.theme_name]

    @staticmethod
    def _settings_path() -> Path:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "ExamCue" / "settings.json"

    def _load_app_settings(self) -> dict[str, object]:
        settings = dict(DEFAULT_APP_SETTINGS)
        try:
            loaded = json.loads(self._settings_path().read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                settings.update(loaded)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if settings.get("theme") not in THEMES:
            settings["theme"] = DEFAULT_APP_SETTINGS["theme"]
        if settings.get("chunking_mode") not in CHUNKING_MODE_OPTIONS:
            settings["chunking_mode"] = DEFAULT_APP_SETTINGS["chunking_mode"]
        if settings.get("analysis_scan_mode") not in ANALYSIS_SCAN_OPTIONS:
            settings["analysis_scan_mode"] = DEFAULT_APP_SETTINGS["analysis_scan_mode"]
        return settings

    def _save_app_settings(self) -> None:
        settings = {
            "theme": self.theme_name,
            "chunking_mode": self.chunking_mode_var.get(),
            "analysis_scan_mode": self.analysis_scan_mode_var.get(),
            "auto_show_results": self.auto_show_results_var.get(),
            "show_completion_warnings": self.show_completion_warnings_var.get(),
        }
        path = self._settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(path)
        except OSError:
            pass

    def _build_styles(self) -> None:
        self.style.theme_use("clam")
        self.option_add("*Font", ("Segoe UI", 11))
        self.option_add("*TCombobox*Listbox.font", ("Segoe UI", 11))

    def _apply_theme(self) -> None:
        colors = self.colors
        self.configure(background=colors["background"])
        self.style.configure(
            ".",
            background=colors["surface"],
            foreground=colors["text"],
            font=("Segoe UI", 11),
        )
        self.style.configure("App.TFrame", background=colors["background"])
        self.style.configure("Card.TFrame", background=colors["surface"])
        self.style.configure("SoftCard.TFrame", background=colors["card"])
        self.style.configure(
            "Title.TLabel",
            background=colors["background"],
            foreground=colors["text"],
            font=("Segoe UI", 25, "bold"),
        )
        self.style.configure(
            "DialogTitle.TLabel",
            background=colors["background"],
            foreground=colors["text"],
            font=("Segoe UI", 22, "bold"),
        )
        self.style.configure(
            "Heading.TLabel",
            background=colors["surface"],
            foreground=colors["text"],
            font=("Segoe UI", 15, "bold"),
        )
        self.style.configure(
            "Subtitle.TLabel",
            background=colors["background"],
            foreground=colors["muted"],
            font=("Segoe UI", 11),
        )
        self.style.configure(
            "CardMuted.TLabel",
            background=colors["surface"],
            foreground=colors["muted"],
            font=("Segoe UI", 11),
        )
        self.style.configure(
            "SoftCard.TLabel",
            background=colors["card"],
            foreground=colors["text"],
            font=("Segoe UI", 11),
        )
        self.style.configure(
            "SoftCardTitle.TLabel",
            background=colors["card"],
            foreground=colors["text"],
            font=("Segoe UI", 13, "bold"),
        )
        self.style.configure(
            "Drop.TLabel",
            background=colors["surface"],
            foreground=colors["muted"],
            bordercolor=colors["border"],
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 11),
        )
        self.style.configure(
            "DropActive.TLabel",
            background=colors["selected"],
            foreground=colors["accent"],
            bordercolor=colors["accent"],
            borderwidth=2,
            relief="solid",
            font=("Segoe UI", 11, "bold"),
        )
        self.style.configure(
            "TButton",
            background=colors["card"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            lightcolor=colors["card"],
            darkcolor=colors["card"],
            padding=(14, 9),
            font=("Segoe UI", 11),
        )
        self.style.map(
            "TButton",
            background=[("active", colors["card_hover"]), ("disabled", colors["surface"])],
            foreground=[("disabled", colors["muted"])],
        )
        self.style.configure(
            "Accent.TButton",
            background=colors["accent"],
            foreground=colors["accent_text"],
            bordercolor=colors["accent"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
            padding=(20, 10),
            font=("Segoe UI", 11, "bold"),
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", colors["accent_hover"]), ("disabled", colors["border"])],
            foreground=[("disabled", colors["muted"])],
        )
        self.style.configure(
            "Danger.TButton",
            background=colors["surface"],
            foreground=colors["danger"],
            bordercolor=colors["border"],
            padding=(10, 7),
        )
        self.style.configure(
            "Check.TButton",
            background=colors["surface"],
            foreground=colors["text"],
            borderwidth=0,
            padding=(0, 5),
            font=("Segoe UI", 11),
            anchor="w",
        )
        self.style.map(
            "Check.TButton",
            background=[("active", colors["surface"])],
            foreground=[("active", colors["accent"])],
        )
        self.style.configure(
            "TEntry",
            fieldbackground=colors["input"],
            foreground=colors["text"],
            insertcolor=colors["text"],
            bordercolor=colors["border"],
            padding=(10, 8),
        )
        self.style.map("TEntry", bordercolor=[("focus", colors["accent"])])
        self.style.configure(
            "TCombobox",
            fieldbackground=colors["input"],
            background=colors["card"],
            foreground=colors["text"],
            arrowcolor=colors["text"],
            bordercolor=colors["border"],
            padding=(8, 7),
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", colors["input"])],
            foreground=[("readonly", colors["text"])],
            bordercolor=[("focus", colors["accent"])],
        )
        self.style.configure(
            "TCheckbutton",
            background=colors["surface"],
            foreground=colors["text"],
            font=("Segoe UI", 11),
        )
        self.style.map("TCheckbutton", background=[("active", colors["surface"])])
        self.style.configure("TNotebook", background=colors["background"], borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            background=colors["card"],
            foreground=colors["muted"],
            padding=(24, 12),
            font=("Segoe UI", 11, "bold"),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", colors["surface"]), ("active", colors["card_hover"])],
            foreground=[("selected", colors["accent"])],
            padding=[("selected", (24, 12)), ("!selected", (24, 12))],
        )
        self.style.configure(
            "Card.TLabelframe",
            background=colors["surface"],
            bordercolor=colors["border"],
            borderwidth=1,
            relief="solid",
        )
        self.style.configure(
            "Card.TLabelframe.Label",
            background=colors["surface"],
            foreground=colors["text"],
            font=("Segoe UI", 14, "bold"),
        )
        self.style.configure(
            "Treeview",
            background=colors["input"],
            fieldbackground=colors["input"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            rowheight=38,
            font=("Segoe UI", 11),
        )
        self.style.map(
            "Treeview",
            background=[("selected", colors["selected"])],
            foreground=[("selected", colors["text"])],
        )
        self.style.configure(
            "Treeview.Heading",
            background=colors["card"],
            foreground=colors["text"],
            padding=(8, 10),
            font=("Segoe UI", 11, "bold"),
        )
        self.style.map("Treeview.Heading", background=[("active", colors["card_hover"])])
        self.style.configure(
            "Horizontal.TProgressbar",
            background=colors["accent"],
            troughcolor=colors["card"],
            bordercolor=colors["border"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
            thickness=18,
        )
        if hasattr(self, "source_list"):
            self.source_list.tag_configure(
                "odd", background=colors["input"], foreground=colors["text"]
            )
            self.source_list.tag_configure(
                "even", background=colors["card"], foreground=colors["text"]
            )
        if hasattr(self, "source_delete_button"):
            self.source_delete_button.configure(
                background=colors["surface"],
                foreground=colors["danger"],
                activebackground=colors["card_hover"],
                activeforeground=colors["danger"],
                highlightbackground=colors["danger"],
            )
        if hasattr(self, "result_canvas"):
            self.result_table.configure(background=colors["surface"])
            self.result_header.configure(background=colors["card"])
            self.result_canvas.configure(background=colors["surface"])
            self.result_content.configure(background=colors["surface"])
            self._render_results(self.report)
        if hasattr(self, "settings_canvas"):
            self.settings_canvas.configure(background=colors["surface"])
        if hasattr(self, "quiz_text"):
            self.quiz_text.configure(
                background=colors["input"],
                foreground=colors["text"],
                insertbackground=colors["text"],
            )
        self.theme_button.configure(text="☀  淺色" if self.theme_name == "dark" else "☾  深色")
        _set_native_title_bar(self, self.theme_name == "dark")

    def _toggle_theme(self) -> None:
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self._apply_theme()
        self._save_app_settings()

    def _open_app_settings(self) -> None:
        self._show_embedded_page(AppSettingsDialog(self))

    def _show_embedded_page(self, page: ttk.Frame) -> None:
        if self.embedded_page is not None and self.embedded_page.winfo_exists():
            self.embedded_page.destroy()
        self.embedded_page = page
        page.place(x=0, y=0, relwidth=1, relheight=1)
        page.lift()

    def _close_embedded_page(self, page: ttk.Frame) -> None:
        if self.embedded_page is page:
            self.embedded_page = None
        if page.winfo_exists():
            page.destroy()

    def _apply_app_settings(
        self,
        theme_name: str,
        chunking_mode: str,
        analysis_scan_mode: str,
        auto_show_results: bool,
        show_completion_warnings: bool,
    ) -> None:
        self.theme_name = theme_name if theme_name in THEMES else "light"
        self.chunking_mode_var.set(
            chunking_mode
            if chunking_mode in CHUNKING_MODE_OPTIONS
            else next(iter(CHUNKING_MODE_OPTIONS))
        )
        self.analysis_scan_mode_var.set(
            analysis_scan_mode
            if analysis_scan_mode in ANALYSIS_SCAN_OPTIONS
            else next(iter(ANALYSIS_SCAN_OPTIONS))
        )
        self.auto_show_results_var.set(auto_show_results)
        self.show_completion_warnings_var.set(show_completion_warnings)
        self._apply_theme()
        self._save_app_settings()
        self._update_run_summary()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))
        title_area = ttk.Frame(header, style="App.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        ttk.Label(title_area, text="ExamCue", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_area,
            text="把課堂影音整理成有證據、有時間戳的考試重點",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        self.theme_button = ttk.Button(header, text="☾  深色", command=self._toggle_theme)
        self.theme_button.pack(side="right", anchor="n")
        self.settings_button = ttk.Button(header, text="⚙  設定", command=self._open_app_settings)
        self.settings_button.pack(side="right", anchor="n", padx=(0, 8))

        self.status_var = tk.StringVar(value="準備就緒")
        self.progress_var = tk.DoubleVar(value=0)
        self._progress_item = 0
        self._progress_total = 0
        self._transcription_item = 0
        self._transcription_total = 0
        self._analysis_item = 0
        self._analysis_total = 0
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)
        input_tab = ttk.Frame(self.notebook, padding=18, style="Card.TFrame")
        settings_tab = ttk.Frame(self.notebook, style="Card.TFrame")
        self.run_tab = ttk.Frame(self.notebook, padding=18, style="Card.TFrame")
        self.results_tab = ttk.Frame(self.notebook, padding=18, style="Card.TFrame")
        self.quiz_tab = ttk.Frame(self.notebook, padding=18, style="Card.TFrame")
        self.notebook.add(input_tab, text="1  加入課程來源")
        self.notebook.add(settings_tab, text="2  調整分析設定")
        self.notebook.add(self.run_tab, text="3  開始課程分析")
        self.notebook.add(self.results_tab, text="4  查看分析結果")
        self.notebook.add(self.quiz_tab, text="5  練習選擇題")

        self._build_input_tab(input_tab)
        self._build_scrollable_settings_tab(settings_tab)
        self._build_run_tab(self.run_tab)
        self._build_results_tab(self.results_tab)
        self._build_quiz_tab(self.quiz_tab)

    def _build_input_tab(self, parent: ttk.Frame) -> None:
        intro = ttk.Frame(parent, padding=18, style="SoftCard.TFrame")
        intro.pack(fill="x", pady=(0, 16))
        ttk.Label(intro, text="加入課程內容", style="SoftCardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="支援多個本機影音檔、YouTube 影片與播放清單。你可以混合加入不同來源。",
            style="SoftCard.TLabel",
        ).pack(anchor="w", pady=(4, 12))
        actions = ttk.Frame(intro, style="SoftCard.TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="＋  加入影音檔", command=self._open_media_page).pack(side="left")
        self.add_url_button = ttk.Button(
            actions, text="🔗  加入 YouTube 連結", command=self._add_urls
        )
        self.add_url_button.pack(side="left", padx=8)
        ttk.Button(
            actions,
            text="▤  加入投影片／教材",
            command=self._open_reference_page,
        ).pack(side="left")
        self.reference_summary_var = tk.StringVar(value="尚未加入輔助教材")
        ttk.Label(
            intro,
            textvariable=self.reference_summary_var,
            style="SoftCard.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(12, 0))

        list_header = ttk.Frame(parent, style="Card.TFrame")
        list_header.pack(fill="x", pady=(0, 8))
        ttk.Label(list_header, text="待分析來源", style="Heading.TLabel").pack(side="left")
        ttk.Button(list_header, text="清空", command=self._clear_sources).pack(side="right")

        self.source_frame = ttk.Frame(parent, style="Card.TFrame")
        self.source_frame.pack(fill="both", expand=True)
        self.source_list = ttk.Treeview(
            self.source_frame,
            columns=("number", "source", "action"),
            show="headings",
            selectmode="none",
        )
        self.source_list.heading("number", text="序號")
        self.source_list.heading("source", text="來源")
        self.source_list.heading("action", text="")
        self.source_list.column("number", width=80, minwidth=70, stretch=False, anchor="center")
        self.source_list.column("source", width=850, minwidth=300, stretch=True, anchor="w")
        self.source_list.column("action", width=52, minwidth=52, stretch=False, anchor="center")
        self.source_list.pack(fill="both", expand=True)
        self.source_list.bind("<MouseWheel>", self._scroll_source_list)
        self.source_list.bind("<Motion>", self._source_row_motion)
        self.source_list.bind("<Leave>", self._schedule_source_delete_hide)
        self.source_list.bind("<Configure>", lambda _event: self._hide_source_delete())
        self._source_delete_row: str | None = None
        self._source_delete_hide_job: str | None = None
        self.source_delete_button = tk.Button(
            self.source_list,
            text="✕",
            command=self._remove_hovered_source,
            font=("Segoe UI", 12, "bold"),
            cursor="hand2",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            takefocus=False,
        )
        self.source_delete_button.bind("<Enter>", self._cancel_source_delete_hide)
        self.source_delete_button.bind("<Leave>", self._schedule_source_delete_hide)
        self.source_empty = ttk.Label(
            self.source_frame,
            text="尚未加入任何來源\n從上方選擇影音檔或貼上 YouTube 連結",
            style="CardMuted.TLabel",
            justify="center",
            font=("Segoe UI", 13),
        )
        self.source_empty.place(relx=0.5, rely=0.5, anchor="center")

    def _build_scrollable_settings_tab(self, parent: ttk.Frame) -> None:
        self.settings_canvas = tk.Canvas(
            parent,
            background=self.colors["surface"],
            bd=0,
            highlightthickness=0,
            takefocus=True,
        )
        self.settings_canvas.pack(fill="both", expand=True)
        self.settings_content = ttk.Frame(self.settings_canvas, padding=18, style="Card.TFrame")
        self.settings_window = self.settings_canvas.create_window(
            (0, 0), window=self.settings_content, anchor="nw"
        )
        self.settings_content.bind("<Configure>", self._settings_content_changed)
        self.settings_canvas.bind("<Configure>", self._settings_canvas_changed)
        self._build_settings_tab(self.settings_content)
        self._bind_settings_wheel(self.settings_content)

    def _settings_content_changed(self, _event: object = None) -> None:
        self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))

    def _settings_canvas_changed(self, event: tk.Event) -> None:
        self.settings_canvas.itemconfigure(self.settings_window, width=event.width)

    def _bind_settings_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._scroll_settings, add="+")
        for child in widget.winfo_children():
            self._bind_settings_wheel(child)

    def _scroll_settings(self, event: tk.Event) -> str:
        self.settings_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _scroll_settings_to_llm(self) -> None:
        self.settings_content.update_idletasks()
        self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all"))
        self.settings_canvas.yview_moveto(1.0)

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        self.project_name_var = tk.StringVar(value="我的課程")
        self.output_dir_var = tk.StringVar(value=str(Path.cwd() / "projects" / "我的課程"))
        self.output_language_var = tk.StringVar(value="繁體中文")
        self.whisper_model_var = tk.StringVar(value="small")
        self.transcription_source_var = tk.StringVar(value=next(iter(TRANSCRIPTION_SOURCE_OPTIONS)))
        self.transcription_mode_var = tk.StringVar(value=list(TRANSCRIPTION_MODE_OPTIONS)[1])
        self.use_llm_var = tk.BooleanVar(value=False)
        self.capture_frames_var = tk.BooleanVar(value=True)
        self.frame_interval_var = tk.StringVar(value="5")
        self.analyze_frames_var = tk.BooleanVar(value=True)
        self.generate_quiz_var = tk.BooleanVar(value=True)
        self.quiz_count_var = tk.StringVar(value="10")
        self.provider_var = tk.StringVar(value=next(iter(LLM_PROVIDERS)))
        self.endpoint_var = tk.StringVar(value="")
        self.llm_model_var = tk.StringVar(value="")
        self.api_key_var = tk.StringVar(value="")
        self.llm_test_status_var = tk.StringVar(value="")
        self._syncing_project_path = False
        self.project_name_var.trace_add("write", self._project_name_changed)
        self.output_dir_var.trace_add("write", self._output_dir_changed)

        basic = ttk.LabelFrame(
            parent, text="專案與轉錄", padding=(18, 14), style="Card.TLabelframe"
        )
        basic.pack(fill="x", pady=(0, 16))
        basic.columnconfigure(1, weight=1)
        row = self._setting_entry(basic, 0, "專案名稱", self.project_name_var)
        ttk.Label(basic, text="輸出資料夾").grid(row=row, column=0, sticky="w", pady=7)
        ttk.Entry(basic, textvariable=self.output_dir_var).grid(
            row=row, column=1, sticky="ew", padx=(16, 8), pady=4
        )
        ttk.Button(basic, text="瀏覽", command=self._choose_output_dir).grid(
            row=row, column=2, pady=4
        )
        row += 1
        row = self._setting_entry(basic, row, "報告語言", self.output_language_var)
        ttk.Label(basic, text="語音模型").grid(row=row, column=0, sticky="w", pady=7)
        ttk.Combobox(
            basic,
            textvariable=self.whisper_model_var,
            values=("tiny", "base", "small", "medium", "large-v3"),
            state="readonly",
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=4)
        row += 1
        ttk.Label(basic, text="轉錄來源").grid(row=row, column=0, sticky="w", pady=7)
        ttk.Combobox(
            basic,
            textvariable=self.transcription_source_var,
            values=tuple(TRANSCRIPTION_SOURCE_OPTIONS),
            state="readonly",
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=4)
        row += 1
        ttk.Label(basic, text="辨識模式").grid(row=row, column=0, sticky="w", pady=7)
        ttk.Combobox(
            basic,
            textvariable=self.transcription_mode_var,
            values=tuple(TRANSCRIPTION_MODE_OPTIONS),
            state="readonly",
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=4)

        v2 = ttk.LabelFrame(
            parent, text="V2 畫面與題庫", padding=(18, 14), style="Card.TLabelframe"
        )
        v2.pack(fill="x", pady=(0, 16))
        v2.columnconfigure(1, weight=1)
        self.capture_frames_button = ttk.Button(
            v2,
            command=lambda: self._toggle_v2_option(
                self.capture_frames_var,
                self.capture_frames_button,
                "定時擷取課程影片畫面",
            ),
            style="Check.TButton",
        )
        self.capture_frames_button.grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(v2, text="擷取間隔（分鐘）").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            v2,
            textvariable=self.frame_interval_var,
            values=("1", "2", "3", "5", "10", "15"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=(16, 0), pady=4)
        self.analyze_frames_button = ttk.Button(
            v2,
            command=lambda: self._toggle_v2_option(
                self.analyze_frames_var,
                self.analyze_frames_button,
                "若模型支援圖片，使用 LLM 判斷畫面內容",
            ),
            style="Check.TButton",
        )
        self.analyze_frames_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self.generate_quiz_button = ttk.Button(
            v2,
            command=lambda: self._toggle_v2_option(
                self.generate_quiz_var,
                self.generate_quiz_button,
                "分析完成後依考點生成四選一題目",
            ),
            style="Check.TButton",
        )
        self.generate_quiz_button.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(v2, text="最多題數").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Combobox(
            v2,
            textvariable=self.quiz_count_var,
            values=("5", "10", "15", "20", "30"),
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", padx=(16, 0), pady=4)
        ttk.Label(
            v2,
            text=(
                "畫面會存到專案 frames 資料夾。視覺模型不相容時只略過畫面理解，"
                "不會中止語音與考點分析。題庫功能需啟用 LLM。"
            ),
            style="CardMuted.TLabel",
            wraplength=820,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._refresh_v2_option_buttons()

        llm = ttk.LabelFrame(parent, text="AI 分析", padding=(18, 14), style="Card.TLabelframe")
        llm.pack(fill="x")
        llm.columnconfigure(1, weight=1)
        self.llm_toggle_button = ttk.Button(
            llm,
            text="☐  使用 LLM 分析；停用或連線失敗時改用規則式備援",
            command=self._toggle_llm_details,
            style="Check.TButton",
        )
        self.llm_toggle_button.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.llm_details = ttk.Frame(llm, padding=(0, 0, 0, 8), style="Card.TFrame")
        self.llm_details.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.llm_details.columnconfigure(1, weight=1)
        ttk.Label(self.llm_details, text="LLM 來源").grid(row=0, column=0, sticky="w", pady=7)
        provider_box = ttk.Combobox(
            self.llm_details,
            textvariable=self.provider_var,
            values=tuple(LLM_PROVIDERS),
            state="readonly",
        )
        provider_box.grid(row=0, column=1, sticky="ew", padx=(16, 8), pady=4)
        provider_box.bind("<<ComboboxSelected>>", self._provider_changed)
        self.llm_test_button = ttk.Button(
            self.llm_details, text="測試 LLM 分析", command=self._test_llm_connection
        )
        self.llm_test_button.grid(row=0, column=2, sticky="e", pady=4)
        row = self._setting_entry(self.llm_details, 1, "LLM Endpoint", self.endpoint_var)
        row = self._setting_entry(self.llm_details, row, "LLM Model", self.llm_model_var)
        ttk.Label(self.llm_details, text="API Key（不會儲存）").grid(
            row=row, column=0, sticky="w", pady=7
        )
        ttk.Entry(self.llm_details, textvariable=self.api_key_var, show="●").grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=4
        )
        ttk.Label(
            self.llm_details,
            text=(
                "外接 LLM 會收到逐字稿片段，可能產生 API 費用。API Key 僅保留於本次執行的記憶體中。"
            ),
            style="CardMuted.TLabel",
            wraplength=820,
        ).grid(row=row + 1, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Label(
            self.llm_details,
            textvariable=self.llm_test_status_var,
            style="CardMuted.TLabel",
            wraplength=820,
        ).grid(row=row + 2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.llm_details.grid_remove()

    def _build_quiz_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="考點練習題", style="Heading.TLabel").pack(anchor="w")
        self.quiz_text = tk.Text(
            parent,
            wrap="word",
            font=("Segoe UI", 11),
            relief="flat",
            padx=18,
            pady=16,
        )
        self.quiz_text.pack(fill="both", expand=True, pady=(12, 0))
        self.quiz_text.insert("1.0", "分析完成後，LLM 產生的四選一題目會顯示在這裡。")
        self.quiz_text.configure(state="disabled")

    def _setting_entry(
        self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=7)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=(16, 0), pady=4
        )
        return row + 1

    def _provider_changed(self, _event: object = None) -> None:
        endpoint = LLM_PROVIDERS[self.provider_var.get()]
        if endpoint != "custom":
            self.endpoint_var.set(endpoint)
        self.llm_test_status_var.set("")

    def _project_name_changed(self, *_args: object) -> None:
        if self._syncing_project_path:
            return
        name = self.project_name_var.get().strip()
        current = self.output_dir_var.get().strip()
        if not name or not current:
            return
        self._syncing_project_path = True
        try:
            self.output_dir_var.set(str(Path(current).parent / name))
        finally:
            self._syncing_project_path = False

    def _output_dir_changed(self, *_args: object) -> None:
        if self._syncing_project_path:
            return
        current = self.output_dir_var.get().strip().rstrip("\\/")
        if not current:
            return
        name = Path(current).name
        if not name:
            return
        self._syncing_project_path = True
        try:
            self.project_name_var.set(name)
        finally:
            self._syncing_project_path = False

    def _test_llm_connection(self) -> None:
        if self.llm_test_worker and self.llm_test_worker.is_alive():
            return
        client = OpenAICompatibleClient(
            self.endpoint_var.get().strip(),
            self.llm_model_var.get().strip(),
            self.api_key_var.get().strip(),
        )
        self.llm_test_button.configure(text="測試中…", state="disabled")
        self.llm_test_status_var.set("正在測試連線、JSON 格式與考點判斷…")

        def test_connection() -> None:
            try:
                status = client.test_connection()
                self.events.put(("llm_test", status))
            except Exception as exc:  # noqa: BLE001 - surface provider errors in the UI
                self.events.put(("llm_test_error", str(exc)))

        self.llm_test_worker = threading.Thread(target=test_connection, daemon=True)
        self.llm_test_worker.start()

    def _toggle_llm_details(self) -> None:
        self.use_llm_var.set(not self.use_llm_var.get())
        if self.use_llm_var.get():
            self.llm_toggle_button.configure(
                text="☑  使用 LLM 分析；停用或連線失敗時改用規則式備援"
            )
            self.llm_details.grid()
            self.after_idle(self._scroll_settings_to_llm)
        else:
            self.llm_toggle_button.configure(
                text="☐  使用 LLM 分析；停用或連線失敗時改用規則式備援"
            )
            self.llm_details.grid_remove()
            self.settings_canvas.yview_moveto(0.0)

    @staticmethod
    def _set_boolean_button_text(button: ttk.Button, variable: tk.BooleanVar, label: str) -> None:
        button.configure(text=f"{'☑' if variable.get() else '☐'}  {label}")

    def _toggle_v2_option(self, variable: tk.BooleanVar, button: ttk.Button, label: str) -> None:
        variable.set(not variable.get())
        self._set_boolean_button_text(button, variable, label)

    def _refresh_v2_option_buttons(self) -> None:
        self._set_boolean_button_text(
            self.capture_frames_button,
            self.capture_frames_var,
            "定時擷取課程影片畫面",
        )
        self._set_boolean_button_text(
            self.analyze_frames_button,
            self.analyze_frames_var,
            "若模型支援圖片，使用 LLM 判斷畫面內容",
        )
        self._set_boolean_button_text(
            self.generate_quiz_button,
            self.generate_quiz_var,
            "分析完成後依考點生成四選一題目",
        )

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        intro = ttk.Frame(parent, padding=20, style="SoftCard.TFrame")
        intro.pack(fill="x", pady=(0, 18))
        ttk.Label(intro, text="準備開始分析", style="SoftCardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="確認來源與設定後開始處理。分析期間可以留在此頁查看目前階段與進度。",
            style="SoftCard.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        summary = ttk.LabelFrame(
            parent, text="分析摘要", padding=(20, 16), style="Card.TLabelframe"
        )
        summary.pack(fill="x", pady=(0, 18))
        self.run_source_count_var = tk.StringVar(value="0 個來源")
        self.run_model_var = tk.StringVar(value="轉錄：small／標準／自動")
        self.run_chunk_var = tk.StringVar(value="逐字稿分段：標準／120 秒／4,000 字元")
        self.run_scan_var = tk.StringVar(value="掃描策略：完整掃描")
        self.run_ai_var = tk.StringVar(value="考點分析：規則式備援")
        ttk.Label(summary, textvariable=self.run_source_count_var).pack(anchor="w", pady=3)
        ttk.Label(summary, textvariable=self.run_model_var).pack(anchor="w", pady=3)
        ttk.Label(summary, textvariable=self.run_chunk_var).pack(anchor="w", pady=3)
        ttk.Label(summary, textvariable=self.run_scan_var).pack(anchor="w", pady=3)
        ttk.Label(summary, textvariable=self.run_ai_var).pack(anchor="w", pady=3)

        progress_card = ttk.LabelFrame(
            parent, text="目前進度", padding=(20, 18), style="Card.TLabelframe"
        )
        progress_card.pack(fill="x")
        ttk.Label(progress_card, textvariable=self.status_var, style="CardMuted.TLabel").pack(
            anchor="w", pady=(0, 10)
        )
        self.progress_bar = ttk.Progressbar(
            progress_card,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.pack(fill="x", pady=(0, 18))
        actions = ttk.Frame(progress_card, style="Card.TFrame")
        actions.pack(fill="x")
        self.start_button = ttk.Button(
            actions, text="開始分析", command=self._start, style="Accent.TButton"
        )
        self.start_button.pack(side="right")
        self.cancel_button = ttk.Button(
            actions, text="取消", command=self._cancel, state="disabled"
        )
        self.cancel_button.pack(side="right", padx=(0, 10))
        self.notebook.bind("<<NotebookTabChanged>>", self._notebook_changed)

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        header = ttk.Frame(parent, style="Card.TFrame")
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="考點分析結果", style="Heading.TLabel").pack(side="left")
        self.open_report_button = ttk.Button(
            header, text="開啟完整報告", command=self._open_report, state="disabled"
        )
        self.open_report_button.pack(side="right")

        self.result_table = tk.Frame(parent, bd=0, highlightthickness=0)
        self.result_table.pack(fill="both", expand=True)
        self.result_header = tk.Frame(self.result_table, bd=0, highlightthickness=1)
        self.result_header.pack(fill="x")
        self._configure_result_columns(self.result_header)
        for column, heading in enumerate(("分類", "分析方式", "可信度", "來源", "時間", "考點")):
            label = tk.Label(
                self.result_header,
                text=heading,
                font=("Segoe UI", 11, "bold"),
                padx=10,
                pady=10,
                anchor="center",
                bd=0,
                highlightthickness=1,
            )
            label.grid(row=0, column=column, sticky="nsew")
            self._bind_result_wheel(label)

        self.result_canvas = tk.Canvas(
            self.result_table, bd=0, highlightthickness=0, takefocus=True
        )
        self.result_canvas.pack(fill="both", expand=True)
        self.result_content = tk.Frame(self.result_canvas, bd=0, highlightthickness=0)
        self.result_window = self.result_canvas.create_window(
            (0, 0), window=self.result_content, anchor="nw"
        )
        self.result_content.bind("<Configure>", self._result_content_changed)
        self.result_canvas.bind("<Configure>", self._result_canvas_changed)
        self._bind_result_wheel(self.result_canvas)
        self._bind_result_wheel(self.result_content)
        self._render_results(None)

    def _open_media_page(self) -> None:
        self._show_embedded_page(MediaDialog(self))

    def _open_reference_page(self) -> None:
        self._show_embedded_page(ReferenceDialog(self))

    def _apply_reference_sources(self, paths: list[str]) -> None:
        self.reference_sources = list(dict.fromkeys(paths))
        if self.reference_sources:
            names = "、".join(Path(value).name for value in self.reference_sources)
            self.reference_summary_var.set(
                f"輔助教材 {len(self.reference_sources)} 份：{names}（再次加入可追加）"
            )
        else:
            self.reference_summary_var.set("尚未加入輔助教材")
        self._update_run_summary()

    def _add_urls(self) -> None:
        if self.url_expand_worker and self.url_expand_worker.is_alive():
            messagebox.showinfo("正在展開連結", "請稍候，目前仍在讀取播放清單。")
            return
        self._show_embedded_page(UrlDialog(self))

    def _start_url_expansion(self, values: list[str]) -> None:
        if not values:
            return
        self.add_url_button.configure(text="正在展開播放清單…", state="disabled")
        self.start_button.configure(state="disabled")
        self.url_expand_worker = threading.Thread(
            target=self._expand_url_sources,
            args=(values,),
            daemon=True,
        )
        self.url_expand_worker.start()

    def _expand_url_sources(self, values: list[str]) -> None:
        expanded: list[str] = []
        errors: list[str] = []
        for value in values:
            try:
                video_urls = expand_youtube_urls(value)
                expanded.extend(video_urls)
            except MediaPreparationError as exc:
                errors.append(f"{value}\n{exc}")
        self.events.put(
            (
                "urls_expanded",
                {
                    "urls": list(dict.fromkeys(expanded)),
                    "errors": errors,
                },
            )
        )

    def _append_sources(self, values: list[str]) -> None:
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in self.sources:
                self.sources.append(cleaned)
        self._refresh_source_rows()
        self._update_source_empty_state()

    def _source_row_motion(self, event: tk.Event) -> None:
        row_id = self.source_list.identify_row(event.y)
        if not row_id:
            self._hide_source_delete()
            return
        bounds = self.source_list.bbox(row_id, "action")
        if not bounds:
            self._hide_source_delete()
            return
        self._cancel_source_delete_hide()
        self._source_delete_row = row_id
        x, y, width, height = bounds
        size = min(30, max(22, height - 6))
        self.source_delete_button.place(
            x=x + (width - size) // 2,
            y=y + (height - size) // 2,
            width=size,
            height=size,
        )
        self.source_delete_button.lift()

    def _remove_hovered_source(self) -> None:
        row_id = self._source_delete_row
        children = list(self.source_list.get_children())
        if row_id not in children:
            self._hide_source_delete()
            return
        self.sources.pop(children.index(row_id))
        self._hide_source_delete()
        self._refresh_source_rows()
        self._update_source_empty_state()

    def _schedule_source_delete_hide(self, _event: object = None) -> None:
        self._cancel_source_delete_hide()
        self._source_delete_hide_job = self.after(80, self._hide_source_delete_if_outside)

    def _cancel_source_delete_hide(self, _event: object = None) -> None:
        if self._source_delete_hide_job is not None:
            self.after_cancel(self._source_delete_hide_job)
            self._source_delete_hide_job = None

    def _hide_source_delete_if_outside(self) -> None:
        self._source_delete_hide_job = None
        widget = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if widget in (self.source_list, self.source_delete_button):
            return
        self._hide_source_delete()

    def _hide_source_delete(self) -> None:
        self._cancel_source_delete_hide()
        self._source_delete_row = None
        self.source_delete_button.place_forget()

    def _clear_sources(self) -> None:
        self.sources.clear()
        self._refresh_source_rows()
        self._update_source_empty_state()

    def _refresh_source_rows(self) -> None:
        self._hide_source_delete()
        self.source_list.delete(*self.source_list.get_children())
        for index, source in enumerate(self.sources, start=1):
            tag = "odd" if index % 2 else "even"
            self.source_list.insert("", "end", values=(index, source, ""), tags=(tag,))
        self._update_run_summary()

    def _update_source_empty_state(self) -> None:
        if self.sources:
            self.source_empty.place_forget()
        else:
            self.source_empty.place(relx=0.5, rely=0.5, anchor="center")

    def _scroll_source_list(self, event: tk.Event) -> str:
        self._hide_source_delete()
        self.source_list.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _scroll_result_tree(self, event: tk.Event) -> str:
        self.result_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _bind_result_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._scroll_result_tree)

    def _result_content_changed(self, _event: tk.Event) -> None:
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))

    def _result_canvas_changed(self, event: tk.Event) -> None:
        self.result_canvas.itemconfigure(self.result_window, width=event.width)
        self._sync_result_columns(event.width)

    @staticmethod
    def _configure_result_columns(
        frame: tk.Misc, widths: tuple[int, ...] = RESULT_COLUMN_MIN_WIDTHS
    ) -> None:
        for column, width in enumerate(widths):
            frame.grid_columnconfigure(column, minsize=width, weight=0)

    def _sync_result_columns(self, canvas_width: int) -> None:
        # The header and category groups all use a one-pixel highlight border.
        widths = _result_column_widths(max(canvas_width - 2, 0))
        self._configure_result_columns(self.result_header, widths)
        for child in self.result_content.winfo_children():
            if isinstance(child, tk.Frame):
                self._configure_result_columns(child, widths)

    def _render_results(self, report: AnalysisReport | None) -> None:
        if not hasattr(self, "result_content"):
            return
        for child in self.result_content.winfo_children():
            child.destroy()
        colors = self.colors
        self.result_header.configure(
            background=colors["card"], highlightbackground=colors["border"]
        )
        for label in self.result_header.winfo_children():
            label.configure(
                background=colors["card"],
                foreground=colors["text"],
                highlightbackground=colors["border"],
            )
        self.result_canvas.yview_moveto(0)
        if report is None or not report.cues:
            empty = tk.Label(
                self.result_content,
                text="尚無分析結果",
                background=colors["surface"],
                foreground=colors["muted"],
                font=("Segoe UI", 13),
                pady=80,
            )
            empty.pack(fill="x")
            self._bind_result_wheel(empty)
            return

        grouped = {kind: [] for kind in RESULT_KIND_ORDER}
        for cue in report.cues:
            grouped.setdefault(cue.kind, []).append(cue)
        for kind in RESULT_KIND_ORDER:
            cues = sorted(grouped.get(kind, []), key=lambda cue: (-cue.confidence, cue.start))
            if cues:
                self._build_result_group(kind, cues)

    def _build_result_group(self, kind: CueKind, cues: list[ExamCue]) -> None:
        colors = self.colors
        rail, row_color, alternate = RESULT_GROUP_COLORS[self.theme_name][kind]
        group = tk.Frame(
            self.result_content,
            background=rail,
            bd=0,
            highlightthickness=1,
            highlightbackground=rail,
        )
        group.pack(fill="x", pady=(0, 10))
        self._configure_result_columns(
            group,
            _result_column_widths(max(self.result_canvas.winfo_width() - 2, 0)),
        )
        for row in range(len(cues)):
            group.grid_rowconfigure(row, minsize=46, weight=1)

        category = tk.Frame(group, background=rail, bd=0)
        category.grid(row=0, column=0, rowspan=len(cues), sticky="nsew")
        category_label = tk.Label(
            category,
            text=f"{RESULT_KIND_ICONS[kind]}\n{KIND_LABELS[kind]}\n{len(cues)} 筆",
            background=rail,
            foreground="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
            justify="center",
        )
        category_label.place(relx=0.5, rely=0.5, anchor="center")
        self._bind_result_wheel(category)
        self._bind_result_wheel(category_label)

        for row, cue in enumerate(cues):
            background = row_color if row % 2 == 0 else alternate
            values = (
                ANALYSIS_METHOD_LABELS.get(cue.analysis_method, cue.analysis_method),
                f"{cue.confidence:.0%}",
                cue.source_title,
                seconds_to_timestamp(cue.start),
                cue.title,
            )
            for offset, value in enumerate(values, start=1):
                anchor = "center" if offset in (1, 2, 4) else "w"
                wraplength = 160 if offset == 3 else 300 if offset == 5 else 0
                cell = tk.Label(
                    group,
                    text=value,
                    background=background,
                    foreground=colors["text"],
                    font=("Segoe UI", 11),
                    padx=10,
                    pady=9,
                    anchor=anchor,
                    justify="left",
                    wraplength=wraplength,
                    bd=0,
                    highlightthickness=1,
                    highlightbackground=colors["border"],
                )
                cell.grid(row=row, column=offset, sticky="nsew")
                self._bind_result_wheel(cell)

    def _notebook_changed(self, _event: object = None) -> None:
        self._update_run_summary()

    def _update_run_summary(self) -> None:
        if not hasattr(self, "run_source_count_var"):
            return
        self.run_source_count_var.set(f"來源數量：{len(self.sources)}")
        if self.reference_sources:
            self.run_source_count_var.set(
                f"課程來源：{len(self.sources)}；輔助教材：{len(self.reference_sources)}"
            )
        self.run_model_var.set(
            "轉錄："
            f"{self.whisper_model_var.get()}／"
            f"{self.transcription_mode_var.get().split('（', 1)[0]}／"
            f"{self.transcription_source_var.get().split('（', 1)[0]}"
        )
        seconds, chars = CHUNKING_MODE_OPTIONS[self.chunking_mode_var.get()]
        chunk_name = self.chunking_mode_var.get().split("（", 1)[0]
        self.run_chunk_var.set(f"逐字稿分段：{chunk_name}／{seconds} 秒／{chars:,} 字元")
        scan_name = self.analysis_scan_mode_var.get().split("（", 1)[0]
        self.run_scan_var.set(f"掃描策略：{scan_name}")
        if self.use_llm_var.get():
            self.run_ai_var.set(f"考點分析：{self.provider_var.get()}")
        else:
            self.run_ai_var.set("考點分析：規則式備援")

    def _choose_output_dir(self) -> None:
        current = self.output_dir_var.get().strip()
        options: dict[str, object] = {"title": "選擇輸出資料夾"}
        if current:
            options["initialdir"] = current
        selected = filedialog.askdirectory(**options)
        if selected:
            self.output_dir_var.set(selected)

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if not self.sources:
            messagebox.showwarning("尚未加入影片", "請先加入至少一個影音檔或 YouTube 連結。")
            return
        output_dir = Path(self.output_dir_var.get().strip()).expanduser()
        project_name = output_dir.name or self.project_name_var.get().strip() or "ExamCue 專案"
        if self.project_name_var.get() != project_name:
            self.project_name_var.set(project_name)
        config = PipelineConfig(
            project_name=project_name,
            project_dir=output_dir,
            sources=list(self.sources),
            output_language=self.output_language_var.get().strip() or "繁體中文",
            whisper_model=self.whisper_model_var.get(),
            transcription_source=TRANSCRIPTION_SOURCE_OPTIONS[self.transcription_source_var.get()],
            transcription_mode=TRANSCRIPTION_MODE_OPTIONS[self.transcription_mode_var.get()],
            analysis_chunk_seconds=CHUNKING_MODE_OPTIONS[self.chunking_mode_var.get()][0],
            analysis_chunk_chars=CHUNKING_MODE_OPTIONS[self.chunking_mode_var.get()][1],
            analysis_scan_mode=ANALYSIS_SCAN_OPTIONS[self.analysis_scan_mode_var.get()],
            llm_endpoint=self.endpoint_var.get().strip(),
            llm_model=self.llm_model_var.get().strip(),
            llm_api_key=self.api_key_var.get().strip(),
            use_llm=self.use_llm_var.get(),
            reference_sources=list(self.reference_sources),
            capture_frames=self.capture_frames_var.get(),
            frame_interval_seconds=max(int(self.frame_interval_var.get()), 1) * 60,
            analyze_frames_with_llm=self.analyze_frames_var.get(),
            generate_quiz=self.generate_quiz_var.get(),
            max_quiz_questions=max(int(self.quiz_count_var.get()), 1),
        )
        self.cancel_event.clear()
        self.resume_event.set()
        self.is_paused = False
        self._update_run_summary()
        self.notebook.select(self.run_tab)
        self.start_button.configure(text="暫停", command=self._toggle_pause, state="normal")
        self.cancel_button.configure(state="normal")
        self.open_report_button.configure(state="disabled")
        self.status_var.set("正在啟動…")
        self.progress_var.set(0)
        self._progress_item = 0
        self._progress_total = 0
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(12)
        self.report = None
        self._render_results(None)
        self.worker = threading.Thread(target=self._run_pipeline, args=(config,), daemon=True)
        self.worker.start()

    def _run_pipeline(self, config: PipelineConfig) -> None:
        try:
            pipeline = ExamCuePipeline(
                config,
                progress=lambda message: self.events.put(("progress", message)),
                cancel_event=self.cancel_event,
                resume_event=self.resume_event,
            )
            report = pipeline.run()
            self.events.put(("done", report))
        except PipelineCancelled as exc:
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:  # noqa: BLE001 - GUI thread must surface all failures
            self.events.put(("error", str(exc)))

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.resume_event.set()
        self.is_paused = False
        self.status_var.set("正在取消；目前步驟結束後停止…")
        self.cancel_button.configure(state="disabled")

    def _toggle_pause(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        if self.is_paused:
            self.is_paused = False
            self.resume_event.set()
            self.start_button.configure(text="暫停")
            self.status_var.set("繼續分析中…")
            if str(self.progress_bar.cget("mode")) == "indeterminate":
                self.progress_bar.start(12)
        else:
            self.is_paused = True
            self.resume_event.clear()
            self.progress_bar.stop()
            self.start_button.configure(text="繼續")
            self.status_var.set("已暫停；按「繼續」恢復分析")

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "progress":
                    if not self.is_paused:
                        self._update_progress(str(payload))
                elif event == "done":
                    self._show_report(payload)
                elif event == "cancelled":
                    self._finish_state(str(payload))
                elif event == "error":
                    self._finish_state("處理失敗")
                    messagebox.showerror("ExamCue 發生錯誤", str(payload))
                elif event == "llm_test":
                    self._show_llm_test_result(payload)
                elif event == "llm_test_error":
                    self._show_llm_test_error(str(payload))
                elif event == "urls_expanded":
                    self._show_expanded_urls(payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_events)

    def _show_expanded_urls(self, payload: object) -> None:
        self.add_url_button.configure(text="🔗  加入 YouTube 連結", state="normal")
        self.start_button.configure(state="normal")
        if not isinstance(payload, dict):
            messagebox.showerror("無法加入連結", "沒有收到有效的播放清單資料。")
            return

        urls = payload.get("urls", [])
        errors = payload.get("errors", [])
        if isinstance(urls, list):
            before = len(self.sources)
            self._append_sources([str(url) for url in urls])
            added = len(self.sources) - before
        else:
            added = 0

        if errors:
            detail = "\n\n".join(str(error) for error in errors)
            messagebox.showwarning(
                "部分連結無法讀取",
                f"已加入 {added} 支影片。以下連結未加入：\n\n{detail}",
            )

    def _show_llm_test_result(self, payload: object) -> None:
        self.llm_test_button.configure(text="測試 LLM 分析", state="normal")
        if isinstance(payload, LLMStatus):
            if not self.endpoint_var.get().strip():
                self.endpoint_var.set(payload.endpoint)
            if not self.llm_model_var.get().strip():
                self.llm_model_var.set(payload.model)
            self.llm_test_status_var.set(
                f"✓ 連線與考點分析測試成功：{payload.endpoint}　／　{payload.model}"
            )
        else:
            self._show_llm_test_error("沒有收到有效的連線結果")

    def _show_llm_test_error(self, detail: str) -> None:
        self.llm_test_button.configure(text="測試 LLM 分析", state="normal")
        self.llm_test_status_var.set("✕ LLM 連線或考點分析測試失敗")
        messagebox.showerror(
            "LLM 測試失敗",
            "請確認伺服器、Endpoint、Model、API Key，以及模型是否能遵循 JSON 考點指令。\n\n"
            f"詳細資訊：{detail}",
        )

    def _show_report(self, payload: object) -> None:
        assert isinstance(payload, AnalysisReport)
        self.report = payload
        self._render_results(payload)
        self._render_quiz(payload)
        self.open_report_button.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_var.set(100)
        if self.auto_show_results_var.get():
            self.notebook.select(self.results_tab)
        self._finish_state(f"完成：找到 {len(payload.cues)} 個項目")
        if payload.warnings and self.show_completion_warnings_var.get():
            messagebox.showwarning("分析完成，但有警告", "\n".join(dict.fromkeys(payload.warnings)))

    def _render_quiz(self, report: AnalysisReport) -> None:
        self.quiz_text.configure(state="normal")
        self.quiz_text.delete("1.0", "end")
        if not report.quiz_questions:
            self.quiz_text.insert(
                "1.0",
                "目前沒有題目。請確認已啟用 LLM 與『分析完成後依考點生成四選一題目』。",
            )
        else:
            blocks: list[str] = []
            for number, question in enumerate(report.quiz_questions, start=1):
                options = "\n".join(
                    f"   {chr(65 + index)}. {option}"
                    for index, option in enumerate(question.options)
                )
                blocks.append(
                    f"{number}. {question.question}\n{options}\n\n"
                    f"答案：{chr(65 + question.answer_index)}\n"
                    f"解析：{question.explanation}\n"
                    f"來源：{question.source_title} {seconds_to_timestamp(question.start)}"
                )
            self.quiz_text.insert("1.0", "\n\n".join(blocks))
        self.quiz_text.configure(state="disabled")

    def _finish_state(self, message: str) -> None:
        self.progress_bar.stop()
        self.resume_event.set()
        self.is_paused = False
        self.status_var.set(message)
        self.start_button.configure(text="開始分析", command=self._start, state="normal")
        self.cancel_button.configure(state="disabled")

    def _update_progress(self, message: str) -> None:
        self.status_var.set(message)
        transcription_item = re.search(r"\[語音辨識\s+(\d+)/(\d+)\]", message)
        transcription_time = re.search(r"語音辨識進度\s+([\d.]+)/([\d.]+)", message)
        transcription_done = re.search(r"語音辨識完成\s+(\d+)/(\d+)", message)
        analysis_item = re.search(r"\[考點分析\s+(\d+)/(\d+)\]", message)
        chunk_match = re.search(r"分析考點\s+(\d+)/(\d+)", message)
        overall_analysis_work = re.search(r"整體分析工作進度\s+(\d+)/(\d+)", message)
        chunk_work = re.search(r"分析工作進度\s+(\d+)/(\d+)", message)
        analysis_done = re.search(r"考點分析完成\s+(\d+)/(\d+)", message)
        if transcription_item:
            self._transcription_item = int(transcription_item.group(1))
            self._transcription_total = int(transcription_item.group(2))
            completed = max(self._transcription_item - 1, 0)
            value = completed / max(self._transcription_total, 1) * 50
            self._set_determinate_progress(value)
        elif transcription_time and self._transcription_total:
            current = float(transcription_time.group(1))
            duration = max(float(transcription_time.group(2)), 1)
            item_fraction = min(max(current / duration, 0), 1)
            completed = max(self._transcription_item - 1, 0) + item_fraction
            value = completed / self._transcription_total * 50
            self._set_determinate_progress(value)
        elif transcription_done:
            completed = int(transcription_done.group(1))
            total = max(int(transcription_done.group(2)), 1)
            self._set_determinate_progress(completed / total * 50)
        elif message == "語音辨識階段完成":
            self._set_determinate_progress(50)
        elif analysis_item:
            self._analysis_item = int(analysis_item.group(1))
            self._analysis_total = int(analysis_item.group(2))
            completed = max(self._analysis_item - 1, 0)
            value = 50 + completed / max(self._analysis_total, 1) * 50
            self._set_determinate_progress(value)
        elif overall_analysis_work:
            current = int(overall_analysis_work.group(1))
            total = max(int(overall_analysis_work.group(2)), 1)
            self._set_determinate_progress(50 + min(max(current / total, 0), 1) * 50)
        elif chunk_match and self._analysis_total:
            chunk = int(chunk_match.group(1))
            chunks = max(int(chunk_match.group(2)), 1)
            completed = max(self._analysis_item - 1, 0) + max(chunk - 1, 0) / chunks
            value = 50 + completed / self._analysis_total * 50
            self._set_determinate_progress(value)
        elif chunk_work and self._analysis_total:
            current = int(chunk_work.group(1))
            total = max(int(chunk_work.group(2)), 1)
            item_fraction = min(max(current / total, 0), 1)
            completed = max(self._analysis_item - 1, 0) + item_fraction
            value = 50 + completed / self._analysis_total * 50
            self._set_determinate_progress(value)
        elif analysis_done:
            completed = int(analysis_done.group(1))
            total = max(int(analysis_done.group(2)), 1)
            self._set_determinate_progress(50 + completed / total * 50)
        elif message.startswith("完成："):
            self._set_determinate_progress(100)

    def _set_determinate_progress(self, value: float) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_var.set(max(0, min(100, value)))

    def _open_report(self) -> None:
        path = Path(self.output_dir_var.get()) / "report.md"
        if path.exists():
            os.startfile(path)


class MediaDialog(ttk.Frame):
    """In-window page for adding multiple local audio/video files."""

    def __init__(self, parent: ExamCueApp):
        super().__init__(parent, style="App.TFrame")
        self.parent_app = parent
        self.paths: list[str] = []
        self.status_var = tk.StringVar()
        self.count_var = tk.StringVar()
        self._build_ui()
        self._refresh_rows()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x", pady=(0, 16))
        ttk.Button(header, text="←  返回", command=self._save).pack(side="left")
        ttk.Label(header, text="＋  加入課程影音", style="DialogTitle.TLabel").pack(
            side="left", padx=(14, 0)
        )

        self.drop_zone = ttk.Label(
            container,
            text=(
                "⬇\n\n將多個影片或音訊檔案拖曳到這裡\n"
                "支援 MP4、MKV、MOV、AVI、WebM、MP3、M4A、WAV、AAC、FLAC、OGG"
                if DND_AVAILABLE
                else "拖放元件尚未安裝，請使用下方按鈕一次選擇多個影音檔"
            ),
            style="Drop.TLabel",
            anchor="center",
            justify="center",
            padding=(20, 28),
        )
        self.drop_zone.pack(fill="x")
        if DND_AVAILABLE:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<DropEnter>>", self._drop_enter)
            self.drop_zone.dnd_bind("<<DropLeave>>", self._drop_leave)
            self.drop_zone.dnd_bind("<<Drop>>", self._drop_files)

        actions = ttk.Frame(container, style="App.TFrame")
        actions.pack(fill="x", pady=(14, 10))
        ttk.Button(actions, text="＋  選擇影音檔", command=self._select_files).pack(side="left")
        ttk.Button(actions, text="移除選取", command=self._remove_selected).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="清空", command=self._clear).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.count_var, style="Subtitle.TLabel").pack(side="right")

        self.file_list = ttk.Treeview(
            container,
            columns=("number", "name", "type", "path"),
            show="headings",
            selectmode="extended",
        )
        for column, title in (
            ("number", "序號"),
            ("name", "檔案名稱"),
            ("type", "格式"),
            ("path", "完整路徑"),
        ):
            self.file_list.heading(column, text=title)
        self.file_list.column("number", width=70, stretch=False, anchor="center")
        self.file_list.column("name", width=260, minwidth=160, stretch=False)
        self.file_list.column("type", width=80, stretch=False, anchor="center")
        self.file_list.column("path", width=560, minwidth=260, stretch=True)
        self.file_list.pack(fill="both", expand=True)
        self.file_list.bind(
            "<MouseWheel>",
            lambda event: self.file_list.yview_scroll(int(-event.delta / 120), "units"),
        )

        ttk.Label(
            container,
            textvariable=self.status_var,
            style="Subtitle.TLabel",
            wraplength=1000,
        ).pack(anchor="w", pady=(10, 0))

        footer = ttk.Frame(container, style="App.TFrame")
        footer.pack(fill="x", pady=(18, 0))
        ttk.Button(
            footer,
            text="完成加入",
            command=self._save,
            style="Accent.TButton",
        ).pack(side="right")
        ttk.Button(footer, text="取消", command=self._close).pack(side="right", padx=(0, 10))

    def _select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="選擇課堂影音",
            filetypes=[
                (
                    "影音檔",
                    "*.mp4 *.mkv *.mov *.avi *.webm *.mp3 *.m4a *.wav *.aac *.flac *.ogg",
                ),
                ("所有檔案", "*.*"),
            ],
        )
        self._append_paths(list(paths))

    def _drop_enter(self, _event: tk.Event) -> str:
        self.drop_zone.configure(style="DropActive.TLabel")
        return "copy"

    def _drop_leave(self, _event: tk.Event) -> str:
        self.drop_zone.configure(style="Drop.TLabel")
        return "copy"

    def _drop_files(self, event: tk.Event) -> str:
        self.drop_zone.configure(style="Drop.TLabel")
        try:
            values = list(self.tk.splitlist(str(event.data)))
        except tk.TclError:
            values = [str(event.data)]
        self._append_paths(values)
        return "copy"

    def _append_paths(self, values: list[str]) -> None:
        accepted, rejected = _filter_media_paths(values)
        added = 0
        for value in accepted:
            if value not in self.paths and value not in self.parent_app.sources:
                self.paths.append(value)
                added += 1
        messages = []
        if added:
            messages.append(f"已加入清單 {added} 個影音檔")
        if rejected:
            messages.append(f"已略過 {len(rejected)} 個不支援或不存在的檔案")
        if not messages and accepted:
            messages.append("選取的影音檔已在待加入或待分析清單中")
        self.status_var.set("；".join(messages))
        self._refresh_rows()

    def _remove_selected(self) -> None:
        indexes = sorted(
            (
                int(self.file_list.item(item_id, "values")[0]) - 1
                for item_id in self.file_list.selection()
            ),
            reverse=True,
        )
        for index in indexes:
            if 0 <= index < len(self.paths):
                self.paths.pop(index)
        if indexes:
            self.status_var.set(f"已移除 {len(indexes)} 個影音檔")
        self._refresh_rows()

    def _clear(self) -> None:
        self.paths.clear()
        self.status_var.set("待加入影音清單已清空")
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        for item_id in self.file_list.get_children():
            self.file_list.delete(item_id)
        for index, value in enumerate(self.paths, start=1):
            path = Path(value)
            self.file_list.insert(
                "", "end", values=(index, path.name, path.suffix.upper().lstrip("."), value)
            )
        self.count_var.set(f"待加入 {len(self.paths)} 個影音檔")

    def _save(self) -> None:
        self.parent_app._append_sources(self.paths)
        self._close()

    def _close(self) -> None:
        self.parent_app._close_embedded_page(self)


class ReferenceDialog(ttk.Frame):
    """In-window page for staging, dropping, and managing reference documents."""

    def __init__(self, parent: ExamCueApp):
        super().__init__(parent, style="App.TFrame")
        self.parent_app = parent
        self.paths = list(parent.reference_sources)
        self.status_var = tk.StringVar()
        self.count_var = tk.StringVar()
        self._build_ui()
        self._refresh_rows()

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x")
        ttk.Button(header, text="←  返回", command=self._save).pack(side="left")
        ttk.Label(
            header,
            text="▤  加入投影片與教材",
            style="DialogTitle.TLabel",
        ).pack(side="left", padx=(14, 0))
        ttk.Label(
            container,
            text=("教材只用於校正術語與補充上下文，不會單獨被當作老師明示的考點證據。"),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 16))

        self.drop_zone = ttk.Label(
            container,
            text=(
                "⬇\n\n將多個 PDF、PPTX、TXT 或 Markdown 檔案拖曳到這裡\n"
                "也可以使用下方的「選擇檔案」按鈕"
                if DND_AVAILABLE
                else "拖放元件尚未安裝，請使用下方按鈕一次選擇多個教材"
            ),
            style="Drop.TLabel",
            anchor="center",
            justify="center",
            padding=(20, 28),
        )
        self.drop_zone.pack(fill="x")
        if DND_AVAILABLE:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<DropEnter>>", self._drop_enter)
            self.drop_zone.dnd_bind("<<DropLeave>>", self._drop_leave)
            self.drop_zone.dnd_bind("<<Drop>>", self._drop_files)

        actions = ttk.Frame(container, style="App.TFrame")
        actions.pack(fill="x", pady=(14, 10))
        ttk.Button(actions, text="＋  選擇檔案", command=self._select_files).pack(side="left")
        ttk.Button(actions, text="移除選取", command=self._remove_selected).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="清空", command=self._clear).pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.count_var, style="Subtitle.TLabel").pack(side="right")

        self.file_list = ttk.Treeview(
            container,
            columns=("number", "name", "path"),
            show="headings",
            selectmode="extended",
        )
        self.file_list.heading("number", text="序號")
        self.file_list.heading("name", text="檔案名稱")
        self.file_list.heading("path", text="完整路徑")
        self.file_list.column("number", width=70, stretch=False, anchor="center")
        self.file_list.column("name", width=250, minwidth=150, stretch=False)
        self.file_list.column("path", width=650, minwidth=280, stretch=True)
        self.file_list.pack(fill="both", expand=True)
        self.file_list.bind(
            "<MouseWheel>",
            lambda event: self.file_list.yview_scroll(int(-event.delta / 120), "units"),
        )

        ttk.Label(
            container,
            textvariable=self.status_var,
            style="Subtitle.TLabel",
            wraplength=1000,
        ).pack(anchor="w", pady=(10, 0))

        footer = ttk.Frame(container, style="App.TFrame")
        footer.pack(fill="x", pady=(18, 0))
        ttk.Button(
            footer,
            text="完成",
            command=self._save,
            style="Accent.TButton",
        ).pack(side="right")
        ttk.Button(footer, text="取消", command=self._close).pack(side="right", padx=(0, 10))

    def _select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="加入投影片或教材",
            filetypes=[
                ("支援的教材", "*.pdf *.pptx *.txt *.md"),
                ("PDF", "*.pdf"),
                ("PowerPoint", "*.pptx"),
                ("文字文件", "*.txt *.md"),
            ],
        )
        self._append_paths(list(paths))

    def _drop_enter(self, _event: tk.Event) -> str:
        self.drop_zone.configure(style="DropActive.TLabel")
        return "copy"

    def _drop_leave(self, _event: tk.Event) -> str:
        self.drop_zone.configure(style="Drop.TLabel")
        return "copy"

    def _drop_files(self, event: tk.Event) -> str:
        self.drop_zone.configure(style="Drop.TLabel")
        try:
            values = list(self.tk.splitlist(str(event.data)))
        except tk.TclError:
            values = [str(event.data)]
        self._append_paths(values)
        return "copy"

    def _append_paths(self, values: list[str]) -> None:
        accepted, rejected = _filter_reference_paths(values)
        added = 0
        for value in accepted:
            if value not in self.paths:
                self.paths.append(value)
                added += 1
        messages = []
        if added:
            messages.append(f"已加入 {added} 份教材")
        if rejected:
            messages.append(f"已略過 {len(rejected)} 個不支援或不存在的檔案")
        if not messages and accepted:
            messages.append("選取的教材已經在清單中")
        self.status_var.set("；".join(messages))
        self._refresh_rows()

    def _remove_selected(self) -> None:
        indexes = sorted(
            (
                int(self.file_list.item(item_id, "values")[0]) - 1
                for item_id in self.file_list.selection()
            ),
            reverse=True,
        )
        for index in indexes:
            if 0 <= index < len(self.paths):
                self.paths.pop(index)
        if indexes:
            self.status_var.set(f"已移除 {len(indexes)} 份教材")
        self._refresh_rows()

    def _clear(self) -> None:
        self.paths.clear()
        self.status_var.set("教材清單已清空；按完成後才會套用")
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        for item_id in self.file_list.get_children():
            self.file_list.delete(item_id)
        for index, value in enumerate(self.paths, start=1):
            path = Path(value)
            self.file_list.insert("", "end", values=(index, path.name, value))
        self.count_var.set(f"目前 {len(self.paths)} 份教材")

    def _save(self) -> None:
        self.parent_app._apply_reference_sources(self.paths)
        self._close()

    def _close(self) -> None:
        self.parent_app._close_embedded_page(self)


class AppSettingsDialog(ttk.Frame):
    def __init__(self, parent: ExamCueApp):
        super().__init__(parent, style="App.TFrame")
        self.parent_app = parent

        current_theme = next(
            (label for label, value in THEME_OPTIONS.items() if value == parent.theme_name),
            "淺色",
        )
        self.theme_var = tk.StringVar(value=current_theme)
        self.chunking_var = tk.StringVar(value=parent.chunking_mode_var.get())
        self.scan_mode_var = tk.StringVar(value=parent.analysis_scan_mode_var.get())
        self.auto_show_results_var = tk.BooleanVar(value=parent.auto_show_results_var.get())
        self.show_warnings_var = tk.BooleanVar(value=parent.show_completion_warnings_var.get())
        self.chunk_description_var = tk.StringVar()
        self.cuda_status_var = tk.StringVar(value="正在檢測語音辨識加速環境…")
        self.cuda_detail_var = tk.StringVar()
        self.cuda_test_worker: threading.Thread | None = None
        self.cuda_test_result: tuple[str, object] | None = None
        self._build_ui()
        self._bind_content_wheel(self.content_frame)
        self._update_chunk_description()
        self._refresh_cuda_diagnostics()
        self._refresh_option_buttons()
        self.bind("<MouseWheel>", self._scroll_content)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)
        header = ttk.Frame(container, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ttk.Button(header, text="←  返回", command=self._close).pack(side="left")
        ttk.Label(header, text="⚙  應用程式設定", style="DialogTitle.TLabel").pack(
            side="left", padx=(14, 0)
        )

        self.content_canvas = tk.Canvas(
            container,
            background=self.parent_app.colors["background"],
            bd=0,
            highlightthickness=0,
            takefocus=True,
        )
        self.content_canvas.grid(row=1, column=0, sticky="nsew")
        self.content_frame = ttk.Frame(self.content_canvas, style="App.TFrame")
        self.content_window = self.content_canvas.create_window(
            (0, 0), window=self.content_frame, anchor="nw"
        )
        self.content_frame.bind("<Configure>", self._content_changed)
        self.content_canvas.bind("<Configure>", self._canvas_changed)

        actions = ttk.Frame(container, style="App.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(18, 0))
        ttk.Button(
            actions,
            text="儲存設定",
            command=self._save,
            style="Accent.TButton",
        ).pack(side="right")
        ttk.Button(actions, text="取消", command=self._close).pack(side="right", padx=(0, 10))

        analysis = ttk.LabelFrame(
            self.content_frame,
            text="逐字稿分段",
            padding=(18, 14),
            style="Card.TLabelframe",
        )
        analysis.pack(fill="x", pady=(0, 14))
        analysis.columnconfigure(1, weight=1)
        ttk.Label(analysis, text="分段模式").grid(row=0, column=0, sticky="w", pady=6)
        chunk_box = ttk.Combobox(
            analysis,
            textvariable=self.chunking_var,
            values=tuple(CHUNKING_MODE_OPTIONS),
            state="readonly",
        )
        chunk_box.grid(row=0, column=1, sticky="ew", padx=(16, 0), pady=4)
        chunk_box.bind("<<ComboboxSelected>>", self._update_chunk_description)
        ttk.Label(
            analysis,
            textvariable=self.chunk_description_var,
            style="CardMuted.TLabel",
            wraplength=580,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 2))
        ttk.Label(analysis, text="掃描策略").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(
            analysis,
            textvariable=self.scan_mode_var,
            values=tuple(ANALYSIS_SCAN_OPTIONS),
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=(16, 0), pady=4)
        ttk.Label(
            analysis,
            text=(
                "完整掃描會把每段逐字稿交給 LLM，較能找到沒有固定關鍵字的考點；"
                "提示優先只分析規則找到的候選範圍，速度較快但可能遺漏內容。"
            ),
            style="CardMuted.TLabel",
            wraplength=580,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))

        acceleration = ttk.LabelFrame(
            self.content_frame,
            text="語音辨識加速檢測",
            padding=(18, 14),
            style="Card.TLabelframe",
        )
        acceleration.pack(fill="x", pady=(0, 14))
        acceleration.columnconfigure(0, weight=1)
        ttk.Label(
            acceleration,
            textvariable=self.cuda_status_var,
            style="CardTitle.TLabel",
            wraplength=580,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            acceleration,
            textvariable=self.cuda_detail_var,
            style="CardMuted.TLabel",
            wraplength=580,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 10))
        ttk.Button(
            acceleration,
            text="重新檢測環境",
            command=self._refresh_cuda_diagnostics,
        ).grid(row=2, column=0, sticky="w")
        self.cuda_test_button = ttk.Button(
            acceleration,
            text="測試 GPU 語音辨識",
            command=self._start_cuda_test,
        )
        self.cuda_test_button.grid(row=2, column=1, sticky="e")

        appearance = ttk.LabelFrame(
            self.content_frame,
            text="介面與完成行為",
            padding=(18, 14),
            style="Card.TLabelframe",
        )
        appearance.pack(fill="x")
        appearance.columnconfigure(1, weight=1)
        ttk.Label(appearance, text="主題").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Combobox(
            appearance,
            textvariable=self.theme_var,
            values=tuple(THEME_OPTIONS),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=(16, 0), pady=4)
        self.auto_show_button = ttk.Button(
            appearance,
            command=self._toggle_auto_show_results,
            style="Check.TButton",
        )
        self.auto_show_button.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 4))
        self.show_warnings_button = ttk.Button(
            appearance,
            command=self._toggle_show_warnings,
            style="Check.TButton",
        )
        self.show_warnings_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

    def _content_changed(self, _event: object = None) -> None:
        self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all"))

    def _canvas_changed(self, event: tk.Event) -> None:
        self.content_canvas.itemconfigure(self.content_window, width=event.width)

    def _scroll_content(self, event: tk.Event) -> str:
        self.content_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _bind_content_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._scroll_content, add="+")
        for child in widget.winfo_children():
            self._bind_content_wheel(child)

    def _toggle_auto_show_results(self) -> None:
        self.auto_show_results_var.set(not self.auto_show_results_var.get())
        self._refresh_option_buttons()

    def _refresh_cuda_diagnostics(self) -> None:
        diagnostics = cuda_diagnostics()
        gpu_name = str(diagnostics.get("gpu_name") or "NVIDIA GPU")
        if diagnostics.get("cuda_ready"):
            self.cuda_status_var.set(f"✓ GPU 加速環境可用：{gpu_name}")
            self.cuda_detail_var.set(
                "Whisper 將使用 CUDA／float16。按下實際測試可載入目前選擇的語音模型，"
                "確認模型推論也能在 GPU 上執行。正式分析若採用現有字幕或逐字稿快取，"
                "不會啟動 Whisper，這是正常行為。"
            )
            self.cuda_test_button.configure(state="normal")
            return
        missing = diagnostics.get("missing_dlls") or []
        if diagnostics.get("gpu_count"):
            detail = (
                "已偵測到 NVIDIA GPU，但缺少或找不到 "
                f"{', '.join(missing) if missing else 'CUDA 執行庫'}。"
                "請關閉 ExamCue，在專案資料夾執行 "
                '.\\.venv\\Scripts\\python.exe -m pip install -e ".[cuda-windows]"，'
                "重新啟動後再按『重新檢測環境』。未安裝時 Whisper 會自動改用 CPU／int8。"
            )
        else:
            detail = (
                "未偵測到可供 CTranslate2 使用的 NVIDIA CUDA GPU；"
                "Whisper 將使用 CPU／int8，仍可正常執行但速度較慢。"
                "若電腦有 NVIDIA GPU，請先更新顯示卡驅動，再安裝 CUDA Windows 選用套件。"
            )
        error = str(diagnostics.get("error") or "").strip()
        if error:
            detail = f"{detail} 檢測訊息：{error}"
        self.cuda_status_var.set("⚠ GPU 加速目前不可用")
        self.cuda_detail_var.set(detail)
        self.cuda_test_button.configure(state="disabled")

    def _start_cuda_test(self) -> None:
        if self.cuda_test_worker and self.cuda_test_worker.is_alive():
            return
        model_name = self.parent_app.whisper_model_var.get().strip() or "small"
        self.cuda_test_button.configure(text="測試中…", state="disabled")
        self.cuda_status_var.set(f"正在以 {model_name} 執行 GPU 語音辨識測試…")
        self.cuda_detail_var.set("首次使用該模型時可能需要下載；測試會執行一小段合成音訊。")
        self.cuda_test_result = None

        def worker() -> None:
            try:
                self.cuda_test_result = (
                    "success",
                    test_cuda_transcription(model_name),
                )
            except Exception as exc:  # noqa: BLE001 - surface diagnostic detail
                self.cuda_test_result = ("error", str(exc))

        self.cuda_test_worker = threading.Thread(target=worker, daemon=True)
        self.cuda_test_worker.start()
        self.after(100, self._poll_cuda_test)

    def _poll_cuda_test(self) -> None:
        if self.cuda_test_worker and self.cuda_test_worker.is_alive():
            self.after(100, self._poll_cuda_test)
            return
        self.cuda_test_button.configure(text="測試 GPU 語音辨識", state="normal")
        if not self.cuda_test_result:
            self.cuda_status_var.set("✕ GPU 測試沒有回傳結果")
            return
        status, payload = self.cuda_test_result
        if status == "success" and isinstance(payload, dict):
            self.cuda_status_var.set("✓ GPU 語音辨識測試成功")
            self.cuda_detail_var.set(
                f"模型：{payload.get('tested_model', '')}；裝置：CUDA／float16；"
                f"測試耗時：{float(payload.get('test_seconds', 0)):.2f} 秒。"
            )
        else:
            self.cuda_status_var.set("✕ GPU 語音辨識測試失敗")
            self.cuda_detail_var.set(str(payload))

    def _toggle_show_warnings(self) -> None:
        self.show_warnings_var.set(not self.show_warnings_var.get())
        self._refresh_option_buttons()

    def _refresh_option_buttons(self) -> None:
        self.auto_show_button.configure(
            text=(
                "☑  分析完成後自動切換到結果頁"
                if self.auto_show_results_var.get()
                else "☐  分析完成後自動切換到結果頁"
            )
        )
        self.show_warnings_button.configure(
            text=(
                "☑  分析完成後以視窗顯示警告（警告仍會寫入報告）"
                if self.show_warnings_var.get()
                else "☐  分析完成後以視窗顯示警告（警告仍會寫入報告）"
            )
        )

    def _update_chunk_description(self, _event: object = None) -> None:
        seconds, chars = CHUNKING_MODE_OPTIONS[self.chunking_var.get()]
        self.chunk_description_var.set(
            f"每段連續上下文最多約 {seconds} 秒或 {chars:,} 字元，"
            "任一條件先達到就切開。較長設定能保留更多上下文，但會使用更多 token，"
            "也更依賴模型的上下文能力。"
        )

    def _save(self) -> None:
        self.parent_app._apply_app_settings(
            THEME_OPTIONS[self.theme_var.get()],
            self.chunking_var.get(),
            self.scan_mode_var.get(),
            self.auto_show_results_var.get(),
            self.show_warnings_var.get(),
        )
        self._close()

    def _close(self) -> None:
        self.parent_app._close_embedded_page(self)


class UrlDialog(ttk.Frame):
    def __init__(self, parent: ExamCueApp):
        super().__init__(parent, style="App.TFrame")
        self.parent_app = parent
        self.value: list[str] | None = None
        self.rows: list[dict[str, object]] = []
        self._build_ui()
        self._add_row(focus=True)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=24, style="App.TFrame")
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="App.TFrame")
        header.pack(fill="x")
        ttk.Button(header, text="←  返回", command=self._close).pack(side="left")
        ttk.Label(header, text="🔗  加入課程影片", style="DialogTitle.TLabel").pack(
            side="left", padx=(14, 0)
        )
        info = ttk.Frame(container, padding=16, style="SoftCard.TFrame")
        info.pack(fill="x", pady=(14, 16))
        ttk.Label(info, text="支援 YouTube 影片與播放清單", style="SoftCardTitle.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            info,
            text="貼上連結後會自動新增下一列；每列只放一個網址。",
            style="SoftCard.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        list_shell = ttk.Frame(container, style="Card.TFrame")
        list_shell.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            list_shell,
            borderwidth=0,
            highlightthickness=0,
            height=72,
            background=self.parent_app.colors["surface"],
        )
        self.rows_frame = ttk.Frame(self.canvas, style="Card.TFrame")
        self.rows_window = self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.rows_frame.bind("<Configure>", self._refresh_scroll_region)
        self.canvas.bind("<Configure>", self._resize_rows_frame)
        self.canvas.pack(fill="both", expand=True)
        self.bind("<MouseWheel>", self._scroll_rows)

        footer = ttk.Frame(container, padding=(0, 16, 0, 0), style="App.TFrame")
        footer.pack(fill="x")
        self.count_var = tk.StringVar(value="已加入 0 個連結")
        ttk.Label(footer, textvariable=self.count_var, style="Subtitle.TLabel").pack(side="left")
        self.done_button = ttk.Button(
            footer,
            text="完成",
            command=self._accept,
            style="Accent.TButton",
            state="disabled",
        )
        self.done_button.pack(side="right")
        ttk.Button(footer, text="取消", command=self._close).pack(side="right", padx=(0, 10))

    def _add_row(self, focus: bool = False) -> None:
        row = ttk.Frame(self.rows_frame, padding=(12, 9), style="SoftCard.TFrame")
        row.pack(fill="x", pady=(0, 9))
        number = ttk.Label(row, width=3, anchor="center", style="SoftCardTitle.TLabel")
        number.pack(side="left", padx=(0, 8))
        variable = tk.StringVar()
        entry = ttk.Entry(row, textvariable=variable, font=("Segoe UI", 12))
        entry.pack(side="left", fill="x", expand=True)
        remove = ttk.Button(row, text="刪除", style="Danger.TButton")
        remove.pack(side="right", padx=(8, 0))
        record: dict[str, object] = {
            "frame": row,
            "number": number,
            "variable": variable,
            "entry": entry,
            "remove": remove,
        }
        self.rows.append(record)
        remove.configure(command=lambda current=record: self._remove_row(current))
        variable.trace_add("write", lambda *_args, current=record: self._row_changed(current))
        entry.bind("<Return>", lambda _event, current=record: self._focus_next(current))
        self._renumber_rows()
        self._update_count()
        self._resize_for_rows()
        if focus:
            entry.focus_set()

    def _row_changed(self, record: dict[str, object]) -> None:
        variable = record["variable"]
        assert isinstance(variable, tk.StringVar)
        if self.rows and record is self.rows[-1] and variable.get().strip():
            self._add_row()
        self._update_count()

    def _remove_row(self, record: dict[str, object]) -> None:
        if record not in self.rows:
            return
        frame = record["frame"]
        assert isinstance(frame, ttk.Frame)
        frame.destroy()
        self.rows.remove(record)
        if not self.rows or self._row_value(self.rows[-1]):
            self._add_row()
        self._renumber_rows()
        self._update_count()
        self._resize_for_rows()

    def _focus_next(self, record: dict[str, object]) -> str:
        try:
            index = self.rows.index(record)
        except ValueError:
            return "break"
        if index == len(self.rows) - 1:
            self._add_row()
        next_entry = self.rows[min(index + 1, len(self.rows) - 1)]["entry"]
        assert isinstance(next_entry, ttk.Entry)
        next_entry.focus_set()
        return "break"

    def _renumber_rows(self) -> None:
        for index, record in enumerate(self.rows, start=1):
            number = record["number"]
            assert isinstance(number, ttk.Label)
            number.configure(text=str(index))
            remove = record["remove"]
            assert isinstance(remove, ttk.Button)
            only_empty_row = len(self.rows) == 1 and not self._row_value(record)
            remove.configure(state="disabled" if only_empty_row else "normal")

    def _update_count(self) -> None:
        count = sum(bool(self._row_value(record)) for record in self.rows)
        self.count_var.set(f"已加入 {count} 個連結")
        self.done_button.configure(state="normal" if count else "disabled")
        self.after_idle(lambda: self.canvas.yview_moveto(1.0))

    @staticmethod
    def _row_value(record: dict[str, object]) -> str:
        variable = record["variable"]
        assert isinstance(variable, tk.StringVar)
        return variable.get().strip()

    def _refresh_scroll_region(self, _event: object = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _scroll_rows(self, event: tk.Event) -> str:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _resize_for_rows(self) -> None:
        visible_rows = min(max(len(self.rows), 1), 4)
        self.canvas.configure(height=72 + (visible_rows - 1) * 60)

    def _resize_rows_frame(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.rows_window, width=event.width)

    def _accept(self) -> None:
        values = [self._row_value(record) for record in self.rows]
        self.value = list(dict.fromkeys(value for value in values if value))
        accepted = self.value
        self._close()
        if accepted:
            self.parent_app._start_url_expansion(accepted)

    def _close(self) -> None:
        self.parent_app._close_embedded_page(self)


def main() -> None:
    app = ExamCueApp()
    app.mainloop()


if __name__ == "__main__":
    main()
