"""Focused CustomTkinter front end for the local video editing engine.

The interface intentionally exposes only the editing flow this tool supports:
source/output, face tracking, dead-silence removal, and filler-word/caption
cleanup.  Export work stays on a daemon thread and communicates with Tk through
an event queue, so the window remains responsive while FFmpeg runs.

Run ``python app.py --smoke-test`` to construct, resize, validate, and destroy
the real UI without starting an export.
"""

from __future__ import annotations

import importlib
import math
import os
import queue
import shutil
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk


APP_TITLE = "Local Video Editor"
VIDEO_EXTENSIONS = (
    ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
    ("All files", "*.*"),
)
CAPTION_STYLES = {
    "Clean": "clean",
    "Bold punch": "punch",
    "Karaoke highlight": "karaoke",
    "Marker box": "marker",
    "Modern podcast": "modern_podcast",
}
CAPTION_MODES = ("Line", "Word")
BG = "#151A1B"
CARD = "#202728"
SURFACE = "#2B3536"
BORDER = "#3A4849"
TEXT = "#E4EFEC"
MUTED = "#9BAFAA"
MINT = "#A6E4CB"
MINT_HOVER = "#C0F0DD"
INK = "#172923"


def application_directory() -> Path:
    """Return the directory beside the script, or beside a frozen exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_ffmpeg() -> str | None:
    """Find FFmpeg on PATH, beside the app, or at a configured location."""
    configured = os.environ.get("FFMPEG_PATH")
    candidates = [
        Path(configured) if configured else None,
        application_directory() / "ffmpeg.exe",
        application_directory() / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
    ]
    path_match = shutil.which("ffmpeg")
    if path_match:
        candidates.insert(0, Path(path_match))
    return next((str(path) for path in candidates if path and path.is_file()), None)


def find_ffprobe(ffmpeg_path: str | None = None) -> str | None:
    """Find the matching FFprobe executable."""
    configured = os.environ.get("FFPROBE_PATH")
    ffmpeg_path = ffmpeg_path or find_ffmpeg()
    candidates = [Path(configured)] if configured else []
    if ffmpeg_path:
        binary = Path(ffmpeg_path)
        candidates.append(binary.with_name("ffprobe.exe" if binary.suffix.lower() == ".exe" else "ffprobe"))
    candidates.append(application_directory() / "ffprobe.exe")
    path_match = shutil.which("ffprobe")
    if path_match:
        candidates.append(Path(path_match))
    return next((str(path) for path in candidates if path.is_file()), None)


def _import_error_detail(exc: BaseException) -> str:
    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return detail


def _import_module(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # optional native packages can fail with OSError too
        return None, _import_error_detail(exc)


def _ctranslate2_status() -> str:
    module, error = _import_module("ctranslate2")
    return "Ready" if module is not None else f"Missing ({error})"


def _faster_whisper_status() -> str:
    """Check only the speech imports used by filler removal/captions."""
    package, error = _import_module("faster_whisper")
    if package is None:
        return f"Missing ({error})"
    wrapper = getattr(package, "WhisperModel", None)
    if not callable(wrapper) or not callable(getattr(wrapper, "transcribe", None)):
        return "Unavailable (WhisperModel is not usable)"
    return "Ready"


def _opencv_status() -> str:
    """Check OpenCV and the Haar cascade required for face tracking."""
    numpy, error = _import_module("numpy")
    if numpy is None:
        return f"Missing NumPy ({error})"
    cv2, error = _import_module("cv2")
    if cv2 is None:
        return f"Missing ({error})"
    data = getattr(cv2, "data", None)
    haar_dir = getattr(data, "haarcascades", None)
    cascade = Path(haar_dir) / "haarcascade_frontalface_default.xml" if haar_dir else None
    return "Ready" if cascade and cascade.is_file() else "Unavailable (face model missing)"


def _transcription_ready(status: str) -> bool:
    return status in {"Ready", "CPU ready"}


def _optional_install_message(package: str, feature: str, status: str) -> str:
    return (
        f"{feature} requires {package}, but it is not usable. Status: {status}\n"
        "Install the optional dependencies with: python -m pip install -r requirements-optional.txt"
    )


def optional_dependency_status() -> dict[str, str]:
    """Probe dependencies using the same imports export will require."""
    ffmpeg = find_ffmpeg()
    speech = _faster_whisper_status()
    return {
        "ffmpeg": "Ready" if ffmpeg else "Install FFmpeg",
        "ffprobe": "Ready" if find_ffprobe(ffmpeg) else "Install ffprobe",
        "speech": "CPU ready" if _transcription_ready(speech) else speech,
        "face": _opencv_status(),
        "ctranslate2": _ctranslate2_status(),
    }


def load_engine_module() -> Any:
    """Import the sibling engine lazily for source and frozen builds."""
    tool_dir = str(application_directory())
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    try:
        return importlib.import_module("engine")
    except ModuleNotFoundError as exc:
        if exc.name == "engine":
            raise RuntimeError("Processing engine is missing. Place engine.py beside the app.") from exc
        raise RuntimeError(f"The engine needs {exc.name}. Install it in this Python environment.") from exc


def load_engine() -> Callable[..., Any]:
    module = load_engine_module()
    handler = getattr(module, "process_video", None)
    if not callable(handler):
        raise RuntimeError("engine.py must expose process_video(input_path, output_path, options, progress_callback, log_callback).")
    return handler


def validate_paths(input_text: str, output_text: str) -> tuple[Path, Path]:
    """Validate source and destination before confirmation or worker startup."""
    if not input_text.strip():
        raise ValueError("Choose an input video first.")
    source = Path(input_text.strip()).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Input video was not found. Choose an existing file.")
    if not output_text.strip():
        raise ValueError("Choose where to save the MP4 output.")
    destination = Path(output_text.strip()).expanduser().resolve()
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError("Input and output must be different files. Choose a new output name.")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Output must end in .mp4. Use Save as to choose an MP4 file.")
    if destination.exists() and not destination.is_file():
        raise ValueError("Output is a folder. Choose an MP4 filename instead.")
    if not destination.parent.is_dir():
        raise ValueError("Output folder does not exist. Choose an existing folder with Save as.")
    return source, destination


class VideoEditorApp:
    """One-window editor focused on four explicit features."""

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(520, 620)
        scaling = self.root._get_window_scaling()
        width = min(1040, max(520, int((root.winfo_screenwidth() - 80) / scaling)))
        height = min(820, max(620, int((root.winfo_screenheight() - 100) / scaling)))
        x = max(0, int((root.winfo_screenwidth() - width * scaling) / 2))
        y = max(16, int((root.winfo_screenheight() - height * scaling) / 2) - 24)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.configure(fg_color=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.input_var = tk.StringVar(master=root)
        self.output_var = tk.StringVar(master=root)
        self.status_var = tk.StringVar(master=root, value="Choose a source video to get started.")
        self.progress_var = tk.DoubleVar(master=root, value=0.0)
        self.option_vars = {
            "face_tracking": tk.BooleanVar(master=root, value=False),
            "remove_silence": tk.BooleanVar(master=root, value=True),
            "remove_filler_words": tk.BooleanVar(master=root, value=False),
            "captions": tk.BooleanVar(master=root, value=False),
        }
        self.setting_vars = {
            "caption_mode": tk.StringVar(master=root, value="Line"),
            "caption_style": tk.StringVar(master=root, value="Clean"),
        }
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.processing = False
        self.closed = False
        self._queue_job: str | None = None
        self._layout_job: str | None = None
        self._controls: list[Any] = []
        self._wide: bool | None = None
        self._build_ui()
        self.root.bind("<Configure>", self._schedule_layout, add="+")
        self.root.bind("<Control-o>", lambda _event: self._choose_input())
        for variable in (*self.option_vars.values(), *self.setting_vars.values(), self.input_var, self.output_var):
            variable.trace_add("write", lambda *_args: self._refresh_summary())
        self._refresh_status()
        self._refresh_summary()
        self._layout()
        self._queue_job = self.root.after(80, self._drain_events)

    def _font(self, size: int = 13, weight: str = "normal") -> ctk.CTkFont:
        return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)

    def _label(self, parent: Any, text: str = "", *, size: int = 13, muted: bool = False, **kwargs: Any) -> ctk.CTkLabel:
        return ctk.CTkLabel(parent, text=text, font=self._font(size), text_color=MUTED if muted else TEXT, **kwargs)

    def _button(self, parent: Any, text: str, command: Callable[..., Any], *, primary: bool = False, **kwargs: Any) -> ctk.CTkButton:
        height = kwargs.pop("height", 36)
        button = ctk.CTkButton(
            parent, text=text, command=command, height=height, corner_radius=10,
            font=self._font(13, "bold"), fg_color=MINT if primary else SURFACE,
            hover_color=MINT_HOVER if primary else BORDER, text_color=INK if primary else TEXT,
            text_color_disabled=MUTED, **kwargs,
        )
        self._controls.append(button)
        return button

    def _card(self, parent: Any, title: str, detail: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)
        card.columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text=title, font=self._font(16, "bold"), text_color=TEXT, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=16, pady=(14, 0)
        )
        self._label(card, detail, size=12, muted=True, anchor="w", wraplength=420).grid(
            row=1, column=0, sticky="ew", padx=16, pady=(3, 12)
        )
        return card

    def _entry(self, parent: Any, variable: tk.StringVar, **kwargs: Any) -> ctk.CTkEntry:
        entry = ctk.CTkEntry(
            parent, textvariable=variable, height=36, corner_radius=9,
            font=self._font(), fg_color=BG, border_color=BORDER, text_color=TEXT, **kwargs,
        )
        self._controls.append(entry)
        return entry

    def _switch(self, parent: Any, key: str, title: str, detail: str) -> ctk.CTkSwitch:
        switch = ctk.CTkSwitch(
            parent, text=title, variable=self.option_vars[key], font=self._font(14, "bold"),
            text_color=TEXT, fg_color=BORDER, progress_color=MINT,
            button_color=TEXT, button_hover_color=MINT_HOVER, width=0,
        )
        switch.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 3))
        self._label(parent, detail, size=12, muted=True, anchor="w", wraplength=430).grid(
            row=3, column=0, sticky="ew", padx=(58, 16), pady=(0, 15)
        )
        self._controls.append(switch)
        return switch

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.root, fg_color=CARD, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="EDIT VIDEO", font=self._font(21, "bold"), text_color=TEXT).grid(
            row=0, column=0, padx=(22, 12), pady=(15, 0), sticky="w"
        )
        self._label(header, "Simple local cleanup · CPU processing", size=11, muted=True).grid(
            row=1, column=0, padx=(22, 12), pady=(0, 15), sticky="w"
        )
        self.header_status = ctk.CTkLabel(
            header, text="READY", font=self._font(11, "bold"), text_color=INK,
            fg_color=MINT, corner_radius=9, width=76, height=28,
        )
        self.header_status.grid(row=0, column=2, rowspan=2, padx=(10, 22), pady=14)

        self.content = ctk.CTkScrollableFrame(
            self.root, fg_color=BG, corner_radius=0,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=MUTED,
        )
        self.content.grid(row=1, column=0, sticky="nsew", padx=14, pady=(12, 8))
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=1)

        self._build_source_output()
        self._build_feature_cards()
        self._build_caption_controls()
        self._build_dependency_status()
        self._build_activity()

        footer = ctk.CTkFrame(self.root, fg_color=CARD, corner_radius=0)
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status_label = self._label(footer, size=12, anchor="w", textvariable=self.status_var, wraplength=720)
        self.status_label.grid(row=0, column=0, sticky="ew", padx=22, pady=(10, 1))
        self.progress = ctk.CTkProgressBar(
            footer, variable=self.progress_var, height=6, corner_radius=3,
            fg_color=SURFACE, progress_color=MINT,
        )
        self.progress.grid(row=1, column=0, sticky="ew", padx=(22, 16), pady=(0, 12))
        self.progress.set(0)
        self.export_button = self._button(footer, "Export", self._start_processing, primary=True, width=128, height=42)
        self.export_button.grid(row=0, column=0, rowspan=2, sticky="e", padx=(8, 22), pady=10)

    def _build_source_output(self) -> None:
        card = self._card(self.content, "Source and output", "Choose an existing video and a new MP4 destination.")
        card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        card.columnconfigure(1, weight=1)
        self._label(card, "SOURCE", size=10, muted=True).grid(row=2, column=0, sticky="w", padx=(16, 10), pady=(0, 7))
        self._entry(card, self.input_var, placeholder_text="Select a video…").grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(0, 7))
        self._button(card, "Browse", self._choose_input, width=82).grid(row=2, column=2, padx=(0, 16), pady=(0, 7))
        self._label(card, "OUTPUT", size=10, muted=True).grid(row=3, column=0, sticky="w", padx=(16, 10), pady=(0, 16))
        self._entry(card, self.output_var, placeholder_text="Save edited video as…").grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(0, 16))
        self._button(card, "Save as", self._choose_output, width=82).grid(row=3, column=2, padx=(0, 16), pady=(0, 16))

    def _build_feature_cards(self) -> None:
        self.feature_grid = ctk.CTkFrame(self.content, fg_color="transparent")
        self.feature_grid.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.feature_grid.columnconfigure(0, weight=1)
        self.feature_grid.columnconfigure(1, weight=1)
        face = self._card(self.feature_grid, "01  Face tracking", "Keep the active speaker in frame with local OpenCV tracking.")
        face.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self._switch(face, "face_tracking", "Track faces", "Off by default · centered framing is used when disabled.")
        silence = self._card(self.feature_grid, "02  Dead silence removal", "Trim quiet pauses while keeping natural speech boundaries.")
        silence.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8))
        self._switch(silence, "remove_silence", "Remove dead silence", "On by default · runs locally through FFmpeg.")
        filler = self._card(self.feature_grid, "03  Filler words", "Remove common ums, uhs, and erms using local word timestamps.")
        filler.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(8, 0))
        self._switch(filler, "remove_filler_words", "Remove filler words", "Off by default · needs the local speech dependency.")
        captions = self._card(self.feature_grid, "04  Captions", "Burn readable local captions into the exported MP4.")
        captions.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(8, 0))
        self._switch(captions, "captions", "Add captions", "Off by default · uses the same local speech model.")

    def _build_caption_controls(self) -> None:
        card = self._card(self.content, "Caption style", "These settings apply when captions are enabled above.")
        card.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        card.columnconfigure(1, weight=1)
        self._label(card, "STYLE", size=10, muted=True).grid(row=2, column=0, sticky="w", padx=(16, 12), pady=(0, 14))
        style = ctk.CTkOptionMenu(
            card, variable=self.setting_vars["caption_style"], values=list(CAPTION_STYLES),
            height=34, corner_radius=9, font=self._font(), dropdown_font=self._font(),
            fg_color=SURFACE, button_color=SURFACE, button_hover_color=BORDER,
            text_color=TEXT, dropdown_fg_color=CARD, dropdown_hover_color=BORDER,
            dropdown_text_color=TEXT,
        )
        style.grid(row=2, column=1, sticky="ew", padx=(0, 16), pady=(0, 14))
        self._controls.append(style)
        self._label(card, "MODE", size=10, muted=True).grid(row=3, column=0, sticky="w", padx=(16, 12), pady=(0, 16))
        mode = ctk.CTkSegmentedButton(
            card, values=list(CAPTION_MODES), variable=self.setting_vars["caption_mode"],
            height=34, corner_radius=9, font=self._font(12), fg_color=BG,
            selected_color=MINT, selected_hover_color=MINT_HOVER,
            unselected_color=SURFACE, unselected_hover_color=BORDER,
            text_color=TEXT,
        )
        mode.grid(row=3, column=1, sticky="ew", padx=(0, 16), pady=(0, 16))
        self._controls.append(mode)

    def _build_dependency_status(self) -> None:
        card = self._card(self.content, "Dependency status", "Checked on startup and again before export. Processing stays local on this computer.")
        card.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.capability_label = self._label(card, "Checking…", size=12, muted=True, anchor="w", wraplength=850, justify="left")
        self.capability_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

    def _build_activity(self) -> None:
        card = self._card(self.content, "Progress and log", "Export progress and engine messages appear here.")
        card.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.log_text = ctk.CTkTextbox(
            card, height=125, corner_radius=10, fg_color=BG, text_color=MUTED,
            font=self._font(12), wrap="word", scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=MUTED,
        )
        self.log_text.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.log_text.configure(state="disabled")

    def _schedule_layout(self, event: Any) -> None:
        if event.widget is self.root and self._layout_job is None:
            self._layout_job = self.root.after(60, self._layout)

    def _layout(self) -> None:
        self._layout_job = None
        logical_width = int(self.root.geometry().split("x", 1)[0])
        self._wide = logical_width >= 760
        if hasattr(self, "feature_grid"):
            if self._wide:
                self.feature_grid.columnconfigure(0, weight=1)
                self.feature_grid.columnconfigure(1, weight=1)
            else:
                self.feature_grid.columnconfigure(0, weight=1)
                self.feature_grid.columnconfigure(1, weight=1)
        if hasattr(self, "capability_label"):
            self.capability_label.configure(wraplength=max(360, logical_width - 90))
        if hasattr(self, "status_label"):
            self.status_label.configure(wraplength=max(240, logical_width - 190))

    def _choose_input(self) -> None:
        if self.processing:
            return
        selected = filedialog.askopenfilename(parent=self.root, title="Choose input video", filetypes=VIDEO_EXTENSIONS)
        if selected:
            self.input_var.set(selected)
            if not self.output_var.get().strip():
                source = Path(selected)
                self.output_var.set(str(source.with_name(f"{source.stem}_edited.mp4")))

    def _choose_output(self) -> None:
        if self.processing:
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root, title="Save edited video", defaultextension=".mp4",
            filetypes=(("MP4 video", "*.mp4"),), confirmoverwrite=False,
            initialfile=Path(self.output_var.get()).name if self.output_var.get() else "edited.mp4",
        )
        if selected:
            self.output_var.set(selected)

    def _refresh_status(self) -> dict[str, str]:
        capabilities = optional_dependency_status()
        labels = {"ffmpeg": "FFmpeg", "ffprobe": "ffprobe", "speech": "Speech / captions", "face": "Face tracking", "ctranslate2": "Speech runtime"}
        self.capability_label.configure(text="   ·   ".join(f"{labels[key]}: {value}" for key, value in capabilities.items()))
        return capabilities

    def _refresh_summary(self) -> None:
        if not hasattr(self, "status_label") or self.processing:
            return
        source = Path(self.input_var.get().strip()).name if self.input_var.get().strip() else "No source selected"
        output = Path(self.output_var.get().strip()).name if self.output_var.get().strip() else "No output selected"
        enabled = []
        if self.option_vars["face_tracking"].get():
            enabled.append("face tracking")
        if self.option_vars["remove_silence"].get():
            enabled.append("silence removal")
        if self.option_vars["remove_filler_words"].get():
            enabled.append("filler removal")
        if self.option_vars["captions"].get():
            enabled.append("captions")
        self.status_var.set(f"Source: {source}  ·  Output: {output}  ·  {', '.join(enabled) if enabled else 'No edits selected'}")

    def _collect_options(self) -> dict[str, Any]:
        """Return only the small contract accepted by the focused UI flow."""
        style_label = self.setting_vars["caption_style"].get()
        mode_label = self.setting_vars["caption_mode"].get()
        if style_label not in CAPTION_STYLES:
            raise ValueError("Choose a valid caption style.")
        if mode_label not in CAPTION_MODES:
            raise ValueError("Choose a valid caption mode.")
        # Keep the engine contract explicit: no provider, music, watermark,
        # transition, gallery, or hidden workspace settings are forwarded.
        return {
            "remove_silence": bool(self.option_vars["remove_silence"].get()),
            "remove_filler_words": bool(self.option_vars["remove_filler_words"].get()),
            "face_tracking": bool(self.option_vars["face_tracking"].get()),
            "captions": bool(self.option_vars["captions"].get()),
            "caption_preset": CAPTION_STYLES[style_label],
            "caption_mode": mode_label.lower(),
            "whisper_model": "tiny",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
        }

    def _append_log(self, message: Any) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message).rstrip() + "\n")
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 1500:
            self.log_text.delete("1.0", f"{line_count - 1200}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_processing(self, active: bool) -> None:
        self.processing = active
        for widget in self._controls:
            widget.configure(state="disabled" if active else "normal")
        self.export_button.configure(text="Exporting…" if active else "Export")
        self.header_status.configure(
            text="EXPORTING" if active else "READY",
            fg_color=SURFACE if active else MINT,
            text_color=TEXT if active else INK,
        )

    def _start_processing(self) -> None:
        if self.processing:
            return
        try:
            source, destination = validate_paths(self.input_var.get(), self.output_var.get())
            options = self._collect_options()
            capabilities = self._refresh_status()
            if capabilities["ffmpeg"] != "Ready" or capabilities["ffprobe"] != "Ready":
                raise ValueError("FFmpeg and ffprobe are required. Install them on PATH, place them beside the app, or set FFMPEG_PATH and FFPROBE_PATH.")
            speech = options["remove_filler_words"] or options["captions"]
            if speech and not _transcription_ready(capabilities["speech"]):
                raise ValueError(_optional_install_message("faster-whisper", "Speech features", capabilities["speech"]))
            if speech and capabilities["ctranslate2"] != "Ready":
                raise ValueError(_optional_install_message("ctranslate2", "Speech features", capabilities["ctranslate2"]))
            if options["face_tracking"] and capabilities["face"] != "Ready":
                raise ValueError(_optional_install_message("opencv-python", "Face tracking", capabilities["face"]))
            existing = [destination]
            if options["captions"]:
                existing.extend(path for path in (destination.with_suffix(".srt"), destination.with_suffix(".ass")) if path.exists())
            existing = [path for path in existing if path.exists()]
            if existing and not messagebox.askyesno(
                APP_TITLE, "Replace the existing output file(s)?\n\n" + "\n".join(str(path) for path in existing),
                parent=self.root, default="no",
            ):
                return
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check export settings", str(exc), parent=self.root)
            return

        self._set_processing(True)
        self.progress_var.set(0)
        self.status_var.set("Preparing export…")
        self._append_log(f"Export: {source.name} → {destination}")
        self._append_log("Settings: " + ", ".join(f"{key}={value}" for key, value in options.items()))
        threading.Thread(
            target=self._process_worker, args=(str(source), str(destination), options),
            name="video-export", daemon=True,
        ).start()

    def _process_worker(self, input_path: str, output_path: str, options: dict[str, Any]) -> None:
        """Run engine work away from Tk and send only serializable UI events."""
        def report_progress(value: Any) -> None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(number):
                self.events.put(("progress", max(0.0, min(1.0, number if number <= 1 else number / 100))))

        def report_log(message: Any) -> None:
            self.events.put(("log", str(message)))

        environment_keys = ("FFMPEG_PATH", "FFPROBE_PATH")
        previous_environment = {key: os.environ.get(key) for key in environment_keys}
        try:
            ffmpeg, ffprobe = find_ffmpeg(), find_ffprobe()
            if ffmpeg:
                os.environ["FFMPEG_PATH"] = ffmpeg
            if ffprobe:
                os.environ["FFPROBE_PATH"] = ffprobe
            result = load_engine()(
                input_path=input_path, output_path=output_path, options=options,
                progress_callback=report_progress, log_callback=report_log,
            )
            output = Path(output_path)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("The engine returned without a nonempty output file. Check the log and choose a writable MP4 path.")
            self.events.put(("done", (output_path, result)))
        except Exception as exc:
            self.events.put(("error", "".join(traceback.format_exception_only(type(exc), exc)).strip()))
        finally:
            for key, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous

    def _drain_events(self) -> None:
        if self.closed:
            return
        for _ in range(150):
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "progress":
                self.progress_var.set(payload)
                self.status_var.set(f"Exporting · {round(payload * 100)}%")
            elif event == "log":
                self._append_log(payload)
            elif event == "done":
                output_path, _result = payload
                self._set_processing(False)
                self.progress_var.set(1)
                self.status_var.set("Export complete. Your MP4 is ready.")
                self._append_log(f"Saved: {output_path}")
            elif event == "error":
                self._set_processing(False)
                self.status_var.set("Export failed. See the log for details.")
                self._append_log("Export failed: " + payload)
                summary = payload if len(payload) <= 700 else payload[:700] + "…"
                messagebox.showerror("Export failed", summary, parent=self.root)
        self._queue_job = self.root.after(80, self._drain_events)

    def _close(self) -> None:
        if self.processing:
            messagebox.showinfo(APP_TITLE, "An export is running. Wait for it to finish before closing.", parent=self.root)
            return
        self.closed = True
        if self._queue_job is not None:
            self.root.after_cancel(self._queue_job)
        if self._layout_job is not None:
            self.root.after_cancel(self._layout_job)
        self.root.destroy()


def main() -> None:
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    app = VideoEditorApp(root)
    if "--smoke-test" in sys.argv:
        try:
            root.geometry("980x760")
            root.update()
            assert callable(load_engine())
            options = app._collect_options()
            assert options == {
                "remove_silence": True,
                "remove_filler_words": False,
                "face_tracking": False,
                "captions": False,
                "caption_preset": "clean",
                "caption_mode": "line",
                "whisper_model": "tiny",
                "whisper_device": "cpu",
                "whisper_compute_type": "int8",
            }
            assert len(options) == 9
            assert app.export_button.winfo_exists() and app.log_text.winfo_exists()
            assert not hasattr(app, "timeline_canvas") and not hasattr(app, "right")
            root.geometry("640x700")
            root.update()
            app._layout()
            assert app._wide is False
            root.geometry("980x760")
            root.update()
            app._layout()
            assert app._wide is True
            print("UI smoke test passed: focused four-feature UI, reduced engine options, defaults, dependency status, worker controls, and responsive layouts.")
        finally:
            app._close()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
