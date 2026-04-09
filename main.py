"""
main.py  –  Contactless Human Identification using Ear Biometrics
            Real-time GUI powered by CustomTkinter + OpenCV + SVM
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
import os, sys, time, queue, logging

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from modules.ear_detector    import EarDetector
from modules.feature_extractor import FeatureExtractor
from modules.classifier      import EarClassifier
from modules.dataset_manager import DatasetManager
from modules.utils           import load_image, cv_to_pil, cv_to_pil_rgb, confidence_to_label

# ── appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── palette ───────────────────────────────────────────────────────────────────
C_BG   = "#0d0d1a"
C_CARD = "#1a1a2e"
C_ACCENT = "#4cc9f0"
C_GREEN  = "#06d6a0"
C_RED    = "#f72585"
C_PURPLE = "#7209b7"
C_TEXT   = "#e0e0e0"
C_SUBTEXT= "#8888aa"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEADER = ("Segoe UI", 14, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 10)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper Widgets
# ══════════════════════════════════════════════════════════════════════════════

class StatCard(ctk.CTkFrame):
    """Small card showing a number + label."""
    def __init__(self, parent, title, value="—", accent=C_ACCENT, **kw):
        super().__init__(parent, fg_color=C_CARD, corner_radius=12, **kw)
        self._val_var = tk.StringVar(value=value)
        ctk.CTkLabel(self, text=title, font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(pady=(14, 0))
        ctk.CTkLabel(self, textvariable=self._val_var, font=("Segoe UI", 28, "bold"),
                     text_color=accent).pack(pady=(2, 14))

    def set(self, value): self._val_var.set(str(value))


class ConfidenceMeter(ctk.CTkFrame):
    """Animated confidence progress bar."""
    def __init__(self, parent, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        ctk.CTkLabel(self, text="Confidence", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w")
        self.bar = ctk.CTkProgressBar(self, height=16, corner_radius=8,
                                      progress_color=C_GREEN,
                                      fg_color="#222244")
        self.bar.pack(fill="x", pady=(4, 2))
        self.bar.set(0)
        self._pct_var = tk.StringVar(value="0 %")
        ctk.CTkLabel(self, textvariable=self._pct_var, font=FONT_SMALL,
                     text_color=C_TEXT).pack(anchor="e")

    def set(self, value: float):
        self.bar.set(value)
        self._pct_var.set(f"{value*100:.1f} %")
        colour = C_GREEN if value >= 0.75 else (C_ACCENT if value >= 0.5 else C_RED)
        self.bar.configure(progress_color=colour)


class LogBox(ctk.CTkTextbox):
    """Thread-safe scrolling log window."""
    def __init__(self, parent, **kw):
        kw.setdefault("state", "disabled")
        kw.setdefault("fg_color", "#0a0a18")
        kw.setdefault("text_color", C_TEXT)
        kw.setdefault("font", FONT_MONO)
        super().__init__(parent, **kw)

    def append(self, msg: str):
        self.configure(state="normal")
        self.insert("end", msg + "\n")
        self.see("end")
        self.configure(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application Window
# ══════════════════════════════════════════════════════════════════════════════

class EarBiometricsApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Ear Biometrics Identification System")
        self.geometry("1280x780")
        self.minsize(1100, 680)
        self.configure(fg_color=C_BG)

        # ── shared backend objects ────────────────────────────────────────────
        self.detector   = EarDetector()
        self.extractor  = FeatureExtractor()
        self.classifier = EarClassifier()

        # ── state ────────────────────────────────────────────────────────────
        self._cam_running   = False
        self._cam_thread    = None
        self._cam_cap       = None
        self._frame_queue   = queue.Queue(maxsize=2)
        self._result_queue  = queue.Queue(maxsize=10)
        self._train_thread  = None
        self._stats         = {"classes": 0, "images": 0, "accuracy": "—", "model": "Not Trained"}
        self._dataset_root  = tk.StringVar(value="")
        self._identify_path = tk.StringVar(value="")

        # ── layout ───────────────────────────────────────────────────────────
        self._build_sidebar()
        self._build_content()
        self._show_tab("dashboard")

        # auto-load model if exists
        if self.classifier.load():
            self._stats["model"]   = "Loaded"
            self._stats["classes"] = len(self.classifier.class_names)
            self._refresh_stats()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    #  Sidebar
    # ══════════════════════════════════════════════════════════════════════════

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=220, fg_color=C_CARD, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # Logo / title
        ctk.CTkLabel(sb, text="👂", font=("Segoe UI", 36)).pack(pady=(28, 4))
        ctk.CTkLabel(sb, text="EarBio ID", font=("Segoe UI", 18, "bold"),
                     text_color=C_ACCENT).pack()
        ctk.CTkLabel(sb, text="Biometric System", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(pady=(0, 28))

        ctk.CTkFrame(sb, height=1, fg_color="#333355").pack(fill="x", padx=20, pady=4)

        nav_items = [
            ("🏠  Dashboard",       "dashboard"),
            ("🎓  Train Model",     "train"),
            ("📷  Live Camera",     "live"),
            ("🔍  Identify Image",  "identify"),
            ("📊  Reports",         "reports"),
        ]

        self._nav_buttons = {}
        for label, tab in nav_items:
            btn = ctk.CTkButton(
                sb, text=label, anchor="w",
                fg_color="transparent", hover_color="#2a2a4a",
                text_color=C_TEXT, font=FONT_BODY,
                height=44, corner_radius=8,
                command=lambda t=tab: self._show_tab(t),
            )
            btn.pack(fill="x", padx=12, pady=2)
            self._nav_buttons[tab] = btn

        ctk.CTkFrame(sb, height=1, fg_color="#333355").pack(fill="x", padx=20, pady=16)
        ctk.CTkLabel(sb, text="v1.0  |  LBP+HOG+SVM", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(side="bottom", pady=16)

    # ══════════════════════════════════════════════════════════════════════════
    #  Content area + tab switcher
    # ══════════════════════════════════════════════════════════════════════════

    def _build_content(self):
        self._content = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self._content.pack(side="right", fill="both", expand=True)

        self._tab_dashboard = self._make_dashboard_tab()
        self._tab_train     = self._make_train_tab()
        self._tab_live      = self._make_live_tab()
        self._tab_identify  = self._make_identify_tab()
        self._tab_reports   = self._make_reports_tab()

        self._all_tabs = {
            "dashboard": self._tab_dashboard,
            "train":     self._tab_train,
            "live":      self._tab_live,
            "identify":  self._tab_identify,
            "reports":   self._tab_reports,
        }

    def _show_tab(self, name: str):
        # stop camera if leaving live tab
        if name != "live" and self._cam_running:
            self._stop_camera()

        for tab_name, frame in self._all_tabs.items():
            frame.pack_forget()

        self._all_tabs[name].pack(fill="both", expand=True)

        # highlight nav button
        for tab, btn in self._nav_buttons.items():
            btn.configure(
                fg_color=C_ACCENT if tab == name else "transparent",
                text_color=C_BG   if tab == name else C_TEXT,
            )

    # ══════════════════════════════════════════════════════════════════════════
    #  DASHBOARD TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _make_dashboard_tab(self):
        f = ctk.CTkFrame(self._content, fg_color=C_BG)

        ctk.CTkLabel(f, text="Dashboard", font=FONT_TITLE,
                     text_color=C_ACCENT).pack(anchor="w", padx=32, pady=(28, 4))
        ctk.CTkLabel(f, text="Contactless Human Identification using Ear Biometrics",
                     font=FONT_BODY, text_color=C_SUBTEXT).pack(anchor="w", padx=32)

        ctk.CTkFrame(f, height=1, fg_color="#333355").pack(fill="x", padx=32, pady=16)

        # stat cards
        cards_row = ctk.CTkFrame(f, fg_color="transparent")
        cards_row.pack(fill="x", padx=32, pady=4)

        self._card_classes  = StatCard(cards_row, "Classes", accent=C_ACCENT)
        self._card_images   = StatCard(cards_row, "Training Images", accent=C_GREEN)
        self._card_accuracy = StatCard(cards_row, "Test Accuracy", accent=C_PURPLE)
        self._card_model    = StatCard(cards_row, "Model Status", accent=C_RED)

        for card in [self._card_classes, self._card_images,
                     self._card_accuracy, self._card_model]:
            card.pack(side="left", fill="both", expand=True, padx=8, pady=4)

        # pipeline description
        info = ctk.CTkFrame(f, fg_color=C_CARD, corner_radius=12)
        info.pack(fill="x", padx=32, pady=24)

        ctk.CTkLabel(info, text="System Pipeline", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(anchor="w", padx=20, pady=(16,4))

        steps = [
            ("1. Ear Detection",      "Haar Cascade detects ear ROI from input frame."),
            ("2. Preprocessing",      "Resize to 128×128, CLAHE contrast enhancement."),
            ("3. LBP Extraction",     "Local Binary Pattern – captures texture info."),
            ("4. HOG Extraction",     "Histogram of Oriented Gradients – shape features."),
            ("5. Feature Fusion",     "LBP hist + HOG descriptor concatenated."),
            ("6. SVM Classification", "RBF-kernel SVM identifies the person with confidence."),
        ]

        for step, desc in steps:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(row, text=step, font=("Segoe UI", 11, "bold"),
                         text_color=C_TEXT, width=220, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=desc, font=FONT_SMALL,
                         text_color=C_SUBTEXT, anchor="w").pack(side="left")

        ctk.CTkFrame(info, fg_color="transparent", height=12).pack()

        # quick-action buttons
        btn_row = ctk.CTkFrame(f, fg_color="transparent")
        btn_row.pack(pady=16)
        ctk.CTkButton(btn_row, text="▶  Start Live Camera", width=200, height=42,
                      fg_color=C_GREEN, hover_color="#04a87f", text_color=C_BG,
                      font=("Segoe UI", 12, "bold"), corner_radius=10,
                      command=lambda: self._show_tab("live")).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="🎓  Train Model", width=160, height=42,
                      fg_color=C_PURPLE, hover_color="#5a0898", text_color=C_TEXT,
                      font=("Segoe UI", 12, "bold"), corner_radius=10,
                      command=lambda: self._show_tab("train")).pack(side="left", padx=8)

        return f

    def _refresh_stats(self):
        self._card_classes.set(self._stats["classes"])
        self._card_images.set(self._stats["images"])
        self._card_accuracy.set(self._stats["accuracy"])
        self._card_model.set(self._stats["model"])

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAIN TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _make_train_tab(self):
        f = ctk.CTkFrame(self._content, fg_color=C_BG)

        ctk.CTkLabel(f, text="Train Model", font=FONT_TITLE,
                     text_color=C_ACCENT).pack(anchor="w", padx=32, pady=(28, 4))
        ctk.CTkLabel(f, text="Load your ear dataset and train the SVM classifier.",
                     font=FONT_BODY, text_color=C_SUBTEXT).pack(anchor="w", padx=32)
        ctk.CTkFrame(f, height=1, fg_color="#333355").pack(fill="x", padx=32, pady=16)

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=32)

        # Left panel
        left = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12, width=400)
        left.pack(side="left", fill="y", padx=(0,12), pady=4)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Dataset Configuration", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(anchor="w", padx=20, pady=(18,10))

        # Dataset folder
        ctk.CTkLabel(left, text="Dataset Root Folder", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=20)
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(4,12))
        ctk.CTkEntry(row, textvariable=self._dataset_root,
                     placeholder_text="Select folder…",
                     font=FONT_SMALL, height=34).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="Browse", width=70, height=34,
                      fg_color=C_ACCENT, text_color=C_BG,
                      command=self._browse_dataset).pack(side="left", padx=(6,0))

        # Options
        ctk.CTkLabel(left, text="Test Split", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=20)
        self._test_split = ctk.CTkSlider(left, from_=0.1, to=0.4, number_of_steps=6)
        self._test_split.set(0.25)
        self._test_split.pack(fill="x", padx=20, pady=(4,4))
        self._test_split_lbl = ctk.CTkLabel(left, text="25 %", font=FONT_SMALL,
                                            text_color=C_TEXT)
        self._test_split_lbl.pack(anchor="e", padx=20)
        self._test_split.configure(command=lambda v: self._test_split_lbl.configure(
            text=f"{int(float(v)*100)} %"))

        self._augment_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Enable Augmentation", variable=self._augment_var,
                        font=FONT_BODY, text_color=C_TEXT,
                        checkmark_color=C_BG, fg_color=C_ACCENT).pack(anchor="w", padx=20, pady=8)

        self._grid_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(left, text="Auto Hyper-parameter Tuning (Grid Search)",
                        variable=self._grid_var,
                        font=FONT_BODY, text_color=C_TEXT,
                        checkmark_color=C_BG, fg_color=C_ACCENT).pack(anchor="w", padx=20, pady=4)

        ctk.CTkFrame(left, height=1, fg_color="#333355").pack(fill="x", padx=20, pady=16)

        self._train_btn = ctk.CTkButton(
            left, text="🎓  Start Training", height=44,
            fg_color=C_GREEN, hover_color="#04a87f", text_color=C_BG,
            font=("Segoe UI", 13, "bold"), corner_radius=10,
            command=self._start_training,
        )
        self._train_btn.pack(fill="x", padx=20, pady=8)

        self._train_progress = ctk.CTkProgressBar(left, height=10, corner_radius=5,
                                                   progress_color=C_ACCENT)
        self._train_progress.pack(fill="x", padx=20, pady=4)
        self._train_progress.set(0)

        self._train_status = ctk.CTkLabel(left, text="Idle", font=FONT_SMALL,
                                          text_color=C_SUBTEXT)
        self._train_status.pack(anchor="w", padx=20)

        # Right panel – log
        right = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        right.pack(side="right", fill="both", expand=True, pady=4)

        ctk.CTkLabel(right, text="Training Log", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(anchor="w", padx=20, pady=(18,8))
        self._train_log = LogBox(right, height=400)
        self._train_log.pack(fill="both", expand=True, padx=12, pady=(0,12))

        return f

    # ══════════════════════════════════════════════════════════════════════════
    #  LIVE CAMERA TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _make_live_tab(self):
        f = ctk.CTkFrame(self._content, fg_color=C_BG)

        hdr = ctk.CTkFrame(f, fg_color="transparent")
        hdr.pack(fill="x", padx=32, pady=(28,4))
        ctk.CTkLabel(hdr, text="Live Camera  —  Real-Time Ear Identification",
                     font=FONT_TITLE, text_color=C_ACCENT).pack(side="left")

        btn_row = ctk.CTkFrame(hdr, fg_color="transparent")
        btn_row.pack(side="right")
        self._cam_start_btn = ctk.CTkButton(
            btn_row, text="▶  Start Camera", width=150, height=36,
            fg_color=C_GREEN, hover_color="#04a87f", text_color=C_BG,
            font=("Segoe UI", 11, "bold"), corner_radius=8,
            command=self._toggle_camera,
        )
        self._cam_start_btn.pack(side="left", padx=4)

        # Camera index selector
        ctk.CTkLabel(btn_row, text="Cam:", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(side="left", padx=(10,2))
        self._cam_index = ctk.CTkOptionMenu(btn_row, values=["0","1","2"],
                                            width=60, height=36,
                                            fg_color=C_CARD, text_color=C_TEXT)
        self._cam_index.pack(side="left")

        ctk.CTkFrame(f, height=1, fg_color="#333355").pack(fill="x", padx=32, pady=12)

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=32, pady=4)

        # Feed frame
        feed_frame = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        feed_frame.pack(side="left", fill="both", expand=True, padx=(0,12))

        ctk.CTkLabel(feed_frame, text="Camera Feed", font=FONT_HEADER,
                     text_color=C_SUBTEXT).pack(pady=(12,4))

        self._cam_label = ctk.CTkLabel(feed_frame, text="Camera not started.",
                                       font=FONT_BODY, text_color=C_SUBTEXT,
                                       width=560, height=420)
        self._cam_label.pack(padx=12, pady=(0,12))

        # Results panel
        right = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Identification Result", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(pady=(18,8), padx=16, anchor="w")

        self._live_name_var = tk.StringVar(value="—")
        ctk.CTkLabel(right, textvariable=self._live_name_var,
                     font=("Segoe UI", 22, "bold"), text_color=C_GREEN,
                     wraplength=240).pack(pady=(4,4), padx=16)

        self._live_conf_meter = ConfidenceMeter(right)
        self._live_conf_meter.pack(fill="x", padx=16, pady=8)

        self._live_grade_var = tk.StringVar(value="")
        ctk.CTkLabel(right, textvariable=self._live_grade_var,
                     font=FONT_BODY, text_color=C_SUBTEXT).pack()

        ctk.CTkFrame(right, height=1, fg_color="#333355").pack(fill="x", padx=16, pady=12)

        # Top-3 list
        ctk.CTkLabel(right, text="Top-3 Candidates", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=16)
        self._top3_labels = []
        for _ in range(3):
            lbl = ctk.CTkLabel(right, text="", font=FONT_SMALL,
                               text_color=C_TEXT, anchor="w")
            lbl.pack(anchor="w", padx=20, pady=2)
            self._top3_labels.append(lbl)

        ctk.CTkFrame(right, height=1, fg_color="#333355").pack(fill="x", padx=16, pady=12)

        # FPS
        self._fps_var = tk.StringVar(value="FPS: 0")
        ctk.CTkLabel(right, textvariable=self._fps_var, font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=16)

        self._ear_detected_var = tk.StringVar(value="Ear: Not detected")
        ctk.CTkLabel(right, textvariable=self._ear_detected_var, font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=16, pady=(2,16))

        return f

    # ══════════════════════════════════════════════════════════════════════════
    #  IDENTIFY IMAGE TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _make_identify_tab(self):
        f = ctk.CTkFrame(self._content, fg_color=C_BG)

        ctk.CTkLabel(f, text="Identify from Image", font=FONT_TITLE,
                     text_color=C_ACCENT).pack(anchor="w", padx=32, pady=(28,4))
        ctk.CTkLabel(f, text="Load a static ear image and run identification.",
                     font=FONT_BODY, text_color=C_SUBTEXT).pack(anchor="w", padx=32)
        ctk.CTkFrame(f, height=1, fg_color="#333355").pack(fill="x", padx=32, pady=16)

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=32)

        # Left: image display
        left = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0,12), pady=4)

        ctk.CTkLabel(left, text="Input Image", font=FONT_HEADER,
                     text_color=C_SUBTEXT).pack(pady=(14,4))

        self._id_img_label = ctk.CTkLabel(left, text="No image loaded.",
                                          font=FONT_BODY, text_color=C_SUBTEXT,
                                          width=480, height=360)
        self._id_img_label.pack(padx=12)

        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(pady=12)
        ctk.CTkButton(row, text="📂  Browse Image", width=160, height=38,
                      fg_color=C_ACCENT, text_color=C_BG,
                      font=("Segoe UI", 11, "bold"),
                      command=self._browse_image).pack(side="left", padx=6)
        ctk.CTkButton(row, text="🔍  Identify", width=140, height=38,
                      fg_color=C_PURPLE, text_color=C_TEXT,
                      font=("Segoe UI", 11, "bold"),
                      command=self._run_identify).pack(side="left", padx=6)

        # Right panel
        right = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12, width=300)
        right.pack(side="right", fill="y", pady=4)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Result", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(pady=(18,6), padx=16, anchor="w")

        self._id_name_var = tk.StringVar(value="—")
        ctk.CTkLabel(right, textvariable=self._id_name_var,
                     font=("Segoe UI", 24, "bold"), text_color=C_GREEN,
                     wraplength=260).pack(padx=16, pady=(0,4))

        self._id_conf_meter = ConfidenceMeter(right)
        self._id_conf_meter.pack(fill="x", padx=16, pady=8)

        self._id_grade_var = tk.StringVar(value="")
        ctk.CTkLabel(right, textvariable=self._id_grade_var,
                     font=FONT_BODY, text_color=C_SUBTEXT).pack()

        ctk.CTkFrame(right, height=1, fg_color="#333355").pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(right, text="Feature Extraction", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=16)

        # LBP image preview
        ctk.CTkLabel(right, text="LBP Image:", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=20, pady=(8,2))
        self._lbp_preview = ctk.CTkLabel(right, text="", width=120, height=90)
        self._lbp_preview.pack()

        # HOG image preview
        ctk.CTkLabel(right, text="HOG Image:", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=20, pady=(8,2))
        self._hog_preview = ctk.CTkLabel(right, text="", width=120, height=90)
        self._hog_preview.pack()

        # Top-3
        ctk.CTkFrame(right, height=1, fg_color="#333355").pack(fill="x", padx=16, pady=10)
        ctk.CTkLabel(right, text="Top-3 Candidates", font=FONT_SMALL,
                     text_color=C_SUBTEXT).pack(anchor="w", padx=16)
        self._id_top3 = []
        for _ in range(3):
            lbl = ctk.CTkLabel(right, text="", font=FONT_SMALL,
                               text_color=C_TEXT, anchor="w")
            lbl.pack(anchor="w", padx=20, pady=2)
            self._id_top3.append(lbl)

        return f

    # ══════════════════════════════════════════════════════════════════════════
    #  REPORTS TAB
    # ══════════════════════════════════════════════════════════════════════════

    def _make_reports_tab(self):
        f = ctk.CTkFrame(self._content, fg_color=C_BG)

        ctk.CTkLabel(f, text="Reports & Analytics", font=FONT_TITLE,
                     text_color=C_ACCENT).pack(anchor="w", padx=32, pady=(28,4))
        ctk.CTkLabel(f, text="Performance metrics generated after training.",
                     font=FONT_BODY, text_color=C_SUBTEXT).pack(anchor="w", padx=32)
        ctk.CTkFrame(f, height=1, fg_color="#333355").pack(fill="x", padx=32, pady=16)

        # Metric cards
        m_row = ctk.CTkFrame(f, fg_color="transparent")
        m_row.pack(fill="x", padx=32, pady=4)
        self._rpt_acc   = StatCard(m_row, "Overall Accuracy", accent=C_GREEN)
        self._rpt_prec  = StatCard(m_row, "Macro Precision",  accent=C_ACCENT)
        self._rpt_rec   = StatCard(m_row, "Macro Recall",     accent=C_PURPLE)
        self._rpt_f1    = StatCard(m_row, "Macro F1-Score",   accent=C_RED)
        for c in [self._rpt_acc, self._rpt_prec, self._rpt_rec, self._rpt_f1]:
            c.pack(side="left", fill="both", expand=True, padx=8, pady=4)

        body = ctk.CTkFrame(f, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=32, pady=8)

        # Confusion matrix canvas area
        self._cm_frame = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        self._cm_frame.pack(side="left", fill="both", expand=True, padx=(0,8))

        ctk.CTkLabel(self._cm_frame, text="Confusion Matrix", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(pady=(14,4))
        self._cm_img_label = ctk.CTkLabel(self._cm_frame, text="No report yet.",
                                          font=FONT_BODY, text_color=C_SUBTEXT)
        self._cm_img_label.pack(pady=20, padx=16)

        # Per-class bar chart area
        self._bar_frame = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        self._bar_frame.pack(side="right", fill="both", expand=True, padx=(8,0))

        ctk.CTkLabel(self._bar_frame, text="Per-Class Metrics", font=FONT_HEADER,
                     text_color=C_ACCENT).pack(pady=(14,4))
        self._bar_img_label = ctk.CTkLabel(self._bar_frame, text="No report yet.",
                                           font=FONT_BODY, text_color=C_SUBTEXT)
        self._bar_img_label.pack(pady=20, padx=16)

        return f

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAINING LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Root Folder")
        if path:
            self._dataset_root.set(path)

    def _start_training(self):
        root = self._dataset_root.get().strip()
        if not root or not os.path.isdir(root):
            messagebox.showerror("Error", "Please select a valid dataset folder.")
            return
        if self._train_thread and self._train_thread.is_alive():
            messagebox.showinfo("Info", "Training is already running.")
            return
        self._train_btn.configure(state="disabled", text="Training…")
        self._train_progress.set(0)
        self._train_log.configure(state="normal")
        self._train_log.delete("1.0", "end")
        self._train_log.configure(state="disabled")
        self._train_thread = threading.Thread(target=self._train_worker,
                                              args=(root,), daemon=True)
        self._train_thread.start()

    def _train_worker(self, dataset_root: str):
        def log(msg):
            self.after(0, lambda m=msg: self._train_log.append(m))
        def set_status(msg, prog=None):
            self.after(0, lambda m=msg: self._train_status.configure(text=m))
            if prog is not None:
                self.after(0, lambda p=prog: self._train_progress.set(p))
        try:
            log("=== EarBio Training Started ===")
            detector  = EarDetector()
            extractor = FeatureExtractor()
            augment    = self._augment_var.get()
            test_split = float(self._test_split.get())
            grid_srch  = self._grid_var.get()
            dm = DatasetManager(dataset_root, detector, extractor,
                                augment=augment, test_size=test_split)
            stats = dm.scan()
            log(f"Dataset: {len(stats)} classes found")
            for cls, cnt in stats.items():
                log(f"  {cls}: {cnt} image(s)")
            set_status("Loading dataset…", 0.05)

            def progress_cb(cur, total):
                pct = cur / max(total, 1)
                self.after(0, lambda p=pct: self._train_progress.set(p * 0.5))
                self.after(0, lambda c=cur, t=total:
                    self._train_status.configure(text=f"Loading {c}/{t}…"))

            X, y = dm.load(progress_callback=progress_cb)
            log(f"Loaded {len(X)} samples, feature dim={X.shape[1]}")
            X_train, X_test, y_train, y_test = dm.split(X, y)
            log(f"Train={len(X_train)}  Test={len(X_test)}")
            set_status("Training SVM…", 0.55)

            clf = EarClassifier()
            result = clf.train(X_train, y_train, dm.class_names,
                               grid_search=grid_srch,
                               progress_callback=lambda msg: log(msg))
            set_status("Evaluating…", 0.85)

            metrics = clf.evaluate(X_test, y_test)
            acc = metrics["accuracy"]
            rpt = metrics["report"]
            log(f"\n=== Results ===")
            log(f"Test Accuracy  : {acc*100:.2f}%")
            log(f"Macro Precision: {rpt['macro avg']['precision']*100:.2f}%")
            log(f"Macro Recall   : {rpt['macro avg']['recall']*100:.2f}%")
            log(f"Macro F1       : {rpt['macro avg']['f1-score']*100:.2f}%")
            clf.save()
            log("Model saved → models/ear_svm_model.pkl")
            self.classifier = clf
            set_status("Done ✓", 1.0)
            self._stats.update({
                "classes":  len(dm.class_names),
                "images":   len(X_train),
                "accuracy": f"{acc*100:.1f}%",
                "model":    "Trained ✓",
            })
            self.after(0, self._refresh_stats)
            os.makedirs("reports", exist_ok=True)
            cm_path  = os.path.join("reports", "confusion_matrix.png")
            bar_path = os.path.join("reports", "per_class_metrics.png")
            clf.plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)
            clf.plot_accuracy_bar(metrics, save_path=bar_path)
            self.after(0, lambda: self._load_reports(metrics, cm_path, bar_path))
            log("=== Training Complete ===")
        except Exception as ex:
            import traceback
            log(f"[ERROR] {ex}")
            log(traceback.format_exc())
            set_status(f"Error: {ex}", 0)
        finally:
            self.after(0, lambda: self._train_btn.configure(
                state="normal", text="🎓  Start Training"))

    # ══════════════════════════════════════════════════════════════════════════
    #  LIVE CAMERA LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if not self.classifier.trained:
            messagebox.showwarning("Model Not Trained",
                                   "Please train the model first.")
            return
        cam_idx = int(self._cam_index.get())
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            messagebox.showerror("Camera Error",
                                 f"Cannot open camera index {cam_idx}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cam_cap     = cap
        self._cam_running = True
        self._cam_start_btn.configure(text="⏹  Stop Camera",
                                       fg_color=C_RED, hover_color="#c01060",
                                       text_color=C_TEXT)
        self._cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self._cam_thread.start()
        self._cam_update_ui()

    def _stop_camera(self):
        self._cam_running = False
        if self._cam_cap:
            self._cam_cap.release()
            self._cam_cap = None
        self._cam_start_btn.configure(text="▶  Start Camera",
                                       fg_color=C_GREEN, hover_color="#04a87f",
                                       text_color=C_BG)
        self._cam_label.configure(image=None, text="Camera stopped.")

    def _camera_worker(self):
        prev_time = time.time()
        while self._cam_running and self._cam_cap and self._cam_cap.isOpened():
            ret, frame = self._cam_cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi, bbox = self.detector.detect(gray)
            label, confidence, top3 = "—", 0.0, []
            if self.classifier.trained:
                try:
                    feat = self.extractor.extract(roi)
                    label, confidence = self.classifier.predict(feat)
                    top3 = self.classifier.predict_top_k(feat, k=3)
                except Exception:
                    pass
            vis = self.detector.draw_detection(frame, bbox)
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(vis, (x, y - 28), (x + w, y), (0, 0, 0), -1)
                cv2.putText(vis, f"{label}  {confidence*100:.1f}%",
                            (x + 4, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 150), 1)
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            try:
                self._frame_queue.put_nowait(rgb)
            except queue.Full:
                pass
            try:
                self._result_queue.put_nowait({
                    "label": label, "confidence": confidence,
                    "top3": top3, "fps": fps, "detected": bbox is not None,
                })
            except queue.Full:
                pass

    def _cam_update_ui(self):
        """Pull from queues and refresh GUI at ~30 fps (main thread)."""
        if not self._cam_running:
            return
        try:
            rgb = self._frame_queue.get_nowait()
            pil_img = Image.fromarray(rgb).resize((560, 420), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                                   size=(560, 420))
            self._cam_label.configure(image=ctk_img, text="")
            self._cam_label._image = ctk_img
        except queue.Empty:
            pass
        try:
            res = self._result_queue.get_nowait()
            self._live_name_var.set(res["label"])
            self._live_conf_meter.set(res["confidence"])
            self._live_grade_var.set(
                f"Grade: {confidence_to_label(res['confidence'])}")
            self._fps_var.set(f"FPS: {res['fps']:.1f}")
            self._ear_detected_var.set(
                "Ear: Detected ✓" if res["detected"] else "Ear: Searching…")
            for i, w in enumerate(self._top3_labels):
                if i < len(res["top3"]):
                    n, p = res["top3"][i]
                    w.configure(text=f"#{i+1}  {n}  ({p*100:.1f}%)")
                else:
                    w.configure(text="")
        except queue.Empty:
            pass
        self.after(33, self._cam_update_ui)

    # ══════════════════════════════════════════════════════════════════════════
    #  IDENTIFY IMAGE LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Ear Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.pgm"), ("All", "*.*")],
        )
        if path:
            self._identify_path.set(path)
            try:
                pil = Image.open(path).convert("RGB")
                pil.thumbnail((480, 360))
                ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
                self._id_img_label.configure(image=ctk_img, text="")
                self._id_img_label._image = ctk_img
            except Exception as e:
                messagebox.showerror("Error", f"Cannot load image: {e}")

    def _run_identify(self):
        path = self._identify_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid image.")
            return
        if not self.classifier.trained:
            messagebox.showwarning("Not Trained", "Train the model first.")
            return
        try:
            gray = load_image(path, grayscale=True)
            if gray is None:
                messagebox.showerror("Error", "Failed to read image.")
                return
            roi, bbox = self.detector.detect(gray)
            visuals = self.extractor.extract_with_visuals(roi)
            fv = visuals["feature_vector"]
            label, confidence = self.classifier.predict(fv)
            top3 = self.classifier.predict_top_k(fv, k=3)
            self._id_name_var.set(label)
            self._id_conf_meter.set(confidence)
            self._id_grade_var.set(
                f"Grade: {confidence_to_label(confidence)}")
            for i, w in enumerate(self._id_top3):
                if i < len(top3):
                    n, p = top3[i]
                    w.configure(text=f"#{i+1}  {n}  ({p*100:.1f}%)")
                else:
                    w.configure(text="")
            # LBP preview
            lbp_pil = Image.fromarray(visuals["lbp_image"]).resize((120, 90))
            lbp_ctk = ctk.CTkImage(light_image=lbp_pil, dark_image=lbp_pil,
                                   size=(120, 90))
            self._lbp_preview.configure(image=lbp_ctk, text="")
            self._lbp_preview._image = lbp_ctk
            # HOG preview
            hog_pil = Image.fromarray(visuals["hog_image"]).resize((120, 90))
            hog_ctk = ctk.CTkImage(light_image=hog_pil, dark_image=hog_pil,
                                   size=(120, 90))
            self._hog_preview.configure(image=hog_ctk, text="")
            self._hog_preview._image = hog_ctk
        except Exception as ex:
            import traceback
            messagebox.showerror("Identification Error", str(ex))
            traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════════
    #  REPORTS LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _load_reports(self, metrics: dict, cm_path: str, bar_path: str):
        rpt = metrics["report"]
        acc = metrics["accuracy"]
        self._rpt_acc.set(f"{acc*100:.1f}%")
        self._rpt_prec.set(f"{rpt['macro avg']['precision']*100:.1f}%")
        self._rpt_rec.set(f"{rpt['macro avg']['recall']*100:.1f}%")
        self._rpt_f1.set(f"{rpt['macro avg']['f1-score']*100:.1f}%")

        def _embed(label_widget, img_path, size=(380, 280)):
            if os.path.isfile(img_path):
                pil = Image.open(img_path).resize(size, Image.LANCZOS)
                ci  = ctk.CTkImage(light_image=pil, dark_image=pil, size=size)
                label_widget.configure(image=ci, text="")
                label_widget._image = ci

        _embed(self._cm_img_label,  cm_path)
        _embed(self._bar_img_label, bar_path)
        self._show_tab("reports")

    # ══════════════════════════════════════════════════════════════════════════
    #  CLOSE
    # ══════════════════════════════════════════════════════════════════════════

    def _on_close(self):
        self._cam_running = False
        if self._cam_cap:
            self._cam_cap.release()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = EarBiometricsApp()
    app.mainloop()
