"""
DeepScan — Tkinter Desktop Application
Standalone GUI for fake image detection.
Run: python desktop_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import io
import os
import threading
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageTk
import cv2

# PyTorch optional
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH = True
except ImportError:
    TORCH = False

# ─────────────────────────────────────────────
# CNN MODEL
# ─────────────────────────────────────────────
if TORCH:
    class FakeImageCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 2),
            )
        def forward(self, x):
            return self.classifier(self.features(x))

    _cnn = FakeImageCNN()
    model_path = os.environ.get('MODEL_PATH', 'best_model.pt')
    if os.path.isfile(model_path):
        _cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
    _cnn.eval()

    _tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    def cnn_score(img):
        with torch.no_grad():
            return float(torch.softmax(_cnn(_tfm(img).unsqueeze(0)), 1).squeeze()[1])
else:
    def cnn_score(img):
        return 0.5

# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def ela_analysis(img, quality=90):
    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    recomp = Image.open(buf).convert('RGB')
    diff = ImageChops.difference(img.convert('RGB'), recomp)
    ex = diff.getextrema()
    mx = max(e[1] for e in ex)
    diff = ImageEnhance.Brightness(diff).enhance(255.0 / mx if mx else 1.0)
    score = float(np.mean(np.array(diff)) / 255.0)
    return diff, score

def noise_analysis(img):
    arr = np.array(img.convert('L')).astype(np.float32)
    blur = cv2.GaussianBlur(arr, (5, 5), 0)
    return float(np.std(np.abs(arr - blur)) / 128.0)

def freq_analysis(img):
    arr = cv2.resize(np.array(img.convert('L')).astype(np.float32), (256, 256))
    dct = cv2.dct(arr)
    hf = np.abs(dct[128:, 128:])
    return min(float(np.mean(hf) / (np.mean(np.abs(dct)) + 1e-9)), 1.0)

def analyze(img):
    ela_img, ela_s = ela_analysis(img)
    ns = noise_analysis(img)
    fs = freq_analysis(img)
    cs = cnn_score(img)
    final = float(np.clip(0.4*cs + 0.3*ela_s + 0.2*ns + 0.1*fs, 0, 1))
    is_fake = final >= 0.5
    label = 'FAKE' if is_fake else 'REAL'
    conf  = final if is_fake else 1.0 - final
    return {
        'label': label,
        'conf':  round(conf * 100, 1),
        'score': round(final, 4),
        'cnn':   round(cs,    4),
        'ela':   round(ela_s, 4),
        'noise': round(ns,    4),
        'freq':  round(fs,    4),
        'ela_img': ela_img,
    }

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
BG      = '#0a0a0f'
SURFACE = '#16161f'
BORDER  = '#1e1e2e'
BLUE    = '#4d9fff'
GREEN   = '#00e5a0'
RED     = '#ff3c6e'
TEXT    = '#e8e8f0'
MUTED   = '#6b6b8a'

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
class DeepScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('DeepScan — Fake Image Detector')
        self.geometry('900x700')
        self.minsize(760, 580)
        self.configure(bg=BG)
        self._image = None
        self._build_ui()

    # ── UI CONSTRUCTION ───────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_main()
        self._build_statusbar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=SURFACE, height=54)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        tk.Label(hdr, text='Deep', font=('Courier New', 15, 'bold'),
                 bg=SURFACE, fg=TEXT).pack(side='left', padx=(18, 0), pady=14)
        tk.Label(hdr, text='Scan', font=('Courier New', 15, 'bold'),
                 bg=SURFACE, fg=BLUE).pack(side='left', pady=14)
        tk.Label(hdr, text='  —  Fake Image Detector',
                 font=('Helvetica', 9), bg=SURFACE, fg=MUTED).pack(side='left', pady=14)

        if not TORCH:
            tk.Label(hdr, text='[PyTorch not installed — CNN disabled]',
                     font=('Courier New', 8), bg=SURFACE, fg=RED).pack(side='right', padx=16)

    def _build_main(self):
        main = tk.Frame(self, bg=BG)
        main.pack(fill='both', expand=True, padx=16, pady=14)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_left(main)
        self._build_right(main)

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=SURFACE)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 7))

        # Image preview canvas
        tk.Label(left, text='// UPLOADED IMAGE', font=('Courier New', 7),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', padx=10, pady=(10, 0))
        self.img_canvas = tk.Canvas(left, bg='#0d0d14', bd=0,
                                    highlightthickness=0, height=270)
        self.img_canvas.pack(fill='both', expand=True, padx=10, pady=(4, 0))
        self._canvas_placeholder(self.img_canvas, 'No image loaded')

        # ELA preview canvas
        tk.Label(left, text='// ELA VISUALISATION', font=('Courier New', 7),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', padx=10, pady=(8, 0))
        self.ela_canvas = tk.Canvas(left, bg='#0d0d14', bd=0,
                                    highlightthickness=0, height=180)
        self.ela_canvas.pack(fill='both', expand=True, padx=10, pady=(4, 0))
        self._canvas_placeholder(self.ela_canvas, 'Run prediction to see ELA')

        # Buttons
        btn_row = tk.Frame(left, bg=SURFACE)
        btn_row.pack(fill='x', padx=10, pady=10)

        self.upload_btn = tk.Button(
            btn_row, text='📂  Upload Image',
            font=('Helvetica', 9, 'bold'),
            bg=BORDER, fg=TEXT, bd=0, padx=12, pady=9,
            cursor='hand2', activebackground='#2a2a3e', activeforeground=TEXT,
            command=self._upload,
        )
        self.upload_btn.pack(side='left', fill='x', expand=True, padx=(0, 5))

        self.predict_btn = tk.Button(
            btn_row, text='🔍  Predict',
            font=('Helvetica', 9, 'bold'),
            bg=BLUE, fg='#000', bd=0, padx=12, pady=9,
            cursor='hand2', activebackground='#6db3ff', activeforeground='#000',
            state='disabled', command=self._predict,
        )
        self.predict_btn.pack(side='left', fill='x', expand=True)

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=SURFACE)
        right.grid(row=0, column=1, sticky='nsew', padx=(7, 0))

        # Verdict area
        tk.Label(right, text='// VERDICT', font=('Courier New', 7),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', padx=10, pady=(10, 0))

        verdict_frame = tk.Frame(right, bg=BORDER, height=90)
        verdict_frame.pack(fill='x', padx=10, pady=(4, 0))
        verdict_frame.pack_propagate(False)

        inner = tk.Frame(verdict_frame, bg=BORDER)
        inner.place(relx=0.5, rely=0.5, anchor='center')

        self.verdict_lbl = tk.Label(inner, text='—',
                                     font=('Helvetica', 32, 'bold'),
                                     bg=BORDER, fg=MUTED)
        self.verdict_lbl.pack()
        self.conf_lbl = tk.Label(inner, text='Upload an image and click Predict',
                                  font=('Courier New', 8), bg=BORDER, fg=MUTED)
        self.conf_lbl.pack()

        # Score bars
        tk.Label(right, text='// SIGNAL SCORES', font=('Courier New', 7),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', padx=10, pady=(12, 0))

        self.score_widgets = {}
        for key, name in [('cnn', 'CNN Score (40%)'),
                           ('ela', 'ELA Score (30%)'),
                           ('noise', 'Noise Score (20%)'),
                           ('freq', 'Freq Score (10%)')]:
            f = tk.Frame(right, bg=SURFACE)
            f.pack(fill='x', padx=10, pady=3)

            hdr_row = tk.Frame(f, bg=SURFACE)
            hdr_row.pack(fill='x')
            tk.Label(hdr_row, text=name, font=('Helvetica', 8),
                     bg=SURFACE, fg=MUTED).pack(side='left')
            val_lbl = tk.Label(hdr_row, text='—', font=('Courier New', 8),
                               bg=SURFACE, fg=MUTED)
            val_lbl.pack(side='right')

            bar_bg = tk.Frame(f, bg=BORDER, height=5)
            bar_bg.pack(fill='x', pady=(2, 0))
            bar_fill = tk.Frame(bar_bg, bg=BLUE, height=5, width=0)
            bar_fill.place(x=0, y=0, height=5)

            self.score_widgets[key] = (val_lbl, bar_fill, bar_bg)

        # Log
        tk.Label(right, text='// LOG', font=('Courier New', 7),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', padx=10, pady=(12, 0))

        log_frame = tk.Frame(right, bg=SURFACE)
        log_frame.pack(fill='both', expand=True, padx=10, pady=(4, 10))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side='right', fill='y')

        self.log_box = tk.Text(
            log_frame, bg='#0d0d14', fg=MUTED,
            font=('Courier New', 8), bd=0,
            highlightthickness=0, wrap='word',
            yscrollcommand=scrollbar.set,
            insertbackground=TEXT,
        )
        self.log_box.pack(fill='both', expand=True)
        scrollbar.config(command=self.log_box.yview)

        self._log('DeepScan ready.')
        if not TORCH:
            self._log('PyTorch not found — CNN disabled (score fixed at 0.50).')
        self._log('Upload an image and click Predict.')

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(self, textvariable=self.status_var,
                 font=('Courier New', 8), bg=BORDER, fg=MUTED,
                 anchor='w', padx=12, pady=3).pack(fill='x', side='bottom')

    # ── HELPERS ───────────────────────────────
    def _canvas_placeholder(self, canvas, text):
        canvas.update_idletasks()
        w = canvas.winfo_width()  or 400
        h = canvas.winfo_height() or 200
        canvas.create_text(w//2, h//2, text=text, fill=MUTED,
                           font=('Courier New', 9))

    def _show_on_canvas(self, pil_img, canvas):
        canvas.update_idletasks()
        w = canvas.winfo_width()  or 400
        h = canvas.winfo_height() or 200
        copy = pil_img.copy()
        copy.thumbnail((w - 4, h - 4), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(copy)
        canvas._tk_img = tk_img          # prevent GC
        canvas.delete('all')
        canvas.create_image(w//2, h//2, anchor='center', image=tk_img)

    def _log(self, msg):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', f'> {msg}\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _reset_scores(self):
        self.verdict_lbl.configure(text='—', fg=MUTED)
        self.conf_lbl.configure(text='Running…', fg=MUTED)
        for key in self.score_widgets:
            lbl, fill, _ = self.score_widgets[key]
            lbl.configure(text='—', fg=MUTED)
            fill.configure(width=0, bg=BLUE)

    # ── ACTIONS ───────────────────────────────
    def _upload(self):
        path = filedialog.askopenfilename(
            title='Select an image',
            filetypes=[
                ('Images', '*.jpg *.jpeg *.png *.webp *.bmp'),
                ('All files', '*.*'),
            ]
        )
        if not path:
            return

        try:
            self._image = Image.open(path).convert('RGB')
        except Exception as e:
            messagebox.showerror('Error', f'Could not open image:\n{e}')
            return

        self._show_on_canvas(self._image, self.img_canvas)
        self.ela_canvas.delete('all')
        self._canvas_placeholder(self.ela_canvas, 'Click Predict to see ELA')

        self.predict_btn.configure(state='normal')
        document = os.path.basename(path)
        w, h = self._image.size
        self._log(f'Loaded: {document}  ({w}×{h} px)')
        self.status_var.set(f'Image loaded: {document}')
        document.strip()

    def _predict(self):
        if not self._image:
            return
        self._reset_scores()
        self.predict_btn.configure(state='disabled', text='Analysing…')
        self.status_var.set('Running analysis…')
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            self._log('Starting analysis pipeline…')
            result = analyze(self._image)
            self.after(0, lambda: self._show_result(result))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Analysis Error', str(e)))
            self.after(0, lambda: self.status_var.set('Error — see log'))
        finally:
            self.after(0, lambda: self.predict_btn.configure(
                state='normal', text='🔍  Predict'))

    def _show_result(self, r):
        is_fake = r['label'] == 'FAKE'
        color   = RED if is_fake else GREEN
        icon    = '⚠  FAKE' if is_fake else '✓  REAL'

        self.verdict_lbl.configure(text=icon, fg=color)
        self.conf_lbl.configure(
            text=f'Confidence: {r["conf"]}%   ·   Final Score: {r["score"]}',
            fg=color
        )

        bar_color = RED if is_fake else BLUE
        for key, val in [('cnn', r['cnn']), ('ela', r['ela']),
                          ('noise', r['noise']), ('freq', r['freq'])]:
            pct = int(val * 100)
            lbl, fill, bg = self.score_widgets[key]
            lbl.configure(text=f'{pct}%', fg=TEXT)
            bg.update_idletasks()
            bar_w = bg.winfo_width()
            fill.configure(width=max(1, int(bar_w * val)), bg=bar_color)

        self._show_on_canvas(r['ela_img'].copy(), self.ela_canvas)

        self._log(f'Result  : {r["label"]}  ({r["conf"]}% confidence)')
        self._log(f'CNN={r["cnn"]}  ELA={r["ela"]}  Noise={r["noise"]}  Freq={r["freq"]}')
        self.status_var.set(f'Result: {r["label"]} — {r["conf"]}% confidence')


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app = DeepScanApp()
    app.mainloop()
