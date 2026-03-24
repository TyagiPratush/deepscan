"""
DeepScan — Fake Image Detection
Flask Backend (Render-ready, production-safe)
"""

import os
import io
import uuid
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2

# PyTorch is optional — falls back gracefully if not installed
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─────────────────────────────────────────────
# FLASK APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']            = os.environ.get('SECRET_KEY', 'deepscan-secret-2026')
app.config['UPLOAD_FOLDER']         = 'static/uploads'
app.config['MAX_CONTENT_LENGTH']    = 16 * 1024 * 1024   # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─────────────────────────────────────────────
# DATABASE  (SQLite — swap for MySQL below)
# ─────────────────────────────────────────────
DB_PATH = os.environ.get('DB_PATH', 'detections.db')

# ── To use MySQL instead of SQLite, install mysql-connector-python and replace
#    the three db functions below with:
#
#   import mysql.connector
#   def get_conn():
#       return mysql.connector.connect(
#           host=os.environ.get('MYSQL_HOST', 'localhost'),
#           user=os.environ.get('MYSQL_USER', 'root'),
#           password=os.environ.get('MYSQL_PASSWORD', ''),
#           database=os.environ.get('MYSQL_DB', 'deepscan')
#       )

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            result      TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            ela_score   REAL,
            noise_score REAL,
            freq_score  REAL,
            timestamp   TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(filename, result, confidence, ela_score, noise_score, freq_score):
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections
            (filename, result, confidence, ela_score, noise_score, freq_score, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (filename, result, confidence, ela_score, noise_score, freq_score,
          datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def get_recent_detections(limit=10):
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# ─────────────────────────────────────────────
# CNN MODEL  (PyTorch)
# ─────────────────────────────────────────────
if TORCH_AVAILABLE:
    class FakeImageCNN(nn.Module):
        """
        4-block Convolutional Neural Network.
        Input : 128×128 RGB tensor
        Output: 2-class logits  [real, fake]
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                nn.MaxPool2d(2),
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                nn.MaxPool2d(2),
                # Block 4
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    _model = FakeImageCNN()

    # Load pre-trained weights if available
    MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model.pt')
    if os.path.isfile(MODEL_PATH):
        _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f'[INFO] Loaded model weights from {MODEL_PATH}')
    else:
        print('[INFO] No weights file found — using random CNN weights.')

    _model.eval()

    _transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    def cnn_predict(pil_image):
        tensor = _transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(_model(tensor), dim=1).squeeze().numpy()
        return float(probs[1])   # probability of FAKE

else:
    def cnn_predict(pil_image):
        """Fallback when PyTorch is not installed."""
        return 0.5

# ─────────────────────────────────────────────
# IMAGE ANALYSIS UTILITIES
# ─────────────────────────────────────────────
def ela_analysis(pil_image, quality=90):
    """
    Error Level Analysis.
    Re-compresses the image at lower quality and measures pixel differences.
    Tampered regions tend to show higher error levels.
    """
    buf = io.BytesIO()
    pil_image.convert('RGB').save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')

    ela_img = ImageChops.difference(pil_image.convert('RGB'), recompressed)
    extrema = ela_img.getextrema()
    max_diff = max(ex[1] for ex in extrema)
    scale    = 255.0 / max_diff if max_diff > 0 else 1.0
    ela_img  = ImageEnhance.Brightness(ela_img).enhance(scale)

    score = float(np.mean(np.array(ela_img).astype(np.float32)) / 255.0)
    return ela_img, score


def noise_analysis(pil_image):
    """
    Gaussian noise forensics.
    GAN-generated images often have unnaturally consistent noise patterns.
    """
    arr  = np.array(pil_image.convert('L')).astype(np.float32)
    blur = cv2.GaussianBlur(arr, (5, 5), 0)
    noise = np.abs(arr - blur)
    return float(np.std(noise) / 128.0)


def frequency_analysis(pil_image):
    """
    DCT high-frequency anomaly score.
    AI upsampling leaves distinctive artifacts in the frequency domain.
    """
    arr     = np.array(pil_image.convert('L')).astype(np.float32)
    resized = cv2.resize(arr, (256, 256))
    dct     = cv2.dct(resized)
    high_freq = np.abs(dct[128:, 128:])
    score   = float(np.mean(high_freq) / (np.mean(np.abs(dct)) + 1e-9))
    return float(min(score, 1.0))


def aggregate_score(cnn_score, ela_score, noise_score, freq_score):
    """Weighted ensemble of all four analysis signals."""
    weighted = (
        0.40 * cnn_score   +
        0.30 * ela_score   +
        0.20 * noise_score +
        0.10 * freq_score
    )
    return float(np.clip(weighted, 0.0, 1.0))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    recent = get_recent_detections(8)
    return render_template('index.html', recent=recent)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Use JPG, PNG, WEBP or BMP.'}), 400

    # Save uploaded file with a unique name
    ext         = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path   = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(save_path)

    # Open image
    try:
        pil_img = Image.open(save_path).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read image file. Please try another.'}), 400

    # Run all four analysis signals
    ela_img, ela_score  = ela_analysis(pil_img)
    noise_score         = noise_analysis(pil_img)
    freq_score          = frequency_analysis(pil_img)
    cnn_score           = cnn_predict(pil_img)
    final_score         = aggregate_score(cnn_score, ela_score, noise_score, freq_score)

    # Save ELA output image
    ela_name = f"ela_{unique_name}"
    ela_path = os.path.join(app.config['UPLOAD_FOLDER'], ela_name)
    ela_img.save(ela_path)

    # Decision threshold: 0.50
    is_fake    = final_score >= 0.50
    label      = 'FAKE' if is_fake else 'REAL'
    confidence = final_score if is_fake else (1.0 - final_score)

    # Persist to database
    save_detection(
        unique_name, label,
        round(confidence * 100, 2),
        round(ela_score,   4),
        round(noise_score, 4),
        round(freq_score,  4),
    )

    return jsonify({
        'result':          label,
        'confidence':      round(confidence * 100, 1),
        'final_score':     round(final_score,  4),
        'cnn_score':       round(cnn_score,    4),
        'ela_score':       round(ela_score,    4),
        'noise_score':     round(noise_score,  4),
        'freq_score':      round(freq_score,   4),
        'image_url':       f'/static/uploads/{unique_name}',
        'ela_url':         f'/static/uploads/{ela_name}',
        'torch_available': TORCH_AVAILABLE,
    })


@app.route('/history')
def history():
    rows = get_recent_detections(50)
    return render_template('history.html', detections=rows)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    """Health check endpoint for Render."""
    return jsonify({'status': 'ok', 'torch': TORCH_AVAILABLE}), 200


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
