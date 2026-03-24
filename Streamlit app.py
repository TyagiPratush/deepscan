"""
DeepScan — Streamlit Interface
Run: streamlit run streamlit_app.py
"""

import io
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import streamlit as st

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

    @st.cache_resource
    def load_model():
        m = FakeImageCNN()
        model_path = os.environ.get('MODEL_PATH', 'best_model.pt')
        if os.path.isfile(model_path):
            m.load_state_dict(torch.load(model_path, map_location='cpu'))
        m.eval()
        return m

    _tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    def cnn_pred(img):
        m = load_model()
        with torch.no_grad():
            return float(torch.softmax(m(_tfm(img).unsqueeze(0)), 1).squeeze()[1])
else:
    def cnn_pred(img):
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
    cs = cnn_pred(img)
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
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title='DeepScan — Fake Image Detector',
    page_icon='🔍',
    layout='wide',
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0a0f; }
  [data-testid="stHeader"]           { background: #0a0a0f; }
  section[data-testid="stSidebar"]   { background: #12121a; }
  .block-container { padding-top: 2rem; }
  h1, h2, h3, p, label, div { color: #e8e8f0; }
  .stButton > button {
    background: #4d9fff; color: #000;
    font-weight: 700; border: none;
    border-radius: 8px; padding: 0.65rem 2rem;
    width: 100%; font-size: 1rem;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #6db3ff; border: none; }
  .verdict-fake { color: #ff3c6e !important; font-size: 2.2rem; font-weight: 800; }
  .verdict-real { color: #00e5a0 !important; font-size: 2.2rem; font-weight: 800; }
  [data-testid="stMetricValue"] { color: #4d9fff !important; }
  .stProgress > div > div { background: #4d9fff; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🔍 DeepScan")
st.markdown("**Multi-signal fake image detection** — CNN + ELA + Noise + Frequency Analysis")

if not TORCH:
    st.warning("PyTorch not installed — CNN signal disabled. ELA, Noise & Frequency signals are active.")

st.divider()

col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.markdown("#### Upload Image")
    uploaded = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        label_visibility='collapsed',
    )

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.markdown(f"**Size:** {img.size[0]} × {img.size[1]} px")

        analyse_clicked = st.button('🔍  Analyse for Fake Detection', use_container_width=True)

        if analyse_clicked:
            with st.spinner('Running multi-signal analysis…'):
                r = analyze(img)

            with col2:
                st.markdown("#### Results")

                cls  = 'fake' if r['label'] == 'FAKE' else 'real'
                icon = '⚠️'   if r['label'] == 'FAKE' else '✅'
                st.markdown(
                    f'<p class="verdict-{cls}">{icon} {r["label"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** `{r['conf']}%` &nbsp;|&nbsp; **Final Score:** `{r['score']}`")
                st.divider()

                # Score metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('CNN',   f"{round(r['cnn']   * 100)}%")
                c2.metric('ELA',   f"{round(r['ela']   * 100)}%")
                c3.metric('Noise', f"{round(r['noise'] * 100)}%")
                c4.metric('Freq',  f"{round(r['freq']  * 100)}%")

                # Progress bars
                st.markdown("**Signal strengths:**")
                for label, val in [('CNN (40%)', r['cnn']), ('ELA (30%)', r['ela']),
                                    ('Noise (20%)', r['noise']), ('Freq (10%)', r['freq'])]:
                    st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
                    st.progress(float(val))

                # ELA image
                st.markdown("#### ELA Visualisation")
                st.image(r['ela_img'], caption='Error Level Analysis — bright = high inconsistency',
                         use_column_width=True)

                # Explanation
                with st.expander("📊 What do these scores mean?"):
                    st.markdown("""
| Signal | Weight | What it detects |
|--------|--------|-----------------|
| CNN Score | 40% | Deep learned manipulation features |
| ELA Score | 30% | JPEG compression inconsistencies |
| Noise Score | 20% | Unnatural GAN noise patterns |
| Freq Score | 10% | DCT high-frequency AI artefacts |

**Decision rule:** Final score ≥ 0.50 → FAKE, < 0.50 → REAL
                    """)
    else:
        with col2:
            st.markdown("#### How it works")
            st.info("""
**Four-signal analysis pipeline:**

1. 🧠 **CNN** — PyTorch model extracts deep image features
2. 🔬 **ELA** — Error Level Analysis detects compression inconsistencies
3. 📡 **Noise** — Gaussian noise forensics via OpenCV
4. 📊 **Frequency** — DCT analysis for AI upsampling artefacts

Upload an image on the left to begin.
            """)
            st.markdown("#### Signal Weights")
            for label, pct in [('CNN', 0.40), ('ELA', 0.30), ('Noise', 0.20), ('Freq', 0.10)]:
                st.markdown(f"<small>{label} — {int(pct*100)}%</small>", unsafe_allow_html=True)
                st.progress(pct)

st.divider()
st.caption("DeepScan © 2024 — Built with Flask · PyTorch · OpenCV · Streamlit")
