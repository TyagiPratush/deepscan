# DeepScan — Fake Image Detector

Multi-signal deep learning system for detecting AI-generated and manipulated images.

## Project Structure

```
deepscan/
├── app.py               ← Flask web backend (main entry point)
├── streamlit_app.py     ← Streamlit alternative interface
├── desktop_app.py       ← Tkinter desktop GUI
├── train_model.py       ← CNN training script
├── requirements.txt     ← Python dependencies (Render-safe)
├── Procfile             ← Render/Heroku start command
├── runtime.txt          ← Python version for Render
├── .gitignore
├── README.md
├── templates/
│   ├── index.html       ← Upload + predict UI
│   ├── history.html     ← Detection history
│   └── about.html       ← Architecture overview
└── static/
    └── uploads/         ← Uploaded + ELA images (auto-managed)
        └── .gitkeep
```

## Detection Pipeline

| Signal | Weight | Library |
|--------|--------|---------|
| CNN Inference | 40% | PyTorch |
| ELA Analysis | 30% | Pillow + OpenCV |
| Noise Forensics | 20% | OpenCV |
| DCT Frequency | 10% | OpenCV + NumPy |

**Decision:** Final score ≥ 0.50 → FAKE, < 0.50 → REAL

## Local Setup (Windows)

```cmd
cd "C:\Users\YourName\deepscan"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python app.py
```

Visit: http://localhost:5000

## Run Streamlit

```cmd
venv\Scripts\activate
streamlit run streamlit_app.py
```

## Run Desktop GUI

```cmd
venv\Scripts\activate
python desktop_app.py
```

## Train the CNN

```
dataset/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

```cmd
python train_model.py --dataset ./dataset --epochs 20 --batch 32
```

Set `MODEL_PATH=best_model.pt` environment variable before running app.py.

## Deploy to Render

1. Push to GitHub
2. Connect repo on render.com → New Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Instance type: Free

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| PORT | 5000 | Web server port (set by Render automatically) |
| SECRET_KEY | deepscan-secret-2024 | Flask session key |
| MODEL_PATH | best_model.pt | Path to trained CNN weights |
| DB_PATH | detections.db | SQLite database path |

## MySQL (Optional)

Replace SQLite with MySQL by swapping the database functions in app.py:

```python
import mysql.connector
def get_conn():
    return mysql.connector.connect(
        host=os.environ.get('MYSQL_HOST', 'localhost'),
        user=os.environ.get('MYSQL_USER', 'root'),
        password=os.environ.get('MYSQL_PASSWORD', ''),
        database=os.environ.get('MYSQL_DB', 'deepscan')
    )
```

Install: `pip install mysql-connector-python`

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Backend | Python 3.11 / Flask / Gunicorn |
| Deep Learning | PyTorch CNN |
| Alt Interface | Streamlit |
| Desktop GUI | Tkinter |
| Image Processing | OpenCV (headless), Pillow |
| Numerics | NumPy |
| Visualisation | Matplotlib |
| Database | SQLite (MySQL-compatible) |
| Templates | Jinja2 / HTML / CSS |
| Hosting | Render |
