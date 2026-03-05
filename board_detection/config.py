from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
LABELS_DIR = DATA_DIR / "labels"
YOLO_DATASET_DIR = DATA_DIR / "yolo_dataset"
TEMPLATE_PATH = ROOT / "templates" / "chessboard_template.png"
RUNS_DIR = ROOT / "runs"

# ── Frame extraction ───────────────────────────────────────────────
FRAME_INTERVAL_SEC = 30  # 1 frame every 30 seconds

# ── Existing 15 video IDs (from piece classification dataset) ─────
EXISTING_VIDEO_IDS = [
    "-ZVbDR3sRRo",
    "3I_ESQVyxNc",
    "65VWIFlc4C4",
    "9dQzTnvsNG4",
    "B4lR3NYwI8",
    "CmM1zxS_Ae8",
    "NFod-ozimmM",
    "PmQs1KhB948",
    "QNcO9CJyDBc",
    "Uc7Kf_-hsgw",
    "Xxqi7IvwekE",
    "ghJRGPXsjfk",
    "r0jMl4qNdyQ",
    "u0o38cEaGdw",
    "wYCYWpx3CFM",
]

# ── New video IDs (populated by find_videos.py) ───────────────────
NEW_VIDEO_IDS = [
    # Add IDs here after running find_videos.py
]

VIDEO_IDS = EXISTING_VIDEO_IDS + NEW_VIDEO_IDS

# ── GothamChess channel ───────────────────────────────────────────
GOTHAMCHESS_CHANNEL_URL = "https://www.youtube.com/@GothamChess"

# ── Download settings ─────────────────────────────────────────────
MAX_VIDEO_HEIGHT = 720
SLEEP_INTERVAL = 5
MAX_SLEEP_INTERVAL = 15

# ── YOLO class mapping ────────────────────────────────────────────
CLASS_NAMES = {0: "chessboard"}
