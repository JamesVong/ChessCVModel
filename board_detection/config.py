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
    "BmyY31di9B4",  # 29.7m - The Gukesh Situation Is Insane
    "WlmTRwcJzzo",  # 22.0m - Evil Baby vs Martin: STUPID CHESS
    "w5_WiD3sfxU",  # 17.7m - Can I Beat My Own Baby Chess Bot?
    "jTeJonyDjoA",  # 23.6m - 6 Brilliant Moves IN ONE GAME
    "KNko92SXggk",  # 22.8m - How?
    "dPLLaPsNbJs",  # 25.5m - Claude vs DeepSeek: The End Of Chess
    "iBJ-DurxcoM",  # 25.7m - ChatGPT HITS 5,000 CHESS ELO
    "YhQNCuCE4no",  # 22.3m - DeepSeek Solved Chess. Goodbye.
    "hzzPs17gavs",  # 21.9m - Claude Tried Chess. It's TERRIFYING. 
    "7g-jN3DTkWQ",  # 23.3m - Grok vs Copilot: DUMBEST CHESS MATCH
    "7S8QPpeCyD8",  # 23.2m - CHATBOT CHESS CHAMPIONSHIP IS BACK!!!!!!
    "DmFE2j9dzR4",  # 28.2m - MAGNUS CARLSEN WINS THE WORLD CHAMPIONSH
    "cTx8qu1m8EU",  # 24.2m - Magnus Carlsen Watches Gotham Play Chess
    "cQAedm_gWrw",  # 25.1m - I Played GM Pia in an Official Tournamen
    "6iWVZb6n8GM",  # 23.9m - GUESS THE ELO: Magnus vs Levy
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
