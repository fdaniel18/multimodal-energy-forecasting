import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.dirname(BACKEND_DIR)
ROOT = os.path.dirname(DEMO_DIR)
DATA_DIR = os.path.join(DEMO_DIR, "data")
INFER_DIR = os.path.join(DATA_DIR, "infer")


CHECKPOINTS = {
    "global_notext": os.path.join(ROOT, "global_notext_v10.pt"),
    "global_text": os.path.join(ROOT, "global_model_v10.pt"),
    "patchtst": os.path.join(ROOT, "patchtst_v10.pt"),
}

MODEL_KIND = {"global_notext": "notext", "global_text": "hybrid", "patchtst": "patchtst"}
NUM_MATH_FEATURES = 15
NUM_BUILDING_TYPES = 16

PREDICTIONS = os.path.join(DATA_DIR, "predictions.parquet")
METRICS = os.path.join(DATA_DIR, "metrics.parquet")
BUILDINGS = os.path.join(DATA_DIR, "buildings.parquet")
HISTORY = os.path.join(DATA_DIR, "history.parquet")
EMBEDDINGS = os.path.join(DATA_DIR, "embeddings.npy")
EMBEDDING_META = os.path.join(DATA_DIR, "embedding_meta.parquet")

CORS_ORIGINS = ["*"]
