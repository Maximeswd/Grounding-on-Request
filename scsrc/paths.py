"""Cluster paths + HF offline env. Import this FIRST (it sets env vars) before
transformers/datasets so the local cache is used and no network is hit on
compute nodes."""
import os

USER = os.environ.get("USER", "user")
# canonical HF cache: ./.cache/huggingface (most datasets+models live here)
HF_CACHE = (os.environ.get("SC_HF_CACHE") or os.environ.get("HF_HOME")
            or os.path.expanduser("~/.cache/huggingface"))
# one cache root: hub snapshots -> $HF_CACHE/hub, processed Arrow -> $HF_CACHE/datasets
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE, "hub")   # legacy alias
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
# offline on compute nodes (download happens on the login node via download_data.py)
os.environ.setdefault("HF_HUB_OFFLINE", os.environ.get("SC_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("SC_OFFLINE", "1"))
# wandb cannot reach the internet from compute nodes -> offline, sync from login
os.environ.setdefault("WANDB_MODE", os.environ.get("SC_WANDB_MODE", "offline"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT = os.environ.get("SC_PROJECT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# checkpoints/results can be large & are regenerable -> prefer expscratch if given
RUNS = os.environ.get("SC_RUNS", os.path.join(PROJECT, "runs"))
WANDB_DIR = os.environ.get("SC_WANDB_DIR", os.path.join(PROJECT, "logs"))

MODEL_ID = os.environ.get("SC_MODEL", "google/gemma-3-4b-it")
DATASET_ID = os.environ.get("SC_DATASET", "stanfordnlp/colorswap")

os.makedirs(RUNS, exist_ok=True)
os.makedirs(WANDB_DIR, exist_ok=True)
