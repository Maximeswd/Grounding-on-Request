"""Configuration: the YAML base config (configs/base.yaml) merged with command-line overrides, and the per-run
config.json record that evaluate.apply_checkpoint reads back to reproduce a run's prompt protocol."""
import yaml, os, json, argparse


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def merge(base: dict, override: dict):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def build_config(default_path, cli_overrides: dict):
    cfg = load_yaml(default_path)
    cfg = merge(cfg, cli_overrides)
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
