from __future__ import annotations
import os
import time
import numpy as np
import torch

from . import config
from .. import models as M


def _metrics(real: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - real
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum(np.abs(real)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else None
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape": wape}


class InferenceEngine:
    def __init__(self):
        self.device = "cpu"
        self.models: dict[str, torch.nn.Module] = {}
        self._bundles: dict[str, dict] = {}
        self._load_models()

    def _load_models(self):
        for name, path in config.CHECKPOINTS.items():
            if not os.path.exists(path):
                continue
            try:
                self.models[name] = M.load_model(
                    config.MODEL_KIND[name], path, config.NUM_MATH_FEATURES, self.device,
                    num_building_types=config.NUM_BUILDING_TYPES)
            except Exception as e:
                print(f"[infer] nu s-a putut incarca {name} ({path}): {e}")

    @property
    def available_models(self):
        return sorted(self.models.keys())

    def _bundle(self, building: str) -> dict | None:
        if building in self._bundles:
            return self._bundles[building]
        path = os.path.join(config.INFER_DIR, f"{building}.npz")
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=True)
        bundle = {
            "X_day": z["X_day"], "dyn_day": z["dyn_day"], "static": z["static"],
            "y_real": z["y_real"], "days": [str(d) for d in z["days"]],
            "scale": float(z["scale"]), "min": float(z["min"]),
            "text_static": str(z["text_static"]),
        }
        self._bundles[building] = bundle
        return bundle

    def test_days(self, building: str):
        b = self._bundle(building)
        return b["days"] if b else []

    def infer(self, building: str, day: str, model_names: list[str]):
        b = self._bundle(building)
        if b is None:
            raise FileNotFoundError(f"Nu exista bundle de inferenta pentru {building} "
                                    f"(ruleaza save_infer_bundles in notebook).")
        if day not in b["days"]:
            raise KeyError(f"Ziua {day} nu e disponibila pentru {building}.")
        i = b["days"].index(day)
        scale, mn = b["scale"], b["min"]

        def inv(x):
            return (np.asarray(x) - mn) / scale

        X = torch.tensor(b["X_day"][i:i + 1])
        static = torch.tensor(b["static"][None, :])
        dyn = torch.tensor(b["dyn_day"][i:i + 1])
        real = b["y_real"][i].astype(float)

        out = {
            "building": building, "day": day,
            "hours": list(range(len(real))),
            "real": real.tolist(),
            "models": {},
            "skipped": [],
        }
        for name in model_names:
            model = self.models.get(name)
            if model is None:
                out["skipped"].append(name)
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                if name == "global_notext":
                    pred = inv(model(X)[0].numpy())
                    entry = {"pred": pred.tolist()}
                elif name == "global_text":
                    p, _logits, math_pred, offset = model(X, static, dyn)
                    pred = inv(p[0].numpy())
                    base = inv(math_pred[0].numpy())
                    entry = {
                        "pred": pred.tolist(),
                        "base": base.tolist(),
                        "text_offset": (offset[0].numpy() / scale).tolist(),
                    }
                elif name == "patchtst":
                    pred = inv(model(X)[..., 0][0].numpy())
                    entry = {"pred": pred.tolist()}
                else:
                    out["skipped"].append(name)
                    continue
            entry["infer_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            entry["metrics"] = _metrics(real, np.asarray(entry["pred"]))
            out["models"][name] = entry

        out["infer_ms_total"] = round(
            sum(e.get("infer_ms", 0) for e in out["models"].values()), 2)

        out["text_static"] = b["text_static"]
        return out


_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine
