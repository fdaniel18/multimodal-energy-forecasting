
from __future__ import annotations
import os
import glob
import numpy as np
import pandas as pd

from . import config

MODEL_LABELS = {
    "perhouse_notext": "BiLSTM per clădire",
    "perhouse_text": "Hibrid per clădire",
    "global_notext": "BiLSTM global",
    "global_text": "Hibrid global",
    "patchtst": "PatchTST",
}
LOWER_BETTER = {"rmse": True, "mae": True, "wape": True, "r2": False}

WEEKDAYS = ["Luni", "Marti", "Miercuri", "Joi", "Vineri", "Sambata", "Duminica"]
MONTHS = ["Ian", "Feb", "Mar", "Apr", "Mai", "Iun",
          "Iul", "Aug", "Sep", "Oct", "Noi", "Dec"]


def _clean(records):
    out = []
    for r in records:
        out.append({k: (None if isinstance(v, float) and np.isnan(v) else v)
                    for k, v in r.items()})
    return out


class Store:
    def __init__(self):
        self.predictions = pd.read_parquet(config.PREDICTIONS)
        self.metrics = pd.read_parquet(config.METRICS)
        self.buildings = pd.read_parquet(config.BUILDINGS)
        self.history = pd.read_parquet(config.HISTORY)
        for df in (self.predictions, self.history):
            df["ts"] = pd.to_datetime(df["ts"])
        self.ids = sorted(self.buildings["building_id"].tolist())

        self._merge_extra_models()

        evaluated = set(self.metrics["building_id"].unique())
        if evaluated:
            self.buildings = self.buildings[
                self.buildings["building_id"].isin(evaluated)].reset_index(drop=True)
            self.ids = sorted(self.buildings["building_id"].tolist())

        self.pred_models = [m for m in MODEL_LABELS if m in self.predictions.columns]
        self.metric_models = [m for m in MODEL_LABELS if m in self.metrics["model"].unique().tolist()]

        self.embeddings = None
        self.embedding_meta = None
        if os.path.exists(config.EMBEDDING_META):
            self.embedding_meta = pd.read_parquet(config.EMBEDDING_META)
        if os.path.exists(config.EMBEDDINGS):
            self.embeddings = np.load(config.EMBEDDINGS)

    def _merge_extra_models(self):
        for path in sorted(glob.glob(os.path.join(config.DATA_DIR, "predictions_*.parquet"))):
            df = pd.read_parquet(path)
            df["ts"] = pd.to_datetime(df["ts"])
            extra_cols = [c for c in df.columns if c not in ("building_id", "ts")]
            self.predictions = self.predictions.merge(
                df[["building_id", "ts"] + extra_cols], on=["building_id", "ts"], how="left")
        for path in sorted(glob.glob(os.path.join(config.DATA_DIR, "metrics_*.parquet"))):
            self.metrics = pd.concat([self.metrics, pd.read_parquet(path)], ignore_index=True)
        self.metrics = self.metrics.drop_duplicates(subset=["building_id", "model"], keep="last")

    def list_buildings(self):
        df = self.buildings.copy()
        return _clean(df.to_dict("records"))

    def _metrics_for(self, bid: str) -> dict:
        m = self.metrics[self.metrics["building_id"] == bid]
        out = {}
        for _, row in m.iterrows():
            out[row["model"]] = {k: float(row[k]) for k in ("rmse", "mae", "r2", "wape")}
        if "global_notext" in out and "global_text" in out and out["global_notext"]["rmse"] > 0:
            out["rmse_improvement_pct"] = (
                (out["global_notext"]["rmse"] - out["global_text"]["rmse"])
                / out["global_notext"]["rmse"] * 100
            )
        return out

    def _text_static(self, bid: str):
        if self.embedding_meta is None:
            return None
        m = self.embedding_meta[self.embedding_meta["building_id"] == bid]
        if m.empty:
            return None
        t = m.iloc[0].get("text_static")
        return None if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)

    def building(self, bid: str):
        row = self.buildings[self.buildings["building_id"] == bid]
        if row.empty:
            return None
        info = _clean(row.to_dict("records"))[0]
        info["metrics"] = self._metrics_for(bid)
        info["text_static"] = self._text_static(bid)
        return info

    def predictions_for(self, bid: str, start=None, end=None, limit=None):
        df = self.predictions[self.predictions["building_id"] == bid].sort_values("ts")
        if start is not None:
            df = df[df["ts"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["ts"] <= pd.to_datetime(end)]
        if limit:
            df = df.head(int(limit))
        df = df.assign(ts=df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        cols = ["ts", "real"] + [m for m in self.pred_models if m in df.columns]
        return _clean(df[[c for c in cols if c in df.columns]].to_dict("records"))

    def prediction_weeks(self, bid: str):
        df = self.predictions[self.predictions["building_id"] == bid]
        if df.empty:
            return []
        iso = df["ts"].dt.isocalendar()
        weeks = sorted(iso["week"].unique().tolist())
        return [int(w) for w in weeks]

    def history_for(self, bid: str, resample: str | None = "D"):
        df = self.history[self.history["building_id"] == bid].sort_values("ts")
        if df.empty:
            return []
        s = df.set_index("ts")["value"]
        if resample:
            s = s.resample(resample).mean().dropna()
        out = pd.DataFrame({"ts": s.index.strftime("%Y-%m-%d %H:%M:%S"), "value": s.values})
        return _clean(out.to_dict("records"))

    def weekday_profile(self, bid: str):
        df = self.history[self.history["building_id"] == bid]
        if df.empty:
            return []
        g = df.groupby(df["ts"].dt.dayofweek)["value"].mean()
        return [{"weekday": WEEKDAYS[int(d)], "mean": float(v)} for d, v in g.items()]

    def month_profile(self, bid: str):
        df = self.history[self.history["building_id"] == bid]
        if df.empty:
            return []
        g = df.groupby(df["ts"].dt.month)["value"].mean()
        return [{"month": MONTHS[int(m) - 1], "mean": float(v)} for m, v in g.items()]

    def hourly_profile(self, bid: str):
        df = self.history[self.history["building_id"] == bid]
        if df.empty:
            return []
        g = df.groupby(df["ts"].dt.hour)["value"].mean()
        return [{"hour": int(h), "mean": float(v)} for h, v in g.items()]

    def metrics_summary(self):
        piv = self.metrics.pivot(index="building_id", columns="model", values="rmse")
        win = None
        if "global_text" in piv and "global_notext" in piv:
            pair = piv[["global_text", "global_notext"]].dropna()
            win = float((pair["global_text"] < pair["global_notext"]).mean() * 100)
        agg = {}
        for model in self.metric_models:
            sub = self.metrics[self.metrics["model"] == model]
            agg[model] = {k: float(sub[k].median()) for k in ("rmse", "mae", "r2", "wape")}
        return {
            "n_buildings": int(len(self.ids)),
            "models": self.metric_models,
            "text_win_rate_pct": win,
            "median": agg,
            "per_building": _clean(self.metrics.to_dict("records")),
        }

    def models_overview(self):
        out = []
        for model in self.metric_models:
            sub = self.metrics[self.metrics["model"] == model]
            stats = {}
            for k in ("rmse", "mae", "r2", "wape"):
                col = sub[k].dropna()
                stats[k] = {
                    "mean": float(col.mean()), "median": float(col.median()),
                    "std": float(col.std()), "min": float(col.min()), "max": float(col.max()),
                }
            out.append({"model": model, "label": MODEL_LABELS.get(model, model),
                        "n": int(len(sub)), "stats": stats})
        return out

    def models_compare(self, metric: str = "rmse"):
        from itertools import combinations
        piv = self.metrics.pivot(index="building_id", columns="model", values=metric)
        lower_better = LOWER_BETTER.get(metric, True)
        pairs = []
        for a, b in combinations(self.metric_models, 2):
            if a not in piv or b not in piv:
                continue
            d = piv[[a, b]].dropna()
            if len(d) < 3:
                continue
            a_better = (d[a] < d[b]) if lower_better else (d[a] > d[b])
            win = float(a_better.mean() * 100)
            med_a, med_b = float(d[a].median()), float(d[b].median())
            improve = ((med_b - med_a) / med_b * 100) if lower_better and med_b else \
                      ((med_a - med_b) / med_b * 100 if med_b else None)
            pairs.append({
                "a": a, "b": b, "metric": metric, "n_common": int(len(d)),
                "a_win_rate_pct": win, "median_a": med_a, "median_b": med_b,
                "median_improvement_pct": improve,
            })
        return pairs

    def embedding_scatter(self):
        if self.embedding_meta is None:
            return None
        return _clean(self.embedding_meta.to_dict("records"))


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
