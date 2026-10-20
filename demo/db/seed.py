from __future__ import annotations
import os
import glob
import pandas as pd

from .database import SessionLocal, engine, DATABASE_URL
from . import schema
from .database import Base
from ..backend import config

MODEL_REGISTRY = [
    dict(name="perhouse_notext", label="BiLSTM per clădire", kind="baseline",
         checkpoint_path=None, n_params=61080, lookback=168, forecast=24,
         hyperparams={"hidden_dim": 64, "bidirectional": True, "per_building": True},
         description="BiLSTM antrenat separat pe fiecare cladire, fara text."),
    dict(name="perhouse_text", label="Hibrid per clădire", kind="text",
         checkpoint_path=None, n_params=None, lookback=168, forecast=24,
         hyperparams={"per_building": True, "text": True},
         description="Model hibrid antrenat separat pe fiecare cladire, cu text."),
    dict(name="global_notext", label="BiLSTM global", kind="baseline",
         checkpoint_path="global_notext_v10.pt", n_params=61080, lookback=168, forecast=24,
         hyperparams={"hidden_dim": 64, "bidirectional": True},
         description="BiLSTM pur, pooled global, fara context textual."),
    dict(name="global_text", label="Hibrid global", kind="text",
         checkpoint_path="global_model_v10.pt", n_params=567040, lookback=168, forecast=24,
         hyperparams={"hidden_dim": 64, "text_emb_dim": 768, "film": True, "attention": True},
         description="GlobalTextConditionedModel: FiLM static + offset prin atentie pe text dinamic."),
    dict(name="patchtst", label="PatchTST", kind="sota",
         checkpoint_path="patchtst_v10.pt", n_params=133720, lookback=168, forecast=24,
         hyperparams={"patch_len": 16, "stride": 8, "d_model": 64, "n_heads": 4, "e_layers": 3},
         description="PatchTST (Nie et al., ICLR 2023), acelasi setup/ferestre ca modelul global."),
]


def _opt(v):
    return None if pd.isna(v) else v


def seed():
    Base.metadata.create_all(engine)
    s = SessionLocal()
    try:
        for m in MODEL_REGISTRY:
            s.merge(schema.Model(**m))

        bdf = pd.read_parquet(config.BUILDINGS)
        for _, r in bdf.iterrows():
            s.merge(schema.Building(
                building_id=r["building_id"], site=_opt(r.get("site")),
                usage=_opt(r.get("usage")), sub_usage=_opt(r.get("sub_usage")),
                sqm=_opt(r.get("sqm")),
                yearbuilt=(int(r["yearbuilt"]) if not pd.isna(r.get("yearbuilt")) else None),
                mean_consumption=_opt(r.get("mean_consumption")),
            ))

        mfiles = [config.METRICS] + sorted(glob.glob(os.path.join(config.DATA_DIR, "metrics_*.parquet")))
        seen = set()
        for f in mfiles:
            if not os.path.exists(f):
                continue
            mdf = pd.read_parquet(f)
            for _, r in mdf.iterrows():
                key = (r["building_id"], r["model"])
                if key in seen:
                    continue
                seen.add(key)
                s.merge(schema.Metric(
                    building_id=r["building_id"], model_name=r["model"],
                    rmse=_opt(r.get("rmse")), mae=_opt(r.get("mae")),
                    r2=_opt(r.get("r2")), wape=_opt(r.get("wape")),
                ))

        pdf = pd.read_parquet(config.PREDICTIONS, columns=["building_id", "ts"])
        pdf["day"] = pd.to_datetime(pdf["ts"]).dt.date
        days = pdf[["building_id", "day"]].drop_duplicates()
        for _, r in days.iterrows():
            s.merge(schema.TestDay(building_id=r["building_id"], day=r["day"]))

        s.commit()
        print(f"Seed OK ({DATABASE_URL.split('@')[-1]}): "
              f"{len(MODEL_REGISTRY)} modele, {len(bdf)} cladiri, "
              f"{len(seen)} metrici, {len(days)} zile de test.")
    finally:
        s.close()


if __name__ == "__main__":
    seed()
