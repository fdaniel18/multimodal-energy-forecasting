from __future__ import annotations
import os
import re
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "demo", "data")
META_CSV = os.path.join(ROOT, "data_set_genome", "metadata.csv")
ELEC_CSV = os.path.join(ROOT, "data_set_genome", "electricity_cleaned.csv")
GLOBAL_PRED = os.path.join(OUT, "predictions_global.parquet")

PREFIX = "Rezultate_Licenta_"
SUFFIX = "Bert"


def folder_to_bid(folder: str) -> str:
    s = folder[len(PREFIX):]
    if s.endswith(SUFFIX):
        s = s[: -len(SUFFIX)]
    return s


def is_ireland(bid: str) -> bool:
    return bool(re.match(r"H\d+$", bid))


def metrics(real: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - real
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum(np.abs(real)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape": wape}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    if not os.path.exists(GLOBAL_PRED):
        print(f"EROARE: lipseste {GLOBAL_PRED}. Ruleaza intai exportul global din notebook.")
        return 1
    gp = pd.read_parquet(GLOBAL_PRED, columns=["building_id", "ts"])
    gp["ts"] = pd.to_datetime(gp["ts"])
    target = sorted(gp["building_id"].unique())
    # fereastra de test per cladire, din predictiile globale (valabila pentru toate 107)
    test_win = gp.groupby("building_id")["ts"].agg(["min", "max", "count"])
    print(f"Cladiri tinta (din predictions_global): {len(target)}")

    meta = pd.read_csv(META_CSV)
    for col in ("sqm", "sqft", "yearbuilt", "numberoffloors", "occupants", "eui"):
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta = meta.set_index("building_id")

    elec_cols = pd.read_csv(ELEC_CSV, nrows=0).columns
    have_elec = [b for b in target if b in elec_cols]
    elec = pd.read_csv(ELEC_CSV, usecols=["timestamp"] + have_elec, parse_dates=["timestamp"])

    pred_rows = []
    building_rows = []
    no_perhouse = []

    for bid in target:
        # predictiile per-cladire exista doar daca a fost antrenat individual (are folder)
        csv = os.path.join(ROOT, f"{PREFIX}{bid}{SUFFIX}", "Evolutie_Predictie_Offset.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv, parse_dates=["Timestamp"])
            notext = df["Baseline_Prediction"].to_numpy(dtype=float)
            pred_rows.append(pd.DataFrame({
                "building_id": bid,
                "ts": df["Timestamp"].values,
                "real": df["Real_Consumption"].to_numpy(dtype=float),
                "perhouse_notext": notext,
                "perhouse_text": df["Hybrid_Prediction"].to_numpy(dtype=float),
            }))
        else:
            no_perhouse.append(bid)

        mrow = meta.loc[bid] if bid in meta.index else None
        hist = elec[bid] if bid in elec.columns else None
        mean_cons = float(hist.mean()) if hist is not None else float("nan")
        tw = test_win.loc[bid]
        building_rows.append({
            "building_id": bid,
            "site": (mrow["site_id"] if mrow is not None else None),
            "usage": (mrow["primaryspaceusage"] if mrow is not None else None),
            "sub_usage": (mrow["sub_primaryspaceusage"] if mrow is not None else None),
            "sqm": (float(mrow["sqm"]) if mrow is not None and pd.notna(mrow["sqm"]) else None),
            "yearbuilt": (int(mrow["yearbuilt"]) if mrow is not None and pd.notna(mrow["yearbuilt"]) else None),
            "floors": (float(mrow["numberoffloors"]) if mrow is not None and pd.notna(mrow["numberoffloors"]) else None),
            "occupants": (float(mrow["occupants"]) if mrow is not None and pd.notna(mrow["occupants"]) else None),
            "eui": (float(mrow["eui"]) if mrow is not None and pd.notna(mrow["eui"]) else None),
            "mean_consumption": mean_cons,
            "test_start": tw["min"],
            "test_end": tw["max"],
            "n_test": int(tw["count"]),
        })

    if pred_rows:
        predictions = pd.concat(pred_rows, ignore_index=True)
    else:
        predictions = pd.DataFrame(columns=["building_id", "ts", "real", "perhouse_notext", "perhouse_text"])
    buildings = pd.DataFrame(building_rows)

    history = elec.melt(id_vars="timestamp", value_vars=have_elec,
                        var_name="building_id", value_name="value")
    history = history.rename(columns={"timestamp": "ts"}).dropna(subset=["value"])
    history = history[["building_id", "ts", "value"]].reset_index(drop=True)

    predictions.to_parquet(os.path.join(OUT, "predictions.parquet"), index=False)
    buildings.to_parquet(os.path.join(OUT, "buildings.parquet"), index=False)
    history.to_parquet(os.path.join(OUT, "history.parquet"), index=False)

    def mb(name):
        return os.path.getsize(os.path.join(OUT, name)) / 1e6

    print("\nScris in demo/data/ (metrics.parquet nu e atins):")
    print(f"  buildings.parquet    {len(buildings):>8} cladiri  {mb('buildings.parquet'):6.2f} MB")
    print(f"  history.parquet      {len(history):>8} randuri  {mb('history.parquet'):6.2f} MB")
    print(f"  predictions.parquet  {len(predictions):>8} randuri  {mb('predictions.parquet'):6.2f} MB "
          f"({predictions['building_id'].nunique()} cladiri cu rezultate per-cladire)")
    if no_perhouse:
        print(f"\n{len(no_perhouse)} cladiri fara antrenare per-cladire (apar doar cu rezultate globale):")
        print("  " + ", ".join(no_perhouse))
    return 0


if __name__ == "__main__":
    sys.exit(main())
