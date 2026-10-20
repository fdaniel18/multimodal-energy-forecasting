from __future__ import annotations
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _metrics(real: np.ndarray, pred: np.ndarray) -> dict:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - real
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum(np.abs(real)))
    wape = float(np.sum(np.abs(err)) / denom) if denom > 0 else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((real - real.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "wape": wape}


def save_patchtst(pt_model, pt_buildings, global_store, device,
                  model_name: str = "patchtst", out_dir: str = DATA_DIR):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    pt_model.eval()
    pred_rows, metric_rows = [], []

    for b in pt_buildings:
        s = global_store[b]["splits"]["test"]
        with torch.no_grad():
            mb = torch.tensor(np.asarray(s["X"], dtype=np.float32)).to(device)
            pred = pt_model(mb)[..., 0].cpu().numpy()
        es = global_store[b]["energy_scaler"]
        pa = es.inverse_transform(pred.reshape(-1, 1)).reshape(pred.shape)
        ya = es.inverse_transform(np.asarray(s["y"]).reshape(-1, 1)).reshape(np.asarray(s["y"]).shape)
        ts = np.asarray(s["ts"])

        n = min(len(ts), pa.shape[0])
        first_pred = pa[:n, 0]
        first_real = ya[:n, 0]
        pred_rows.append(pd.DataFrame({
            "building_id": b,
            "ts": pd.to_datetime(ts[:n]),
            model_name: first_pred,
        }))
        metric_rows.append({"building_id": b, "model": model_name,
                            **_metrics(first_real, first_pred)})

    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    pred_path = os.path.join(out_dir, f"predictions_{model_name}.parquet")
    met_path = os.path.join(out_dir, f"metrics_{model_name}.parquet")
    predictions.to_parquet(pred_path, index=False)
    metrics.to_parquet(met_path, index=False)
    print(f"[demo] scris {pred_path}  ({len(predictions)} randuri, {len(pt_buildings)} cladiri)")
    print(f"[demo] scris {met_path}   (median RMSE={metrics['rmse'].median():.3f})")
    return predictions, metrics


def _project_3d(X: np.ndarray) -> np.ndarray:
    """UMAP 3D daca e disponibil (metric cosinus), altfel PCA."""
    n = len(X)
    try:
        import umap
        reducer = umap.UMAP(n_components=3, n_neighbors=min(15, max(2, n - 1)),
                            min_dist=0.15, metric="cosine", random_state=42)
        return reducer.fit_transform(X)
    except Exception as e:
        print(f"[demo] UMAP indisponibil ({e}); folosesc PCA.")
        from sklearn.decomposition import PCA
        return PCA(n_components=3, random_state=42).fit_transform(X)


def _clean_metric_dict(d):
    return {bid: {k: float(v) for k, v in m.items()} for bid, m in (d or {}).items()}


def _load_perhouse_dicts(out_dir: str = DATA_DIR):
    import json
    p = os.path.join(out_dir, "perhouse_results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_perhouse_dicts(results_dict=None, results_hybrid_dict=None, out_dir: str = DATA_DIR):
    import json
    os.makedirs(out_dir, exist_ok=True)
    data = _load_perhouse_dicts(out_dir)
    if results_dict:
        data["results_dict"] = _clean_metric_dict(results_dict)
    if results_hybrid_dict:
        data["results_hybrid_dict"] = _clean_metric_dict(results_hybrid_dict)
    with open(os.path.join(out_dir, "perhouse_results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[demo] salvat rezultate per-cladire pe disc: {list(data.keys())}.")


def save_metrics_24h(global_store, num_math_features=15, device="cpu",
                     results_dict=None, results_hybrid_dict=None, out_dir: str = DATA_DIR):
    import sys
    import torch
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from demo import models as M

    specs = {
        "global_notext": (os.path.join(_root, "global_notext_v10.pt"), "notext"),
        "global_text": (os.path.join(_root, "global_model_v10.pt"), "hybrid"),
        "patchtst": (os.path.join(_root, "patchtst_v10.pt"), "patchtst"),
    }
    nbt = 16
    if os.path.exists(specs["global_text"][0]):
        nbt = torch.load(specs["global_text"][0], map_location="cpu")["building_classifier.weight"].shape[0]
    loaded = {}
    for name, (path, kind) in specs.items():
        if os.path.exists(path):
            loaded[name] = M.load_model(kind, path, num_math_features, device, num_building_types=nbt)
        else:
            print(f"  [info] checkpoint lipsa pentru {name}, il sar.")

    rows = []
    for b, gs in global_store.items():
        s = gs["splits"]["test"]
        X = torch.tensor(np.asarray(s["X"], dtype=np.float32)).to(device)
        n = X.shape[0]
        st = np.asarray(gs["static"], dtype=np.float32).ravel()
        static = torch.tensor(np.broadcast_to(st, (n, st.shape[0])).copy()).to(device)
        dyn = torch.tensor(np.asarray(s["dyn"], dtype=np.float32)).to(device)
        es = gs["energy_scaler"]
        y = np.asarray(s["y"])
        y_a = es.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
        with torch.no_grad():
            preds = {}
            if "global_notext" in loaded:
                preds["global_notext"] = loaded["global_notext"](X).cpu().numpy()
            if "global_text" in loaded:
                preds["global_text"] = loaded["global_text"](X, static, dyn)[0].cpu().numpy()
            if "patchtst" in loaded:
                preds["patchtst"] = loaded["patchtst"](X)[..., 0].cpu().numpy()
        for name, p in preds.items():
            pa = es.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)
            rows.append({"building_id": b, "model": name,
                         **_metrics(y_a.flatten(), pa.flatten())})

    if results_dict is None or results_hybrid_dict is None:
        disk = _load_perhouse_dicts(out_dir)
        results_dict = results_dict or disk.get("results_dict")
        results_hybrid_dict = results_hybrid_dict or disk.get("results_hybrid_dict")

    keys = set(global_store.keys())
    for model, d in (("perhouse_notext", results_dict), ("perhouse_text", results_hybrid_dict)):
        if not d:
            continue
        n_added = 0
        for bid, m in d.items():
            if bid in keys:
                rows.append({"building_id": bid, "model": model,
                             "rmse": float(m["RMSE"]), "mae": float(m["MAE"]),
                             "r2": float(m["R2"]), "wape": float(m["WAPE"])})
                n_added += 1
        print(f"  [demo] {model}: {n_added} cladiri (din {len(d)} in dict, intersectie cu global).")

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "metrics.parquet"), index=False)
    for f in ("metrics_global.parquet", "metrics_patchtst.parquet"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)

    piv = df.pivot(index="building_id", columns="model", values="rmse")
    if "global_text" in piv and "global_notext" in piv:
        pair = piv[["global_text", "global_notext"]].dropna()
        win = (pair["global_text"] < pair["global_notext"]).mean() * 100
        print(f"[demo] metrics.parquet (24h) scris: {len(df)} randuri, {df['model'].nunique()} modele.")
        print(f"[demo] Global: textul ajuta la {win:.0f}% din cladiri "
              f"(median notext={pair['global_notext'].median():.2f}, "
              f"text={pair['global_text'].median():.2f}).")
    else:
        print(f"[demo] metrics.parquet (24h) scris: {len(df)} randuri.")


def save_platform_metrics(results_dict=None, results_hybrid_dict=None,
                          results_notext_dict=None, results_global_dict=None,
                          results_patchtst_dict=None, out_dir: str = DATA_DIR):

    mapping = {
        "perhouse_notext": results_dict,
        "perhouse_text": results_hybrid_dict,
        "global_notext": results_notext_dict,
        "global_text": results_global_dict,
        "patchtst": results_patchtst_dict,
    }
    rows = []
    for model, d in mapping.items():
        if not d:
            print(f"  [info] lipseste dictul pentru {model}, il sar.")
            continue
        for bid, m in d.items():
            rows.append({
                "building_id": bid, "model": model,
                "rmse": float(m["RMSE"]), "mae": float(m["MAE"]),
                "r2": float(m["R2"]), "wape": float(m["WAPE"]),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "metrics.parquet"), index=False)
    for f in ("metrics_global.parquet", "metrics_patchtst.parquet"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            os.remove(p)

    piv = df.pivot(index="building_id", columns="model", values="rmse")
    if "global_text" in piv and "global_notext" in piv:
        pair = piv[["global_text", "global_notext"]].dropna()
        win = (pair["global_text"] < pair["global_notext"]).mean() * 100
        print(f"[demo] metrics.parquet scris ({len(df)} randuri, {df['model'].nunique()} modele, 24h).")
        print(f"[demo] Global: textul ajuta la {win:.0f}% din cladiri "
              f"(median notext={pair['global_notext'].median():.2f}, "
              f"text={pair['global_text'].median():.2f}).")
    else:
        print(f"[demo] metrics.parquet scris ({len(df)} randuri).")


def save_global_predictions(global_store, num_math_features=15, device="cpu",
                            notext_ckpt=None, hybrid_ckpt=None, out_dir: str = DATA_DIR):

    import sys
    import torch
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from demo import models as M

    notext_ckpt = notext_ckpt or os.path.join(_root, "global_notext_v10.pt")
    hybrid_ckpt = hybrid_ckpt or os.path.join(_root, "global_model_v10.pt")
    nbt = torch.load(hybrid_ckpt, map_location="cpu")["building_classifier.weight"].shape[0]
    notext = M.load_model("notext", notext_ckpt, num_math_features, device)
    hybrid = M.load_model("hybrid", hybrid_ckpt, num_math_features, device, num_building_types=nbt)

    pred_rows, metric_rows = [], []
    for b, gs in global_store.items():
        s = gs["splits"]["test"]
        X = torch.tensor(np.asarray(s["X"], dtype=np.float32)).to(device)
        n = X.shape[0]
        st = np.asarray(gs["static"], dtype=np.float32).ravel()
        static = torch.tensor(np.broadcast_to(st, (n, st.shape[0])).copy()).to(device)
        dyn = torch.tensor(np.asarray(s["dyn"], dtype=np.float32)).to(device)
        with torch.no_grad():
            nt = notext(X).cpu().numpy()
            hy = hybrid(X, static, dyn)[0].cpu().numpy()
        es = gs["energy_scaler"]
        nt_a = es.inverse_transform(nt.reshape(-1, 1)).reshape(nt.shape)
        hy_a = es.inverse_transform(hy.reshape(-1, 1)).reshape(hy.shape)
        y_a = es.inverse_transform(np.asarray(s["y"]).reshape(-1, 1)).reshape(np.asarray(s["y"]).shape)
        ts = pd.to_datetime(np.asarray(s["ts"]))
        m = min(len(ts), nt_a.shape[0])
        real, ntp, hyp = y_a[:m, 0], nt_a[:m, 0], hy_a[:m, 0]
        pred_rows.append(pd.DataFrame({
            "building_id": b, "ts": ts[:m],
            "global_notext": ntp, "global_text": hyp,
        }))
        for model, p in (("global_notext", ntp), ("global_text", hyp)):
            metric_rows.append({"building_id": b, "model": model, **_metrics(real, p)})

    predictions = pd.concat(pred_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_parquet(os.path.join(out_dir, "predictions_global.parquet"), index=False)
    metrics.to_parquet(os.path.join(out_dir, "metrics_global.parquet"), index=False)
    piv = metrics.pivot(index="building_id", columns="model", values="rmse")
    win = (piv["global_text"] < piv["global_notext"]).mean() * 100
    print(f"[demo] GLOBAL scris: predictions_global.parquet ({len(predictions)} randuri), "
          f"metrics_global.parquet ({len(global_store)} cladiri).")
    print(f"[demo] Text imbunatateste RMSE la {win:.0f}% din cladiri "
          f"(median global_notext={piv['global_notext'].median():.2f}, "
          f"global_text={piv['global_text'].median():.2f}).")


def save_infer_bundles(valid_buildings, global_store, generate_static_text=None,
                       df_metadata=None, out_dir: str = DATA_DIR):
    infer_dir = os.path.join(out_dir, "infer")
    os.makedirs(infer_dir, exist_ok=True)
    n_ok = 0
    for b in valid_buildings:
        gs = global_store.get(b)
        if gs is None:
            continue
        s = gs["splits"]["test"]
        ts = pd.to_datetime(np.asarray(s["ts"]))
        mask = ts.hour == 0
        if mask.sum() == 0:
            print(f"  [info] {b}: nicio fereastra aliniata pe zi, sarit.")
            continue

        es = gs["energy_scaler"]
        y = np.asarray(s["y"], dtype=np.float32)[mask]
        y_real = es.inverse_transform(y.reshape(-1, 1)).reshape(y.shape).astype(np.float32)

        text_static = ""
        if generate_static_text is not None and df_metadata is not None:
            text_static = generate_static_text(b, df_metadata[df_metadata["building_id"] == b])

        np.savez_compressed(
            os.path.join(infer_dir, f"{b}.npz"),
            X_day=np.asarray(s["X"], dtype=np.float32)[mask],
            dyn_day=np.asarray(s["dyn"], dtype=np.float32)[mask],
            static=np.asarray(gs["static"], dtype=np.float32).ravel(),
            y_real=y_real,
            days=np.array([d.strftime("%Y-%m-%d") for d in ts[mask]]),
            scale=np.float32(es.scale_[0]),
            min=np.float32(es.min_[0]),
            text_static=np.array(text_static),
        )
        n_ok += 1
    print(f"[demo] scris {n_ok} bundle-uri de inferenta in {infer_dir}")


def save_embeddings(valid_buildings, df_metadata, build_static_emb, generate_static_text,
                    n_clusters: int | None = None, out_dir: str = DATA_DIR):
    from sklearn.cluster import KMeans
    os.makedirs(out_dir, exist_ok=True)

    ids, embs, texts, rows = [], [], [], []
    for b in valid_buildings:
        meta_row = df_metadata[df_metadata["building_id"] == b]
        e = np.asarray(build_static_emb(b, meta_row), dtype=np.float32).ravel()
        t = generate_static_text(b, meta_row)
        mr = meta_row.iloc[0] if not meta_row.empty else {}
        ids.append(b)
        embs.append(e)
        texts.append(t)
        rows.append({
            "building_id": b,
            "usage": mr.get("primaryspaceusage") if hasattr(mr, "get") else None,
            "sub_usage": mr.get("sub_primaryspaceusage") if hasattr(mr, "get") else None,
            "site": mr.get("site_id") if hasattr(mr, "get") else None,
        })

    X = np.vstack(embs)
    xyz = _project_3d(X)

    usages = [r["usage"] for r in rows if r["usage"]]
    if n_clusters is None:
        n_clusters = max(2, min(8, len(set(usages)) or 2))
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

    meta = pd.DataFrame(rows)
    meta["x"] = xyz[:, 0]
    meta["y"] = xyz[:, 1]
    meta["z"] = xyz[:, 2]
    meta["cluster"] = labels.astype(int)
    meta["text_static"] = texts

    np.save(os.path.join(out_dir, "embeddings.npy"), X)
    meta.to_parquet(os.path.join(out_dir, "embedding_meta.parquet"), index=False)
    print(f"[demo] scris embeddings.npy {X.shape} + embedding_meta.parquet "
          f"({len(meta)} cladiri, {n_clusters} clustere)")
    return meta
