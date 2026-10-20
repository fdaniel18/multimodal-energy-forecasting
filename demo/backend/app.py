from __future__ import annotations
from datetime import date
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .store import get_store
from .inference import get_engine

app = FastAPI(title="Predictii consum energetic - API", version="0.2")


class InferRequest(BaseModel):
    building: str
    day: str
    models: list[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    s = get_store()
    return {
        "status": "ok",
        "buildings": len(s.ids),
        "pred_models": s.pred_models,
        "metric_models": s.metric_models,
        "has_embeddings": s.embedding_meta is not None,
    }


@app.get("/api/buildings")
def buildings():
    return get_store().list_buildings()


@app.get("/api/buildings/{bid}")
def building(bid: str):
    info = get_store().building(bid)
    if info is None:
        raise HTTPException(404, f"Cladire necunoscuta: {bid}")
    return info


@app.get("/api/buildings/{bid}/predictions")
def predictions(bid: str,
                start: str | None = Query(None, description="ISO datetime"),
                end: str | None = Query(None, description="ISO datetime"),
                limit: int | None = Query(None)):
    s = get_store()
    if bid not in s.ids:
        raise HTTPException(404, f"Cladire necunoscuta: {bid}")
    return s.predictions_for(bid, start, end, limit)


@app.get("/api/buildings/{bid}/weeks")
def weeks(bid: str):
    return get_store().prediction_weeks(bid)


@app.get("/api/buildings/{bid}/history")
def history(bid: str, resample: str | None = Query("D", description="D, W, M sau gol pentru brut")):
    s = get_store()
    if bid not in s.ids:
        raise HTTPException(404, f"Cladire necunoscuta: {bid}")
    return s.history_for(bid, resample or None)


@app.get("/api/buildings/{bid}/profile")
def profile(bid: str):
    s = get_store()
    if bid not in s.ids:
        raise HTTPException(404, f"Cladire necunoscuta: {bid}")
    return {
        "weekday": s.weekday_profile(bid),
        "month": s.month_profile(bid),
        "hour": s.hourly_profile(bid),
    }


@app.get("/api/metrics")
def metrics():
    return get_store().metrics_summary()


@app.get("/api/models")
def models():
    try:
        from ..db.database import SessionLocal
        from ..db import schema
        s = SessionLocal()
        try:
            rows = s.query(schema.Model).all()
            if rows:
                avail = get_engine().available_models
                return [{
                    "name": m.name, "label": m.label, "kind": m.kind,
                    "n_params": m.n_params, "lookback": m.lookback, "forecast": m.forecast,
                    "hyperparams": m.hyperparams, "description": m.description,
                    "live": m.name in config.CHECKPOINTS,
                    "checkpoint_available": m.name in avail,
                } for m in rows]
        finally:
            s.close()
    except Exception as e:
        print(f"[api] DB indisponibil pentru /api/models ({e})")
    from ..db.seed import MODEL_REGISTRY
    avail = get_engine().available_models
    return [{**{k: m[k] for k in ("name", "label", "kind", "n_params",
                                   "lookback", "forecast", "hyperparams", "description")},
             "live": m["name"] in config.CHECKPOINTS,
             "checkpoint_available": m["name"] in avail} for m in MODEL_REGISTRY]


@app.get("/api/models/stats")
def models_stats():
    return get_store().models_overview()


@app.get("/api/models/compare")
def models_compare(metric: str = Query("rmse", description="rmse, mae, r2 sau wape")):
    return get_store().models_compare(metric)


@app.get("/api/inference/status")
def inference_status():
    return {"loaded_models": get_engine().available_models}


@app.get("/api/inference/runs")
def inference_runs(limit: int = Query(50)):
    try:
        from ..db.database import SessionLocal
        from ..db import schema
        s = SessionLocal()
        try:
            rows = (s.query(schema.InferenceRun)
                    .order_by(schema.InferenceRun.created_at.desc(),
                              schema.InferenceRun.id.desc())
                    .limit(limit).all())
            return [{
                "id": r.id, "building_id": r.building_id, "day": str(r.day),
                "model_names": r.model_names or [],
                "created_at": r.created_at.isoformat() if r.created_at else None,
            } for r in rows]
        finally:
            s.close()
    except Exception as e:
        print(f"[api] DB indisponibil pentru /api/inference/runs ({e}).")
        return []


@app.get("/api/inference/runs/{run_id}")
def inference_run(run_id: int):
    try:
        from ..db.database import SessionLocal
        from ..db import schema
        s = SessionLocal()
        try:
            r = s.get(schema.InferenceRun, run_id)
            if r is None:
                raise HTTPException(404, f"Rularea {run_id} nu exista.")
            return {"id": r.id, "building_id": r.building_id, "day": str(r.day),
                    "model_names": r.model_names or [],
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "result": r.result}
        finally:
            s.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Baza de date indisponibila: {e}")


@app.get("/api/buildings/{bid}/test-days")
def test_days(bid: str):
    return get_engine().test_days(bid)


@app.post("/api/infer")
def infer(req: InferRequest):
    eng = get_engine()
    try:
        result = eng.infer(req.building, req.day, req.models)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except KeyError as e:
        raise HTTPException(400, str(e))
    try:
        from ..db.database import SessionLocal
        from ..db import schema
        s = SessionLocal()
        try:
            s.add(schema.InferenceRun(building_id=req.building, day=date.fromisoformat(req.day),
                                      model_names=req.models, result=result))
            s.commit()
        finally:
            s.close()
    except Exception:
        pass
    return result


@app.get("/api/embeddings")
def embeddings():
    data = get_store().embedding_scatter()
    if data is None:
        raise HTTPException(404, "Embeddings inca neexportate din notebook (embedding_meta.parquet lipseste)")
    return data
