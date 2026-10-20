from __future__ import annotations
from sqlalchemy import inspect
from .database import engine, DATABASE_URL, SessionLocal
from . import schema


def main():
    safe_url = DATABASE_URL.split("@")[-1]
    try:
        with engine.connect():
            pass
    except Exception as e:
        print(f"NU ma pot conecta la {safe_url}\n  {e}")
        print("\nVerifica: Postgres porneste, baza exista, iar demo/.env are connection string-ul corect.")
        return 1

    print(f"Conectat la: {safe_url}")
    tables = inspect(engine).get_table_names()
    print("Tabele:", ", ".join(sorted(tables)) or "(niciuna - ruleaza init_db + seed)")
    if {"buildings", "models", "metrics"} <= set(tables):
        s = SessionLocal()
        try:
            print(f"  models   : {s.query(schema.Model).count()}")
            print(f"  buildings: {s.query(schema.Building).count()}")
            print(f"  metrics  : {s.query(schema.Metric).count()}")
            print(f"  test_days: {s.query(schema.TestDay).count()}")
            print(f"  runs     : {s.query(schema.InferenceRun).count()}")
        finally:
            s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
