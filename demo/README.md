# Platforma demo - ghid de pornire

Backend FastAPI care serveste predictiile + inferenta live + statistici, cu metadata in Postgres.
Toate comenzile se ruleaza din **radacina proiectului** (`Proiect-Licenta/`), cu interpretorul din `.venv`.

## 0. Dependinte (o singura data)
```
python -m pip install -r demo/requirements.txt
```

## 1. Conexiunea la Postgres
Copiaza `demo/.env.example` ca `demo/.env` si completeaza cu parola si numele bazei tale:
```
DATABASE_URL=postgresql+psycopg2://postgres:PAROLA_TA@localhost:5432/licenta_demo
```
Verifica conexiunea:
```
python -m demo.db.check
```

## 2. Datele platformei (din notebook)
Platforma citeste fisierele din `demo/data/`, generate de celulele de export de la finalul
`Notebook-licenta.ipynb` (functiile din `demo/notebook_export.py`):

| Functie apelata | Ce scrie |
|---|---|
| `save_perhouse_dicts` | `perhouse_results.json` (rezultatele modelelor per cladire) |
| `save_global_predictions` | `predictions_global.parquet` (BiLSTM global, cu si fara text) |
| `save_embeddings` | `embeddings.npy` + `embedding_meta.parquet` |
| `save_infer_bundles` | `demo/data/infer/*.npz` (intrarile pentru inferenta live) |
| `save_patchtst` | `predictions_patchtst.parquet` |
| `save_platform_metrics`, `save_metrics_24h` | `metrics.parquet` (metrici pe orizontul de 24h) |

Antrenarea scrie checkpoint-urile `global_model_v10.pt`, `global_notext_v10.pt` si `patchtst_v10.pt`
in radacina proiectului (folosite de inferenta live).

## 3. Catalog cladiri + istoric
```
python demo/scripts/export_predictions.py
```
Scrie `buildings / history / predictions.parquet` in `demo/data/` (citeste `data_set_genome/metadata.csv`
si `electricity_cleaned.csv`). Necesita `predictions_global.parquet` de la pasul 2.

## 4. Baza de date
```
python -m demo.db.init_db    # creeaza tabelele
python -m demo.db.seed       # le umple din parquet (modele, cladiri, metrici, zile)
```

## 5. Porneste backend-ul
```
python -m uvicorn demo.backend.app:app --reload --port 8000
```
Documentatie interactiva: http://localhost:8000/docs

## Verificare rapida
- `GET /api/health` - cate cladiri, ce modele de predictie
- `GET /api/inference/status` - ce checkpoint-uri sunt incarcate (notext/hybrid; patchtst dupa antrenare)
- `GET /api/models` - registrul modelelor (din Postgres)
- `POST /api/infer` cu `{"building":"...","day":"YYYY-MM-DD","models":["global_notext","global_text","patchtst"]}`

## Ordinea pe scurt
```
notebook (antrenare + export)  ->  export_predictions.py  ->  init_db  ->  seed  ->  uvicorn
```

## Frontend (React)
In `demo/frontend/` (Vite + React + React Router). Cere backend-ul pornit pe :8000.
```
npm install --prefix demo/frontend     # o singura data
npm run dev --prefix demo/frontend     # porneste pe http://localhost:5173
```
Dev server-ul redirecteaza automat `/api/...` catre backend (:8000), deci nu ai griji de CORS.
Pagini: Acasa (`/`), Cladire (`/cladire`), Inferenta (`/inferenta`), Embeddings (`/embeddings`), Arhiva (`/arhiva`).

