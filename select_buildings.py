import json
import pandas as pd
import numpy as np

DATA = "data_set_genome"
NOTEBOOK = "Notebook-licenta.ipynb"

MIN_HOURS = 8760
MAX_ZERO_FRAC = 0.15
MIN_CV = 0.10
N_TO_ADD = 60

QUOTA = {
    "Education": 24,
    "Office": 20,
}
OTHER_TOTAL = N_TO_ADD - sum(QUOTA.values())

nb = json.load(open(NOTEBOOK, encoding="utf-8"))
cell0 = "".join(nb["cells"][0]["source"])
raw = cell0.split("TARGET_BUILDINGS")[1].split("[", 1)[1].split("]", 1)[0]
current = [x.strip().strip("'\"") for x in raw.split(",") if x.strip().strip("'\"")]
print(f"Set curent: {len(current)} cladiri")

md = pd.read_csv(f"{DATA}/metadata.csv")
weather_sites = set(pd.read_csv(f"{DATA}/weather.csv", usecols=["site_id"])["site_id"].dropna().unique())

print("Citesc electricity.csv (poate dura)...")
elec = pd.read_csv(f"{DATA}/electricity.csv")
val_cols = [c for c in elec.columns if c != "timestamp"]

stats = {}
for c in val_cols:
    s = elec[c]
    nn = int(s.notna().sum())
    if nn == 0:
        continue
    v = s.dropna()
    mean = float(v.mean())
    zero_frac = float((v == 0).mean())
    cv = float(v.std() / mean) if mean > 0 else 0.0
    stats[c] = (nn, zero_frac, cv)

st = pd.DataFrame(
    [(b, *stats[b]) for b in stats],
    columns=["building_id", "nn", "zero_frac", "cv"],
)

md = md.merge(st, on="building_id", how="inner")
md["area"] = md["sqm"].fillna(md["sqft"] * 0.092903)

def richness(r):
    score = 0
    for f in ("yearbuilt", "numberoffloors", "occupants", "heatingtype"):
        score += int(pd.notna(r.get(f)))
    for m in ("hotwater", "chilledwater", "steam", "gas", "water"):
        score += int(str(r.get(m, "")).strip().lower() == "yes")
    return score

md["rich"] = md.apply(richness, axis=1)

cand = md[
    (~md["building_id"].isin(current))
    & (md["site_id"].isin(weather_sites))
    & (md["nn"] >= MIN_HOURS)
    & (md["zero_frac"] <= MAX_ZERO_FRAC)
    & (md["cv"] >= MIN_CV)
    & (md["area"].notna())
    & (md["primaryspaceusage"].notna())
].copy()

cand["nn_norm"] = (cand["nn"] - cand["nn"].min()) / (cand["nn"].max() - cand["nn"].min() + 1e-9)
cand["score"] = cand["nn_norm"] + 0.25 * cand["rich"]
cand = cand.sort_values("score", ascending=False)

print(f"\nCandidati care trec toate filtrele: {len(cand)}")
print(cand["primaryspaceusage"].value_counts().to_string())

picked = []
for usage, q in QUOTA.items():
    sub = cand[cand["primaryspaceusage"] == usage].head(q)
    picked.append(sub)

used_types = set(QUOTA)
rest = cand[~cand["primaryspaceusage"].isin(used_types)]
by_type = {u: g.copy() for u, g in rest.groupby("primaryspaceusage")}
others = []
while len(others) < OTHER_TOTAL and by_type:
    for u in list(by_type):
        if not by_type[u].empty and len(others) < OTHER_TOTAL:
            others.append(by_type[u].iloc[0])
            by_type[u] = by_type[u].iloc[1:]
        if by_type[u].empty:
            del by_type[u]
if others:
    picked.append(pd.DataFrame(others))

selection = pd.concat(picked).drop_duplicates("building_id")

print(f"\n=== {len(selection)} cladiri selectate de adaugat ===")
print(selection["primaryspaceusage"].value_counts().to_string())

after = pd.concat([
    md[md.building_id.isin(current)]["primaryspaceusage"],
    selection["primaryspaceusage"],
])
print(f"\n=== Distributie DUPA adaugare (total {len(current) + len(selection)}) ===")
print(after.value_counts().to_string())

with open("cladiri_de_adaugat.txt", "w", encoding="utf-8") as f:
    for b in selection["building_id"]:
        f.write(f"    '{b}',\n")
print("\nLista scrisa in cladiri_de_adaugat.txt (gata de lipit in TARGET_BUILDINGS).")
print(selection[["building_id", "primaryspaceusage", "area", "nn", "rich", "cv"]]
      .to_string(index=False))
