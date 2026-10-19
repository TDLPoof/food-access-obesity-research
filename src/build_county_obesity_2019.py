# Run from the repository root after restoring data/raw inputs.
import pandas as pd, numpy as np, re, os, sys, json

os.makedirs("data/processed", exist_ok=True)
os.makedirs("logs", exist_ok=True)

LOG = []
def log(m): LOG.append(m); print(m)
def std_fips(s, width=5):
    if pd.isna(s): return np.nan
    s = re.sub(r"[^\d]", "", str(s))
    return s.zfill(width) if s else np.nan

# ---------- 1) PLACES (y: age-adjusted obesity %, 2019) ----------
places = pd.read_csv("data/raw/PLACES__Local_Data_for_Better_Health,_County_Data_2021_release_20251101.csv",
                     dtype={"LocationID": str})

# Be robust to schema differences (some releases use "Year" vs "Data_Value_Year")
cols = {c.lower(): c for c in places.columns}
YEAR    = cols.get("data_value_year") or cols.get("year")
TYPE    = cols.get("data_value_type") or cols.get("datavaluetype")
MEASURE = cols.get("measure")         # should be "Measure"
FIPS    = cols.get("locationid")      # should be "CountyFIPS"

if not all([YEAR, TYPE, MEASURE, FIPS]):
    raise RuntimeError(f"Unexpected PLACES columns: {list(places.columns)}")

mask = (
    places[MEASURE].str.contains("Obesity", case=False, na=False) &
    places[TYPE].str.contains("age", case=False, na=False) &
    (places[YEAR].astype(int) == 2019) &
    (places[FIPS] != "59")
)

y = places.loc[mask, [FIPS, "Data_Value"]].rename(
    columns={FIPS: "fips", "Data_Value": "obesity_adj_pct"}
)
y["fips"] = y["fips"].str.replace(r"[^\d]", "", regex=True).str.zfill(5)
y["y_prop"] = y["obesity_adj_pct"] / 100.0

log(f"PLACES rows kept: {len(y)}")

# # ---------- 7) 2019 population ----------
# pop = pd.read_csv("County Populations 2019.csv", encoding="ISO-8859-1", dtype=str)
# pop["fips"] = pop["STATE"] + pop["COUNTY"]
# pcol = next((c for c in pop.columns if re.search(r"pop.*2019|2019.*pop|population", c, re.I)), None)
# if not pcol: raise RuntimeError("Population: missing 2019 population column.")
# pop = pop[["fips", pcol]].rename(columns={pcol:"pop_2019"})
# pop["pop_2019"] = pd.to_numeric(pop["pop_2019"], errors="coerce")
# log(f"Population rows: {len(pop)}")

# ---------- 2) FARA (tract -> county low_access_share, 2019) ----------
fa = pd.read_csv(os.path.join("data/raw/FARA","Food Access Research Atlas.csv"), dtype=str)

fa["fips"] = fa["CensusTract"].str.zfill(11).str[:5]
needed = ["Pop2010","LAPOP1_10"]  # 1 mile (urban) / 10 miles (rural) combined definition
missing = [c for c in needed if c not in fa.columns]
if missing:
    raise RuntimeError(f"FARA missing expected columns: {missing}. "
                       f"Present columns include: {fa.columns[:20].tolist()} ...")

fa["Pop2010"]   = pd.to_numeric(fa["Pop2010"], errors="coerce")
fa["LAPOP1_10"] = pd.to_numeric(fa["LAPOP1_10"], errors="coerce")

fara_cnty = (fa.groupby("fips", as_index=False)
               .agg(low_access_pop=("LAPOP1_10","sum"),
                    tract_pop_2010=("Pop2010","sum")))
fara_cnty["low_access_share"] = fara_cnty["low_access_pop"] / fara_cnty["tract_pop_2010"]

# Optional: low-income variant if available
if "LALOWI1_10" in fa.columns:
    fa["LALOWI1_10"] = pd.to_numeric(fa["LALOWI1_10"], errors="coerce")
    tmp = fa.groupby("fips", as_index=False)["LALOWI1_10"].sum().rename(columns={"LALOWI1_10":"lowi_pop"})
    fara_cnty = fara_cnty.merge(tmp, on="fips", how="left")
    fara_cnty["lowi_low_access_share"] = fara_cnty["lowi_pop"] / fara_cnty["tract_pop_2010"]

fara_keep = ["fips","low_access_share"] + (["lowi_low_access_share"] if "lowi_low_access_share" in fara_cnty.columns else [])
fara_cnty = fara_cnty[fara_keep]
log(f"FARA counties: {len(fara_cnty)}; mean low_access_share={fara_cnty['low_access_share'].mean():.3f}")

# ---------- 3) Food Environment Atlas (2016 per-1k proxies) ----------
AGG = "first"                 # how to handle duplicates per (FIPS,State,County,Variable_Code): "first", "max", "sum", ...

# 1) Load as strings (preserves leading zeros in FIPS) and coerce Value to numeric
dtypes = {"FIPS": "string", "State": "string", "County": "string", "Variable_Code": "string"}
df = pd.read_csv("data/raw/FoodEnvironmentAtlas/StateAndCountyData.csv", dtype=dtypes)
df = df[df["County"] != "Total"]
df["FIPS"] = df["FIPS"].str.zfill(5)
df.drop(['State', 'County'], axis = 1, inplace = True)
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# 2) Pivot: rows = (FIPS, State, County), columns = Variable_Code, values = Value
fea = df.pivot_table(
    index=["FIPS"],
    columns="Variable_Code",
    values="Value",
    aggfunc=AGG,
)

# 4) Order columns naturally: V1, V2, ..., V10, ...
def _nat_key(col):
    base = col.split("_")[0]  # "V123" from "V123_Value"
    m = re.fullmatch(r"V(\d+)", base)
    return (0, int(m.group(1))) if m else (1, base)

fea = fea.sort_index(axis=1, key=lambda cols: [ _nat_key(c) for c in cols ])

# 5) Back to a regular frame and write files
fea = fea.reset_index()
fea.rename(columns={"FIPS": "fips"}, inplace=True)
PTH = {
    "FFRPTH16":  "unhealthy_ffr_pth",      # fast-food / 1,000 pop
    "CONVSPTH16":"unhealthy_convs_pth",    # convenience / 1,000 pop
    "GROCPTH16": "healthy_groc_pth",       # grocery / 1,000 pop
    "SPECSPTH16":"healthy_specs_pth",      # specialized food stores / 1,000 pop
}

# 2016 counts (fallback if a PTH column is missing)
CNT = {
    "FFR16":   "unhealthy_ffr_cnt",
    "CONVS16": "unhealthy_convs_cnt",
    "GROC16":  "healthy_groc_cnt",
    "SPECS16": "healthy_specs_cnt",
}

# Pull whatever per-1k columns exist
env_cols = ["fips"] + [c for c in PTH if c in fea.columns]
env = fea[env_cols].copy()

need_fallback = any(c not in fea.columns for c in PTH)
if need_fallback:
    # Use counts -> convert to per-1k via your County Populations 2019 file
    cnt_cols = ["fips"] + [c for c in CNT if c in fea.columns]
    env = fea[cnt_cols].copy()

    pop = pd.read_csv("data/raw/County Populations 2019.csv", dtype=str)
    # normalize FIPS and find 2019 pop column
    if "fips" not in [c.lower() for c in pop.columns]:
        fips_col = next(c for c in pop.columns if re.search(r"fips", c, re.I))
        pop["fips"] = pop[fips_col].astype(str).str.extract(r"(\d+)").astype(str).str.zfill(5)
    popcol = next(c for c in pop.columns if re.search(r"(?:^|_)pop(?:ulation)?_?2019$|2019.*pop|pop.*2019", c, re.I))
    pop = pop[["fips", popcol]].rename(columns={popcol:"pop_2019"})
    pop["pop_2019"] = pd.to_numeric(pop["pop_2019"], errors="coerce")

    env = env.merge(pop, on="fips", how="left")
    for code in CNT:
        if code in env.columns:
            env[code+"_PTH16"] = pd.to_numeric(env[code], errors="coerce") / env["pop_2019"] * 1000.0

    # unify names to PTH keys the rest of the pipeline expects
    rename_map = {
        "FFR16_PTH16":   "FFRPTH16",
        "CONVS16_PTH16": "CONVSPTH16",
        "GROC16_PTH16":  "GROCPTH16",
        "SPECS16_PTH16": "SPECSPTH16",
    }
    env = env.rename(columns=rename_map)

# Ensure all expected PTH columns exist
for req in ["FFRPTH16","CONVSPTH16","GROCPTH16","SPECSPTH16"]:
    if req not in env.columns:
        env[req] = np.nan

# Cast and compute densities
for req in ["FFRPTH16","CONVSPTH16","GROCPTH16","SPECSPTH16"]:
    env[req] = pd.to_numeric(env[req], errors="coerce")

# Treat missing specialty density as 0 (common, sparse)
env["SPECSPTH16"] = env["SPECSPTH16"].fillna(0)

env["unhealthy_density"] = env["FFRPTH16"].fillna(0) + env["CONVSPTH16"].fillna(0)
env["healthy_density"]   = env["GROCPTH16"].fillna(0) + env["SPECSPTH16"].fillna(0)
env["swamp_index"]       = env["unhealthy_density"] - env["healthy_density"]

# Optional: log ratio (nice for linear models)
eps = 0.1
env["swamp_logratio"] = np.log((env["unhealthy_density"]+eps) / (env["healthy_density"]+eps))

env = env[["fips","unhealthy_density","healthy_density","swamp_index","swamp_logratio"]].drop_duplicates("fips")
log(f"FEA env rows: {len(env)}")


# ---------- 4) SAIPE (2019 county poverty %) ----------
saipe = pd.read_excel("data/raw/SAIPE.xls", sheet_name="est19ALL", dtype=str, skiprows=[0, 1, 2])
saipe = saipe[saipe["County FIPS Code"] != "000"]
saipe["fips"] = saipe["State FIPS Code"] + saipe["County FIPS Code"]
saipe = saipe[["fips", "Poverty Percent, All Ages"]].rename(columns={"Poverty Percent, All Ages": "poverty_pct"})
saipe["poverty_pct"] = pd.to_numeric(saipe["poverty_pct"], errors="coerce")
log(f"SAIPE rows: {len(saipe)}")

# ---------- 5) LAUS (2019 unemployment rate %) ----------
laus = pd.read_excel("data/raw/BLS LAUS 2019 unemployment.xlsx", sheet_name=0, dtype=str, skiprows=[0])
laus = laus[pd.notna(laus["State FIPS Code"])]
laus = laus[laus["State FIPS Code"] != "72"] # Exclude PR
laus["fips"] = laus["State FIPS Code"] + laus["County FIPS Code"]
laus = laus[["fips", "Unemployment Rate (%)"]].rename(columns={"Unemployment Rate (%)": "unemp_rate"})
laus["unemp_rate"] = pd.to_numeric(laus["unemp_rate"], errors="coerce")
log(f"LAUS rows: {len(laus)}")

# ---------- 6) ACS education (S1501: hs_plus_pct, ba_plus_pct) ----------
with open("data/raw/CensusEducationData.csv", "r", encoding="utf-8") as f:
    data = json.load(f)
edu = pd.DataFrame(data[1:], columns=data[0])
edu = edu[edu["state"] != "72"]
edu["fips"] = edu["state"] + edu["county"]
edu = edu[["fips", "S1501_C02_014E", "S1501_C02_015E"]].rename(columns={"S1501_C02_014E": "hs_plus_pct", "S1501_C02_015E": "ba_plus_pct"})
edu["hs_plus_pct"] = pd.to_numeric(edu["hs_plus_pct"], errors="coerce")
edu["ba_plus_pct"] = pd.to_numeric(edu["ba_plus_pct"], errors="coerce")
log(f"Education rows: {len(edu)}")

# ---------- 8) RUCC (2013) ----------
rucc_df = pd.read_excel("data/raw/ruralurbancodes2013.xls", sheet_name=0, dtype=str)
rucc_df = rucc_df[~rucc_df["State"].isin(["AS", "GU", "MP", "VI", "PR"])]
rucc = rucc_df[["FIPS", "RUCC_2013"]].rename(columns={"FIPS":"fips", "RUCC_2013":"rucc"})
rucc = pd.get_dummies(rucc, columns=["rucc"], prefix="rucc", drop_first=True)
log(f"RUCC dummies: {rucc.shape[1]-1}")

# ---------- 9) Join all ----------
df = (y[["fips","y_prop","obesity_adj_pct"]]
      .merge(fara_cnty[["fips","low_access_share"]], on="fips", how="left")
      .merge(env, on="fips", how="left")
      .merge(saipe, on="fips", how="left")
      .merge(edu, on="fips", how="left")
      .merge(laus, on="fips", how="left")
      .merge(rucc, on="fips", how="left"))
df = df.drop_duplicates(subset=["fips"])
log(f"Final shape: {df.shape}")
log("Sample:\n" + df.head(10).to_string(index=False))

# ---------- 10) Save ----------
df.dropna(inplace=True)
df.to_csv("data/processed/county_obesity_2019_model_data.csv", index=False)
with open("logs/data_prep_log.txt","w") as f: f.write("\n".join(LOG))
print("\nWrote county_obesity_2019_model_data.csv and data_prep_log.txt")
