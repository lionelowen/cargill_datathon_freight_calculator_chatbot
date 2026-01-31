# coba.py
# ============================================================
# FULL VERSION (includes everything in one file)
# Fixes:
#   1) Cargill hire_rate conversion (11.75 -> 11,750/day) when < 1000
#   2) Better default for port_mgo_per_day (use 2.0 instead of 0.1)
#   3) Safe handling if a vessel is missing BALLAST/LADEN row (skip instead of crash)
#   4) Better port normalisation (accents + aliases) to reduce distance mismatches
#   5) Time handling: laycan filter using vessel estimate_time_of_departure + ETA(load)
# Adds:
#   6) Optimal assignment for ALL committed cargoes (3 cargos) using Cargill + Market vessels
#   7) Optional: assign MARKET cargo to UNUSED Cargill vessels (1 cargo per vessel, no cargo reused)
#   8) Print mismatches (distance lookup fail) clearly (BAL / LAD)
#   9) Print NaN profit rows for debugging
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import math
import itertools
import re
import unicodedata

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

BUNKER_XLSX = DATA_DIR / "bunker.xlsx"
CARGILL_VESSEL_XLSX = DATA_DIR / "cargill_vessel.xlsx"
MARKET_VESSEL_XLSX = DATA_DIR / "market_vessel.xlsx"
COMMITTED_CARGO_XLSX = DATA_DIR / "committed_cargoes.xlsx"
MARKET_CARGO_XLSX = DATA_DIR / "market_cargoes.xlsx"
DIST_XLSX = DATA_DIR / "Port Distances.xlsx"
FFA_REPORT_XLSX = DATA_DIR / "ffa_report.xlsx"


# ============================================================
# Port normalisation (reduces mismatch)
# ============================================================
PORT_ALIASES = {
    # accent / alternate spellings
    "TUBARÃO": "TUBARAO",
    "TUBARAO": "TUBARAO",

    # common "anchorage" style differences (edit to match your distance file)
    "PORT KAMSAR": "KAMSAR",
    # if your distance file actually uses "KAMSAR ANCHORAGE", comment above out
    # "KAMSAR": "KAMSAR ANCHORAGE",
}


def _norm_port(p: str) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    s = str(p).strip().upper()

    # remove accents (TUBARÃO -> TUBARAO)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # normalise whitespace
    s = re.sub(r"\s+", " ", s)

    # apply aliases
    s = PORT_ALIASES.get(s, s)
    return s


def _pct_to_float(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    s = str(x).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


# ============================================================
# Dataclasses
# ============================================================
@dataclass
class Vessel:
    name: str = "mv cargill tbn"
    dwt: float = 62000
    grain_capacity_cbm: float = 70000

    sp_ballast: float = 14
    sp_laden: float = 12

    ifo_ballast: float = 23
    ifo_laden: float = 18.5
    mdo_ballast: float = 0.1
    mdo_laden: float = 0.1

    ifo_port_work: float = 5.5
    mdo_port_work: float = 0.1
    ifo_port_idle: float = 5.5
    mdo_port_idle: float = 0.1

    daily_hire: float = 12
    adcoms: float = 0.0375

    position: str = ""
    time_of_departure: pd.Timestamp | None = None

    # Remaining fuel onboard (MT)
    ifo_remaining: float = 0.0
    mdo_remaining: float = 0.0


@dataclass
class Cargo:
    cargo_qty: float = 60500
    stow_factor: float = 1.33

    freight: float = 22
    address_coms: float = 0.0375
    broker_coms: float = 0.0125

    load_rate: float = 8000
    dis_rate: float = 11000

    loadport_tt: float = 0.5
    disport_tt: float = 0.5
    port_idle: float = 0.5

    ballast_bonus: float = 0.0

    name: str = ""
    load_port: str = ""
    discharge_port: str = ""
    total_port_cost: float = 0.0
    earliest_date: pd.Timestamp | None = None
    latest_date: pd.Timestamp | None = None

    # For alternative discharge ports (same TCE basis)
    base_cargo_id: str = ""  # links alternative port options to same cargo
    alt_port_index: int = 0   # 0 = primary, 1+ = alternatives


@dataclass
class Voyage:
    ballast_leg_nm: float = 3000
    laden_leg_nm: float = 3000
    bunker_days: float = 1.0


@dataclass
class PricesAndFees:
    ifo_price: float = 440
    mdo_price: float = 850

    # Previous month prices (for remaining fuel valuation)
    ifo_price_prev: float = 440
    mdo_price_prev: float = 850

    ## insurance, survey etc. (fixed per-voyage fees)
    awrp: float = 1500
    cev: float = 1500
    ilhoc: float = 5000

    bunker_da_per_day: float = 1500


# ============================================================
# Distance lookup
# ============================================================
def load_distance_lookup(dist_xlsx: Path) -> dict[tuple[str, str], float]:
    df = pd.read_excel(dist_xlsx, sheet_name=0)
    df["PORT_NAME_FROM"] = df["PORT_NAME_FROM"].map(_norm_port)
    df["PORT_NAME_TO"] = df["PORT_NAME_TO"].map(_norm_port)
    df["DISTANCE"] = pd.to_numeric(df["DISTANCE"], errors="coerce")
    df = df[df["DISTANCE"].notna()].copy()

    lut: dict[tuple[str, str], float] = {}
    for r in df.itertuples(index=False):
        a, b, d = r.PORT_NAME_FROM, r.PORT_NAME_TO, float(r.DISTANCE)
        lut[(a, b)] = d
        lut[(b, a)] = d
    return lut
## return value{
## ("SINGAPORE", "QINGDAO"): 2500.0,
##  ("QINGDAO", "SINGAPORE"): 2500.0



## return distance
def dist_nm(lut: dict, a: str, b: str) -> float:
    a, b = _norm_port(a), _norm_port(b)
    if (a, b) in lut:
        return float(lut[(a, b)])
    raise KeyError(f"Distance not found for {a} -> {b}")



# ============================================================
# Debug printers
# ============================================================
def print_distance_mismatches(vessels, cargoes, dist_lut, title=""):
    """
    Prints which exact distance pair is missing:
      - BAL: vessel.position -> cargo.load_port
      - LAD: cargo.load_port -> cargo.discharge_port
    """
    print(f"\n==== DISTANCE MISMATCHES: {title} ====")
    found = False

    for v in vessels:
        for c in cargoes:
            if not v.position or not c.load_port or not c.discharge_port:
                print(f"[BLANK PORT] vessel='{v.name}' pos='{v.position}' load='{c.load_port}' dis='{c.discharge_port}'")
                found = True
                continue

            try:
                _ = dist_nm(dist_lut, v.position, c.load_port)
            except KeyError:
                print(f"[MISSING BAL] vessel='{v.name}'  {v.position} -> {c.load_port}  (cargo='{c.name}')")
                found = True
                continue

            try:
                _ = dist_nm(dist_lut, c.load_port, c.discharge_port)
            except KeyError:
                print(f"[MISSING LAD] cargo='{c.name}'  {c.load_port} -> {c.discharge_port}  (vessel='{v.name}')")
                found = True
                continue

    if not found:
        print("No distance mismatches found.")


def print_missing_port_pairs(vessels, cargoes, dist_lut, title=""):
    """
    Prints ONLY the unique missing port pairs (no duplicates).
    Returns two sets: missing_bal_pairs, missing_lad_pairs
    """
    print(f"\n==== MISSING PORT PAIRS: {title} ====")
    missing_bal = set()  # (from_port, to_port)
    missing_lad = set()  # (from_port, to_port)

    for v in vessels:
        for c in cargoes:
            if not v.position or not c.load_port or not c.discharge_port:
                continue

            # Check ballast leg
            a, b = _norm_port(v.position), _norm_port(c.load_port)
            if (a, b) not in dist_lut and (b, a) not in dist_lut:
                missing_bal.add((a, b))

            # Check laden leg
            a2, b2 = _norm_port(c.load_port), _norm_port(c.discharge_port)
            if (a2, b2) not in dist_lut and (b2, a2) not in dist_lut:
                missing_lad.add((a2, b2))

    if missing_bal:
        print(f"\n[BALLAST LEG] {len(missing_bal)} missing pairs:")
        for frm, to in sorted(missing_bal):
            print(f"  {frm} -> {to}")

    if missing_lad:
        print(f"\n[LADEN LEG] {len(missing_lad)} missing pairs:")
        for frm, to in sorted(missing_lad):
            print(f"  {frm} -> {to}")

    if not missing_bal and not missing_lad:
        print("No missing port pairs.")

    return missing_bal, missing_lad


def save_missing_pairs_to_excel(all_missing_pairs: dict, output_path: Path):
    """
    Save all missing port pairs to an Excel file.
    all_missing_pairs: dict with keys like "CARGILL x COMMITTED" and values (missing_bal, missing_lad)
    """
    rows = []
    for title, (missing_bal, missing_lad) in all_missing_pairs.items():
        for frm, to in sorted(missing_bal):
            rows.append({
                "Category": title,
                "Leg_Type": "BALLAST",
                "From_Port": frm,
                "To_Port": to,
                "Distance_NM": ""  # Empty for user to fill in
            })
        for frm, to in sorted(missing_lad):
            rows.append({
                "Category": title,
                "Leg_Type": "LADEN",
                "From_Port": frm,
                "To_Port": to,
                "Distance_NM": ""  # Empty for user to fill in
            })
    
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    print(f"\nSaved {len(rows)} missing port pairs to: {output_path}")
    return df


def print_nan_profit_rows(df: pd.DataFrame, title=""):
    if df is None or len(df) == 0:
        print(f"\n==== NaN PROFIT ROWS: {title} ====\n(no rows)")
        return
    bad = df[df["profit_loss"].isna()].copy()
    print(f"\n==== NaN PROFIT ROWS: {title} ====")
    if len(bad) == 0:
        print("No NaN profit rows.")
        return
    cols = [
        "vessel", "cargo", "from_pos", "load_port", "discharge_port",
        "ballast_nm", "laden_nm",
        "revenue", "hire_net", "bunker_expense", "misc_expense", "profit_loss"
    ]
    cols = [c for c in cols if c in bad.columns]
    print(bad[cols].head(50))


# ============================================================
# Bunker prices
# ============================================================

# Port-to-bunker-location mapping (nearest bunkering hub)
PORT_TO_BUNKER_HUB = {
    # China ports -> Qingdao (or Shanghai)
    "QINGDAO": "Qingdao",
    "SHANGHAI": "Shanghai",
    "FANGCHENG": "Qingdao",
    "CAOFEIDIAN": "Qingdao",
    "XIAMEN": "Shanghai",
    "JINGTANG": "Qingdao",
    "TIANJIN": "Qingdao",
    "RIZHAO": "Qingdao",
    "LIANYUNGANG": "Qingdao",
    "NINGBO": "Shanghai",
    "GUANGZHOU": "Shanghai",
    
    # Korea/Japan -> Singapore (or Qingdao)
    "GWANGYANG": "Singapore",
    "BUSAN": "Singapore",
    "POHANG": "Singapore",
    
    # Southeast Asia -> Singapore
    "SINGAPORE": "Singapore",
    "MAP TA PHUT": "Singapore",
    "TABONEO": "Singapore",
    "TELUK RUBIAH": "Singapore",
    
    # India -> Singapore or Fujairah
    "PARADIP": "Singapore",
    "VIZAG": "Singapore",
    "KANDLA": "Fujairah",
    "MUNDRA": "Fujairah",
    "KRISHNAPATNAM": "Singapore",
    "MANGALORE": "Singapore",
    "CHENNAI": "Singapore",
    
    # Middle East -> Fujairah
    "FUJAIRAH": "Fujairah",
    "JUBAIL": "Fujairah",
    "JEBEL ALI": "Fujairah",
    "RAS LAFFAN": "Fujairah",
    
    # Europe -> Rotterdam or Gibraltar
    "ROTTERDAM": "Rotterdam",
    "PORT TALBOT": "Rotterdam",
    "AMSTERDAM": "Rotterdam",
    "ANTWERP": "Rotterdam",
    "DUNKIRK": "Rotterdam",
    "GIBRALTAR": "Gibraltar",
    
    # West Africa -> Gibraltar or Port Louis
    "KAMSAR": "Gibraltar",
    "KAMSAR ANCHORAGE": "Gibraltar",
    "CONAKRY": "Gibraltar",
    
    # South Africa -> Durban or Richards Bay
    "DURBAN": "Durban",
    "RICHARDS BAY": "Richards Bay",
    "SALDANHA BAY": "Durban",
    
    # Brazil -> Gibraltar (transatlantic) or Durban (via Cape)
    "TUBARAO": "Gibraltar",
    "PONTA DA MADEIRA": "Gibraltar",
    "ITAGUAI": "Gibraltar",
    "SANTOS": "Gibraltar",
    
    # Australia -> Singapore
    "PORT HEDLAND": "Singapore",
    "DAMPIER": "Singapore",
    "NEWCASTLE": "Singapore",
    "HAY POINT": "Singapore",
    
    # Mauritius
    "PORT LOUIS": "Port Louis",
}


def get_bunker_location_for_port(port: str, default: str = "Singapore") -> str:
    """
    Returns the nearest bunkering hub for a given port.
    Falls back to default (Singapore) if port not found.
    """
    port_norm = _norm_port(port)
    
    # Direct match
    if port_norm in PORT_TO_BUNKER_HUB:
        return PORT_TO_BUNKER_HUB[port_norm]
    
    # Partial match (for ports like "PORT HEDLAND ANCHORAGE")
    for key, hub in PORT_TO_BUNKER_HUB.items():
        if key in port_norm or port_norm in key:
            return hub
    
    return default


#using bunker file to look for fuel prices
def load_bunker_prices(bunker_xlsx: Path, location="Singapore", month_col="cal", prev_month_col="cal") -> tuple[float, float, float, float]:
    """
    Returns (vlsfo_current, mgo_current, vlsfo_prev, mgo_prev)
    """
    df = pd.read_excel(bunker_xlsx, sheet_name=0)
    df["location"] = df["location"].astype(str).str.strip().str.upper()
    df["type"] = df["type"].astype(str).str.strip().str.upper()

    loc = location.strip().upper()
    vlsfo_curr = df.loc[(df["location"] == loc) & (df["type"].isin(["VLSFO", "LSFO", "IFO"])), month_col].iloc[0]
    mgo_curr = df.loc[(df["location"] == loc) & (df["type"].isin(["MGO", "MDO"])), month_col].iloc[0]
    vlsfo_prev = df.loc[(df["location"] == loc) & (df["type"].isin(["VLSFO", "LSFO", "IFO"])), prev_month_col].iloc[0]
    mgo_prev = df.loc[(df["location"] == loc) & (df["type"].isin(["MGO", "MDO"])), prev_month_col].iloc[0]
    return float(vlsfo_curr), float(mgo_curr), float(vlsfo_prev), float(mgo_prev)


def load_all_bunker_prices(bunker_xlsx: Path, month_col="cal", prev_month_col="cal") -> dict[str, tuple[float, float, float, float]]:
    """
    Load bunker prices for ALL locations.
    Returns dict: {location: (vlsfo_curr, mgo_curr, vlsfo_prev, mgo_prev)}
    """
    df = pd.read_excel(bunker_xlsx, sheet_name=0)
    df["location"] = df["location"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    
    prices = {}
    for loc in df["location"].unique():
        loc_df = df[df["location"] == loc]
        try:
            vlsfo_curr = loc_df.loc[loc_df["type"].isin(["VLSFO", "LSFO", "IFO"]), month_col].iloc[0]
            mgo_curr = loc_df.loc[loc_df["type"].isin(["MGO", "MDO"]), month_col].iloc[0]
            vlsfo_prev = loc_df.loc[loc_df["type"].isin(["VLSFO", "LSFO", "IFO"]), prev_month_col].iloc[0]
            mgo_prev = loc_df.loc[loc_df["type"].isin(["MGO", "MDO"]), prev_month_col].iloc[0]
            prices[loc] = (float(vlsfo_curr), float(mgo_curr), float(vlsfo_prev), float(mgo_prev))
        except (IndexError, KeyError):
            continue
    
    return prices


# ============================================================
# FFA 5TC hire
# ============================================================
def load_ffa(ffa_xlsx: Path) -> pd.DataFrame:
    ffa = pd.read_excel(ffa_xlsx, sheet_name=0)
    ffa["Route"] = ffa["Route"].astype(str).str.strip()
    return ffa


#
def _find_5tc_row(ffa: pd.DataFrame) -> pd.Series:
    m = ffa["Route"].str.contains("5TC", case=False, na=False)
    if not m.any():
        raise KeyError("Cannot find a 5TC row in ffa_report.xlsx (column 'Route').")
    return ffa.loc[m].iloc[0]
    # filtering row and give the first encountered row



def ffa_5tc_usd_per_day(ffa: pd.DataFrame, dep: pd.Timestamp | None) -> float:
    row = _find_5tc_row(ffa)

    monthly_cols = [c for c in ffa.columns if isinstance(c, pd.Timestamp)]
    monthly_cols = sorted(monthly_cols)

    rate = None

    if dep is not None and pd.notna(dep) and monthly_cols:
        dep = pd.Timestamp(dep)
        month_key = pd.Timestamp(dep.year, dep.month, 1)

        if month_key in monthly_cols and pd.notna(row[month_key]):
            rate = float(row[month_key])
        else:
            earlier = [c for c in monthly_cols if c <= dep and pd.notna(row[c])]
            later = [c for c in monthly_cols if c >= dep and pd.notna(row[c])]
            if earlier and later:
                a = max(earlier)
                b = min(later)
                if a == b:
                    rate = float(row[a])
                else:
                    ra, rb = float(row[a]), float(row[b])
                    w = (dep - a) / (b - a)
                    rate = ra + float(w) * (rb - ra)

    if rate is None and dep is not None and pd.notna(dep):
        q = (int(dep.month) - 1) // 3 + 1
        qcol = f"Q{q} {str(dep.year)[-2:]}"
        if qcol in ffa.columns and pd.notna(row[qcol]):
            rate = float(row[qcol])
        else:
            calcol = f"Cal {str(dep.year)[-2:]}"
            if calcol in ffa.columns and pd.notna(row[calcol]):
                rate = float(row[calcol])

    if rate is None:
        cal_cols = [c for c in ffa.columns if isinstance(c, str) and c.strip().lower().startswith("cal")]
        for c in cal_cols:
            if pd.notna(row[c]):
                rate = float(row[c])
                break

    if rate is None:
        raise ValueError("Could not derive 5TC hire from FFA table.")

    # Convert rate from thousands to actual USD/day (14.157 -> 14,157)
    if rate < 1000:
        rate = rate * 1000

    return rate


# ============================================================
# Estimate freight rate from FFA 5TC
# ============================================================
# Route-based typical voyage days and distances for Panamax/Capesize
ROUTE_PROFILES = {
    # (load_region, discharge_region): (typical_voyage_days, typical_cargo_mt)
    # Australia -> China (C5)
    ("AUSTRALIA", "CHINA"): (35, 170000),
    ("DAMPIER", "QINGDAO"): (35, 170000),
    ("PORT HEDLAND", "QINGDAO"): (35, 170000),
    ("PORT HEDLAND", "GWANGYANG"): (32, 165000),
    
    # Brazil -> China (C3)
    ("BRAZIL", "CHINA"): (85, 180000),
    ("TUBARAO", "QINGDAO"): (85, 180000),
    ("PONTA DA MADEIRA", "QINGDAO"): (90, 190000),
    ("PONTA DA MADEIRA", "CAOFEIDIAN"): (90, 190000),
    ("ITAGUAI", "QINGDAO"): (85, 180000),
    
    # South Africa -> China
    ("SOUTH AFRICA", "CHINA"): (55, 180000),
    ("SALDANHA BAY", "TIANJIN"): (55, 180000),
    
    # West Africa -> India/China
    ("WEST AFRICA", "CHINA"): (75, 175000),
    ("KAMSAR ANCHORAGE", "QINGDAO"): (75, 175000),
    ("KAMSAR ANCHORAGE", "MANGALORE"): (45, 175000),
    
    # Indonesia -> India
    ("INDONESIA", "INDIA"): (25, 150000),
    ("TABONEO", "KRISHNAPATNAM"): (25, 150000),
    
    # Canada -> China
    ("CANADA", "CHINA"): (45, 160000),
    ("VANCOUVER", "FANGCHENG"): (45, 160000),
    
    # Brazil -> Malaysia
    ("BRAZIL", "MALAYSIA"): (70, 180000),
    ("TUBARAO", "TELUK RUBIAH"): (70, 180000),
    
    # Default fallback
    ("DEFAULT", "DEFAULT"): (50, 170000),
}


def estimate_freight_from_ffa(
    ffa: pd.DataFrame,
    load_port: str,
    discharge_port: str,
    cargo_qty: float,
    dep_date: pd.Timestamp | None = None
) -> float:
    """
    Estimate freight rate ($/MT) based on FFA TCE rates and route benchmarks.
    
    FFA Routes provide TCE (Time Charter Equivalent = daily revenue after voyage costs):
    - C3: Tubarao–Qingdao (Brazil to China) - ~$17,000-21,000/day
    - C5: West Australia–Qingdao - ~$6,600-8,700/day
    - C7: Bolivar–Rotterdam - ~$10,600-11,800/day
    
    Benchmark freight rates (from committed cargoes):
    - Brazil -> China: ~$22/MT
    - Australia -> China: ~$9/MT  
    - West Africa -> China: ~$23/MT
    """
    # Normalize port names for lookup
    load_norm = _norm_port(load_port)
    dis_norm = _norm_port(discharge_port)
    
    # Route mapping with BENCHMARK FREIGHT RATES ($/MT) based on market data
    # Format: (route_name, load_keywords, discharge_keywords, base_freight_rate)
    ROUTE_BENCHMARKS = [
        # C5: West Australia -> China/Asia (~$9/MT based on BHP committed)
        ("C5", ["PORT HEDLAND", "DAMPIER", "AUSTRALIA", "WEST AUSTRALIA", "GERALDTON"], 
         ["QINGDAO", "CHINA", "CAOFEIDIAN", "RIZHAO", "TIANJIN", "LIANYUNGANG", "GWANGYANG", "KOREA"], 9.0),
        # C3: Brazil -> China (~$22/MT based on CSN committed)
        ("C3", ["TUBARAO", "ITAGUAI", "PONTA DA MADEIRA", "BRAZIL", "PDM", "TUBARÃO"], 
         ["QINGDAO", "CHINA", "CAOFEIDIAN", "RIZHAO", "TIANJIN", "TELUK RUBIAH"], 22.0),
        # West Africa (Guinea) -> China (~$23/MT based on EGA committed)
        ("WAFR_CHINA", ["KAMSAR", "GUINEA", "CONAKRY", "WEST AFRICA"], 
         ["QINGDAO", "CHINA", "CAOFEIDIAN", "RIZHAO", "TIANJIN", "MANGALORE", "INDIA"], 23.0),
        # West Africa -> India (~$18/MT)
        ("WAFR_INDIA", ["KAMSAR", "GUINEA", "CONAKRY", "WEST AFRICA"], 
         ["MANGALORE", "INDIA", "PARADIP", "VIZAG", "KRISHNAPATNAM"], 18.0),
        # South Africa -> China (~$14/MT)
        ("SAFR_CHINA", ["SALDANHA", "RICHARDS BAY", "SOUTH AFRICA", "DURBAN"], 
         ["QINGDAO", "CHINA", "CAOFEIDIAN", "RIZHAO", "TIANJIN"], 14.0),
        # Indonesia -> India (~$10/MT)
        ("INDO_INDIA", ["TABONEO", "INDONESIA", "KALIMANTAN", "SAMARINDA"], 
         ["KRISHNAPATNAM", "INDIA", "PARADIP", "VIZAG", "MANGALORE"], 10.0),
        # Canada/Pacific -> China (~$16/MT)
        ("PAC_CHINA", ["VANCOUVER", "CANADA", "PACIFIC"], 
         ["FANGCHENG", "CHINA", "QINGDAO", "CAOFEIDIAN"], 16.0),
        # C7: South America -> Europe (~$12/MT)
        ("C7", ["BOLIVAR", "COLOMBIA", "SOUTH AMERICA"], 
         ["ROTTERDAM", "EUROPE", "AMSTERDAM"], 12.0),
    ]
    
    # Try to match route and get benchmark freight rate
    for route_name, load_keys, dis_keys, base_rate in ROUTE_BENCHMARKS:
        load_match = any(key in load_norm for key in load_keys)
        dis_match = any(key in dis_norm for key in dis_keys)
        if load_match and dis_match:
            # Adjust rate based on FFA market movement (optional scaling)
            # Get current FFA rate vs baseline to adjust
            try:
                if route_name in ["C3", "C5", "C7"]:
                    route_row = ffa[ffa["Route"].str.contains(route_name, case=False, na=False)]
                    if len(route_row) > 0:
                        row = route_row.iloc[0]
                        monthly_cols = [c for c in ffa.columns if isinstance(c, pd.Timestamp)]
                        monthly_cols = sorted(monthly_cols)
                        
                        # Get current rate
                        current_rate = None
                        if dep_date is not None and pd.notna(dep_date) and monthly_cols:
                            dep = pd.Timestamp(dep_date)
                            month_key = pd.Timestamp(dep.year, dep.month, 1)
                            if month_key in monthly_cols and pd.notna(row[month_key]):
                                current_rate = float(row[month_key])
                        
                        if current_rate is None and monthly_cols:
                            for c in monthly_cols:
                                if pd.notna(row[c]):
                                    current_rate = float(row[c])
                                    break
                        
                        # Scale freight based on FFA movement (baseline rates for reference)
                        # C3 baseline: ~19,000/day, C5 baseline: ~7,500/day, C7 baseline: ~11,000/day
                        baseline_tce = {"C3": 19000, "C5": 7500, "C7": 11000}
                        if current_rate and route_name in baseline_tce:
                            scale_factor = current_rate / baseline_tce[route_name]
                            # Apply moderate scaling (cap at +/- 30%)
                            scale_factor = max(0.7, min(1.3, scale_factor))
                            adjusted_rate = base_rate * scale_factor
                            return round(adjusted_rate, 2)
            except Exception:
                pass
            
            return base_rate
    
    # Fallback: estimate based on distance (rough approximation)
    # Long haul (Brazil/Africa -> Asia): ~$20/MT
    # Medium haul (Australia -> Asia): ~$9/MT
    # Short haul (Intra-Asia): ~$6/MT
    
    # Check for long-haul routes
    long_haul_origins = ["BRAZIL", "TUBARAO", "ITAGUAI", "GUINEA", "KAMSAR", "AFRICA", "SALDANHA"]
    long_haul_dests = ["CHINA", "QINGDAO", "CAOFEIDIAN", "RIZHAO", "TIANJIN"]
    
    is_long_origin = any(key in load_norm for key in long_haul_origins)
    is_long_dest = any(key in dis_norm for key in long_haul_dests)
    
    if is_long_origin and is_long_dest:
        return 20.0
    elif is_long_dest:  # Medium haul to China
        return 12.0
    else:  # Default
        return 10.0


# ============================================================
# Load vessels (2 rows per vessel: BALLAST + LADEN)
# ============================================================
def load_vessels_from_excel(
    vessel_xlsx: Path,
    *,
    speed_mode: str = "economical",
    is_market: bool = False,
    ffa: pd.DataFrame | None = None,
    market_hire_multiplier: float = 1.00,
    avg_cargill_hire: float | None = None,  # Average hire rate from Cargill vessels
) -> list[Vessel]:
    df = pd.read_excel(vessel_xlsx, sheet_name=0)

    df["vessel_name"] = df["vessel_name"].astype(str).str.strip()
    df["movement"] = df["movement"].astype(str).str.strip().str.upper()
    df["voyage_position"] = df["voyage_position"].astype(str).str.strip()
    df["estimate_time_of_departure"] = pd.to_datetime(df["estimate_time_of_departure"], errors="coerce")

    sp_col = "economical_speed" if speed_mode.lower() == "economical" else "warranted_speed"
    vlsf_col = "economical_vlsf" if speed_mode.lower() == "economical" else "waranted_vlsf"
    mgo_col = "economical_mgo" if speed_mode.lower() == "economical" else "waranted_mgo"

    vessels: list[Vessel] = []
    for name, g in df.groupby("vessel_name"):
        first = g.iloc[0]
        dep = first["estimate_time_of_departure"]

        dwt_val = float(first["dwt"])
        # Convert DWT from thousands to actual MT (180.803 -> 180,803)
        if dwt_val < 1000:
            dwt = dwt_val * 1000
        else:
            dwt = dwt_val

        gb_df = g[g["movement"] == "BALLAST"]
        gl_df = g[g["movement"] == "LADEN"]
        if len(gb_df) == 0 or len(gl_df) == 0:
            # skip instead of crashing
            continue
        gb = gb_df.iloc[0]
        gl = gl_df.iloc[0]

        # better default than 0.1
        port_mgo = first.get("port_mgo_per_day", np.nan)
        port_mgo = float(port_mgo) if pd.notna(port_mgo) else 2.0

        # daily hire
        if is_market:
            # Use average of Cargill vessel hire rates for market vessels
            if avg_cargill_hire is not None:
                hire = avg_cargill_hire * float(market_hire_multiplier)
            else:
                # Fallback to FFA 5TC if no average provided
                if ffa is None:
                    raise ValueError("is_market=True requires either avg_cargill_hire or ffa=load_ffa(...).")
                hire = ffa_5tc_usd_per_day(ffa, dep if pd.notna(dep) else None) * float(market_hire_multiplier)
        else:
            if "hire_rate" in df.columns and pd.notna(first.get("hire_rate", np.nan)):
                hire = float(first["hire_rate"])
                # Convert hire_rate from thousands to actual USD/day (11.75 -> 11,750)
                if hire < 1000:
                    hire = hire * 1000
            else:
                hire = 0.0

        vessels.append(
            Vessel(
                name=name,
                dwt=dwt,
                grain_capacity_cbm=70000,

                sp_ballast=float(gb[sp_col]),
                sp_laden=float(gl[sp_col]),

                # treat vlsfo as ifo in this model
                ifo_ballast=float(gb[vlsf_col]),
                ifo_laden=float(gl[vlsf_col]),
                mdo_ballast=float(gb[mgo_col]),
                mdo_laden=float(gl[mgo_col]),

                ifo_port_work=5.5,
                ifo_port_idle=5.5,
                mdo_port_work=port_mgo,
                mdo_port_idle=port_mgo,

                daily_hire=float(hire),
                adcoms=0.0375,
                # why adcoms 3.75%? standard
                position=_norm_port(first["voyage_position"]),
                time_of_departure=dep if pd.notna(dep) else None,

                # Remaining fuel onboard (MT)
                ifo_remaining=float(first.get("vslf_remaining", 0) or 0),
                mdo_remaining=float(first.get("mgo_remaining", 0) or 0),
            )
        )

    return vessels


# ============================================================
# Load cargoes
# ============================================================
def load_cargoes_from_excel(cargo_xlsx: Path, tag: str, ffa: pd.DataFrame | None = None) -> list[Cargo]:
    df = pd.read_excel(cargo_xlsx, sheet_name=0)

    df["load_port"] = df["load_port"].map(_norm_port)
    df["discharge_port1"] = df["discharge_port1"].map(_norm_port)

    # Check for alternative discharge ports (discharge_port2, discharge_port3, etc.)
    alt_port_cols = [c for c in df.columns if c.startswith("discharge_port") and c != "discharge_port1"]
    for col in alt_port_cols:
        df[col] = df[col].map(_norm_port)

    # laycan columns (if present)
    if "earliest_date" in df.columns:
        df["earliest_date"] = pd.to_datetime(df["earliest_date"], errors="coerce")
    else:
        df["earliest_date"] = pd.NaT

    if "latest_date" in df.columns:
        df["latest_date"] = pd.to_datetime(df["latest_date"], errors="coerce")
    else:
        df["latest_date"] = pd.NaT

    def _num(x):
        if pd.isna(x):
            return 0.0
        return float(str(x).replace(",", "").strip())

    cargoes: list[Cargo] = []
    for i, r in df.iterrows():
        base_id = f"{tag}_{r.get('customer','CUST')}_{r.get('commodity','CARGO')}_{i}"

        qty = float(r["qty_mt"]) if pd.notna(r.get("qty_mt")) else np.nan
        fr = float(r["freight_rate"]) if pd.notna(r.get("freight_rate")) else np.nan
        
        # If freight rate is missing, estimate from FFA (if ffa provided)
        if pd.isna(fr) and ffa is not None:
            load_p = r.get("load_port", "")
            dis_p = r.get("discharge_port1", "")
            dep_date = r.get("earliest_date") if pd.notna(r.get("earliest_date")) else None
            fr = estimate_freight_from_ffa(ffa, load_p, dis_p, qty, dep_date)

        broker = _pct_to_float(r.get("broker_commission"))
        charterer = _pct_to_float(r.get("charterer_commission"))

        load_rate = _num(r.get("loading_rate", 0))
        dis_rate = _num(r.get("discharge_rate", 0))

        # Handle NaN in turn time values
        load_tt_raw = r.get("loading_turn_time_hr", 0)
        load_tt = float(load_tt_raw) / 24.0 if pd.notna(load_tt_raw) and load_tt_raw else 0.0
        
        dis_tt_raw = r.get("discharge_turn_time_hr", 0)
        dis_tt = float(dis_tt_raw) / 24.0 if pd.notna(dis_tt_raw) and dis_tt_raw else 0.0

        # total_port_cost should be (load + discharge) already (as you said)
        port_cost_raw = r.get("total_port_cost", 0)
        if pd.isna(port_cost_raw):
            port_cost = 0.0
        else:
            port_cost = float(str(port_cost_raw).replace(",", "").replace("USD", "").strip() or 0.0)

        # Collect all discharge ports (primary + alternatives)
        discharge_ports = [r["discharge_port1"]]
        for col in alt_port_cols:
            port = r.get(col, "")
            if port and isinstance(port, str) and port.strip():
                discharge_ports.append(port)

        # Create a Cargo object for each discharge port option
        for port_idx, dis_port in enumerate(discharge_ports):
            if not dis_port:  # skip empty
                continue
            nm = f"{base_id}" if port_idx == 0 else f"{base_id}_ALT{port_idx}"
            cargoes.append(
                Cargo(
                    name=nm,
                    cargo_qty=qty,
                    stow_factor=1.33,

                    freight=fr,
                    address_coms=charterer,
                    broker_coms=broker,

                    load_rate=load_rate,
                    dis_rate=dis_rate,

                    loadport_tt=load_tt,
                    disport_tt=dis_tt,
                    port_idle=0.5,

                    ballast_bonus=0.0,

                    load_port=r["load_port"],
                    discharge_port=dis_port,
                    total_port_cost=port_cost,
                    earliest_date=r["earliest_date"] if pd.notna(r["earliest_date"]) else None,
                    latest_date=r["latest_date"] if pd.notna(r["latest_date"]) else None,

                    base_cargo_id=base_id,
                    alt_port_index=port_idx,
                )
            )

    return cargoes


# ============================================================
# Calculator (profit/loss)
# ============================================================
def calc(v: Vessel, c: Cargo, voy: Voyage, pf: PricesAndFees) -> dict:
    # durations at sea
    dur_ballast_days = voy.ballast_leg_nm / (v.sp_ballast * 24.0)
    dur_laden_days = voy.laden_leg_nm / (v.sp_laden * 24.0)

    # port time (loading/discharging)
    loadport_days = (c.cargo_qty / c.load_rate) + c.loadport_tt + c.port_idle if c.load_rate > 0 else (c.loadport_tt + c.port_idle)
    disport_days = (c.cargo_qty / c.dis_rate) + c.disport_tt if c.dis_rate > 0 else c.disport_tt

    total_duration = (dur_ballast_days + dur_laden_days) + voy.bunker_days + loadport_days + disport_days

    # revenue
    # Use DWT as the vessel's cargo capacity constraint instead of grain_capacity_cbm
    loaded_qty = min(v.dwt, c.cargo_qty)
    revenue = loaded_qty * c.freight * (1 - c.address_coms - c.broker_coms)

    # hire
    hire_gross = v.daily_hire * total_duration + c.ballast_bonus
    hire_net = hire_gross * (1 - v.adcoms)

    # fuel usage
    ifo_sea = dur_ballast_days * v.ifo_ballast + dur_laden_days * v.ifo_laden
    mdo_sea = dur_ballast_days * v.mdo_ballast + dur_laden_days * v.mdo_laden

    ifo_port = loadport_days * v.ifo_port_work + disport_days * v.ifo_port_work + c.port_idle * v.ifo_port_idle
    mdo_port = loadport_days * v.mdo_port_work + disport_days * v.mdo_port_work + c.port_idle * v.mdo_port_idle

    total_ifo = ifo_sea + ifo_port
    total_mdo = mdo_sea + mdo_port

    # bunker expense: remaining fuel at prev month price + additional fuel at current price
    # IFO expense
    if total_ifo <= v.ifo_remaining:
        ifo_expense = total_ifo * pf.ifo_price_prev
    else:
        ifo_expense = v.ifo_remaining * pf.ifo_price_prev + (total_ifo - v.ifo_remaining) * pf.ifo_price
    
    # MDO expense
    if total_mdo <= v.mdo_remaining:
        mdo_expense = total_mdo * pf.mdo_price_prev
    else:
        mdo_expense = v.mdo_remaining * pf.mdo_price_prev + (total_mdo - v.mdo_remaining) * pf.mdo_price
    
    bunker_expense = ifo_expense + mdo_expense

    # misc expense
    bunker_da = pf.bunker_da_per_day * voy.bunker_days
    misc_expense = (
        pf.awrp + pf.cev + pf.ilhoc
        + bunker_da
        + float(c.total_port_cost or 0.0)
    )

    profit_loss = revenue - hire_net - bunker_expense - misc_expense

    # if any component is NaN -> profit is NaN
    if any(pd.isna(x) for x in [revenue, hire_net, bunker_expense, misc_expense]):
        profit_loss = np.nan

    # TCE (Time Charter Equivalent) = (Revenue - Voyage Costs) / Duration
    voyage_costs = bunker_expense + misc_expense
    tce = (revenue - voyage_costs) / total_duration if total_duration > 0 else 0.0

    return {
        "vessel": v.name,
        "cargo": c.name,
        "base_cargo_id": c.base_cargo_id or c.name,
        "from_pos": v.position,
        "load_port": c.load_port,
        "discharge_port": c.discharge_port,
        "ballast_nm": voy.ballast_leg_nm,
        "laden_nm": voy.laden_leg_nm,
        "dur_ballast_days": dur_ballast_days,
        "dur_laden_days": dur_laden_days,
        "total_duration": total_duration,
        "loaded_qty": loaded_qty,
        "revenue": revenue,
        "hire_net": hire_net,
        "bunker_expense": bunker_expense,
        "misc_expense": misc_expense,
        "profit_loss": profit_loss,
        "tce": tce,
    }


# ============================================================
# Build combinations table (with laycan time filter)
# ============================================================
def build_profit_table(
    vessels: list[Vessel],
    cargoes: list[Cargo],
    dist_lut: dict[tuple[str, str], float],
    pf: PricesAndFees,
    bunker_days: float = 1.0,
    enforce_laycan: bool = True,
    bunker_prices_by_location: dict[str, tuple[float, float, float, float]] | None = None,
) -> pd.DataFrame:
    rows = []

    for v in vessels:
        v_dep = v.time_of_departure if isinstance(v.time_of_departure, pd.Timestamp) else None

        for c in cargoes:
            try:
                ballast_nm = dist_nm(dist_lut, v.position, c.load_port)
                laden_nm = dist_nm(dist_lut, c.load_port, c.discharge_port)
            except KeyError:
                continue

            # laycan filter based on ETA at load port
            if enforce_laycan and v_dep is not None:
                dur_ballast_days = ballast_nm / (v.sp_ballast * 24.0)
                eta_load = v_dep + pd.Timedelta(days=float(dur_ballast_days))

                if isinstance(c.earliest_date, pd.Timestamp) and pd.notna(c.earliest_date):
                    if eta_load < c.earliest_date:
                        continue
                if isinstance(c.latest_date, pd.Timestamp) and pd.notna(c.latest_date):
                    if eta_load > c.latest_date:
                        continue

            # Select bunker prices based on vessel position (nearest hub)
            pf_to_use = pf
            bunker_location = "Singapore"  # default
            if bunker_prices_by_location:
                bunker_location = get_bunker_location_for_port(v.position)
                if bunker_location in bunker_prices_by_location:
                    vlsfo, mgo, vlsfo_prev, mgo_prev = bunker_prices_by_location[bunker_location]
                    pf_to_use = PricesAndFees(
                        ifo_price=vlsfo,
                        mdo_price=mgo,
                        ifo_price_prev=vlsfo_prev,
                        mdo_price_prev=mgo_prev,
                        awrp=pf.awrp,
                        cev=pf.cev,
                        ilhoc=pf.ilhoc,
                        bunker_da_per_day=pf.bunker_da_per_day,
                    )

            voy = Voyage(ballast_leg_nm=ballast_nm, laden_leg_nm=laden_nm, bunker_days=bunker_days)
            out = calc(v, c, voy, pf_to_use)

            # Add bunker location info to output
            out["bunker_location"] = bunker_location

            # keep eta in table for debugging (optional)
            if enforce_laycan and v_dep is not None:
                out["eta_load"] = eta_load

            rows.append(out)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("profit_loss", ascending=False).reset_index(drop=True)


# ============================================================
# OPTIMISATION: committed cargo assignment + optional market cargo
# ============================================================
def optimal_committed_assignment(
    df_cv_cc: pd.DataFrame, 
    df_mv_cc: pd.DataFrame,
    cargill_vessels: list = None,
    idle_days: float = 1.0,
    consider_sunk_cost: bool = True
) -> tuple[pd.DataFrame, float]:
    """
    Maximise total P/L for committed cargoes, considering sunk cost for idle Cargill vessels.
    
    Logic:
    - Each BASE cargo assigned exactly once (can pick any discharge port option)
    - Each vessel used at most once
    - If consider_sunk_cost=True: Factor in the cost of leaving Cargill vessels idle
      → Prefer using Cargill vessel even at a loss if loss < idle penalty
    
    Args:
        df_cv_cc: Cargill vessels x Committed cargoes profit table
        df_mv_cc: Market vessels x Committed cargoes profit table
        cargill_vessels: List of Cargill Vessel objects (needed for sunk cost calc)
        idle_days: Number of days to assume for idle penalty calculation
        consider_sunk_cost: If True, include sunk cost in optimization
    """
    a = df_cv_cc.copy()
    a["vessel_type"] = "CARGILL"
    b = df_mv_cc.copy()
    b["vessel_type"] = "MARKET"
    cand = pd.concat([a, b], ignore_index=True)

    if len(cand) == 0:
        raise ValueError("No feasible combos for committed cargoes. Check distances / port matching / laycan filter.")

    # Build Cargill vessel penalty lookup: {vessel_name: idle_cost}
    cargill_idle_cost = {}
    if consider_sunk_cost and cargill_vessels:
        for v in cargill_vessels:
            # Sunk cost = hire rate * days * (1 - commission)
            cargill_idle_cost[v.name] = (v.daily_hire * idle_days) * (1 - v.adcoms)
    
    all_cargill_names = set(cargill_idle_cost.keys())

    # Use base_cargo_id to group alternative ports together
    # Each base cargo should be assigned exactly once (any port option)
    base_cargos = sorted(cand["base_cargo_id"].unique().tolist())
    opts = {c: cand[cand["base_cargo_id"] == c].to_dict("records") for c in base_cargos}
    for c in base_cargos:
        if len(opts[c]) == 0:
            raise ValueError(f"No feasible vessel found for committed cargo: {c}")

    best_choice, best_total = None, -float("inf")

    for choice in itertools.product(*(opts[c] for c in base_cargos)):
        used = set()
        ok = True
        total = 0.0

        for r in choice:
            if r["vessel"] in used:
                ok = False
                break
            used.add(r["vessel"])

            pl = r["profit_loss"]
            if pd.isna(pl):
                ok = False
                break
            total += float(pl)

        if not ok:
            continue
            
        # Add sunk cost penalty for unused Cargill vessels
        if consider_sunk_cost:
            used_cargill = used & all_cargill_names
            unused_cargill = all_cargill_names - used_cargill
            sunk_cost_penalty = sum(cargill_idle_cost.get(v, 0) for v in unused_cargill)
            total_with_sunk = total - sunk_cost_penalty
        else:
            total_with_sunk = total

        if total_with_sunk > best_total:
            best_total = total_with_sunk
            best_choice = choice

    assign_df = pd.DataFrame(best_choice)
    # Include comprehensive columns for detailed output
    cols = ["base_cargo_id", "cargo", "discharge_port", "vessel_type", "vessel", 
            "from_pos", "load_port", "ballast_nm", "laden_nm", 
            "dur_ballast_days", "dur_laden_days", "total_duration",
            "loaded_qty", "revenue", "hire_net", "bunker_expense", "misc_expense",
            "profit_loss", "tce", "bunker_location", "eta_load"]
    cols = [c for c in cols if c in assign_df.columns]
    assign_df = assign_df[cols].sort_values("base_cargo_id").reset_index(drop=True)

    return assign_df, best_total


def assign_market_cargo_to_unused_cargill(
    df_cv_mc: pd.DataFrame, 
    used_cargill: set[str],
    force_assignment: bool = True
) -> tuple[pd.DataFrame, float]:
    """
    For each unused Cargill vessel, assign at most 1 market cargo (greedy):
    - choose best profit market cargo for that vessel
    - do not reuse a market cargo
    - if force_assignment=True: assign even if P/L is negative (vessel must be used)
    - if force_assignment=False: only assign if P/L > 0
    """
    if df_cv_mc is None or len(df_cv_mc) == 0:
        return pd.DataFrame(), 0.0

    all_cargill = set(df_cv_mc["vessel"].unique())
    unused = sorted(all_cargill - set(used_cargill))

    used_market_cargo = set()
    rows = []

    for v in unused:
        sub = df_cv_mc[df_cv_mc["vessel"] == v].sort_values("profit_loss", ascending=False)
        if len(sub) == 0:
            continue

        picked = None
        for _, r in sub.iterrows():
            if r["cargo"] in used_market_cargo:
                continue
            if pd.isna(r["profit_loss"]):
                continue
            # If force_assignment, take best available even if negative P/L
            # Otherwise, only take if positive P/L
            if not force_assignment and float(r["profit_loss"]) <= 0:
                break
            picked = r
            break

        if picked is not None:
            used_market_cargo.add(picked["cargo"])
            rows.append(picked)

    if not rows:
        return pd.DataFrame(columns=df_cv_mc.columns), 0.0

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out, float(out["profit_loss"].sum())


def unused_committed_vessel_penalty(cargill_vessels, used_cargill: set[str], idle_days: float = 1.0) -> float:
    """
    Cost for committed vessels even if not used.
    idle_days = how many days you assume you must pay when unused.
    """
    penalty = 0.0
    for v in cargill_vessels:
        if v.name not in used_cargill:
            penalty += (v.daily_hire * idle_days) * (1 - v.adcoms)
    return penalty


def compare_alt_discharge_ports(df: pd.DataFrame, tolerance_pct: float = 0.02, show_all: bool = False) -> pd.DataFrame:
    """
    Compare alternative discharge ports for the same base cargo.
    Returns only the BEST port option per cargo (or all if show_all=True).
    
    Args:
        df: profit table with base_cargo_id, discharge_port, tce columns
        tolerance_pct: e.g. 0.02 = 2% tolerance
        show_all: if True, show all options with comparison; if False, show only optimal
    
    Returns:
        DataFrame with discharge port comparison
    """
    if df is None or len(df) == 0 or "base_cargo_id" not in df.columns:
        return pd.DataFrame()
    
    # Group by vessel + base_cargo_id to compare port options
    results = []
    
    for (vessel, base_id), grp in df.groupby(["vessel", "base_cargo_id"]):
        if len(grp) <= 1:
            # No alternatives for this cargo - skip
            continue
        
        # Sort by TCE descending
        grp_sorted = grp.sort_values("tce", ascending=False).reset_index(drop=True)
        best_tce = grp_sorted.iloc[0]["tce"]
        
        for idx, row in grp_sorted.iterrows():
            tce = row["tce"]
            tce_diff_pct = (best_tce - tce) / best_tce if best_tce != 0 else 0
            within_tolerance = tce_diff_pct <= tolerance_pct
            
            # Only include best option (or all if show_all=True)
            if not show_all and idx > 0:
                continue
            
            results.append({
                "vessel": vessel,
                "base_cargo_id": base_id,
                "discharge_port": row["discharge_port"],
                "laden_nm": row["laden_nm"],
                "total_duration": round(row["total_duration"], 2),
                "profit_loss": round(row["profit_loss"], 2),
                "tce": round(tce, 2),
                "best_tce": round(best_tce, 2),
                "tce_diff_%": round(tce_diff_pct * 100, 2),
                "within_2%": "YES" if within_tolerance else "NO",
            })
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results).sort_values(["vessel", "base_cargo_id", "tce"], ascending=[True, True, False])


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) lookups
    dist_lut = load_distance_lookup(DIST_XLSX)

    # 2) bunker prices - load ALL locations for dynamic selection
    all_bunker_prices = load_all_bunker_prices(BUNKER_XLSX, month_col="mar", prev_month_col="feb")
    print(f"\nLoaded bunker prices for {len(all_bunker_prices)} locations: {list(all_bunker_prices.keys())}")
    
    # Default prices (Singapore) for fallback
    vlsfo_price, mgo_price, vlsfo_prev, mgo_prev = load_bunker_prices(
        BUNKER_XLSX, location="Singapore", month_col="mar", prev_month_col="feb"
    )
    pf = PricesAndFees(
        ifo_price=vlsfo_price,
        mdo_price=mgo_price,
        ifo_price_prev=vlsfo_prev,
        mdo_price_prev=mgo_prev,
    )

    # 3) FFA
    ffa = load_ffa(FFA_REPORT_XLSX)

    # 4) vessels
    # Load Cargill vessels first
    cargill_vessels = load_vessels_from_excel(
        CARGILL_VESSEL_XLSX,
        speed_mode="economical",
        is_market=False,
        ffa=ffa,
    )
    
    # Calculate average Cargill hire rate for market vessels
    avg_cargill_hire = sum(v.daily_hire for v in cargill_vessels) / len(cargill_vessels) if cargill_vessels else 0
    print(f"Average Cargill vessel hire rate: ${avg_cargill_hire:,.2f}/day")
    
    # Load market vessels using average Cargill hire rate
    market_vessels = load_vessels_from_excel(
        MARKET_VESSEL_XLSX,
        speed_mode="economical",
        is_market=True,
        ffa=ffa,
        market_hire_multiplier=1.00,
        avg_cargill_hire=avg_cargill_hire,  # Use average of Cargill hire rates
    )

    # 5) cargoes
    committed = load_cargoes_from_excel(COMMITTED_CARGO_XLSX, tag="COMMITTED")
    market_cargoes = load_cargoes_from_excel(MARKET_CARGO_XLSX, tag="MARKET", ffa=ffa)

    #print distance mismatches
    #print_distance_mismatches(cargill_vessels, committed, dist_lut, "CARGILL x COMMITTED")
    #print_distance_mismatches(market_vessels, committed, dist_lut, "MARKET x COMMITTED")
    #print_distance_mismatches(cargill_vessels, market_cargoes, dist_lut, "CARGILL x MARKET")

    # print unique missing port pairs only and collect them
    #all_missing = {}
    #all_missing["CARGILL x COMMITTED"] = print_missing_port_pairs(cargill_vessels, committed, dist_lut, "CARGILL x COMMITTED")
    #all_missing["MARKET x COMMITTED"] = print_missing_port_pairs(market_vessels, committed, dist_lut, "MARKET x COMMITTED")
    #all_missing["CARGILL x MARKET"] = print_missing_port_pairs(cargill_vessels, market_cargoes, dist_lut, "CARGILL x MARKET")

    # Save missing pairs to Excel
    #OUTPUT_DIR = BASE_DIR / "outputs"
    #OUTPUT_DIR.mkdir(exist_ok=True)
    #save_missing_pairs_to_excel(all_missing, OUTPUT_DIR / "missing_port_pairs.xlsx")

    # 6) build profit tables (enforce laycan) with location-based bunker prices
    df_cv_cc = build_profit_table(cargill_vessels, committed, dist_lut, pf, bunker_days=1.0, enforce_laycan=True, bunker_prices_by_location=all_bunker_prices)
    df_cv_mc = build_profit_table(cargill_vessels, market_cargoes, dist_lut, pf, bunker_days=1.0, enforce_laycan=True, bunker_prices_by_location=all_bunker_prices)
    df_mv_cc = build_profit_table(market_vessels, committed, dist_lut, pf, bunker_days=1.0, enforce_laycan=True, bunker_prices_by_location=all_bunker_prices)

    # 7) print top rows
    print("\n=== Cargill vessels x Committed cargoes (Top 10) ===")
    print(df_cv_cc.head(10)[["vessel", "cargo", "profit_loss", "total_duration", "ballast_nm", "laden_nm", "eta_load"]]
          if len(df_cv_cc) else "No rows.")

    print("\n=== Cargill vessels x Market cargoes (Top 10) ===")
    print(df_cv_mc.head(10)[["vessel", "cargo", "profit_loss", "total_duration", "ballast_nm", "laden_nm", "eta_load"]]
          if len(df_cv_mc) else "No rows.")

    print("\n=== Market vessels x Committed cargoes (Top 10) ===")
    print(df_mv_cc.head(10)[["vessel", "cargo", "profit_loss", "total_duration", "ballast_nm", "laden_nm", "eta_load"]]
          if len(df_mv_cc) else "No rows.")

    # print NaN profit rows
    print_nan_profit_rows(df_cv_cc, "CARGILL x COMMITTED")
    print_nan_profit_rows(df_cv_mc, "CARGILL x MARKET")
    print_nan_profit_rows(df_mv_cc, "MARKET x COMMITTED")

    # 8) optimal plan (with sunk cost consideration for unused Cargill vessels)
    if len(df_cv_cc) and len(df_mv_cc):
        commit_plan, commit_total = optimal_committed_assignment(
            df_cv_cc, df_mv_cc, 
            cargill_vessels=cargill_vessels,
            idle_days=1.0,
            consider_sunk_cost=True  # Factor in sunk cost for idle Cargill vessels
        )

        print("\n" + "="*80)
        print("OPTIMAL PLAN: COMMITTED CARGOES (must carry)")
        print("(Optimized with sunk cost consideration for unused Cargill vessels)")
        print("="*80)
        
        # Print detailed info for each optimal assignment
        for idx, row in commit_plan.iterrows():
            print(f"\n{'─'*80}")
            print(f"ASSIGNMENT {idx+1}: {row['base_cargo_id']}")
            print(f"{'─'*80}")
            
            # Vessel Info
            print(f"\n  📦 VESSEL INFO:")
            print(f"     Type:              {row['vessel_type']}")
            print(f"     Name:              {row['vessel']}")
            if 'from_pos' in row:
                print(f"     Start Position:    {row['from_pos']}")
            
            # Route Info
            print(f"\n  🚢 ROUTE INFO:")
            if 'load_port' in row:
                print(f"     Load Port:         {row['load_port']}")
            print(f"     Discharge Port:    {row['discharge_port']}")
            print(f"     Ballast Distance:  {row['ballast_nm']:,.2f} nm")
            print(f"     Laden Distance:    {row['laden_nm']:,.2f} nm")
            print(f"     Total Distance:    {row['ballast_nm'] + row['laden_nm']:,.2f} nm")
            
            # Time Info
            print(f"\n  ⏱️  TIME INFO:")
            if 'dur_ballast_days' in row:
                print(f"     Ballast Duration:  {row['dur_ballast_days']:.2f} days")
            if 'dur_laden_days' in row:
                print(f"     Laden Duration:    {row['dur_laden_days']:.2f} days")
            print(f"     Total Duration:    {row['total_duration']:.2f} days")
            print(f"     ETA at Load Port:  {row['eta_load']}")
            
            # Financial Info
            print(f"\n  💰 FINANCIAL INFO:")
            if 'loaded_qty' in row:
                print(f"     Loaded Quantity:   {row['loaded_qty']:,.0f} MT")
            if 'revenue' in row:
                print(f"     Revenue:           ${row['revenue']:,.2f}")
            if 'hire_net' in row:
                print(f"     Hire Cost (Net):   ${row['hire_net']:,.2f}")
            if 'bunker_expense' in row:
                print(f"     Bunker Expense:    ${row['bunker_expense']:,.2f}")
            if 'misc_expense' in row:
                print(f"     Misc Expense:      ${row['misc_expense']:,.2f}")
            print(f"     ─────────────────────────────────")
            print(f"     PROFIT/LOSS:       ${row['profit_loss']:,.2f}")
            print(f"     TCE:               ${row['tce']:,.2f}/day")
            if 'bunker_location' in row:
                print(f"     Bunker Location:   {row['bunker_location']}")
        
        print(f"\n{'='*80}")
        print(f"COMMITTED CARGO TOTAL P/L: ${commit_total:,.2f}")
        print(f"{'='*80}")

        used_cargill = set(commit_plan.loc[commit_plan["vessel_type"] == "CARGILL", "vessel"].tolist())
        all_cargill = set([v.name for v in cargill_vessels])
        unused_cargill = sorted(all_cargill - used_cargill)

        print("\nCargill vessels USED for committed cargo:", sorted(used_cargill))
        print("Cargill vessels UNUSED:", unused_cargill)

        # FORCE unused Cargill vessels to carry market cargo (they must be used)
        market_plan, market_total = assign_market_cargo_to_unused_cargill(
            df_cv_mc, used_cargill, force_assignment=True
        )

        print("\n" + "="*80)
        print("MARKET CARGO ON UNUSED CARGILL VESSELS (MUST USE)")
        print("="*80)
        if len(market_plan):
            for idx, row in market_plan.iterrows():
                print(f"\n{'─'*80}")
                print(f"MARKET CARGO ASSIGNMENT: {row['cargo']}")
                print(f"{'─'*80}")
                print(f"  Cargill Vessel:       {row['vessel']}")
                if 'from_pos' in row:
                    print(f"  Start Position:       {row['from_pos']}")
                if 'load_port' in row:
                    print(f"  Load Port:            {row['load_port']}")
                if 'discharge_port' in row:
                    print(f"  Discharge Port:       {row['discharge_port']}")
                print(f"  Ballast Distance:     {row['ballast_nm']:,.2f} nm")
                print(f"  Laden Distance:       {row['laden_nm']:,.2f} nm")
                print(f"  Total Duration:       {row['total_duration']:.2f} days")
                print(f"  ETA at Load Port:     {row['eta_load']}")
                if 'revenue' in row:
                    print(f"  Revenue:              ${row['revenue']:,.2f}")
                if 'bunker_expense' in row:
                    print(f"  Bunker Expense:       ${row['bunker_expense']:,.2f}")
                print(f"  PROFIT/LOSS:          ${row['profit_loss']:,.2f}")
                if 'tce' in row:
                    print(f"  TCE:                  ${row['tce']:,.2f}/day")
        else:
            print("\nNo market cargo available for unused Cargill vessels.")

        print(f"\nMarket cargo total P/L: ${market_total:,.2f}")

        portfolio_total = commit_total + market_total
        print("\n" + "="*80)
        print(f"PORTFOLIO TOTAL P/L: ${portfolio_total:,.2f}")
        print("="*80)

        # Calculate penalty for unused committed vessels
        penalty = unused_committed_vessel_penalty(cargill_vessels, used_cargill, idle_days=1.0)
        portfolio_total_after_penalty = commit_total + market_total - penalty
        print(f"\nUnused committed vessel penalty: ${penalty:,.2f}")
        print(f"PORTFOLIO TOTAL P/L (after penalty): ${portfolio_total_after_penalty:,.2f}")
    else:
        print("\nCannot compute optimal plan (one of the committed tables is empty). Check distances / port matching / laycan filter.")

    # 9) Compare alternative discharge ports (TCE with 2% tolerance)
    print("\n==============================")
    print("\n==============================")
    print("ALTERNATIVE DISCHARGE PORT COMPARISON (2% TCE tolerance)")
    print("==============================")
    alt_port_comparison = compare_alt_discharge_ports(df_cv_cc, tolerance_pct=0.02, show_all=True)
    if len(alt_port_comparison) > 0:
        print(alt_port_comparison.to_string(index=False))
    else:
        print("No alternative discharge ports to compare. ")

    # 10) save output
    OUT_DIR = BASE_DIR / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    df_cv_cc.to_csv(OUT_DIR / "cargill_vs_committed.csv", index=False)
    df_cv_mc.to_csv(OUT_DIR / "cargill_vs_marketcargo.csv", index=False)
    df_mv_cc.to_csv(OUT_DIR / "marketvs_committed.csv", index=False)
    if len(alt_port_comparison) > 0:
        alt_port_comparison.to_csv(OUT_DIR / "alt_port_comparison.csv", index=False)

    # Save optimal assignments to a single comprehensive file
    if commit_plan is not None and len(commit_plan) > 0:
        # Add assignment type column
        commit_plan_out = commit_plan.copy()
        commit_plan_out['assignment_type'] = 'Committed Cargo'
        commit_plan_out['assignment_category'] = commit_plan_out['vessel_type'].apply(
            lambda x: 'Cargill Vessel → Committed Cargo' if x == 'CARGILL' else 'Market Vessel → Committed Cargo'
        )
        
        # Combine with market cargo assignments if any
        if market_plan is not None and len(market_plan) > 0:
            market_plan_out = market_plan.copy()
            market_plan_out['assignment_type'] = 'Market Cargo'
            market_plan_out['assignment_category'] = 'Cargill Vessel → Market Cargo'
            all_assignments = pd.concat([commit_plan_out, market_plan_out], ignore_index=True)
        else:
            all_assignments = commit_plan_out
        
        # Reorder columns for better readability
        priority_cols = ['assignment_category', 'assignment_type', 'vessel', 'vessel_type', 'cargo', 
                        'from_pos', 'load_port', 'discharge_port', 
                        'ballast_nm', 'laden_nm', 'total_duration',
                        'eta_load', 'loaded_qty', 'revenue', 'hire_net', 
                        'bunker_expense', 'misc_expense', 'profit_loss', 'tce', 'bunker_location']
        existing_cols = [c for c in priority_cols if c in all_assignments.columns]
        other_cols = [c for c in all_assignments.columns if c not in existing_cols]
        all_assignments = all_assignments[existing_cols + other_cols]
        
        all_assignments.to_csv(OUT_DIR / "optimal_assignments.csv", index=False)
        print(f"\n✅ Saved OPTIMAL ASSIGNMENTS to: {OUT_DIR / 'optimal_assignments.csv'}")

    print(f"\nSaved CSV outputs to: {OUT_DIR}")

    