# coba.py
# ============================================================
# FULL VERSION (includes everything in one file)
# ✅ Fixes:
#   1) Cargill hire_rate conversion (11.75 -> 11,750/day) when < 1000
#   2) Better default for port_mgo_per_day (use 2.0 instead of 0.1)
#   3) Safe handling if a vessel is missing BALLAST/LADEN row (skip instead of crash)
#   4) ✅ Better port normalisation (accents + aliases) to reduce distance mismatches
#   5) ✅ Time handling: laycan filter using vessel estimate_time_of_departure + ETA(load)
# ✅ Adds:
#   6) Optimal assignment for ALL committed cargoes (3 cargos) using Cargill + Market vessels
#   7) Optional: assign MARKET cargo to UNUSED Cargill vessels (1 cargo per vessel, no cargo reused)
#   8) ✅ Print mismatches (distance lookup fail) clearly (BAL / LAD)
#   9) ✅ Print NaN profit rows for debugging
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
    "KAMSAR ANCHORAGE": "KAMSAR",
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


@dataclass
class Voyage:
    ballast_leg_nm: float = 3000
    laden_leg_nm: float = 3000
    bunker_days: float = 1.0


@dataclass
class PricesAndFees:
    ifo_price: float = 440
    mdo_price: float = 850

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
def load_bunker_prices(bunker_xlsx: Path, location="Singapore", month_col="cal") -> tuple[float, float]:
    df = pd.read_excel(bunker_xlsx, sheet_name=0)
    df["location"] = df["location"].astype(str).str.strip().str.upper()
    df["type"] = df["type"].astype(str).str.strip().str.upper()

    loc = location.strip().upper()
    vlsfo = df.loc[(df["location"] == loc) & (df["type"].isin(["VLSFO", "LSFO", "IFO"])), month_col].iloc[0]
    mgo = df.loc[(df["location"] == loc) & (df["type"].isin(["MGO", "MDO"])), month_col].iloc[0]
    return float(vlsfo), float(mgo)


# ============================================================
# FFA 5TC hire
# ============================================================
def load_ffa(ffa_xlsx: Path) -> pd.DataFrame:
    ffa = pd.read_excel(ffa_xlsx, sheet_name=0)
    ffa["Route"] = ffa["Route"].astype(str).str.strip()
    return ffa


def _find_5tc_row(ffa: pd.DataFrame) -> pd.Series:
    m = ffa["Route"].str.contains("5TC", case=False, na=False)
    if not m.any():
        raise KeyError("Cannot find a 5TC row in ffa_report.xlsx (column 'Route').")
    return ffa.loc[m].iloc[0]


def ffa_5tc_usd_per_day(ffa: pd.DataFrame, dep: pd.Timestamp | None) -> float:
    row = _find_5tc_row(ffa)

    monthly_cols = [c for c in ffa.columns if isinstance(c, pd.Timestamp)]
    monthly_cols = sorted(monthly_cols)

    if dep is not None and pd.notna(dep) and monthly_cols:
        dep = pd.Timestamp(dep)
        month_key = pd.Timestamp(dep.year, dep.month, 1)

        if month_key in monthly_cols and pd.notna(row[month_key]):
            return float(row[month_key])

        earlier = [c for c in monthly_cols if c <= dep and pd.notna(row[c])]
        later = [c for c in monthly_cols if c >= dep and pd.notna(row[c])]
        if earlier and later:
            a = max(earlier)
            b = min(later)
            if a == b:
                return float(row[a])
            ra, rb = float(row[a]), float(row[b])
            w = (dep - a) / (b - a)
            return ra + float(w) * (rb - ra)

    if dep is not None and pd.notna(dep):
        q = (int(dep.month) - 1) // 3 + 1
        qcol = f"Q{q} {str(dep.year)[-2:]}"
        if qcol in ffa.columns and pd.notna(row[qcol]):
            return float(row[qcol])

        calcol = f"Cal {str(dep.year)[-2:]}"
        if calcol in ffa.columns and pd.notna(row[calcol]):
            return float(row[calcol])

    cal_cols = [c for c in ffa.columns if isinstance(c, str) and c.strip().lower().startswith("cal")]
    for c in cal_cols:
        if pd.notna(row[c]):
            return float(row[c])

    raise ValueError("Could not derive 5TC hire from FFA table.")


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
            if ffa is None:
                raise ValueError("is_market=True requires ffa=load_ffa(...).")
            hire = ffa_5tc_usd_per_day(ffa, dep if pd.notna(dep) else None) * float(market_hire_multiplier)
        else:
            if "hire_rate" in df.columns and pd.notna(first.get("hire_rate", np.nan)):
                hire = float(first["hire_rate"])
               
                
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

                position=_norm_port(first["voyage_position"]),
                time_of_departure=dep if pd.notna(dep) else None,
            )
        )

    return vessels


# ============================================================
# Load cargoes
# ============================================================
def load_cargoes_from_excel(cargo_xlsx: Path, tag: str) -> list[Cargo]:
    df = pd.read_excel(cargo_xlsx, sheet_name=0)

    df["load_port"] = df["load_port"].map(_norm_port)
    df["discharge_port1"] = df["discharge_port1"].map(_norm_port)

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
        nm = f"{tag}_{r.get('customer','CUST')}_{r.get('commodity','CARGO')}_{i}"

        qty = float(r["qty_mt"]) if pd.notna(r.get("qty_mt")) else np.nan
        fr = float(r["freight_rate"]) if pd.notna(r.get("freight_rate")) else np.nan

        broker = _pct_to_float(r.get("broker_commission"))
        charterer = _pct_to_float(r.get("charterer_commission"))

        load_rate = _num(r.get("loading_rate", 0))
        dis_rate = _num(r.get("discharge_rate", 0))

        load_tt = float(r.get("loading_turn_time_hr", 0) or 0) / 24.0
        dis_tt = float(r.get("discharge_turn_time_hr", 0) or 0) / 24.0

        # total_port_cost should be (load + discharge) already (as you said)
        port_cost_raw = r.get("total_port_cost", 0)
        if pd.isna(port_cost_raw):
            port_cost = 0.0
        else:
            port_cost = float(str(port_cost_raw).replace(",", "").replace("USD", "").strip() or 0.0)

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
                discharge_port=r["discharge_port1"],
                total_port_cost=port_cost,
                earliest_date=r["earliest_date"] if pd.notna(r["earliest_date"]) else None,
                latest_date=r["latest_date"] if pd.notna(r["latest_date"]) else None,
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
    loaded_qty = min(v.grain_capacity_cbm / c.stow_factor, c.cargo_qty)
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

    # bunker expense
    bunker_expense = total_ifo * pf.ifo_price + total_mdo * pf.mdo_price

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

    return {
        "vessel": v.name,
        "cargo": c.name,
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
    }


# ============================================================
# Build combinations table (✅ with laycan time filter)
# ============================================================
def build_profit_table(
    vessels: list[Vessel],
    cargoes: list[Cargo],
    dist_lut: dict[tuple[str, str], float],
    pf: PricesAndFees,
    bunker_days: float = 1.0,
    enforce_laycan: bool = True,
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

            # ✅ laycan filter based on ETA at load port
            if enforce_laycan and v_dep is not None:
                dur_ballast_days = ballast_nm / (v.sp_ballast * 24.0)
                eta_load = v_dep + pd.Timedelta(days=float(dur_ballast_days))

                if isinstance(c.earliest_date, pd.Timestamp) and pd.notna(c.earliest_date):
                    if eta_load < c.earliest_date:
                        continue
                if isinstance(c.latest_date, pd.Timestamp) and pd.notna(c.latest_date):
                    if eta_load > c.latest_date:
                        continue

            voy = Voyage(ballast_leg_nm=ballast_nm, laden_leg_nm=laden_nm, bunker_days=bunker_days)
            out = calc(v, c, voy, pf)

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
def optimal_committed_assignment(df_cv_cc: pd.DataFrame, df_mv_cc: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Maximise total P/L for committed cargoes:
    - each committed cargo assigned exactly once
    - each vessel used at most once
    """
    a = df_cv_cc.copy()
    a["vessel_type"] = "CARGILL"
    b = df_mv_cc.copy()
    b["vessel_type"] = "MARKET"
    cand = pd.concat([a, b], ignore_index=True)

    if len(cand) == 0:
        raise ValueError("No feasible combos for committed cargoes. Check distances / port matching / laycan filter.")

    cargos = sorted(cand["cargo"].unique().tolist())
    opts = {c: cand[cand["cargo"] == c].to_dict("records") for c in cargos}
    for c in cargos:
        if len(opts[c]) == 0:
            raise ValueError(f"No feasible vessel found for committed cargo: {c}")

    best_choice, best_total = None, -float("inf")

    for choice in itertools.product(*(opts[c] for c in cargos)):
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

        if ok and total > best_total:
            best_total = total
            best_choice = choice

    assign_df = pd.DataFrame(best_choice)
    cols = ["cargo", "vessel_type", "vessel", "profit_loss", "total_duration", "ballast_nm", "laden_nm", "eta_load"]
    cols = [c for c in cols if c in assign_df.columns]
    assign_df = assign_df[cols].sort_values("cargo").reset_index(drop=True)

    return assign_df, best_total


def assign_market_cargo_to_unused_cargill(df_cv_mc: pd.DataFrame, used_cargill: set[str]) -> tuple[pd.DataFrame, float]:
    """
    For each unused Cargill vessel, assign at most 1 market cargo (greedy):
    - choose best profit market cargo for that vessel
    - do not reuse a market cargo
    - only take if profit_loss > 0
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
            if pd.isna(r["profit_loss"]) or float(r["profit_loss"]) <= 0:
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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) lookups
    dist_lut = load_distance_lookup(DIST_XLSX)

    # 2) bunker prices (Singapore / "cal")
    vlsfo_price, mgo_price = load_bunker_prices(BUNKER_XLSX, location="Singapore", month_col="cal")
    pf = PricesAndFees(ifo_price=vlsfo_price, mdo_price=mgo_price)

    # 3) FFA
    ffa = load_ffa(FFA_REPORT_XLSX)

    # 4) vessels
    cargill_vessels = load_vessels_from_excel(
        CARGILL_VESSEL_XLSX,
        speed_mode="economical",
        is_market=False,
        ffa=ffa,
    )
    market_vessels = load_vessels_from_excel(
        MARKET_VESSEL_XLSX,
        speed_mode="economical",
        is_market=True,
        ffa=ffa,
        market_hire_multiplier=1.00,
    )

    # 5) cargoes
    committed = load_cargoes_from_excel(COMMITTED_CARGO_XLSX, tag="COMMITTED")
    market_cargoes = load_cargoes_from_excel(MARKET_CARGO_XLSX, tag="MARKET")

    # ✅ print distance mismatches
    print_distance_mismatches(cargill_vessels, committed, dist_lut, "CARGILL x COMMITTED")
    print_distance_mismatches(market_vessels, committed, dist_lut, "MARKET x COMMITTED")
    print_distance_mismatches(cargill_vessels, market_cargoes, dist_lut, "CARGILL x MARKET")

    # 6) build profit tables (✅ enforce laycan)
    df_cv_cc = build_profit_table(cargill_vessels, committed, dist_lut, pf, bunker_days=1.0, enforce_laycan=True)
    df_cv_mc = build_profit_table(cargill_vessels, market_cargoes, dist_lut, pf, bunker_days=1.0, enforce_laycan=True)
    df_mv_cc = build_profit_table(market_vessels, committed, dist_lut, pf, bunker_days=1.0, enforce_laycan=True)

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

    # ✅ print NaN profit rows
    print_nan_profit_rows(df_cv_cc, "CARGILL x COMMITTED")
    print_nan_profit_rows(df_cv_mc, "CARGILL x MARKET")
    print_nan_profit_rows(df_mv_cc, "MARKET x COMMITTED")

    # 8) optimal plan
    if len(df_cv_cc) and len(df_mv_cc):
        commit_plan, commit_total = optimal_committed_assignment(df_cv_cc, df_mv_cc)

        print("\n==============================")
        print("OPTIMAL PLAN: COMMITTED CARGOES (must carry)")
        print("==============================")
        print(commit_plan)
        print(f"\nCommitted cargo total P/L: {commit_total:,.2f}")

        used_cargill = set(commit_plan.loc[commit_plan["vessel_type"] == "CARGILL", "vessel"].tolist())
        all_cargill = set([v.name for v in cargill_vessels])
        unused_cargill = sorted(all_cargill - used_cargill)

        print("\nCargill vessels USED for committed cargo:", sorted(used_cargill))
        print("Cargill vessels UNUSED:", unused_cargill)

        # optional: use unused cargill for market cargo
        market_plan, market_total = assign_market_cargo_to_unused_cargill(df_cv_mc, used_cargill)

        print("\n==============================")
        print("OPTIONAL UPSIDE: MARKET CARGO ON UNUSED CARGILL VESSELS")
        print("==============================")
        if len(market_plan):
            print(market_plan[["vessel", "cargo", "profit_loss", "total_duration", "ballast_nm", "laden_nm", "eta_load"]])
        else:
            print("No profitable market cargo assigned.")

        print(f"\nMarket cargo total P/L: {market_total:,.2f}")

        portfolio_total = commit_total + market_total
        print("\n==============================")
        print(f"PORTFOLIO TOTAL P/L: {portfolio_total:,.2f}")
        print("==============================")
    else:
        print("\nCannot compute optimal plan (one of the committed tables is empty). Check distances / port matching / laycan filter.")

    # 9) save outputs
    OUT_DIR = BASE_DIR / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    df_cv_cc.to_csv(OUT_DIR / "cargill_vs_committed.csv", index=False)
    df_cv_mc.to_csv(OUT_DIR / "cargill_vs_marketcargo.csv", index=False)
    df_mv_cc.to_csv(OUT_DIR / "marketvs_committed.csv", index=False)

    print(f"\nSaved CSV outputs to: {OUT_DIR}")

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

# after computing used_cargill and commit_total/market_total:
penalty = unused_committed_vessel_penalty(cargill_vessels, used_cargill, idle_days=1.0)
portfolio_total = commit_total + market_total - penalty

print(f"\nUnused committed vessel penalty: {penalty:,.2f}")
print(f"PORTFOLIO TOTAL P/L (after penalty): {portfolio_total:,.2f}")
