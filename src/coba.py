from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import math

# =========================
# Paths (your uploaded files)
# =========================
BUNKER_XLSX = Path("data/bunker.xlsx")
CARGILL_VESSEL_XLSX = Path("data/cargill_vessel.xlsx")
COMMITTED_CARGO_XLSX = Path("data/committed_cargoes.xlsx")
MARKET_VESSEL_XLSX = Path("data/market_vessel.xlsx")
MARKET_CARGO_XLSX = Path("data/market_cargoes.xlsx")
DIST_XLSX = Path("data/Port Distances.xlsx")


# =========================
# Your dataclasses (same idea, but we’ll fill from Excel)
# =========================
@dataclass
class Vessel:
    name: str
    dwt: float

    # Sea speeds (knots)
    sp_ballast: float
    sp_laden: float

    # Sea fuel consumption (tons/day) — we treat VLSFO as "IFO"
    ifo_ballast: float
    ifo_laden: float
    mdo_ballast: float
    mdo_laden: float

    # Port fuel consumption (tons/day)
    # NOTE: your vessel file has port_mgo_per_day; it doesn’t have port VLSFO.
    # We'll keep a constant VLSFO port burn (editable) + read MGO from file.
    ifo_port_work: float = 5.5
    mdo_port_work: float = 2.0
    ifo_port_idle: float = 5.5
    mdo_port_idle: float = 2.0

    # Hire (USD/day)
    daily_hire: float = 0.0
    adcoms: float = 0.0375  # default if you need it

    # Current position / timing
    position: str = ""
    time_of_departure: pd.Timestamp | None = None


@dataclass
class Cargo:
    name: str
    cargo_qty: float

    freight: float
    address_coms: float
    broker_coms: float

    load_rate: float
    dis_rate: float

    loadport_tt: float
    disport_tt: float
    port_idle: float

    load_port: str
    discharge_port: str

    # Optional costs
    total_port_cost: float = 0.0

    # Laycan (optional)
    earliest_date: pd.Timestamp | None = None
    latest_date: pd.Timestamp | None = None

    ballast_bonus: float = 0.0


@dataclass
class Voyage:
    ballast_leg_nm: float
    laden_leg_nm: float
    bunker_days: float = 1.0  # keep as your Excel reference, editable


@dataclass
class PricesAndFees:
    ifo_price: float
    mdo_price: float

    awrp: float = 1500
    cev: float = 1500
    ilhoc: float = 5000

    pda_loadport: float = 0
    pda_disport: float = 0

    bunker_da_per_day: float = 1500


# =========================
# Helpers: cleaning + distances + prices
# =========================
def _norm_port(p: str) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return str(p).strip().upper()


def load_distance_lookup(dist_xlsx: Path) -> dict[tuple[str, str], float]:
    df = pd.read_excel(dist_xlsx, sheet_name=0)
    df["PORT_NAME_FROM"] = df["PORT_NAME_FROM"].map(_norm_port)
    df["PORT_NAME_TO"] = df["PORT_NAME_TO"].map(_norm_port)
    # keep only rows with a number
    df = df[pd.to_numeric(df["DISTANCE"], errors="coerce").notna()].copy()
    df["DISTANCE"] = df["DISTANCE"].astype(float)

    # Make lookup BOTH directions (A->B and B->A) in case table is one-way.
    lut = {}
    for r in df.itertuples(index=False):
        a, b, d = r.PORT_NAME_FROM, r.PORT_NAME_TO, r.DISTANCE
        lut[(a, b)] = d
        lut[(b, a)] = d
    return lut


def dist_nm(lut: dict, a: str, b: str) -> float:
    a, b = _norm_port(a), _norm_port(b)
    if (a, b) in lut:
        return float(lut[(a, b)])
    # If not found, you can decide:
    # - return a big number (penalty)
    # - or raise error
    raise KeyError(f"Distance not found for {a} -> {b}")


def load_bunker_prices(bunker_xlsx: Path, location="Singapore", month_col="cal") -> tuple[float, float]:
    """
    bunker.xlsx has rows like:
      location, type(VLSFO/MGO), feb..dec, cal
    We use Singapore + cal by default.
    """
    df = pd.read_excel(bunker_xlsx, sheet_name=0)
    df["location"] = df["location"].astype(str).str.strip().str.upper()
    df["type"] = df["type"].astype(str).str.strip().str.upper()

    loc = location.strip().upper()
    vlsfo = df.loc[(df["location"] == loc) & (df["type"].isin(["VLSFO", "LSFO", "IFO"])), month_col].iloc[0]
    mgo = df.loc[(df["location"] == loc) & (df["type"].isin(["MGO", "MDO"])), month_col].iloc[0]
    return float(vlsfo), float(mgo)


def _pct_to_float(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    s = str(x).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)


# =========================
# Load vessels from Excel
# =========================
def load_vessels(vessel_xlsx: Path, speed_mode="economical") -> list[Vessel]:
    """
    vessel sheet has 2 rows per vessel: movement = Laden/Ballast
    We'll pivot into one Vessel object with both ballast & laden speeds/cons.
    speed_mode: "economical" or "warranted"
    """
    df = pd.read_excel(vessel_xlsx, sheet_name=0)

    # normalize
    df["vessel_name"] = df["vessel_name"].astype(str).str.strip()
    df["movement"] = df["movement"].astype(str).str.strip().str.upper()
    df["voyage_position"] = df["voyage_position"].astype(str).str.strip()
    df["estimate_time_of_departure"] = pd.to_datetime(df["estimate_time_of_departure"], errors="coerce")

    sp_col = "economical_speed" if speed_mode.lower() == "economical" else "warranted_speed"
    vlsf_col = "economical_vlsf" if speed_mode.lower() == "economical" else "waranted_vlsf"
    mgo_col = "economical_mgo" if speed_mode.lower() == "economical" else "waranted_mgo"

    vessels = []
    for name, g in df.groupby("vessel_name"):
        # pick common fields from first row
        first = g.iloc[0]
        dwt = float(first["dwt"]) * 1000 if float(first["dwt"]) < 1000 else float(first["dwt"])

        # ballast row
        gb = g[g["movement"] == "BALLAST"].iloc[0]
        # laden row
        gl = g[g["movement"] == "LADEN"].iloc[0]

        print("Vessel columns:", df.columns.tolist())

        # hire_rate looks like 11.75 meaning $11,750/day → *1000
        hire_rate = float(first["hire_rate"]) * 1000.0

        port_mgo = float(first["port_mgo_per_day"]) if pd.notna(first["port_mgo_per_day"]) else 2.0

        vessels.append(
            Vessel(
                name=name,
                dwt=dwt,
                sp_ballast=float(gb[sp_col]),
                sp_laden=float(gl[sp_col]),
                ifo_ballast=float(gb[vlsf_col]),
                ifo_laden=float(gl[vlsf_col]),
                mdo_ballast=float(gb[mgo_col]),
                mdo_laden=float(gl[mgo_col]),
                mdo_port_work=port_mgo,
                mdo_port_idle=port_mgo,
                daily_hire=hire_rate,
                position=first["voyage_position"],
                time_of_departure=first["estimate_time_of_departure"],
            )
        )
    return vessels


# =========================
# Load cargoes (committed + market use same columns)
# =========================
def load_cargoes(cargo_xlsx: Path) -> list[Cargo]:
    df = pd.read_excel(cargo_xlsx, sheet_name=0)

    # normalize ports
    df["load_port"] = df["load_port"].map(_norm_port)
    df["discharge_port1"] = df["discharge_port1"].map(_norm_port)

    df["earliest_date"] = pd.to_datetime(df.get("earliest_date"), errors="coerce")
    df["latest_date"] = pd.to_datetime(df.get("latest_date"), errors="coerce")

    cargoes = []
    for i, r in df.iterrows():
        name = f"{r.get('customer','CUST')}_{r.get('commodity','CARGO')}_{i}"
        qty = float(r["qty_mt"])

        freight = float(r["freight_rate"])  # assume $/MT
        broker = _pct_to_float(r.get("broker_commission"))
        charterer = _pct_to_float(r.get("charterer_commission"))

        load_rate = float(str(r["loading_rate"]).replace(",", "")) if pd.notna(r["loading_rate"]) else 0.0
        dis_rate = float(str(r["discharge_rate"]).replace(",", "")) if pd.notna(r["discharge_rate"]) else 0.0

        load_tt_days = float(r.get("loading_turn_time_hr", 0)) / 24.0
        dis_tt_days = float(r.get("discharge_turn_time_hr", 0)) / 24.0

        total_port_cost = float(r.get("total_port_cost", 0) or 0)

        cargoes.append(
            Cargo(
                name=name,
                cargo_qty=qty,
                freight=freight,
                address_coms=charterer,   # treat as address/charterer commission
                broker_coms=broker,
                load_rate=load_rate,
                dis_rate=dis_rate,
                loadport_tt=load_tt_days,
                disport_tt=dis_tt_days,
                port_idle=0.5,  # assumption if not given separately
                load_port=r["load_port"],
                discharge_port=r["discharge_port1"],
                total_port_cost=total_port_cost,
                earliest_date=r.get("earliest_date"),
                latest_date=r.get("latest_date"),
            )
        )
    return cargoes


# =========================
# Your core calc (same as yours, but now voyage distances come from Excel)
# =========================
def calc(v: Vessel, c: Cargo, voy: Voyage, pf: PricesAndFees) -> dict:
    # Distances -> durations
    dur_ballast_days = voy.ballast_leg_nm / (v.sp_ballast * 24.0)
    dur_laden_days = voy.laden_leg_nm / (v.sp_laden * 24.0)
    steaming_days = dur_ballast_days + dur_laden_days

    # Port time
    loadport_days = (c.cargo_qty / c.load_rate) + c.loadport_tt + c.port_idle if c.load_rate > 0 else (c.loadport_tt + c.port_idle)
    disport_days = (c.cargo_qty / c.dis_rate) + c.disport_tt if c.dis_rate > 0 else c.disport_tt

    total_duration = steaming_days + voy.bunker_days + loadport_days + disport_days

    # Revenue (use qty as loaded qty for bulk; your earlier cbm/stow factor is grain-specific)
    loaded_qty = c.cargo_qty
    revenue = loaded_qty * c.freight * (1 - c.address_coms - c.broker_coms)

    # Hire (treat daily_hire as opportunity cost for Cargill vessel too; you can adjust later)
    hire_gross = v.daily_hire * total_duration + c.ballast_bonus
    hire_net = hire_gross * (1 - v.adcoms)

    # Fuel at sea
    ifo_sea = dur_ballast_days * v.ifo_ballast + dur_laden_days * v.ifo_laden
    mdo_sea = dur_ballast_days * v.mdo_ballast + dur_laden_days * v.mdo_laden

    # Fuel in port (working + idle)
    ifo_port = (loadport_days + disport_days) * v.ifo_port_work + c.port_idle * v.ifo_port_idle
    mdo_port = (loadport_days + disport_days) * v.mdo_port_work + c.port_idle * v.mdo_port_idle

    total_ifo = ifo_sea + ifo_port
    total_mdo = mdo_sea + mdo_port
    bunker_expense = total_ifo * pf.ifo_price + total_mdo * pf.mdo_price

    # Misc expense: include port costs from cargo sheet
    bunker_da = pf.bunker_da_per_day * voy.bunker_days
    misc_expense = pf.awrp + pf.cev + pf.ilhoc + bunker_da + c.total_port_cost

    profit_loss = revenue - hire_net - bunker_expense - misc_expense

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
        "revenue": revenue,
        "hire_net": hire_net,
        "bunker_expense": bunker_expense,
        "port_cost": c.total_port_cost,
        "misc_expense": misc_expense,
        "profit_loss": profit_loss,
    }


# =========================
# Build voyages for all vessel-cargo pairs
# =========================
def build_profit_matrix(vessels: list[Vessel], cargoes: list[Cargo], dist_lut: dict, pf: PricesAndFees) -> pd.DataFrame:
    rows = []
    for v in vessels:
        for c in cargoes:
            try:
                ballast_nm = dist_nm(dist_lut, v.position, c.load_port)
                laden_nm = dist_nm(dist_lut, c.load_port, c.discharge_port)
            except KeyError:
                # If distance missing, skip (or you can set a penalty)
                continue

            voy = Voyage(ballast_leg_nm=ballast_nm, laden_leg_nm=laden_nm, bunker_days=1.0)
            out = calc(v, c, voy, pf)
            rows.append(out)
    return pd.DataFrame(rows).sort_values("profit_loss", ascending=False)


if __name__ == "__main__":
    # Load lookup tables
    dist_lut = load_distance_lookup(DIST_XLSX)
    vlsfo_price, mgo_price = load_bunker_prices(BUNKER_XLSX, location="Singapore", month_col="cal")
    pf = PricesAndFees(ifo_price=vlsfo_price, mdo_price=mgo_price)

    # Load datasets
    cargill_vessels = load_vessels(CARGILL_VESSEL_XLSX, speed_mode="economical")
    market_vessels = load_vessels(MARKET_VESSEL_XLSX, speed_mode="economical")

    committed = load_cargoes(COMMITTED_CARGO_XLSX)
    market_cargoes = load_cargoes(MARKET_CARGO_XLSX)

    # 1) Cargill vessels carrying committed cargoes
    df_cv_cc = build_profit_matrix(cargill_vessels, committed, dist_lut, pf)
    print("\n=== Cargill vessels x Committed cargoes (Top 10) ===")
    print(df_cv_cc.head(10)[["vessel","cargo","profit_loss","total_duration","ballast_nm","laden_nm"]])

    # 2) Cargill vessels carrying market cargoes (optional arbitrage)
    df_cv_mc = build_profit_matrix(cargill_vessels, market_cargoes, dist_lut, pf)
    print("\n=== Cargill vessels x Market cargoes (Top 10) ===")
    print(df_cv_mc.head(10)[["vessel","cargo","profit_loss","total_duration","ballast_nm","laden_nm"]])

    # 3) Market vessels carrying committed cargoes (if you want to outsource)
    df_mv_cc = build_profit_matrix(market_vessels, committed, dist_lut, pf)
    print("\n=== Market vessels x Committed cargoes (Top 10) ===")
    print(df_mv_cc.head(10)[["vessel","cargo","profit_loss","total_duration","ballast_nm","laden_nm"]])
