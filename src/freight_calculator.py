from dataclasses import dataclass, asdict

# ----------------------------
# Inputs (change these anytime)
# ----------------------------

@dataclass
class Vessel:
    name: str = "mv cargill tbn"
    dwt: float = 62000
    grain_capacity_cbm: float = 70000

    # Sea speeds (knots)
    sp_ballast: float = 14
    sp_laden: float = 12

    # Sea fuel consumption (tons/day)
    ifo_ballast: float = 23
    ifo_laden: float = 18.5
    mdo_ballast: float = 0.1
    mdo_laden: float = 0.1

    # Port fuel consumption (tons/day)
    ifo_port_work: float = 5.5
    mdo_port_work: float = 0.1
    ifo_port_idle: float = 5.5
    mdo_port_idle: float = 0.1

    # Hire
    daily_hire: float = 12000
    adcoms: float = 0.0375  # e.g. 3.75%


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


@dataclass
class Voyage:
    # Distances (NM)
    ballast_leg_nm: float = 3000
    laden_leg_nm: float = 3000

    # Other days
    bunker_days: float = 1.0  # Excel had BUNKER (DAYS) = 1


@dataclass
class PricesAndFees:
    ifo_price: float = 440
    mdo_price: float = 850

    # Misc fixed costs (edit as needed)
    awrp: float = 1500
    cev: float = 1500
    ilhoc: float = 5000

    # Port disbursement account
    pda_loadport: float = 20000
    pda_disport: float = 20000

    # Optional: bunker DA per bunker day (set 0 if not used)
    bunker_da_per_day: float = 1500


# ----------------------------
# Calculator (formulas)
# ----------------------------

def calc(v: Vessel, c: Cargo, voy: Voyage, pf: PricesAndFees) -> dict:
    # --- Distances -> durations (Excel: duration = distance/(speed*24)) ---
    dur_ballast_days = voy.ballast_leg_nm / (v.sp_ballast * 24)
    dur_laden_days = voy.laden_leg_nm / (v.sp_laden * 24)

    steaming_days = dur_ballast_days + dur_laden_days

    # --- Port time (Excel-style) ---
    # LOADPORT days = cargo_qty/load_rate + loadport_tt + port_idle
    loadport_days = (c.cargo_qty / c.load_rate) + c.loadport_tt + c.port_idle

    # DISPORT days = cargo_qty/dis_rate + disport_tt
    disport_days = (c.cargo_qty / c.dis_rate) + c.disport_tt

    total_duration = steaming_days + voy.bunker_days + loadport_days + disport_days

    # --- Loaded qty (Excel: loaded_qty = grain_capacity_cbm / stow_factor) ---
    loaded_qty = v.grain_capacity_cbm / c.stow_factor

    # --- Revenue (Excel: revenue = loaded_qty * freight * (1 - address - broker)) ---
    revenue = loaded_qty * c.freight * (1 - c.address_coms - c.broker_coms)

    # --- Hire (Excel: hire_gross = daily_hire*total_duration + ballast_bonus) ---
    hire_gross = v.daily_hire * total_duration + c.ballast_bonus
    hire_net = hire_gross * (1 - v.adcoms)

    # --- Fuel at sea (Excel: ifo_sea = dur_ballast*ifo_ballast + dur_laden*ifo_laden, etc.) ---
    ifo_sea = dur_ballast_days * v.ifo_ballast + dur_laden_days * v.ifo_laden
    mdo_sea = dur_ballast_days * v.mdo_ballast + dur_laden_days * v.mdo_laden

    # --- Fuel in port (Excel-ish: loadport + disport at working rate + idle at idle rate) ---
    ifo_port = loadport_days * v.ifo_port_work + disport_days * v.ifo_port_work + c.port_idle * v.ifo_port_idle
    mdo_port = loadport_days * v.mdo_port_work + disport_days * v.mdo_port_work + c.port_idle * v.mdo_port_idle

    total_ifo = ifo_sea + ifo_port
    total_mdo = mdo_sea + mdo_port

    # --- Bunker expense (simple) ---
    bunker_expense = total_ifo * pf.ifo_price + total_mdo * pf.mdo_price

    # --- Misc expense (Excel had SUM of a block; keep it editable) ---
    bunker_da = pf.bunker_da_per_day * voy.bunker_days
    misc_expense = (
        pf.awrp + pf.cev + pf.ilhoc
        + pf.pda_loadport + pf.pda_disport
        + bunker_da
    )

    # --- Profit & Loss (structure: revenue - hire - bunker - misc) ---
    profit_loss = revenue - hire_net - bunker_expense - misc_expense

    return {
        "dur_ballast_days": dur_ballast_days,
        "dur_laden_days": dur_laden_days,
        "steaming_days": steaming_days,
        "bunker_days": voy.bunker_days,
        "loadport_days": loadport_days,
        "disport_days": disport_days,
        "total_duration": total_duration,
        "loaded_qty": loaded_qty,
        "revenue": revenue,
        "hire_gross": hire_gross,
        "hire_net": hire_net,
        "ifo_sea": ifo_sea,
        "mdo_sea": mdo_sea,
        "ifo_port": ifo_port,
        "mdo_port": mdo_port,
        "total_ifo": total_ifo,
        "total_mdo": total_mdo,
        "bunker_expense": bunker_expense,
        "misc_expense": misc_expense,
        "profit_loss": profit_loss,
    }


if __name__ == "__main__":
    v = Vessel()
    c = Cargo()
    voy = Voyage()
    pf = PricesAndFees()

    out = calc(v, c, voy, pf)

    # Pretty print key outputs
    keys = [
        "total_duration", "loaded_qty", "revenue",
        "hire_net", "bunker_expense", "misc_expense", "profit_loss"
    ]
    for k in keys:
        print(f"{k:15s}: {out[k]:,.4f}")
