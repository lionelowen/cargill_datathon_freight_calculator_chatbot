# scenario_analysis.py
# ============================================================
# Scenario Analysis: Find the breakeven point for China port delays
# that would change the optimal vessel-cargo recommendation
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
from freight_calculator import (
    load_distance_lookup, load_bunker_prices, load_all_bunker_prices,
    load_ffa, load_vessels_from_excel, load_cargoes_from_excel,
    build_profit_table, optimal_committed_assignment,
    assign_market_cargo_to_unused_cargill,
    PricesAndFees, Cargo, Vessel, Voyage, calc, dist_nm,
    get_bunker_location_for_port, _norm_port,
    BUNKER_XLSX, CARGILL_VESSEL_XLSX, MARKET_VESSEL_XLSX,
    COMMITTED_CARGO_XLSX, MARKET_CARGO_XLSX, DIST_XLSX, FFA_REPORT_XLSX
)
import os

# China ports (discharge ports where delays would apply)
CHINA_PORTS = {
    "QINGDAO", "SHANGHAI", "TIANJIN", "RIZHAO", "CAOFEIDIAN", 
    "JINGTANG", "FANGCHENG", "XIAMEN", "NINGBO", "LIANYUNGANG",
    "GUANGZHOU", "DALIAN", "YANTAI", "ZHANGJIAGANG", "NANTONG",
    "NANJING", "ZHANJIANG", "YINGKOU"
}


def is_china_port(port: str) -> bool:
    """Check if a port is in China"""
    port_norm = _norm_port(port)
    return port_norm in CHINA_PORTS


def add_china_delay_to_profit_table(df: pd.DataFrame, extra_delay_days: float) -> pd.DataFrame:
    """
    Add extra delay days to voyages with China discharge ports.
    This modifies the profit_loss, total_duration, hire_net, etc.
    
    Args:
        df: Profit table from build_profit_table
        extra_delay_days: Additional delay days to add for China ports
    
    Returns:
        Modified DataFrame with adjusted profit/loss for China port delays
    """
    if df.empty or extra_delay_days == 0:
        return df.copy()
    
    df = df.copy()
    
    for idx, row in df.iterrows():
        if is_china_port(row["discharge_port"]):
            # Original values
            original_duration = row["total_duration"]
            original_hire_net = row["hire_net"]
            original_profit = row["profit_loss"]
            
            # Calculate hire rate per day (approximate from hire_net / duration)
            # Note: This is an approximation since hire_net = hire_gross * (1 - adcoms)
            if original_duration > 0:
                hire_per_day = original_hire_net / original_duration
            else:
                hire_per_day = 0
            
            # Additional hire cost due to delay
            extra_hire_cost = hire_per_day * extra_delay_days
            
            # Update values
            df.at[idx, "total_duration"] = original_duration + extra_delay_days
            df.at[idx, "hire_net"] = original_hire_net + extra_hire_cost
            df.at[idx, "profit_loss"] = original_profit - extra_hire_cost
            
            # Mark the extra delay
            if "china_extra_delay" not in df.columns:
                df["china_extra_delay"] = 0.0
            df.at[idx, "china_extra_delay"] = extra_delay_days
    
    # Re-sort by profit_loss
    df = df.sort_values("profit_loss", ascending=False).reset_index(drop=True)
    return df


def build_profit_table_with_china_delay(
    vessels: list[Vessel],
    cargoes: list[Cargo],
    dist_lut: dict,
    pf: PricesAndFees,
    bunker_prices_by_location: dict,
    china_delay_days: float = 0.0,
    bunker_days: float = 1.0,
    enforce_laycan: bool = True,
) -> pd.DataFrame:
    """
    Build profit table with additional delay days for China ports.
    """
    rows = []

    for v in vessels:
        v_dep = v.time_of_departure if isinstance(v.time_of_departure, pd.Timestamp) else None

        for c in cargoes:
            try:
                ballast_nm = dist_nm(dist_lut, v.position, c.load_port)
                laden_nm = dist_nm(dist_lut, c.load_port, c.discharge_port)
            except KeyError:
                continue

            # laycan filter
            if enforce_laycan and v_dep is not None:
                dur_ballast_days = ballast_nm / (v.sp_ballast * 24.0)
                eta_load = v_dep + pd.Timedelta(days=float(dur_ballast_days))

                if isinstance(c.earliest_date, pd.Timestamp) and pd.notna(c.earliest_date):
                    if eta_load < c.earliest_date:
                        continue
                if isinstance(c.latest_date, pd.Timestamp) and pd.notna(c.latest_date):
                    if eta_load > c.latest_date:
                        continue

            # Select bunker prices based on vessel position
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
            else:
                pf_to_use = pf

            # Add China port delay to discharge time if applicable
            extra_delay = china_delay_days if is_china_port(c.discharge_port) else 0.0
            
            # Calculate with delay
            out = calc_with_delay(v, c, ballast_nm, laden_nm, pf_to_use, bunker_days, extra_delay)
            out["bunker_location"] = bunker_location
            out["china_delay_days"] = extra_delay

            if enforce_laycan and v_dep is not None:
                dur_ballast_days = ballast_nm / (v.sp_ballast * 24.0)
                out["eta_load"] = v_dep + pd.Timedelta(days=float(dur_ballast_days))

            rows.append(out)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("profit_loss", ascending=False).reset_index(drop=True)


def calc_with_delay(v: Vessel, c: Cargo, ballast_nm: float, laden_nm: float, 
                    pf: PricesAndFees, bunker_days: float, extra_delay_days: float) -> dict:
    """
    Calculate profit/loss with additional delay days at discharge port.
    """
    # durations at sea
    dur_ballast_days = ballast_nm / (v.sp_ballast * 24.0)
    dur_laden_days = laden_nm / (v.sp_laden * 24.0)

    # port time (loading/discharging) + extra delay
    loadport_days = (c.cargo_qty / c.load_rate) + c.loadport_tt + c.port_idle if c.load_rate > 0 else (c.loadport_tt + c.port_idle)
    disport_days = (c.cargo_qty / c.dis_rate) + c.disport_tt if c.dis_rate > 0 else c.disport_tt
    disport_days += extra_delay_days  # Add China port delay

    total_duration = (dur_ballast_days + dur_laden_days) + bunker_days + loadport_days + disport_days

    # revenue
    loaded_qty = min(v.dwt, c.cargo_qty)
    revenue = loaded_qty * c.freight * (1 - c.address_coms - c.broker_coms)

    # hire (increases with delay)
    hire_gross = v.daily_hire * total_duration + c.ballast_bonus
    hire_net = hire_gross * (1 - v.adcoms)

    # fuel usage (port fuel increases with delay)
    ifo_sea = dur_ballast_days * v.ifo_ballast + dur_laden_days * v.ifo_laden
    mdo_sea = dur_ballast_days * v.mdo_ballast + dur_laden_days * v.mdo_laden

    ifo_port = loadport_days * v.ifo_port_work + disport_days * v.ifo_port_work + c.port_idle * v.ifo_port_idle
    mdo_port = loadport_days * v.mdo_port_work + disport_days * v.mdo_port_work + c.port_idle * v.mdo_port_idle

    total_ifo = ifo_sea + ifo_port
    total_mdo = mdo_sea + mdo_port

    # bunker expense
    if total_ifo <= v.ifo_remaining:
        ifo_expense = total_ifo * pf.ifo_price_prev
    else:
        ifo_expense = v.ifo_remaining * pf.ifo_price_prev + (total_ifo - v.ifo_remaining) * pf.ifo_price
    
    if total_mdo <= v.mdo_remaining:
        mdo_expense = total_mdo * pf.mdo_price_prev
    else:
        mdo_expense = v.mdo_remaining * pf.mdo_price_prev + (total_mdo - v.mdo_remaining) * pf.mdo_price
    
    bunker_expense = ifo_expense + mdo_expense

    # misc expense
    bunker_da = pf.bunker_da_per_day * bunker_days
    misc_expense = (
        pf.awrp + pf.cev + pf.ilhoc
        + bunker_da
        + float(c.total_port_cost or 0.0)
    )

    profit_loss = revenue - hire_net - bunker_expense - misc_expense

    if any(pd.isna(x) for x in [revenue, hire_net, bunker_expense, misc_expense]):
        profit_loss = np.nan

    voyage_costs = bunker_expense + misc_expense
    tce = (revenue - voyage_costs) / total_duration if total_duration > 0 else 0.0

    return {
        "vessel": v.name,
        "cargo": c.name,
        "base_cargo_id": c.base_cargo_id or c.name,
        "from_pos": v.position,
        "load_port": c.load_port,
        "discharge_port": c.discharge_port,
        "ballast_nm": ballast_nm,
        "laden_nm": laden_nm,
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


def run_scenario_analysis(enforce_laycan: bool = False):
    """
    Find the number of additional China port delay days that would make
    the current recommendation no longer optimal.
    
    Args:
        enforce_laycan: Whether to enforce laycan constraints (default False for testing)
    """
    print("=" * 70)
    print("SCENARIO ANALYSIS: China Port Delay Sensitivity")
    print("=" * 70)
    print(f"\nConfiguration: enforce_laycan = {enforce_laycan}")
    
    # Load data
    dist_lut = load_distance_lookup(DIST_XLSX)
    all_bunker_prices = load_all_bunker_prices(BUNKER_XLSX, month_col="mar", prev_month_col="feb")
    
    vlsfo_price, mgo_price, vlsfo_prev, mgo_prev = load_bunker_prices(
        BUNKER_XLSX, location="Singapore", month_col="mar", prev_month_col="feb"
    )
    pf = PricesAndFees(
        ifo_price=vlsfo_price,
        mdo_price=mgo_price,
        ifo_price_prev=vlsfo_prev,
        mdo_price_prev=mgo_prev,
    )

    ffa = load_ffa(FFA_REPORT_XLSX)

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

    committed = load_cargoes_from_excel(COMMITTED_CARGO_XLSX, tag="COMMITTED")
    market_cargoes = load_cargoes_from_excel(MARKET_CARGO_XLSX, tag="MARKET")

    # Identify which cargoes have China discharge ports
    print("\n--- CARGO ANALYSIS ---")
    china_cargoes = []
    non_china_cargoes = []
    for c in committed:
        if is_china_port(c.discharge_port):
            china_cargoes.append(c.name)
            print(f"  🇨🇳 {c.name}: {c.load_port} → {c.discharge_port} (CHINA)")
        else:
            non_china_cargoes.append(c.name)
            print(f"  🌍 {c.name}: {c.load_port} → {c.discharge_port}")
    
    if not china_cargoes:
        print("\n⚠️  No committed cargoes have China discharge ports!")
        print("    China port delay analysis will not affect the assignment.")

    # Get baseline (0 delay days)
    print("\n--- BASELINE (0 extra delay days) ---")
    df_cv_cc_base = build_profit_table_with_china_delay(
        cargill_vessels, committed, dist_lut, pf, all_bunker_prices, 
        china_delay_days=0.0,
        enforce_laycan=enforce_laycan
    )
    df_mv_cc_base = build_profit_table_with_china_delay(
        market_vessels, committed, dist_lut, pf, all_bunker_prices, 
        china_delay_days=0.0,
        enforce_laycan=enforce_laycan
    )
    
    # Pass cargill_vessels for sunk cost calculation
    baseline_plan, baseline_total = optimal_committed_assignment(
        df_cv_cc_base, df_mv_cc_base, 
        cargill_vessels=cargill_vessels,
        consider_sunk_cost=True
    )
    baseline_vessels = set(baseline_plan["vessel"].tolist())
    baseline_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"], row["vessel_type"]) 
                          for _, row in baseline_plan.iterrows()}
    
    print(f"Total P/L (with sunk cost): ${baseline_total:,.2f}")
    print("\nOptimal Assignment:")
    for _, row in baseline_plan.iterrows():
        vessel_type = row.get('vessel_type', 'UNKNOWN')
        china_flag = " 🇨🇳" if is_china_port(row['discharge_port']) else ""
        print(f"  {row['base_cargo_id']}: [{vessel_type}] {row['vessel']} → {row['discharge_port']}{china_flag} (P/L: ${row['profit_loss']:,.2f})")

    # Test increasing delay days
    print("\n--- SENSITIVITY ANALYSIS ---")
    print(f"{'Delay Days':<12} {'Total P/L':>15} {'Change':>12} {'Assignment Changed?':<20}")
    print("-" * 70)

    breakeven_delay = None
    previous_plan = baseline_plan.copy()
    previous_assignment = baseline_assignment.copy()
    
    # Finer granularity for delay testing
    delay_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 75, 100]

    for delay_days in delay_values:
        df_cv_cc = build_profit_table_with_china_delay(
            cargill_vessels, committed, dist_lut, pf, all_bunker_prices,
            china_delay_days=delay_days,
            enforce_laycan=enforce_laycan
        )
        df_mv_cc = build_profit_table_with_china_delay(
            market_vessels, committed, dist_lut, pf, all_bunker_prices,
            china_delay_days=delay_days,
            enforce_laycan=enforce_laycan
        )
        
        try:
            plan, total = optimal_committed_assignment(
                df_cv_cc, df_mv_cc,
                cargill_vessels=cargill_vessels,
                consider_sunk_cost=True
            )
            current_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"], row.get("vessel_type", "")) 
                                 for _, row in plan.iterrows()}
            
            change = total - baseline_total
            assignment_changed = current_assignment != baseline_assignment
            
            status = "YES - CHANGED!" if assignment_changed else "No"
            print(f"{delay_days:<12} ${total:>14,.2f} ${change:>11,.2f} {status:<20}")
            
            if assignment_changed and breakeven_delay is None:
                breakeven_delay = delay_days
                print(f"\n>>> BREAKEVEN FOUND at {delay_days} extra delay days! <<<")
                print("\nAssignment Changes:")
                for cargo_id, (new_vessel, new_port, new_type) in current_assignment.items():
                    old_vessel, old_port, old_type = baseline_assignment.get(cargo_id, ('N/A', 'N/A', 'N/A'))
                    if (new_vessel, new_port) != (old_vessel, old_port):
                        print(f"  {cargo_id}:")
                        print(f"    BEFORE: [{old_type}] {old_vessel} → {old_port}")
                        print(f"    AFTER:  [{new_type}] {new_vessel} → {new_port} ***")
                    else:
                        print(f"  {cargo_id}: [{new_type}] {new_vessel} → {new_port} (unchanged)")
                
        except Exception as e:
            print(f"{delay_days:<12} ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if breakeven_delay is not None:
        print(f"\n🔴 The current recommendation becomes SUB-OPTIMAL at {breakeven_delay} extra days of China port delay.")
        print(f"\n   This means if China ports experience {breakeven_delay}+ additional days of congestion/delay,")
        print(f"   you should reconsider the vessel-cargo assignment.")
    else:
        max_delay = delay_values[-1]
        print(f"\n🟢 The current recommendation remains OPTIMAL even with {max_delay}+ extra days of China port delay.")
        print(f"   The assignment is robust to port congestion scenarios.")

    # Calculate cost per delay day
    print("\n--- COST IMPACT PER DELAY DAY ---")
    df_cv_1 = build_profit_table_with_china_delay(
        cargill_vessels, committed, dist_lut, pf, all_bunker_prices,
        china_delay_days=1.0,
        enforce_laycan=enforce_laycan
    )
    df_mv_1 = build_profit_table_with_china_delay(
        market_vessels, committed, dist_lut, pf, all_bunker_prices,
        china_delay_days=1.0,
        enforce_laycan=enforce_laycan
    )
    _, total_1day = optimal_committed_assignment(
        df_cv_1, df_mv_1,
        cargill_vessels=cargill_vessels,
        consider_sunk_cost=True
    )
    cost_per_day = baseline_total - total_1day
    
    print(f"Each additional day of China port delay costs approximately: ${cost_per_day:,.2f}")
    print(f"This includes: hire cost + port fuel consumption")
    
    # Save results to file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "scenario_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Scenario Analysis Report: China Port Delay Sensitivity\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Enforce Laycan: {enforce_laycan}\n\n")
        f.write(f"## Baseline (0 extra delay)\n")
        f.write(f"- Total P/L: ${baseline_total:,.2f}\n\n")
        f.write("### Optimal Assignment\n")
        for cargo_id, (vessel, port, vtype) in baseline_assignment.items():
            china = " (China)" if is_china_port(port) else ""
            f.write(f"- {cargo_id}: [{vtype}] {vessel} -> {port}{china}\n")
        f.write(f"\n## Key Findings\n")
        if breakeven_delay:
            f.write(f"- **Tipping Point**: {breakeven_delay} extra days of China port delay\n")
        else:
            f.write(f"- **Tipping Point**: Not found (assignment stable up to {delay_values[-1]}+ days)\n")
        f.write(f"- **Cost per delay day**: ${cost_per_day:,.2f}\n")
    
    print(f"\n📄 Report saved to: {report_path}")

    return breakeven_delay


if __name__ == "__main__":
    breakeven = run_scenario_analysis(enforce_laycan=False)
