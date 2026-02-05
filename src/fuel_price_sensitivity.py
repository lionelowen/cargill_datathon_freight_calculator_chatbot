# fuel_price_sensitivity.py
# ============================================================
# Scenario Analysis: Find the fuel price % increase at which
# the current recommendation becomes less profitable than the next option
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
from freight_calculator import (
    load_distance_lookup, load_bunker_prices, load_all_bunker_prices,
    load_ffa, load_vessels_from_excel, load_cargoes_from_excel,
    build_profit_table, optimal_committed_assignment,
    PricesAndFees, Cargo, Vessel, Voyage, calc, dist_nm,
    get_bunker_location_for_port, _norm_port,
    BUNKER_XLSX, CARGILL_VESSEL_XLSX, MARKET_VESSEL_XLSX,
    COMMITTED_CARGO_XLSX, MARKET_CARGO_XLSX, DIST_XLSX, FFA_REPORT_XLSX,
    BASE_DIR
)


def scale_bunker_prices(base_prices: dict, increase_pct: float) -> dict:
    """
    Scale all bunker prices by a percentage increase.
    increase_pct: e.g., 0.10 = 10% increase, 0.50 = 50% increase
    """
    scaled = {}
    multiplier = 1.0 + increase_pct
    for loc, (vlsfo, mgo, vlsfo_prev, mgo_prev) in base_prices.items():
        scaled[loc] = (
            vlsfo * multiplier,
            mgo * multiplier,
            vlsfo_prev * multiplier,
            mgo_prev * multiplier,
        )
    return scaled


def build_profit_table_with_fuel_increase(
    vessels: list[Vessel],
    cargoes: list[Cargo],
    dist_lut: dict,
    base_pf: PricesAndFees,
    base_bunker_prices: dict,
    fuel_increase_pct: float = 0.0,
    bunker_days: float = 1.0,
    enforce_laycan: bool = True,
) -> pd.DataFrame:
    """
    Build profit table with scaled fuel prices.
    """
    # Scale bunker prices
    scaled_prices = scale_bunker_prices(base_bunker_prices, fuel_increase_pct)
    
    # Scale default PricesAndFees
    multiplier = 1.0 + fuel_increase_pct
    pf = PricesAndFees(
        ifo_price=base_pf.ifo_price * multiplier,
        mdo_price=base_pf.mdo_price * multiplier,
        ifo_price_prev=base_pf.ifo_price_prev * multiplier,
        mdo_price_prev=base_pf.mdo_price_prev * multiplier,
        awrp=base_pf.awrp,
        cev=base_pf.cev,
        ilhoc=base_pf.ilhoc,
        bunker_da_per_day=base_pf.bunker_da_per_day,
    )
    
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
            if bunker_location in scaled_prices:
                vlsfo, mgo, vlsfo_prev, mgo_prev = scaled_prices[bunker_location]
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

            # Calculate
            voy = Voyage(ballast_leg_nm=ballast_nm, laden_leg_nm=laden_nm, bunker_days=bunker_days)
            out = calc(v, c, voy, pf_to_use)
            out["bunker_location"] = bunker_location
            out["fuel_increase_pct"] = fuel_increase_pct

            if enforce_laycan and v_dep is not None:
                dur_ballast_days = ballast_nm / (v.sp_ballast * 24.0)
                out["eta_load"] = v_dep + pd.Timedelta(days=float(dur_ballast_days))

            rows.append(out)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("profit_loss", ascending=False).reset_index(drop=True)


def get_all_feasible_assignments(df_cv_cc: pd.DataFrame, df_mv_cc: pd.DataFrame) -> list[tuple[dict, float]]:
    """
    Get all feasible assignments (not just the optimal one).
    Returns list of (assignment_dict, total_pl) sorted by total_pl descending.
    """
    import itertools
    
    a = df_cv_cc.copy()
    a["vessel_type"] = "CARGILL"
    b = df_mv_cc.copy()
    b["vessel_type"] = "MARKET"
    cand = pd.concat([a, b], ignore_index=True)

    if len(cand) == 0:
        return []

    base_cargos = sorted(cand["base_cargo_id"].unique().tolist())
    opts = {c: cand[cand["base_cargo_id"] == c].to_dict("records") for c in base_cargos}
    
    all_assignments = []

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

        if ok:
            assignment = {r["base_cargo_id"]: {
                "vessel": r["vessel"],
                "vessel_type": r["vessel_type"],
                "discharge_port": r["discharge_port"],
                "profit_loss": r["profit_loss"],
                "bunker_expense": r.get("bunker_expense", 0),
            } for r in choice}
            all_assignments.append((assignment, total))

    # Sort by total P/L descending
    all_assignments.sort(key=lambda x: x[1], reverse=True)
    return all_assignments


def run_fuel_price_sensitivity(enforce_laycan: bool = False):
    """
    Find the fuel price % increase at which the current recommendation
    becomes less profitable than the next best option.
    
    Args:
        enforce_laycan: Whether to enforce laycan constraints (default False for testing)
    """
    print("=" * 70)
    print("SCENARIO ANALYSIS: Fuel Price Sensitivity")
    print("=" * 70)
    print(f"\nConfiguration: enforce_laycan = {enforce_laycan}")
    
    # Load data
    dist_lut = load_distance_lookup(DIST_XLSX)
    base_bunker_prices = load_all_bunker_prices(BUNKER_XLSX, month_col="mar", prev_month_col="feb")
    
    vlsfo_price, mgo_price, vlsfo_prev, mgo_prev = load_bunker_prices(
        BUNKER_XLSX, location="Singapore", month_col="mar", prev_month_col="feb"
    )
    base_pf = PricesAndFees(
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

    # Get baseline (0% increase)
    print("\n--- BASELINE (0% fuel price increase) ---")
    df_cv_cc_base = build_profit_table_with_fuel_increase(
        cargill_vessels, committed, dist_lut, base_pf, base_bunker_prices, 
        fuel_increase_pct=0.0,
        enforce_laycan=enforce_laycan
    )
    df_mv_cc_base = build_profit_table_with_fuel_increase(
        market_vessels, committed, dist_lut, base_pf, base_bunker_prices, 
        fuel_increase_pct=0.0,
        enforce_laycan=enforce_laycan
    )
    
    # Get all feasible assignments at baseline
    all_assignments_base = get_all_feasible_assignments(df_cv_cc_base, df_mv_cc_base)
    
    if len(all_assignments_base) < 2:
        print("Not enough feasible assignments to compare. Need at least 2 options.")
        return None
    
    baseline_assignment, baseline_total = all_assignments_base[0]
    second_best_assignment, second_best_total = all_assignments_base[1]
    
    print(f"\nBaseline fuel prices (Singapore): VLSFO ${vlsfo_price:.2f}/MT, MGO ${mgo_price:.2f}/MT")
    print(f"\nTotal feasible assignments: {len(all_assignments_base)}")
    
    print(f"\n#1 OPTIMAL Assignment (P/L: ${baseline_total:,.2f}):")
    for cargo_id, details in baseline_assignment.items():
        print(f"   {cargo_id}: {details['vessel']} ({details['vessel_type']}) → {details['discharge_port']} (P/L: ${details['profit_loss']:,.2f})")
    
    print(f"\n#2 NEXT BEST Assignment (P/L: ${second_best_total:,.2f}):")
    for cargo_id, details in second_best_assignment.items():
        print(f"   {cargo_id}: {details['vessel']} ({details['vessel_type']}) → {details['discharge_port']} (P/L: ${details['profit_loss']:,.2f})")
    
    gap_at_baseline = baseline_total - second_best_total
    print(f"\nGap between #1 and #2 at baseline: ${gap_at_baseline:,.2f}")

    # Test increasing fuel prices
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Fuel Price Increase")
    print("=" * 70)
    print(f"{'Increase %':<12} {'#1 P/L':>15} {'#2 P/L':>15} {'Gap':>12} {'Status':<15}")
    print("-" * 70)

    breakeven_pct = None
    # Test percentages: 0% to 100% in reasonable steps
    test_percentages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    max_test_pct = test_percentages[-1]

    results = []
    
    for pct in test_percentages:
        increase = pct / 100.0
        
        df_cv_cc = build_profit_table_with_fuel_increase(
            cargill_vessels, committed, dist_lut, base_pf, base_bunker_prices,
            fuel_increase_pct=increase,
            enforce_laycan=enforce_laycan
        )
        df_mv_cc = build_profit_table_with_fuel_increase(
            market_vessels, committed, dist_lut, base_pf, base_bunker_prices,
            fuel_increase_pct=increase,
            enforce_laycan=enforce_laycan
        )
        
        try:
            all_assignments = get_all_feasible_assignments(df_cv_cc, df_mv_cc)
            
            if len(all_assignments) < 2:
                continue
                
            current_best, current_best_total = all_assignments[0]
            current_second, current_second_total = all_assignments[1]
            
            gap = current_best_total - current_second_total
            
            # Check if the optimal assignment changed
            current_best_key = tuple(sorted((k, v["vessel"], v["discharge_port"]) 
                                           for k, v in current_best.items()))
            baseline_key = tuple(sorted((k, v["vessel"], v["discharge_port"]) 
                                       for k, v in baseline_assignment.items()))
            
            assignment_changed = current_best_key != baseline_key
            
            if assignment_changed:
                status = ">>> CHANGED!"
                if breakeven_pct is None:
                    breakeven_pct = pct
            else:
                status = "Same"
            
            print(f"{pct:>10}%  ${current_best_total:>14,.2f} ${current_second_total:>14,.2f} ${gap:>11,.2f} {status:<15}")
            
            results.append({
                "increase_pct": pct,
                "best_total": current_best_total,
                "second_total": current_second_total,
                "gap": gap,
                "changed": assignment_changed,
                "best_assignment": current_best,
            })
            
        except Exception as e:
            print(f"{pct:>10}%  ERROR: {e}")

    # Find exact breakeven using binary search if assignment changed
    print("\n" + "=" * 70)
    
    if breakeven_pct is not None:
        print(f"\n🔴 BREAKEVEN FOUND: Assignment changes at {breakeven_pct}% fuel price increase")
        
        # Binary search for exact breakeven
        print("\n--- Refining breakeven point ---")
        low = breakeven_pct - (test_percentages[test_percentages.index(breakeven_pct)] - 
                               test_percentages[test_percentages.index(breakeven_pct) - 1] 
                               if test_percentages.index(breakeven_pct) > 0 else 5)
        high = breakeven_pct
        
        for _ in range(10):  # 10 iterations of binary search
            mid = (low + high) / 2
            increase = mid / 100.0
            
            df_cv_cc = build_profit_table_with_fuel_increase(
                cargill_vessels, committed, dist_lut, base_pf, base_bunker_prices,
                fuel_increase_pct=increase,
                enforce_laycan=enforce_laycan
            )
            df_mv_cc = build_profit_table_with_fuel_increase(
                market_vessels, committed, dist_lut, base_pf, base_bunker_prices,
                fuel_increase_pct=increase,
                enforce_laycan=enforce_laycan
            )
            
            all_assignments = get_all_feasible_assignments(df_cv_cc, df_mv_cc)
            current_best, _ = all_assignments[0]
            
            current_best_key = tuple(sorted((k, v["vessel"], v["discharge_port"]) 
                                           for k, v in current_best.items()))
            baseline_key = tuple(sorted((k, v["vessel"], v["discharge_port"]) 
                                       for k, v in baseline_assignment.items()))
            
            if current_best_key != baseline_key:
                high = mid
            else:
                low = mid
        
        exact_breakeven = high
        print(f"\nExact breakeven: ~{exact_breakeven:.1f}% fuel price increase")
        
        # Show what changes at breakeven
        print("\n--- New Optimal Assignment at Breakeven ---")
        increase = exact_breakeven / 100.0
        df_cv_cc = build_profit_table_with_fuel_increase(
            cargill_vessels, committed, dist_lut, base_pf, base_bunker_prices,
            fuel_increase_pct=increase,
            enforce_laycan=enforce_laycan
        )
        df_mv_cc = build_profit_table_with_fuel_increase(
            market_vessels, committed, dist_lut, base_pf, base_bunker_prices,
            fuel_increase_pct=increase,
            enforce_laycan=enforce_laycan
        )
        all_assignments = get_all_feasible_assignments(df_cv_cc, df_mv_cc)
        new_best, new_best_total = all_assignments[0]
        
        print(f"\nNew optimal (P/L: ${new_best_total:,.2f}):")
        for cargo_id, details in new_best.items():
            old_vessel = baseline_assignment[cargo_id]["vessel"]
            old_port = baseline_assignment[cargo_id]["discharge_port"]
            changed = "***" if (details["vessel"], details["discharge_port"]) != (old_vessel, old_port) else ""
            print(f"   {cargo_id}: {details['vessel']} → {details['discharge_port']} {changed}")
        
        # Calculate fuel price at breakeven
        breakeven_vlsfo = vlsfo_price * (1 + exact_breakeven/100)
        breakeven_mgo = mgo_price * (1 + exact_breakeven/100)
        print(f"\nFuel prices at breakeven:")
        print(f"   VLSFO: ${vlsfo_price:.2f} → ${breakeven_vlsfo:.2f}/MT")
        print(f"   MGO: ${mgo_price:.2f} → ${breakeven_mgo:.2f}/MT")
        
    else:
        print(f"\n🟢 The current recommendation remains OPTIMAL even with {max_test_pct}% fuel price increase!")
        print("   The assignment is highly robust to fuel price volatility.")
        exact_breakeven = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if exact_breakeven:
        print(f"\n📊 Breakeven fuel price increase: {exact_breakeven:.1f}%")
        print(f"\n   At this point, the optimal vessel-cargo assignment changes.")
        print(f"   Current VLSFO ${vlsfo_price:.2f}/MT would need to reach ${vlsfo_price * (1 + exact_breakeven/100):.2f}/MT")
    else:
        print(f"\n📊 No breakeven found up to {max_test_pct}% fuel increase.")
        print(f"   The current assignment is robust to extreme fuel price scenarios.")

    # Cost sensitivity per 10% increase
    print("\n--- Cost Impact per 10% Fuel Increase ---")
    df_cv_10 = build_profit_table_with_fuel_increase(
        cargill_vessels, committed, dist_lut, base_pf, base_bunker_prices,
        fuel_increase_pct=0.10,
        enforce_laycan=enforce_laycan
    )
    df_mv_10 = build_profit_table_with_fuel_increase(
        market_vessels, committed, dist_lut, base_pf, base_bunker_prices,
        fuel_increase_pct=0.10,
        enforce_laycan=enforce_laycan
    )
    _, total_10pct = optimal_committed_assignment(df_cv_10, df_mv_10)
    cost_per_10pct = baseline_total - total_10pct
    
    print(f"Each 10% fuel price increase reduces P/L by approximately: ${cost_per_10pct:,.2f}")

    return exact_breakeven, results


if __name__ == "__main__":
    breakeven, results = run_fuel_price_sensitivity(enforce_laycan=True)
