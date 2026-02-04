# Quick script to compare vessel characteristics
import sys
sys.path.insert(0, r'D:\Personal Study\cargill_datathon_freight_calculator_chatbot\src')

from freight_calculator import (
    load_vessels_from_excel, load_ffa, load_distance_lookup,
    CARGILL_VESSEL_XLSX, FFA_REPORT_XLSX, DIST_XLSX, dist_nm
)

ffa = load_ffa(FFA_REPORT_XLSX)
dist_lut = load_distance_lookup(DIST_XLSX)

vessels = load_vessels_from_excel(
    CARGILL_VESSEL_XLSX, 
    speed_mode='economical', 
    is_market=False, 
    ffa=ffa
)

print("\n=== CARGILL VESSEL COMPARISON ===\n")
print(f"{'Vessel':<18} {'Daily Hire':>12} {'DWT':>10} {'Position':>18} {'Ballast to LIANYUNGANG':>25}")
print("-" * 90)

for v in vessels:
    try:
        ballast_nm = dist_nm(dist_lut, v.position, "LIANYUNGANG")
        ballast_days = ballast_nm / (v.sp_ballast * 24.0)
        ballast_info = f"{ballast_nm:,.0f} nm ({ballast_days:.1f} days)"
    except:
        ballast_info = "N/A"
    
    print(f"{v.name:<18} ${v.daily_hire:>10,.2f} {v.dwt:>10,.0f} {v.position:>18} {ballast_info:>25}")

print("\n=== KEY INSIGHT ===")
print("When delay increases, the HIRE COST per day matters more!")
print("A vessel with LOWER daily hire rate becomes more attractive for longer voyages.")

# Calculate cost impact
print("\n=== COST IMPACT OF 40 EXTRA DELAY DAYS ===\n")
for v in vessels:
    extra_cost = v.daily_hire * 40 * (1 - v.adcoms)  # Net hire cost
    print(f"{v.name}: Extra cost for 40 days delay = ${extra_cost:,.2f}")
