# 🚢 Freight Calculator Analysis Report

**Generated:** January 31, 2026  
**Project:** Cargill Datathon Freight Calculator

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Committed Cargo P/L** | $1,648,907.93 |
| **Market Cargo P/L** | $0.00 |
| **Idle Vessel Penalty** | -$42,831.25 |
| **Net Portfolio P/L** | **$1,606,076.68** |

---

## 📋 Optimal Vessel-Cargo Assignment

The optimization algorithm assigned **3 committed cargoes** to maximize total portfolio profit:

| Cargo | Vessel | Type | Discharge Port | Duration | TCE |
|-------|--------|------|----------------|----------|-----|
| COMMITTED_BHP_Iron Ore_1 | ANN BELL | CARGILL | Primary | 35.27 days | - |
| COMMITTED_CSN_Iron Ore_2 | ANN BELL | CARGILL | RIZHAO | 86.90 days | $18,077/day |
| COMMITTED_EGA_Bauxite_0 | ZENITH GLORY | MARKET | QINGDAO | 79.24 days | $21,029/day |

### Vessel Utilization

| Status | Vessels |
|--------|---------|
| ✅ **Used** | ANN BELL |
| ⚠️ **Unused (Cargill)** | GOLDEN ASCENT, OCEAN HORIZON, PACIFIC GLORY |

---

## 🎯 Rationale Behind Recommendations

### 1. Optimization Objective
The algorithm uses **combinatorial optimization** to:
- Assign each committed cargo exactly **once**
- Use each vessel at most **once**
- Select the **optimal discharge port** among alternatives
- **Maximize total portfolio P/L**

### 2. Why ANN BELL for Iron Ore Cargoes?
- ✅ Cargill vessels are prioritized (already on-hire commitment)
- ✅ Favorable **position proximity** to load ports (reduces ballast distance)
- ✅ Lower ballast distance → lower fuel costs → higher profit

### 3. Why ZENITH GLORY (Market) for Bauxite?
- ✅ Market vessel offered **better TCE ($21,029/day)** for this route
- ✅ ETA aligns within the **laycan window**
- ✅ Better overall portfolio optimization

### 4. Discharge Port Selection Logic
**RIZHAO selected over QINGDAO/TIANJIN for Iron Ore:**
- 2% TCE tolerance rule applied
- RIZHAO: **10,728 nm** laden distance
- QINGDAO: 11,371 nm (6.48% worse TCE)
- TIANJIN: 11,373 nm (6.50% worse TCE)

---

## 🔧 Key Assumptions

### Vessel Operations

| Parameter | Value | Notes |
|-----------|-------|-------|
| Speed Mode | Economical | Uses economical_speed columns |
| Daily Hire (Cargill) | From vessel file | Auto-converted: 11.75 → $11,750/day |
| Daily Hire (Market) | FFA 5TC rate | Interpolated from ffa_report.xlsx |
| Bunker Days | 1.0 day | Default bunkering stop |

### ⛽ Fuel Prices (Location-Based Selection)

The calculator **dynamically selects bunker prices** based on vessel position, using the nearest bunkering hub:

| Bunkering Hub | VLSFO (Mar) | MGO (Mar) | Vessel Positions Using This Hub |
|---------------|-------------|-----------|--------------------------------|
| **Qingdao** | $643/MT | $833/MT | Qingdao, Fangcheng, Caofeidian, Jingtang, Tianjin, Rizhao |
| **Shanghai** | $645/MT | $836/MT | Shanghai, Xiamen, Ningbo |
| **Singapore** | $490/MT | $649/MT | Map Ta Phut, Gwangyang, Paradip, Vizag, Port Hedland, Dampier |
| **Fujairah** | $478/MT | $638/MT | Kandla, Mundra, Jubail |
| **Rotterdam** | $467/MT | $613/MT | Rotterdam, Port Talbot, Amsterdam |
| **Gibraltar** | $474/MT | $623/MT | Kamsar, Tubarao, Ponta da Madeira |
| **Durban** | $437/MT | $510/MT | Durban, Saldanha Bay |
| **Richards Bay** | $441/MT | $519/MT | Richards Bay |
| **Port Louis** | $454/MT | $583/MT | Port Louis |

**Note:** Vessels in China (e.g., ANN BELL at Qingdao) use higher Qingdao bunker prices (~$643/MT) instead of cheaper Singapore prices (~$490/MT), resulting in more realistic P/L calculations.

### Commissions & Fees

| Item | Rate |
|------|------|
| Address Commission (ADCOMS) | 3.75% |
| Broker Commission | 1.25% |
| AWRP (War Risk Premium) | $1,500 |
| CEV (Classification Extra Voyage) | $1,500 |
| ILHOC (Insurance, Legal, etc.) | $5,000 |

### Laycan Enforcement
- ✅ **Enabled** - Vessel ETA must fall within cargo's earliest/latest dates
- Automatically filters out infeasible vessel-cargo combinations

---

## 📐 Calculation Methodology

### Profit/Loss Formula

```
Profit/Loss = Revenue - Hire Cost - Bunker Expense - Misc Expense
```

**Where:**

| Component | Formula |
|-----------|---------|
| **Revenue** | Loaded Qty × Freight Rate × (1 - Address Coms - Broker Coms) |
| **Hire Cost** | (Daily Hire × Total Duration + Ballast Bonus) × (1 - ADCOMS) |
| **Bunker Expense** | (IFO usage × IFO price) + (MDO usage × MDO price) |
| **Misc Expense** | AWRP + CEV + ILHOC + Bunker DA + Port Costs |

**Bunker Price Selection:**
- Bunker prices are selected based on **vessel's current position**
- Each port is mapped to the **nearest bunkering hub** (e.g., Qingdao → Qingdao, Gwangyang → Singapore)
- Remaining fuel onboard is valued at **previous month's prices**

### Time Charter Equivalent (TCE)

```
TCE = (Revenue - Bunker Expense - Misc Expense) / Total Duration
```

### Voyage Duration

```
Total Duration = Ballast Days + Laden Days + Bunker Days + Load Port Days + Discharge Port Days
```

---

## 🔄 Alternative Discharge Port Analysis

Ports within **2% TCE tolerance** are considered equivalent options:

| Vessel | Cargo | Port | Laden (nm) | Duration | P/L | TCE | Within 2%? |
|--------|-------|------|------------|----------|-----|-----|------------|
| ANN BELL | CSN_Iron Ore_2 | **RIZHAO** | 10,728 | 86.90 | $588,088 | $18,077 | ✅ YES |
| ANN BELL | CSN_Iron Ore_2 | QINGDAO | 11,371 | 89.14 | $498,830 | $16,906 | ❌ NO |
| ANN BELL | CSN_Iron Ore_2 | TIANJIN | 11,373 | 89.14 | $498,530 | $16,902 | ❌ NO |
| ANN BELL | EGA_Bauxite_0 | **QINGDAO** | 11,124 | 91.41 | $888,463 | $21,029 | ✅ YES |
| ANN BELL | EGA_Bauxite_0 | TIANJIN | 11,924 | 94.18 | $777,459 | $19,564 | ❌ NO |

---

## ⚠️ Market Cargo Analysis

### Why No Market Cargo Assigned to Unused Cargill Vessels?

The 3 unused Cargill vessels (GOLDEN ASCENT, OCEAN HORIZON, PACIFIC GLORY) were not assigned market cargo because:

1. ❌ Market cargo profit/loss was **≤ 0** for available routes
2. ❌ Cargo laycan windows didn't align with vessel ETAs
3. ❌ Distance/port matching issues

**Result:** $42,831.25 penalty for idle vessels (1 day × daily hire × (1 - ADCOMS) per vessel)

---

## 📁 Output Files Generated

| File | Description |
|------|-------------|
| `cargill_vs_committed.csv` | Cargill vessels × Committed cargoes matrix |
| `cargill_vs_marketcargo.csv` | Cargill vessels × Market cargoes matrix |
| `marketvs_committed.csv` | Market vessels × Committed cargoes matrix |
| `alt_port_comparison.csv` | Alternative discharge port analysis |

---

## 📝 Notes

- Port names are normalized (accents removed, aliases applied)
- DWT values auto-converted from thousands if < 1000
- Hire rates auto-converted from thousands if < 1000
- **Bunker prices are dynamically selected based on vessel position** (nearest hub)
- Remaining fuel onboard is valued at previous month's prices

---

*Report generated by Cargill Datathon Freight Calculator Chatbot*
