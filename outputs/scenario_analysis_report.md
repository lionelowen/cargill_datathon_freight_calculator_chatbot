# 📊 Scenario Analysis Report: China Port Delay Sensitivity

**Generated:** January 31, 2026  
**Project:** Cargill Datathon Freight Calculator

---

## 📋 Executive Summary

| Metric | Value |
|--------|-------|
| **Scenario Tested** | Additional China Port Delay Days |
| **Delay Range Tested** | 0 - 50 days |
| **Cost Per Delay Day** | **$57,795.23** |
| **Assignment Change Point** | **None (robust to 50+ days)** |
| **Breakeven Point (P/L turns negative)** | ~28-30 days |

### Key Finding

🟢 **The current vessel-cargo assignment remains OPTIMAL even with 50+ days of China port delay.**

The recommendation is **robust** to port congestion scenarios - the same vessels and discharge ports remain optimal regardless of delay duration.

---

## 🚢 Baseline Optimal Assignment

| Cargo | Vessel | Type | Discharge Port | P/L (0 days) |
|-------|--------|------|----------------|--------------|
| COMMITTED_BHP_Iron Ore_1 | PACIFIC VANGUARD | MARKET | LIANYUNGANG | -$305,218.02 |
| COMMITTED_CSN_Iron Ore_2 | ANN BELL | CARGILL | RIZHAO | $588,088.48 |
| COMMITTED_EGA_Bauxite_0 | ZENITH GLORY | MARKET | QINGDAO | $1,366,037.47 |

**Total Portfolio P/L (Baseline):** $1,648,907.93

---

## 📈 Sensitivity Analysis Results

### P/L Impact by Delay Days

| Delay Days | Total P/L | Change from Baseline | Assignment Changed? |
|------------|-----------|---------------------|---------------------|
| 0 (baseline) | $1,648,907.93 | - | - |
| 1 | $1,591,112.70 | -$57,795.23 | ❌ No |
| 2 | $1,533,317.48 | -$115,590.45 | ❌ No |
| 3 | $1,475,522.25 | -$173,385.67 | ❌ No |
| 4 | $1,417,727.03 | -$231,180.90 | ❌ No |
| 5 | $1,359,931.80 | -$288,976.12 | ❌ No |
| 6 | $1,302,136.58 | -$346,771.35 | ❌ No |
| 7 | $1,244,341.35 | -$404,566.58 | ❌ No |
| 8 | $1,186,546.13 | -$462,361.80 | ❌ No |
| 9 | $1,128,750.90 | -$520,157.03 | ❌ No |
| 10 | $1,070,955.68 | -$577,952.25 | ❌ No |
| 12 | $955,365.23 | -$693,542.70 | ❌ No |
| 14 | $839,774.78 | -$809,133.15 | ❌ No |
| 16 | $724,184.33 | -$924,723.60 | ❌ No |
| 18 | $608,593.88 | -$1,040,314.05 | ❌ No |
| 20 | $493,003.43 | -$1,155,904.50 | ❌ No |
| 25 | $204,027.30 | -$1,444,880.63 | ❌ No |
| **30** | **-$84,948.82** | **-$1,733,856.75** | ❌ No |
| 40 | -$662,901.07 | -$2,311,809.00 | ❌ No |
| 50 | -$1,240,853.32 | -$2,889,761.25 | ❌ No |

---

## 💰 Cost Impact Analysis

### Per-Day Delay Cost Breakdown

| Cost Component | Approximate Daily Cost |
|----------------|------------------------|
| **Vessel Hire** | ~$45,000 - $50,000/day |
| **Port Fuel (IFO)** | ~$3,000 - $5,000/day |
| **Port Fuel (MGO)** | ~$500 - $1,000/day |
| **Total per Delay Day** | **~$57,795/day** |

### Cumulative Impact

| Delay Scenario | Total Extra Cost | % of Baseline P/L |
|----------------|------------------|-------------------|
| 1 week (7 days) | $404,567 | 24.5% |
| 2 weeks (14 days) | $809,133 | 49.1% |
| 3 weeks (21 days) | $1,213,700 | 73.6% |
| 1 month (30 days) | $1,733,857 | **105.2%** (turns negative) |

---

## 🔍 Why Doesn't the Assignment Change?

### All Discharge Ports are in China

| Cargo | Discharge Port | Country |
|-------|---------------|---------|
| BHP Iron Ore | LIANYUNGANG | 🇨🇳 China |
| CSN Iron Ore | RIZHAO | 🇨🇳 China |
| EGA Bauxite | QINGDAO | 🇨🇳 China |

Since **all committed cargoes discharge to China**, the additional delay affects **all options equally**. The relative profitability ranking remains unchanged:

- Best option at 0 days delay → Still best at 50 days delay
- Second best at 0 days → Still second best at 50 days
- And so on...

### Conditions for Assignment Change

The assignment would change if:
1. ✅ Some cargoes had **non-China discharge alternatives** (e.g., Korea, Vietnam, Malaysia)
2. ✅ Different **port-specific delays** applied (e.g., Qingdao +10 days, Rizhao +2 days)
3. ✅ Delay costs varied by **vessel type** (e.g., market vessels have different hire rates)

---

## 🎯 Scenario Definition

### What This Analysis Tests

```
Scenario: Additional delay days at China discharge ports

Normal Discharge Time = (Cargo Qty ÷ Discharge Rate) + Turn Time
Scenario Discharge Time = Normal + Additional Delay Days

Affected Ports: QINGDAO, SHANGHAI, TIANJIN, RIZHAO, CAOFEIDIAN,
                JINGTANG, FANGCHENG, XIAMEN, NINGBO, LIANYUNGANG,
                GUANGZHOU, DALIAN, YANTAI
```

### Real-World Causes of Port Delays

| Cause | Typical Duration |
|-------|------------------|
| Port congestion (peak season) | 3-7 days |
| Typhoon/weather delays | 2-5 days |
| Berth availability | 1-3 days |
| Customs/inspection | 1-2 days |
| COVID-related restrictions | 7-14+ days |
| Equipment breakdown | 1-3 days |

---

## 📝 Recommendations

### 1. Current Assignment is Robust ✅
The optimal vessel-cargo assignment can withstand significant port delays without needing to be changed. This provides **operational flexibility**.

### 2. Monitor Delay Costs 💰
Each day of China port delay costs approximately **$57,795**. Factor this into voyage estimates when port congestion is expected.

### 3. Breakeven Awareness ⚠️
At approximately **28-30 days** of cumulative delay, the portfolio P/L turns **negative**. This is the critical threshold for voyage viability.

### 4. Consider Contingency Buffer 📊
For voyage planning, consider adding a buffer:
- **Conservative:** 5 days (+$289K cost)
- **Moderate:** 10 days (+$578K cost)
- **Aggressive:** 3 days (+$173K cost)

---

## 📁 Related Files

| File | Description |
|------|-------------|
| `scenario_analysis.py` | Python script for this analysis |
| `freight_calculator.py` | Core freight calculation engine |
| `freight_calculator_analysis_report.md` | Detailed freight calculator report |

---

*Report generated by Cargill Datathon Freight Calculator Chatbot*
