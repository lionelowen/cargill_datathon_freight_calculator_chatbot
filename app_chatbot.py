# app_chatbot.py
# ============================================================
# Streamlit Chatbot with Gemini API Integration
# For Cargill Freight Calculator
# ============================================================

import streamlit as st
import google.generativeai as genai
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.freight_calculator import (
    load_distance_lookup, load_bunker_prices, load_all_bunker_prices,
    load_ffa, load_vessels_from_excel, load_cargoes_from_excel,
    build_profit_table, optimal_committed_assignment,
    PricesAndFees,
    BUNKER_XLSX, CARGILL_VESSEL_XLSX, MARKET_VESSEL_XLSX,
    COMMITTED_CARGO_XLSX, MARKET_CARGO_XLSX, DIST_XLSX, FFA_REPORT_XLSX
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Freight Calculator Chatbot",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Freight Calculator Chatbot")
st.caption("Powered by Gemini AI + Cargill Freight Calculator")

# ============================================================
# Sidebar - API Key & Settings
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Enter your Gemini API Key to enable AI chat")
    
    st.divider()
    
    # Load data button
    if st.button("🔄 Load/Refresh Freight Data"):
        st.session_state.data_loaded = False
    
    st.divider()
    st.markdown("### Quick Actions")
    
    if st.button("📊 Show Optimal Plan"):
        st.session_state.show_optimal = True
    
    if st.button("⛽ Fuel Price Sensitivity"):
        st.session_state.show_fuel = True
    
    if st.button("🕐 Port Delay Analysis"):
        st.session_state.show_delay = True

# ============================================================
# Load Freight Calculator Data
# ============================================================
@st.cache_data
def load_freight_data():
    """Load all freight calculator data"""
    try:
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
        market_cargoes = load_cargoes_from_excel(MARKET_CARGO_XLSX, tag="MARKET", ffa=ffa)

        # Build profit tables
        df_cv_cc = build_profit_table(
            cargill_vessels, committed, dist_lut, pf, 
            bunker_days=1.0, enforce_laycan=True, 
            bunker_prices_by_location=all_bunker_prices
        )
        df_mv_cc = build_profit_table(
            market_vessels, committed, dist_lut, pf, 
            bunker_days=1.0, enforce_laycan=True, 
            bunker_prices_by_location=all_bunker_prices
        )

        # Get optimal plan
        commit_plan, commit_total = optimal_committed_assignment(df_cv_cc, df_mv_cc)
        
        return {
            "cargill_vessels": cargill_vessels,
            "market_vessels": market_vessels,
            "committed": committed,
            "market_cargoes": market_cargoes,
            "commit_plan": commit_plan,
            "commit_total": commit_total,
            "df_cv_cc": df_cv_cc,
            "df_mv_cc": df_mv_cc,
            "bunker_prices": all_bunker_prices,
            "vlsfo_price": vlsfo_price,
            "mgo_price": mgo_price,
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

with st.spinner("Loading freight calculator data..."):
    data = load_freight_data()
    if data:
        st.session_state.data_loaded = True
        st.session_state.freight_data = data

# ============================================================
# Build Context for Gemini
# ============================================================
def build_freight_context():
    """Build context string for Gemini about current freight data"""
    if not st.session_state.get("data_loaded") or not st.session_state.get("freight_data"):
        return "No freight data loaded."
    
    data = st.session_state.freight_data
    plan = data["commit_plan"]
    
    context = f"""
You are a freight calculator assistant for Cargill shipping operations.

CURRENT OPTIMAL ASSIGNMENT:
Total Portfolio P/L: ${data['commit_total']:,.2f}

Vessel-Cargo Assignments:
"""
    for _, row in plan.iterrows():
        context += f"- {row['base_cargo_id']}: {row['vessel']} ({row['vessel_type']}) → {row['discharge_port']}, P/L: ${row['profit_loss']:,.2f}, TCE: ${row['tce']:,.2f}/day, Duration: {row['total_duration']:.1f} days\n"
    
    context += f"""
FUEL PRICES (Current):
- VLSFO: ${data['vlsfo_price']:.2f}/MT
- MGO: ${data['mgo_price']:.2f}/MT

AVAILABLE VESSELS:
Cargill: {', '.join([v.name for v in data['cargill_vessels']])}
Market: {', '.join([v.name for v in data['market_vessels']])}

COMMITTED CARGOES:
{', '.join([c.name for c in data['committed']])}

KEY INSIGHTS:
- The optimal plan uses location-based bunker prices (9 hubs)
- Each 10% fuel increase reduces P/L by ~$428,675
- Breakeven fuel price increase: ~31%
- China port delays cost ~$57,795 per day

When answering questions:
1. Be specific with numbers and data
2. Explain the rationale behind recommendations
3. Reference TCE (Time Charter Equivalent) when comparing options
4. Consider fuel costs, hire rates, and voyage duration
"""
    return context

# ============================================================
# Chat Interface
# ============================================================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Gemini chat
if "gemini_chat" not in st.session_state and api_key:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state.gemini_chat = model.start_chat(history=[])
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")

# Display data summary
if st.session_state.get("data_loaded"):
    data = st.session_state.freight_data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio P/L", f"${data['commit_total']:,.0f}")
    with col2:
        st.metric("VLSFO Price", f"${data['vlsfo_price']:.0f}/MT")
    with col3:
        st.metric("Committed Cargoes", len(data['committed']))

# Show optimal plan if requested
if st.session_state.get("show_optimal"):
    st.subheader("📊 Optimal Assignment Plan")
    if st.session_state.get("data_loaded"):
        st.dataframe(
            st.session_state.freight_data["commit_plan"][
                ["base_cargo_id", "vessel", "vessel_type", "discharge_port", "profit_loss", "tce", "total_duration"]
            ],
            use_container_width=True
        )
    st.session_state.show_optimal = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about freight calculations, routes, or scenarios..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not api_key:
            response = "⚠️ Please enter your Gemini API key in the sidebar to enable AI chat."
            st.warning(response)
        elif not st.session_state.get("data_loaded"):
            response = "⚠️ Freight data not loaded. Please check the data files."
            st.warning(response)
        else:
            try:
                # Build context-enhanced prompt
                context = build_freight_context()
                enhanced_prompt = f"{context}\n\nUser Question: {prompt}"
                
                # Get response from Gemini
                with st.spinner("Thinking..."):
                    gemini_response = st.session_state.gemini_chat.send_message(enhanced_prompt)
                    response = gemini_response.text
                
                st.markdown(response)
                
            except Exception as e:
                response = f"❌ Error: {e}"
                st.error(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================
# Example Questions
# ============================================================
st.divider()
st.markdown("### 💡 Example Questions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - What is the current optimal vessel assignment?
    - Why was ANN BELL assigned to RIZHAO?
    - What is the TCE for each cargo?
    - How does fuel price affect profitability?
    """)

with col2:
    st.markdown("""
    - What if fuel prices increase by 20%?
    - Which vessel has the highest TCE?
    - Explain the port delay cost impact
    - Compare RIZHAO vs QINGDAO for iron ore
    """)

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("Built with Streamlit + Gemini AI | Cargill Datathon Freight Calculator")
