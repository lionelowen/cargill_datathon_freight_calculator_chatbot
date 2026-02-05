import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Page config
st.set_page_config(page_title="Freight Calculator Chatbot", page_icon="🚢", layout="wide")
st.title("🚢 The Musketeers - Freight Calculator Bot")

# Load data
@st.cache_data
def load_optimal_assignments():
    """Load the optimal assignments data"""
    base_dir = Path(__file__).parent
    csv_path = base_dir / "outputs" / "optimal_assignments.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_all_outputs():
    """Load all output files"""
    base_dir = Path(__file__).parent
    outputs = {}
    
    files = {
        'optimal_assignments': 'optimal_assignments.csv',
        'cargill_vs_committed': 'cargill_vs_committed.csv',
        'cargill_vs_marketcargo': 'cargill_vs_marketcargo.csv',
        'market_vs_committed': 'marketvs_committed.csv',
        'alt_port_comparison': 'alt_port_comparison.csv'
    }
    
    for key, filename in files.items():
        path = base_dir / "outputs" / filename
        if path.exists():
            outputs[key] = pd.read_csv(path)
    
    return outputs

# Load data
df_optimal = load_optimal_assignments()
all_outputs = load_all_outputs()

# Calculate summary metrics
def get_summary_metrics(df):
    if df is None or len(df) == 0:
        return {}
    
    # Fill NaN vessel_type with 'CARGILL' for market cargo assignments
    df_copy = df.copy()
    df_copy['vessel_type'] = df_copy['vessel_type'].fillna('CARGILL')
    
    committed = df_copy[df_copy['assignment_type'] == 'Committed Cargo']
    market = df_copy[df_copy['assignment_type'] == 'Market Cargo']
    
    return {
        'total_pl': df_copy['profit_loss'].sum(),
        'total_revenue': df_copy['revenue'].sum(),
        'total_bunker': df_copy['bunker_expense'].sum(),
        'total_hire': df_copy['hire_net'].sum(),
        'avg_tce': df_copy['tce'].mean(),
        'num_assignments': len(df_copy),
        'committed_count': len(committed),
        'market_count': len(market),
        'committed_pl': committed['profit_loss'].sum() if len(committed) > 0 else 0,
        'market_pl': market['profit_loss'].sum() if len(market) > 0 else 0,
        'cargill_vessels': df_copy[df_copy['vessel_type'] == 'CARGILL']['vessel'].nunique(),
        'market_vessels': df_copy[df_copy['vessel_type'] == 'MARKET']['vessel'].nunique(),
    }

metrics = get_summary_metrics(df_optimal)

# Sidebar with summary
with st.sidebar:
    st.header("📊 Portfolio Summary")
    
    if metrics:
        st.metric("Total P/L", f"${metrics['total_pl']:,.2f}")
        st.metric("Avg TCE", f"${metrics['avg_tce']:,.2f}/day")
        
        st.divider()
        st.subheader("Assignments")
        st.write(f"📦 Committed Cargo: {metrics['committed_count']} (P/L: ${metrics['committed_pl']:,.2f})")
        st.write(f"📦 Market Cargo: {metrics['market_count']} (P/L: ${metrics['market_pl']:,.2f})")
        
        st.divider()
        st.subheader("Vessels Used")
        st.write(f"🔵 Cargill Vessels: {metrics['cargill_vessels']}")
        st.write(f"🟢 Market Vessels: {metrics['market_vessels']}")
        
        st.divider()
        st.subheader("Financial Breakdown")
        st.write(f"Revenue: ${metrics['total_revenue']:,.2f}")
        st.write(f"Hire Cost: ${metrics['total_hire']:,.2f}")
        st.write(f"Bunker Cost: ${metrics['total_bunker']:,.2f}")
    else:
        st.warning("No optimal assignments loaded")

# Main content - Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📋 Assignments", "📈 Analysis", "🔄 Port Delay Scenario", "⛽ Fuel Price Sensitivity"])

# Tab 1: Chat Interface
with tab1:
    # Prepare context for Gemini
    def get_data_context():
        """Create a context string with all the data for Gemini"""
        context = """You are a freight calculator assistant for Cargill's shipping operations. 
You have access to the following optimal cargo assignments data:

"""
        if df_optimal is not None:
            context += "OPTIMAL ASSIGNMENTS:\n"
            for idx, row in df_optimal.iterrows():
                context += f"""
Assignment {idx+1}:
- Category: {row['assignment_category']}
- Vessel: {row['vessel']} ({row['vessel_type']})
- Cargo: {row['cargo']}
- Route: {row['from_pos']} → {row['load_port']} → {row['discharge_port']}
- Distances: Ballast {row['ballast_nm']:.0f}nm, Laden {row['laden_nm']:.0f}nm
- Duration: {row['total_duration']:.1f} days
- Revenue: ${row['revenue']:,.2f}
- Bunker Cost: ${row['bunker_expense']:,.2f} (at {row['bunker_location']})
- Profit/Loss: ${row['profit_loss']:,.2f}
- TCE: ${row['tce']:,.2f}/day
"""
            context += f"""
PORTFOLIO SUMMARY:
- Total P/L: ${metrics['total_pl']:,.2f}
- Average TCE: ${metrics['avg_tce']:,.2f}/day
- Total Revenue: ${metrics['total_revenue']:,.2f}
- Total Bunker Cost: ${metrics['total_bunker']:,.2f}
- Cargill Vessels Used: {metrics['cargill_vessels']}
- Market Vessels Used: {metrics['market_vessels']}
"""
        return context

    genai.configure(api_key=st.secrets["API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask about freight assignments, routes, costs..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Build history for Gemini
            history = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                for m in st.session_state.messages[:-1]
            ]
            
            # Add data context to the first message or system instruction
            data_context = get_data_context()
            enhanced_prompt = f"{data_context}\n\nUser Question: {prompt}\n\nProvide a helpful, concise answer based on the data above."
            
            chat = model.start_chat(history=history)
            
            # Use stream=True for a better UX
            response_stream = chat.send_message(enhanced_prompt, stream=True)
            
            # Stream the response to the UI
            full_response = st.write_stream(res.text for res in response_stream)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Tab 2: Assignments Table
with tab2:
    st.header("📋 Optimal Cargo Assignments")
    
    if df_optimal is not None:
        # Fix NaN vessel_type
        df_display = df_optimal.copy()
        df_display['vessel_type'] = df_display['vessel_type'].fillna('CARGILL')
        
        # Separate committed and market cargo
        committed_df = df_display[df_display['assignment_type'] == 'Committed Cargo']
        market_df = df_display[df_display['assignment_type'] == 'Market Cargo']
        
        # ===== COMMITTED CARGO SECTION =====
        st.subheader("📦 Committed Cargo Assignments")
        if len(committed_df) > 0:
            committed_pl = committed_df['profit_loss'].sum()
            st.write(f"**Total Committed Cargo P/L: ${committed_pl:,.2f}**")
            
            for idx, row in committed_df.iterrows():
                pl_color = "🟢" if row['profit_loss'] >= 0 else "🔴"
                vessel_icon = "🔵" if row['vessel_type'] == 'CARGILL' else "🟢"
                
                with st.expander(f"{vessel_icon} {row['vessel']} → {row['cargo'].replace('COMMITTED_', '')} | {pl_color} P/L: ${row['profit_loss']:,.2f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📦 Vessel Info**")
                        st.write(f"Vessel: {row['vessel']}")
                        st.write(f"Type: {row['vessel_type']}")
                        st.write(f"Start Position: {row['from_pos']}")
                    
                    with col2:
                        st.markdown("**🚢 Route Info**")
                        st.write(f"Load Port: {row['load_port']}")
                        st.write(f"Discharge Port: {row['discharge_port']}")
                        st.write(f"Ballast: {row['ballast_nm']:,.0f} nm")
                        st.write(f"Laden: {row['laden_nm']:,.0f} nm")
                        st.write(f"Duration: {row['total_duration']:.1f} days")
                    
                    with col3:
                        st.markdown("**💰 Financial**")
                        st.write(f"Loaded Qty: {row['loaded_qty']:,.0f} MT")
                        st.write(f"Revenue: ${row['revenue']:,.2f}")
                        st.write(f"Hire Cost: ${row['hire_net']:,.2f}")
                        st.write(f"Bunker: ${row['bunker_expense']:,.2f} ({row['bunker_location']})")
                        st.write(f"**P/L: ${row['profit_loss']:,.2f}**")
                        st.write(f"**TCE: ${row['tce']:,.2f}/day**")
        else:
            st.info("No committed cargo assignments")
        
        st.divider()
        
        # ===== MARKET CARGO SECTION =====
        st.subheader("📦 Market Cargo Assignments (Unused Cargill Vessels)")
        if len(market_df) > 0:
            market_pl = market_df['profit_loss'].sum()
            st.write(f"**Total Market Cargo P/L: ${market_pl:,.2f}**")
            
            for idx, row in market_df.iterrows():
                pl_color = "🟢" if row['profit_loss'] >= 0 else "🔴"
                
                with st.expander(f"🔵 {row['vessel']} → {row['cargo'].replace('MARKET_', '')} | {pl_color} P/L: ${row['profit_loss']:,.2f}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📦 Vessel Info**")
                        st.write(f"Vessel: {row['vessel']}")
                        st.write(f"Type: CARGILL (carrying market cargo)")
                        st.write(f"Start Position: {row['from_pos']}")
                    
                    with col2:
                        st.markdown("**🚢 Route Info**")
                        st.write(f"Load Port: {row['load_port']}")
                        st.write(f"Discharge Port: {row['discharge_port']}")
                        st.write(f"Ballast: {row['ballast_nm']:,.0f} nm")
                        st.write(f"Laden: {row['laden_nm']:,.0f} nm")
                        st.write(f"Duration: {row['total_duration']:.1f} days")
                    
                    with col3:
                        st.markdown("**💰 Financial**")
                        st.write(f"Loaded Qty: {row['loaded_qty']:,.0f} MT")
                        st.write(f"Revenue: ${row['revenue']:,.2f}")
                        st.write(f"Hire Cost: ${row['hire_net']:,.2f}")
                        st.write(f"Bunker: ${row['bunker_expense']:,.2f} ({row['bunker_location']})")
                        st.write(f"**P/L: ${row['profit_loss']:,.2f}**")
                        st.write(f"**TCE: ${row['tce']:,.2f}/day**")
        else:
            st.info("No market cargo assignments for unused Cargill vessels")
        
        st.divider()
        
        # ===== SUMMARY TABLE =====
        st.subheader("📊 All Assignments Summary")
        display_cols = ['assignment_category', 'vessel', 'cargo', 'load_port', 'discharge_port', 
                       'total_duration', 'profit_loss', 'tce', 'bunker_location']
        available_cols = [c for c in display_cols if c in df_display.columns]
        
        st.dataframe(
            df_display[available_cols].style.format({
                'profit_loss': '${:,.2f}',
                'tce': '${:,.2f}',
                'total_duration': '{:.1f} days'
            }),
            use_container_width=True
        )
    else:
        st.warning("No optimal assignments found. Run the freight calculator first.")

# Tab 3: Analysis
with tab3:
    st.header("📈 Portfolio Analysis")
    
    if df_optimal is not None:
        # Fix NaN vessel_type
        df_analysis = df_optimal.copy()
        df_analysis['vessel_type'] = df_analysis['vessel_type'].fillna('CARGILL')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("P/L by Assignment")
            chart_data = df_analysis[['cargo', 'profit_loss']].copy()
            chart_data['cargo'] = chart_data['cargo'].str.replace('COMMITTED_', '').str.replace('MARKET_', '')
            st.bar_chart(chart_data.set_index('cargo'))
        
        with col2:
            st.subheader("TCE by Vessel")
            tce_data = df_analysis[['vessel', 'tce']].copy()
            st.bar_chart(tce_data.set_index('vessel'))
        
        st.divider()
        
        # Cost breakdown
        st.subheader("💵 Cost Breakdown")
        cost_data = {
            'Category': ['Revenue', 'Hire Cost', 'Bunker Cost', 'Misc Cost'],
            'Amount': [
                df_analysis['revenue'].sum(),
                df_analysis['hire_net'].sum(),
                df_analysis['bunker_expense'].sum(),
                df_analysis['misc_expense'].sum()
            ]
        }
        cost_df = pd.DataFrame(cost_data)
        st.bar_chart(cost_df.set_index('Category'))
        
        # Assignment type breakdown
        st.subheader("📦 Assignment Type Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            type_counts = df_analysis['assignment_type'].value_counts()
            st.bar_chart(type_counts)
        
        with col2:
            type_pl = df_analysis.groupby('assignment_type')['profit_loss'].sum()
            st.bar_chart(type_pl)
    else:
        st.warning("No data available for analysis.")

# Tab 4: Scenario Analysis
with tab4:
    st.header("🔄 Scenario Analysis: China Port Delay Sensitivity")
    st.markdown("""
    This analysis shows how **China port congestion/delays** affect the optimal vessel-cargo assignment.
    Use the slider to simulate different delay scenarios and see the impact on P/L.
    """)
    
    # Import scenario analysis functions
    try:
        from scenario_analysis import (
            build_profit_table_with_china_delay,
            is_china_port,
            CHINA_PORTS
        )
        from freight_calculator import (
            load_distance_lookup, load_bunker_prices, load_all_bunker_prices,
            load_ffa, load_vessels_from_excel, load_cargoes_from_excel,
            optimal_committed_assignment, assign_market_cargo_to_unused_cargill,
            unused_committed_vessel_penalty, PricesAndFees,
            BUNKER_XLSX, CARGILL_VESSEL_XLSX, MARKET_VESSEL_XLSX,
            COMMITTED_CARGO_XLSX, MARKET_CARGO_XLSX, DIST_XLSX, FFA_REPORT_XLSX
        )
        
        @st.cache_data
        def load_scenario_data():
            """Load all data needed for scenario analysis"""
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
            
            # Calculate average Cargill hire rate
            cargill_vessels_temp = load_vessels_from_excel(
                CARGILL_VESSEL_XLSX, speed_mode="economical", is_market=False, ffa=ffa
            )
            avg_cargill_hire = sum(v.daily_hire for v in cargill_vessels_temp) / len(cargill_vessels_temp)
            
            cargill_vessels = load_vessels_from_excel(
                CARGILL_VESSEL_XLSX, speed_mode="economical", is_market=False, ffa=ffa
            )
            market_vessels = load_vessels_from_excel(
                MARKET_VESSEL_XLSX, speed_mode="economical", is_market=True, ffa=ffa,
                market_hire_multiplier=1.00, avg_cargill_hire=avg_cargill_hire
            )
            
            committed = load_cargoes_from_excel(COMMITTED_CARGO_XLSX, tag="COMMITTED")
            market_cargo = load_cargoes_from_excel(MARKET_CARGO_XLSX, tag="MARKET", ffa=ffa)
            
            return dist_lut, all_bunker_prices, pf, cargill_vessels, market_vessels, committed, market_cargo, ffa
        
        @st.cache_data
        def run_delay_scenario(_dist_lut, _all_bunker_prices, _pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, delay_days, enforce_laycan):
            """Run scenario with specific delay days - returns both committed and market cargo assignments"""
            # Build profit tables for committed cargo
            df_cv_cc = build_profit_table_with_china_delay(
                _cargill_vessels, _committed, _dist_lut, _pf, _all_bunker_prices,
                china_delay_days=delay_days,
                enforce_laycan=enforce_laycan
            )
            df_mv_cc = build_profit_table_with_china_delay(
                _market_vessels, _committed, _dist_lut, _pf, _all_bunker_prices,
                china_delay_days=delay_days,
                enforce_laycan=enforce_laycan
            )
            
            # Optimal committed assignment
            committed_plan, committed_total = optimal_committed_assignment(df_cv_cc, df_mv_cc, _cargill_vessels)
            
            # Get used Cargill vessels
            used_cargill = set(committed_plan[committed_plan['vessel_type'] == 'CARGILL']['vessel'].tolist())
            
            # Build profit table for market cargo with Cargill vessels
            df_cv_mc = build_profit_table_with_china_delay(
                _cargill_vessels, _market_cargo, _dist_lut, _pf, _all_bunker_prices,
                china_delay_days=delay_days,
                enforce_laycan=enforce_laycan
            )
            
            # Assign market cargo to unused Cargill vessels
            market_plan, market_total = assign_market_cargo_to_unused_cargill(df_cv_mc, used_cargill)
            
            # Calculate penalty for truly unused vessels
            if len(market_plan) > 0:
                used_for_market = set(market_plan["vessel"].tolist())
                all_used_cargill = used_cargill | used_for_market
            else:
                all_used_cargill = used_cargill
            
            penalty = unused_committed_vessel_penalty(_cargill_vessels, all_used_cargill, idle_days=1.0)
            
            # Full portfolio total
            portfolio_total = committed_total + market_total - penalty
            
            return committed_plan, market_plan, committed_total, market_total, penalty, portfolio_total
        
        # Load data
        with st.spinner("Loading scenario analysis data..."):
            dist_lut, all_bunker_prices, pf, cargill_vessels, market_vessels, committed, market_cargo, ffa = load_scenario_data()
        
        # Show China ports info
        with st.expander("ℹ️ China Ports Affected by Delay"):
            st.write("The following discharge ports are considered China ports:")
            st.write(", ".join(sorted(CHINA_PORTS)))
        
        # Laycan toggle
        enforce_laycan_delay = st.checkbox(
            "Enforce Laycan Constraints",
            value=False,
            help="If checked, only consider vessel-cargo combinations that meet the laycan (arrival window) requirements",
            key="enforce_laycan_delay"
        )
        
        # Delay slider
        st.subheader("⏱️ Delay Simulation")
        delay_days = st.slider(
            "Additional delay days at China ports",
            min_value=0,
            max_value=30,
            value=0,
            step=1,
            help="Simulate port congestion by adding extra waiting days at China discharge ports"
        )
        
        # Run baseline (0 days)
        baseline_committed, baseline_market, baseline_committed_total, baseline_market_total, baseline_penalty, baseline_total = run_delay_scenario(
            dist_lut, all_bunker_prices, pf, cargill_vessels, market_vessels, committed, market_cargo, 0, enforce_laycan_delay
        )
        baseline_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in baseline_committed.iterrows()}
        baseline_market_assignment = {}
        if len(baseline_market) > 0:
            baseline_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in baseline_market.iterrows()}
        
        # Run current scenario
        current_committed, current_market, current_committed_total, current_market_total, current_penalty, current_total = run_delay_scenario(
            dist_lut, all_bunker_prices, pf, cargill_vessels, market_vessels, committed, market_cargo, delay_days, enforce_laycan_delay
        )
        current_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in current_committed.iterrows()}
        current_market_assignment = {}
        if len(current_market) > 0:
            current_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in current_market.iterrows()}
        
        # Display results - Summary metrics
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Baseline Total P/L", 
                f"${baseline_total:,.2f}"
            )
        
        with col2:
            change = current_total - baseline_total
            st.metric(
                f"Current Total P/L ({delay_days}d)",
                f"${current_total:,.2f}",
                delta=f"${change:,.2f}"
            )
        
        with col3:
            committed_changed = baseline_committed_assignment != current_committed_assignment
            market_changed = baseline_market_assignment != current_market_assignment
            if committed_changed or market_changed:
                st.error("⚠️ Assignment CHANGED!")
            else:
                st.success("✅ Assignment unchanged")
        
        with col4:
            st.metric("Idle Penalty", f"${current_penalty:,.2f}")
        
        # Detailed breakdown
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Committed Cargo P/L", f"${current_committed_total:,.2f}", 
                     delta=f"${current_committed_total - baseline_committed_total:,.2f}")
        with col2:
            st.metric("Market Cargo P/L", f"${current_market_total:,.2f}",
                     delta=f"${current_market_total - baseline_market_total:,.2f}")
        with col3:
            st.metric("Penalty Change", f"${current_penalty:,.2f}",
                     delta=f"${current_penalty - baseline_penalty:,.2f}")
        
        st.divider()
        
        # Show assignment comparison - COMMITTED CARGO
        st.subheader("📦 Committed Cargo Assignment Comparison")
        
        comparison_data = []
        for cargo_id in baseline_committed_assignment.keys():
            base_vessel, base_port = baseline_committed_assignment[cargo_id]
            curr_vessel, curr_port = current_committed_assignment.get(cargo_id, ("N/A", "N/A"))
            
            changed = (base_vessel, base_port) != (curr_vessel, curr_port)
            
            # Get P/L for both scenarios
            base_pl = baseline_committed[baseline_committed['base_cargo_id'] == cargo_id]['profit_loss'].values[0]
            curr_pl = current_committed[current_committed['base_cargo_id'] == cargo_id]['profit_loss'].values[0]
            
            comparison_data.append({
                "Type": "COMMITTED",
                "Cargo": cargo_id.replace("COMMITTED_", ""),
                "Baseline Vessel": base_vessel,
                "Baseline Port": base_port,
                "Current Vessel": curr_vessel,
                "Current Port": curr_port,
                "Baseline P/L": base_pl,
                "Current P/L": curr_pl,
                "P/L Change": curr_pl - base_pl,
                "Changed": "🔴 YES" if changed else "✅ No"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(
            comparison_df.style.format({
                'Baseline P/L': '${:,.2f}',
                'Current P/L': '${:,.2f}',
                'P/L Change': '${:,.2f}'
            }),
            use_container_width=True
        )
        
        # Show assignment comparison - MARKET CARGO
        st.subheader("📦 Market Cargo Assignment Comparison (Unused Cargill Vessels)")
        
        if len(baseline_market_assignment) > 0 or len(current_market_assignment) > 0:
            market_comparison_data = []
            all_market_cargos = set(baseline_market_assignment.keys()) | set(current_market_assignment.keys())
            
            for cargo_id in all_market_cargos:
                base_vessel, base_port = baseline_market_assignment.get(cargo_id, ("N/A", "N/A"))
                curr_vessel, curr_port = current_market_assignment.get(cargo_id, ("N/A", "N/A"))
                
                changed = (base_vessel, base_port) != (curr_vessel, curr_port)
                
                # Get P/L for both scenarios
                base_pl = 0
                curr_pl = 0
                if len(baseline_market) > 0 and cargo_id in baseline_market['cargo'].values:
                    base_pl = baseline_market[baseline_market['cargo'] == cargo_id]['profit_loss'].values[0]
                if len(current_market) > 0 and cargo_id in current_market['cargo'].values:
                    curr_pl = current_market[current_market['cargo'] == cargo_id]['profit_loss'].values[0]
                
                market_comparison_data.append({
                    "Type": "MARKET",
                    "Cargo": cargo_id.replace("MARKET_", ""),
                    "Baseline Vessel": base_vessel,
                    "Baseline Port": base_port,
                    "Current Vessel": curr_vessel,
                    "Current Port": curr_port,
                    "Baseline P/L": base_pl,
                    "Current P/L": curr_pl,
                    "P/L Change": curr_pl - base_pl,
                    "Changed": "🔴 YES" if changed else "✅ No"
                })
            
            market_comparison_df = pd.DataFrame(market_comparison_data)
            st.dataframe(
                market_comparison_df.style.format({
                    'Baseline P/L': '${:,.2f}',
                    'Current P/L': '${:,.2f}',
                    'P/L Change': '${:,.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No market cargo assignments in either scenario")
        
        st.divider()
        
        # Sensitivity chart
        st.subheader("📊 Sensitivity Analysis")
        
        @st.cache_data
        def generate_sensitivity_data(_dist_lut, _all_bunker_prices, _pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, enforce_laycan):
            """Generate data for sensitivity chart - uses binary search to find breakeven quickly"""
            
            def get_assignment_key(delay):
                """Get assignment key for a given delay"""
                curr_committed, curr_market, _, _, _, total = run_delay_scenario(
                    _dist_lut, _all_bunker_prices, _pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, delay, enforce_laycan
                )
                curr_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                                  for _, row in curr_committed.iterrows()}
                curr_market_assignment = {}
                if len(curr_market) > 0:
                    curr_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                                  for _, row in curr_market.iterrows()}
                return (frozenset(curr_committed_assignment.items()), frozenset(curr_market_assignment.items())), total
            
            # Get baseline
            base_key, base_total = get_assignment_key(0)
            
            # Binary search to find breakeven point (where assignment changes)
            def find_breakeven(low, high, base_key):
                """Binary search to find exact breakeven point"""
                if high - low <= 1:
                    return high if get_assignment_key(high)[0] != base_key else None
                
                mid = (low + high) // 2
                mid_key, _ = get_assignment_key(mid)
                
                if mid_key != base_key:
                    return find_breakeven(low, mid, base_key)
                else:
                    return find_breakeven(mid, high, base_key)
            
            # Quick check: does assignment change at max delay (30)?
            max_key, max_total = get_assignment_key(30)
            breakeven = None
            if max_key != base_key:
                breakeven = find_breakeven(0, 30, base_key)
            
            # Generate minimal chart data: baseline, breakeven point (if found), and endpoint
            results = []
            results.append({"Delay Days": 0, "Total P/L": base_total, "Assignment Changed": False})
            
            if breakeven is not None:
                # Add point just before breakeven
                if breakeven > 1:
                    _, pre_total = get_assignment_key(breakeven - 1)
                    results.append({"Delay Days": breakeven - 1, "Total P/L": pre_total, "Assignment Changed": False})
                # Add breakeven point
                _, be_total = get_assignment_key(breakeven)
                results.append({"Delay Days": breakeven, "Total P/L": be_total, "Assignment Changed": True})
            
            # Add endpoint
            results.append({"Delay Days": 30, "Total P/L": max_total, "Assignment Changed": max_key != base_key})
            
            return pd.DataFrame(results), breakeven
        
        sensitivity_df, breakeven = generate_sensitivity_data(
            dist_lut, all_bunker_prices, pf, cargill_vessels, market_vessels, committed, market_cargo, enforce_laycan_delay
        )
        
        # Display chart
        st.line_chart(sensitivity_df.set_index("Delay Days")["Total P/L"])
        
        # Explanation of the linear relationship
        st.markdown("""
        ---
        **📖 Why is this a straight line?**
        
        The P/L decreases **linearly** with each additional delay day because:
        
        1. **Fixed Cost per Delay Day** = Daily Hire Rate + Port Fuel Consumption
           - Each day the vessel waits at a China port costs the same amount
           - Hire cost: ~$14,000/day per vessel
           - Port fuel: ~$300-500/day (MGO for generators, etc.)
        
        2. **Revenue is unchanged** - The freight rate ($/MT) is fixed regardless of delays
        
        3. **Formula:** `P/L = Baseline_P/L - (Cost_per_day × Delay_days)`
        
        The line would show a **step/discontinuity** only if the optimal assignment changes 
        (e.g., switching from a Cargill vessel to a Market vessel, or choosing a different discharge port).
        
        **Key Insight:** This linear relationship helps you quickly estimate the financial impact 
        of port congestion. For example, if delays are expected to be 5 days, multiply the 
        cost per day by 5 to get the total impact.
        """)
        
        st.divider()
        
        # Cost per delay day
        if len(sensitivity_df) >= 2:
            cost_per_day = (sensitivity_df.iloc[0]["Total P/L"] - sensitivity_df.iloc[1]["Total P/L"]) / 2
            st.info(f"💰 **Cost per delay day:** ~${cost_per_day:,.2f}")
        
        # Breakeven info
        if breakeven is not None:
            st.warning(f"⚠️ **Breakeven Point:** Assignment changes at **{breakeven} delay days**")
        else:
            st.success("🟢 Assignment remains optimal even with 30 days of China port delay")
        
    except ImportError as e:
        st.error(f"Could not load scenario analysis module: {e}")
        st.info("Make sure scenario_analysis.py is in the src folder.")

# Tab 5: Fuel Price Sensitivity
with tab5:
    st.header("⛽ Fuel Price Sensitivity Analysis")
    st.markdown("""
    This analysis shows how **fuel price changes** affect the optimal vessel-cargo assignment and portfolio P/L.
    Use the slider to simulate different fuel price scenarios.
    """)
    
    # Clear cache button
    if st.button("🔄 Clear Cache & Refresh Data", key="clear_fuel_cache"):
        st.cache_data.clear()
        st.rerun()
    
    try:
        from fuel_price_sensitivity import (
            build_profit_table_with_fuel_increase,
            scale_bunker_prices,
            get_all_feasible_assignments
        )
        from freight_calculator import (
            load_distance_lookup, load_bunker_prices, load_all_bunker_prices,
            load_ffa, load_vessels_from_excel, load_cargoes_from_excel,
            optimal_committed_assignment, assign_market_cargo_to_unused_cargill,
            unused_committed_vessel_penalty, PricesAndFees, Voyage, calc,
            BUNKER_XLSX, CARGILL_VESSEL_XLSX, MARKET_VESSEL_XLSX,
            COMMITTED_CARGO_XLSX, MARKET_CARGO_XLSX, DIST_XLSX, FFA_REPORT_XLSX
        )
        
        @st.cache_data
        def load_fuel_sensitivity_data():
            """Load all data needed for fuel sensitivity analysis"""
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
            
            # Calculate average Cargill hire rate
            cargill_vessels_temp = load_vessels_from_excel(
                CARGILL_VESSEL_XLSX, speed_mode="economical", is_market=False, ffa=ffa
            )
            avg_cargill_hire = sum(v.daily_hire for v in cargill_vessels_temp) / len(cargill_vessels_temp)
            
            cargill_vessels = load_vessels_from_excel(
                CARGILL_VESSEL_XLSX, speed_mode="economical", is_market=False, ffa=ffa
            )
            market_vessels = load_vessels_from_excel(
                MARKET_VESSEL_XLSX, speed_mode="economical", is_market=True, ffa=ffa,
                market_hire_multiplier=1.00, avg_cargill_hire=avg_cargill_hire
            )
            
            committed = load_cargoes_from_excel(COMMITTED_CARGO_XLSX, tag="COMMITTED")
            market_cargo = load_cargoes_from_excel(MARKET_CARGO_XLSX, tag="MARKET", ffa=ffa)
            
            return dist_lut, base_bunker_prices, base_pf, vlsfo_price, mgo_price, cargill_vessels, market_vessels, committed, market_cargo, ffa
        
        @st.cache_data
        def run_fuel_scenario(_dist_lut, _base_bunker_prices, _base_pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, increase_pct, enforce_laycan):
            """Run scenario with specific fuel price increase - returns both committed and market cargo assignments"""
            # Build profit tables for committed cargo
            df_cv_cc = build_profit_table_with_fuel_increase(
                _cargill_vessels, _committed, _dist_lut, _base_pf, _base_bunker_prices,
                fuel_increase_pct=increase_pct / 100.0,
                enforce_laycan=enforce_laycan
            )
            df_mv_cc = build_profit_table_with_fuel_increase(
                _market_vessels, _committed, _dist_lut, _base_pf, _base_bunker_prices,
                fuel_increase_pct=increase_pct / 100.0,
                enforce_laycan=enforce_laycan
            )
            
            # Optimal committed assignment
            committed_plan, committed_total = optimal_committed_assignment(df_cv_cc, df_mv_cc, _cargill_vessels)
            
            # Get used Cargill vessels
            used_cargill = set(committed_plan[committed_plan['vessel_type'] == 'CARGILL']['vessel'].tolist())
            
            # Build profit table for market cargo with Cargill vessels
            df_cv_mc = build_profit_table_with_fuel_increase(
                _cargill_vessels, _market_cargo, _dist_lut, _base_pf, _base_bunker_prices,
                fuel_increase_pct=increase_pct / 100.0,
                enforce_laycan=enforce_laycan
            )
            
            # Assign market cargo to unused Cargill vessels
            market_plan, market_total = assign_market_cargo_to_unused_cargill(df_cv_mc, used_cargill)
            
            # Calculate penalty for truly unused vessels
            if len(market_plan) > 0:
                used_for_market = set(market_plan["vessel"].tolist())
                all_used_cargill = used_cargill | used_for_market
            else:
                all_used_cargill = used_cargill
            
            penalty = unused_committed_vessel_penalty(_cargill_vessels, all_used_cargill, idle_days=1.0)
            
            # Full portfolio total
            portfolio_total = committed_total + market_total - penalty
            
            return committed_plan, market_plan, committed_total, market_total, penalty, portfolio_total, df_cv_cc, df_mv_cc
        
        # Load data
        with st.spinner("Loading fuel sensitivity data..."):
            dist_lut, base_bunker_prices, base_pf, vlsfo_price, mgo_price, cargill_vessels, market_vessels, committed, market_cargo, ffa = load_fuel_sensitivity_data()
        
        # Show current bunker prices
        with st.expander("ℹ️ Current Bunker Prices by Location"):
            price_data = []
            for loc, (vlsfo, mgo, _, _) in base_bunker_prices.items():
                price_data.append({"Location": loc, "VLSFO ($/MT)": vlsfo, "MGO ($/MT)": mgo})
            st.dataframe(pd.DataFrame(price_data), use_container_width=True)
        
        # Laycan toggle
        enforce_laycan_fuel = st.checkbox(
            "Enforce Laycan Constraints",
            value=False,
            help="If checked, only consider vessel-cargo combinations that meet the laycan (arrival window) requirements",
            key="enforce_laycan_fuel"
        )
        
        # Pre-calculate breakeven point to set as default slider value
        @st.cache_data
        def find_fuel_breakeven(_dist_lut, _base_bunker_prices, _base_pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, enforce_laycan):
            """Find the fuel price breakeven point using binary search"""
            def get_assignment_key(pct):
                curr_committed, curr_market, _, _, _, _, _, _ = run_fuel_scenario(
                    _dist_lut, _base_bunker_prices, _base_pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, pct, enforce_laycan
                )
                curr_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                                  for _, row in curr_committed.iterrows()}
                curr_market_assignment = {}
                if len(curr_market) > 0:
                    curr_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                                  for _, row in curr_market.iterrows()}
                return (frozenset(curr_committed_assignment.items()), frozenset(curr_market_assignment.items()))
            
            base_key = get_assignment_key(0)
            max_key = get_assignment_key(200)
            
            if max_key == base_key:
                return None  # No breakeven found
            
            # Binary search
            low, high = 0, 200
            while high - low > 1:
                mid = (low + high) // 2
                if get_assignment_key(mid) != base_key:
                    high = mid
                else:
                    low = mid
            return high
        
        # Get breakeven point
        with st.spinner("Finding tipping point..."):
            fuel_breakeven_default = find_fuel_breakeven(
                dist_lut, base_bunker_prices, base_pf, cargill_vessels, market_vessels, committed, market_cargo, enforce_laycan_fuel
            )
        
        # Set default slider value to breakeven if found, otherwise 0
        default_fuel_value = fuel_breakeven_default if fuel_breakeven_default is not None else 0
        
        # Fuel price slider
        st.subheader("⛽ Fuel Price Simulation")
        col1, col2 = st.columns(2)
        
        with col1:
            if fuel_breakeven_default is not None:
                st.info(f"📍 Slider set to tipping point: **{fuel_breakeven_default}%** (where assignment changes)")
            fuel_increase_pct = st.slider(
                "Fuel price change (%)",
                min_value=-50,
                max_value=200,
                value=default_fuel_value,
                step=1,
                help="Simulate fuel price changes. Negative = decrease, Positive = increase"
            )
        
        with col2:
            # Show simulated prices
            multiplier = 1 + (fuel_increase_pct / 100.0)
            st.metric("Simulated VLSFO (Singapore)", f"${vlsfo_price * multiplier:.2f}/MT", 
                     f"{fuel_increase_pct:+.0f}%")
            st.metric("Simulated MGO (Singapore)", f"${mgo_price * multiplier:.2f}/MT",
                     f"{fuel_increase_pct:+.0f}%")
        
        # Run baseline (0%)
        baseline_committed, baseline_market, baseline_committed_total, baseline_market_total, baseline_penalty, baseline_total, _, _ = run_fuel_scenario(
            dist_lut, base_bunker_prices, base_pf, cargill_vessels, market_vessels, committed, market_cargo, 0, enforce_laycan_fuel
        )
        baseline_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in baseline_committed.iterrows()}
        baseline_market_assignment = {}
        if len(baseline_market) > 0:
            baseline_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in baseline_market.iterrows()}
        
        # Run current scenario
        current_committed, current_market, current_committed_total, current_market_total, current_penalty, current_total, _, _ = run_fuel_scenario(
            dist_lut, base_bunker_prices, base_pf, cargill_vessels, market_vessels, committed, market_cargo, fuel_increase_pct, enforce_laycan_fuel
        )
        current_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in current_committed.iterrows()}
        current_market_assignment = {}
        if len(current_market) > 0:
            current_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                              for _, row in current_market.iterrows()}
        
        # Display results - Summary metrics
        st.divider()
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Baseline Total P/L", 
                f"${baseline_total:,.2f}"
            )
        
        with col2:
            change = current_total - baseline_total
            st.metric(
                f"Current Total P/L ({fuel_increase_pct:+.0f}%)",
                f"${current_total:,.2f}",
                delta=f"${change:,.2f}"
            )
        
        with col3:
            committed_changed = baseline_committed_assignment != current_committed_assignment
            market_changed = baseline_market_assignment != current_market_assignment
            if committed_changed or market_changed:
                st.error("⚠️ Assignment CHANGED!")
            else:
                st.success("✅ Assignment unchanged")
        
        with col4:
            st.metric("Idle Penalty", f"${current_penalty:,.2f}")
        
        # Detailed breakdown
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Committed Cargo P/L", f"${current_committed_total:,.2f}", 
                     delta=f"${current_committed_total - baseline_committed_total:,.2f}")
        with col2:
            st.metric("Market Cargo P/L", f"${current_market_total:,.2f}",
                     delta=f"${current_market_total - baseline_market_total:,.2f}")
        with col3:
            st.metric("Penalty Change", f"${current_penalty:,.2f}",
                     delta=f"${current_penalty - baseline_penalty:,.2f}")
        
        st.divider()
        
        # Show assignment comparison - COMMITTED CARGO
        st.subheader("📦 Committed Cargo Assignment Comparison")
        
        comparison_data = []
        for cargo_id in baseline_committed_assignment.keys():
            base_vessel, base_port = baseline_committed_assignment[cargo_id]
            curr_vessel, curr_port = current_committed_assignment.get(cargo_id, ("N/A", "N/A"))
            
            changed = (base_vessel, base_port) != (curr_vessel, curr_port)
            
            # Get P/L and bunker for both scenarios
            base_row = baseline_committed[baseline_committed['base_cargo_id'] == cargo_id].iloc[0]
            curr_row = current_committed[current_committed['base_cargo_id'] == cargo_id].iloc[0]
            
            comparison_data.append({
                "Type": "COMMITTED",
                "Cargo": cargo_id.replace("COMMITTED_", ""),
                "Baseline Vessel": base_vessel,
                "Baseline Port": base_port,
                "Current Vessel": curr_vessel,
                "Current Port": curr_port,
                "Baseline Bunker": base_row['bunker_expense'],
                "Current Bunker": curr_row['bunker_expense'],
                "Bunker Change": curr_row['bunker_expense'] - base_row['bunker_expense'],
                "Baseline P/L": base_row['profit_loss'],
                "Current P/L": curr_row['profit_loss'],
                "P/L Change": curr_row['profit_loss'] - base_row['profit_loss'],
                "Changed": "🔴 YES" if changed else "✅ No"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(
            comparison_df.style.format({
                'Baseline Bunker': '${:,.2f}',
                'Current Bunker': '${:,.2f}',
                'Bunker Change': '${:,.2f}',
                'Baseline P/L': '${:,.2f}',
                'Current P/L': '${:,.2f}',
                'P/L Change': '${:,.2f}'
            }),
            use_container_width=True
        )
        
        # Show assignment comparison - MARKET CARGO
        st.subheader("📦 Market Cargo Assignment Comparison (Unused Cargill Vessels)")
        
        if len(baseline_market_assignment) > 0 or len(current_market_assignment) > 0:
            market_comparison_data = []
            all_market_cargos = set(baseline_market_assignment.keys()) | set(current_market_assignment.keys())
            
            for cargo_id in all_market_cargos:
                base_vessel, base_port = baseline_market_assignment.get(cargo_id, ("N/A", "N/A"))
                curr_vessel, curr_port = current_market_assignment.get(cargo_id, ("N/A", "N/A"))
                
                changed = (base_vessel, base_port) != (curr_vessel, curr_port)
                
                # Get P/L and bunker for both scenarios
                base_pl = 0
                base_bunker = 0
                curr_pl = 0
                curr_bunker = 0
                if len(baseline_market) > 0 and cargo_id in baseline_market['cargo'].values:
                    base_row = baseline_market[baseline_market['cargo'] == cargo_id].iloc[0]
                    base_pl = base_row['profit_loss']
                    base_bunker = base_row.get('bunker_expense', 0)
                if len(current_market) > 0 and cargo_id in current_market['cargo'].values:
                    curr_row = current_market[current_market['cargo'] == cargo_id].iloc[0]
                    curr_pl = curr_row['profit_loss']
                    curr_bunker = curr_row.get('bunker_expense', 0)
                
                market_comparison_data.append({
                    "Type": "MARKET",
                    "Cargo": cargo_id.replace("MARKET_", ""),
                    "Baseline Vessel": base_vessel,
                    "Baseline Port": base_port,
                    "Current Vessel": curr_vessel,
                    "Current Port": curr_port,
                    "Baseline Bunker": base_bunker,
                    "Current Bunker": curr_bunker,
                    "Bunker Change": curr_bunker - base_bunker,
                    "Baseline P/L": base_pl,
                    "Current P/L": curr_pl,
                    "P/L Change": curr_pl - base_pl,
                    "Changed": "🔴 YES" if changed else "✅ No"
                })
            
            market_comparison_df = pd.DataFrame(market_comparison_data)
            st.dataframe(
                market_comparison_df.style.format({
                    'Baseline Bunker': '${:,.2f}',
                    'Current Bunker': '${:,.2f}',
                    'Bunker Change': '${:,.2f}',
                    'Baseline P/L': '${:,.2f}',
                    'Current P/L': '${:,.2f}',
                    'P/L Change': '${:,.2f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No market cargo assignments in either scenario")
        
        st.divider()
        
        # Sensitivity chart
        st.subheader("📊 Fuel Price Sensitivity Chart")
        
        @st.cache_data
        def generate_fuel_sensitivity_data(_dist_lut, _base_bunker_prices, _base_pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, enforce_laycan):
            """Generate data for fuel sensitivity chart - uses binary search to find breakeven quickly"""
            
            def get_scenario_data(pct):
                """Get assignment key and totals for a given fuel % change"""
                curr_committed, curr_market, _, _, _, total, _, _ = run_fuel_scenario(
                    _dist_lut, _base_bunker_prices, _base_pf, _cargill_vessels, _market_vessels, _committed, _market_cargo, pct, enforce_laycan
                )
                curr_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) 
                                  for _, row in curr_committed.iterrows()}
                curr_market_assignment = {}
                if len(curr_market) > 0:
                    curr_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) 
                                  for _, row in curr_market.iterrows()}
                
                total_bunker = curr_committed['bunker_expense'].sum()
                if len(curr_market) > 0 and 'bunker_expense' in curr_market.columns:
                    total_bunker += curr_market['bunker_expense'].sum()
                
                key = (frozenset(curr_committed_assignment.items()), frozenset(curr_market_assignment.items()))
                return key, total, total_bunker
            
            # Get baseline (0%)
            base_key, base_total, base_bunker = get_scenario_data(0)
            
            # Binary search to find breakeven point (where assignment changes)
            def find_breakeven(low, high, base_key):
                """Binary search to find exact breakeven point"""
                if high - low <= 1:
                    return high if get_scenario_data(high)[0] != base_key else None
                
                mid = (low + high) // 2
                mid_key, _, _ = get_scenario_data(mid)
                
                if mid_key != base_key:
                    return find_breakeven(low, mid, base_key)
                else:
                    return find_breakeven(mid, high, base_key)
            
            # Quick check: does assignment change at max fuel increase (200%)?
            max_key, max_total, max_bunker = get_scenario_data(200)
            breakeven = None
            if max_key != base_key:
                breakeven = find_breakeven(0, 200, base_key)
            
            # Generate minimal chart data: baseline, key points, breakeven, and endpoint
            results = []
            results.append({"Fuel Change (%)": -50, "Total P/L": get_scenario_data(-50)[1], "Total Bunker Cost": get_scenario_data(-50)[2], "Assignment Changed": False})
            results.append({"Fuel Change (%)": 0, "Total P/L": base_total, "Total Bunker Cost": base_bunker, "Assignment Changed": False})
            
            if breakeven is not None:
                # Add point just before breakeven
                if breakeven > 1:
                    _, pre_total, pre_bunker = get_scenario_data(breakeven - 1)
                    results.append({"Fuel Change (%)": breakeven - 1, "Total P/L": pre_total, "Total Bunker Cost": pre_bunker, "Assignment Changed": False})
                # Add breakeven point
                _, be_total, be_bunker = get_scenario_data(breakeven)
                results.append({"Fuel Change (%)": breakeven, "Total P/L": be_total, "Total Bunker Cost": be_bunker, "Assignment Changed": True})
            
            # Add endpoint
            results.append({"Fuel Change (%)": 200, "Total P/L": max_total, "Total Bunker Cost": max_bunker, "Assignment Changed": max_key != base_key})
            
            # Sort by fuel change %
            results = sorted(results, key=lambda x: x["Fuel Change (%)"])
            
            return pd.DataFrame(results), breakeven
        
        fuel_sensitivity_df, fuel_breakeven = generate_fuel_sensitivity_data(
            dist_lut, base_bunker_prices, base_pf, cargill_vessels, market_vessels, committed, market_cargo, enforce_laycan_fuel
        )
        
        # Display P/L chart with breakeven marker
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Total P/L vs Fuel Price Change**")
            chart_df = fuel_sensitivity_df.set_index("Fuel Change (%)")["Total P/L"]
            st.line_chart(chart_df)
            if fuel_breakeven is not None:
                st.caption(f"⚠️ Assignment changes at {fuel_breakeven}% (vertical shift expected)")
        
        with col2:
            st.write("**Bunker Cost vs Fuel Price Change**")
            st.line_chart(fuel_sensitivity_df.set_index("Fuel Change (%)")["Total Bunker Cost"])
        
        # Explanation
        st.markdown("""
        ---
        **📖 Understanding the Chart**
        
        1. **Linear Relationship**: P/L decreases linearly as fuel prices increase because:
           - Bunker cost = Fuel Consumption (MT) × Fuel Price ($/MT)
           - Fuel consumption is fixed for a given route
           - So: `Change in P/L = -Fuel Consumption × Change in Price`
        
        2. **Slope Indicates Exposure**: A steeper slope means higher fuel consumption and greater exposure to fuel price risk.
        
        3. **Tipping Point (Breakeven)**: When fuel prices increase enough, a different vessel-cargo assignment may become more profitable. This is the "tipping point" where the optimizer switches to a different solution.
        
        **Hedging Insight**: This analysis helps determine at what fuel price level you should consider fuel hedging or bunkering strategies.
        """)
        
        st.divider()
        
        # Key metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost per 10% increase
            if len(fuel_sensitivity_df) >= 2:
                try:
                    idx_0 = fuel_sensitivity_df[fuel_sensitivity_df["Fuel Change (%)"] == 0].index[0]
                    idx_10 = fuel_sensitivity_df[fuel_sensitivity_df["Fuel Change (%)"] == 10].index[0]
                    cost_per_10pct = fuel_sensitivity_df.iloc[idx_0]["Total P/L"] - fuel_sensitivity_df.iloc[idx_10]["Total P/L"]
                    st.info(f"💰 **Cost per 10% fuel increase:** ~${cost_per_10pct:,.2f}")
                except:
                    st.info("Cost per 10% increase data not available")
        
        with col2:
            # Breakeven info
            if fuel_breakeven is not None:
                breakeven_vlsfo = vlsfo_price * (1 + fuel_breakeven/100)
                st.warning(f"⚠️ **Tipping Point:** Assignment changes at **{fuel_breakeven:+.0f}%** (VLSFO ~${breakeven_vlsfo:.2f}/MT)")
            else:
                st.success("🟢 Assignment remains optimal across all tested fuel price scenarios")
        
        # Show what changes at breakeven
        if fuel_breakeven is not None:
            st.divider()
            st.subheader("🔀 What Changes at the Tipping Point?")
            # Get baseline assignment (vessel, port)
            baseline_committed_bp, baseline_market_bp, _, _, _, _, _, _ = run_fuel_scenario(
                dist_lut, base_bunker_prices, base_pf, cargill_vessels, market_vessels, committed, market_cargo, 0, enforce_laycan_fuel
            )
            baseline_committed_assignment_bp = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) for _, row in baseline_committed_bp.iterrows()}
            baseline_market_assignment_bp = {}
            if len(baseline_market_bp) > 0:
                baseline_market_assignment_bp = {row["cargo"]: (row["vessel"], row["discharge_port"]) for _, row in baseline_market_bp.iterrows()}
            
            # Get assignment at breakeven (vessel, port)
            breakeven_committed, breakeven_market, _, _, _, _, _, _ = run_fuel_scenario(
                dist_lut, base_bunker_prices, base_pf, cargill_vessels, market_vessels, committed, market_cargo, fuel_breakeven, enforce_laycan_fuel
            )
            breakeven_committed_assignment = {row["base_cargo_id"]: (row["vessel"], row["discharge_port"]) for _, row in breakeven_committed.iterrows()}
            breakeven_market_assignment = {}
            if len(breakeven_market) > 0:
                breakeven_market_assignment = {row["cargo"]: (row["vessel"], row["discharge_port"]) for _, row in breakeven_market.iterrows()}
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Tipping Point (0%)**")
                st.write("*Committed Cargo:*")
                for cargo, (vessel, port) in baseline_committed_assignment_bp.items():
                    cargo_short = cargo.replace("COMMITTED_", "")
                    st.write(f"• {cargo_short} → **{vessel}** @ {port}")
                if baseline_market_assignment_bp:
                    st.write("*Market Cargo:*")
                    for cargo, (vessel, port) in baseline_market_assignment_bp.items():
                        cargo_short = cargo.replace("MARKET_", "")
                        st.write(f"• {cargo_short} → **{vessel}** @ {port}")
            with col2:
                st.write(f"**At Tipping Point ({fuel_breakeven}%)**")
                st.write("*Committed Cargo:*")
                for cargo, (vessel, port) in breakeven_committed_assignment.items():
                    cargo_short = cargo.replace("COMMITTED_", "")
                    changed = "🔴" if baseline_committed_assignment_bp.get(cargo) != (vessel, port) else ""
                    st.write(f"• {cargo_short} → **{vessel}** @ {port} {changed}")
                if breakeven_market_assignment:
                    st.write("*Market Cargo:*")
                    for cargo, (vessel, port) in breakeven_market_assignment.items():
                        cargo_short = cargo.replace("MARKET_", "")
                        changed = "🔴" if baseline_market_assignment_bp.get(cargo) != (vessel, port) else ""
                        st.write(f"• {cargo_short} → **{vessel}** @ {port} {changed}")
            st.markdown("""
            ---
            **Why does the assignment change?**
            
            Different vessels have different fuel consumption rates. When fuel prices rise:
            - **Fuel-efficient vessels** become relatively more attractive
            - **Fuel-hungry vessels** become relatively less attractive
            
            At the tipping point, the fuel savings from using a more efficient vessel outweigh other cost differences.
            """)
        
    except ImportError as e:
        st.error(f"Could not load fuel sensitivity module: {e}")
        st.info("Make sure fuel_price_sensitivity.py is in the src folder.")