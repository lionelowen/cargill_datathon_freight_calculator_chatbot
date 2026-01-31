import google.generativeai as genai
import streamlit as st
import pandas as pd
from pathlib import Path

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
    
    return {
        'total_pl': df['profit_loss'].sum(),
        'total_revenue': df['revenue'].sum(),
        'total_bunker': df['bunker_expense'].sum(),
        'avg_tce': df['tce'].mean(),
        'num_assignments': len(df),
        'cargill_vessels': df[df['vessel_type'] == 'CARGILL']['vessel'].nunique(),
        'market_vessels': df[df['vessel_type'] == 'MARKET']['vessel'].nunique(),
    }

metrics = get_summary_metrics(df_optimal)

# Sidebar with summary
with st.sidebar:
    st.header("📊 Portfolio Summary")
    
    if metrics:
        st.metric("Total P/L", f"${metrics['total_pl']:,.2f}")
        st.metric("Avg TCE", f"${metrics['avg_tce']:,.2f}/day")
        st.metric("Total Assignments", metrics['num_assignments'])
        
        st.divider()
        st.subheader("Vessels Used")
        st.write(f"🔵 Cargill Vessels: {metrics['cargill_vessels']}")
        st.write(f"🟢 Market Vessels: {metrics['market_vessels']}")
        
        st.divider()
        st.subheader("Financial Breakdown")
        st.write(f"Revenue: ${metrics['total_revenue']:,.2f}")
        st.write(f"Bunker Cost: ${metrics['total_bunker']:,.2f}")
    else:
        st.warning("No optimal assignments loaded")

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📋 Assignments", "📈 Analysis"])

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
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            vessel_filter = st.multiselect(
                "Filter by Vessel",
                options=df_optimal['vessel'].unique(),
                default=df_optimal['vessel'].unique()
            )
        with col2:
            type_filter = st.multiselect(
                "Filter by Vessel Type",
                options=df_optimal['vessel_type'].unique(),
                default=df_optimal['vessel_type'].unique()
            )
        
        # Apply filters
        filtered_df = df_optimal[
            (df_optimal['vessel'].isin(vessel_filter)) &
            (df_optimal['vessel_type'].isin(type_filter))
        ]
        
        # Display key columns
        display_cols = ['assignment_category', 'vessel', 'cargo', 'load_port', 'discharge_port', 
                       'total_duration', 'profit_loss', 'tce', 'bunker_location']
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols].style.format({
                'profit_loss': '${:,.2f}',
                'tce': '${:,.2f}',
                'total_duration': '{:.1f} days'
            }),
            use_container_width=True
        )
        
        # Detailed view
        st.subheader("📝 Detailed Assignment View")
        for idx, row in filtered_df.iterrows():
            with st.expander(f"🚢 {row['vessel']} → {row['cargo']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📦 Vessel Info**")
                    st.write(f"Type: {row['vessel_type']}")
                    st.write(f"Start: {row['from_pos']}")
                
                with col2:
                    st.markdown("**🚢 Route Info**")
                    st.write(f"Load: {row['load_port']}")
                    st.write(f"Discharge: {row['discharge_port']}")
                    st.write(f"Ballast: {row['ballast_nm']:,.0f} nm")
                    st.write(f"Laden: {row['laden_nm']:,.0f} nm")
                
                with col3:
                    st.markdown("**💰 Financial**")
                    st.write(f"Revenue: ${row['revenue']:,.2f}")
                    st.write(f"Bunker: ${row['bunker_expense']:,.2f}")
                    st.write(f"**P/L: ${row['profit_loss']:,.2f}**")
                    st.write(f"**TCE: ${row['tce']:,.2f}/day**")
    else:
        st.warning("No optimal assignments found. Run the freight calculator first.")

# Tab 3: Analysis
with tab3:
    st.header("📈 Portfolio Analysis")
    
    if df_optimal is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("P/L by Assignment")
            chart_data = df_optimal[['cargo', 'profit_loss']].copy()
            chart_data['cargo'] = chart_data['cargo'].str.replace('COMMITTED_', '').str.replace('MARKET_', '')
            st.bar_chart(chart_data.set_index('cargo'))
        
        with col2:
            st.subheader("TCE by Vessel")
            tce_data = df_optimal[['vessel', 'tce']].copy()
            st.bar_chart(tce_data.set_index('vessel'))
        
        st.divider()
        
        # Cost breakdown
        st.subheader("💵 Cost Breakdown")
        cost_data = {
            'Category': ['Revenue', 'Hire Cost', 'Bunker Cost', 'Misc Cost'],
            'Amount': [
                df_optimal['revenue'].sum(),
                df_optimal['hire_net'].sum(),
                df_optimal['bunker_expense'].sum(),
                df_optimal['misc_expense'].sum()
            ]
        }
        cost_df = pd.DataFrame(cost_data)
        st.bar_chart(cost_df.set_index('Category'))
        
        # Vessel utilization
        st.subheader("🚢 Vessel Type Distribution")
        vessel_type_counts = df_optimal['vessel_type'].value_counts()
        st.bar_chart(vessel_type_counts)
    else:
        st.warning("No data available for analysis.")