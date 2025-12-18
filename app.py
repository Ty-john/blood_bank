import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# --- 0. ENSURE DETERMINISM ---
# This ensures that random noise is reproducible. 
# If you return to the same slider settings, you get the exact same numbers.
np.random.seed(42)


# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Blood Bank Dashboard", # Changed from HematoAnalytics
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOADING SCREEN ---
# Only runs once per session
if 'loaded' not in st.session_state:
    loader = st.empty()
    loader_html = """
    <style>
        .stApp { overflow: hidden; }
        #loading-overlay {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background-color: #0e1117; z-index: 9999999;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        }
        .heartbeat { font-size: 80px; animation: beat 1.5s infinite; filter: drop-shadow(0 0 15px rgba(231, 76, 60, 0.6)); }
        .loading-text { margin-top: 20px; font-size: 20px; color: #e0e0e0; font-family: sans-serif; letter-spacing: 2px; animation: fade 1.5s infinite; }
        @keyframes beat { 0% { transform: scale(1); } 10% { transform: scale(1.1); } 20% { transform: scale(1); } 100% { transform: scale(1); } }
        @keyframes fade { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
    </style>
    <div id="loading-overlay"><div class="heartbeat">🩸</div><div class="loading-text">LOADING INVENTORY DATA...</div></div>
    """
    loader.markdown(loader_html, unsafe_allow_html=True)
    time.sleep(1.5)
    loader.empty()
    st.session_state['loaded'] = True

# --- 3. STATIC THEME ENGINE (DARK MODE ONLY) ---
# We hardcode dark_mode to True so that Section 4 (Map) doesn't crash.

dark_mode = True  # <--- THIS IS THE CRITICAL MISSING LINE

# Set Global Chart Variables for Plotly
chart_template = "plotly_dark"
map_style = "carto-darkmatter"

# Apply the CSS to fix the app background and tabs
st.markdown("""
<style>
    /* 1. BACKGROUND GRADIENT */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
    }

    /* 2. TABS (Centered & Full Width) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e2a36; 
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 50px;
        padding: 6px;
        gap: 10px;
        display: flex;
        width: 100%;
    }

    .stTabs [data-baseweb="tab"] {
        flex: 1;
        height: 45px;
        border-radius: 40px;
        border: none !important;
        color: #a0a0a0 !important;
        background-color: transparent;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] > div { background-color: transparent !important; }

    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important; 
        color: white !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.3);
    }

    /* 3. GLOBAL TEXT COLORS */
    h1, h2, h3, h4, h5, h6, p, li, span, label, [data-testid="stSidebar"] * { 
        color: #ffffff !important; 
    }
</style>
""", unsafe_allow_html=True)


# --- 4. DATA LOADING & PREP ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('blood_sample_size.csv')
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['months_active'] = df['months_since_first_donation'].replace(0, 1)
        df['monthly_rate'] = df['pints_donated'] / df['months_active']
        
        # --- A. MAP CITIES TO STATES ---
        city_to_state = {
            "Hobart": "Tasmania",
            "Darwin": "Northern Territory",
            "Melbourne": "Victoria",
            "Canberra": "Australian Capital Territory",
            "Perth": "Western Australia",
            "Adelaide": "South Australia",
            "Sydney": "New South Wales",
            "Brisbane": "Queensland"
        }
        df['state_name'] = df['city'].map(city_to_state)
        
        # --- B. GEOLOCATION MAPPING ---
        city_coords = {
            "Hobart": (-42.8821, 147.3272),
            "Darwin": (-12.4634, 130.8456),
            "Melbourne": (-37.8136, 144.9631),
            "Canberra": (-35.2809, 149.1300),
            "Perth": (-31.9505, 115.8605),
            "Adelaide": (-34.9285, 138.6007),
            "Sydney": (-33.8688, 151.2093),
            "Brisbane": (-27.4698, 153.0251)
        }
        
        df['lat'] = df['city'].map(lambda x: city_coords.get(x, (None, None))[0])
        df['lon'] = df['city'].map(lambda x: city_coords.get(x, (None, None))[1])
        
        df.dropna(subset=['lat', 'lon'], inplace=True)
        
        # Add slight jitter
        np.random.seed(42) 
        df['lat'] = df['lat'] + np.random.normal(0, 0.005, len(df))
        df['lon'] = df['lon'] + np.random.normal(0, 0.005, len(df))
            
        return df
    except FileNotFoundError:
        st.error("File 'blood_sample_size.csv' not found. Please upload it.")
        return pd.DataFrame()

# --- IMPORTANT: THIS EXECUTES THE FUNCTION ---
df = load_data()

# --- 5. LOGIC ENGINE (TUNED VOLATILITY) ---
def run_simulation(df, months_ahead, shock_pct, boost_pct, waste_pct):
    # 1. PREPARE HISTORY
    df['created_at'] = pd.to_datetime(df['created_at'])
    history = df.set_index('created_at').resample('MS')['pints_donated'].sum().reset_index()
    
    # 2. DETECT TREND & VOLATILITY
    lookback = 24
    if len(history) > lookback:
        recent = history.iloc[-lookback:].copy()
    else:
        recent = history.copy()
        
    recent['idx'] = np.arange(len(recent))
    
    # Calculate Slope & Volatility
    if len(recent) > 1:
        slope, intercept = np.polyfit(recent['idx'], recent['pints_donated'], 1)
        
        # Calculate how much the data naturally bounces (Standard Deviation)
        predicted_values = (slope * recent['idx']) + intercept
        residuals = recent['pints_donated'] - predicted_values
        volatility = residuals.std() 
    else:
        slope = 0
        volatility = 10 

    # 3. FORECAST GENERATION
    forecast_data = []
    
    last_actual_supply = history['pints_donated'].iloc[-1] if not history.empty else 1000
    current_inventory = last_actual_supply 
    current_date = history['created_at'].iloc[-1] if not history.empty else datetime.now()
    
    demand_ratio = 0.95 

    # Seed for consistent "randomness" (so lines don't jitter on every click)
    np.random.seed(42)

    for m in range(1, months_ahead + 1):
        # A. PROJECT SUPPLY
        trend_component = last_actual_supply + (slope * m)
        
        # FIX: Dampen the volatility (0.3x) so it doesn't drown out the sliders
        # This keeps the "shape" but prevents it from hitting zero or masking the boost
        random_bounce = np.random.normal(0, volatility * 0.3)
        
        base_supply = trend_component + random_bounce
        
        # Safety clamp (but now less likely to be hit due to dampening)
        base_supply = max(base_supply, 0)
        
        # B. APPLY SLIDERS
        sim_supply = base_supply * (1 + boost_pct/100) * (1 - waste_pct/100)
        
        # Demand follows supply
        base_demand = base_supply * demand_ratio
        
        # Reduced noise for demand as well
        demand_noise = np.random.normal(0, volatility * 0.15) 
        sim_demand = (base_demand + demand_noise) * (1 + shock_pct/100)

        # C. UPDATE INVENTORY
        current_inventory += (sim_supply - sim_demand)
        
        # D. DATE HANDLING
        next_date = current_date + pd.DateOffset(months=m)
        
        forecast_data.append({
            "Date": next_date.strftime("%b %Y"),
            "Inventory": int(current_inventory),
            "MonthlySupply": int(sim_supply),
            "MonthlyDemand": int(sim_demand)
        })

    forecast_df = pd.DataFrame(forecast_data)
    
    avg_sup = forecast_df['MonthlySupply'].mean() if not forecast_df.empty else 0
    avg_dem = forecast_df['MonthlyDemand'].mean() if not forecast_df.empty else 0
        
    return forecast_df, avg_sup, avg_dem

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    # OPTION A: Robust Emoji Logo
    st.markdown("<div style='font-size: 80px; line-height: 0.8; margin-bottom: 10px;'>🩸</div>", unsafe_allow_html=True)
    
    st.title("Blood Bank Dashboard") # Changed from HematoLink
    st.caption("Supply & Shortage Monitor") # Changed from v4.1 | Geospatial Engine
    
    st.markdown("---")
    
    # [LOCATION FILTER REMOVED] - Now handled in Tab 1
    
    st.subheader("🛠️ Control Panel")
    months_to_predict = st.slider("Forecast Horizon (Months)", 1, 12, 6)
    
    st.markdown("### 🔥 Crisis Simulation")
    demand_shock = st.slider("Casualty Surge (Demand)", 0, 50, 0, format="+%d%%", key="demand_slider")
    supply_boost = st.slider("Donation Campaign (Supply)", 0, 30, 0, format="+%d%%", key="supply_slider")
    
    with st.expander("⚙️ Logistics Efficiency"):
        wastage_rate = st.slider("Spoilage Rate", 0, 15, 5, format="%d%%", key="waste_slider")

    # Run simulation with the FULL DataFrame (df)
    # This ensures the CSV export contains the complete organizational data
    if not df.empty:
        forecast_df, monthly_sup, monthly_dem = run_simulation(df, months_to_predict, demand_shock, supply_boost, wastage_rate)
        
        st.markdown("---")
        st.subheader("💾 Export Data")
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")


# --- 7. MAIN DASHBOARD UI ---
# Check if dataframe exists
if df.empty:
    st.stop()

# --- A. CALCULATE GLOBAL METRICS ---
# We use the 'forecast_df' calculated in the Sidebar (Global View)
final_inventory = forecast_df.iloc[-1]['Inventory']
monthly_balance = monthly_sup - monthly_dem
days_of_supply = (final_inventory / (monthly_dem/30)) if monthly_dem > 0 else float('inf')

# --- B. HEADER & BADGE ---
col_head, col_badge = st.columns([3, 1])
with col_head:
    st.title(f"Clinical Supply Forecast: {months_to_predict} Months")
    st.markdown("Real-time monitoring of blood bank inventory and predicted shortages.")

with col_badge:
    if monthly_balance >= 0: 
        bg, text = "#238636", "INVENTORY GROWING"
    elif days_of_supply > 10:
        bg, text = "#d29922", "SLOW DEPLETION"
    else: 
        bg, text = "#da3633", "CRITICAL DEPLETION"
    # Badge styling
    st.markdown(f'<div style="text-align: right; margin-top: 20px;"><span class="custom-badge" style="background-color: {bg}; box-shadow: 0 0 10px {bg}60;">● {text}</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- C. TABS (THIS IS THE MISSING PART) ---
# You must define tab1, tab2, tab3 here so the rest of the code knows what they are.
tab1, tab2, tab3 = st.tabs(["Overview", "By Blood Type", "Utilization"])

# --- TAB 1: OVERVIEW & GEOSPATIAL ENGINE ---
# --- TAB 1: OVERVIEW & GEOSPATIAL ENGINE ---
with tab1:
    # --- 0. HANDLE MAP INTERACTION (Must be at the top) ---
    # This catches the map click BEFORE the widgets are drawn to avoid the API Exception.
    
    # Check if a map interaction occurred
    if "aus_map_interaction" in st.session_state:
        map_data = st.session_state.aus_map_interaction
        
        # Initialize a history tracker to prevent infinite loops
        if "last_map_data" not in st.session_state:
            st.session_state["last_map_data"] = None
            
        # Only process if the map selection has CHANGED since the last run
        if map_data != st.session_state["last_map_data"]:
            
            # Update history so we don't re-process this click
            st.session_state["last_map_data"] = map_data
            
            # Check if there is a valid selection
            if map_data and "selection" in map_data and map_data["selection"]["points"]:
                clicked_point = map_data["selection"]["points"][0]
                clicked_state = clicked_point.get("location") or clicked_point.get("hovertext")
                
                if clicked_state:
                    # Get current active filters
                    current_pills = st.session_state.get("state_pills", [])
                    
                    # TOGGLE LOGIC: Add if missing, Remove if present
                    if clicked_state in current_pills:
                        current_pills.remove(clicked_state)
                    else:
                        current_pills.append(clicked_state)
                        
                    # CRITICAL: Update the widget state BEFORE it is instantiated
                    st.session_state["state_pills"] = current_pills
    
   # 1. GLOBAL METRICS (Gradient Glass Cards)
    # Added 'linear-gradient' to the background for a premium 3D look.
    
    # --- Logic for Dynamic Colors & Icons ---
    if monthly_balance >= 0:
        net_color = "#4ade80" # Green
        net_bg = "rgba(74, 222, 128, 0.15)"
        net_icon = "📈"
        net_label = "Surplus"
    else:
        net_color = "#f87171" # Red
        net_bg = "rgba(248, 113, 113, 0.15)"
        net_icon = "📉"
        net_label = "Deficit"

    if days_of_supply < 7:
        dos_color = "#f87171" # Red (Critical)
        dos_bg = "rgba(248, 113, 113, 0.15)"
        dos_icon = "🚨"
    elif days_of_supply < 14:
        dos_color = "#fbbf24" # Amber (Warning)
        dos_bg = "rgba(251, 191, 36, 0.15)"
        dos_icon = "⚠️"
    else:
        dos_color = "#4ade80" # Green (Healthy)
        dos_bg = "rgba(74, 222, 128, 0.15)"
        dos_icon = "✅"

    # --- CSS Styles (Now with Gradients) ---
    st.markdown("""
    <style>
        .metric-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        .metric-card {
            /* GRADIENT BACKGROUND: Lighter Top-Left -> Darker Bottom-Right */
            background: linear-gradient(135deg, #2c3e50 0%, #1e2a36 100%);
            
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            /* On hover, slightly brighten the gradient */
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
            transform: translateY(-5px);
            border-color: rgba(255, 255, 255, 0.3);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        }
        .metric-title {
            color: #b0c4de; /* Slightly lighter text for contrast */
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .metric-value {
            color: white;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.5); /* Text pops against gradient */
        }
        .metric-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            box-shadow: inset 0 0 5px rgba(0,0,0,0.1); /* Inner shadow for depth */
        }
        .card-icon {
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 1.5rem;
            opacity: 0.7;
            filter: drop-shadow(0 2px 2px rgba(0,0,0,0.5));
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Generate HTML ---
    html_cards = f"""
    <div class="metric-container">
        <div class="metric-card" style="border-left: 4px solid #3b82f6;">
            <div class="metric-title">Avg Monthly Inflow</div>
            <div class="metric-value">{int(monthly_sup):,}</div>
            <div class="metric-badge" style="background: rgba(59, 130, 246, 0.2); color: #93c5fd;">
                ↑ {supply_boost}% Boost
            </div>
            <div class="card-icon">🩸</div>
        </div>
        <div class="metric-card" style="border-left: 4px solid #f97316;">
            <div class="metric-title">Avg Monthly Outflow</div>
            <div class="metric-value">{int(monthly_dem):,}</div>
            <div class="metric-badge" style="background: rgba(249, 115, 22, 0.2); color: #fdba74;">
                ↑ {demand_shock}% Surge
            </div>
            <div class="card-icon">🚑</div>
        </div>
        <div class="metric-card" style="border-left: 4px solid {net_color};">
            <div class="metric-title">Net Monthly Flow</div>
            <div class="metric-value" style="color: {net_color};">{int(monthly_balance):,}</div>
            <div class="metric-badge" style="background: {net_bg}; color: {net_color};">
                {net_label}
            </div>
            <div class="card-icon">{net_icon}</div>
        </div>
        <div class="metric-card" style="border-left: 4px solid {dos_color};">
            <div class="metric-title">Est. Days of Supply</div>
            <div class="metric-value" style="color: {dos_color};">{days_of_supply:.1f} Days</div>
            <div class="metric-badge" style="background: {dos_bg}; color: {dos_color};">
                Target: >7 Days
            </div>
            <div class="card-icon">{dos_icon}</div>
        </div>
    </div>
    """
    
    # --- RENDER ---
    st.markdown(html_cards, unsafe_allow_html=True)

   # 2. CLEAN HEADER & FILTER (Selection Pills)
    # Uses clickable "Pills" instead of a text box. No typing required!
    
    col_title, col_status = st.columns([3, 1])
    
    with col_title:
        st.subheader("📍 National Overview")
        
    with col_status:
        # Status indicator
        count = len(st.session_state.get('state_pills', []))
        if count == 0:
             st.markdown("*Viewing: **All Australia***")
        else:
             st.markdown(f"*Viewing: **{count} Regions***")

    # 3. FILTER PANEL (Clickable Chips)
    with st.expander("🔎 Filter States", expanded=True):
        
        if 'state_name' in df.columns:
            all_states = sorted(df['state_name'].dropna().unique())
            
            # CHECK: Does this Streamlit version support st.pills?
            if hasattr(st, "pills"):
                # MODERN: Clickable Buttons (Not Writeable!)
                selected_states = st.pills(
                    "Click to select states:",
                    all_states,
                    selection_mode="multi",
                    key="state_pills"
                )
            else:
                # COMPATIBILITY: Fallback for older Streamlit versions
                # We use a clean multiselect but remove the "Type..." text to discourage typing
                selected_states = st.multiselect(
                    "Select States:", 
                    all_states, 
                    default=[],
                    placeholder="Choose states...", # Simple text, no instructions to type
                    key="state_pills"
                )
        else:
            selected_states = []

    # 4. FILTERING LOGIC
    if selected_states:
        df_local = df[df['state_name'].isin(selected_states)].copy()
    else:
        df_local = df.copy()

    # Re-Run Simulation for this local view
    local_forecast, _, _ = run_simulation(df_local, months_to_predict, demand_shock, supply_boost, wastage_rate)

  # 4. MAP (Interactive: Click to Select)
    # Allows users to click on the map to toggle states in the filter.

    # --- A. PREPARE DATA ---
    state_data = df[['state_name']].dropna().drop_duplicates().copy()

    # Logic: Check if filter is active
    if selected_states:
        state_data['active_flag'] = state_data['state_name'].isin(selected_states).astype(int)
    else:
        state_data['active_flag'] = 0

    pints_by_state = df_local.groupby('state_name')['pints_donated'].sum()
    state_data['pints_donated'] = state_data['state_name'].map(pints_by_state).fillna(0)

    # --- B. LOAD GEOJSON ---
    import json
    import os

    map_file_path = 'states_min.geojson' 
    aus_geo = None

    if os.path.exists(map_file_path):
        with open(map_file_path, 'r') as f:
            aus_geo = json.load(f)
    else:
        st.error(f"⚠️ Map file missing! Please ensure '{map_file_path}' is in this folder.")

    # --- C. GENERATE MAP ---
    if aus_geo:
        active_color = "#D32F2F" if dark_mode else "#FF4B4B"
        inactive_color = "#7a7a7a" if dark_mode else "#E0E0E0"
        dot_color = "#D30000" if not dark_mode else "white"

        custom_scale = [[0.0, inactive_color], [1.0, active_color]]

        fig_map = px.choropleth_mapbox(
            state_data,
            geojson=aus_geo,
            locations='state_name',
            featureidkey="properties.STATE_NAME", 
            color='active_flag',
            color_continuous_scale=custom_scale,
            range_color=[0, 1],
            hover_name='state_name',
            hover_data={'active_flag': False, 'pints_donated': True},
            mapbox_style=map_style,
            zoom=2.4,
            center={"lat": -28, "lon": 133},
            opacity=0.9
        )
        
        fig_map.update_traces(
            marker_line_width=1,
            marker_line_color="rgba(255,255,255,0.25)" if dark_mode else "rgba(0,0,0,0.2)",
            showscale=False
        )
        
        # Only add dots if selection is active
        if selected_states:
            fig_map.add_trace(go.Scattermapbox(
                lat=df_local['lat'],
                lon=df_local['lon'],
                mode='markers+text',
                marker=go.scattermapbox.Marker(size=8, color=dot_color),
                text=df_local['city'],
                textfont=dict(size=12, color=dot_color, family="Arial Black"),
                textposition="top center",
                showlegend=False,
                hoverinfo='none'
            ))
        
        fig_map.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            # IMPORTANT: Enable click selection mode
            clickmode='event+select'
        )
        
        # --- D. RENDER MAP WITH SELECTION EVENT ---
        # on_select="rerun" makes the app reload when you click the map
        map_event = st.plotly_chart(
            fig_map, 
            use_container_width=True, 
            on_select="rerun",
            key="aus_map_interaction" # Unique key to track state
        )

        # --- E. HANDLE CLICK EVENT ---
        # If user clicked a state, we update the main filter
        if map_event and "selection" in map_event and map_event["selection"]["points"]:
            
            # 1. Grab the clicked state name
            clicked_point = map_event["selection"]["points"][0]
            clicked_state = clicked_point.get("location") or clicked_point.get("hovertext")
            
            if clicked_state:
                # 2. Get current filter list (handle safely)
                current_pills = st.session_state.get("state_pills", [])
                
                # 3. Toggle Logic (If active -> remove. If inactive -> add)
                if clicked_state in current_pills:
                    current_pills.remove(clicked_state)
                else:
                    current_pills.append(clicked_state)
                
                # 4. Push update to Session State & Rerun
                st.session_state["state_pills"] = current_pills
                st.rerun()

    else:
        st.warning("Showing basic map (GeoJSON file not found).")
        fig_dots = px.scatter_mapbox(df_local, lat="lat", lon="lon", size="pints_donated", zoom=2.4)
        st.plotly_chart(fig_dots, use_container_width=True)

  # 5. CLEAN & FRIENDLY SUPPLY VS DEMAND CHART
    st.markdown("---")
    
    st.subheader(f"📈 {'National Inventory Trends' if not selected_states else 'Local Inventory Trends'}")
    st.caption("Historical performance compared against future predicted requirements.")
    
    # 1. Prepare Data
    hist_supply = df_local.set_index('created_at').resample('MS')['pints_donated'].sum().reset_index()
    hist_supply.columns = ['Date', 'Supply']
    np.random.seed(42)
    hist_supply['Demand'] = hist_supply['Supply'] * np.random.uniform(0.85, 1.15, len(hist_supply))
    
    forecast_plot = local_forecast.copy()
    forecast_plot['Date'] = pd.to_datetime(forecast_plot['Date'], format="%b %Y")
    
    fig_sd = go.Figure()

    # --- THE CLEAN FIX: USE AREA FILLS FOR CLARITY ---
    # Supply History (Blue Fill)
    fig_sd.add_trace(go.Scatter(
        x=hist_supply['Date'], y=hist_supply['Supply'], 
        mode='lines', name='Supply (Donations)', 
        fill='tozeroy', fillcolor='rgba(54, 162, 235, 0.1)',
        line=dict(color='#36a2eb', width=3),
        legendgroup="Sup"
    ))
    
    # Demand History (Red Fill)
    fig_sd.add_trace(go.Scatter(
        x=hist_supply['Date'], y=hist_supply['Demand'], 
        mode='lines', name='Demand (Usage)', 
        fill='tozeroy', fillcolor='rgba(255, 99, 132, 0.1)',
        line=dict(color='#ff6384', width=3),
        legendgroup="Dem"
    ))

    # --- FORECAST LINES (Dashed & No Fill to look "lighter") ---
    fig_sd.add_trace(go.Scatter(
        x=forecast_plot['Date'], y=forecast_plot['MonthlySupply'], 
        mode='lines+markers', name='Supply Forecast', 
        line=dict(color='#36a2eb', width=2, dash='dot'),
        marker=dict(size=4, symbol='circle'),
        legendgroup="Sup", showlegend=False
    ))
    
    fig_sd.add_trace(go.Scatter(
        x=forecast_plot['Date'], y=forecast_plot['MonthlyDemand'], 
        mode='lines+markers', name='Demand Forecast', 
        line=dict(color='#ff6384', width=2, dash='dot'),
        marker=dict(size=4, symbol='circle'),
        legendgroup="Dem", showlegend=False
    ))

    # --- MINIMALIST LAYOUT ---
    # Shading the Forecast Zone (Lighter)
    fig_sd.add_vrect(
        x0=forecast_plot['Date'].min(), x1=forecast_plot['Date'].max(),
        fillcolor="rgba(255, 255, 255, 0.03)", layer="below", line_width=0
    )

    fig_sd.update_layout(
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        height=400, 
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center'),
        xaxis=dict(
            showgrid=False, 
            tickfont=dict(color="rgba(255,255,255,0.5)"),
            range=[hist_supply['Date'].min(), forecast_plot['Date'].max()]
        ),
        yaxis=dict(
            title="Pints of Blood", 
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="rgba(255,255,255,0.5)")
        ),
        margin=dict(l=0, r=20, t=20, b=0)
    )
    
    st.plotly_chart(fig_sd, use_container_width=True)

# --- TAB 2 & 3 (Preserve your existing code if you have it) ---
with tab2:
    st.subheader("🧬 Blood Type distribution")
    
    # --- 1. DATA PREPARATION (Updated for Coverage Ratio) ---
    supply_dist = df.groupby('blood_group')['pints_donated'].sum() / df['pints_donated'].sum()
    type_data = []
    
    for bg_name, ratio in supply_dist.items():
        t_sup = int(monthly_sup * ratio)
        t_dem = int(monthly_dem * ratio) 
        # Calculate coverage (e.g., 0.8 means only 80% of demand is met)
        coverage = t_sup / t_dem if t_dem > 0 else 1.0
        
        type_data.append({"Type": bg_name, "Category": "Supply", "Units": t_sup, "Coverage": coverage})
        type_data.append({"Type": bg_name, "Category": "Demand", "Units": t_dem, "Coverage": coverage})
    
    df_bar = pd.DataFrame(type_data)

    # --- 2. DYNAMIC PRIORITY LOGIC ---
    # We find the type with the LOWEST coverage (The real bottleneck)
    priority_df = df_bar.sort_values('Coverage', ascending=True)
    critical_type = priority_df.iloc[0]['Type']
    lowest_coverage = priority_df.iloc[0]['Coverage'] * 100

    # Determine Banner style based on shortage severity
    if lowest_coverage < 100:
        msg = f"Type <b>{critical_type}</b> coverage has dropped to {lowest_coverage:.1f}%. Immediate donation drive required."
        b_color = "#ff4b4b" # Red for deficit
        bg_rgb = "255, 75, 75"
    else:
        msg = f"All types are currently stable. Type <b>{critical_type}</b> has the narrowest safety margin."
        b_color = "#fbbf24" # Amber for low margin
        bg_rgb = "251, 191, 36"

    # --- 3. DYNAMIC INSIGHT BANNER ---
    st.markdown(f"""
        <div style="background: rgba({bg_rgb}, 0.1); border-left: 5px solid {b_color}; padding: 12px 20px; border-radius: 8px; margin-bottom: 30px;">
            <span style="color: {b_color}; font-weight: bold; font-size: 1.1rem;">⚠️ Inventory Priority:</span> 
            <span style="color: white; font-size: 1rem;">{msg}</span>
        </div>
    """, unsafe_allow_html=True)

    # --- 4. IMPROVED CHARTS ---
    c_bar, c_pie = st.columns([1.8, 1.2], gap="large")
    
    with c_bar:
        st.markdown("#### 📊 Projected Monthly Flow")
        st.caption("Detailed Supply vs Demand comparison per blood group")
        
        fig_bar = px.bar(
            df_bar, x="Type", y="Units", color="Category", barmode="group",
            color_discrete_map={"Supply": "#00C851", "Demand": "#ff4444"},
            text_auto='.2s'
        )
        
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(title="", tickfont=dict(size=14, color='white')),
            yaxis=dict(title="Units", gridcolor="rgba(255,255,255,0.1)"),
            bargap=0.25
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_pie:
        st.markdown("#### 🎯 Volume Distribution")
        pie_mode = st.radio("Focus View:", ["Supply", "Demand"], horizontal=True, key="blood_pie_final_clean")
        
        is_supply = pie_mode == "Supply"
        df_pie = df_bar[df_bar['Category'] == ('Supply' if is_supply else 'Demand')]
        center_val = int(monthly_sup if is_supply else monthly_dem)
        
        fig_pie = px.pie(df_pie, values='Units', names='Type', hole=0.7,
                         color_discrete_sequence=px.colors.qualitative.Prism if is_supply else px.colors.qualitative.Safe)
        
        fig_pie.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400, showlegend=True, 
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(text=f"<span style='font-size:24px; font-weight:bold;'>{center_val}</span><br><span style='font-size:14px; color:#a0a0a0;'>{pie_mode} Total</span>", 
                         x=0.5, y=0.5, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.subheader("♻️ Inventory Flow & Utilization Risk")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. CALCULATE FLOW DATA (The Math for the Waterfall)
    # We define the starting stock and the subtractive values (Used/Wasted)
    start_inv = monthly_sup * 0.25  # Assume 25% base stock for visualization
    used = -monthly_dem 
    waste = -(monthly_sup * (wastage_rate/100))
    
    # Coverage Ratio for the Gauge
    coverage_ratio = (monthly_sup / monthly_dem * 100) if monthly_dem > 0 else 0
    
    # Color logic for Gauge
    if 90 <= coverage_ratio <= 120:
        health_color = "#39d353" # Green
        health_status = "OPTIMAL"
    else:
        health_color = "#fbbf24" # Yellow/Amber
        health_status = "IMBALANCED"

    col_u1, col_u2 = st.columns([2, 1])
    
    # --- CHART 1: PRO WATERFALL (The "Story" of the Blood) ---
    with col_u1:
        st.caption("Monthly Inventory Flow: How donations are utilized")
        
        fig_water = go.Figure(go.Waterfall(
            name = "Inventory", 
            orientation = "v",
            # 'measure' defines if the bar starts from 0 (absolute/total) or from the previous bar (relative)
            measure = ["absolute", "relative", "relative", "relative", "total"],
            x = ["Starting Stock", "New Donations", "Transfused", "Wastage", "End Inventory"],
            textposition = "outside",
            # We show the + / - values clearly
            text = [f"+{int(start_inv)}", f"+{int(monthly_sup)}", f"{int(used)}", f"{int(waste)}", "Total Stock"],
            y = [start_inv, monthly_sup, used, waste, 0],
            
            # THE CONNECTION FIX: Adding bridge lines between the bars
            connector = {"line":{"color":"rgba(255, 255, 255, 0.4)", "width": 1, "dash": "dot"}},
            
            decreasing = {"marker":{"color":"#f87171"}}, # Red for blood leaving
            increasing = {"marker":{"color":"#39d353"}}, # Green for blood entering
            totals = {"marker":{"color":"#3b82f6"}}      # Blue for final total
        ))

        fig_water.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend = False,
            margin=dict(l=10, r=10, t=30, b=20)
        )
        st.plotly_chart(fig_water, use_container_width=True)
        
    # --- CHART 2: THE GAUGE (The "Health Check") ---
    with col_u2:
        st.caption(f"Supply Health: {health_status}")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = coverage_ratio,
            number = {'suffix': "%", 'font': {'size': 34, 'color': "white"}},
            title = {'text': "Supply/Demand Balance", 'font': {'size': 14, 'color': "#a0a0a0"}},
            gauge = {
                'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': health_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 90], 'color': 'rgba(248, 113, 113, 0.15)'},   # Shortage
                    {'range': [90, 120], 'color': 'rgba(74, 222, 128, 0.15)'},  # Optimal
                    {'range': [120, 200], 'color': 'rgba(251, 191, 36, 0.15)'}  # Waste Risk
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'value': 100}
            }
        ))
        
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 3. OPTIONAL SUMMARY BOX
    st.markdown(
        f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border-left: 5px solid {health_color};">
            <strong>Logistics Summary:</strong> Your current strategy yields a <b>{coverage_ratio:.1f}%</b> coverage ratio. 
            To maintain optimal stock levels (90-120%), aim to keep the blue "End Inventory" bar in the Waterfall chart above 0 at all times.
        </div>
        """, unsafe_allow_html=True
    )

# --- FOOTER ---
st.markdown("---")
f1, f2 = st.columns([1, 1])

with f1:
    st.caption("© 2025 TaZz • Health Informatics Project")

with f2:
    # Right-aligned link to the dataset
    st.markdown(
        """
        <div style='text-align: right; font-size: 12px; color: #666;'>
            Data Source: 
            <a href='https://www.kaggle.com/datasets/sumedh1507/blood-donor-dataset?resource=download' target='_blank' style='color: #36a2eb; text-decoration: none;'>
                Kaggle Blood Donor Dataset
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )