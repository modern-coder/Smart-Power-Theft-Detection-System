import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import random
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Power Theft Detection System", page_icon="⚡", layout="wide")

# --- 2. SESSION STATE FOR RANDOMIZATION ---
if 'seed' not in st.session_state:
    st.session_state.seed = int(time.time())

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("⚡ PowerAdmin v3.0")
st.sidebar.write(f"**Session ID:** `{st.session_state.seed}`")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "⚙️ Generate Dynamic Data", "📊 Upload & Analyze", "📱 Live Field Check"])

# --- 4. HOME PAGE ---
if menu == "🏠 Home":
    st.title("Smart Power Theft Detection System")
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=100)
    st.write("""
    ### Project Overview
    This AI-powered system detects electricity fraud by analyzing consumption patterns.
    
    **Key Features:**
    * 🔍 **Unsupervised Learning:** Uses Isolation Forest to find anomalies.
    * 📈 **Real-time Visualization:** Graphs showing usage vs. suspected theft.
    * 💰 **Financial Tracking:** Calculates estimated revenue loss in Rupees.
    * 📱 **Field Ready:** Includes a manual check tool for linemen.
    """)
    st.info("Status: System Ready | Engine: IsolationForest (Scikit-Learn)")

# --- 5. GENERATE DYNAMIC DATA PAGE ---
elif menu == "⚙️ Generate Dynamic Data":
    st.title("Smart Power Theft Detection System")
    st.subheader("⚙️ Generate Randomized Test Dataset")
    
    theft_scenario = st.selectbox(
        "Select Theft Condition:",
        ["Evening Meter Bypass", "Direct Hooking (High Spikes)", "Meter Tampering (Flatline)", "Clean Data"]
    )
    
    if st.button("🔄 Refresh Random Seed & Generate"):
        st.session_state.seed = int(time.time())
        st.rerun()

    # Apply the unique seed
    np.random.seed(st.session_state.seed)
    random.seed(st.session_state.seed)

    # Data Generation Logic
    date_rng = pd.date_range(start='2026-03-01', periods=720, freq='h')
    df = pd.DataFrame(date_rng, columns=['Datetime'])
    df['Date'] = df['Datetime'].dt.date.astype(str)
    df['Time'] = df['Datetime'].dt.time.astype(str)
    
    # Normal Usage Pattern
    base_power = np.random.normal(1.5, 0.2, len(df))
    # Evening Spikes (Normal behavior)
    evening_mask = (df['Datetime'].dt.hour >= 18) & (df['Datetime'].dt.hour <= 22)
    base_power[evening_mask] += np.random.normal(2.5, 0.4, evening_mask.sum())
    
    df['Voltage'] = np.random.normal(230, 1.5, len(df))
    df['Power_kW'] = base_power

    # Injecting Random Theft
    all_days = df['Date'].unique()
    target_days = random.sample(list(all_days), 4)
    
    if "Evening Meter Bypass" in theft_scenario:
        for day in target_days:
            h = random.randint(18, 20)
            mask = (df['Date'] == day) & (df['Datetime'].dt.hour >= h) & (df['Datetime'].dt.hour <= h+2)
            df.loc[mask, 'Power_kW'] = np.random.uniform(0.01, 0.09)

    elif "Direct Hooking" in theft_scenario:
        for day in target_days:
            h = random.randint(11, 15)
            mask = (df['Date'] == day) & (df['Datetime'].dt.hour == h)
            df.loc[mask, 'Power_kW'] = np.random.uniform(11.0, 15.0)

    elif "Meter Tampering" in theft_scenario:
        for day in target_days[:2]:
            mask = (df['Date'] == day)
            df.loc[mask, 'Power_kW'] = 0.12

    df['Current'] = (df['Power_kW'] * 1000) / df['Voltage']
    df_final = df[['Date', 'Time', 'Voltage', 'Current', 'Power_kW']]

    # Download
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Randomized Dataset (CSV)",
        data=csv,
        file_name=f'meter_readings_{st.session_state.seed}.csv',
        mime='text/csv',
        key=f"btn_{st.session_state.seed}"
    )
    st.write(f"Active Dataset ID: `{st.session_state.seed}`")
    st.dataframe(df_final.head(10))

# --- 6. UPLOAD & ANALYZE PAGE ---
elif menu == "📊 Upload & Analyze":
    st.title("Smart Power Theft Detection System")
    st.subheader("📊 Consumption Pattern Analysis")
    
    file = st.file_uploader("Upload CSV Data", type="csv")
    
    if file:
        data = pd.read_csv(file)
        # Combine Date and Time for better analysis
        data['DT'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
        
        # ML Engine
        st.write("### 🧠 Running Anomaly Detection...")
        # Automatically tune contamination based on data variance
        model = IsolationForest(contamination=0.05, random_state=42)
        data['Anomaly'] = model.fit_predict(data[['Power_kW']])
        
        thefts = data[data['Anomaly'] == -1]
        
        # Financials
        loss = len(thefts) * 3.0 * 7.5 # Estimated 3kW loss at Rs 7.5 per unit
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", len(data))
        m2.metric("Theft Alerts 🚨", len(thefts), delta_color="inverse")
        m3.metric("Revenue Loss (Est.)", f"₹ {loss:,.2f}")
        
        st.markdown("---")
        
        # Visuals
        c_graph, c_pie = st.columns([2, 1])
        
        with c_graph:
            st.write("#### Power Usage Timeline")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['DT'], data['Power_kW'], label='Normal Usage', color='#1f77b4', alpha=0.7)
            ax.scatter(thefts['DT'], thefts['Power_kW'], color='red', label='Suspected Theft', s=20, zorder=5)
            ax.set_ylabel("Power (kW)")
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)
            
        with c_pie:
            st.write("#### Integrity Status")
            fig2, ax2 = plt.subplots()
            ax2.pie([len(data)-len(thefts), len(thefts)], labels=['Normal', 'Theft'], 
                    colors=['#2ca02c', '#d62728'], autopct='%1.1f%%', startangle=140)
            st.pyplot(fig2)
            
        if not thefts.empty:
            st.error("📋 Detailed Theft Evidence Report")
            final_report = thefts[['Date', 'Time', 'Voltage', 'Current', 'Power_kW']].reset_index(drop=True)
            st.dataframe(final_report, use_container_width=True)
            
            csv_out = final_report.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Report as CSV", csv_out, "Theft_Evidence.csv", "text/csv")
        else:
            st.success("✅ Analysis Complete: No significant anomalies detected.")

# --- 7. LIVE FIELD CHECK PAGE ---
elif menu == "📱 Live Field Check":
    st.title("Smart Power Theft Detection System")
    st.subheader("📱 Field Staff Verification Tool")
    
    with st.container():
        v = st.number_input("Meter Voltage (V)", value=230.0)
        i = st.number_input("Meter Current (A)", value=0.0, step=0.1)
        load = st.selectbox("Customer Load Type", ["Residential", "Commercial", "Industrial"])
        
        if st.button("Run Instant Audit", type="primary"):
            p = (v * i) / 1000
            st.write(f"### Current Load: `{p:.3f} kW`")
            
            if p < 0.05 and i < 0.3:
                st.error("🚨 RESULT: THEFT DETECTED. Reading is too low for active line (Bypass Suspected).")
            elif p > 10.0 and load == "Residential":
                st.warning("⚠️ RESULT: ABNORMAL SPIKE. Possible direct hooking or unauthorized heavy load.")
            else:
                st.success("✅ RESULT: NORMAL. Reading is within approved parameters.")
