import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import random
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Power Theft Detection System", page_icon="⚡", layout="wide")

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("⚡ PowerAdmin v3.0")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "⚙️ Generate Dynamic Data", "📊 Upload & Analyze", "📱 Live Field Check"])

# --- 3. HOME PAGE ---
if menu == "🏠 Home":
    st.title("Smart Power Theft Detection System")
    st.write("""
    Welcome to the automated theft detection dashboard. 
    This system uses **Machine Learning (Isolation Forest)** to analyze consumer power consumption 
    patterns and flag unnatural drops or spikes that indicate meter bypassing or tampering.
    """)
    st.info("System Status: Online | ML Engine: Active | Version: 3.0 Ultimate")
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933245.png", width=100)

# --- 4. GENERATE DYNAMIC DATA PAGE ---
elif menu == "⚙️ Generate Dynamic Data":
    st.title("Smart Power Theft Detection System")
    st.subheader("⚙️ Generate Dynamic Test Dataset")
    st.write("Use this tool to generate 30 days of hourly power consumption data. The theft events are randomized so the results are never the same!")
    
    theft_scenario = st.selectbox(
        "Select Test Condition to Inject:",
        ["Evening Meter Bypass", "Direct Hooking (Spikes)", "Meter Tampering", "Clean Data"]
    )
    
    if st.button("Generate Randomized CSV", type="primary"):
        # Use current time as seed to ensure unique results every click
        np.random.seed(int(time.time()))
        
        date_rng = pd.date_range(start='2026-01-01', end='2026-01-30 23:00:00', freq='h')
        df = pd.DataFrame(date_rng, columns=['Datetime'])
        df['Date'] = df['Datetime'].dt.date
        df['Time'] = df['Datetime'].dt.time
        
        # Base normal usage
        base_power = np.random.normal(1.5, 0.3, len(df))
        # Evening peak loads
        evening_mask = (df['Datetime'].dt.hour >= 18) & (df['Datetime'].dt.hour <= 22)
        base_power[evening_mask] += np.random.normal(2.0, 0.5, evening_mask.sum())
        
        df['Voltage'] = np.random.normal(230, 2, len(df)) 
        df['Power_kW'] = base_power
        
        # PICK 3 RANDOM DAYS FOR THEFT
        all_days = df['Date'].unique()
        random_theft_days = random.sample(list(all_days), 3)
        
        # INJECT SCENARIOS
        if "Evening Meter Bypass" in theft_scenario:
            for day in random_theft_days:
                start_h = random.randint(18, 20)
                mask = (df['Date'] == day) & (df['Datetime'].dt.hour >= start_h) & (df['Datetime'].dt.hour <= start_h + 2)
                df.loc[mask, 'Power_kW'] = np.random.uniform(0.01, 0.08)
                
        elif "Direct Hooking" in theft_scenario:
            for day in random_theft_days:
                start_h = random.randint(10, 16)
                mask = (df['Date'] == day) & (df['Datetime'].dt.hour == start_h)
                df.loc[mask, 'Power_kW'] = np.random.uniform(9.0, 13.0)
                
        elif "Meter Tampering" in theft_scenario:
            for day in random_theft_days:
                mask = (df['Date'] == day)
                df.loc[mask, 'Power_kW'] = 0.1 # Constant low value
        
        # Recalculate current
        df['Current'] = (df['Power_kW'] * 1000) / df['Voltage']
        df = df[['Date', 'Time', 'Voltage', 'Current', 'Power_kW']]
        
        st.success(f"Generated data for: {theft_scenario}")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Dynamic_Meter_Data.csv", csv, "meter_data.csv", "text/csv")

# --- 5. UPLOAD & ANALYZE PAGE ---
elif menu == "📊 Upload & Analyze":
    st.title("Smart Power Theft Detection System")
    st.subheader("📊 Bulk Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        if 'Power_kW' not in data.columns:
            data['Power_kW'] = (data['Voltage'] * data['Current']) / 1000
            
        st.write("### 🧠 ML Processing...")
        model = IsolationForest(contamination=0.06, random_state=42)
        data['Anomaly_Score'] = model.fit_predict(data[['Power_kW']])
        theft_records = data[data['Anomaly_Score'] == -1]
        
        # Metrics
        tariff = 7.0 # Rs per unit
        loss = len(theft_records) * 2.5 * tariff
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Readings", len(data))
        c2.metric("Theft Alerts 🚨", len(theft_records))
        c3.metric("Est. Revenue Saved", f"₹{loss:.2f}")
        
        # Charts
        col_graph, col_pie = st.columns([2, 1])
        with col_graph:
            st.write("### 📈 Consumption Pattern")
            data['DT'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['DT'], data['Power_kW'], label='Usage', color='#1f77b4')
            anom = data[data['Anomaly_Score'] == -1]
            ax.scatter(pd.to_datetime(anom['Date'] + ' ' + anom['Time']), anom['Power_kW'], color='red', label='Theft')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col_pie:
            st.write("### 🥧 Distribution")
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie([len(data)-len(theft_records), len(theft_records)], labels=['Normal', 'Theft'], colors=['green', 'red'], autopct='%1.1f%%')
            st.pyplot(fig_pie)
            
        if not theft_records.empty:
            st.error("Evidence Report:")
            report = theft_records[['Date', 'Time', 'Power_kW']].reset_index(drop=True)
            st.dataframe(report, use_container_width=True)
            csv_rep = report.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Theft Report", csv_rep, "Evidence_Report.csv", "text/csv")

# --- 6. LIVE FIELD CHECK PAGE ---
elif menu == "📱 Live Field Check":
    st.title("Smart Power Theft Detection System")
    st.subheader("📱 Field Verification App")
    v = st.number_input("Voltage", value=230.0)
    i = st.number_input("Current", value=0.0)
    if st.button("Check Status"):
        p = (v * i) / 1000
        st.write(f"Power: {p:.3f} kW")
        if p < 0.1 and i < 0.5:
            st.error("🚨 ALERT: Meter Bypass Detected!")
        else:
            st.success("✅ Status: Normal")
