import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Smart Power Theft Detection System", page_icon="⚡", layout="wide")

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("⚡ PowerAdmin")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "⚙️ Generate Dummy Data", "📊 Upload & Analyze", "📱 Live Field Check"])

# --- 3. HOME PAGE ---
if menu == "🏠 Home":
    st.title("Smart Power Theft Detection System")
    st.write("""
    Welcome to the automated theft detection dashboard. 
    This system uses **Machine Learning (Isolation Forest)** to analyze consumer power consumption 
    patterns and flag unnatural drops or spikes that indicate meter bypassing or tampering.
    """)
    st.info("System Status: Online | ML Engine: Active | Version: 3.0 Ultimate")

# --- 4. GENERATE DUMMY DATA PAGE (UPGRADED) ---
elif menu == "⚙️ Generate Dummy Data":
    st.title("⚙️ Generate Smart Meter Dataset")
    st.write("Use this tool to generate 30 days of hourly power consumption data. Select a specific theft scenario to test the ML engine.")
    
    # NEW: Dropdown for multiple conditions
    theft_scenario = st.selectbox(
        "Select Theft Condition to Inject:",
        [
            "1. Evening Meter Bypass (Drops to 0 during peak hours)",
            "2. Direct Hooking / Overload (Massive 10kW spikes)",
            "3. Meter Tampering (Flatline low usage all day)",
            "4. Clean Data (Normal Usage Only)"
        ]
    )
    
    if st.button("Generate CSV Dataset", type="primary"):
        # Generate 30 days of hourly data
        date_rng = pd.date_range(start='2026-01-01', end='2026-01-30 23:00:00', freq='h')
        df = pd.DataFrame(date_rng, columns=['Datetime'])
        df['Date'] = df['Datetime'].dt.date
        df['Time'] = df['Datetime'].dt.time
        
        # Simulate normal residential usage
        np.random.seed(42)
        base_power = np.random.normal(1.5, 0.3, len(df)) 
        
        # Add evening peak loads (6 PM to 10 PM)
        evening_mask = (df['Datetime'].dt.hour >= 18) & (df['Datetime'].dt.hour <= 22)
        base_power[evening_mask] += np.random.normal(2.0, 0.5, evening_mask.sum())
        
        df['Voltage'] = np.random.normal(230, 2, len(df)) 
        df['Current'] = (base_power * 1000) / df['Voltage']
        df['Power_kW'] = base_power
        
        # --- APPLY MULTIPLE CONDITIONS BASED ON USER CHOICE ---
        
        # Condition 1: Evening Bypass
        if "Evening Meter Bypass" in theft_scenario:
            theft_days = ['2026-01-15', '2026-01-16', '2026-01-17']
            for day in theft_days:
                theft_mask = (df['Date'].astype(str) == day) & (df['Datetime'].dt.hour >= 19) & (df['Datetime'].dt.hour <= 21)
                df.loc[theft_mask, 'Power_kW'] = np.random.uniform(0.01, 0.05)
                
        # Condition 2: Direct Hooking (Massive Spikes)
        elif "Direct Hooking" in theft_scenario:
            spike_days = ['2026-01-10', '2026-01-20']
            for day in spike_days:
                spike_mask = (df['Date'].astype(str) == day) & (df['Datetime'].dt.hour >= 12) & (df['Datetime'].dt.hour <= 15)
                df.loc[spike_mask, 'Power_kW'] = np.random.uniform(9.0, 12.0) # Illegal heavy machinery
                
        # Condition 3: Constant Tampering (Flatline)
        elif "Meter Tampering" in theft_scenario:
            tamper_days = ['2026-01-25', '2026-01-26', '2026-01-27']
            for day in tamper_days:
                tamper_mask = (df['Date'].astype(str) == day)
                df.loc[tamper_mask, 'Power_kW'] = 0.1 # Flat flatline reading all day and night
                
        # (Condition 4 does nothing, leaves data clean)

        # Recalculate current based on new power
        df['Current'] = (df['Power_kW'] * 1000) / df['Voltage']
        df = df[['Date', 'Time', 'Voltage', 'Current', 'Power_kW']]
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.success(f"Dataset generated successfully with scenario: {theft_scenario.split('(')[0]}")
        st.download_button(label="⬇️ Download meter_data.csv", data=csv, file_name='meter_data.csv', mime='text/csv')
        
        with st.expander("Preview Generated Data"):
            st.dataframe(df.head(15))

# --- 5. UPLOAD & ANALYZE PAGE ---
elif menu == "📊 Upload & Analyze":
    st.title("Smart Power Theft Detection System")
    st.subheader("📊 Bulk Analysis Dashboard")
    st.write("Upload a consumer's historical meter data (CSV) to run the ML Anomaly Detection engine.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        if 'Power_kW' not in data.columns:
            data['Power_kW'] = (data['Voltage'] * data['Current']) / 1000
            
        st.write("### 🧠 ML Analysis Results")
        with st.spinner('Running Isolation Forest algorithm...'):
            # Contamination increased slightly to catch the big spikes and flatlines
            model = IsolationForest(contamination=0.06, random_state=42)
            data['Anomaly_Score'] = model.fit_predict(data[['Power_kW']])
            
        theft_records = data[data['Anomaly_Score'] == -1]
        
        tariff_per_unit = 7.0 
        estimated_loss = len(theft_records) * 2.5 * tariff_per_unit 
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Readings", len(data))
        col2.metric("Theft Alerts 🚨", len(theft_records), delta_color="inverse")
        col3.metric("Est. Revenue Saved", f"₹ {estimated_loss:.2f}")
        
        st.markdown("---")
        
        col_graph, col_pie = st.columns([2, 1])
        
        with col_graph:
            st.write("### 📈 Consumption Pattern Graph")
            fig, ax = plt.subplots(figsize=(10, 5))
            data['Datetime_Plot'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
            theft_records['Datetime_Plot'] = pd.to_datetime(theft_records['Date'] + ' ' + theft_records['Time'])
            
            ax.plot(data['Datetime_Plot'], data['Power_kW'], label='Normal Power (kW)', color='#1f77b4', linewidth=1)
            ax.scatter(theft_records['Datetime_Plot'], theft_records['Power_kW'], color='red', label='🚨 Anomaly/Theft', zorder=5)
            ax.set_ylabel('Power (kW)')
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col_pie:
            st.write("### 🥧 Distribution")
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            sizes = [len(data) - len(theft_records), len(theft_records)]
            labels = ['Normal', 'Anomaly']
            colors = ['#2ca02c', '#d62728']
            ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            st.pyplot(fig_pie)
        
        if not theft_records.empty:
            st.error(f"⚠️ {len(theft_records)} suspicious reading(s) found! See details below:")
            report_df = theft_records[['Date', 'Time', 'Voltage', 'Current', 'Power_kW']].reset_index(drop=True)
            st.dataframe(report_df, use_container_width=True)
            
            csv_report = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Export Final Theft Report (CSV)", data=csv_report, file_name='Theft_Report.csv', mime='text/csv')
        else:
            st.success("✅ No suspicious patterns detected in this dataset.")

# --- 6. LIVE FIELD CHECK PAGE ---
elif menu == "📱 Live Field Check":
    st.title("Smart Power Theft Detection System")
    st.subheader("📱 Live Field Check (Lineman App)")
    st.write("Manually verify real-time meter readings against expected thresholds.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        voltage = st.number_input("Recorded Voltage (V)", min_value=0.0, max_value=500.0, value=230.0)
    with col2:
        current = st.number_input("Recorded Current (A)", min_value=0.0, max_value=100.0, value=0.0)
        
    expected_load = st.selectbox("Sanctioned Load Type", ["1 BHK (Low Load)", "2 BHK (Medium Load)", "Commercial (High Load)"])
    
    if st.button("Calculate & Verify Status", type="primary"):
        power_kw = (voltage * current) / 1000
        st.write(f"### Calculated Power: **{power_kw:.3f} kW**")
        
        if expected_load == "2 BHK (Medium Load)" and power_kw < 0.1:
            st.error("🚨 ALERT: Extremely low consumption for a 2BHK. Possible meter bypass!")
        elif expected_load == "1 BHK (Low Load)" and power_kw > 5.0:
            st.warning("⚠️ WARNING: Load exceeds normal capacity. Possible direct hooking/overload.")
        elif current == 0 and voltage > 0:
            st.error("🚨 ALERT: Zero current flowing while voltage is active. Strong indicator of tampering.")
        else:
            st.success("✅ Status: Normal. Reading falls within acceptable parameters.")