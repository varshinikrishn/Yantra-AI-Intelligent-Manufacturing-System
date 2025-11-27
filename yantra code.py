import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import openai
import os
import time

# Set page configuration
st.set_page_config(page_title="Yantra InsightGen", layout="wide")

# Define the Modern Blue & Teal color palette
colors = {
    "Primary": "#0077b6",      # Deep Blue
    "Secondary": "#90e0ef",    # Light Teal
    "Accent": "#00b4d8",       # Bright Teal
    "Background": "#f5f5f5",   # Light Grey
    "Text": "#023e8a",         # Dark Blue
    "Highlight": "#48cae4",    # Light Teal
    "Success": "#28a745",      # Green for Success Messages
    "Warning": "#ffc107",      # Yellow for Warnings
    "Danger": "#dc3545",       # Red for Errors
    "LightGray": "#6c757d"     # Light Gray for borders, etc.
}

# Example usage in a Streamlit component
st.markdown(f"""
    <style>
    body {{
        background-color: {colors['Background']};
        font-family: 'Roboto', sans-serif;
    }}
    .stButton>button {{
        background-color: {colors['Primary']};
        color: white;
        border-radius: 4px;
    }}
    .stTextInput>div>div>input {{
        border-radius: 4px;
        border: 1px solid {colors['LightGray']};
        padding: 10px;
        font-size: 16px;
        color: {colors['Text']};
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {colors['Primary']};
    }}
    .stMarkdown p {{
        color: {colors['Text']};
    }}
    .widget-container {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }}
    .widget {{
        background-color: {colors['Secondary']};
        padding: 10px;
        border-radius: 8px;
        color: {colors['Text']};
        font-weight: bold;
    }}
    .conversation-history {{
        background-color: {colors['Background']};
        padding: 15px;
        border-radius: 10px;
        border: 1px solid {colors['LightGray']};
        color: {colors['Text']};
    }}
    .user-bubble {{
        background-color: {colors['LightGray']};
        color: {colors['Text']};
        border-radius: 15px;
        padding: 10px;
    }}
    .yantra-bubble {{
        background-color: {colors['Accent']};
        color: white;
        border-radius: 15px;
        padding: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Yantra InsightGen")
st.sidebar.markdown("### Navigation")
sections = ["Dashboard Overview", "Predictive Maintenance", "Anomaly Detection", 
            "Process Optimization", "Energy Management", "Virtual Assistant"]
selected_section = st.sidebar.radio("Go to", sections)

# Simulated data for EV battery manufacturing mirroring XRIT
def generate_ev_battery_data(days=30):
    date_rng = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='H')
    data = {
        'Timestamp': date_rng,
        'MachineID': np.random.choice(['Raw Material Feeder', 'Mixer', 'Coating Machine', 'Cutting Machine', 
                                       'Stacking/Winding Machine', 'Electrolyte Filling Machine', 'Sealing Machine', 
                                       'Formation Equipment', 'Aging Chamber', 'Electrical Testing', 
                                       'Thermal Imaging', 'X-Ray Inspection', 'Packaging Machine', 'AGV', 'Storage Rack'], 
                                      size=len(date_rng)),
        'MaterialFeedRate': np.random.normal(500, 50, size=len(date_rng)),  # kg/min
        'MixingSpeed': np.random.normal(1500, 100, size=len(date_rng)),  # rpm
        'CoatingThickness': np.random.normal(100, 5, size=len(date_rng)),  # µm
        'CuttingPrecision': np.random.normal(0.01, 0.001, size=len(date_rng)),  # mm
        'WindingSpeed': np.random.normal(200, 20, size=len(date_rng)),  # rpm
        'ElectrolyteVolume': np.random.normal(5, 0.5, size=len(date_rng)),  # ml
        'SealingTemperature': np.random.normal(150, 5, size=len(date_rng)),  # °C
        'FormationVoltage': np.random.normal(3.7, 0.1, size=len(date_rng)),  # V
        'FormationCurrent': np.random.normal(1.5, 0.1, size=len(date_rng)),  # A
        'AgingTemperature': np.random.normal(25, 2, size=len(date_rng)),  # °C
        'AgingDuration': np.random.normal(48, 2, size=len(date_rng)),  # hours
        'Capacity': np.random.normal(3.0, 0.1, size=len(date_rng)),  # Ah
        'Voltage': np.random.normal(3.7, 0.1, size=len(date_rng)),  # V
        'Impedance': np.random.normal(0.005, 0.0005, size=len(date_rng)),  # Ohms
        'VibrationLevel': np.random.normal(0.5, 0.1, size=len(date_rng)),  # Vibration level for anomaly detection
        'Temperature': np.random.normal(75, 10, size=len(date_rng)),  # Temperature for anomaly detection
        'Pressure': np.random.normal(100, 5, size=len(date_rng)),  # Pressure for anomaly detection
        'ThermalAnomaly': np.random.choice([True, False], size=len(date_rng), p=[0.01, 0.99]),
        'XrayDefects': np.random.poisson(0.1, size=len(date_rng)),
        'PackagingRate': np.random.normal(60, 5, size=len(date_rng)),  # units/min
        'AGVLoad': np.random.normal(500, 50, size=len(date_rng)),  # kg
        'StorageTemperature': np.random.normal(20, 2, size=len(date_rng)),  # °C
        'StorageHumidity': np.random.normal(40, 5, size=len(date_rng)),  # %
    }
    return pd.DataFrame(data)

# Simulated data
df = generate_ev_battery_data()

# Define color palette
colors = {
    "Primary": "#007bff",
    "Secondary": "#f7f7f7",
    "Gray": "#e0e0e0",
    "DarkGray": "#333333",
    "Green": "#28a745",
    "Yellow": "#ffc107",
    "Red": "#dc3545",
    "LightGray": "#6c757d"
}

# Section: Dashboard Overview
def dashboard_overview():
    st.markdown(f"<h1 style='color:{colors['Primary']}; text-align:center;'>Dashboard Overview</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Real-Time Data Stream
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Real-Time Data Stream</h2>", unsafe_allow_html=True)
    st.dataframe(df.tail(10))
    
    # KPIs
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Key Performance Indicators (KPIs)</h2>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Production Rate", value=f"{np.random.randint(80, 100)}%")
    kpi2.metric(label="Downtime", value=f"{np.random.randint(5, 20)}%")
    kpi3.metric(label="Energy Consumption", value=f"{np.random.randint(200, 500)} kWh")
    kpi4.metric(label="Maintenance Status", value="Stable", delta="+1.2%")
    
    # Customizable Data Filtering and Sorting
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Customizable Data Filtering and Sorting</h2>", unsafe_allow_html=True)
    columns = df.columns.tolist()
    filter_column = st.selectbox("Select Column to Filter", columns)
    filter_value = st.text_input("Enter Value to Filter by")
    sort_column = st.selectbox("Select Column to Sort", columns)
    sort_order = st.radio("Sort Order", ['Ascending', 'Descending'])

    filtered_df = df[df[filter_column].astype(str).str.contains(filter_value, na=False)]
    sorted_df = filtered_df.sort_values(by=sort_column, ascending=(sort_order == 'Ascending'))

    st.dataframe(sorted_df)
    
    # Real-Time Alerts and Notifications
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Real-Time Alerts and Notifications</h2>", unsafe_allow_html=True)
    alert_thresholds = {
        'FormationVoltage': 3.8,
        'AgingTemperature': 28.0
    }

    alerts = []
    for index, row in df.tail(10).iterrows():
        if row['FormationVoltage'] > alert_thresholds['FormationVoltage']:
            alerts.append(f"Alert: FormationVoltage too high at {row['FormationVoltage']}V on Machine {row['MachineID']}")
        if row['AgingTemperature'] > alert_thresholds['AgingTemperature']:
            alerts.append(f"Alert: AgingTemperature too high at {row['AgingTemperature']}°C on Machine {row['MachineID']}")

    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("No alerts in the last 10 entries.")
    
    # Historical Data Comparison
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Historical Data Comparison</h2>", unsafe_allow_html=True)
    compare_column = st.selectbox("Select Column for Historical Comparison", columns)
    comparison_period = st.slider("Select Number of Days for Comparison", min_value=1, max_value=30, value=7)

    historical_df = df[df['Timestamp'] >= datetime.now() - timedelta(days=comparison_period)]
    fig = px.line(historical_df, x='Timestamp', y=compare_column, title=f'{compare_column} Over Last {comparison_period} Days')
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive Charts and Graphs
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Interactive Data Visualization</h2>", unsafe_allow_html=True)
    visualization_type = st.selectbox("Select Visualization Type", ['Line Chart', 'Bar Chart', 'Heatmap'])
    visualization_column = st.selectbox("Select Column to Visualize", columns)

    if visualization_type == 'Line Chart':
        fig = px.line(df, x='Timestamp', y=visualization_column, title=f'{visualization_column} Line Chart')
    elif visualization_type == 'Bar Chart':
        fig = px.bar(df, x='Timestamp', y=visualization_column, title=f'{visualization_column} Bar Chart')
    elif visualization_type == 'Heatmap':
        fig = px.density_heatmap(df, x='Timestamp', y=visualization_column, title=f'{visualization_column} Heatmap')

    st.plotly_chart(fig, use_container_width=True)

# Section: Predictive Maintenance
def predictive_maintenance():
    st.markdown(f"<h1 style='color:{colors['Primary']}; text-align:center;'>Predictive Maintenance</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Train Predictive Maintenance Model
    X = df[['FormationVoltage', 'FormationCurrent', 'AgingTemperature', 'AgingDuration', 'Capacity']]
    y = df['ThermalAnomaly']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Machine Health Overview
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Machine Health Overview</h2>", unsafe_allow_html=True)
    st.dataframe(df[['Timestamp', 'MachineID', 'FormationVoltage', 'FormationCurrent']].tail(10))
    
    # Maintenance Schedule
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Maintenance Schedule</h2>", unsafe_allow_html=True)
    st.table(df[df['ThermalAnomaly'] == True][['Timestamp', 'MachineID']])
    
    # Predictive Maintenance Insights
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Predictive Maintenance Insights</h2>", unsafe_allow_html=True)
    maintenance_predictions = model.predict(X)
    df['PredictedAnomaly'] = maintenance_predictions
    maintenance_alerts = df[df['PredictedAnomaly'] == 1].tail(10)
    st.table(maintenance_alerts[['Timestamp', 'MachineID', 'PredictedAnomaly']])

    if not maintenance_alerts.empty:
        st.error("Predictive Maintenance Required for Machines in the Table Above")
    else:
        st.success("No immediate maintenance required based on current predictions.")

    fig = px.histogram(df, x='FormationVoltage', title='Formation Voltage Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Actionable Insights
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Actionable Insights</h2>", unsafe_allow_html=True)
    actionable_data = {
        'Task': ['Check Voltage Stability', 'Replace Heater', 'Inspect Insulation', 'Calibrate Sensors'],
        'Priority': ['High', 'Medium', 'Low', 'Medium'],
        'Reason': ['Voltage Fluctuation', 'Heater Failure', 'Insulation Weakness', 'Sensor Drift'],
        'MachineID': ['Formation Equipment', 'Sealing Machine', 'Aging Chamber', 'Electrolyte Filling Machine'],
        'Scheduled Date': [datetime.now() + timedelta(days=i) for i in range(1, 5)]
    }
    actionable_df = pd.DataFrame(actionable_data)
    st.table(actionable_df)
    
    # Real-Time Predictive Maintenance Alerts
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Real-Time Predictive Maintenance Alerts</h2>", unsafe_allow_html=True)
    maintenance_alerts = df[df['PredictedAnomaly'] == 1].tail(10)
    if not maintenance_alerts.empty:
        for i, row in maintenance_alerts.iterrows():
            st.error(f"Maintenance Alert: {row['MachineID']} requires maintenance (Predicted Anomaly)")
    else:
        st.success("No predictive maintenance alerts at this time.")

    # Historical Maintenance Analysis
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Historical Maintenance Analysis</h2>", unsafe_allow_html=True)
    maintenance_history_period = st.slider("Select History Period (Days)", 1, 30, 7)
    historical_maintenance_df = df[df['Timestamp'] >= datetime.now() - timedelta(days=maintenance_history_period)]
    fig = px.line(historical_maintenance_df, x='Timestamp', y='PredictedAnomaly', title=f'Historical Maintenance Analysis for Last {maintenance_history_period} Days')
    st.plotly_chart(fig, use_container_width=True)

    # Maintenance Priority and Impact Assessment
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Maintenance Priority and Impact Assessment</h2>", unsafe_allow_html=True)
    priority_map = {
        'Critical': '🔴 Critical Impact',
        'High': '🟠 High Impact',
        'Medium': '🟡 Medium Impact',
        'Low': '🟢 Low Impact'
    }

    maintenance_alerts['Priority'] = np.where(maintenance_alerts['FormationVoltage'] > 3.8, 'Critical', 
                                              np.where(maintenance_alerts['FormationVoltage'] > 3.6, 'High', 
                                                       np.where(maintenance_alerts['FormationVoltage'] > 3.5, 'Medium', 'Low')))

    maintenance_alerts['Impact'] = maintenance_alerts['Priority'].map(priority_map)

    st.table(maintenance_alerts[['Timestamp', 'MachineID', 'PredictedAnomaly', 'Priority', 'Impact']])

    # Integration with Spare Parts Inventory
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Spare Parts Inventory</h2>", unsafe_allow_html=True)
    spare_parts_inventory = {'Formation Equipment': 3, 'Sealing Machine': 1, 'Aging Chamber': 0}

    for machine in maintenance_alerts['MachineID'].unique():
        if spare_parts_inventory.get(machine, 0) <= 0:
            st.warning(f"Warning: Spare parts for {machine} are out of stock. Please reorder.")
        else:
            st.info(f"{spare_parts_inventory[machine]} spare parts available for {machine}.")

    # Maintenance Task Assignment and Tracking
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Maintenance Task Assignment and Tracking</h2>", unsafe_allow_html=True)
    task_assignees = ['Technician A', 'Technician B', 'Technician C']
    maintenance_alerts['AssignedTo'] = st.selectbox("Assign Task to", task_assignees)

    for i, row in maintenance_alerts.iterrows():
        st.markdown(f"Task for {row['MachineID']} (Anomaly detected): Assigned to {row['AssignedTo']}")

    if st.button("Mark Task as Completed"):
        st.success("Task marked as completed.")

# Section: Anomaly Detection and Quality Control
def anomaly_detection():
    st.markdown(f"<h1 style='color:{colors['Primary']}; text-align:center;'>Anomaly Detection and Quality Control</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Train Anomaly Detection Model
    X_anomaly = df[['VibrationLevel', 'Temperature', 'Pressure']]
    anomaly_model = IsolationForest(contamination=0.05)
    anomaly_model.fit(X_anomaly)
    df['Anomaly'] = anomaly_model.predict(X_anomaly)
    df['Anomaly'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

    # Anomaly Alerts
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Anomaly Alerts</h2>", unsafe_allow_html=True)
    st.table(df[df['Anomaly'] == 'Anomaly'][['Timestamp', 'MachineID', 'Anomaly']].tail(10))
    
    # Quality Control Dashboard
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Quality Control Dashboard</h2>", unsafe_allow_html=True)
    fig = px.box(df, x='MachineID', y='Capacity', title='Quality Metrics')
    st.plotly_chart(fig, use_container_width=True)
    
    # Root Cause Analysis
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Root Cause Analysis</h2>", unsafe_allow_html=True)
    root_cause_data = {
        'Anomaly': ['Voltage Instability', 'Thermal Runaway', 'Electrolyte Leakage', 'Mechanical Failure'],
        'Possible Causes': [
            'Power Supply Issues, Calibration Errors, Environmental Conditions',
            'Overcharging, External Short, Manufacturing Defects',
            'Seal Failure, Excessive Pressure, Contamination',
            'Wear and Tear, Overuse, Poor Maintenance'
        ],
        'MachineID': ['Formation Equipment', 'Thermal Imaging', 'Electrolyte Filling Machine', 'AGV']
    }
    root_cause_df = pd.DataFrame(root_cause_data)
    st.table(root_cause_df)
    
    # Corrective Actions
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Corrective Actions</h2>", unsafe_allow_html=True)
    corrective_actions_data = {
        'Action': ['Stabilize Voltage Supply', 'Check and Repair Insulation', 'Replace Damaged Seals', 'Perform Routine Maintenance'],
        'Priority': ['High', 'Medium', 'High', 'Low'],
        'MachineID': ['Formation Equipment', 'Thermal Imaging', 'Electrolyte Filling Machine', 'AGV']
    }
    corrective_actions_df = pd.DataFrame(corrective_actions_data)
    st.table(corrective_actions_df)
    
    # Automated Anomaly Classification and Severity Ranking
    def classify_anomaly(row):
        if row['Temperature'] > 85 or row['Pressure'] > 110:
            return "Severe"
        elif row['Temperature'] > 80 or row['Pressure'] > 105:
            return "Moderate"
        else:
            return "Minor"

    df['AnomalySeverity'] = df.apply(classify_anomaly, axis=1)

    st.markdown(f"<h2 style='color:{colors['Primary']};'>Anomaly Classification and Severity Ranking</h2>", unsafe_allow_html=True)
    st.table(df[['Timestamp', 'MachineID', 'Anomaly', 'AnomalySeverity']].tail(10))

    # Real-Time Anomaly Notification System
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Real-Time Anomaly Notifications</h2>", unsafe_allow_html=True)
    for index, row in df[df['Anomaly'] == 'Anomaly'].tail(10).iterrows():
        st.error(f"Notification: {row['MachineID']} has detected an anomaly of {row['AnomalySeverity']} severity at {row['Timestamp']}.")

    # Historical Anomaly Trend Analysis
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Historical Anomaly Trend Analysis</h2>", unsafe_allow_html=True)
    anomaly_trend_df = df.groupby([pd.Grouper(key='Timestamp', freq='D'), 'AnomalySeverity']).size().reset_index(name='Count')

    fig = px.line(anomaly_trend_df, x='Timestamp', y='Count', color='AnomalySeverity', title='Anomaly Trend Over Time')
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Root Cause Exploration Tool
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Interactive Root Cause Exploration</h2>", unsafe_allow_html=True)
    selected_machine = st.selectbox("Select Machine", df['MachineID'].unique())
    selected_severity = st.selectbox("Select Anomaly Severity", ['Severe', 'Moderate', 'Minor'])

    filtered_root_cause_df = df[(df['MachineID'] == selected_machine) & (df['AnomalySeverity'] == selected_severity)]
    st.table(filtered_root_cause_df[['Timestamp', 'MachineID', 'Anomaly', 'AnomalySeverity']])

    # Predictive Quality Control Insights
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Predictive Quality Control Insights</h2>", unsafe_allow_html=True)

    # Placeholder model for prediction - replace with actual predictive model
    def predict_quality_issues(df):
        return np.random.choice(['Stable', 'Warning', 'Critical'], size=len(df))

    df['PredictedQuality'] = predict_quality_issues(df)

    st.table(df[['Timestamp', 'MachineID', 'Capacity', 'PredictedQuality']].tail(10))

# Section: Process Optimization
def process_optimization():
    st.markdown(f"<h1 style='color:{colors['Primary']}; text-align:center;'>Process Optimization</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Process Flow Visualization
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Process Flow Visualization</h2>", unsafe_allow_html=True)
    process_flow_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Raw Material Feed", "Mixing", "Coating", "Cutting", "Winding", "Electrolyte Filling", "Sealing", "Formation", "Aging", "Testing", "Packaging"],
            color=[colors['Primary'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Gray'], colors['Primary']]
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=[8, 6, 4, 3, 5, 6, 4, 3, 2, 1]
        )
    ))
    process_flow_fig.update_layout(title_text="Process Flow Visualization", font_size=10)
    st.plotly_chart(process_flow_fig, use_container_width=True)
    
    # Optimization Suggestions
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Optimization Suggestions</h2>", unsafe_allow_html=True)
    optimization_suggestions = {
        'Suggestion': ['Optimize Mixing Speed', 'Improve Coating Thickness Control', 'Enhance Cutting Precision'],
        'Potential Benefit': ['10% Increased Efficiency', '5% Material Savings', '15% Improved Quality']
    }
    optimization_df = pd.DataFrame(optimization_suggestions)
    st.table(optimization_df)
    
    # Scenario Simulation
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Scenario Simulation</h2>", unsafe_allow_html=True)
    scenario_simulation_fig = go.Figure()
    for scenario in ['Current', 'Optimized']:
        scenario_simulation_fig.add_trace(go.Scatter(
            x=df['Timestamp'], 
            y=np.random.normal(100, 10, size=len(df)),
            mode='lines',
            name=scenario
        ))
    scenario_simulation_fig.update_layout(title="Scenario Simulation", xaxis_title="Time", yaxis_title="Efficiency")
    st.plotly_chart(scenario_simulation_fig, use_container_width=True)
    
    # Performance Comparison
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Performance Comparison</h2>", unsafe_allow_html=True)
    fig = px.line(df, x='Timestamp', y='MaterialFeedRate', title='Performance Over Time')
    st.plotly_chart(fig, use_container_width=True)

    # Dynamic Parameter Adjustment for Simulation
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Dynamic Parameter Adjustment</h2>", unsafe_allow_html=True)

    mixing_speed = st.slider("Adjust Mixing Speed (rpm)", min_value=1000, max_value=2000, value=1500, step=50)
    coating_thickness = st.slider("Adjust Coating Thickness (µm)", min_value=50, max_value=150, value=100, step=5)
    cutting_precision = st.slider("Adjust Cutting Precision (mm)", min_value=0.005, max_value=0.02, value=0.01, step=0.001)

    # Simulate the impact of parameter adjustments
    adjusted_df = df.copy()
    adjusted_df['MixingSpeed'] = mixing_speed
    adjusted_df['CoatingThickness'] = coating_thickness
    adjusted_df['CuttingPrecision'] = cutting_precision

    st.write(f"Impact of Adjusted Parameters: Mixing Speed = {mixing_speed} rpm, Coating Thickness = {coating_thickness} µm, Cutting Precision = {cutting_precision} mm")

    # Real-Time Process Monitoring with Alerts
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Real-Time Process Monitoring</h2>", unsafe_allow_html=True)

    for index, row in df.tail(10).iterrows():
        if row['MixingSpeed'] > 1800 or row['CoatingThickness'] > 120 or row['CuttingPrecision'] > 0.015:
            st.error(f"Alert: {row['MachineID']} exceeds safe operational limits at {row['Timestamp']}")
        else:
            st.success(f"Normal Operation: {row['MachineID']} at {row['Timestamp']}")

    # Cost-Benefit Analysis of Optimization Suggestions
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Cost-Benefit Analysis</h2>", unsafe_allow_html=True)

    cost_savings = {"Optimize Mixing Speed": 5000, "Improve Coating Thickness Control": 3000, "Enhance Cutting Precision": 2000}
    optimization_df['CostBenefit'] = optimization_df['Suggestion'].map(cost_savings)

    st.table(optimization_df[['Suggestion', 'Potential Benefit', 'CostBenefit']])
    st.write("Total Potential Savings: $", sum(cost_savings.values()))

    # Historical Performance Comparison
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Historical Performance Comparison</h2>", unsafe_allow_html=True)

    comparison_period = st.slider("Select Period for Comparison (Days)", min_value=1, max_value=30, value=7)
    historical_df = df[df['Timestamp'] >= datetime.now() - timedelta(days=comparison_period)]

    fig = px.line(historical_df, x='Timestamp', y='MaterialFeedRate', title=f'Performance Comparison Over Last {comparison_period} Days')
    st.plotly_chart(fig, use_container_width=True)

    # Interactive Process Flow Customization
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Interactive Process Flow Customization</h2>", unsafe_allow_html=True)

    # Example of simple customization: Allow the user to include/exclude steps
    include_steps = st.multiselect("Select Process Steps to Include", ["Raw Material Feed", "Mixing", "Coating", "Cutting", "Winding", "Electrolyte Filling", "Sealing", "Formation", "Aging", "Testing", "Packaging"], default=["Raw Material Feed", "Mixing", "Coating", "Cutting", "Winding"])

    custom_process_flow_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=include_steps,
            color=[colors['Primary']] * len(include_steps)
        ),
        link=dict(
            source=list(range(len(include_steps) - 1)),
            target=list(range(1, len(include_steps))),
            value=[1] * (len(include_steps) - 1)
        )
    ))
    custom_process_flow_fig.update_layout(title_text="Customized Process Flow Visualization", font_size=10)
    st.plotly_chart(custom_process_flow_fig, use_container_width=True)

# Section: Energy Management
def energy_management():
    st.markdown(f"<h1 style='color:{colors['Primary']}; text-align:center;'>Energy Management</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Ensure the required columns exist
    if 'Grid' not in df.columns:
        df['Grid'] = np.random.normal(300, 20, len(df))
    if 'Renewable' not in df.columns:
        df['Renewable'] = np.random.normal(200, 15, len(df))
    if 'Battery' not in df.columns:
        df['Battery'] = np.random.normal(100, 10, len(df))
    
    # Energy Usage Dashboard
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Energy Usage Dashboard</h2>", unsafe_allow_html=True)
    fig = px.line(df, x='Timestamp', y='AGVLoad', title='Energy Usage Over Time')
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimization Recommendations
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Optimization Recommendations</h2>", unsafe_allow_html=True)
    optimization_recommendations = {
        'Recommendation': ['Upgrade to Energy-efficient Equipment', 'Optimize Production Schedules', 'Enhance HVAC Efficiency'],
        'Expected Savings': ['20%', '10%', '15%']
    }
    recommendations_df = pd.DataFrame(optimization_recommendations)
    st.table(recommendations_df)
    
    # Renewable Integration
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Renewable Integration</h2>", unsafe_allow_html=True)
    fig = px.pie(df, names='MachineID', values='MaterialFeedRate', title='Renewable Energy Usage')
    st.plotly_chart(fig, use_container_width=True)
    
    # **Enhanced Energy Consumption Forecasting with Random Forest Regressor**
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Energy Consumption Forecasting</h2>", unsafe_allow_html=True)

    # Feature selection for forecasting
    features = ['Grid', 'Renewable', 'Battery', 'AGVLoad']
    X = df[features]
    y = df['AGVLoad']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Forecasting
    df['ForecastedEnergy'] = rf_model.predict(X)

    st.line_chart(df[['Timestamp', 'AGVLoad', 'ForecastedEnergy']].set_index('Timestamp'))
    
    # **Real-Time Energy Efficiency Anomaly Detection with Isolation Forest**
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Real-Time Energy Efficiency Alerts</h2>", unsafe_allow_html=True)

    # Train Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly'] = isolation_forest.fit_predict(df[['AGVLoad']])

    # Alert for anomalies
    for index, row in df.tail(10).iterrows():
        if row['Anomaly'] == -1:
            st.error(f"Alert: Anomaly detected in energy efficiency on {row['Timestamp']} with AGVLoad of {row['AGVLoad']} kWh")
        else:
            st.success(f"Normal Efficiency: {row['Timestamp']} with AGVLoad of {row['AGVLoad']} kWh")
    
    # Calculate Carbon Footprint
    carbon_intensity = {'Grid': 0.5, 'Renewable': 0.05, 'Battery': 0.1}
    df['CarbonFootprint'] = (df['Grid'] * carbon_intensity['Grid'] +
                             df['Renewable'] * carbon_intensity['Renewable'] +
                             df['Battery'] * carbon_intensity['Battery'])

    # Interactive Detailed Breakdown of Energy Sources
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Detailed Breakdown of Energy Sources</h2>", unsafe_allow_html=True)
 
    fig_sources = go.Figure()

    for source in ['Grid', 'Renewable', 'Battery']:
       fig_sources.add_trace(go.Scatter(
        x=df['Timestamp'], 
        y=df[source],
        mode='lines',
        name=source
    ))

    fig_sources.update_layout(title="Energy Sources Over Time", xaxis_title="Time", yaxis_title="Energy (kWh)", hovermode="x unified")
    st.plotly_chart(fig_sources, use_container_width=True)

     # Carbon Footprint Tracking
    st.markdown(f"<h2 style='color:{colors['Primary']};'>Carbon Footprint Tracking</h2>", unsafe_allow_html=True)

    fig_carbon = go.Figure()

    fig_carbon.add_trace(go.Scatter(
    x=df['Timestamp'], 
    y=df['CarbonFootprint'],
    mode='lines',
    name='Carbon Footprint',
    line=dict(color='firebrick')
     ))

    fig_carbon.update_layout(title="Carbon Footprint Over Time", xaxis_title="Time", yaxis_title="Carbon Footprint (kg CO2)", hovermode="x unified")
    st.plotly_chart(fig_carbon, use_container_width=True)

# Initialize GPT-4 model using OpenAI API
openai.api_key = "sk-ANDNMYc7wS6b0xbCezW7JPDw8AFgPY3fGb1fW48VEgT3BlbkFJd1Bth16s4c3ycsoYsQ7z0M-2oGPhZ7SEF8ZAIglMsA"

# Function to determine if the input is a data query and return the result
def handle_data_query(user_input):
    response = None
    
    if "material feed rate" in user_input.lower():
        response = f"The current material feed rate is {df['MaterialFeedRate'].iloc[-1]:.2f} kg/min."
    elif "mixing speed" in user_input.lower():
        response = f"The current mixing speed is {df['MixingSpeed'].iloc[-1]:.2f} rpm."
    elif "coating thickness" in user_input.lower():
        response = f"The current coating thickness is {df['CoatingThickness'].iloc[-1]:.2f} µm."
    elif "cutting precision" in user_input.lower():
        response = f"The current cutting precision is {df['CuttingPrecision'].iloc[-1]:.4f} mm."
    elif "winding speed" in user_input.lower():
        response = f"The current winding speed is {df['WindingSpeed'].iloc[-1]:.2f} rpm."
    elif "electrolyte volume" in user_input.lower():
        response = f"The current electrolyte volume is {df['ElectrolyteVolume'].iloc[-1]:.2f} ml."
    elif "sealing temperature" in user_input.lower():
        response = f"The current sealing temperature is {df['SealingTemperature'].iloc[-1]:.2f} °C."
    elif "formation voltage" in user_input.lower():
        response = f"The current formation voltage is {df['FormationVoltage'].iloc[-1]:.2f} V."
    elif "formation current" in user_input.lower():
        response = f"The current formation current is {df['FormationCurrent'].iloc[-1]:.2f} A."
    elif "aging temperature" in user_input.lower():
        response = f"The current aging temperature is {df['AgingTemperature'].iloc[-1]:.2f} °C."
    elif "aging duration" in user_input.lower():
        response = f"The current aging duration is {df['AgingDuration'].iloc[-1]:.2f} hours."
    elif "capacity" in user_input.lower():
        response = f"The current capacity is {df['Capacity'].iloc[-1]:.2f} Ah."
    elif "voltage" in user_input.lower():
        response = f"The current voltage is {df['Voltage'].iloc[-1]:.2f} V."
    elif "impedance" in user_input.lower():
        response = f"The current impedance is {df['Impedance'].iloc[-1]:.5f} Ohms."
    elif "vibration level" in user_input.lower():
        response = f"The current vibration level is {df['VibrationLevel'].iloc[-1]:.2f}."
    elif "temperature" in user_input.lower():
        response = f"The current temperature is {df['Temperature'].iloc[-1]:.2f} °C."
    elif "pressure" in user_input.lower():
        response = f"The current pressure is {df['Pressure'].iloc[-1]:.2f} kPa."
    elif "thermal anomaly" in user_input.lower():
        response = f"Thermal anomaly detected: {df['ThermalAnomaly'].iloc[-1]}."
    elif "x-ray defects" in user_input.lower():
        response = f"The current X-ray defects count is {df['XrayDefects'].iloc[-1]:.2f}."
    elif "packaging rate" in user_input.lower():
        response = f"The current packaging rate is {df['PackagingRate'].iloc[-1]:.2f} units/min."
    elif "agv load" in user_input.lower():
        response = f"The current AGV load is {df['AGVLoad'].iloc[-1]:.2f} kg."
    elif "storage temperature" in user_input.lower():
        response = f"The current storage temperature is {df['StorageTemperature'].iloc[-1]:.2f} °C."
    elif "storage humidity" in user_input.lower():
        response = f"The current storage humidity is {df['StorageHumidity'].iloc[-1]:.2f}%."
    
    return response


# Function for the Virtual Assistant using GPT-4
def virtual_assistant():
    # Custom CSS for styling and layout adjustments
    st.markdown("""
        <style>
        /* 1. Sidebar and Navigation */
        .sidebar .css-1lcbmhc {
            background-color: #2a3e52;
            padding: 20px;
        }
        
        .sidebar .css-1lcbmhc a {
            color: #ffffff !important;
            font-size: 16px !important;
            padding: 10px 15px !important;
            border-radius: 5px !important;
            transition: background-color 0.3s !important;
            display: block;
            text-decoration: none;
        }
        
        .sidebar .css-1lcbmhc a:hover, .sidebar .css-1lcbmhc a.active {
            background-color: #1f2c3a !important;
            color: #ffffff !important;
        }

        /* 2. Chat Interface (Main Area) */
        .chat-bubble-user {
            background-color: #e6e6e6;
            color: #333333;
            border-radius: 15px;
            padding: 10px 15px;
            margin-bottom: 10px;
            max-width: 60%;
            word-wrap: break-word;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }

        .chat-bubble-yantra {
            background-color: #00468b;
            color: white;
            border-radius: 15px;
            padding: 10px 15px;
            margin-bottom: 10px;
            max-width: 60%;
            word-wrap: break-word;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        .chat-interface .css-1v3fvcr {
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
        }

        /* 3. Conversation History */
        .conversation-history {
            background-color: #f4f6f8;
            padding: 15px;
            border-radius: 10px;
            height: 250px; /* Adjust as needed */
            overflow-y: scroll;
            border: 1px solid #dcdfe3;
        }

        .conversation-history {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            height: 200px;
            overflow-y: scroll;
            margin-bottom: 15px;
        }
        
        .conversation-history .history-item:hover {
            background-color: #e0e6ed;
        }

        .processing-indicator {
            font-size: 18px;
            color: #0073e6;
            text-align: center;
            margin-top: 10px;
        }

        /* 4. Header and Subheader */
        .header {
            text-align: center;
            padding: 20px;
        }
        
        .header .logo {
            width: 150px;
        }
        
        .header h2 {
            color: #2a3e52;
            font-family: 'Roboto', sans-serif;
            font-weight: bold;
            font-size: 28px;
        }

        .header h4 {
            color: #888888;
            font-family: 'Roboto', sans-serif;
            font-size: 20px;
        }

        /* 5. Widgets Section */
        .widget-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        
        .widget {
            background-color: #ffffff;
            padding: 10px 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
            font-size: 16px;
            color: #333333;
            transition: transform 0.3s ease, background-color 0.3s ease;
            cursor: pointer;
        }

        .widget:hover {
            transform: translateY(-5px);
            background-color: #f1f1f1;
        }

        /* 6. Buttons and Forms */
        .stButton>button {
            background-color: #00468b;
            color: white;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px 20px;
            transition: background-color 0.3s;
            border: none;
        }

        .stButton>button:hover {
            background-color: #003366;
        }
        
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            padding: 10px;
            font-size: 16px;
            margin-bottom: 15px;
        }

        /* 7. Background and Layout */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f6f8;
        }
        
        .main-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 20px;
            background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
            min-height: 100vh;
        }

        /* Additional global styles */
        .css-1v3fvcr {
            font-size: 16px;
        }
        
        </style>
    """, unsafe_allow_html=True)

    # Top Header
    st.markdown("<h2 style='text-align: center;'>Yantra: Your AI Co-Pilot</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #888;'>Unlock the Potential of AI. Seamlessly integrate machine learning, natural language understanding, and predictive analytics into your workflows.</h4>", unsafe_allow_html=True)

    # Create the Widgets Section (First 6 use cases as widgets)
    st.markdown("""
    <div class="widget-container">
        <div class="widget">Real-Time Monitoring</div>
        <div class="widget">Data Analytics and Reporting</div>
        <div class="widget">Predictive Maintenance</div>
        <div class="widget">Process Optimization</div>
        <div class="widget">Training and Onboarding</div>
        <div class="widget">Compliance and Safety Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat Interface Section
    st.markdown("<div class='chat-interface'>", unsafe_allow_html=True)
    st.markdown("<h4>Chat with Yantra</h4>", unsafe_allow_html=True)

    # Chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Function to display the chat history
    def display_chat():
        st.markdown("<div class='conversation-history'>", unsafe_allow_html=True)
        for chat in st.session_state["chat_history"]:
            st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-yantra'><b>AIQ:</b> {chat['aiq']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Display the chat history
    display_chat()

    # Text input for the user's query
    user_input = st.text_input("Type your message here...", key="user_input", placeholder="Ask a question...")

    # Example Plotly Chart - Responding to User's Request
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
        'MaterialFeedRate': np.random.randint(100, 200, size=100)
    })

    if user_input.lower() == "show material feed rate chart":
        fig = px.line(df, x='Timestamp', y='MaterialFeedRate', title='Material Feed Rate Over Time')
        st.plotly_chart(fig)

    if st.button("Send"):
        if user_input:
            # Show processing animation
            with st.spinner('Processing...'):
                time.sleep(2)  # Simulate processing time

                # First check if the query relates to the data
                data_response = handle_data_query(user_input)
                
                if data_response:
                    # If it's a data-related response, use it directly
                    response_text = data_response
                else:
                    # If not data-related, generate a response from GPT-4
                    gpt_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    response_text = gpt_response['choices'][0]['message']['content']

            # Append the user input and AIQ response to chat history
            st.session_state["chat_history"].append({"user": user_input, "aiq": response_text})
            st.session_state["user_input"] = ""  # Clear the input box after sending

            # Refresh the interface to display the new chat bubbles
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# Main content selection based on sidebar
if selected_section == "Dashboard Overview":
    dashboard_overview()
elif selected_section == "Predictive Maintenance":
    predictive_maintenance()
elif selected_section == "Anomaly Detection":
    anomaly_detection()
elif selected_section == "Process Optimization":
    process_optimization()
elif selected_section == "Energy Management":
    energy_management()
elif selected_section == "Virtual Assistant":
    virtual_assistant()
