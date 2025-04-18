import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from emissions_model import BitRobotEmissionsModel

# Set page config for wide layout
st.set_page_config(layout="wide")

# Title
st.title("BitRobot Emissions Model")

# Sidebar for configuration
st.sidebar.header("Model Parameters")

# Default values from BitRobotEmissionsModel constructor
default_params = {
    "total_supply": 1_000_000_000,
    "team_allocation_percentage": 0.3,
    "vesting_months": 36,
    "t_burn": 48,
    "burn_emission_factor": 0.9,
    "burn_coefficient": 100000,
    "simulation_months": 120
}

# Create sliders in sidebar
total_supply = st.sidebar.number_input(
    "Supply start (tokens)",
    min_value=100_000_000,
    max_value=10_000_000_000,
    value=default_params["total_supply"],
    step=100_000_000,
    help="Starting supply of the token"
)

team_allocation = st.sidebar.slider(
    "Team Allocation Percentage",
    min_value=0.0,
    max_value=1.0,
    value=default_params["team_allocation_percentage"],
    step=0.01,
    format="%.2f",
    help="Percentage of total supply allocated to team/consultants, subject to vesting"
)

vesting_months = st.sidebar.slider(
    "Vesting Months",
    min_value=12,
    max_value=60,
    value=default_params["vesting_months"],
    step=1,
    help="Number of months over which team tokens are gradually released"
)

t_burn = st.sidebar.slider(
    "Burn Start Month",
    min_value=12,
    max_value=120,
    value=default_params["t_burn"],
    step=1,
    help="Month at which burn-based emissions begin (after fixed emissions period)"
)

burn_emission_factor = st.sidebar.slider(
    "Burn Emission Factor",
    min_value=0.1,
    max_value=2.0,
    value=default_params["burn_emission_factor"],
    step=0.1,
    help="Multiplier that determines how much new tokens are emitted based on burn rate"
)

burn_coefficient = st.sidebar.slider(
    "Burn Coefficient",
    min_value=10000,
    max_value=1000000,
    value=default_params["burn_coefficient"],
    step=10000,
    help="Coefficient that determines the base level of token burning"
)

simulation_months = st.sidebar.slider(
    "Simulation Months",
    min_value=60,
    max_value=240,
    value=default_params["simulation_months"],
    step=12,
    help="Total number of months to simulate"
)

# Create model instance with current parameters
model = BitRobotEmissionsModel(
    total_supply=total_supply,
    team_allocation_percentage=team_allocation,
    vesting_months=vesting_months,
    t_burn=t_burn,
    burn_emission_factor=burn_emission_factor,
    burn_coefficient=burn_coefficient,
    simulation_months=simulation_months
)

# Run simulation
model.run_simulation()

# Get results as DataFrame
df = model.get_results_dataframe()

# Create Altair charts
def create_chart(data, y_col, title, color):
    # Create the main line chart
    line = alt.Chart(data).mark_line(color=color).encode(
        x=alt.X('Month:Q', title='Month'),
        y=alt.Y(f'{y_col}:Q', title='Tokens', axis=alt.Axis(format='~s')),
        tooltip=['Month', alt.Tooltip(y_col, format='~s')]
    )
    
    # Create the vertical rule for burn start
    rule = alt.Chart(pd.DataFrame({'x': [t_burn]})).mark_rule(
        color='red',
        strokeDash=[5, 5]
    ).encode(x='x:Q')
    
    # Layer the charts and configure
    chart = (line + rule).properties(
        title=title,
        width=400,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16
    ).interactive()
    
    return chart

# Create charts
col1, col2 = st.columns(2)

with col1:
    st.altair_chart(create_chart(df, 'Circulating Supply', 'Circulating Supply Over Time', 'blue'), use_container_width=True)
    st.altair_chart(create_chart(df, 'Cumulative Emissions', 'Cumulative Emissions Over Time', 'purple'), use_container_width=True)

with col2:
    st.altair_chart(create_chart(df, 'Emissions', 'Monthly Emissions', 'green'), use_container_width=True)
    st.altair_chart(create_chart(df, 'Burn', 'Monthly Burns', 'orange'), use_container_width=True)

# Display raw data if requested
if st.checkbox("Show Raw Data"):
    st.dataframe(df) 