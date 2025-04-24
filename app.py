import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from emissions_model import BitRobotEmissionsModel

# Set page config for wide layout
st.set_page_config(
    layout="wide",
    page_title="BitRobot Emissions Model",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Enable LaTeX support
st.markdown("""
<style>
.katex-html {
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("BitRobot Emissions Model")

# Create tabs
tab1, tab2 = st.tabs(["Documentation", "Simulation"])

with tab1:
    # Render the readme.md content with LaTeX support
    st.markdown("""
    # Emissions & Circulating Supply Model

    ## Overview

    This document provides a mathematical definition of the BitRobot emissions model described [here](https://docs.google.com/spreadsheets/d/1xgWB7Z7pZE_DYL3E6_krAk0EY9xS1zQUdY5lSOyfh8M/edit?gid=1290723036#gid=1290723036):

    - We use monthly time indexing rather than yearly to have a finer grained view of the emissions.
    - Assume a linear vesting schedule for the team & consultants over 3 years.
    - Specifies **fixed emissions** for an initial period
    - Switches to **burn-adjusted emissions** from a configurable month

    ---

    ## Time Indexing

    Define:

    - Let $t \in \\{0, 1, 2, \dots, T_{\\text{max}} \\}$ represent **months**, where $t = 0$ is the **TGE** (Token Generation Event)
    - Let $S(t)$ be the **circulating supply** at the **end** of month $t$
    - Let $E(t)$ be the **emissions** introduced during month $t$
    - Let $V(t)$ be the **tokens vested** (unlocked) from the team/consultant allocation in month $t$
    - Let $B(t)$ be the **tokens burned** during month $t$
    - Let $T_{\\text{burn}}$ be the **first month** in which burn-based emissions begin
    - Let $F$ be the **burn-based emission factor**. This is a modeling assumption to understand difference scenarios of how token inflation may occur.

    ---

    ## Initial Allocation at TGE

    - Total allocated at TGE:  
      $$
      A_{\\text{total}} = 1,000,000,000 \\text{ tokens}
      $$

    - Team + Consultants allocation:
      $$
      A_{\\text{team}} = a \cdot A_{\\text{total}}, \quad \\text{e.g., } a = 0.3 \Rightarrow A_{\\text{team}} = 300,000,000
      $$

    - The remainder is allocated to other buckets (liquidity, community, etc.), but only what's vested enters circulating supply.

    ---

    ## Vesting Schedule

    Assuming **linear monthly vesting over 3 years** (36 months):

    $$
    V(t) = \\begin{cases}
    \\frac{A_{\\text{team}}}{36}, & 0 \leq t < 36 \\\\
    0, & \\text{otherwise}
    \\end{cases}
    $$

    ---

    ## Emissions Schedule

    ### Fixed Emissions (Months $0 \leq t < T_{\\text{burn}}$)

    Fixed emissions are according to a user-defined schedule:

    $$
    E(t) = E_{\\text{fixed}}(t), \quad \\text{for } t < T_{\\text{burn}}
    $$

    Where $E_{\\text{fixed}}(t)$ is manually defined per month.

    > Example (in Excel Sheet):
    > - Months 1–12: $\\frac{100,000,000}{12}$
    > - Months 13–24: $\\frac{88,000,000}{12}$
    > - Months 25–36: $\\frac{60,000,000}{12}$
    > - Months 37–48: $\\frac{25,000,000}{12}$

    ---

    ### Burn-Based Emissions (Months $t \geq T_{\\text{burn}}$)

    Emissions are calculated based on burn from the **previous N months**:

    Let the base factor be:

    $$
    B'(t) = \\sum_{i=t-N}^{t-1} B(i)
    $$

    Then emissions are then:

    $$
    E(t) = F \cdot \\frac{B'(t)}{N}
    $$

    Where:

    - $F$ is a tunable scalar (e.g., 0.9), and $N$ is the lookback window size.
    - This causes emissions to **scale with burn**, which is assumed to correlate with protocol usage and value capture.

    ---

    ## Circulating Supply

    At each month $t$:

    $$
    S(t) = S(t-1) + V(t) + E(t) - B(t)
    $$

    With:

    $$
    S(0) = V(0)
    $$

    """, unsafe_allow_html=True)

with tab2:
    # Default values from BitRobotEmissionsModel constructor
    default_params = {
        "total_supply": 1_000_000_000,
        "team_allocation_percentage": 0.3,
        "vesting_months": 36,
        "t_burn": 48,
        "burn_emission_factor": 0.9,
        "burn_coefficient": 100000,
        "burn_lookback_months": 12,
        "burn_volatility": 0.2,
        "burn_pattern": "logarithmic",
        "simulation_months": 120
    }

    # Sidebar for configuration
    st.sidebar.header("Model Parameters")

    # Group settings into expandable sections
    with st.sidebar.expander("Token Supply Settings", expanded=True):
        total_supply = st.number_input(
            "Supply start (tokens)",
            min_value=100_000_000,
            max_value=10_000_000_000,
            value=default_params["total_supply"],
            step=100_000_000,
            help="Starting supply of the token"
        )

        team_allocation = st.slider(
            "Team Allocation Percentage",
            min_value=0.0,
            max_value=1.0,
            value=default_params["team_allocation_percentage"],
            step=0.01,
            format="%.2f",
            help="Percentage of total supply allocated to team/consultants, subject to vesting"
        )

        vesting_months = st.slider(
            "Vesting Months",
            min_value=12,
            max_value=60,
            value=default_params["vesting_months"],
            step=1,
            help="Number of months over which team tokens are gradually released"
        )

    with st.sidebar.expander("Burn Settings", expanded=True):
        t_burn = st.slider(
            "Burn Start Month",
            min_value=12,
            max_value=120,
            value=default_params["t_burn"],
            step=1,
            help="Month at which burn-based emissions begin (after fixed emissions period)"
        )

        burn_pattern = st.selectbox(
            "Burn Pattern",
            options=["logarithmic", "exponential", "sigmoid"],
            index=["logarithmic", "exponential", "sigmoid"].index(default_params["burn_pattern"]),
            help="Type of burn pattern to use"
        )

        burn_coefficient = st.slider(
            "Burn Coefficient",
            min_value=10000,
            max_value=1000000,
            value=default_params["burn_coefficient"],
            step=10000,
            help="Coefficient that determines the base level of token burning"
        )

        burn_volatility = st.slider(
            "Burn Volatility",
            min_value=0.0,
            max_value=1.0,
            value=default_params["burn_volatility"],
            step=0.05,
            format="%.2f",
            help="Standard deviation of random burn variation as percentage of base burn"
        )

    with st.sidebar.expander("Emission Settings", expanded=True):
        burn_emission_factor = st.slider(
            "Burn Emission Factor",
            min_value=0.1,
            max_value=2.0,
            value=default_params["burn_emission_factor"],
            step=0.1,
            help="Multiplier that determines how much new tokens are emitted based on burn rate"
        )

        burn_lookback_months = st.slider(
            "Burn Lookback Months",
            min_value=1,
            max_value=36,
            value=default_params["burn_lookback_months"],
            step=1,
            help="Number of months to look back when calculating burn-based emissions"
        )

    with st.sidebar.expander("Simulation Settings", expanded=True):
        simulation_months = st.slider(
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
        burn_lookback_months=burn_lookback_months,
        burn_volatility=burn_volatility,
        burn_pattern=burn_pattern,
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

    def create_stacked_area_chart(data, title):
        # Reshape data for stacked area chart
        df_stacked = pd.melt(
            data,
            id_vars=['Month'],
            value_vars=['Cumulative Vested', 'Cumulative Emitted'],
            var_name='Source',
            value_name='Tokens'
        )
        
        # Create the stacked area chart
        area = alt.Chart(df_stacked).mark_area().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Tokens:Q', title='Tokens', axis=alt.Axis(format='~s')),
            color=alt.Color('Source:N', 
                           scale=alt.Scale(domain=['Cumulative Vested', 'Cumulative Emitted'],
                                         range=['#1f77b4', '#2ca02c'])),
            tooltip=['Month', 'Source', alt.Tooltip('Tokens', format='~s')]
        )
        
        # Create the vertical rule for burn start
        rule = alt.Chart(pd.DataFrame({'x': [t_burn]})).mark_rule(
            color='red',
            strokeDash=[5, 5]
        ).encode(x='x:Q')
        
        # Layer the charts and configure
        chart = (area + rule).properties(
            title=title,
            width=800,
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
        st.altair_chart(create_chart(df, 'Circulating Supply', 'Circulating Supply Over Time (Net of Burns)', 'blue'), use_container_width=True)
        st.altair_chart(create_chart(df, 'Cumulative Emissions', 'Total Tokens Ever Emitted (Gross of Burns)', 'purple'), use_container_width=True)

    with col2:
        st.altair_chart(create_chart(df, 'Emissions', 'Monthly New Emissions', 'green'), use_container_width=True)
        st.altair_chart(create_chart(df, 'Burn', 'Monthly Token Burns', 'orange'), use_container_width=True)

    # Add supply composition chart
    st.altair_chart(create_stacked_area_chart(df, 'Circulating Supply Composition Over Time'), use_container_width=True)

    # Add net supply change chart
    st.altair_chart(create_chart(df, 'Net Supply Change', 'Monthly Net Supply Change (Emissions + Vesting - Burn)', 'red'), use_container_width=True)

    # Add explanation
    st.markdown("""
    ### Understanding the Metrics

    - **Circulating Supply**: The actual number of tokens available in the market (net of burns)
    - **Cumulative Emissions**: The total number of tokens ever created (gross of burns)
    - **Monthly Emissions**: New tokens created each month
    - **Monthly Burns**: Tokens removed from circulation each month
    - **Net Supply Change**: The net change in circulating supply each month (emissions + vesting - burns)
    """)

    # Display raw data if requested
    if st.checkbox("Show Raw Data"):
        st.dataframe(df) 