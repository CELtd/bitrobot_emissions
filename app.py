import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from bitrobot_supply_model import BitRobotSupplyModel

# Configure Streamlit for widescreen mode
st.set_page_config(
    page_title="BitRobot Emissions Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 BitRobot Emissions Model")
st.markdown("Simulate and visualize the BitRobot token emissions schedule and supply breakdown.")

# Sidebar for parameters
st.sidebar.header("Configuration")

# Emissions Schedule Configuration Expander
with st.sidebar.expander("Emissions Schedule Configuration", expanded=True):
    emissions_schedule_type = st.selectbox(
        "Emissions Schedule Type",
        ["static", "linear", "exponential", "percentage"],
        index=3,
        help="Choose the static emissions schedule type."
    )

    # Static schedule parameters (only show if static is selected)
    if emissions_schedule_type == "static":
        static_year1 = st.number_input(
            "Year 1 Emission (tokens)",
            value=80_000_000,
            min_value=0,
            max_value=500_000_000,
            step=1_000_000,
            help="Total tokens to emit in year 1 (months 1-12)."
        )
        static_year2 = st.number_input(
            "Year 2 Emission (tokens)",
            value=64_000_000,
            min_value=0,
            max_value=500_000_000,
            step=1_000_000,
            help="Total tokens to emit in year 2 (months 13-24)."
        )
        static_year3 = st.number_input(
            "Year 3 Emission (tokens)",
            value=40_000_000,
            min_value=0,
            max_value=500_000_000,
            step=1_000_000,
            help="Total tokens to emit in year 3 (months 25-36)."
        )
        static_year4 = st.number_input(
            "Year 4 Emission (tokens)",
            value=16_000_000,
            min_value=0,
            max_value=500_000_000,
            step=1_000_000,
            help="Total tokens to emit in year 4 (months 37-48)."
        )
    else:
        static_year1 = 80_000_000
        static_year2 = 64_000_000
        static_year3 = 40_000_000
        static_year4 = 16_000_000

    # Linear schedule parameters (only show if linear is selected)
    if emissions_schedule_type == "linear":
        linear_total_emissions = st.number_input(
            "Linear Total Emissions (tokens)",
            value=200_000_000,
            min_value=0,
            max_value=500_000_000,
            step=1_000_000,
            help="Total tokens to emit over 48 months for linear schedule."
        )
        
        linear_start_emission = st.number_input(
            "Linear Start Emission (per month, tokens)",
            value=12_000_000,
            min_value=0,
            max_value=50_000_000,
            step=100_000,
            help="Starting monthly emission for linear schedule."
        )
        linear_end_emission = st.number_input(
            "Linear End Emission (per month, tokens)",
            value=2_000_000,
            min_value=0,
            max_value=50_000_000,
            step=100_000,
            help="Ending monthly emission for linear schedule."
        )
        
        # Calculate what the actual total would be with current start/end values
        raw_total = (linear_start_emission + linear_end_emission) * 48 / 2
        
        # Show if values need adjustment
        if abs(raw_total - linear_total_emissions) > 1_000_000:  # Allow 1M tolerance
            st.warning(f"⚠️ Start/end values will be automatically scaled to achieve {linear_total_emissions:,.0f} total emissions")
            st.caption(f"Raw calculation: {raw_total:,.0f} tokens")
        else:
            st.success(f"✅ Start/end values will achieve {linear_total_emissions:,.0f} total emissions")
    else:
        linear_start_emission = 12_000_000
        linear_end_emission = 2_000_000
        linear_total_emissions = 200_000_000

    # Exponential schedule parameters (only show if exponential is selected)
    if emissions_schedule_type == "exponential":
        exponential_start_emission = st.number_input(
            "Exponential Start Emission (per month, tokens)",
            value=12_000_000,
            min_value=0,
            max_value=50_000_000,
            step=100_000,
            help="Starting monthly emission for exponential decay schedule."
        )
        exponential_end_emission = st.number_input(
            "Exponential End Emission (per month, tokens)",
            value=2_000_000,
            min_value=0,
            max_value=50_000_000,
            step=100_000,
            help="Ending monthly emission for exponential decay schedule."
        )
        
        # Calculate and display total emissions for exponential schedule
        # For exponential decay, we need to sum the actual curve
        # This is an approximation: total ≈ (start + end) * 48 / 2 * adjustment_factor
        # The adjustment factor accounts for the curve shape
        if exponential_start_emission > 0 and exponential_end_emission > 0:
            decay_rate = -np.log(exponential_end_emission / exponential_start_emission) / 47
            # Calculate actual total by summing the curve
            total_emissions = 0
            for month in range(1, 49):
                emission = exponential_start_emission * np.exp(-decay_rate * (month - 1))
                total_emissions += emission
            st.info(f"**Exponential Schedule Total:** {total_emissions:,.0f} tokens over 48 months")
    else:
        exponential_start_emission = 12_000_000
        exponential_end_emission = 2_000_000

    # Percentage schedule parameters (only show if percentage is selected)
    if emissions_schedule_type == "percentage":
        percentage_total_emissions = st.number_input(
            "Percentage Total Emissions (tokens)",
            value=250_000_000,
            min_value=0,
            max_value=500_000_000,
            step=1_000_000,
            help="Total tokens to emit over 48 months for percentage schedule."
        )
        
        percentage_start_pct = st.number_input(
            "Percentage Start (% of 1B supply per month)",
            value=0.8,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            help="Starting monthly emission as percentage of 1 billion total supply."
        )
        percentage_end_pct = st.number_input(
            "Percentage End (% of 1B supply per month)",
            value=0.25,
            min_value=0.01,
            max_value=10.0,
            step=0.01,
            help="Ending monthly emission as percentage of 1 billion total supply."
        )
        
        # Calculate what the actual total would be with current start/end values
        raw_total = (percentage_start_pct + percentage_end_pct) * 48 / 2 * 1_000_000_000 / 100
        
        # Show if values need adjustment
        if abs(raw_total - percentage_total_emissions) > 1_000_000:  # Allow 1M tolerance
            st.warning(f"⚠️ Start/end percentages will be automatically scaled to achieve {percentage_total_emissions:,.0f} total emissions")
            st.caption(f"Raw calculation: {raw_total:,.0f} tokens")
        else:
            st.success(f"✅ Start/end percentages will achieve {percentage_total_emissions:,.0f} total emissions")
    else:
        percentage_start_pct = 0.8
        percentage_end_pct = 0.25
        percentage_total_emissions = 250_000_000

simulation_months = st.number_input(
    "Simulation Months",
    value=120,
    min_value=12,
    max_value=240
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Emissions Schedule", "Fee Simulation", "📊 Download Data"])

with tab1:
    st.header("Emissions Schedule Analysis")
    
    # Supply Configuration Expander
    with st.sidebar.expander("Supply Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            team_allocation = st.number_input(
                "Team Allocation (M)", 
                value=269, 
                min_value=0, 
                max_value=1000,
                help="Tokens allocated to team in millions"
            ) * 1_000_000
            
            investor_allocation = st.number_input(
                "Investor Allocation (M)", 
                value=351, 
                min_value=0, 
                max_value=1000,
                help="Tokens allocated to investors in millions"
            ) * 1_000_000
            
            foundation_allocation = st.number_input(
                "Foundation & Ecosystem Growth (M)", 
                value=307, 
                min_value=0, 
                max_value=1000,
                help="Tokens allocated to Foundation in millions"
            ) * 1_000_000
            
            foundation_initial_liquidity = st.number_input(
                "Foundation Initial Liquidity (M)", 
                value=50, 
                min_value=0, 
                max_value=100,
                help="Initial Foundation liquidity release in millions"
            ) * 1_000_000
        
        with col2:
            team_cliff_months = st.number_input(
                "Team Cliff (months)", 
                value=12, 
                min_value=0, 
                max_value=60
            )
            
            team_vesting_months = st.number_input(
                "Team Vesting (months)", 
                value=24, 
                min_value=1, 
                max_value=120
            )
            
            foundation_vesting_months = st.number_input(
                "Foundation Vesting (months)", 
                value=48, 
                min_value=1, 
                max_value=120
            )
            
            t_burn = st.number_input(
                "Burn Start Month", 
                value=48, 
                min_value=1, 
                max_value=120
            )
    
    # Burn Parameters Expander
    with st.sidebar.expander("Burn Parameters", expanded=False):
        burn_emission_factor = st.slider(
            "Burn Emission Factor", 
            value=0.9, 
            min_value=0.1, 
            max_value=2.0, 
            step=0.1
        )
        
        # Remove Streamlit widgets for burn_coefficient and burn_pattern
        # (Lines 229 and 237)
        # Remove their inclusion in parameter dictionaries (lines 432-433 and similar for other params)

    # Fee Simulation Configuration Expander (placeholder for now)
    with st.sidebar.expander("Fee Simulation Configuration", expanded=True):
        st.info("Configure fee simulation parameters for different market scenarios.")
        
        # Fee configuration
        st.subheader("Fee Configuration")
        F_base_ent = st.number_input(
            "ENT Registration Fee", 
            value=10.0, 
            step=0.1,
            help="Base fee for ENT registration",
            key="F_base_ent"
        )
        
        F_base_subnet = st.number_input(
            "Subnet Registration Fee", 
            value=20.0, 
            step=0.1,
            help="Base fee for subnet registration",
            key="F_base_subnet"
        )
        
        subnet_maintenance_fee_pct = st.slider(
            "Subnet Maintenance Fee (%)", 
            value=25.0,
            min_value=1.0,
            max_value=40.0,
            step=1.0,
            help="Percentage of subnet rewards charged as maintenance fee",
            key="subnet_maintenance_fee_pct"
        )
        
        subnet_collateral_amount = st.number_input(
            "Subnet Collateral Amount (tokens)", 
            value=100_000, 
            min_value=0,
            max_value=1_000_000,
            step=10_000,
            help="Amount of tokens required as collateral per subnet",
            key="subnet_collateral_amount"
        )
        
        staking_percentage = st.slider(
            "Staking Percentage (% of Circulating Supply)", 
            value=30.0,
            min_value=0.0,
            max_value=50.0,
            step=0.5,
            help="Percentage of circulating supply that users choose to stake",
            key="staking_percentage"
        )
        

        
        target_staking_percentage = st.slider(
            "Target Staking Percentage (% of Circulating Supply)", 
            value=30.0,
            min_value=10.0,
            max_value=80.0,
            step=5.0,
            help="Target percentage of circulating supply that should be staked for optimal APY",
            key="target_staking_percentage"
        )
        
        target_staking_apy = st.slider(
            "Target Staking APY (% per year)", 
            value=4.0,
            min_value=1.0,
            max_value=20.0,
            step=0.5,
            help="Target annual percentage yield for stakers when target staking percentage is reached",
            key="target_staking_apy"
        )
        
        subnet_min_emissions_pct = st.slider(
            "Subnet Minimum Emissions (%)", 
            value=80.0,
            min_value=50.0,
            max_value=95.0,
            step=5.0,
            help="Minimum percentage of emissions guaranteed to subnets (remainder available for staking)",
            key="subnet_min_emissions_pct"
        )
        
        st.subheader("Scenario Parameters")
        
        # Shared starting parameters
        col1, col2 = st.columns(2)
        with col1:
            starting_ents = st.number_input(
                "Starting ENTs", 
                value=500, 
                min_value=0, 
                max_value=1000,
                help="Initial number of ENTs (same for all scenarios)",
                key="starting_ents"
            )
        with col2:
            starting_subnets = st.number_input(
                "Starting Subnets", 
                value=5, 
                min_value=0, 
                max_value=500,
                help="Initial number of subnets (same for all scenarios)",
                key="starting_subnets"
            )
        
        # Staking participants for each scenario
        st.subheader("Staking Participants")
        col1, col2, col3 = st.columns(3)
        with col1:
            staking_participants_bear = st.number_input(
                "Staking Participants (Bear)", 
                value=1000, 
                min_value=1, 
                max_value=100000,
                help="Number of staking participants in bearish scenario",
                key="staking_participants_bear"
            )
        with col2:
            staking_participants_neutral = st.number_input(
                "Staking Participants (Neutral)", 
                value=2000, 
                min_value=1, 
                max_value=100000,
                help="Number of staking participants in neutral scenario",
                key="staking_participants_neutral"
            )
        with col3:
            staking_participants_bull = st.number_input(
                "Staking Participants (Bull)", 
                value=5000, 
                min_value=1, 
                max_value=100000,
                help="Number of staking participants in bullish scenario",
                key="staking_participants_bull"
            )
        
        # Lifetime parameters
        st.subheader("Lifetime Configuration")
        col1, col2 = st.columns(2)
        with col1:
            ent_lifetime = st.number_input(
                "ENT Lifetime (months)", 
                value=12, 
                min_value=1, 
                max_value=60,
                help="How long ENTs stay active before expiring",
                key="ent_lifetime"
            )
        with col2:
            subnet_lifetime = st.number_input(
                "Subnet Lifetime (months)", 
                value=36, 
                min_value=1, 
                max_value=120,
                help="How long subnets stay active before expiring",
                key="subnet_lifetime"
            )
        
        # Bearish scenario
        st.write("**Bearish Scenario**")
        col1, col2 = st.columns(2)
        with col1:
            lambda_ent_bear = st.number_input(
                "ENT Rate (Bear)", 
                value=250.0, 
                min_value=0.1, 
                max_value=5000.0,
                help="ENT arrival rate for bearish scenario",
                key="lambda_ent_bear"
            )
        with col2:
            lambda_subnet_bear = st.number_input(
                "Subnet Rate (Bear)", 
                value=1.0/12.0, 
                min_value=0.01, 
                max_value=100.0,
                help="Subnet arrival rate for bearish scenario",
                key="lambda_subnet_bear"
            )
        
        # Neutral scenario
        st.write("**Neutral Scenario**")
        col1, col2 = st.columns(2)
        with col1:
            lambda_ent_neutral = st.number_input(
                "ENT Rate (Neutral)", 
                value=500.0, 
                min_value=0.1, 
                max_value=5000.0,
                help="ENT arrival rate for neutral scenario",
                key="lambda_ent_neutral"
            )
        with col2:
            lambda_subnet_neutral = st.number_input(
                "Subnet Rate (Neutral)", 
                value=3.0/12.0, 
                min_value=0.01, 
                max_value=100.0,
                help="Subnet arrival rate for neutral scenario",
                key="lambda_subnet_neutral"
            )
        
        # Bullish scenario
        st.write("**Bullish Scenario**")
        col1, col2 = st.columns(2)
        with col1:
            lambda_ent_bull = st.number_input(
                "ENT Rate (Bull)", 
                value=1000.0, 
                min_value=0.1, 
                max_value=5000.0,
                help="ENT arrival rate for bullish scenario",
                key="lambda_ent_bull"
            )
        with col2:
            lambda_subnet_bull = st.number_input(
                "Subnet Rate (Bull)", 
                value=9.0/12.0, 
                min_value=0.01, 
                max_value=100.0,
                help="Subnet arrival rate for bullish scenario",
                key="lambda_subnet_bull"
            )
    
    # Run simulation button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Running emissions and fee simulations..."):
            # Run three scenarios: bear, neutral, and bull
            models = {}
            results_dfs = {}
            
            # Common parameters for all scenarios
            common_params = {
                'team_allocation': team_allocation,
                'investor_allocation': investor_allocation,
                'foundation_allocation': foundation_allocation,
                'foundation_initial_liquidity': foundation_initial_liquidity,
                'foundation_target_48m': foundation_allocation,
                'team_cliff_months': team_cliff_months,
                'team_vesting_months': team_vesting_months,
                'foundation_vesting_months': foundation_vesting_months,
                't_burn': t_burn,
                'burn_emission_factor': burn_emission_factor,
                'simulation_months': simulation_months,
                'emissions_schedule_type': emissions_schedule_type,
                'linear_start_emission': linear_start_emission,
                'linear_end_emission': linear_end_emission,
                'linear_total_emissions': linear_total_emissions,
                'static_year1': static_year1,
                'static_year2': static_year2,
                'static_year3': static_year3,
                'static_year4': static_year4,
                'exponential_start_emission': exponential_start_emission,
                'exponential_end_emission': exponential_end_emission,
                'percentage_start_pct': percentage_start_pct,
                'percentage_end_pct': percentage_end_pct,
                'percentage_total_emissions': percentage_total_emissions,
                'starting_ents': st.session_state.starting_ents,
                'starting_subnets': st.session_state.starting_subnets,
                'ent_lifetime': st.session_state.ent_lifetime,
                'subnet_lifetime': st.session_state.subnet_lifetime,
                'F_base_ent': st.session_state.F_base_ent,
                'F_base_subnet': st.session_state.F_base_subnet,
                'subnet_maintenance_fee_pct': st.session_state.subnet_maintenance_fee_pct / 100.0,
                'subnet_collateral_amount': st.session_state.subnet_collateral_amount,
                'staking_percentage': st.session_state.staking_percentage / 100.0,
                'target_staking_percentage': st.session_state.target_staking_percentage / 100.0,
                'target_staking_apy': st.session_state.target_staking_apy / 100.0,
                'subnet_min_emissions_pct': st.session_state.subnet_min_emissions_pct / 100.0
            }
            
            # Bear scenario (lower growth)
            bear_params = common_params.copy()
            bear_params.update({
                'lambda_ent': st.session_state.lambda_ent_bear,
                'lambda_subnet': st.session_state.lambda_subnet_bear,
                'staking_participants': st.session_state.staking_participants_bear,
                'random_seed': 42
            })
            models['bear'] = BitRobotSupplyModel(**bear_params)
            models['bear'].run_simulation()
            results_dfs['bear'] = models['bear'].get_results_dataframe()
            
            # Neutral scenario (baseline growth)
            neutral_params = common_params.copy()
            neutral_params.update({
                'lambda_ent': st.session_state.lambda_ent_neutral,
                'lambda_subnet': st.session_state.lambda_subnet_neutral,
                'staking_participants': st.session_state.staking_participants_neutral,
                'random_seed': 43
            })
            models['neutral'] = BitRobotSupplyModel(**neutral_params)
            models['neutral'].run_simulation()
            results_dfs['neutral'] = models['neutral'].get_results_dataframe()
            
            # Bull scenario (higher growth)
            bull_params = common_params.copy()
            bull_params.update({
                'lambda_ent': st.session_state.lambda_ent_bull,
                'lambda_subnet': st.session_state.lambda_subnet_bull,
                'staking_participants': st.session_state.staking_participants_bull,
                'random_seed': 44
            })
            models['bull'] = BitRobotSupplyModel(**bull_params)
            models['bull'].run_simulation()
            results_dfs['bull'] = models['bull'].get_results_dataframe()
            
            # Store results in session state
            st.session_state.results_dfs = results_dfs
            st.session_state.models = models
            st.session_state.results_df = results_dfs['neutral']  # Default to neutral for main tab
            
            st.success("Multi-scenario simulation completed!")
    
    # Display plots if results are available
    if 'results_df' in st.session_state:
        results_df = st.session_state.results_df
        
        # Calculate components for the breakdown plot
        # Combine staking and subnet rewards as community
        community_portion = results_df['Staking Rewards'].cumsum() + results_df['Subnet Rewards'].cumsum()
        team_portion = results_df['Team Vested']
        investor_portion = results_df['Investor Vested']
        foundation_portion = results_df['Foundation Vested']
        sum_val = community_portion + team_portion + foundation_portion + investor_portion
        
        # Create data for breakdown plot - properly structured
        breakdown_data = []
        for i, month in enumerate(results_df['Month']):
            breakdown_data.append({
                'Month': month,
                'Component': 'Community',
                'Percentage': (community_portion.iloc[i] / sum_val.iloc[i]) * 100
            })
            breakdown_data.append({
                'Month': month,
                'Component': 'Team',
                'Percentage': (team_portion.iloc[i] / sum_val.iloc[i]) * 100
            })
            breakdown_data.append({
                'Month': month,
                'Component': 'Investors',
                'Percentage': (investor_portion.iloc[i] / sum_val.iloc[i]) * 100
            })
            breakdown_data.append({
                'Month': month,
                'Component': 'Foundation & Ecosystem Growth',
                'Percentage': (foundation_portion.iloc[i] / sum_val.iloc[i]) * 100
            })
        breakdown_data = pd.DataFrame(breakdown_data)
        
        # Plot 1: Breakdown of Cumulative Supply
        breakdown_chart = alt.Chart(breakdown_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Percentage:Q', title='Percentage (%)'),
            color=alt.Color('Component:N', scale=alt.Scale(
                domain=['Community', 'Team', 'Investors', 'Foundation & Ecosystem Growth'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            )),
            strokeDash=alt.condition(
                alt.datum.Component == 'Investors',
                alt.value([5, 5]),
                alt.value([0])
            )
        ).properties(
            title='Breakdown of Cumulative Supply',
            width=400,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Add vertical line at month 48
        rule = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            color='gray',
            strokeDash=[5, 5],
            strokeWidth=2
        ).encode(x='x:Q')
        
        breakdown_chart = alt.layer(breakdown_chart, rule)
        
        # Plot 2: Net Flow - properly structured with all components
        net_flow_data = []
        for i, month in enumerate(results_df['Month']):
            net_flow_data.append({
                'Month': month,
                'Type': 'Emissions',
                'Amount': results_df['Emissions'].iloc[i] / 1e6
            })
            net_flow_data.append({
                'Month': month,
                'Type': 'Burn',
                'Amount': results_df['Burn'].iloc[i] / 1e6
            })
            net_flow_data.append({
                'Month': month,
                'Type': 'Net Flow',
                'Amount': (results_df['Emissions'].iloc[i] - results_df['Burn'].iloc[i]) / 1e6
            })
        net_flow_data = pd.DataFrame(net_flow_data)
        
        net_flow_chart = alt.Chart(net_flow_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Amount:Q', title='BRB (Millions)'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Emissions', 'Burn', 'Net Flow'],
                range=['#2ca02c', '#d62728', '#9467bd']
            )),
            strokeDash=alt.condition(
                alt.datum.Type == 'Net Flow',
                alt.value([5, 5]),
                alt.value([0])
            )
        ).properties(
            title='Supply Flow Components (Neutral Scenario)',
            width=400,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Add vertical line at month 48
        net_flow_chart = alt.layer(net_flow_chart, rule)
        
        # Plot 3: Cumulative Supply - properly structured
        cumulative_data = []
        for i, month in enumerate(results_df['Month']):
            cumulative_data.append({
                'Month': month,
                'Component': 'Community',
                'Amount': (results_df['Staking Rewards'].cumsum().iloc[i] + results_df['Subnet Rewards'].cumsum().iloc[i]) / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Team',
                'Amount': results_df['Team Vested'].iloc[i] / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Investors',
                'Amount': results_df['Investor Vested'].iloc[i] / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Foundation & Ecosystem Growth',
                'Amount': results_df['Foundation Vested'].iloc[i] / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Circulating Supply',
                'Amount': results_df['Circulating Supply'].iloc[i] / 1e9
            })
        cumulative_data = pd.DataFrame(cumulative_data)
        
        cumulative_chart = alt.Chart(cumulative_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Amount:Q', title='BRB (Billions)'),
            color=alt.Color('Component:N', scale=alt.Scale(
                domain=['Community', 'Team', 'Investors', 'Foundation & Ecosystem Growth', 'Circulating Supply'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ))
        ).properties(
            title='Cumulative Supply',
            width=400,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Add vertical line at month 48
        cumulative_chart = alt.layer(cumulative_chart, rule)
        
        # Plot 4: Pie Chart for Month 48
        target_month = 48
        month_data = results_df[results_df['Month'] == target_month].iloc[0]
        
        # print(month_data)
        # Calculate the cumulative supply for each component
        # Combine staking and subnet rewards as community
        community_supply = results_df['Staking Rewards'].cumsum().iloc[48] + results_df['Subnet Rewards'].cumsum().iloc[48]
        team_supply = month_data['Team Vested']
        investor_supply = month_data['Investor Vested']
        foundation_team_supply = month_data['Foundation Vested']
        
        # Create data for pie chart
        pie_data = pd.DataFrame({
            'Component': ['Community', 'Team', 'Investors', 'Foundation & Ecosystem Growth'],
            'Supply': [community_supply, team_supply, investor_supply, foundation_team_supply]
        })
        
        # Calculate percentages for labels
        total_supply = pie_data['Supply'].sum()
        pie_data['Percentage'] = (pie_data['Supply'] / total_supply * 100).round(1)
        pie_data['Label'] = pie_data['Component'] + ': ' + pie_data['Percentage'].astype(str) + '%'
        
        pie_chart = alt.Chart(pie_data).mark_arc().encode(
            theta=alt.Theta('Supply:Q', type='quantitative'),
            color=alt.Color('Component:N', scale=alt.Scale(
                domain=['Community', 'Team', 'Investors', 'Foundation & Ecosystem Growth'],
                range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ), legend=alt.Legend(title='Token Allocation')),
            tooltip=['Component', alt.Tooltip('Supply:Q', format='.0f'), alt.Tooltip('Percentage:Q', format='.1f')]
        ).properties(
            title=f'Token Allocation at Month {target_month} (Cumulative Supply)',
            width=400,
            height=350
        )
        
        # Display charts in a 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.altair_chart(breakdown_chart, use_container_width=True)
        
        with col2:
            st.altair_chart(net_flow_chart, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.altair_chart(cumulative_chart, use_container_width=True)
        
        with col4:
            st.altair_chart(pie_chart, use_container_width=True)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get month 48 data
        month_48_data = results_df[results_df['Month'] == 48].iloc[0]
        
        # Calculate total allocated supply (team + investor + foundation + staking + subnet emissions)
        total_allocated = (team_allocation + investor_allocation + foundation_allocation + 
                          results_df['Staking Rewards'].cumsum().iloc[48] + results_df['Subnet Rewards'].cumsum().iloc[48])
        
        with col1:
            st.metric(
                "Total Allocated Supply at Month 48",
                f"{total_allocated / 1e9:.2f}B BRB"
            )
        
        with col2:
            st.metric(
                "Circulating Supply at Month 48",
                f"{month_48_data['Circulating Supply'] / 1e9:.2f}B BRB"
            )
        
        with col3:
            st.metric(
                "Total Emissions by Month 48",
                f"{month_48_data['Cumulative Emissions'] / 1e6:.1f}M BRB"
            )
        
        with col4:
            st.metric(
                "Total Burn by Month 48 (Neutral)",
                f"{results_df['Burn'].cumsum().iloc[48] / 1e6:.1f}M BRB"
            )
        
        # Add allocation breakdown
        st.subheader("Allocation Breakdown at Month 48")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Team Allocation",
                f"{team_allocation / 1e6:.0f}M BRB",
                f"{(team_allocation / total_allocated * 100):.1f}%"
            )
        
        with col2:
            st.metric(
                "Investor Allocation", 
                f"{investor_allocation / 1e6:.0f}M BRB",
                f"{(investor_allocation / total_allocated * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Foundation Allocation",
                f"{foundation_allocation / 1e6:.0f}M BRB", 
                f"{(foundation_allocation / total_allocated * 100):.1f}%"
            )
        
        with col4:
            st.metric(
                "Community Emissions",
                f"{community_supply / 1e6:.0f}M BRB",
                f"{(community_supply / total_allocated * 100):.1f}%"
            )
    
    else:
        st.info("Click 'Run Simulation' in the sidebar to generate the emissions schedule and plots.")

with tab2:
    st.header("Fee Simulation")
    
    # Add token price slider in the sidebar
    with st.sidebar.expander("Token Price Configuration", expanded=True):
        token_price = st.slider(
            "BRB Token Price ($)",
            min_value=0.01,
            max_value=5.0,
            value=0.30,
            step=0.01,
            help="Set the BRB token price in USD"
        )
    
    # Check if we have simulation results with integrated fee data
    if 'results_dfs' not in st.session_state:
        st.info("Please run the simulation in the 'Emissions Schedule' tab to generate fee analysis plots.")
    else:
        st.info("Fee simulation based on the integrated supply model with multiple scenarios.")
        
        # Store token price in session state for dynamic updates
        st.session_state.token_price = token_price
        
        # Get integrated fee data from all scenarios
        results_dfs = st.session_state.results_dfs
        bear_df = results_dfs['bear']
        neutral_df = results_dfs['neutral']
        bull_df = results_dfs['bull']
        
        # Plot 1: Active ENTs and Subnets Over Time (all scenarios)
        active_entities_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            active_entities_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Type': 'Active ENTs',
                'Count': bear_df['Active ENTs'].iloc[i]
            })
            active_entities_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Type': 'Active Subnets',
                'Count': bear_df['Active Subnets'].iloc[i]
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            active_entities_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Type': 'Active ENTs',
                'Count': neutral_df['Active ENTs'].iloc[i]
            })
            active_entities_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Type': 'Active Subnets',
                'Count': neutral_df['Active Subnets'].iloc[i]
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            active_entities_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Type': 'Active ENTs',
                'Count': bull_df['Active ENTs'].iloc[i]
            })
            active_entities_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Type': 'Active Subnets',
                'Count': bull_df['Active Subnets'].iloc[i]
            })
        
        active_entities_data = pd.DataFrame(active_entities_data)
        
        # Split data into ENTs and Subnets
        active_ents_data = active_entities_data[active_entities_data['Type'] == 'Active ENTs']
        active_subnets_data = active_entities_data[active_entities_data['Type'] == 'Active Subnets']
        
        # Create chart for Active ENTs
        active_ents_chart = alt.Chart(active_ents_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Count:Q', title='Count'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Active ENTs Over Time',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Create chart for Active Subnets
        active_subnets_chart = alt.Chart(active_subnets_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Count:Q', title='Count'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Active Subnets Over Time',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Add vertical line at month 48 for both charts
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        active_ents_chart = alt.layer(active_ents_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        active_subnets_chart = alt.layer(active_subnets_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Plot 2: Fee Categories Over Time (all scenarios)
        fee_categories_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Type': 'ENT Registration',
                'Amount': bear_df['ENT Registration Fees'].iloc[i]
            })
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Type': 'Subnet Registration',
                'Amount': bear_df['Subnet Registration Fees'].iloc[i]
            })
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Type': 'Subnet Maintenance',
                'Amount': bear_df['Subnet Maintenance Fees'].iloc[i]
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Type': 'ENT Registration',
                'Amount': neutral_df['ENT Registration Fees'].iloc[i]
            })
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Type': 'Subnet Registration',
                'Amount': neutral_df['Subnet Registration Fees'].iloc[i]
            })
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Type': 'Subnet Maintenance',
                'Amount': neutral_df['Subnet Maintenance Fees'].iloc[i]
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Type': 'ENT Registration',
                'Amount': bull_df['ENT Registration Fees'].iloc[i]
            })
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Type': 'Subnet Registration',
                'Amount': bull_df['Subnet Registration Fees'].iloc[i]
            })
            fee_categories_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Type': 'Subnet Maintenance',
                'Amount': bull_df['Subnet Maintenance Fees'].iloc[i]
            })
        
        fee_categories_data = pd.DataFrame(fee_categories_data)
        
        # # Debug: Print fee categories data to check for differences between scenarios
        # st.write("### Debug: Fee Categories Data")
        # st.write("Sample of fee categories data (first 10 rows):")
        # st.dataframe(fee_categories_data.head(10))
        
        # # Check for differences between scenarios
        # st.write("### Debug: Fee Values by Scenario")
        # for scenario in ['Bearish', 'Neutral', 'Bullish']:
        #     scenario_data = fee_categories_data[fee_categories_data['Scenario'] == scenario]
        #     st.write(f"**{scenario} Scenario - Sample values:**")
        #     st.write(f"ENT Registration Fees: {scenario_data[scenario_data['Type'] == 'ENT Registration']['Amount'].head(5).tolist()}")
        #     st.write(f"Subnet Registration Fees: {scenario_data[scenario_data['Type'] == 'Subnet Registration']['Amount'].head(5).tolist()}")
        #     st.write(f"Subnet Maintenance Fees: {scenario_data[scenario_data['Type'] == 'Subnet Maintenance']['Amount'].head(5).tolist()}")
        #     st.write("---")
        
        fee_categories_chart = alt.Chart(fee_categories_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Amount:Q', title='Fees (BRB)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            )),
            strokeDash=alt.StrokeDash('Type:N', scale=alt.Scale(
                domain=['ENT Registration', 'Subnet Registration', 'Subnet Maintenance'],
                range=[[0], [5, 5], [10, 5]]
            ))
        ).properties(
            title='Fee Categories Over Time',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        fee_categories_chart = alt.layer(fee_categories_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Plot 3: Cumulative Total Fees Collected (all scenarios)
        cumulative_fees_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            cumulative_fees_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Cumulative Fees': bear_df['Cumulative Fees'].iloc[i]
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            cumulative_fees_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Cumulative Fees': neutral_df['Cumulative Fees'].iloc[i]
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            cumulative_fees_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Cumulative Fees': bull_df['Cumulative Fees'].iloc[i]
            })
        
        cumulative_fees_data = pd.DataFrame(cumulative_fees_data)
        
        cumulative_fees_chart = alt.Chart(cumulative_fees_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Cumulative Fees:Q', title='Cumulative Fees (BRB)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Cumulative Total Fees Collected',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        cumulative_fees_chart = alt.layer(cumulative_fees_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Plot 4: Total Fees per Month (all scenarios)
        total_fees_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            total_fees_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Total Fees': bear_df['Total Fees'].iloc[i]
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            total_fees_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Total Fees': neutral_df['Total Fees'].iloc[i]
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            total_fees_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Total Fees': bull_df['Total Fees'].iloc[i]
            })
        
        total_fees_data = pd.DataFrame(total_fees_data)
        
        # Debug: Print total fees data to check for differences between scenarios
        # st.write("### Debug: Total Fees Data")
        # st.write("Sample of total fees data (first 10 rows):")
        # st.dataframe(total_fees_data.head(10))
        
        # # Check for differences between scenarios
        # st.write("### Debug: Total Fees by Scenario")
        # for scenario in ['Bearish', 'Neutral', 'Bullish']:
        #     scenario_data = total_fees_data[total_fees_data['Scenario'] == scenario]
        #     st.write(f"**{scenario} Scenario - Sample values:**")
        #     st.write(f"Total Fees: {scenario_data['Total Fees'].head(5).tolist()}")
        #     st.write("---")
        
        total_fees_chart = alt.Chart(total_fees_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Total Fees:Q', title='Total Fees (BRB)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Total Fees per Month',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        total_fees_chart = alt.layer(total_fees_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Plot 5: Average Reward per Subnet (all scenarios)
        avg_reward_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            active_subnets = bear_df['Active Subnets'].iloc[i]
            monthly_emission = bear_df['Emissions'].iloc[i]
            avg_reward = monthly_emission / max(active_subnets, 1)  # Avoid division by zero
            avg_reward_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Avg Reward per Subnet': avg_reward
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            active_subnets = neutral_df['Active Subnets'].iloc[i]
            monthly_emission = neutral_df['Emissions'].iloc[i]
            avg_reward = monthly_emission / max(active_subnets, 1)  # Avoid division by zero
            avg_reward_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Avg Reward per Subnet': avg_reward
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            active_subnets = bull_df['Active Subnets'].iloc[i]
            monthly_emission = bull_df['Emissions'].iloc[i]
            avg_reward = monthly_emission / max(active_subnets, 1)  # Avoid division by zero
            avg_reward_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Avg Reward per Subnet': avg_reward
            })
        
        avg_reward_data = pd.DataFrame(avg_reward_data)
        
        # Filter to start from month 1
        avg_reward_data = avg_reward_data[avg_reward_data['Month'] >= 1]
        
        avg_reward_chart = alt.Chart(avg_reward_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Avg Reward per Subnet:Q', title='Avg Reward per Subnet (BRB)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Average Reward per Subnet',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        avg_reward_chart = alt.layer(avg_reward_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Plot 6: Locked Supply Over Time (Neutral Scenario - Component Breakdown)
        locked_supply_data = []
        
        # Neutral scenario only - component breakdown
        for i, month in enumerate(neutral_df['Month']):
            locked_supply_data.append({
                'Month': month,
                'Type': 'Collateral',
                'Amount': neutral_df['Locked Collateral'].iloc[i] / 1e6  # Convert to millions
            })
            locked_supply_data.append({
                'Month': month,
                'Type': 'Staking',
                'Amount': neutral_df['Staking Supply'].iloc[i] / 1e6  # Convert to millions
            })
            locked_supply_data.append({
                'Month': month,
                'Type': 'Total',
                'Amount': neutral_df['Total Locked Supply'].iloc[i] / 1e6  # Convert to millions
            })
        
        locked_supply_data = pd.DataFrame(locked_supply_data)
        
        locked_supply_chart = alt.Chart(locked_supply_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Amount:Q', title='Locked Supply (Millions BRB)'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Collateral', 'Staking', 'Total'],
                range=['#ff7f0e', '#9467bd', '#2ca02c']
            )),
            strokeDash=alt.condition(
                alt.datum.Type == 'Total',
                alt.value([5, 5]),
                alt.value([0])
            )
        ).properties(
            title='Locked Supply Over Time (Neutral Scenario)',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        locked_supply_chart = alt.layer(locked_supply_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Plot 6b: Total Locked Supply Over Time (All Scenarios)
        total_locked_supply_data = []
        
        # All scenarios - total locked supply only
        for i, month in enumerate(bear_df['Month']):
            total_locked_supply_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Total Locked Supply': bear_df['Total Locked Supply'].iloc[i] / 1e6  # Convert to millions
            })
        
        for i, month in enumerate(neutral_df['Month']):
            total_locked_supply_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Total Locked Supply': neutral_df['Total Locked Supply'].iloc[i] / 1e6  # Convert to millions
            })
        
        for i, month in enumerate(bull_df['Month']):
            total_locked_supply_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Total Locked Supply': bull_df['Total Locked Supply'].iloc[i] / 1e6  # Convert to millions
            })
        
        total_locked_supply_data = pd.DataFrame(total_locked_supply_data)
        
        total_locked_supply_chart = alt.Chart(total_locked_supply_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Total Locked Supply:Q', title='M-BRB'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Total Locked Supply Over Time (All Scenarios)',
            width=500,
            height=350
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        total_locked_supply_chart = alt.layer(total_locked_supply_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Display charts in a 2x3 grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.altair_chart(active_ents_chart, use_container_width=True)
        
        with col2:
            st.altair_chart(active_subnets_chart, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.altair_chart(fee_categories_chart, use_container_width=True)
        
        with col4:
            st.altair_chart(cumulative_fees_chart, use_container_width=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.altair_chart(total_fees_chart, use_container_width=True)
        
        with col6:
            st.altair_chart(avg_reward_chart, use_container_width=True)
        
        # Add the locked collateral chart in a new row
        col7, col8 = st.columns(2)
        
        with col7:
            st.altair_chart(locked_supply_chart, use_container_width=True)
        
        # Plot 7: Locked Supply as % of Circulating Supply (Neutral Scenario - Component Breakdown)
        locked_pct_data = []
        
        # Neutral scenario only - component breakdown
        for i, month in enumerate(neutral_df['Month']):
            circulating = neutral_df['Circulating Supply'].iloc[i]
            collateral = neutral_df['Locked Collateral'].iloc[i]
            staking = neutral_df['Staking Supply'].iloc[i]
            total_locked = neutral_df['Total Locked Supply'].iloc[i]
            
            collateral_pct = 100 * collateral / circulating if circulating > 0 else 0
            staking_pct = 100 * staking / circulating if circulating > 0 else 0
            total_pct = 100 * total_locked / circulating if circulating > 0 else 0
            
            locked_pct_data.append({
                'Month': month,
                'Type': 'Collateral',
                'Percentage': collateral_pct
            })
            locked_pct_data.append({
                'Month': month,
                'Type': 'Staking',
                'Percentage': staking_pct
            })
            locked_pct_data.append({
                'Month': month,
                'Type': 'Total',
                'Percentage': total_pct
            })
        locked_pct_data = pd.DataFrame(locked_pct_data)
        
        locked_pct_chart = alt.Chart(locked_pct_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Percentage:Q', title='%'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Collateral', 'Staking', 'Total'],
                range=['#ff7f0e', '#9467bd', '#2ca02c']
            )),
            strokeDash=alt.condition(
                alt.datum.Type == 'Total',
                alt.value([5, 5]),
                alt.value([0])
            )
        ).properties(
            title='Locked / CircSupply (Neutral Scenario)',
            width=450,
            height=300
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')
        
        locked_pct_chart = alt.layer(locked_pct_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Plot 7b: Total Locked Supply as % of Circulating Supply (All Scenarios)
        total_locked_pct_data = []
        
        # All scenarios - total locked supply percentage only
        for i, month in enumerate(bear_df['Month']):
            circulating = bear_df['Circulating Supply'].iloc[i]
            total_locked = bear_df['Total Locked Supply'].iloc[i]
            total_pct = 100 * total_locked / circulating if circulating > 0 else 0
            
            total_locked_pct_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Total Locked %': total_pct
            })
        
        for i, month in enumerate(neutral_df['Month']):
            circulating = neutral_df['Circulating Supply'].iloc[i]
            total_locked = neutral_df['Total Locked Supply'].iloc[i]
            total_pct = 100 * total_locked / circulating if circulating > 0 else 0
            
            total_locked_pct_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Total Locked %': total_pct
            })
        
        for i, month in enumerate(bull_df['Month']):
            circulating = bull_df['Circulating Supply'].iloc[i]
            total_locked = bull_df['Total Locked Supply'].iloc[i]
            total_pct = 100 * total_locked / circulating if circulating > 0 else 0
            
            total_locked_pct_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Total Locked %': total_pct
            })
        
        total_locked_pct_data = pd.DataFrame(total_locked_pct_data)
        
        total_locked_pct_chart = alt.Chart(total_locked_pct_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Total Locked %:Q', title='%'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Total Locked % of Circulating Supply (All Scenarios)',
            width=450,
            height=300
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        total_locked_pct_chart = alt.layer(total_locked_pct_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        with col8:
            st.altair_chart(locked_pct_chart, use_container_width=True)
        
        # Add the total locked supply charts in another row
        col9, col10 = st.columns(2)
        
        with col9:
            st.altair_chart(total_locked_supply_chart, use_container_width=True)
        
        with col10:
            st.altair_chart(total_locked_pct_chart, use_container_width=True)

        # Plot 8: Per-Participant Staking Rewards Over Time (All Scenarios)
        per_participant_rewards_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            per_participant_rewards_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'Rewards per Participant': bear_df['Per Staker Rewards'].iloc[i]
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            per_participant_rewards_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'Rewards per Participant': neutral_df['Per Staker Rewards'].iloc[i]
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            per_participant_rewards_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'Rewards per Participant': bull_df['Per Staker Rewards'].iloc[i]
            })
        
        per_participant_rewards_data = pd.DataFrame(per_participant_rewards_data)
        
        # Filter to start from month 1
        per_participant_rewards_data = per_participant_rewards_data[per_participant_rewards_data['Month'] >= 1]
        
        per_participant_rewards_chart = alt.Chart(per_participant_rewards_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Rewards per Participant:Q', title='Rewards per Participant (BRB)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Per-Participant Staking Rewards Over Time',
            width=450,
            height=300
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        per_participant_rewards_chart = alt.layer(per_participant_rewards_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Add the per-participant rewards chart in a new row
        col11, col12 = st.columns(2)
        
        with col11:
            st.altair_chart(per_participant_rewards_chart, use_container_width=True)
        
        # Add info about the staking plots
        st.info("""
        **Staking Analysis:**
        - **Per-Participant Rewards**: Shows actual BRB rewards each staker receives (varies by scenario due to different participant counts)
        - **APY per Token**: Shows the return rate per token staked (similar across scenarios since we assume equal staking amounts per participant)
        """)
        
        # Plot 9: Staking APY Over Time (All Scenarios) - for reference
        staking_apy_data = []
        
        # Bearish scenario
        for i, month in enumerate(bear_df['Month']):
            staking_apy_data.append({
                'Month': month,
                'Scenario': 'Bearish',
                'APY': bear_df['Staking APY'].iloc[i]
            })
        
        # Neutral scenario
        for i, month in enumerate(neutral_df['Month']):
            staking_apy_data.append({
                'Month': month,
                'Scenario': 'Neutral',
                'APY': neutral_df['Staking APY'].iloc[i]
            })
        
        # Bullish scenario
        for i, month in enumerate(bull_df['Month']):
            staking_apy_data.append({
                'Month': month,
                'Scenario': 'Bullish',
                'APY': bull_df['Staking APY'].iloc[i]
            })
        
        staking_apy_data = pd.DataFrame(staking_apy_data)
        
        # Filter to start from month 1
        staking_apy_data = staking_apy_data[staking_apy_data['Month'] >= 1]
        
        staking_apy_chart = alt.Chart(staking_apy_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('APY:Q', title='APY (%)'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Staking APY per Token Over Time',
            width=450,
            height=300
        ).add_params(
            alt.selection_interval(bind='scales')
        )

        # Add vertical line at month 48
        vertical_line = alt.Chart(pd.DataFrame({'x': [48]})).mark_rule(
            strokeDash=[5, 5],
            color='gray',
            strokeWidth=2
        ).encode(x='x:Q')

        staking_apy_chart = alt.layer(staking_apy_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        with col12:
            st.altair_chart(staking_apy_chart, use_container_width=True)

        # Display summary statistics for all scenarios
        st.subheader("Fee Simulation Summary by Scenario")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Bearish Scenario**")
            st.metric("Total Fees", f"{bear_df['Total Fees'].sum():.0f} BRB")
            st.metric("Avg Monthly Fees", f"{bear_df['Total Fees'].mean():.1f} BRB")
            st.metric("Peak ENTs", f"{bear_df['Active ENTs'].max():.0f}")
            st.metric("Peak Subnets", f"{bear_df['Active Subnets'].max():.0f}")
            st.metric("Peak Locked Collateral", f"{bear_df['Locked Collateral'].max() / 1e6:.1f}M BRB")
            st.metric("Peak Staking Supply", f"{bear_df['Staking Supply'].max() / 1e6:.1f}M BRB")
            st.metric("Peak Total Locked", f"{bear_df['Total Locked Supply'].max() / 1e6:.1f}M BRB")
            
            # Calculate average reward per subnet for bearish scenario
            avg_reward_bear = []
            for i, month in enumerate(bear_df['Month']):
                active_subnets = bear_df['Active Subnets'].iloc[i]
                monthly_emission = bear_df['Subnet Rewards'].iloc[i]  # Use subnet rewards, not total emissions
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_bear.append(avg_reward)
            st.metric("Avg Reward per Subnet", f"{np.mean(avg_reward_bear):.1f} BRB")
            
            # Calculate average staking APY for bearish scenario
            avg_apy_bear = bear_df['Staking APY'].mean()
            st.metric("Avg Staking APY", f"{avg_apy_bear:.2f}%")
        
        with col2:
            st.write("**Neutral Scenario**")
            st.metric("Total Fees", f"{neutral_df['Total Fees'].sum():.0f} BRB")
            st.metric("Avg Monthly Fees", f"{neutral_df['Total Fees'].mean():.1f} BRB")
            st.metric("Peak ENTs", f"{neutral_df['Active ENTs'].max():.0f}")
            st.metric("Peak Subnets", f"{neutral_df['Active Subnets'].max():.0f}")
            st.metric("Peak Locked Collateral", f"{neutral_df['Locked Collateral'].max() / 1e6:.1f}M BRB")
            st.metric("Peak Staking Supply", f"{neutral_df['Staking Supply'].max() / 1e6:.1f}M BRB")
            st.metric("Peak Total Locked", f"{neutral_df['Total Locked Supply'].max() / 1e6:.1f}M BRB")
            
            # Calculate average reward per subnet for neutral scenario
            avg_reward_neutral = []
            for i, month in enumerate(neutral_df['Month']):
                active_subnets = neutral_df['Active Subnets'].iloc[i]
                monthly_emission = neutral_df['Subnet Rewards'].iloc[i]  # Use subnet rewards, not total emissions
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_neutral.append(avg_reward)
            st.metric("Avg Reward per Subnet", f"{np.mean(avg_reward_neutral):.1f} BRB")
            
            # Calculate average staking APY for neutral scenario
            avg_apy_neutral = neutral_df['Staking APY'].mean()
            st.metric("Avg Staking APY", f"{avg_apy_neutral:.2f}%")
        
        with col3:
            st.write("**Bullish Scenario**")
            st.metric("Total Fees", f"{bull_df['Total Fees'].sum():.0f} BRB")
            st.metric("Avg Monthly Fees", f"{bull_df['Total Fees'].mean():.1f} BRB")
            st.metric("Peak ENTs", f"{bull_df['Active ENTs'].max():.0f}")
            st.metric("Peak Subnets", f"{bull_df['Active Subnets'].max():.0f}")
            st.metric("Peak Locked Collateral", f"{bull_df['Locked Collateral'].max() / 1e6:.1f}M BRB")
            st.metric("Peak Staking Supply", f"{bull_df['Staking Supply'].max() / 1e6:.1f}M BRB")
            st.metric("Peak Total Locked", f"{bull_df['Total Locked Supply'].max() / 1e6:.1f}M BRB")
            
            # Calculate average reward per subnet for bullish scenario
            avg_reward_bull = []
            for i, month in enumerate(bull_df['Month']):
                active_subnets = bull_df['Active Subnets'].iloc[i]
                monthly_emission = bull_df['Subnet Rewards'].iloc[i]  # Use subnet rewards, not total emissions
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_bull.append(avg_reward)
            st.metric("Avg Reward per Subnet", f"{np.mean(avg_reward_bull):.1f} BRB")
            
            # Calculate average staking APY for bullish scenario
            avg_apy_bull = bull_df['Staking APY'].mean()
            st.metric("Avg Staking APY", f"{avg_apy_bull:.2f}%")
        
        # Add yearly breakdown of earnings per subnet
        st.subheader("Earnings per Subnet by Year")
        
        def calculate_yearly_earnings(results_df):
            """Calculate both average monthly and total annual BRB per subnet for each year (net after fees)"""
            yearly_metrics = {}
            
            for year in range(1, 5):  # Years 1-4
                start_month = (year - 1) * 12 + 1
                end_month = year * 12
                
                # Filter data for this year
                year_data = results_df[(results_df['Month'] >= start_month) & (results_df['Month'] <= end_month)]
                
                if len(year_data) > 0:
                    # Calculate average monthly net reward per subnet (after fees)
                    year_net_rewards = []
                    for _, row in year_data.iterrows():
                        active_subnets = row['Active Subnets']
                        monthly_emission = row['Subnet Rewards']  # Use subnet rewards, not total emissions
                        subnet_reg_fees = row['Subnet Registration Fees']
                        subnet_maint_fees = row['Subnet Maintenance Fees']
                        
                        # Gross reward per subnet
                        gross_reward_per_subnet = monthly_emission / max(active_subnets, 1)
                        
                        # Net reward per subnet (gross - maintenance fees per subnet - registration fees amortized)
                        # Registration fees are one-time, so we amortize them over the subnet lifetime
                        # For simplicity, we'll assume they're amortized over 12 months
                        amortized_reg_fee = subnet_reg_fees / max(active_subnets, 1) / 12
                        maint_fee_per_subnet = subnet_maint_fees / max(active_subnets, 1)
                        
                        net_reward = gross_reward_per_subnet - maint_fee_per_subnet - amortized_reg_fee
                        year_net_rewards.append(max(0, net_reward))  # Ensure non-negative
                    
                    avg_monthly = np.mean(year_net_rewards)
                    
                    # Calculate total annual net rewards
                    total_yearly_gross = year_data['Subnet Rewards'].sum()  # Use subnet rewards, not total emissions
                    total_yearly_reg_fees = year_data['Subnet Registration Fees'].sum()
                    total_yearly_maint_fees = year_data['Subnet Maintenance Fees'].sum()
                    avg_subnets_that_year = year_data['Active Subnets'].mean()
                    
                    total_annual = (total_yearly_gross - total_yearly_reg_fees - total_yearly_maint_fees) / max(avg_subnets_that_year, 1)
                    total_annual = max(0, total_annual)  # Ensure non-negative
                    
                    yearly_metrics[f'Year {year}'] = {
                        'avg_monthly': avg_monthly,
                        'total_annual': total_annual
                    }
                else:
                    yearly_metrics[f'Year {year}'] = {
                        'avg_monthly': 0,
                        'total_annual': 0
                    }
            
            return yearly_metrics
        
        # Calculate yearly earnings for each scenario
        bear_yearly = calculate_yearly_earnings(bear_df)
        neutral_yearly = calculate_yearly_earnings(neutral_df)
        bull_yearly = calculate_yearly_earnings(bull_df)
        
        # Display yearly breakdown in columns with both BRB and USD values
        col1, col2, col3 = st.columns(3)
        
        def display_earnings_metrics(yearly_data, token_price, collateral_amount):
            """Display both average monthly and total annual earnings metrics with ROI"""
            st.markdown("##### Average Monthly Net Earnings per Subnet (After Fees) ℹ️")
            st.info("""
            **Monthly Net Earnings = Gross Subnet Reward - Maintenance Fees - Amortized Registration Fees**
            
            - **Gross Reward**: Monthly subnet rewards ÷ active subnets
            - **Maintenance Fees**: Monthly maintenance fees ÷ active subnets  
            - **Registration Fees**: Registration fees ÷ active subnets ÷ 12 (amortized over 1 year)
            """)
            
            # Create three columns for the headers
            brb_col, usd_col, roi_col = st.columns(3)
            with brb_col:
                st.markdown("**BRB**")
            with usd_col:
                st.markdown("**USD**")
            with roi_col:
                st.markdown("**ROI**")
            
            # Display each year's average monthly data
            for year, metrics in yearly_data.items():
                avg_monthly = metrics['avg_monthly']
                usd_value = avg_monthly * token_price
                roi_monthly = (avg_monthly / collateral_amount) * 100 if collateral_amount > 0 else 0
                
                # Create columns for this year's values
                year_brb_col, year_usd_col, year_roi_col = st.columns(3)
                
                with year_brb_col:
                    st.markdown(f"**{year}:** {avg_monthly:,.0f}")
                with year_usd_col:
                    st.markdown(f"${usd_value:,.2f}")
                with year_roi_col:
                    st.markdown(f"{roi_monthly:.2f}%")
            
            st.markdown("---")
            st.markdown("##### Total Annual Net Earnings per Subnet (After Fees) ℹ️")
            st.info("""
            **Annual Net Earnings = Total Annual Gross - Total Annual Registration Fees - Total Annual Maintenance Fees**
            
            - **Total Gross**: Sum of all monthly subnet rewards for the year
            - **Total Registration Fees**: Sum of all registration fees for the year
            - **Total Maintenance Fees**: Sum of all maintenance fees for the year
            - **Per Subnet**: Total net ÷ average active subnets for the year
            """)
            
            # Create three columns for the headers
            brb_col, usd_col, roi_col = st.columns(3)
            with brb_col:
                st.markdown("**BRB**")
            with usd_col:
                st.markdown("**USD**")
            with roi_col:
                st.markdown("**ROI**")
            
            # Display each year's total annual data
            for year, metrics in yearly_data.items():
                total_annual = metrics['total_annual']
                usd_value = total_annual * token_price
                roi_annual = (total_annual / collateral_amount) * 100 if collateral_amount > 0 else 0
                
                # Create columns for this year's values
                year_brb_col, year_usd_col, year_roi_col = st.columns(3)
                
                with year_brb_col:
                    st.markdown(f"**{year}:** {total_annual:,.0f}")
                with year_usd_col:
                    st.markdown(f"${usd_value:,.2f}")
                with year_roi_col:
                    st.markdown(f"{roi_annual:.2f}%")
        
        with col1:
            st.markdown("### Bearish Scenario")
            st.markdown("---")
            display_earnings_metrics(bear_yearly, token_price, st.session_state.subnet_collateral_amount)
        
        with col2:
            st.markdown("### Neutral Scenario")
            st.markdown("---")
            display_earnings_metrics(neutral_yearly, token_price, st.session_state.subnet_collateral_amount)
        
        with col3:
            st.markdown("### Bullish Scenario")
            st.markdown("---")
            display_earnings_metrics(bull_yearly, token_price, st.session_state.subnet_collateral_amount) 

with tab3:
    st.header("📊 Download Simulation Data")
    
    if 'results_dfs' in st.session_state:
        results_dfs = st.session_state.results_dfs
        
        # Create comprehensive data with all scenarios and calculated columns
        scenario_data = {}
        scenarios = ['bear', 'neutral', 'bull']
        scenario_names = ['Bear', 'Neutral', 'Bull']
        
        for scenario, scenario_name in zip(scenarios, scenario_names):
            df = results_dfs[scenario].copy()
            
            # Add calculated supply columns for this scenario
            df['Cumulative Burn'] = df['Burn'].cumsum()
            df['Net Supply Change'] = df['Emissions'] - df['Burn']
            df['Cumulative Net Supply Change'] = (df['Emissions'] - df['Burn']).cumsum()
            df['Total Supply'] = df['Circulating Supply'] + df['Total Locked Supply']
            df['Locked % of Circulating'] = (df['Total Locked Supply'] / df['Circulating Supply'] * 100).fillna(0)
            df['Staking % of Circulating'] = (df['Staking Supply'] / df['Circulating Supply'] * 100).fillna(0)
            df['Collateral % of Circulating'] = (df['Locked Collateral'] / df['Circulating Supply'] * 100).fillna(0)
            
            # Add calculated fee metrics for this scenario
            # Average reward per subnet
            avg_reward_per_subnet = []
            for i, month in enumerate(df['Month']):
                active_subnets = df['Active Subnets'].iloc[i]
                monthly_emission = df['Subnet Rewards'].iloc[i]
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_per_subnet.append(avg_reward)
            df['Avg Reward per Subnet'] = avg_reward_per_subnet
            
            # Net reward per subnet (after fees)
            net_reward_per_subnet = []
            for i, month in enumerate(df['Month']):
                active_subnets = df['Active Subnets'].iloc[i]
                monthly_emission = df['Subnet Rewards'].iloc[i]
                subnet_reg_fees = df['Subnet Registration Fees'].iloc[i]
                subnet_maint_fees = df['Subnet Maintenance Fees'].iloc[i]
                
                gross_reward_per_subnet = monthly_emission / max(active_subnets, 1)
                amortized_reg_fee = subnet_reg_fees / max(active_subnets, 1) / 12
                maint_fee_per_subnet = subnet_maint_fees / max(active_subnets, 1)
                
                net_reward = gross_reward_per_subnet - maint_fee_per_subnet - amortized_reg_fee
                net_reward_per_subnet.append(max(0, net_reward))
            df['Net Reward per Subnet'] = net_reward_per_subnet
            
            # ROI per subnet (net reward / collateral)
            collateral_amount = st.session_state.subnet_collateral_amount
            df['Subnet ROI %'] = (df['Net Reward per Subnet'] / collateral_amount * 100).fillna(0)
            
            # Monthly fee revenue per active subnet
            df['Fees per Active Subnet'] = (df['Total Fees'] / df['Active Subnets']).fillna(0)
            
            # Monthly fee revenue per active ENT
            df['Fees per Active ENT'] = (df['Total Fees'] / df['Active ENTs']).fillna(0)
            
            scenario_data[scenario] = df
        
        # 1. Individual Scenario Downloads (Supply + Fee data combined)
        st.subheader("📥 Download Individual Scenarios")
        st.markdown("Each scenario includes all supply components, fees, and calculated metrics.")
        st.info("💡 **Note:** 'Vested' columns show cumulative totals (what's actually available in circulation), not monthly vesting amounts.")
        
        # Monthly downloads
        st.markdown("**📅 Monthly Data Downloads:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter out monthly vesting columns to keep only the important ones
            bear_df_clean = scenario_data['bear'].drop(columns=['Team Vesting', 'Investor Vesting', 'Foundation Vesting'])
            bear_csv = bear_df_clean.to_csv(index=False)
            st.download_button(
                label="🐻 Bear Scenario (Monthly)",
                data=bear_csv,
                file_name=f"bitrobot_bear_scenario_monthly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download Bear scenario monthly data with all supply components, fees, and calculated metrics."
            )
        
        with col2:
            # Filter out monthly vesting columns to keep only the important ones
            neutral_df_clean = scenario_data['neutral'].drop(columns=['Team Vesting', 'Investor Vesting', 'Foundation Vesting'])
            neutral_csv = neutral_df_clean.to_csv(index=False)
            st.download_button(
                label="⚖️ Neutral Scenario (Monthly)",
                data=neutral_csv,
                file_name=f"bitrobot_neutral_scenario_monthly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download Neutral scenario monthly data with all supply components, fees, and calculated metrics."
            )
        
        with col3:
            # Filter out monthly vesting columns to keep only the important ones
            bull_df_clean = scenario_data['bull'].drop(columns=['Team Vesting', 'Investor Vesting', 'Foundation Vesting'])
            bull_csv = bull_df_clean.to_csv(index=False)
            st.download_button(
                label="🐂 Bull Scenario (Monthly)",
                data=bull_csv,
                file_name=f"bitrobot_bull_scenario_monthly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download Bull scenario monthly data with all supply components, fees, and calculated metrics."
            )
        
        # Yearly aggregate downloads
        st.markdown("**📊 Yearly Aggregate Downloads:**")
        col1, col2, col3 = st.columns(3)
        
        def create_yearly_aggregate(df, scenario_name):
            """Create yearly aggregate data from monthly data"""
            yearly_data = []
            
            for year in range(1, 11):  # Years 1-10 (120 months)
                start_month = (year - 1) * 12 + 1
                end_month = year * 12
                
                # Filter data for this year
                year_data = df[(df['Month'] >= start_month) & (df['Month'] <= end_month)]
                
                if len(year_data) > 0:
                    # Calculate yearly aggregates
                    yearly_row = {
                        'Year': year,
                        'Start Month': start_month,
                        'End Month': end_month,
                        'Months in Year': len(year_data)
                    }
                    
                    # Sum columns (cumulative values)
                    sum_columns = [
                        'Emissions', 'Burn', 'Total Fees', 'ENT Registration Fees', 
                        'Subnet Registration Fees', 'Subnet Maintenance Fees', 'Cumulative Fees',
                        'Staking Rewards', 'Subnet Rewards', 'Cumulative Burn', 'Net Supply Change'
                    ]
                    
                    for col in sum_columns:
                        if col in year_data.columns:
                            yearly_row[f'Total {col}'] = year_data[col].sum()
                    
                    # Average columns (flow values)
                    avg_columns = [
                        'Active ENTs', 'Active Subnets', 'Staking Supply', 'Locked Collateral',
                        'Total Locked Supply', 'Per Staker Rewards', 'Staking APY',
                        'Avg Reward per Subnet', 'Net Reward per Subnet', 'Subnet ROI %',
                        'Fees per Active Subnet', 'Fees per Active ENT'
                    ]
                    
                    for col in avg_columns:
                        if col in year_data.columns:
                            yearly_row[f'Avg {col}'] = year_data[col].mean()
                    
                    # End-of-year values (stock values)
                    end_columns = [
                        'Circulating Supply', 'Team Vested', 'Investor Vested', 'Foundation Vested',
                        'Total Supply', 'Locked % of Circulating', 'Staking % of Circulating',
                        'Collateral % of Circulating'
                    ]
                    
                    for col in end_columns:
                        if col in year_data.columns:
                            yearly_row[f'End of Year {col}'] = year_data[col].iloc[-1]
                    
                    # Calculate some additional metrics
                    if 'Total Emissions' in yearly_row and 'Total Burn' in yearly_row:
                        yearly_row['Net Emissions'] = yearly_row['Total Emissions'] - yearly_row['Total Burn']
                    
                    if 'Total Subnet Rewards' in yearly_row and 'Total Subnet Registration Fees' in yearly_row and 'Total Subnet Maintenance Fees' in yearly_row:
                        yearly_row['Net Subnet Rewards'] = yearly_row['Total Subnet Rewards'] - yearly_row['Total Subnet Registration Fees'] - yearly_row['Total Subnet Maintenance Fees']
                    
                    yearly_data.append(yearly_row)
            
            return pd.DataFrame(yearly_data)
        
        with col1:
            bear_yearly = create_yearly_aggregate(scenario_data['bear'], 'Bear')
            bear_yearly_csv = bear_yearly.to_csv(index=False)
            st.download_button(
                label="🐻 Bear Scenario (Yearly)",
                data=bear_yearly_csv,
                file_name=f"bitrobot_bear_scenario_yearly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download Bear scenario yearly aggregate data with annual totals and averages."
            )
        
        with col2:
            neutral_yearly = create_yearly_aggregate(scenario_data['neutral'], 'Neutral')
            neutral_yearly_csv = neutral_yearly.to_csv(index=False)
            st.download_button(
                label="⚖️ Neutral Scenario (Yearly)",
                data=neutral_yearly_csv,
                file_name=f"bitrobot_neutral_scenario_yearly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download Neutral scenario yearly aggregate data with annual totals and averages."
            )
        
        with col3:
            bull_yearly = create_yearly_aggregate(scenario_data['bull'], 'Bull')
            bull_yearly_csv = bull_yearly.to_csv(index=False)
            st.download_button(
                label="🐂 Bull Scenario (Yearly)",
                data=bull_yearly_csv,
                file_name=f"bitrobot_bull_scenario_yearly_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download Bull scenario yearly aggregate data with annual totals and averages."
            )
        
        # 2. Comprehensive Comparison CSV
        st.subheader("📊 Download Scenario Comparison")
        st.markdown("Key metrics across all scenarios for easy side-by-side analysis.")
        
        # Select key columns for comparison
        key_columns = [
            'Month', 'Circulating Supply', 'Total Locked Supply', 'Emissions', 'Burn', 
            'Total Fees', 'Active Subnets', 'Per Staker Rewards', 'Staking APY',
            'Cumulative Burn', 'Net Supply Change', 'Locked % of Circulating',
            'Avg Reward per Subnet', 'Net Reward per Subnet', 'Subnet ROI %'
        ]
        
        comparison_df = pd.DataFrame({'Month': scenario_data['neutral']['Month']})
        
        for scenario, scenario_name in zip(scenarios, scenario_names):
            for col in key_columns:
                if col in scenario_data[scenario].columns:
                    comparison_df[f'{col} ({scenario_name})'] = scenario_data[scenario][col]
        
        # Add range columns for easy comparison
        comparison_df['Circulating Supply Range (Max-Min)'] = comparison_df[['Circulating Supply (Bear)', 'Circulating Supply (Neutral)', 'Circulating Supply (Bull)']].max(axis=1) - comparison_df[['Circulating Supply (Bear)', 'Circulating Supply (Neutral)', 'Circulating Supply (Bull)']].min(axis=1)
        comparison_df['Total Fees Range (Max-Min)'] = comparison_df[['Total Fees (Bear)', 'Total Fees (Neutral)', 'Total Fees (Bull)']].max(axis=1) - comparison_df[['Total Fees (Bear)', 'Total Fees (Neutral)', 'Total Fees (Bull)']].min(axis=1)
        comparison_df['Per Staker Rewards Range (Max-Min)'] = comparison_df[['Per Staker Rewards (Bear)', 'Per Staker Rewards (Neutral)', 'Per Staker Rewards (Bull)']].max(axis=1) - comparison_df[['Per Staker Rewards (Bear)', 'Per Staker Rewards (Neutral)', 'Per Staker Rewards (Bull)']].min(axis=1)
        comparison_df['Net Reward per Subnet Range (Max-Min)'] = comparison_df[['Net Reward per Subnet (Bear)', 'Net Reward per Subnet (Neutral)', 'Net Reward per Subnet (Bull)']].max(axis=1) - comparison_df[['Net Reward per Subnet (Bear)', 'Net Reward per Subnet (Neutral)', 'Net Reward per Subnet (Bull)']].min(axis=1)
        
        comparison_csv = comparison_df.to_csv(index=False)
        
        st.download_button(
            label="📈 Download Scenario Comparison CSV",
            data=comparison_csv,
            file_name=f"bitrobot_scenario_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download key metrics comparison across all scenarios for easy analysis."
        )
        
        # 3. Complete Master CSV (Everything)
        st.subheader("📋 Download Complete Master CSV")
        st.markdown("All data from all scenarios in one comprehensive file.")
        
        # Create master CSV with all scenarios and all columns
        master_df = scenario_data['neutral'].copy()
        
        # Rename neutral columns
        neutral_columns = {col: f"{col} (Neutral)" for col in master_df.columns if col != 'Month'}
        master_df = master_df.rename(columns=neutral_columns)
        
        # Add bear and bull columns
        for col in scenario_data['bear'].columns:
            if col != 'Month':
                master_df[f"{col} (Bear)"] = scenario_data['bear'][col]
        
        for col in scenario_data['bull'].columns:
            if col != 'Month':
                master_df[f"{col} (Bull)"] = scenario_data['bull'][col]
        
        master_csv = master_df.to_csv(index=False)
        
        st.download_button(
            label="📚 Download Complete Master CSV",
            data=master_csv,
            file_name=f"bitrobot_complete_master_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download all simulation data from all scenarios in one comprehensive file."
        )
        
        # Show column descriptions
        with st.expander("📋 CSV Column Descriptions"):
            st.markdown("""
            **Monthly Individual Scenario CSVs include:**
            - `Month`: Simulation month (0-120)
            - `Team Vested`: Cumulative team tokens vested (total available)
            - `Investor Vested`: Cumulative investor tokens vested (total available)
            - `Foundation Vested`: Cumulative foundation tokens vested (total available)
            - `Emissions`: Monthly token emissions
            - `Burn`: Monthly tokens burned (equals total fees)
            - `Circulating Supply`: Available tokens in circulation
            - `Total Locked Supply`: Tokens locked in staking + collateral
            - `ENT Registration Fees`: Monthly ENT registration fees
            - `Subnet Registration Fees`: Monthly subnet registration fees
            - `Subnet Maintenance Fees`: Monthly subnet maintenance fees
            - `Total Fees`: Combined monthly fees
            - `Cumulative Fees`: Running total of all fees
            - `Staking Supply`: Tokens locked in staking
            - `Staking Rewards`: Monthly rewards for stakers
            - `Subnet Rewards`: Monthly rewards for subnets
            - `Per Staker Rewards`: Average rewards per staking participant
            - `Staking APY`: Annual percentage yield for staking
            - `Cumulative Burn`: Running total of burned tokens
            - `Net Supply Change`: Monthly emissions minus burn
            - `Total Supply`: Circulating + locked supply
            - `Locked % of Circulating`: Percentage of circulating supply that is locked
            - `Staking % of Circulating`: Percentage of circulating supply in staking
            - `Collateral % of Circulating`: Percentage of circulating supply as collateral
            - `Avg Reward per Subnet`: Average gross rewards per subnet
            - `Net Reward per Subnet`: Average net rewards per subnet (after fees)
            - `Subnet ROI %`: Return on investment for subnet operators
            - `Fees per Active Subnet`: Average fees paid per active subnet
            - `Fees per Active ENT`: Average fees paid per active ENT
            
            **Yearly Individual Scenario CSVs include:**
            - `Year`: Year number (1-10)
            - `Start Month` / `End Month`: Month range for the year
            - `Months in Year`: Number of months with data
            - `Total [Metric]`: Annual sum for flow metrics (emissions, fees, rewards, etc.)
            - `Avg [Metric]`: Annual average for stock metrics (active entities, APY, etc.)
            - `End of Year [Metric]`: Value at the end of the year for cumulative metrics
            - `Net Emissions`: Total emissions minus total burn for the year
            - `Net Subnet Rewards`: Total subnet rewards minus total fees for the year
            
            **Note:** "Vested" columns show cumulative totals (what's actually available), not monthly vesting amounts.
            
            **Comparison CSV includes:**
            - Key metrics for all scenarios side-by-side
            - Range columns showing differences between scenarios
            - Focused on the most important metrics for analysis
            
            **Master CSV includes:**
            - All columns from all scenarios (including monthly vesting amounts)
            - Complete dataset for advanced analysis
            """)
        
        # Show data preview
        with st.expander("👀 Preview Comparison Data (First 10 rows)"):
            st.dataframe(comparison_df.head(10), use_container_width=True)
    
    else:
        st.info("Run the simulation first to generate downloadable results.")