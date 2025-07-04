import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from emissions_model import BitRobotEmissionsModel
from fee_simulation import run_fee_simulation

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

# Create tabs
tab1, tab2 = st.tabs(["Emissions Schedule", "Fee Simulation"])

with tab1:
    st.header("Emissions Schedule Analysis")
    
    # Supply Configuration Expander
    with st.sidebar.expander("Supply Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            team_allocation = st.number_input(
                "Team Allocation (M)", 
                value=260, 
                min_value=0, 
                max_value=1000,
                help="Tokens allocated to team in millions"
            ) * 1_000_000
            
            investor_allocation = st.number_input(
                "Investor Allocation (M)", 
                value=260, 
                min_value=0, 
                max_value=1000,
                help="Tokens allocated to investors in millions"
            ) * 1_000_000
            
            dao_allocation = st.number_input(
                "Foundation Allocation (M)", 
                value=480, 
                min_value=0, 
                max_value=1000,
                help="Tokens allocated to Foundation in millions"
            ) * 1_000_000
            
            dao_initial_liquidity = st.number_input(
                "Foundation Initial Liquidity (M)", 
                value=50, 
                min_value=0, 
                max_value=100,
                help="Initial Foundation liquidity release in millions"
            ) * 1_000_000
            
            dao_target_48m = st.number_input(
                "Foundation Target 48M (M)", 
                value=480, 
                min_value=0, 
                max_value=500,
                help="Target Foundation tokens released by month 48 in millions"
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
            
            dao_vesting_months = st.number_input(
                "Foundation Vesting (months)", 
                value=48, 
                min_value=1, 
                max_value=120
            )
            
            fixed_emissions_target = st.number_input(
                "Fixed Emissions Target (M)", 
                value=272.3209, 
                min_value=0.0, 
                max_value=1000.0,
                help="Total fixed emissions target in millions"
            ) * 1_000_000
            
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
        
        burn_coefficient = st.number_input(
            "Burn Coefficient", 
            value=1_000_000, 
            min_value=100_000, 
            max_value=10_000_000,
            step=100_000
        )
        
        burn_pattern = st.selectbox(
            "Burn Pattern",
            ["logarithmic", "exponential", "sigmoid"],
            index=0
        )
        
        simulation_months = st.number_input(
            "Simulation Months", 
            value=120, 
            min_value=12, 
            max_value=240
        )
    
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
            value=5.0,
            min_value=0.1,
            max_value=20.0,
            step=0.1,
            help="Percentage of subnet rewards charged as maintenance fee",
            key="subnet_maintenance_fee_pct"
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
                value=24, 
                min_value=1, 
                max_value=60,
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
            # Initialize and run the emissions model
            model = BitRobotEmissionsModel(
                team_allocation=team_allocation,
                investor_allocation=investor_allocation,
                foundation_allocation=dao_allocation,
                foundation_initial_liquidity=dao_initial_liquidity,
                foundation_target_48m=dao_target_48m,
                fixed_emissions_target=fixed_emissions_target,
                team_cliff_months=team_cliff_months,
                team_vesting_months=team_vesting_months,
                dao_vesting_months=dao_vesting_months,
                t_burn=t_burn,
                burn_emission_factor=burn_emission_factor,
                burn_coefficient=burn_coefficient,
                burn_pattern=burn_pattern,
                simulation_months=simulation_months
            )
            
            model.run_simulation()
            results_df = model.get_results_dataframe()
            
            # Store emissions results in session state
            st.session_state.results_df = results_df
            st.session_state.model = model
            
            # Run fee simulation using the emission schedule
            emission_schedule = results_df['Emissions'].values
            
            # Get fee simulation parameters from sidebar
            F_base_ent = st.session_state.F_base_ent
            F_base_subnet = st.session_state.F_base_subnet
            subnet_maintenance_fee_pct = st.session_state.subnet_maintenance_fee_pct / 100.0
            lambda_ent_bear = st.session_state.lambda_ent_bear
            lambda_subnet_bear = st.session_state.lambda_subnet_bear
            starting_ents = st.session_state.starting_ents
            starting_subnets = st.session_state.starting_subnets
            lambda_ent_neutral = st.session_state.lambda_ent_neutral
            lambda_subnet_neutral = st.session_state.lambda_subnet_neutral
            lambda_ent_bull = st.session_state.lambda_ent_bull
            lambda_subnet_bull = st.session_state.lambda_subnet_bull
            ent_lifetime = st.session_state.ent_lifetime
            subnet_lifetime = st.session_state.subnet_lifetime
            
            # Run the fee simulation for all scenarios
            fee_results_bear = run_fee_simulation(
                epochs=len(emission_schedule),
                emission_schedule=emission_schedule,
                lambda_ent=lambda_ent_bear,
                lambda_subnet=lambda_subnet_bear,
                starting_ents=starting_ents,
                starting_subnets=starting_subnets,
                F_base_ent=F_base_ent,
                F_base_subnet=F_base_subnet,
                subnet_maintenance_fee_pct=subnet_maintenance_fee_pct,
                ent_lifetime=ent_lifetime,
                subnet_lifetime=subnet_lifetime,
                random_seed=42  # Bear scenario seed
            )
            
            fee_results_neutral = run_fee_simulation(
                epochs=len(emission_schedule),
                emission_schedule=emission_schedule,
                lambda_ent=lambda_ent_neutral,
                lambda_subnet=lambda_subnet_neutral,
                starting_ents=starting_ents,
                starting_subnets=starting_subnets,
                F_base_ent=F_base_ent,
                F_base_subnet=F_base_subnet,
                subnet_maintenance_fee_pct=subnet_maintenance_fee_pct,
                ent_lifetime=ent_lifetime,
                subnet_lifetime=subnet_lifetime,
                random_seed=43  # Neutral scenario seed
            )
            
            fee_results_bull = run_fee_simulation(
                epochs=len(emission_schedule),
                emission_schedule=emission_schedule,
                lambda_ent=lambda_ent_bull,
                lambda_subnet=lambda_subnet_bull,
                starting_ents=starting_ents,
                starting_subnets=starting_subnets,
                F_base_ent=F_base_ent,
                F_base_subnet=F_base_subnet,
                subnet_maintenance_fee_pct=subnet_maintenance_fee_pct,
                ent_lifetime=ent_lifetime,
                subnet_lifetime=subnet_lifetime,
                random_seed=44  # Bull scenario seed
            )
            
            # Store fee results in session state
            st.session_state.fee_results_bear = fee_results_bear
            st.session_state.fee_results_neutral = fee_results_neutral
            st.session_state.fee_results_bull = fee_results_bull
            
            st.success("Both simulations completed!")
    
    # Display plots if results are available
    if 'results_df' in st.session_state:
        results_df = st.session_state.results_df
        
        # Calculate components for the breakdown plot
        foundation_community_portion = 20.0 / 48.0
        community_portion = results_df['DAO Vested'] * foundation_community_portion + results_df['Emissions'].cumsum()
        team_portion = results_df['Team Vested']
        investor_portion = results_df['Investor Vested']
        foundation_team_portion = results_df['DAO Vested'] * (1 - foundation_community_portion)
        sum_val = community_portion + team_portion + foundation_team_portion + investor_portion
        
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
                'Component': 'Foundation',
                'Percentage': (foundation_team_portion.iloc[i] / sum_val.iloc[i]) * 100
            })
        breakdown_data = pd.DataFrame(breakdown_data)
        
        # Plot 1: Breakdown of Cumulative Supply
        breakdown_chart = alt.Chart(breakdown_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Percentage:Q', title='Percentage (%)'),
            color=alt.Color('Component:N', scale=alt.Scale(
                domain=['Community', 'Team', 'Investors', 'Foundation'],
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
        
        # Plot 2: Net Flow - properly structured
        net_flow_data = []
        for i, month in enumerate(results_df['Month']):
            net_flow_data.append({
                'Month': month,
                'Type': 'Burn',
                'Amount': results_df['Burn'].iloc[i] / 1e6
            })
            net_flow_data.append({
                'Month': month,
                'Type': 'Emissions',
                'Amount': results_df['Emissions'].iloc[i] / 1e6
            })
        net_flow_data = pd.DataFrame(net_flow_data)
        
        net_flow_chart = alt.Chart(net_flow_data).mark_line().encode(
            x=alt.X('Month:Q', title='Month'),
            y=alt.Y('Amount:Q', title='BRB (Millions)'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Burn', 'Emissions'],
                range=['#d62728', '#2ca02c']
            ))
        ).properties(
            title='Net Flow',
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
                'Component': 'Foundation',
                'Amount': results_df['DAO Vested'].iloc[i] / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Team',
                'Amount': results_df['Team Vested'].iloc[i] / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Emissions',
                'Amount': results_df['Emissions'].cumsum().iloc[i] / 1e9
            })
            cumulative_data.append({
                'Month': month,
                'Component': 'Burn',
                'Amount': results_df['Burn'].cumsum().iloc[i] / 1e9
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
                domain=['Foundation', 'Team', 'Emissions', 'Burn', 'Circulating Supply'],
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
        
        # Calculate the cumulative supply for each component
        foundation_community_portion = 20.0 / 48.0
        community_supply = month_data['DAO Vested'] * foundation_community_portion + results_df['Emissions'].cumsum()[target_month]
        team_supply = month_data['Team Vested']
        investor_supply = month_data['Investor Vested']
        foundation_team_supply = month_data['DAO Vested'] * (1 - foundation_community_portion)
        
        # Create data for pie chart
        pie_data = pd.DataFrame({
            'Component': ['Community', 'Team', 'Investors', 'Foundation'],
            'Supply': [community_supply, team_supply, investor_supply, foundation_team_supply]
        })
        
        # Calculate percentages for labels
        total_supply = pie_data['Supply'].sum()
        pie_data['Percentage'] = (pie_data['Supply'] / total_supply * 100).round(1)
        pie_data['Label'] = pie_data['Component'] + ': ' + pie_data['Percentage'].astype(str) + '%'
        
        pie_chart = alt.Chart(pie_data).mark_arc().encode(
            theta=alt.Theta('Supply:Q', type='quantitative'),
            color=alt.Color('Component:N', scale=alt.Scale(
                domain=['Community', 'Team', 'Investors', 'Foundation'],
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
        
        with col1:
            st.metric(
                "Total Supply at Month 48",
                f"{results_df[results_df['Month'] == 48]['Total Supply'].iloc[0] / 1e9:.2f}B BRB"
            )
        
        with col2:
            st.metric(
                "Circulating Supply at Month 48",
                f"{results_df[results_df['Month'] == 48]['Circulating Supply'].iloc[0] / 1e9:.2f}B BRB"
            )
        
        with col3:
            st.metric(
                "Total Emissions by Month 48",
                f"{results_df[results_df['Month'] == 48]['Emissions'].cumsum().iloc[-1] / 1e6:.1f}M BRB"
            )
        
        with col4:
            st.metric(
                "Total Burn by Month 48",
                f"{results_df[results_df['Month'] == 48]['Burn'].cumsum().iloc[-1] / 1e6:.1f}M BRB"
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
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Set the BRB token price in USD"
        )
    
    # Check if we have fee simulation results
    if 'fee_results_bear' not in st.session_state:
        st.info("Please run the simulation in the 'Emissions Schedule' tab to generate fee analysis plots.")
    else:
        st.info("Fee simulation based on the emissions schedule from the first tab.")
        
        # Store token price in session state for dynamic updates
        st.session_state.token_price = token_price
        
        # Display fee simulation results
        fee_results_bear = st.session_state.fee_results_bear
        fee_results_neutral = st.session_state.fee_results_neutral
        fee_results_bull = st.session_state.fee_results_bull
        
        # Plot 1: Active ENTs and Subnets Over Time (all scenarios)
        active_entities_data = []
        
        # Bearish scenario
        for i, epoch in enumerate(fee_results_bear['epoch']):
            active_entities_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Type': 'Active ENTs',
                'Count': fee_results_bear['active_ENTs'].iloc[i]
            })
            active_entities_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Type': 'Active Subnets',
                'Count': fee_results_bear['active_subnets'].iloc[i]
            })
        
        # Neutral scenario
        for i, epoch in enumerate(fee_results_neutral['epoch']):
            active_entities_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Type': 'Active ENTs',
                'Count': fee_results_neutral['active_ENTs'].iloc[i]
            })
            active_entities_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Type': 'Active Subnets',
                'Count': fee_results_neutral['active_subnets'].iloc[i]
            })
        
        # Bullish scenario
        for i, epoch in enumerate(fee_results_bull['epoch']):
            active_entities_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Type': 'Active ENTs',
                'Count': fee_results_bull['active_ENTs'].iloc[i]
            })
            active_entities_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Type': 'Active Subnets',
                'Count': fee_results_bull['active_subnets'].iloc[i]
            })
        
        active_entities_data = pd.DataFrame(active_entities_data)
        
        # Split data into ENTs and Subnets
        active_ents_data = active_entities_data[active_entities_data['Type'] == 'Active ENTs']
        active_subnets_data = active_entities_data[active_entities_data['Type'] == 'Active Subnets']
        
        # Create chart for Active ENTs
        active_ents_chart = alt.Chart(active_ents_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Count:Q', title='Count'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Active ENTs Over Time',
            width=450,
            height=300
        ).add_params(
            alt.selection_interval(bind='scales')
        )
        
        # Create chart for Active Subnets
        active_subnets_chart = alt.Chart(active_subnets_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Count:Q', title='Count'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Active Subnets Over Time',
            width=450,
            height=300
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
        )
        
        active_subnets_chart = alt.layer(active_subnets_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Plot 2: Fee Categories Over Time (all scenarios)
        fee_categories_data = []
        
        # Bearish scenario
        for i, epoch in enumerate(fee_results_bear['epoch']):
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Type': 'ENT Registration',
                'Amount': fee_results_bear['fees_ENT_reg'].iloc[i]
            })
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Type': 'Subnet Registration',
                'Amount': fee_results_bear['fees_subnet_reg'].iloc[i]
            })
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Type': 'Subnet Maintenance',
                'Amount': fee_results_bear['fees_subnet_maint'].iloc[i]
            })
        
        # Neutral scenario
        for i, epoch in enumerate(fee_results_neutral['epoch']):
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Type': 'ENT Registration',
                'Amount': fee_results_neutral['fees_ENT_reg'].iloc[i]
            })
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Type': 'Subnet Registration',
                'Amount': fee_results_neutral['fees_subnet_reg'].iloc[i]
            })
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Type': 'Subnet Maintenance',
                'Amount': fee_results_neutral['fees_subnet_maint'].iloc[i]
            })
        
        # Bullish scenario
        for i, epoch in enumerate(fee_results_bull['epoch']):
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Type': 'ENT Registration',
                'Amount': fee_results_bull['fees_ENT_reg'].iloc[i]
            })
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Type': 'Subnet Registration',
                'Amount': fee_results_bull['fees_subnet_reg'].iloc[i]
            })
            fee_categories_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Type': 'Subnet Maintenance',
                'Amount': fee_results_bull['fees_subnet_maint'].iloc[i]
            })
        
        fee_categories_data = pd.DataFrame(fee_categories_data)
        
        fee_categories_chart = alt.Chart(fee_categories_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Amount:Q', title='$BRB'),
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

        fee_categories_chart = alt.layer(fee_categories_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Plot 3: Cumulative Total Fees Collected (all scenarios)
        cumulative_fees_data = []
        
        # Bearish scenario
        for i, epoch in enumerate(fee_results_bear['epoch']):
            cumulative_fees_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Cumulative Fees': fee_results_bear['total_fees'].cumsum().iloc[i]
            })
        
        # Neutral scenario
        for i, epoch in enumerate(fee_results_neutral['epoch']):
            cumulative_fees_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Cumulative Fees': fee_results_neutral['total_fees'].cumsum().iloc[i]
            })
        
        # Bullish scenario
        for i, epoch in enumerate(fee_results_bull['epoch']):
            cumulative_fees_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Cumulative Fees': fee_results_bull['total_fees'].cumsum().iloc[i]
            })
        
        cumulative_fees_data = pd.DataFrame(cumulative_fees_data)
        
        cumulative_fees_chart = alt.Chart(cumulative_fees_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Cumulative Fees:Q', title='$BRB'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Cumulative Total Fees Collected',
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

        cumulative_fees_chart = alt.layer(cumulative_fees_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Plot 4: Total Fees per Month (all scenarios)
        total_fees_data = []
        
        # Bearish scenario
        for i, epoch in enumerate(fee_results_bear['epoch']):
            total_fees_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Total Fees': fee_results_bear['total_fees'].iloc[i]
            })
        
        # Neutral scenario
        for i, epoch in enumerate(fee_results_neutral['epoch']):
            total_fees_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Total Fees': fee_results_neutral['total_fees'].iloc[i]
            })
        
        # Bullish scenario
        for i, epoch in enumerate(fee_results_bull['epoch']):
            total_fees_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Total Fees': fee_results_bull['total_fees'].iloc[i]
            })
        
        total_fees_data = pd.DataFrame(total_fees_data)
        
        total_fees_chart = alt.Chart(total_fees_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Total Fees:Q', title='$BRB'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Total Fees per Month',
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

        total_fees_chart = alt.layer(total_fees_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Plot 5: Average Reward per Subnet (all scenarios)
        avg_reward_data = []
        
        # Get emission schedule from results
        emission_schedule = results_df['Emissions'].values
        
        # Bearish scenario
        for i, epoch in enumerate(fee_results_bear['epoch']):
            active_subnets = fee_results_bear['active_subnets'].iloc[i]
            monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
            avg_reward = monthly_emission / max(active_subnets, 1)  # Avoid division by zero
            avg_reward_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Avg Reward per Subnet': avg_reward
            })
        
        # Neutral scenario
        for i, epoch in enumerate(fee_results_neutral['epoch']):
            active_subnets = fee_results_neutral['active_subnets'].iloc[i]
            monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
            avg_reward = monthly_emission / max(active_subnets, 1)  # Avoid division by zero
            avg_reward_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Avg Reward per Subnet': avg_reward
            })
        
        # Bullish scenario
        for i, epoch in enumerate(fee_results_bull['epoch']):
            active_subnets = fee_results_bull['active_subnets'].iloc[i]
            monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
            avg_reward = monthly_emission / max(active_subnets, 1)  # Avoid division by zero
            avg_reward_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Avg Reward per Subnet': avg_reward
            })
        
        avg_reward_data = pd.DataFrame(avg_reward_data)
        
        # Filter to start from month 1
        avg_reward_data = avg_reward_data[avg_reward_data['Epoch'] >= 1]
        
        avg_reward_chart = alt.Chart(avg_reward_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Avg Reward per Subnet:Q', title='$BRB'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Average Reward per Subnet',
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

        avg_reward_chart = alt.layer(avg_reward_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Plot 6: Cumulative Fees as Fraction of Circulating Supply (all scenarios)
        fees_fraction_data = []
        
        # Get circulating supply from results
        circulating_supply = results_df['Circulating Supply'].values
        
        # Bearish scenario
        for i, epoch in enumerate(fee_results_bear['epoch']):
            cumulative_fees = fee_results_bear['total_fees'].cumsum().iloc[i]
            circulating_supply_at_epoch = circulating_supply[i] if i < len(circulating_supply) else circulating_supply[-1]
            fraction = (cumulative_fees / max(circulating_supply_at_epoch, 1)) * 100  # Convert to percentage
            fees_fraction_data.append({
                'Epoch': epoch,
                'Scenario': 'Bearish',
                'Fees as % of Circulating Supply': fraction
            })
        
        # Neutral scenario
        for i, epoch in enumerate(fee_results_neutral['epoch']):
            cumulative_fees = fee_results_neutral['total_fees'].cumsum().iloc[i]
            circulating_supply_at_epoch = circulating_supply[i] if i < len(circulating_supply) else circulating_supply[-1]
            fraction = (cumulative_fees / max(circulating_supply_at_epoch, 1)) * 100  # Convert to percentage
            fees_fraction_data.append({
                'Epoch': epoch,
                'Scenario': 'Neutral',
                'Fees as % of Circulating Supply': fraction
            })
        
        # Bullish scenario
        for i, epoch in enumerate(fee_results_bull['epoch']):
            cumulative_fees = fee_results_bull['total_fees'].cumsum().iloc[i]
            circulating_supply_at_epoch = circulating_supply[i] if i < len(circulating_supply) else circulating_supply[-1]
            fraction = (cumulative_fees / max(circulating_supply_at_epoch, 1)) * 100  # Convert to percentage
            fees_fraction_data.append({
                'Epoch': epoch,
                'Scenario': 'Bullish',
                'Fees as % of Circulating Supply': fraction
            })
        
        fees_fraction_data = pd.DataFrame(fees_fraction_data)
        
        fees_fraction_chart = alt.Chart(fees_fraction_data).mark_line().encode(
            x=alt.X('Epoch:Q', title='Month'),
            y=alt.Y('Fees as % of Circulating Supply:Q', title='% of Circulating Supply'),
            color=alt.Color('Scenario:N', scale=alt.Scale(
                domain=['Bearish', 'Neutral', 'Bullish'],
                range=['#d62728', '#2ca02c', '#1f77b4']
            ))
        ).properties(
            title='Cumulative Fees as % of Circulating Supply',
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

        fees_fraction_chart = alt.layer(fees_fraction_chart, vertical_line).configure_view(
            strokeWidth=0
        ).configure_axisLeft(
            labelPadding=10,
            titlePadding=10
        )
        
        # Display charts in a 2x2 grid
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
            st.altair_chart(fees_fraction_chart, use_container_width=True)
        
        with col6:
            st.altair_chart(avg_reward_chart, use_container_width=True)
        
        # Display summary statistics for all scenarios
        st.subheader("Fee Simulation Summary by Scenario")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Bearish Scenario**")
            st.metric("Total Fees", f"{fee_results_bear['total_fees'].sum():.0f} BRB")
            st.metric("Avg Monthly Fees", f"{fee_results_bear['total_fees'].mean():.1f} BRB")
            st.metric("Peak ENTs", f"{fee_results_bear['active_ENTs'].max():.0f}")
            st.metric("Peak Subnets", f"{fee_results_bear['active_subnets'].max():.0f}")
            
            # Calculate average reward per subnet for bearish scenario
            avg_reward_bear = []
            for i, epoch in enumerate(fee_results_bear['epoch']):
                active_subnets = fee_results_bear['active_subnets'].iloc[i]
                monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_bear.append(avg_reward)
            st.metric("Avg Reward per Subnet", f"{np.mean(avg_reward_bear):.1f} BRB")
        
        with col2:
            st.write("**Neutral Scenario**")
            st.metric("Total Fees", f"{fee_results_neutral['total_fees'].sum():.0f} BRB")
            st.metric("Avg Monthly Fees", f"{fee_results_neutral['total_fees'].mean():.1f} BRB")
            st.metric("Peak ENTs", f"{fee_results_neutral['active_ENTs'].max():.0f}")
            st.metric("Peak Subnets", f"{fee_results_neutral['active_subnets'].max():.0f}")
            
            # Calculate average reward per subnet for neutral scenario
            avg_reward_neutral = []
            for i, epoch in enumerate(fee_results_neutral['epoch']):
                active_subnets = fee_results_neutral['active_subnets'].iloc[i]
                monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_neutral.append(avg_reward)
            st.metric("Avg Reward per Subnet", f"{np.mean(avg_reward_neutral):.1f} BRB")
        
        with col3:
            st.write("**Bullish Scenario**")
            st.metric("Total Fees", f"{fee_results_bull['total_fees'].sum():.0f} BRB")
            st.metric("Avg Monthly Fees", f"{fee_results_bull['total_fees'].mean():.1f} BRB")
            st.metric("Peak ENTs", f"{fee_results_bull['active_ENTs'].max():.0f}")
            st.metric("Peak Subnets", f"{fee_results_bull['active_subnets'].max():.0f}")
            
            # Calculate average reward per subnet for bullish scenario
            avg_reward_bull = []
            for i, epoch in enumerate(fee_results_bull['epoch']):
                active_subnets = fee_results_bull['active_subnets'].iloc[i]
                monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
                avg_reward = monthly_emission / max(active_subnets, 1)
                avg_reward_bull.append(avg_reward)
            st.metric("Avg Reward per Subnet", f"{np.mean(avg_reward_bull):.1f} BRB")
        
        # Add yearly breakdown of average BRB per subnet
        st.subheader("Average Earnings per Subnet by Year")
        
        def calculate_yearly_avg_reward(fee_results, emission_schedule):
            """Calculate average BRB per subnet for each year"""
            yearly_avg = {}
            
            for year in range(1, 5):  # Years 1-4
                start_month = (year - 1) * 12 + 1
                end_month = year * 12
                
                year_rewards = []
                for i, epoch in enumerate(fee_results['epoch']):
                    if start_month <= epoch <= end_month:
                        active_subnets = fee_results['active_subnets'].iloc[i]
                        monthly_emission = emission_schedule[i] if i < len(emission_schedule) else 0
                        avg_reward = monthly_emission / max(active_subnets, 1)
                        year_rewards.append(avg_reward)
                
                if year_rewards:
                    yearly_avg[f'Year {year}'] = np.mean(year_rewards)
                else:
                    yearly_avg[f'Year {year}'] = 0
            
            return yearly_avg
        
        # Calculate yearly averages for each scenario
        bear_yearly = calculate_yearly_avg_reward(fee_results_bear, emission_schedule)
        neutral_yearly = calculate_yearly_avg_reward(fee_results_neutral, emission_schedule)
        bull_yearly = calculate_yearly_avg_reward(fee_results_bull, emission_schedule)
        
        # Display yearly breakdown in columns with both BRB and USD values
        col1, col2, col3 = st.columns(3)
        
        def display_earnings_metrics(yearly_data, token_price):
            """Display earnings metrics in both BRB and USD with improved formatting"""
            st.markdown("##### Annual Earnings per Subnet")
            
            # Create two columns for the headers
            brb_col, usd_col = st.columns(2)
            with brb_col:
                st.markdown("**BRB**")
            with usd_col:
                st.markdown("**USD**")
            
            # Display each year's data
            for year, avg_reward in yearly_data.items():
                usd_value = avg_reward * token_price
                
                # Create columns for this year's values
                year_brb_col, year_usd_col = st.columns(2)
                
                with year_brb_col:
                    st.markdown(f"**{year}:** {avg_reward:,.0f}")
                with year_usd_col:
                    st.markdown(f"${usd_value:,.2f}")
        
        with col1:
            st.markdown("### Bearish Scenario")
            st.markdown("---")
            display_earnings_metrics(bear_yearly, token_price)
        
        with col2:
            st.markdown("### Neutral Scenario")
            st.markdown("---")
            display_earnings_metrics(neutral_yearly, token_price)
        
        with col3:
            st.markdown("### Bullish Scenario")
            st.markdown("---")
            display_earnings_metrics(bull_yearly, token_price) 