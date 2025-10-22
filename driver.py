#!/usr/bin/env python3

import robo_supply_model
import numpy as np

NUM_MONTHS = 10*12

def sim1(
    seed=42,  # random seed for reproducible results
    initial_ents=100,  # initial number of entities
    initial_subnets=5,  # initial number of subnets
    ent_arrival_rate=1.0,  # average new entities per month
    subnet_arrival_rate=0.25,  # average new subnets per month
    ent_lifetime_months=24,  # entities depart after this many months
    subnet_lifetime_months=36,  # subnets depart after this many months
    initial_subnet_revenue=100000,  # $100k initial revenue per subnet per month
    max_subnet_revenue=1000000,  # $1M max revenue per subnet per month
    revenue_growth_months=12,  # months to reach max revenue
    revenue_burn_pct=0.3,  # 30% of revenue used for burn
    initial_token_price=0.30,  # $0.30 initial token price
    annual_price_growth_rate=0.01,  # 1% annual growth rate
    initial_target_staking_apy=0.08,  # 8% initial target staking APY
    final_target_staking_apy=0.04,  # 4% final target staking APY
    staking_apy_transition_months=48,  # months to transition from initial to final APY
    linear_start_emission=12_000_000,
    linear_end_emission=2_000_000,
    linear_total_emissions=200_000_000,
    airdrop_allocation=50_000_000,  # 50M airdrop allocation
    community_round_allocation=23_000_000,  # 23M community round allocation
):
    """
    Run a simulation with Poisson processes for arrivals and lifetime-based departures.
    
    Args:
        seed: Random seed for reproducible results
        initial_ents: Initial number of entities on the network
        initial_subnets: Initial number of subnets on the network
        ent_arrival_rate: Average number of new entities per month
        subnet_arrival_rate: Average number of new subnets per month  
        ent_lifetime_months: Number of months before entities depart
        subnet_lifetime_months: Number of months before subnets depart
        initial_subnet_revenue: Initial licensing revenue per subnet per month
        max_subnet_revenue: Maximum licensing revenue per subnet per month
        revenue_growth_months: Number of months to reach maximum revenue
        initial_token_price: Starting token price in USD
        annual_price_growth_rate: Annual growth rate for token price (e.g., 0.01 for 1%)
        initial_target_staking_apy: Initial target staking APY (e.g., 0.08 for 8%)
        final_target_staking_apy: Final target staking APY after transition (e.g., 0.04 for 4%)
        staking_apy_transition_months: Number of months to transition from initial to final APY
        linear_start_emission: Initial emission rate in M-ROBO per month
        linear_end_emission: Final emission rate in M-ROBO per month
        linear_total_emissions: Total emissions in M-ROBO
        airdrop_allocation: Total airdrop allocation in ROBO tokens
        community_round_allocation: Total community round allocation in ROBO tokens
    """
    # Create random number generator with seed for reproducible results
    rng = np.random.default_rng(seed)
    
    model = robo_supply_model.RoboSupplyModel(
        linear_start_emission=linear_start_emission,
        linear_end_emission=linear_end_emission,
        linear_total_emissions=linear_total_emissions,
        airdrop_allocation=airdrop_allocation,
        community_round_allocation=community_round_allocation,
    )
    
    # Initialize with starting entities and subnets
    entity_cohorts = [(0, initial_ents)]  # [(join_month, count), ...]
    subnet_cohorts = [(0, initial_subnets)]  # [(join_month, count), ...]

    for month in range(NUM_MONTHS):
        # Generate Poisson random variables for new arrivals
        new_ents = rng.poisson(ent_arrival_rate)
        new_subnets = rng.poisson(subnet_arrival_rate)
        
        # Add new cohorts
        if new_ents > 0: entity_cohorts.append((month, new_ents))
        if new_subnets > 0: subnet_cohorts.append((month, new_subnets))
        
        # Calculate departures based on lifetime
        ent_departures = 0
        subnet_departures = 0
        
        # Check which entities should depart this month
        entities_to_remove = []
        for i, (join_month, count) in enumerate(entity_cohorts):
            if month - join_month >= ent_lifetime_months:
                ent_departures += count
                entities_to_remove.append(i)
        for i in reversed(entities_to_remove):
            entity_cohorts.pop(i)
        
        # Check which subnets should depart this month
        subnets_to_remove = []
        for i, (join_month, count) in enumerate(subnet_cohorts):
            if month - join_month >= subnet_lifetime_months:
                subnet_departures += count
                subnets_to_remove.append(i)
        for i in reversed(subnets_to_remove):
            subnet_cohorts.pop(i)
        
        # Calculate total licensing revenue based on subnet cohorts
        total_licensing_revenue = 0
        for join_month, count in subnet_cohorts:
            months_on_network = month - join_month
            if months_on_network >= revenue_growth_months:
                # Subnet has reached max revenue
                revenue_per_subnet = max_subnet_revenue
            else:
                # Linear growth from initial to max revenue
                growth_factor = months_on_network / revenue_growth_months
                revenue_per_subnet = initial_subnet_revenue + (max_subnet_revenue - initial_subnet_revenue) * growth_factor
            
            total_licensing_revenue += count * revenue_per_subnet
        
        # Calculate token price with compound growth
        monthly_growth_rate = (1 + annual_price_growth_rate) ** (1/12) - 1
        current_token_price = initial_token_price * (1 + monthly_growth_rate) ** month
        
        # Calculate target staking APY with linear decrease
        if month < staking_apy_transition_months:
            # Linear interpolation from initial to final APY
            progress = month / staking_apy_transition_months
            current_target_staking_apy = initial_target_staking_apy + (final_target_staking_apy - initial_target_staking_apy) * progress
        else:
            # Stay at final APY after transition period
            current_target_staking_apy = final_target_staking_apy
        
        # Create network events for this month
        network_events = {
            'new_ents': new_ents,
            'new_subnets': new_subnets,
            'ent_departures': ent_departures,
            'subnet_departures': subnet_departures,
            'token_price': current_token_price,  # Dynamic token price
            'licensing_revenue_usd': total_licensing_revenue,  # Dynamic licensing revenue
            'revenue_burn_pct': revenue_burn_pct,  
        }

        # Create protocol parameters for this month
        protocol_params = {
            'ent_registration_fee': 1000,
            'subnet_registration_fee': 5000,
            'ent_collateral_amount': 5000,
            'subnet_collateral_amount': 100000,
            'subnet_maintenance_fee_pct': 0.25,
            'burn_emission_factor': 0.9,  # burn-based emission factor
            'target_staking_apy': current_target_staking_apy,  # Dynamic target staking APY
            'staking_percentage': 0.3,  # 30% of tokens staked
            'target_staking_percentage': 0.3,  # 30% target staking
            'max_staking_apy': 0.15,  # 15% max staking APY
            'maintenance_fee_burn_rate': 0.3,  # 0% of maintenance fees burned (100% to vault)
            # Note: fee_burn_rate is no longer used - registration fees are always burned, maintenance fees configurable
        }

        model.simulate_next_month(network_events, protocol_params)

    return model.get_results_dataframe()

def sim2(
    seed=42,  # random seed for reproducible results
    initial_ents=500,  # initial number of entities
    initial_subnets=10,  # initial number of subnets
    target_subnets_month_48=30,  # target number of subnets by month 48
    post_48_subnets_per_year=6,  # subnets added per year after month 48
    ent_arrival_rate=1.0,  # average new entities per month
    ent_lifetime_months=24,  # entities depart after this many months
    subnet_lifetime_months=36,  # subnets depart after this many months
    initial_subnet_revenue=100000,  # $100k initial revenue per subnet per month
    max_subnet_revenue=1000000,  # $1M max revenue per subnet per month
    revenue_growth_months=12,  # months to reach max revenue
    revenue_burn_pct=0.3,  # 30% of revenue used for burn
    initial_token_price=0.30,  # $0.30 initial token price
    annual_price_growth_rate=0.01,  # 1% annual growth rate
    initial_target_staking_apy=0.08,  # 8% initial target staking APY
    final_target_staking_apy=0.04,  # 4% final target staking APY
    staking_apy_transition_months=48,  # months to transition from initial to final APY
    linear_start_emission=12_000_000,
    linear_end_emission=2_000_000,
    linear_total_emissions=200_000_000,
    dynamic_staking_fees=False,
    max_maintenance_fee_pct=0.5,
    airdrop_allocation=50_000_000,  # 50M airdrop allocation
    community_round_allocation=23_000_000,  # 23M community round allocation
    airdrop_vesting_months=0, 
    community_round_vesting_months=0,
    subnet_maintenance_fee_pct=0.25,
):
    """
    Run a simulation with deterministic subnet growth and Poisson processes for entities.
    
    Subnet growth pattern:
    - Start with initial_subnets
    - Grow linearly to target_subnets_month_48 by month 48
    - After month 48, increase by post_48_subnets_per_year subnets per year
    - When subnets expire, create new ones to maintain target count
    
    Args:
        seed: Random seed for reproducible results
        initial_ents: Initial number of entities on the network
        initial_subnets: Initial number of subnets on the network
        target_subnets_month_48: Target number of subnets by month 48
        post_48_subnets_per_year: Number of subnets added per year after month 48
        ent_arrival_rate: Average number of new entities per month
        ent_lifetime_months: Number of months before entities depart
        subnet_lifetime_months: Number of months before subnets depart
        initial_subnet_revenue: Initial licensing revenue per subnet per month
        max_subnet_revenue: Maximum licensing revenue per subnet per month
        revenue_growth_months: Number of months to reach maximum revenue
        initial_token_price: Starting token price in USD
        annual_price_growth_rate: Annual growth rate for token price (e.g., 0.01 for 1%)
        initial_target_staking_apy: Initial target staking APY (e.g., 0.08 for 8%)
        final_target_staking_apy: Final target staking APY after transition (e.g., 0.04 for 4%)
        staking_apy_transition_months: Number of months to transition from initial to final APY
        linear_start_emission: Initial emission rate in M-ROBO per month
        linear_end_emission: Final emission rate in M-ROBO per month
        linear_total_emissions: Total emissions in M-ROBO
        dynamic_staking_fees: Whether to use dynamic staking fees
        max_maintenance_fee_pct: Maximum maintenance fee percentage
        airdrop_allocation: Total airdrop allocation in ROBO tokens
        community_round_allocation: Total community round allocation in ROBO tokens
        subnet_maintenance_fee_pct: Maintenance fee percentage for subnets
        airdrop_vesting_months: Number of months to vest the airdrop allocation
        community_round_vesting_months: Number of months to vest the community round allocation
    """
    # Create random number generator with seed for reproducible results
    rng = np.random.default_rng(seed)
    
    model = robo_supply_model.RoboSupplyModel(
        linear_start_emission=linear_start_emission,
        linear_end_emission=linear_end_emission,
        linear_total_emissions=linear_total_emissions,
        dynamic_staking_fees=dynamic_staking_fees,
        max_maintenance_fee_pct=max_maintenance_fee_pct,
        airdrop_allocation=airdrop_allocation,
        community_round_allocation=community_round_allocation,
        airdrop_vesting_months=airdrop_vesting_months,
        community_round_vesting_months=community_round_vesting_months,
    )
    
    # Initialize with starting entities and subnets
    entity_cohorts = [(0, initial_ents)]  # [(join_month, count), ...]
    subnet_cohorts = [(0, initial_subnets)]  # Start with initial subnets

    def get_target_subnet_count(month):
        """Calculate target number of subnets for a given month."""
        if month <= 48:
            # Linear growth from initial_subnets to target_subnets_month_48 over 48 months
            return initial_subnets + (target_subnets_month_48 - initial_subnets) * (month / 48)
        else:
            # After month 48, increase by post_48_subnets_per_year per year
            months_after_48 = month - 48
            return target_subnets_month_48 + (months_after_48 / 12) * post_48_subnets_per_year

    for month in range(NUM_MONTHS):
        # Generate Poisson random variables for new entity arrivals
        new_ents = rng.poisson(ent_arrival_rate)
        
        # For month 0, include the initial entities in new_ents
        if month == 0:
            new_ents += initial_ents
        
        # Add new entity cohorts
        if new_ents > 0 and month > 0:  # Don't double-add initial entities
            entity_cohorts.append((month, new_ents))
        
        # Calculate entity departures based on lifetime
        ent_departures = 0
        entities_to_remove = []
        for i, (join_month, count) in enumerate(entity_cohorts):
            if month - join_month >= ent_lifetime_months:
                ent_departures += count
                entities_to_remove.append(i)
        for i in reversed(entities_to_remove):
            entity_cohorts.pop(i)
        
        # Calculate subnet departures based on lifetime
        subnet_departures = 0
        subnets_to_remove = []
        for i, (join_month, count) in enumerate(subnet_cohorts):
            if month - join_month >= subnet_lifetime_months:
                subnet_departures += count
                subnets_to_remove.append(i)
        for i in reversed(subnets_to_remove):
            subnet_cohorts.pop(i)
        
        # Calculate current active subnets
        current_active_subnets = sum(count for _, count in subnet_cohorts)
        
        # Calculate target subnet count for this month
        target_subnet_count = get_target_subnet_count(month)
        
        # Determine how many new subnets to create
        # We need to replace departed subnets AND reach target count
        new_subnets_needed = max(0, target_subnet_count - current_active_subnets)
        new_subnets = new_subnets_needed
        
        # Add new subnet cohorts
        if new_subnets > 0: subnet_cohorts.append((month, new_subnets))
        
        # Special handling for month 0: we need to create the initial subnets
        if month == 0:
            new_subnets = initial_subnets  # Create initial subnets
            subnet_cohorts = [(0, initial_subnets)]  # Reset to just the initial cohort
        
        # Calculate total licensing revenue based on subnet cohorts
        total_licensing_revenue = 0
        for join_month, count in subnet_cohorts:
            months_on_network = month - join_month
            if months_on_network >= revenue_growth_months:
                # Subnet has reached max revenue
                revenue_per_subnet = max_subnet_revenue
            else:
                # Linear growth from initial to max revenue
                growth_factor = months_on_network / revenue_growth_months
                revenue_per_subnet = initial_subnet_revenue + (max_subnet_revenue - initial_subnet_revenue) * growth_factor
            
            total_licensing_revenue += count * revenue_per_subnet
        
        # Calculate token price with compound growth
        monthly_growth_rate = (1 + annual_price_growth_rate) ** (1/12) - 1
        current_token_price = initial_token_price * (1 + monthly_growth_rate) ** month
        
        # Calculate target staking APY with linear decrease
        if month < staking_apy_transition_months:
            # Linear interpolation from initial to final APY
            progress = month / staking_apy_transition_months
            current_target_staking_apy = initial_target_staking_apy + (final_target_staking_apy - initial_target_staking_apy) * progress
        else:
            # Stay at final APY after transition period
            current_target_staking_apy = final_target_staking_apy
        
        # Create network events for this month
        network_events = {
            'new_ents': new_ents,
            'new_subnets': new_subnets,
            'ent_departures': ent_departures,
            'subnet_departures': subnet_departures,
            'token_price': current_token_price,  # Dynamic token price
            'licensing_revenue_usd': total_licensing_revenue,  # Dynamic licensing revenue
            'revenue_burn_pct': revenue_burn_pct,  
        }

        # Create protocol parameters for this month
        protocol_params = {
            'ent_registration_fee': 100,
            'subnet_registration_fee': 5000,
            'ent_collateral_amount': 100,
            'subnet_collateral_amount': 100000,
            'subnet_maintenance_fee_pct': subnet_maintenance_fee_pct,
            'burn_emission_factor': 0.9,  # burn-based emission factor
            'target_staking_apy': current_target_staking_apy,  # Dynamic target staking APY
            'staking_percentage': 0.3,  # 30% of tokens staked
            'target_staking_percentage': 0.3,  # 30% target staking
            'max_staking_apy': 0.25,  # 15% max staking APY
            'maintenance_fee_burn_rate': 0.0,  # 0% of maintenance fees burned (100% to vault)
            # Note: fee_burn_rate is no longer used - registration fees are always burned, maintenance fees configurable
        }

        model.simulate_next_month(network_events, protocol_params)

    return model.get_results_dataframe()

if __name__ == "__main__":
    # Run simulation with default seed for reproducible results
    print("Running sim1 (Poisson process for subnets)...")
    results1 = sim1(seed=42)
    print(f"Sim1 completed with {len(results1)} months of data")
    print(f"Final month summary:")
    final_month1 = results1.iloc[-1]
    print(f"  Active ENTs: {final_month1['active_ents']}")
    print(f"  Active Subnets: {final_month1['active_subnets']}")
    print(f"  Circulating Supply: {final_month1['circulating_supply']:,.0f}")
    print(f"  Total Locked Supply: {final_month1['total_locked_supply']:,.0f}")
    print(f"  Total Collateral: {final_month1['total_collateral']:,.0f}")
    
    print("\nRunning sim2 (deterministic subnet growth)...")
    results2 = sim2(seed=42)
    print(f"Sim2 completed with {len(results2)} months of data")
    print(f"Final month summary:")
    final_month2 = results2.iloc[-1]
    print(f"  Active ENTs: {final_month2['active_ents']}")
    print(f"  Active Subnets: {final_month2['active_subnets']}")
    print(f"  Circulating Supply: {final_month2['circulating_supply']:,.0f}")
    print(f"  Total Locked Supply: {final_month2['total_locked_supply']:,.0f}")
    print(f"  Total Collateral: {final_month2['total_collateral']:,.0f}")