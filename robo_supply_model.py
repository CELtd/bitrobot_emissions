import numpy as np
import pandas as pd

def print_yearly_subnet_rewards(df):
    """
    Print yearly subnet rewards summary from simulation dataframe.
    
    Args:
        df: DataFrame containing simulation results with 'month', 'base_emissions', 
            'token_price', and 'active_subnets' columns
    """
    # Add a year column to the dataframe
    df_copy = df.copy()
    df_copy['year'] = (df_copy['month'] // 12) + 1
    
    # Group by year and sum the base_emissions (subnet rewards)
    yearly_subnet_rewards = df_copy.groupby('year')['base_emissions'].sum()
    
    # Print the results
    print("Total Subnet Rewards by Year:")
    print("=" * 40)
    for year, total_rewards in yearly_subnet_rewards.items():
        print(f"Year {year}: {total_rewards:,.0f} ROBO ({total_rewards/1e6:.2f}M ROBO)")
    
    # Also show the USD value
    print("\nTotal Subnet Rewards by Year (USD):")
    print("=" * 40)
    for year in yearly_subnet_rewards.index:
        year_data = df_copy[df_copy['year'] == year]
        # Calculate weighted average token price for the year
        weighted_avg_price = (year_data['base_emissions'] * year_data['token_price']).sum() / year_data['base_emissions'].sum()
        total_rewards = yearly_subnet_rewards[year]
        usd_value = total_rewards * weighted_avg_price
        print(f"Year {year}: ${usd_value:,.0f} (avg price: ${weighted_avg_price:.2f})")
    
    # Show per-subnet rewards by year
    print("\nAverage Per-Subnet Rewards by Year:")
    print("=" * 40)
    for year in yearly_subnet_rewards.index:
        year_data = df_copy[df_copy['year'] == year]
        total_rewards = yearly_subnet_rewards[year]
        avg_active_subnets = year_data['active_subnets'].mean()
        per_subnet_rewards = total_rewards / avg_active_subnets if avg_active_subnets > 0 else 0
        # Calculate weighted average token price for the year
        weighted_avg_price = (year_data['base_emissions'] * year_data['token_price']).sum() / year_data['base_emissions'].sum()
        per_subnet_usd = per_subnet_rewards * weighted_avg_price
        print(f"Year {year}: {per_subnet_rewards:,.0f} ROBO (${per_subnet_usd:,.0f}) per subnet (avg {avg_active_subnets:.1f} subnets)")
    
    # Show the breakdown by regime (before/after month 48)
    print("\nSubnet Rewards by Regime:")
    print("=" * 40)
    
    # Before month 48 (inflationary regime)
    pre_48_data = df_copy[df_copy['month'] < 48]
    pre_48_total = pre_48_data['base_emissions'].sum()
    print(f"Pre-month 48 (Inflationary): {pre_48_total:,.0f} ROBO ({pre_48_total/1e6:.2f}M ROBO)")
    
    # After month 48 (deflationary regime)
    post_48_data = df_copy[df_copy['month'] >= 48]
    post_48_total = post_48_data['base_emissions'].sum()
    print(f"Post-month 48 (Deflationary): {post_48_total:,.0f} ROBO ({post_48_total/1e6:.2f}M ROBO)")
    
    print(f"Total: {pre_48_total + post_48_total:,.0f} ROBO")
    
    return yearly_subnet_rewards

# Usage:
# yearly_rewards = print_yearly_subnet_rewards(df)

class RoboSupplyModel:
    def __init__(
            self,
            team_allocation=269_000_000,
            investor_allocation=351_000_000,
            foundation_allocation=307_000_000,
            foundation_initial_liquidity=50_000_000,
            foundation_target_48m=307_000_000,
            team_cliff_months=12,
            team_vesting_months=36,  # Changed from 24 to 36 (3 years)
            investor_cliff_months=12,  # Added separate investor cliff
            investor_vesting_months=24,  # Added separate investor vesting (2 years)
            foundation_vesting_months=48,
            t_burn=48,
            burn_lookback_months=12,
            simulation_months=120,
            linear_start_emission=12_000_000,
            linear_end_emission=2_000_000,
            linear_total_emissions=200_000_000,
            dynamic_staking_fees=False,  # New parameter for fee-funded staking
            max_maintenance_fee_pct=0.5,  # Maximum maintenance fee percentage (50% default)
            airdrop_allocation=50_000_000,  # 50M airdrop allocation
            community_round_allocation=23_000_000,  # 23M community round allocation
        ):
        """
        Initialize the Robo supply model with single-step simulation capability.
        
        Tokenomics parameters are set in constructor and remain constant.
        Protocol parameters are passed to simulate_month() and can vary each month.
        """
        # Store tokenomics parameters
        self.team_allocation = team_allocation
        self.investor_allocation = investor_allocation
        self.foundation_allocation = foundation_allocation
        self.foundation_initial_liquidity = foundation_initial_liquidity
        self.foundation_target_48m = foundation_target_48m
        self.team_cliff_months = team_cliff_months
        self.team_vesting_months = team_vesting_months
        self.investor_cliff_months = investor_cliff_months
        self.investor_vesting_months = investor_vesting_months
        self.foundation_vesting_months = foundation_vesting_months
        self.t_burn = t_burn
        self.burn_lookback_months = burn_lookback_months
        self.simulation_months = simulation_months
        self.linear_start_emission = linear_start_emission
        self.linear_end_emission = linear_end_emission
        self.linear_total_emissions = linear_total_emissions
        self.dynamic_staking_fees = dynamic_staking_fees
        self.max_maintenance_fee_pct = max_maintenance_fee_pct
        self.airdrop_allocation = airdrop_allocation
        self.community_round_allocation = community_round_allocation
        
        # Initialize fixed emissions schedule
        self.fixed_emissions = np.zeros(simulation_months + 1)
        self._set_linear_fixed_emissions()
        
        # Initialize internal state tracking
        self.cumulative_emissions = 0
        self.cumulative_burn = 0
        self.cumulative_fixed_emissions = 0
        self.cumulative_fees = 0
        self.burn_history = []  # Track burn for lookback calculations
        self.fee_vault = 0  # Track accumulated fees and excess staking rewards
        # Stateful simulation tracking
        self.current_month = 0
        self.history = []
        # Network state tracking
        self.active_ents = 0
        self.active_subnets = 0
        
    def _set_linear_fixed_emissions(self):
        """Set a linear emissions schedule that scales from start to end value over 48 months."""
        required_sum = self.linear_total_emissions * 2 / 48
        current_sum = self.linear_start_emission + self.linear_end_emission
        
        if abs(current_sum - required_sum) > 1:
            scale_factor = required_sum / current_sum
            adjusted_start = self.linear_start_emission * scale_factor
            adjusted_end = self.linear_end_emission * scale_factor
        else:
            adjusted_start = self.linear_start_emission
            adjusted_end = self.linear_end_emission
        print(adjusted_start, adjusted_end)
        
        for month in range(0, 48):
            emission = adjusted_start + (adjusted_end - adjusted_start) * month / 47
            self.fixed_emissions[month] = emission
            
    def _calculate_team_vested(self, month):
        """Calculate total team tokens vested by given month."""
        if month <= self.team_cliff_months:
            return 0
        
        vesting_start = self.team_cliff_months + 1
        if month < vesting_start:
            return 0
        
        vesting_end = vesting_start + self.team_vesting_months
        if month >= vesting_end:
            return self.team_allocation
        
        months_vested = month - vesting_start + 1
        monthly_vesting = self.team_allocation / self.team_vesting_months
        return months_vested * monthly_vesting
        
    def _calculate_investor_vested(self, month):
        """Calculate total investor tokens vested by given month."""
        if month <= self.investor_cliff_months:
            return 0
        
        vesting_start = self.investor_cliff_months + 1
        if month < vesting_start:
            return 0
        
        vesting_end = vesting_start + self.investor_vesting_months
        if month >= vesting_end:
            return self.investor_allocation
        
        months_vested = month - vesting_start + 1
        monthly_vesting = self.investor_allocation / self.investor_vesting_months
        return months_vested * monthly_vesting
        
    def _calculate_foundation_vested(self, month):
        """Calculate total foundation tokens vested by given month."""
        if month == 0:
            return self.foundation_initial_liquidity
        
        remaining_to_48m = self.foundation_target_48m - self.foundation_initial_liquidity
        monthly_vesting = remaining_to_48m / self.foundation_vesting_months
        
        if month >= self.foundation_vesting_months:
            return self.foundation_target_48m
        
        return self.foundation_initial_liquidity + (month * monthly_vesting)
        
    def _calculate_airdrop_vested(self, month):
        """Calculate total airdrop tokens vested by given month."""
        if month >= 48:
            return self.airdrop_allocation
        
        # Linear vesting over 48 months
        monthly_vesting = self.airdrop_allocation / 48
        return month * monthly_vesting
        
    def _calculate_community_round_vested(self, month):
        """Calculate total community round tokens vested by given month."""
        if month >= 48:
            return self.community_round_allocation
        
        # Linear vesting over 48 months
        monthly_vesting = self.community_round_allocation / 48
        return month * monthly_vesting
        
    def _calculate_airdrop_monthly_emission(self, month):
        """Calculate airdrop tokens emitted (vested) this month."""
        if month >= 48:
            return 0  # No more airdrop emissions after month 48
        
        # Linear vesting over 48 months
        monthly_vesting = self.airdrop_allocation / 48
        return monthly_vesting
        
    def _calculate_community_round_monthly_emission(self, month):
        """Calculate community round tokens emitted (vested) this month."""
        if month >= 48:
            return 0  # No more community round emissions after month 48
        
        # Linear vesting over 48 months
        monthly_vesting = self.community_round_allocation / 48
        return monthly_vesting
    
    def _calculate_base_emissions(self, month, burn_emission_factor):
        """Calculate base emissions for the given month."""
        if month < self.t_burn:
            return self.fixed_emissions[month]
        else:
            if len(self.burn_history) >= self.burn_lookback_months:
                burn_sum = sum(self.burn_history[-self.burn_lookback_months:])
                return burn_emission_factor * burn_sum / self.burn_lookback_months
            else:
                return self.fixed_emissions[month]
    
    def simulate_next_month(self, network_events, protocol_params):
        """
        Simulate the next month of the supply model.
        
        Args:
            network_events: Dict with network events and external data for this month
                {
                    'new_ents': int,
                    'new_subnets': int,
                    'ent_departures': int,
                    'subnet_departures': int,
                    'token_price': float,
                    'licensing_revenue_usd': float,
                    'revenue_burn_pct': float,
                }
            protocol_params: Dict with protocol parameters for this month
                {
                    'ent_registration_fee': float,
                    'subnet_registration_fee': float,
                    'ent_collateral_amount': float,
                    'subnet_collateral_amount': float,
                    'subnet_maintenance_fee_pct': float,
                    'burn_emission_factor': float,
                    'target_staking_apy': float,
                    'staking_percentage': float,
                    'target_staking_percentage': float,
                    'max_staking_apy': float,
                    'maintenance_fee_burn_rate': float,
                }
        
        Returns:
            dict: Results for this month
        """
        month = self.current_month
        # Update cumulative values from previous state
        if self.history:
            previous_state = self.history[-1]
            self.cumulative_emissions = previous_state['cumulative_emissions']
            self.cumulative_burn = previous_state['cumulative_burn']
            self.cumulative_fixed_emissions = previous_state['cumulative_fixed_emissions']
            self.cumulative_fees = previous_state['cumulative_fees']
            self.fee_vault = previous_state.get('fee_vault', 0)
            self.active_ents = previous_state['active_ents']
            self.active_subnets = previous_state['active_subnets']
        else:
            self.cumulative_emissions = 0
            self.cumulative_burn = 0
            self.cumulative_fixed_emissions = 0
            self.cumulative_fees = 0
            self.burn_history = []
            self.fee_vault = 0
            self.active_ents = 0
            self.active_subnets = 0
        
        # Calculate vesting for this month
        team_vested = self._calculate_team_vested(month)
        investor_vested = self._calculate_investor_vested(month)
        foundation_vested = self._calculate_foundation_vested(month)
        airdrop_vested = self._calculate_airdrop_vested(month)
        community_round_vested = self._calculate_community_round_vested(month)
        
        # Calculate monthly emissions for this month
        airdrop_monthly_emission = self._calculate_airdrop_monthly_emission(month)
        community_round_monthly_emission = self._calculate_community_round_monthly_emission(month)
        
        # Calculate base emissions
        base_emissions = self._calculate_base_emissions(month, protocol_params['burn_emission_factor'])
        
        # Calculate circulating supply before locking
        circulating_before_locking = (
            team_vested + 
            investor_vested + 
            foundation_vested + 
            airdrop_vested +
            community_round_vested +
            self.cumulative_emissions - 
            self.cumulative_burn
        )
        
        # Calculate fees first
        # Note: Registration fees (ENT + subnet) will be burned, maintenance fees will go to vault
        ent_registration_fees = network_events['new_ents'] * protocol_params['ent_registration_fee']
        subnet_registration_fees = network_events['new_subnets'] * protocol_params['subnet_registration_fee']
        
        # Initialize vault_contribution for both regimes
        vault_contribution = 0
        
        # Staking calculations
        if month < self.t_burn:
            subnet_rewards = self.fixed_emissions[month]
        else:
            subnet_rewards = base_emissions
            
        target_staked_amount = circulating_before_locking * protocol_params['target_staking_percentage']
        actual_staked_amount = circulating_before_locking * protocol_params['staking_percentage']
        monthly_target_apy = protocol_params['target_staking_apy'] / 12
        max_monthly_apy = protocol_params['max_staking_apy'] / 12
        
        if self.dynamic_staking_fees:
            # In dynamic mode, staking is funded entirely through fees
            budgeted_rewards = actual_staked_amount * monthly_target_apy
            target_staking_rewards = budgeted_rewards  # For result dictionary
            if actual_staked_amount > 0:
                potential_apy = (budgeted_rewards / actual_staked_amount) * 12
            else:
                potential_apy = 0
                
            # Apply APY cap if needed
            if potential_apy > protocol_params['max_staking_apy']:
                staking_rewards = actual_staked_amount * max_monthly_apy
                vault_contribution = budgeted_rewards - staking_rewards
            else:
                staking_rewards = budgeted_rewards
                vault_contribution = 0
                
            # Store excess in vault for future use
            self.fee_vault += max(0, vault_contribution)
            # additional_staking_emissions will be set in the fee calculation section below
            additional_staking_emissions = 0  # Initialize, will be updated in fee calculation
            fee_vault_distribution = 0
        else:
            # Original logic for static fee mode
            if month < self.t_burn:
                budgeted_rewards = target_staked_amount * monthly_target_apy
                target_staking_rewards = budgeted_rewards  # For result dictionary
                if actual_staked_amount > 0:
                    potential_apy = (budgeted_rewards / actual_staked_amount) * 12
                else:
                    potential_apy = 0
                if potential_apy > protocol_params['max_staking_apy']:
                    staking_rewards = actual_staked_amount * max_monthly_apy
                    vault_contribution = budgeted_rewards - staking_rewards
                else:
                    staking_rewards = budgeted_rewards
                    vault_contribution = 0
                self.fee_vault += max(0, vault_contribution)
                additional_staking_emissions = staking_rewards
                fee_vault_distribution = 0
            else:
                # Deflationary regime - use fee vault for staking rewards
                target_staking_rewards = actual_staked_amount * monthly_target_apy
                if self.fee_vault >= target_staking_rewards:
                    fee_vault_distribution = target_staking_rewards
                    additional_staking_emissions = 0
                else:
                    fee_vault_distribution = self.fee_vault
                    additional_staking_emissions = target_staking_rewards - fee_vault_distribution
                staking_rewards = fee_vault_distribution + additional_staking_emissions
        
        total_emissions = subnet_rewards + additional_staking_emissions + airdrop_monthly_emission + community_round_monthly_emission
        
        # Calculate maintenance fees and total fees
        rewards_per_subnet = subnet_rewards / max(self.active_subnets, 1)
        
        if self.dynamic_staking_fees:
            # Calculate required staking budget
            actual_staked_amount = circulating_before_locking * protocol_params['staking_percentage']
            monthly_target_apy = protocol_params['target_staking_apy'] / 12
            required_staking_budget = actual_staked_amount * monthly_target_apy
            
            # Use vault first if available
            vault_used_for_staking = min(self.fee_vault, required_staking_budget)
            remaining_staking_needs = required_staking_budget - vault_used_for_staking
            
            # Back-compute maintenance fee percentage to cover remaining staking needs
            if subnet_rewards > 0:
                dynamic_maintenance_fee_pct = remaining_staking_needs / subnet_rewards
            else:
                dynamic_maintenance_fee_pct = protocol_params['subnet_maintenance_fee_pct']
            
            # Apply cap to maintenance fee percentage
            if dynamic_maintenance_fee_pct > self.max_maintenance_fee_pct:
                capped_maintenance_fee_pct = self.max_maintenance_fee_pct
                # Calculate how much staking budget we can't cover with fees
                max_fee_budget = subnet_rewards * self.max_maintenance_fee_pct
                uncovered_staking_needs = remaining_staking_needs - max_fee_budget
                # Use inflation to cover the remainder
                additional_staking_emissions = max(0, uncovered_staking_needs)
            else:
                capped_maintenance_fee_pct = dynamic_maintenance_fee_pct
                additional_staking_emissions = 0  # No additional emissions needed
            
            subnet_maintenance_fees = self.active_subnets * rewards_per_subnet * capped_maintenance_fee_pct
        else:
            # Use static maintenance fee percentage
            subnet_maintenance_fees = self.active_subnets * rewards_per_subnet * protocol_params['subnet_maintenance_fee_pct']
            vault_used_for_staking = 0
            dynamic_maintenance_fee_pct = protocol_params['subnet_maintenance_fee_pct']
            capped_maintenance_fee_pct = protocol_params['subnet_maintenance_fee_pct']
        
        total_fees = ent_registration_fees + subnet_registration_fees + subnet_maintenance_fees
        
        # Handle selective fee routing: registration fees burned, maintenance fees configurable
        # This implements the desired behavior where:
        # - ENT registration fees are burned (deflationary pressure)
        # - Subnet registration fees are burned (deflationary pressure)  
        # - Subnet maintenance fees are split between burn and vault based on maintenance_fee_burn_rate
        # - Revenue percentage is burned separately (additional deflation)
        
        # Registration fees (ENT + subnet) are always burned
        registration_fees_burned = ent_registration_fees + subnet_registration_fees
        
        # Maintenance fees are split between burn and vault based on configuration
        maintenance_fee_burn_rate = protocol_params.get('maintenance_fee_burn_rate', 0.0)  # Default: 0% burned
        maintenance_fees_burned = subnet_maintenance_fees * maintenance_fee_burn_rate
        maintenance_fees_to_vault = subnet_maintenance_fees * (1.0 - maintenance_fee_burn_rate)
        
        # Total fees burned (registration fees + portion of maintenance fees)
        fee_burn = registration_fees_burned + maintenance_fees_burned
        
        # Total fees going to vault (remaining portion of maintenance fees)
        fee_vault_contribution = maintenance_fees_to_vault
        
        if self.dynamic_staking_fees:
            # In dynamic mode, vault is used immediately for staking, then replenished
            self.fee_vault = self.fee_vault - vault_used_for_staking + fee_vault_contribution
        else:
            # In static mode, use existing logic
            self.fee_vault += fee_vault_contribution - fee_vault_distribution
        
        # Calculate revenue-based burn
        revenue_burn_tokens = (network_events['licensing_revenue_usd'] * network_events['revenue_burn_pct']) / network_events['token_price']
        
        # Total burn (fees + revenue)
        total_burn = fee_burn + revenue_burn_tokens
        
        # Update cumulative values
        self.cumulative_emissions += total_emissions
        self.cumulative_burn += total_burn
        self.cumulative_fixed_emissions += self.fixed_emissions[month]
        self.cumulative_fees += total_fees
        self.burn_history.append(total_burn)
        
        # Update active entities and subnets
        self.active_ents += network_events['new_ents'] - network_events['ent_departures']
        self.active_subnets += network_events['new_subnets'] - network_events['subnet_departures']
        
        # Calculate staking metrics
        if actual_staked_amount > 0:
            staking_apy = (staking_rewards / actual_staked_amount) * 12 * 100
        else:
            staking_apy = 0
        
        staking_supply = circulating_before_locking * protocol_params['staking_percentage']
        ent_collateral_supply = self.active_ents * protocol_params['ent_collateral_amount']
        subnet_collateral_supply = self.active_subnets * protocol_params['subnet_collateral_amount']
        total_locked_supply = ent_collateral_supply + subnet_collateral_supply + staking_supply
        circulating_supply = circulating_before_locking - total_locked_supply
        
        result = {
            'month': month,
            'team_vested': team_vested,
            'investor_vested': investor_vested,
            'foundation_vested': foundation_vested,
            'airdrop_vested': airdrop_vested,
            'community_round_vested': community_round_vested,
            'airdrop_monthly_emission': airdrop_monthly_emission,
            'community_round_monthly_emission': community_round_monthly_emission,
            'base_emissions': base_emissions,
            'additional_staking_emissions': additional_staking_emissions,
            'total_emissions': total_emissions,
            'burn': total_burn,
            'circulating_supply': circulating_supply,
            'total_locked_supply': total_locked_supply,
            'staking_supply': staking_supply,
            'ent_collateral': ent_collateral_supply,
            'subnet_collateral': subnet_collateral_supply,
            'staking_rewards': staking_rewards,
            'subnet_rewards': subnet_rewards,
            'staking_apy': staking_apy,
            'target_staking_budget': budgeted_rewards if month < self.t_burn else target_staking_rewards,
            'actual_staking_budget': actual_staked_amount * monthly_target_apy if month < self.t_burn else target_staking_rewards,
            'subnet_guaranteed_emissions': subnet_rewards if month < self.t_burn else subnet_rewards,
            'ent_registration_fees': ent_registration_fees,
            'subnet_registration_fees': subnet_registration_fees,
            'subnet_maintenance_fees': subnet_maintenance_fees,
            'total_fees': total_fees,
            'active_ents': self.active_ents,
            'active_subnets': self.active_subnets,
            'total_collateral': ent_collateral_supply + subnet_collateral_supply,
            'new_ents': network_events['new_ents'],
            'new_subnets': network_events['new_subnets'],
            'ent_departures': network_events['ent_departures'],
            'subnet_departures': network_events['subnet_departures'],
            'cumulative_emissions': self.cumulative_emissions,
            'cumulative_burn': self.cumulative_burn,
            'cumulative_fixed_emissions': self.cumulative_fixed_emissions,
            'cumulative_fees': self.cumulative_fees,
            'ent_registration_fee': protocol_params['ent_registration_fee'],
            'subnet_registration_fee': protocol_params['subnet_registration_fee'],
            'ent_collateral_amount': protocol_params['ent_collateral_amount'],
            'subnet_collateral_amount': protocol_params['subnet_collateral_amount'],
            'subnet_maintenance_fee_pct': protocol_params['subnet_maintenance_fee_pct'],
            'burn_emission_factor': protocol_params['burn_emission_factor'],
            'target_staking_apy': protocol_params['target_staking_apy'],
            'staking_percentage': protocol_params['staking_percentage'],
            'target_staking_percentage': protocol_params['target_staking_percentage'],
            'max_staking_apy': protocol_params['max_staking_apy'],
            'vault_contribution': vault_contribution,
            'fee_vault': self.fee_vault,
            'fee_vault_contribution': fee_vault_contribution,
            'fee_vault_distribution': fee_vault_distribution,
            'vault_used_for_staking': vault_used_for_staking if self.dynamic_staking_fees else 0,
            'dynamic_maintenance_fee_pct': dynamic_maintenance_fee_pct if self.dynamic_staking_fees else protocol_params['subnet_maintenance_fee_pct'],
            'capped_maintenance_fee_pct': capped_maintenance_fee_pct if self.dynamic_staking_fees else protocol_params['subnet_maintenance_fee_pct'],
            'token_price': network_events['token_price'],
            'licensing_revenue_usd': network_events['licensing_revenue_usd'],
            'revenue_burn_pct': network_events['revenue_burn_pct'],
            'fee_burn': fee_burn,
            'revenue_burn_tokens': revenue_burn_tokens,
            'registration_fees_burned': registration_fees_burned,
            'maintenance_fees_burned': maintenance_fees_burned,
            'maintenance_fees_to_vault': maintenance_fees_to_vault,
            'maintenance_fee_burn_rate': maintenance_fee_burn_rate
        }
        # Store result and increment month
        self.history.append(result)
        self.current_month += 1
        return result

    def get_results_dataframe(self):
        """Return the simulation results as a pandas DataFrame."""
        return pd.DataFrame(self.history)

    def reset(self):
        """Reset the simulation to month 0."""
        self.current_month = 0
        self.history = []
        self.cumulative_emissions = 0
        self.cumulative_burn = 0
        self.cumulative_fixed_emissions = 0
        self.cumulative_fees = 0
        self.burn_history = []
        self.fee_vault = 0
        self.active_ents = 0
        self.active_subnets = 0 