import numpy as np
import pandas as pd

## TODO: we need to update how the APY is dealt with in the simulation.
##  the subnet_min_emissions_pct and staking_emissions_cap_pct are no longer valid.

class RoboSupplyModel:
    def __init__(self,
                 team_allocation=269_000_000,
                 investor_allocation=351_000_000,
                 foundation_allocation=307_000_000,
                 foundation_initial_liquidity=50_000_000,
                 foundation_target_48m=307_000_000,
                 team_cliff_months=12,
                 team_vesting_months=24,
                 foundation_vesting_months=48,
                 t_burn=48,
                 burn_lookback_months=12,
                 simulation_months=120,
                 linear_start_emission=12_000_000,
                 linear_end_emission=2_000_000,
                 linear_total_emissions=200_000_000,
                 staking_emissions_cap_pct=0.1):
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
        self.foundation_vesting_months = foundation_vesting_months
        self.t_burn = t_burn
        self.burn_lookback_months = burn_lookback_months
        self.simulation_months = simulation_months
        self.linear_start_emission = linear_start_emission
        self.linear_end_emission = linear_end_emission
        self.linear_total_emissions = linear_total_emissions
        self.staking_emissions_cap_pct = staking_emissions_cap_pct
        
        # Initialize fixed emissions schedule
        self.fixed_emissions = np.zeros(simulation_months + 1)
        self._set_linear_fixed_emissions()
        
        # Initialize internal state tracking
        self.cumulative_emissions = 0
        self.cumulative_burn = 0
        self.cumulative_fixed_emissions = 0
        self.cumulative_fees = 0
        self.burn_history = []  # Track burn for lookback calculations
        self.staking_vault = 0  # Track cumulative excess staking rewards
        self.fee_vault = 0  # Track accumulated fees
        # Stateful simulation tracking
        self.current_month = 0
        self.history = []
        
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
            
        for month in range(1, 49):
            emission = adjusted_start + (adjusted_end - adjusted_start) * (month - 1) / 47
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
        if month <= self.team_cliff_months:
            return 0
        
        vesting_start = self.team_cliff_months + 1
        if month < vesting_start:
            return 0
        
        vesting_end = vesting_start + self.team_vesting_months
        if month >= vesting_end:
            return self.investor_allocation
        
        months_vested = month - vesting_start + 1
        monthly_vesting = self.investor_allocation / self.team_vesting_months
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
    
    def simulate_next_month(self, network_state, protocol_params):
        """
        Simulate the next month of the supply model.
        
        Args:
            network_state: Dict with current network state
                {
                    'active_ents': int,
                    'active_subnets': int,
                    'total_collateral': float,
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
                    'subnet_collateral_amount': float,
                    'subnet_maintenance_fee_pct': float,
                    'burn_emission_factor': float,
                    'target_staking_apy': float,
                    'staking_percentage': float,
                    'target_staking_percentage': float,
                    'max_staking_apy': float,
                    'fee_burn_rate': float,
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
            self.staking_vault = previous_state.get('staking_vault', 0)
            self.fee_vault = previous_state.get('fee_vault', 0)
        else:
            self.cumulative_emissions = 0
            self.cumulative_burn = 0
            self.cumulative_fixed_emissions = 0
            self.cumulative_fees = 0
            self.burn_history = []
            self.staking_vault = 0
            self.fee_vault = 0
        
        # Calculate vesting for this month
        team_vested = self._calculate_team_vested(month)
        investor_vested = self._calculate_investor_vested(month)
        foundation_vested = self._calculate_foundation_vested(month)
        
        # Calculate base emissions
        base_emissions = self._calculate_base_emissions(month, protocol_params['burn_emission_factor'])
        
        # Calculate circulating supply before locking
        circulating_before_locking = (
            team_vested + 
            investor_vested + 
            foundation_vested + 
            self.cumulative_emissions - 
            self.cumulative_burn
        )
        
        # Calculate fees first
        ent_registration_fees = network_state['new_ents'] * protocol_params['ent_registration_fee']
        subnet_registration_fees = network_state['new_subnets'] * protocol_params['subnet_registration_fee']
        
        # Staking calculations (inflationary regime)
        if month < self.t_burn:
            subnet_rewards = self.fixed_emissions[month]
            target_staked_amount = circulating_before_locking * protocol_params['target_staking_percentage']
            actual_staked_amount = circulating_before_locking * protocol_params['staking_percentage']
            monthly_target_apy = protocol_params['target_staking_apy'] / 12
            max_monthly_apy = protocol_params['max_staking_apy'] / 12
            budgeted_rewards = target_staked_amount * monthly_target_apy
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
            self.staking_vault += max(0, vault_contribution)
            additional_staking_emissions = staking_rewards
            total_emissions = subnet_rewards + staking_rewards
            fee_vault_distribution = 0  # No distribution from fee vault in inflationary regime
        else:
            # Deflationary regime - use fee vault for staking rewards
            subnet_rewards = base_emissions
            actual_staked_amount = circulating_before_locking * protocol_params['staking_percentage']
            monthly_target_apy = protocol_params['target_staking_apy'] / 12
            target_staking_rewards = actual_staked_amount * monthly_target_apy
            
            # Try to fund staking rewards from fee vault first
            if self.fee_vault >= target_staking_rewards:
                fee_vault_distribution = target_staking_rewards
                additional_staking_emissions = 0
            else:
                fee_vault_distribution = self.fee_vault
                additional_staking_emissions = target_staking_rewards - fee_vault_distribution
            
            staking_rewards = fee_vault_distribution + additional_staking_emissions
            vault_contribution = 0  # No staking vault contribution in deflationary regime
            total_emissions = base_emissions + additional_staking_emissions
        
        # Calculate maintenance fees and total fees
        rewards_per_subnet = subnet_rewards / max(network_state['active_subnets'], 1)
        subnet_maintenance_fees = network_state['active_subnets'] * rewards_per_subnet * protocol_params['subnet_maintenance_fee_pct']
        total_fees = ent_registration_fees + subnet_registration_fees + subnet_maintenance_fees
        
        # Handle fee burn vs accumulation
        fee_burn = total_fees * protocol_params['fee_burn_rate']
        fee_vault_contribution = total_fees * (1.0 - protocol_params['fee_burn_rate'])
        self.fee_vault += fee_vault_contribution - fee_vault_distribution
        
        # Calculate revenue-based burn
        revenue_burn_tokens = (network_state['licensing_revenue_usd'] * network_state['revenue_burn_pct']) / network_state['token_price']
        
        # Total burn (fees + revenue)
        total_burn = fee_burn + revenue_burn_tokens
        
        # Update cumulative values
        self.cumulative_emissions += total_emissions
        self.cumulative_burn += total_burn
        self.cumulative_fixed_emissions += self.fixed_emissions[month]
        self.cumulative_fees += total_fees
        self.burn_history.append(total_burn)
        
        # Calculate staking metrics
        if actual_staked_amount > 0:
            staking_apy = (staking_rewards / actual_staked_amount) * 12 * 100
        else:
            staking_apy = 0
        
        staking_supply = circulating_before_locking * protocol_params['staking_percentage']
        total_locked_supply = network_state['total_collateral'] + staking_supply
        circulating_supply = circulating_before_locking - total_locked_supply
        
        result = {
            'month': month,
            'team_vested': team_vested,
            'investor_vested': investor_vested,
            'foundation_vested': foundation_vested,
            'base_emissions': base_emissions,
            'additional_staking_emissions': additional_staking_emissions,
            'total_emissions': total_emissions,
            'burn': total_burn,
            'circulating_supply': circulating_supply,
            'total_locked_supply': total_locked_supply,
            'staking_supply': staking_supply,
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
            'active_ents': network_state['active_ents'],
            'active_subnets': network_state['active_subnets'],
            'total_collateral': network_state['total_collateral'],
            'new_ents': network_state['new_ents'],
            'new_subnets': network_state['new_subnets'],
            'ent_departures': network_state['ent_departures'],
            'subnet_departures': network_state['subnet_departures'],
            'cumulative_emissions': self.cumulative_emissions,
            'cumulative_burn': self.cumulative_burn,
            'cumulative_fixed_emissions': self.cumulative_fixed_emissions,
            'cumulative_fees': self.cumulative_fees,
            'ent_registration_fee': protocol_params['ent_registration_fee'],
            'subnet_registration_fee': protocol_params['subnet_registration_fee'],
            'subnet_collateral_amount': protocol_params['subnet_collateral_amount'],
            'subnet_maintenance_fee_pct': protocol_params['subnet_maintenance_fee_pct'],
            'burn_emission_factor': protocol_params['burn_emission_factor'],
            'target_staking_apy': protocol_params['target_staking_apy'],
            'staking_percentage': protocol_params['staking_percentage'],
            'target_staking_percentage': protocol_params['target_staking_percentage'],
            'max_staking_apy': protocol_params['max_staking_apy'],
            'fee_burn_rate': protocol_params['fee_burn_rate'],
            'staking_vault': self.staking_vault,
            'vault_contribution': vault_contribution,
            'fee_vault': self.fee_vault,
            'fee_vault_contribution': fee_vault_contribution,
            'fee_vault_distribution': fee_vault_distribution,
            'token_price': network_state['token_price'],
            'licensing_revenue_usd': network_state['licensing_revenue_usd'],
            'revenue_burn_pct': network_state['revenue_burn_pct'],
            'fee_burn': fee_burn,
            'revenue_burn_tokens': revenue_burn_tokens
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
        self.staking_vault = 0
        self.fee_vault = 0 