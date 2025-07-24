import numpy as np
import pandas as pd

class BitRobotSupplyModel:
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
                 burn_emission_factor=0.9,
                 burn_lookback_months=12,
                 simulation_months=120,
                 emissions_schedule_type="static",
                 linear_start_emission=12_000_000,
                 linear_end_emission=2_000_000,
                 linear_total_emissions=200_000_000,
                 exponential_start_emission=12_000_000,
                 exponential_end_emission=2_000_000,
                 static_year1=80_000_000,
                 static_year2=64_000_000,
                 static_year3=40_000_000,
                 static_year4=16_000_000,
                 # Percentage schedule parameters
                 percentage_start_pct=1.0,
                 percentage_end_pct=0.3,
                 percentage_total_emissions=200_000_000,
                 # Fee simulation parameters
                 lambda_ent=20,
                 lambda_subnet=10,
                 starting_ents=0,
                 starting_subnets=0,
                 ent_lifetime=12,
                 subnet_lifetime=24,
                 fee_lookback_window=12,
                 F_base_ent=10.0,
                 F_base_subnet=20.0,
                 subnet_maintenance_fee_pct=0.05,
                 # Subnet collateral parameters
                 subnet_collateral_amount=100_000,
                 # Staking parameters
                 staking_percentage=0.0,
                 target_staking_percentage=0.3,  # Target percentage of circulating supply to be staked
                 staking_participants=1000,  # New parameter: number of staking participants
                 target_staking_apy=0.04,  # Target APY for stakers (4% per year)
                 subnet_min_emissions_pct=0.8,  # Minimum percentage of emissions guaranteed to subnets
                 staking_emissions_cap_pct=0.1,  # Maximum percentage of average burn for staking emissions (default: 10%)
                 alpha=0.5,
                 eta=0.5,
                 gamma=1.0,
                 delta=1.0,
                 kappa=0.05,
                 random_seed=42):
        """
        Initialize the BitRobot supply model V2.
        
        Parameters:
        - team_allocation: Tokens allocated to team (default: 269M)
        - investor_allocation: Tokens allocated to investors (default: 351M)
        - foundation_allocation: Tokens allocated to foundation (default: 307M)
        - foundation_initial_liquidity: Initial foundation liquidity release (default: 50M)
        - foundation_target_48m: Target foundation tokens released by month 48 (default: 307M)
        - team_cliff_months: Number of months for team cliff (default: 12)
        - team_vesting_months: Number of months for team vesting after cliff (default: 24)
        - foundation_vesting_months: Number of months for foundation vesting (default: 48)
        - t_burn: Month at which burn-based emissions start (default: 48)
        - burn_emission_factor: Factor to multiply burn sum by for emissions (default: 0.9)
        - burn_lookback_months: Number of months to look back for burn sum (default: 12)
        - simulation_months: Total number of months to simulate (default: 120)
        - emissions_schedule_type: Type of emissions schedule ("static", "linear", "exponential", "percentage") (default: "static")
        - linear_start_emission: Starting monthly emission for linear schedule (default: 12M)
        - linear_end_emission: Ending monthly emission for linear schedule (default: 2M)
        - linear_total_emissions: Target total emissions for linear schedule (default: 200M)
        - exponential_start_emission: Starting monthly emission for exponential schedule (default: 12M)
        - exponential_end_emission: Ending monthly emission for exponential schedule (default: 2M)
        - static_year1: Year 1 emission for static schedule (default: 80M)
        - static_year2: Year 2 emission for static schedule (default: 64M)
        - static_year3: Year 3 emission for static schedule (default: 40M)
        - static_year4: Year 4 emission for static schedule (default: 16M)
        - percentage_start_pct: Starting monthly emission as percentage of 1B total supply (default: 1.0)
        - percentage_end_pct: Ending monthly emission as percentage of 1B total supply (default: 0.3)
        - percentage_total_emissions: Target total emissions for percentage schedule (default: 200M)
        - lambda_ent: Average number of new ENTs per month (default: 20)
        - lambda_subnet: Average number of new subnets per month (default: 10)
        - starting_ents: Number of ENTs at simulation start (default: 0)
        - starting_subnets: Number of subnets at simulation start (default: 0)
        - ent_lifetime: Lifetime of ENT registrations in months (default: 12)
        - subnet_lifetime: Lifetime of subnet registrations in months (default: 24)
        - fee_lookback_window: Window for computing rolling average rewards (default: 12)
        - F_base_ent: Static fee for ENT registration (default: 10.0)
        - F_base_subnet: Static fee for subnet registration (default: 20.0)
        - subnet_maintenance_fee_pct: Percentage of rewards charged as maintenance fee (default: 0.05)
        - subnet_collateral_amount: Amount of tokens required as collateral per subnet (default: 100,000)
        - staking_percentage: Percentage of circulating supply that is staked (default: 0.0)
        - target_staking_percentage: Target percentage of circulating supply to be staked (default: 0.3)
        - staking_participants: Number of staking participants (default: 1000)
        - target_staking_apy: Target APY for stakers (default: 0.04 = 4% per year)
        - subnet_min_emissions_pct: Minimum percentage of emissions guaranteed to subnets (default: 0.8)
        - staking_emissions_cap_pct: Maximum percentage of average burn for staking emissions (default: 0.1 = 10%)
        - alpha, eta, gamma, delta, kappa: Fee model parameters (defaults: 0.5, 0.5, 1.0, 1.0, 0.05)
        - random_seed: Random seed for reproducibility (default: 42)
        
        Note: Burn is now calculated as total fees collected, not using stochastic simulation.
        
        Staking Rewards Mechanism:
        - Target APY is specified (default: 4% per year = 0.33% per month)
        - Target staking percentage is specified (default: 30% of circulating supply)
        - Actual staking budget = monthly_rate * actual_staked_amount
        - Subnets are guaranteed a minimum percentage of emissions (default: 80%)
        - Staking rewards are capped at remaining emissions after subnet guarantee
        - Actual APY is calculated based on actual rewards given and actual staked amount
        """
        # Validate allocations
        # if team_allocation + dao_allocation != initial_supply:
        #     raise ValueError("Team and DAO allocations must sum to initial supply")
        
        # Store parameters
        self.team_allocation = team_allocation
        self.investor_allocation = investor_allocation
        self.foundation_allocation = foundation_allocation
        self.foundation_initial_liquidity = foundation_initial_liquidity
        self.foundation_target_48m = foundation_target_48m
        self.team_cliff_months = team_cliff_months
        self.team_vesting_months = team_vesting_months
        self.foundation_vesting_months = foundation_vesting_months
        self.t_burn = t_burn
        self.burn_emission_factor = burn_emission_factor
        self.burn_lookback_months = burn_lookback_months
        self.simulation_months = simulation_months
        self.emissions_schedule_type = emissions_schedule_type
        self.linear_start_emission = linear_start_emission
        self.linear_end_emission = linear_end_emission
        self.linear_total_emissions = linear_total_emissions
        self.exponential_start_emission = exponential_start_emission
        self.exponential_end_emission = exponential_end_emission
        self.static_year1 = static_year1
        self.static_year2 = static_year2
        self.static_year3 = static_year3
        self.static_year4 = static_year4
        
        # Percentage schedule parameters
        self.percentage_start_pct = percentage_start_pct
        self.percentage_end_pct = percentage_end_pct
        self.percentage_total_emissions = percentage_total_emissions
        
        # Fee simulation parameters
        self.lambda_ent = lambda_ent
        self.lambda_subnet = lambda_subnet
        self.starting_ents = starting_ents
        self.starting_subnets = starting_subnets
        self.ent_lifetime = ent_lifetime
        self.subnet_lifetime = subnet_lifetime
        self.fee_lookback_window = fee_lookback_window
        self.F_base_ent = F_base_ent
        self.F_base_subnet = F_base_subnet
        self.subnet_maintenance_fee_pct = subnet_maintenance_fee_pct
        self.subnet_collateral_amount = subnet_collateral_amount
        self.staking_percentage = staking_percentage
        self.target_staking_percentage = target_staking_percentage
        self.staking_participants = staking_participants
        self.target_staking_apy = target_staking_apy
        self.subnet_min_emissions_pct = subnet_min_emissions_pct
        self.staking_emissions_cap_pct = staking_emissions_cap_pct
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.delta = delta
        self.kappa = kappa
        self.random_seed = random_seed
        
        # Initialize arrays for time series data
        self.months = np.arange(0, simulation_months + 1)
        
        # Team vesting arrays
        self.team_vesting = np.zeros(simulation_months + 1)
        self.team_vested = np.zeros(simulation_months + 1)
        
        # Investor vesting arrays
        self.investor_vesting = np.zeros(simulation_months + 1)
        self.investor_vested = np.zeros(simulation_months + 1)
        
        # Foundation vesting arrays
        self.foundation_vesting = np.zeros(simulation_months + 1)
        self.foundation_vested = np.zeros(simulation_months + 1)
        
        # Emissions arrays
        self.emissions = np.zeros(simulation_months + 1)
        self.burn = np.zeros(simulation_months + 1)
        self.cumulative_emissions = np.zeros(simulation_months + 1)
        self.cumulative_fixed_emissions = np.zeros(simulation_months + 1)
        
        # Supply tracking arrays
        self.circulating_supply = np.zeros(simulation_months + 1)
        
        # Component tracking arrays
        self.team_contribution = np.zeros(simulation_months + 1)
        self.investor_contribution = np.zeros(simulation_months + 1)
        self.foundation_contribution = np.zeros(simulation_months + 1)
        self.emissions_contribution = np.zeros(simulation_months + 1)
        
        # Fee tracking arrays
        self.active_ents = np.zeros(simulation_months + 1)
        self.active_subnets = np.zeros(simulation_months + 1)
        self.fees_ent_reg = np.zeros(simulation_months + 1)
        self.fees_subnet_reg = np.zeros(simulation_months + 1)
        self.fees_subnet_maint = np.zeros(simulation_months + 1)
        self.total_fees = np.zeros(simulation_months + 1)
        self.cumulative_fees = np.zeros(simulation_months + 1)
        
        # Collateral tracking arrays
        self.locked_collateral = np.zeros(simulation_months + 1)
        self.cumulative_locked_collateral = np.zeros(simulation_months + 1)
        
        # Individual subnet tracking for proper collateral management
        self.subnet_collateral_tracking = []  # List of (creation_month, expiration_month, collateral_amount)
        
        # Staking tracking arrays
        self.staking_supply = np.zeros(simulation_months + 1)
        self.total_locked_supply = np.zeros(simulation_months + 1)
        
        # Staking rewards tracking arrays
        self.staking_rewards = np.zeros(simulation_months + 1)
        self.subnet_rewards = np.zeros(simulation_months + 1)
        self.per_staker_rewards = np.zeros(simulation_months + 1)
        self.staking_apy = np.zeros(simulation_months + 1)
        
        # New staking mechanism tracking arrays
        self.target_staking_budget = np.zeros(simulation_months + 1)
        self.actual_staking_budget = np.zeros(simulation_months + 1)
        self.subnet_guaranteed_emissions = np.zeros(simulation_months + 1)
        
        # Additional tracking for deflationary regime
        self.base_emissions = np.zeros(simulation_months + 1)  # Base emissions (fixed or burn-based)
        self.additional_staking_emissions = np.zeros(simulation_months + 1)  # Additional minting for staking
        
        # Define fixed emissions schedule
        self.fixed_emissions = np.zeros(simulation_months + 1)
        self._set_fixed_emissions()
        
        # Initialize fee simulation registries
        self.ent_registry = [-1] * starting_ents
        self.subnet_registry = [-1] * starting_subnets
        
    def _set_fixed_emissions(self):
        """Set the fixed emissions schedule based on the selected type."""
        if self.emissions_schedule_type == "static":
            self._set_static_fixed_emissions()
        elif self.emissions_schedule_type == "linear":
            self._set_linear_fixed_emissions()
        elif self.emissions_schedule_type == "exponential":
            self._set_exponential_fixed_emissions()
        elif self.emissions_schedule_type == "percentage":
            self._set_percentage_fixed_emissions()
        else:
            raise ValueError(f"Unknown emissions schedule type: {self.emissions_schedule_type}")
        
    def _set_static_fixed_emissions(self):
        """Set the static fixed emissions schedule for the first 48 months using user-provided values."""
        # Year 1
        monthly_emission_y1 = self.static_year1 / 12
        self.fixed_emissions[1:13] = monthly_emission_y1
        # Year 2
        monthly_emission_y2 = self.static_year2 / 12
        self.fixed_emissions[13:25] = monthly_emission_y2
        # Year 3
        monthly_emission_y3 = self.static_year3 / 12
        self.fixed_emissions[25:37] = monthly_emission_y3
        # Year 4
        monthly_emission_y4 = self.static_year4 / 12
        self.fixed_emissions[37:49] = monthly_emission_y4
        
    def _set_linear_fixed_emissions(self):
        """Set a linear emissions schedule that scales from start to end value over 48 months."""
        # Calculate the slope for linear decrease
        # We have 48 months (months 1-48) with emissions
        # Total area under the line should equal linear_total_emissions
        # Area = (start + end) * 48 / 2
        # So: linear_total_emissions = (start + end) * 48 / 2
        # Therefore: start + end = linear_total_emissions * 2 / 48
        
        # Calculate what the start and end should be to achieve linear_total_emissions
        required_sum = self.linear_total_emissions * 2 / 48  # This should equal start + end
        
        # If user provided start and end don't sum to required_sum, adjust them proportionally
        current_sum = self.linear_start_emission + self.linear_end_emission
        if abs(current_sum - required_sum) > 1:  # Allow small floating point differences
            # Scale both values proportionally to achieve the required total
            scale_factor = required_sum / current_sum
            adjusted_start = self.linear_start_emission * scale_factor
            adjusted_end = self.linear_end_emission * scale_factor
        else:
            adjusted_start = self.linear_start_emission
            adjusted_end = self.linear_end_emission
            
        # Calculate monthly emissions using linear interpolation
        for month in range(1, 49):  # months 1-48
            # Linear interpolation: emission = start + (end - start) * (month - 1) / 47
            emission = adjusted_start + (adjusted_end - adjusted_start) * (month - 1) / 47
            self.fixed_emissions[month] = emission
        
    def _set_exponential_fixed_emissions(self):
        """Set an exponential decay emissions schedule from start to end value over 48 months."""
        # Calculate the decay rate to go from start to end over 48 months
        # We want: end = start * exp(-decay_rate * 47)  (47 because we start at month 1)
        # So: decay_rate = -ln(end/start) / 47
        
        if self.exponential_start_emission <= 0 or self.exponential_end_emission <= 0:
            raise ValueError("Start and end emissions must be positive")
        
        if self.exponential_start_emission == self.exponential_end_emission:
            # If start equals end, use constant emissions
            constant_emission = self.exponential_start_emission
            self.fixed_emissions[1:49] = constant_emission
        else:
            # Calculate decay rate
            decay_rate = -np.log(self.exponential_end_emission / self.exponential_start_emission) / 47
            
            # Calculate monthly emissions using exponential decay
            for month in range(1, 49):  # months 1-48
                emission = self.exponential_start_emission * np.exp(-decay_rate * (month - 1))
                self.fixed_emissions[month] = emission
            
    def _set_percentage_fixed_emissions(self):
        """Set a percentage-based emissions schedule that scales linearly from start to end over 48 months."""
        # Total supply is 1 billion
        total_supply = 1_000_000_000
        
        # Calculate the total percentage emissions needed
        total_percentage_emissions = self.percentage_total_emissions
        
        # Calculate what the start and end percentages should be to achieve total_percentage_emissions
        # We want: total_percentage_emissions = (start_pct + end_pct) * 48 / 2 * total_supply / 100
        # So: start_pct + end_pct = total_percentage_emissions * 2 / 48 * 100 / total_supply
        required_sum = total_percentage_emissions * 2 / 48 * 100 / total_supply
        
        # If user provided start and end don't sum to required_sum, adjust them proportionally
        current_sum = self.percentage_start_pct + self.percentage_end_pct
        if abs(current_sum - required_sum) > 1e-9:  # Allow small floating point differences
            # Scale both values proportionally to achieve the required total
            scale_factor = required_sum / current_sum
            adjusted_start_pct = self.percentage_start_pct * scale_factor
            adjusted_end_pct = self.percentage_end_pct * scale_factor
        else:
            adjusted_start_pct = self.percentage_start_pct
            adjusted_end_pct = self.percentage_end_pct
            
        # Calculate monthly emissions as a percentage of total supply
        for month in range(1, 49):  # months 1-48
            # Linear interpolation: emission_pct = start_pct + (end_pct - start_pct) * (month - 1) / 47
            emission_pct = adjusted_start_pct + (adjusted_end_pct - adjusted_start_pct) * (month - 1) / 47
            self.fixed_emissions[month] = total_supply * emission_pct / 100
            
    def calculate_team_vesting(self):
        """Calculate the monthly team vesting schedule."""
        # No vesting during cliff period
        # After cliff, linear vesting over team_vesting_months
        monthly_vesting = self.team_allocation / self.team_vesting_months
        start_month = self.team_cliff_months + 1
        end_month = start_month + self.team_vesting_months
        self.team_vesting[start_month:end_month] = monthly_vesting
        
    def calculate_investor_vesting(self):
        """Calculate the monthly investor vesting schedule."""
        # Same vesting schedule as team
        monthly_vesting = self.investor_allocation / self.team_vesting_months
        start_month = self.team_cliff_months + 1
        end_month = start_month + self.team_vesting_months
        self.investor_vesting[start_month:end_month] = monthly_vesting
        
    def calculate_foundation_vesting(self):
        """Calculate the monthly DAO vesting schedule."""
        # Initial liquidity release
        self.foundation_vesting[0] = self.foundation_initial_liquidity
        
        # Calculate remaining tokens to vest by month 48
        remaining_to_48m = self.foundation_target_48m - self.foundation_initial_liquidity
        
        # Linear vesting of remaining tokens over 48 months
        monthly_vesting = remaining_to_48m / self.foundation_vesting_months
        self.foundation_vesting[1:self.foundation_vesting_months + 1] = monthly_vesting
        
    def _compute_rolling_average_reward(self, t):
        """Compute rolling average reward for fee calculations."""
        start = max(0, t - self.fee_lookback_window + 1)
        return np.mean(self.emissions[start:t+1])
    
    def _calculate_fees_for_month(self, t):
        """Calculate fees for a given month based on current network state."""
        # Expire old registrations
        self.ent_registry = [e for e in self.ent_registry if t - e < self.ent_lifetime]
        self.subnet_registry = [s for s in self.subnet_registry if t - s < self.subnet_lifetime]

        # New arrivals
        new_ents = np.random.poisson(self.lambda_ent)
        new_subnets = np.random.poisson(self.lambda_subnet)
        self.ent_registry += [t] * new_ents
        self.subnet_registry += [t] * new_subnets
        
        # Track new subnets for collateral management
        for _ in range(new_subnets):
            creation_month = t
            expiration_month = t + self.subnet_lifetime
            self.subnet_collateral_tracking.append((creation_month, expiration_month, self.subnet_collateral_amount))
        
        # Ensure there's always at least one subnet
        if len(self.subnet_registry) == 0:
            self.subnet_registry.append(t)
            # Add a default subnet for collateral tracking
            creation_month = t
            expiration_month = t + self.subnet_lifetime
            self.subnet_collateral_tracking.append((creation_month, expiration_month, self.subnet_collateral_amount))

        # Fees for new ENTs - using static ENT fee
        fee_ent = new_ents * self.F_base_ent
        
        # Fees for new subnets - using static subnet fee
        fee_subnet_reg = new_subnets * self.F_base_subnet

        # Maintenance fees for subnets - using percentage of subnet rewards (not total emissions)
        R_e = self.subnet_rewards[t] / max(len(self.subnet_registry), 1)  # Rewards per subnet
        # print(self.subnet_maintenance_fee_pct)
        fee_subnet_maint = len(self.subnet_registry) * R_e * self.subnet_maintenance_fee_pct

        total_fees = fee_ent + fee_subnet_reg + fee_subnet_maint
        
        return {
            'active_ents': len(self.ent_registry),
            'active_subnets': len(self.subnet_registry),
            'fees_ent_reg': fee_ent,
            'fees_subnet_reg': fee_subnet_reg,
            'fees_subnet_maint': fee_subnet_maint,
            'total_fees': total_fees
        }
            
    def run_simulation(self):
        """Run the full emission, circulating supply, and fee simulation."""
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Calculate vesting schedules
        self.calculate_team_vesting()
        self.calculate_investor_vesting()
        self.calculate_foundation_vesting()
        
        # Initialize tracking arrays
        self.team_vested[0] = self.team_vesting[0]
        self.investor_vested[0] = self.investor_vesting[0]
        self.foundation_vested[0] = self.foundation_vesting[0]
        
        # Initialize fee tracking for month 0
        fee_data = self._calculate_fees_for_month(0)
        self.active_ents[0] = fee_data['active_ents']
        self.active_subnets[0] = fee_data['active_subnets']
        self.fees_ent_reg[0] = fee_data['fees_ent_reg']
        self.fees_subnet_reg[0] = fee_data['fees_subnet_reg']
        self.fees_subnet_maint[0] = fee_data['fees_subnet_maint']
        self.total_fees[0] = fee_data['total_fees']
        self.cumulative_fees[0] = fee_data['total_fees']
        
        # Initialize burn for month 0 (equal to total fees)
        self.burn[0] = self.total_fees[0]
        
        # Initialize collateral tracking for month 0
        self.locked_collateral[0] = self._calculate_locked_collateral(0)
        self.cumulative_locked_collateral[0] = self.locked_collateral[0]
        
        # Calculate circulating supply before locking for month 0
        circulating_before_locking = self.team_vested[0] + self.investor_vested[0] + self.foundation_vested[0] + self.cumulative_emissions[0] - np.sum(self.burn[:1])
        
        # Initialize staking tracking for month 0
        self.staking_supply[0] = circulating_before_locking * self.staking_percentage
        self.total_locked_supply[0] = self.locked_collateral[0] + self.staking_supply[0]
        
        # Initialize circulating supply for month 0
        self.circulating_supply[0] = circulating_before_locking - self.total_locked_supply[0]
        
        # Initialize component contributions for month 0
        self.team_contribution[0] = self.team_vesting[0]
        self.investor_contribution[0] = self.investor_vesting[0]
        self.foundation_contribution[0] = self.foundation_vesting[0]
        self.emissions_contribution[0] = 0
        
        # Simulate each month
        for t in range(1, self.simulation_months + 1):
            # Update vesting amounts
            self.team_vested[t] = self.team_vested[t-1] + self.team_vesting[t]
            self.investor_vested[t] = self.investor_vested[t-1] + self.investor_vesting[t]
            self.foundation_vested[t] = self.foundation_vested[t-1] + self.foundation_vesting[t]
            
            # Calculate base emissions first (needed for fee calculations)
            if t < self.t_burn:
                self.base_emissions[t] = self.fixed_emissions[t]
            else:
                if t >= self.burn_lookback_months:
                    burn_sum = np.sum(self.burn[t-self.burn_lookback_months:t])
                    self.base_emissions[t] = self.burn_emission_factor * burn_sum / self.burn_lookback_months
                else:
                    self.base_emissions[t] = self.fixed_emissions[t]
            
            # Calculate circulating supply before locking for staking calculations
            circulating_before_locking = (
                self.team_vested[t] + 
                self.investor_vested[t] + 
                self.foundation_vested[t] + 
                self.cumulative_emissions[t] - 
                np.sum(self.burn[:t+1])
            )
            
            # Calculate target staking budget based on target APY and target staking percentage
            # Monthly rate = annual rate / 12
            monthly_rate = self.target_staking_apy / 12
            target_staked_amount = circulating_before_locking * self.target_staking_percentage
            self.target_staking_budget[t] = monthly_rate * target_staked_amount
            
            # Calculate actual staking budget based on actual staked amount
            actual_staked_amount = circulating_before_locking * self.staking_percentage
            self.actual_staking_budget[t] = monthly_rate * actual_staked_amount
            
            # Different logic for fixed vs deflationary regime
            if t < self.t_burn:
                # Fixed emissions regime (first 48 months)
                # Calculate guaranteed emissions for subnets
                self.subnet_guaranteed_emissions[t] = self.base_emissions[t] * self.subnet_min_emissions_pct
                
                # Determine staking rewards: use actual budget, but cap at remaining emissions after subnet guarantee
                max_staking_rewards = self.base_emissions[t] - self.subnet_guaranteed_emissions[t]
                self.staking_rewards[t] = min(self.actual_staking_budget[t], max_staking_rewards)
                self.additional_staking_emissions[t] = 0  # No additional minting in fixed regime
                
                # Subnet rewards get the remainder of base emissions
                self.subnet_rewards[t] = self.base_emissions[t] - self.staking_rewards[t]
            else:
                # Deflationary regime (after 48 months)
                # Subnets get all base emissions
                self.subnet_rewards[t] = self.base_emissions[t]
                self.subnet_guaranteed_emissions[t] = self.base_emissions[t]
                
                # Calculate cap for additional staking emissions based on average burn
                if t >= self.burn_lookback_months:
                    burn_sum = np.sum(self.burn[t-self.burn_lookback_months:t])
                    average_monthly_burn = burn_sum / self.burn_lookback_months
                    staking_emissions_cap = average_monthly_burn * self.staking_emissions_cap_pct
                else:
                    # If we don't have enough burn history yet, use a conservative cap
                    staking_emissions_cap = self.base_emissions[t] * self.staking_emissions_cap_pct
                
                # Mint additional tokens for staking rewards, but cap to maintain deflationary behavior
                self.additional_staking_emissions[t] = min(self.actual_staking_budget[t], staking_emissions_cap)
                self.staking_rewards[t] = self.additional_staking_emissions[t]
            
            # Total emissions = base emissions + additional staking emissions
            self.emissions[t] = self.base_emissions[t] + self.additional_staking_emissions[t]
            
            # Calculate per-staker rewards and APY
            if self.staking_participants > 0:
                self.per_staker_rewards[t] = self.staking_rewards[t] / self.staking_participants
                
                # Calculate APY based on actual rewards given and actual staked amount
                if actual_staked_amount > 0:
                    # APY = (monthly rewards / staked amount) * 12 * 100
                    self.staking_apy[t] = (self.staking_rewards[t] / actual_staked_amount) * 12 * 100
                else:
                    self.staking_apy[t] = 0
            else:
                self.per_staker_rewards[t] = 0
                self.staking_apy[t] = 0
            
            # Calculate fees for this month (now emissions[t] is available)
            fee_data = self._calculate_fees_for_month(t)
            self.active_ents[t] = fee_data['active_ents']
            self.active_subnets[t] = fee_data['active_subnets']
            self.fees_ent_reg[t] = fee_data['fees_ent_reg']
            self.fees_subnet_reg[t] = fee_data['fees_subnet_reg']
            self.fees_subnet_maint[t] = fee_data['fees_subnet_maint']
            self.total_fees[t] = fee_data['total_fees']
            
            # Set burn equal to total fees collected
            self.burn[t] = self.total_fees[t]
            
            # Update cumulative emissions
            self.cumulative_emissions[t] = self.cumulative_emissions[t-1] + self.emissions[t]
            
            # Update cumulative fixed emissions (community allocation)
            self.cumulative_fixed_emissions[t] = self.cumulative_fixed_emissions[t-1] + self.fixed_emissions[t]
            
            # Update cumulative fees
            self.cumulative_fees[t] = self.cumulative_fees[t-1] + self.total_fees[t]
            
            # Update collateral tracking - use individual subnet tracking for smooth behavior
            self.locked_collateral[t] = self._calculate_locked_collateral(t)
            self.cumulative_locked_collateral[t] = self.locked_collateral[t]  # This is the current total locked, not cumulative
            
            # Use the circulating supply before locking that was calculated earlier for staking
            
            # Update staking tracking - now based on circulating supply before locking
            self.staking_supply[t] = circulating_before_locking * self.staking_percentage
            self.total_locked_supply[t] = self.locked_collateral[t] + self.staking_supply[t]
            
            # Update circulating supply - subtract locked supply from circulating before locking
            self.circulating_supply[t] = circulating_before_locking - self.total_locked_supply[t]
            
            self.team_contribution[t] = self.team_vested[t]
            self.investor_contribution[t] = self.investor_vested[t]
            self.foundation_contribution[t] = self.foundation_vested[t]
            self.emissions_contribution[t] = self.cumulative_emissions[t] - np.sum(self.burn[:t+1])
            
    def _calculate_locked_collateral(self, t):
        """Calculate total locked collateral at month t based on individual subnet tracking."""
        total_collateral = 0
        for creation_month, expiration_month, collateral_amount in self.subnet_collateral_tracking:
            # Check if this subnet is active at month t
            if creation_month <= t < expiration_month:
                total_collateral += collateral_amount
        return total_collateral

    def get_results_dataframe(self):
        """Return the simulation results as a pandas DataFrame."""
        df = pd.DataFrame({
            'Month': self.months,
            'Team Vesting': self.team_vesting,
            'Team Vested': self.team_vested,
            'Investor Vesting': self.investor_vesting,
            'Investor Vested': self.investor_vested,
            'Foundation Vesting': self.foundation_vesting,
            'Foundation Vested': self.foundation_vested,
            'Emissions': self.emissions,
            'Burn': self.burn,
            'Cumulative Emissions': self.cumulative_emissions,
            'Cumulative Fixed Emissions': self.cumulative_fixed_emissions,
            'Circulating Supply': self.circulating_supply,
            'Team Contribution': self.team_contribution,
            'Investor Contribution': self.investor_contribution,
            'Foundation Contribution': self.foundation_contribution,
            'Emissions Contribution': self.emissions_contribution,
            # Fee tracking data
            'Active ENTs': self.active_ents,
            'Active Subnets': self.active_subnets,
            'ENT Registration Fees': self.fees_ent_reg,
            'Subnet Registration Fees': self.fees_subnet_reg,
            'Subnet Maintenance Fees': self.fees_subnet_maint,
            'Total Fees': self.total_fees,
            'Cumulative Fees': self.cumulative_fees,
            # Collateral tracking data
            'Locked Collateral': self.locked_collateral,
            'Cumulative Locked Collateral': self.cumulative_locked_collateral,
            # Staking tracking data
            'Staking Supply': self.staking_supply,
            'Total Locked Supply': self.total_locked_supply,
            # Staking rewards tracking data
            'Staking Rewards': self.staking_rewards,
            'Subnet Rewards': self.subnet_rewards,
            'Per Staker Rewards': self.per_staker_rewards,
            'Staking APY': self.staking_apy,
            # New staking mechanism tracking data
            'Target Staking Budget': self.target_staking_budget,
            'Actual Staking Budget': self.actual_staking_budget,
            'Subnet Guaranteed Emissions': self.subnet_guaranteed_emissions,
            # Additional tracking for deflationary regime
            'Base Emissions': self.base_emissions,
            'Additional Staking Emissions': self.additional_staking_emissions
        })
        return df