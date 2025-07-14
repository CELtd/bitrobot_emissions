import numpy as np
import pandas as pd

class BitRobotSupplyModel:
    def __init__(self,
                 initial_supply=0,
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
                 burn_coefficient=1000000,
                 burn_lookback_months=12,
                 burn_volatility=0.2,
                 burn_pattern="logarithmic",
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
                 alpha=0.5,
                 eta=0.5,
                 gamma=1.0,
                 delta=1.0,
                 kappa=0.05,
                 random_seed=42):
        """
        Initialize the BitRobot supply model V2.
        
        Parameters:
        - initial_supply: Initial token supply (default: 1 billion)
        - team_allocation: Tokens allocated to team (default: 260M)
        - investor_allocation: Tokens allocated to investors (default: 260M)
        - foundation_allocation: Tokens allocated to foundation (default: 480M)
        - foundation_initial_liquidity: Initial foundation liquidity release (default: 50M)
        - foundation_target_48m: Target foundation tokens released by month 48 (default: 476M)
        - team_cliff_months: Number of months for team cliff (default: 12)
        - team_vesting_months: Number of months for team vesting after cliff (default: 24)
        - foundation_vesting_months: Number of months for foundation vesting (default: 48)
        - t_burn: Month at which burn-based emissions start (default: 48)
        - burn_emission_factor: Factor to multiply burn sum by for emissions (default: 0.9)
        - burn_coefficient: Coefficient b in the burn function (default: 1,000,000)
        - burn_lookback_months: Number of months to look back for burn sum (default: 12)
        - burn_volatility: Standard deviation of random burn variation (default: 0.2)
        - burn_pattern: Type of burn pattern ("logarithmic", "exponential", "sigmoid") (default: "logarithmic")
        - simulation_months: Total number of months to simulate (default: 120)
        - emissions_schedule_type: Type of emissions schedule ("static", "linear", "exponential") (default: "static")
        - linear_start_emission: Starting monthly emission for linear schedule (default: 12M)
        - linear_end_emission: Ending monthly emission for linear schedule (default: 2M)
        - linear_total_emissions: Target total emissions for linear schedule (default: 200M)
        - exponential_start_emission: Starting monthly emission for exponential schedule (default: 12M)
        - exponential_end_emission: Ending monthly emission for exponential schedule (default: 2M)
        - static_year1: Year 1 emission for static schedule (default: 80M)
        - static_year2: Year 2 emission for static schedule (default: 64M)
        - static_year3: Year 3 emission for static schedule (default: 40M)
        - static_year4: Year 4 emission for static schedule (default: 16M)
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
        - alpha, eta, gamma, delta, kappa: Fee model parameters (defaults: 0.5, 0.5, 1.0, 1.0, 0.05)
        - random_seed: Random seed for reproducibility (default: 42)
        """
        # Validate allocations
        # if team_allocation + dao_allocation != initial_supply:
        #     raise ValueError("Team and DAO allocations must sum to initial supply")
        
        # Store parameters
        self.initial_supply = initial_supply
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
        self.burn_coefficient = burn_coefficient
        self.burn_lookback_months = burn_lookback_months
        self.burn_volatility = burn_volatility
        self.burn_pattern = burn_pattern
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
        self.total_supply = np.zeros(simulation_months + 1)
        
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
            
    def _calculate_base_burn(self, t):
        """Calculate the base burn value for a given month based on the selected pattern."""
        if self.burn_pattern == "logarithmic":
            return self.burn_coefficient * np.log(1 + t)
        elif self.burn_pattern == "exponential":
            return self.burn_coefficient * (1 - np.exp(-t/12))
        elif self.burn_pattern == "sigmoid":
            return self.burn_coefficient / (1 + np.exp(-(t - self.simulation_months/2)/10))
        else:
            raise ValueError(f"Unknown burn pattern: {self.burn_pattern}")
            
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
        
    def calculate_burn(self):
        """Calculate the monthly burn based on the selected pattern with random variation."""
        np.random.seed(42)  # Set seed for reproducibility
        for t in range(1, self.simulation_months + 1):
            base_burn = self._calculate_base_burn(t)
            std_dev = self.burn_volatility * base_burn
            random_burn = np.random.normal(base_burn, std_dev)
            self.burn[t] = max(0, random_burn)
    
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

        # Maintenance fees for subnets - using percentage of rewards
        R_e = self.emissions[t] / max(len(self.subnet_registry), 1)  # Rewards per subnet
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
        
        # Calculate burn
        self.calculate_burn()
        
        # Initialize tracking arrays
        self.team_vested[0] = self.team_vesting[0]
        self.investor_vested[0] = self.investor_vesting[0]
        self.foundation_vested[0] = self.foundation_vesting[0]
        self.circulating_supply[0] = self.team_vesting[0] + self.investor_vesting[0] + self.foundation_vesting[0] - self.total_locked_supply[0]
        self.total_supply[0] = self.initial_supply
        self.team_contribution[0] = self.team_vesting[0]
        self.investor_contribution[0] = self.investor_vesting[0]
        self.foundation_contribution[0] = self.foundation_vesting[0]
        self.emissions_contribution[0] = 0
        
        # Initialize fee tracking for month 0
        fee_data = self._calculate_fees_for_month(0)
        self.active_ents[0] = fee_data['active_ents']
        self.active_subnets[0] = fee_data['active_subnets']
        self.fees_ent_reg[0] = fee_data['fees_ent_reg']
        self.fees_subnet_reg[0] = fee_data['fees_subnet_reg']
        self.fees_subnet_maint[0] = fee_data['fees_subnet_maint']
        self.total_fees[0] = fee_data['total_fees']
        self.cumulative_fees[0] = fee_data['total_fees']
        
        # Initialize collateral tracking for month 0
        self.locked_collateral[0] = self._calculate_locked_collateral(0)
        self.cumulative_locked_collateral[0] = self.locked_collateral[0]
        
        # Initialize staking tracking for month 0
        self.staking_supply[0] = self.total_supply[0] * self.staking_percentage
        self.total_locked_supply[0] = self.locked_collateral[0] + self.staking_supply[0]
        
        # Simulate each month
        for t in range(1, self.simulation_months + 1):
            # Update vesting amounts
            self.team_vested[t] = self.team_vested[t-1] + self.team_vesting[t]
            self.investor_vested[t] = self.investor_vested[t-1] + self.investor_vesting[t]
            self.foundation_vested[t] = self.foundation_vested[t-1] + self.foundation_vesting[t]
            
            # Calculate emissions
            if t < self.t_burn:
                self.emissions[t] = self.fixed_emissions[t]
            else:
                if t >= self.burn_lookback_months:
                    burn_sum = np.sum(self.burn[t-self.burn_lookback_months:t])
                    self.emissions[t] = self.burn_emission_factor * burn_sum / self.burn_lookback_months
                else:
                    self.emissions[t] = self.fixed_emissions[t]
            
            # Calculate fees for this month
            fee_data = self._calculate_fees_for_month(t)
            self.active_ents[t] = fee_data['active_ents']
            self.active_subnets[t] = fee_data['active_subnets']
            self.fees_ent_reg[t] = fee_data['fees_ent_reg']
            self.fees_subnet_reg[t] = fee_data['fees_subnet_reg']
            self.fees_subnet_maint[t] = fee_data['fees_subnet_maint']
            self.total_fees[t] = fee_data['total_fees']
            
            # Update cumulative emissions
            self.cumulative_emissions[t] = self.cumulative_emissions[t-1] + self.emissions[t]
            
            # Update cumulative fixed emissions (community allocation)
            self.cumulative_fixed_emissions[t] = self.cumulative_fixed_emissions[t-1] + self.fixed_emissions[t]
            
            # Update cumulative fees
            self.cumulative_fees[t] = self.cumulative_fees[t-1] + self.total_fees[t]
            
            # Update total supply (initial supply + cumulative emissions)
            self.total_supply[t] = self.initial_supply + np.sum(self.emissions[:t+1])
            
            # Update collateral tracking - use individual subnet tracking for smooth behavior
            self.locked_collateral[t] = self._calculate_locked_collateral(t)
            self.cumulative_locked_collateral[t] = self.locked_collateral[t]  # This is the current total locked, not cumulative
            
            # Update staking tracking
            self.staking_supply[t] = self.total_supply[t] * self.staking_percentage
            self.total_locked_supply[t] = self.locked_collateral[t] + self.staking_supply[t]
            

            
            # Update circulating supply and component contributions
            self.circulating_supply[t] = (
                self.team_vested[t] + 
                self.investor_vested[t] + 
                self.foundation_vested[t] + 
                self.cumulative_emissions[t] - 
                np.sum(self.burn[:t+1]) -
                self.total_locked_supply[t]  # Subtract total locked supply (collateral + staking) from circulating supply
            )
            
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
            'Total Supply': self.total_supply,
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
            'Total Locked Supply': self.total_locked_supply
        })
        return df