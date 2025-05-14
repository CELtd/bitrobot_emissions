import numpy as np
import pandas as pd

class BitRobotEmissionsModelV2:
    def __init__(self,
                 initial_supply=0,
                 team_allocation=260_000_000,  # Split from 520M to 260M
                 investor_allocation=260_000_000,  # New parameter
                 dao_allocation=480_000_000,
                 dao_initial_liquidity=50_000_000,
                 dao_target_48m=200_000_000,
                 fixed_emissions_target=200_000_000,
                 team_cliff_months=12,
                 team_vesting_months=24,
                 dao_vesting_months=48,
                 t_burn=48,
                 burn_emission_factor=0.9,
                 burn_coefficient=1000000,
                 burn_lookback_months=12,
                 burn_volatility=0.2,
                 burn_pattern="logarithmic",
                 simulation_months=120):
        """
        Initialize the BitRobot emissions model V2.
        
        Parameters:
        - initial_supply: Initial token supply (default: 1 billion)
        - team_allocation: Tokens allocated to team (default: 260M)
        - investor_allocation: Tokens allocated to investors (default: 260M)
        - dao_allocation: Tokens allocated to DAO (default: 480M)
        - dao_initial_liquidity: Initial DAO liquidity release (default: 50M)
        - dao_target_48m: Target DAO tokens released by month 48 (default: 200M)
        - team_cliff_months: Number of months for team cliff (default: 12)
        - team_vesting_months: Number of months for team vesting after cliff (default: 24)
        - dao_vesting_months: Number of months for DAO vesting (default: 48)
        - t_burn: Month at which burn-based emissions start (default: 48)
        - burn_emission_factor: Factor to multiply burn sum by for emissions (default: 0.9)
        - burn_coefficient: Coefficient b in the burn function (default: 1,000,000)
        - burn_lookback_months: Number of months to look back for burn sum (default: 12)
        - burn_volatility: Standard deviation of random burn variation (default: 0.2)
        - burn_pattern: Type of burn pattern ("logarithmic", "exponential", "sigmoid") (default: "logarithmic")
        - simulation_months: Total number of months to simulate (default: 120)
        """
        # Validate allocations
        # if team_allocation + dao_allocation != initial_supply:
        #     raise ValueError("Team and DAO allocations must sum to initial supply")
        
        # Store parameters
        self.initial_supply = initial_supply
        self.team_allocation = team_allocation
        self.investor_allocation = investor_allocation
        self.dao_allocation = dao_allocation
        self.dao_initial_liquidity = dao_initial_liquidity
        self.dao_target_48m = dao_target_48m
        self.fixed_emissions_target = fixed_emissions_target
        self.team_cliff_months = team_cliff_months
        self.team_vesting_months = team_vesting_months
        self.dao_vesting_months = dao_vesting_months
        self.t_burn = t_burn
        self.burn_emission_factor = burn_emission_factor
        self.burn_coefficient = burn_coefficient
        self.burn_lookback_months = burn_lookback_months
        self.burn_volatility = burn_volatility
        self.burn_pattern = burn_pattern
        self.simulation_months = simulation_months
        
        # Initialize arrays for time series data
        self.months = np.arange(0, simulation_months + 1)
        
        # Team vesting arrays
        self.team_vesting = np.zeros(simulation_months + 1)
        self.team_vested = np.zeros(simulation_months + 1)
        
        # Investor vesting arrays
        self.investor_vesting = np.zeros(simulation_months + 1)
        self.investor_vested = np.zeros(simulation_months + 1)
        
        # DAO vesting arrays
        self.dao_vesting = np.zeros(simulation_months + 1)
        self.dao_vested = np.zeros(simulation_months + 1)
        
        # Emissions arrays
        self.emissions = np.zeros(simulation_months + 1)
        self.burn = np.zeros(simulation_months + 1)
        
        # Supply tracking arrays
        self.circulating_supply = np.zeros(simulation_months + 1)
        self.total_supply = np.zeros(simulation_months + 1)
        
        # Component tracking arrays
        self.team_contribution = np.zeros(simulation_months + 1)
        self.investor_contribution = np.zeros(simulation_months + 1)
        self.dao_contribution = np.zeros(simulation_months + 1)
        self.emissions_contribution = np.zeros(simulation_months + 1)
        
        # Define fixed emissions schedule
        self.fixed_emissions = np.zeros(simulation_months + 1)
        self._set_default_fixed_emissions()
        
    def _set_default_fixed_emissions(self):
        """Set the default fixed emissions schedule for the first 48 months."""
        # Linear emissions over 48 months to reach 200M
        monthly_emission = self.fixed_emissions_target / self.t_burn
        self.fixed_emissions[1:self.t_burn + 1] = monthly_emission
        
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
        
    def calculate_dao_vesting(self):
        """Calculate the monthly DAO vesting schedule."""
        # Initial liquidity release
        self.dao_vesting[0] = self.dao_initial_liquidity
        
        # Calculate remaining tokens to vest by month 48
        remaining_to_48m = self.dao_target_48m - self.dao_initial_liquidity
        
        # Linear vesting of remaining tokens over 48 months
        monthly_vesting = remaining_to_48m / self.dao_vesting_months
        self.dao_vesting[1:self.dao_vesting_months + 1] = monthly_vesting
        
    def calculate_burn(self):
        """Calculate the monthly burn based on the selected pattern with random variation."""
        np.random.seed(42)  # Set seed for reproducibility
        for t in range(1, self.simulation_months + 1):
            base_burn = self._calculate_base_burn(t)
            std_dev = self.burn_volatility * base_burn
            random_burn = np.random.normal(base_burn, std_dev)
            self.burn[t] = max(0, random_burn)
            
    def run_simulation(self):
        """Run the full emission and circulating supply simulation."""
        # Calculate vesting schedules
        self.calculate_team_vesting()
        self.calculate_investor_vesting()
        self.calculate_dao_vesting()
        
        # Calculate burn
        self.calculate_burn()
        
        # Initialize tracking arrays
        self.team_vested[0] = self.team_vesting[0]
        self.investor_vested[0] = self.investor_vesting[0]
        self.dao_vested[0] = self.dao_vesting[0]
        self.circulating_supply[0] = self.team_vesting[0] + self.investor_vesting[0] + self.dao_vesting[0]
        self.total_supply[0] = self.initial_supply
        self.team_contribution[0] = self.team_vesting[0]
        self.investor_contribution[0] = self.investor_vesting[0]
        self.dao_contribution[0] = self.dao_vesting[0]
        self.emissions_contribution[0] = 0
        
        # Simulate each month
        for t in range(1, self.simulation_months + 1):
            # Update vesting amounts
            self.team_vested[t] = self.team_vested[t-1] + self.team_vesting[t]
            self.investor_vested[t] = self.investor_vested[t-1] + self.investor_vesting[t]
            self.dao_vested[t] = self.dao_vested[t-1] + self.dao_vesting[t]
            
            # Calculate emissions
            if t < self.t_burn:
                self.emissions[t] = self.fixed_emissions[t]
            else:
                if t >= self.burn_lookback_months:
                    burn_sum = np.sum(self.burn[t-self.burn_lookback_months:t])
                    self.emissions[t] = self.burn_emission_factor * burn_sum / self.burn_lookback_months
                else:
                    self.emissions[t] = self.fixed_emissions[t]
            
            # Update total supply (initial supply + cumulative emissions)
            self.total_supply[t] = self.initial_supply + np.sum(self.emissions[:t+1])
            
            # Update circulating supply and component contributions
            self.circulating_supply[t] = (
                self.team_vested[t] + 
                self.investor_vested[t] + 
                self.dao_vested[t] + 
                np.sum(self.emissions[:t+1]) - 
                np.sum(self.burn[:t+1])
            )
            
            self.team_contribution[t] = self.team_vested[t]
            self.investor_contribution[t] = self.investor_vested[t]
            self.dao_contribution[t] = self.dao_vested[t]
            self.emissions_contribution[t] = np.sum(self.emissions[:t+1]) - np.sum(self.burn[:t+1])
            
    def get_results_dataframe(self):
        """Return the simulation results as a pandas DataFrame."""
        df = pd.DataFrame({
            'Month': self.months,
            'Team Vesting': self.team_vesting,
            'Team Vested': self.team_vested,
            'Investor Vesting': self.investor_vesting,
            'Investor Vested': self.investor_vested,
            'DAO Vesting': self.dao_vesting,
            'DAO Vested': self.dao_vested,
            'Emissions': self.emissions,
            'Burn': self.burn,
            'Circulating Supply': self.circulating_supply,
            'Total Supply': self.total_supply,
            'Team Contribution': self.team_contribution,
            'Investor Contribution': self.investor_contribution,
            'DAO Contribution': self.dao_contribution,
            'Emissions Contribution': self.emissions_contribution
        })
        return df