import numpy as np
import pandas as pd

"""
Before month 48 (t_burn):
    Emissions are fixed (decreasing from 8.3M to 0.5M)
    Vesting is constant at 8.3M until month 36, then drops to 0
    Burns are relatively small (256K to 389K)
    Net Supply Change is large and positive (16.4M to 4.6M)
After month 48 (t_burn):
    Emissions are now based on burn history (187K to 237K)
    Vesting is 0
    Burns are increasing (389K to 479K)
    Net Supply Change is negative (-201K to -242K)
The shapes look similar after month 48 because:
    - Emissions are calculated as burn_emission_factor * (sum of burns) / burn_lookback_months
    - Burns are increasing logarithmically
    - Net Supply Change is emissions - burn (since vesting is 0)
This means both emissions and net supply change are following the same underlying burn pattern, just with different scaling:
    - Emissions = 0.9 * (average of last 12 months of burns)
    - Net Supply Change = Emissions - current month's burn
"""

class BitRobotEmissionsModel:
    def __init__(self, 
                 total_supply=1_000_000_000,
                 team_allocation_percentage=0.3,
                 vesting_months=36,
                 t_burn=48,
                 burn_emission_factor=0.9,
                 burn_coefficient=1000000,
                 burn_lookback_months=12,
                 burn_volatility=0.2,
                 burn_pattern="logarithmic",
                 simulation_months=120):
        """
        Initialize the BitRobot emissions model.
        
        Parameters:
        - total_supply: Total token supply (default: 1 billion)
        - team_allocation_percentage: Percentage allocated to team/consultants (default: 30%)
        - vesting_months: Number of months for team vesting (default: 36)
        - t_burn: Month at which burn-based emissions start (default: 48)
        - burn_emission_factor: Factor to multiply burn sum by for emissions (default: 0.9)
        - burn_coefficient: Coefficient b in the burn function (default: 1,000,000)
        - burn_lookback_months: Number of months to look back for burn sum (default: 12)
        - burn_volatility: Standard deviation of random burn variation as percentage of base burn (default: 0.2)
        - burn_pattern: Type of burn pattern ("logarithmic", "exponential", "sigmoid") (default: "logarithmic")
        - simulation_months: Total number of months to simulate (default: 120)
        """
        self.total_supply = total_supply
        self.team_allocation = team_allocation_percentage * total_supply
        self.vesting_months = vesting_months
        self.t_burn = t_burn
        self.burn_emission_factor = burn_emission_factor
        self.burn_coefficient = burn_coefficient
        self.burn_lookback_months = burn_lookback_months
        self.burn_volatility = burn_volatility
        self.burn_pattern = burn_pattern
        self.simulation_months = simulation_months
        
        # Initialize arrays for time series data
        self.months = np.arange(0, simulation_months + 1)
        self.vesting = np.zeros(simulation_months + 1)
        self.emissions = np.zeros(simulation_months + 1)
        self.burn = np.zeros(simulation_months + 1)
        self.circulating_supply = np.zeros(simulation_months + 1)
        self.cumulative_emissions = np.zeros(simulation_months + 1)
        self.net_supply_change = np.zeros(simulation_months + 1)  # New tracking array
        
        # Define fixed emissions schedule
        self.fixed_emissions = np.zeros(simulation_months + 1)
        self._set_default_fixed_emissions()
        
    def _set_default_fixed_emissions(self):
        """Set the default fixed emissions schedule as specified in the document."""
        # Months 1-12: 100M/12 per month
        self.fixed_emissions[1:13] = 100_000_000 / 12
        # Months 13-24: 88M/12 per month
        self.fixed_emissions[13:25] = 88_000_000 / 12
        # Months 25-36: 60M/12 per month
        self.fixed_emissions[25:37] = 60_000_000 / 12
        # Months 37-48: 25M/12 per month
        self.fixed_emissions[37:49] = 25_000_000 / 12
        
    def set_custom_fixed_emissions(self, emission_schedule):
        """
        Set a custom fixed emissions schedule.
        
        Parameters:
        - emission_schedule: Dictionary with month ranges as keys and total tokens as values
                           Example: {(1, 12): 100_000_000, (13, 24): 88_000_000}
        """
        self.fixed_emissions = np.zeros(self.simulation_months + 1)
        for (start_month, end_month), total_tokens in emission_schedule.items():
            months_count = end_month - start_month + 1
            monthly_emission = total_tokens / months_count
            self.fixed_emissions[start_month:end_month + 1] = monthly_emission
            
    def _calculate_base_burn(self, t):
        """Calculate the base burn value for a given month based on the selected pattern."""
        if self.burn_pattern == "logarithmic":
            return self.burn_coefficient * np.log(1 + t)
        elif self.burn_pattern == "exponential":
            return self.burn_coefficient * (1 - np.exp(-t/12))  # Scale to reach ~63% of max in 12 months
        elif self.burn_pattern == "sigmoid":
            # Sigmoid function that starts slow, accelerates, then levels off
            return self.burn_coefficient / (1 + np.exp(-(t - self.simulation_months/2)/10))
        else:
            raise ValueError(f"Unknown burn pattern: {self.burn_pattern}")
            
    def calculate_vesting(self):
        """Calculate the monthly vesting based on the linear vesting schedule."""
        monthly_vesting = self.team_allocation / self.vesting_months
        self.vesting[0:self.vesting_months] = monthly_vesting
        
    def calculate_burn(self):
        """Calculate the monthly burn based on the selected pattern with random variation."""
        np.random.seed(42)  # Set seed for reproducibility
        for t in range(1, self.simulation_months + 1):
            # Base burn value from selected pattern
            base_burn = self._calculate_base_burn(t)
            
            # Add random variation with specified volatility
            std_dev = self.burn_volatility * base_burn
            random_burn = np.random.normal(base_burn, std_dev)
            
            # Ensure burn is never negative
            self.burn[t] = max(0, random_burn)
            
    def run_simulation(self):
        """Run the full emission and circulating supply simulation."""
        # Calculate vesting schedule
        self.calculate_vesting()
        
        # Calculate burn
        self.calculate_burn()
        
        # Initial circulating supply is the initial vesting
        self.circulating_supply[0] = self.vesting[0]
        self.cumulative_emissions[0] = 0
        self.net_supply_change[0] = 0
        
        # Simulate each month
        for t in range(1, self.simulation_months + 1):
            # Determine emissions for this month
            if t < self.t_burn:
                # Fixed emissions
                self.emissions[t] = self.fixed_emissions[t]
            else:
                # Burn-based emissions - sum of burns over lookback window * factor
                if t >= self.burn_lookback_months:
                    burn_sum = np.sum(self.burn[t-self.burn_lookback_months:t])
                    self.emissions[t] = self.burn_emission_factor * burn_sum / self.burn_lookback_months
                else:
                    # Fallback if we don't have enough history
                    self.emissions[t] = self.fixed_emissions[t]
            
            # Calculate net supply change (emissions + vesting - burn)
            self.net_supply_change[t] = self.emissions[t] + self.vesting[t] - self.burn[t]
            
            # # Debug print for key months
            # if t % 12 == 0 or t == self.t_burn:
            #     print(f"\nMonth {t}:")
            #     print(f"  Emissions: {self.emissions[t]:.2f}")
            #     print(f"  Vesting: {self.vesting[t]:.2f}")
            #     print(f"  Burn: {self.burn[t]:.2f}")
            #     print(f"  Net Supply Change: {self.net_supply_change[t]:.2f}")
            
            # Update circulating supply
            self.circulating_supply[t] = (
                self.circulating_supply[t-1] + 
                self.vesting[t] + 
                self.emissions[t] - 
                self.burn[t]
            )
            
            # Update cumulative emissions
            self.cumulative_emissions[t] = self.cumulative_emissions[t-1] + self.emissions[t]
            
    def get_results_dataframe(self):
        """Return the simulation results as a pandas DataFrame."""
        df = pd.DataFrame({
            'Month': self.months,
            'Vesting': self.vesting,
            'Emissions': self.emissions,
            'Burn': self.burn,
            'Circulating Supply': self.circulating_supply,
            'Cumulative Emissions': self.cumulative_emissions,
            'Net Supply Change': self.net_supply_change
        })
        return df

# Example usage
if __name__ == "__main__":
    # Create model with default parameters
    model = BitRobotEmissionsModel(
        total_supply=1_000_000_000,
        team_allocation_percentage=0.3,
        vesting_months=36,
        t_burn=48,
        burn_emission_factor=0.9,
        burn_coefficient=1000000,
        burn_lookback_months=12,
        burn_volatility=0.2,
        burn_pattern="logarithmic",
        simulation_months=120
    )
    
    # Run the simulation
    model.run_simulation()
    
    # Get results dataframe
    results_df = model.get_results_dataframe()
    
    # Print first few rows of results
    print(results_df.head(10))