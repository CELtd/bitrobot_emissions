import numpy as np
import pandas as pd

class BitRobotEmissionsModel:
    def __init__(self, 
                 total_supply=1_000_000_000,
                 team_allocation_percentage=0.3,
                 vesting_months=36,
                 t_burn=48,
                 burn_emission_factor=0.9,
                 burn_coefficient=1000000,
                 simulation_months=120):
        """
        Initialize the BitRobot emissions model.
        
        Parameters:
        - total_supply: Total token supply (default: 1 billion)
        - team_allocation_percentage: Percentage allocated to team/consultants (default: 30%)
        - vesting_months: Number of months for team vesting (default: 36)
        - t_burn: Month at which burn-based emissions start (default: 48)
        - burn_emission_factor: F value in the emission formula (default: 0.9)
        - burn_coefficient: Coefficient b in the burn function (default: 1,000,000)
        - simulation_months: Total number of months to simulate (default: 120)
        """
        self.total_supply = total_supply
        self.team_allocation = team_allocation_percentage * total_supply
        self.vesting_months = vesting_months
        self.t_burn = t_burn
        self.burn_emission_factor = burn_emission_factor
        self.burn_coefficient = burn_coefficient
        self.simulation_months = simulation_months
        
        # Initialize arrays for time series data
        self.months = np.arange(0, simulation_months + 1)
        self.vesting = np.zeros(simulation_months + 1)
        self.emissions = np.zeros(simulation_months + 1)
        self.burn = np.zeros(simulation_months + 1)
        self.circulating_supply = np.zeros(simulation_months + 1)
        self.cumulative_emissions = np.zeros(simulation_months + 1)
        
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
            
    def calculate_vesting(self):
        """Calculate the monthly vesting based on the linear vesting schedule."""
        monthly_vesting = self.team_allocation / self.vesting_months
        self.vesting[0:self.vesting_months] = monthly_vesting
        
    def calculate_burn(self):
        """Calculate the monthly burn based on the logarithmic function."""
        for t in range(1, self.simulation_months + 1):
            self.burn[t] = self.burn_coefficient * np.log(1 + t)
            
    def run_simulation(self):
        """Run the full emission and circulating supply simulation."""
        # Calculate vesting schedule
        self.calculate_vesting()
        
        # Calculate burn
        self.calculate_burn()
        
        # Initial circulating supply is the initial vesting
        self.circulating_supply[0] = self.vesting[0]
        self.cumulative_emissions[0] = 0
        
        # Simulate each month
        for t in range(1, self.simulation_months + 1):
            # Determine emissions for this month
            if t < self.t_burn:
                # Fixed emissions
                self.emissions[t] = self.fixed_emissions[t]
            else:
                # Burn-based emissions
                if t >= 13:  # Need at least 12 months of history
                    burnrate = np.sum(self.burn[t-12:t]) / self.circulating_supply[t-12]
                    self.emissions[t] = self.burn_emission_factor * burnrate * self.circulating_supply[t-1]
                else:
                    # Fallback if we don't have 12 months of history
                    self.emissions[t] = self.fixed_emissions[t]
            
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
            'Cumulative Emissions': self.cumulative_emissions
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
        simulation_months=120
    )
    
    # Run the simulation
    model.run_simulation()
    
    # Get results dataframe
    results_df = model.get_results_dataframe()
    
    # Print first few rows of results
    print(results_df.head(10))