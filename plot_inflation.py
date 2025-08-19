#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from driver import sim2

def calculate_yearly_inflation_rate(df):
    """
    Calculate yearly inflation rate from monthly simulation data.
    
    Inflation rate = (Total emissions in year) / (Average circulating supply in year) * 100
    
    Args:
        df: DataFrame from sim2() with monthly data
        
    Returns:
        DataFrame with yearly inflation rates
    """
    # Add year column (0-indexed months to 1-indexed years)
    df_copy = df.copy()
    df_copy['year'] = (df_copy['month'] // 12) + 1
    
    # Group by year and calculate metrics
    yearly_data = []
    
    for year in sorted(df_copy['year'].unique()):
        year_data = df_copy[df_copy['year'] == year]
        
        # Sum total emissions for the year
        total_yearly_emissions = year_data['total_emissions'].sum()
        
        # Calculate average circulating supply for the year
        avg_circulating_supply = year_data['circulating_supply'].mean()
        
        # Calculate inflation rate as percentage
        if avg_circulating_supply > 0:
            inflation_rate = (total_yearly_emissions / avg_circulating_supply) * 100
        else:
            inflation_rate = 0
        
        yearly_data.append({
            'year': year,
            'total_emissions': total_yearly_emissions,
            'avg_circulating_supply': avg_circulating_supply,
            'inflation_rate_pct': inflation_rate,
            'start_month': year_data['month'].min(),
            'end_month': year_data['month'].max(),
            'months_in_year': len(year_data)
        })
    
    return pd.DataFrame(yearly_data)

def plot_yearly_inflation_rate(df, title="Yearly Inflation Rate", save_path=None):
    """
    Create a plot showing yearly inflation rate.
    
    Args:
        df: DataFrame from sim2() with monthly data
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    # Calculate yearly inflation rates
    yearly_df = calculate_yearly_inflation_rate(df)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main plot - inflation rate
    plt.subplot(2, 1, 1)
    plt.plot(yearly_df['year'], yearly_df['inflation_rate_pct'], 
             marker='o', linewidth=2, markersize=6, color='#2E86AB')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().set_xlim(left=0.5)
    
    # Add value labels on points
    for i, row in yearly_df.iterrows():
        plt.annotate(f'{row["inflation_rate_pct"]:.1f}%', 
                    (row['year'], row['inflation_rate_pct']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    # Secondary plot - emissions breakdown
    plt.subplot(2, 1, 2)
    plt.bar(yearly_df['year'], yearly_df['total_emissions'] / 1_000_000, 
            alpha=0.7, color='#A23B72', label='Total Emissions')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Emissions (M tokens)', fontsize=12)
    plt.title('Annual Token Emissions', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.gca().set_xlim(left=0.5)
    
    # Add value labels on bars
    for i, row in yearly_df.iterrows():
        plt.annotate(f'{row["total_emissions"]/1_000_000:.1f}M', 
                    (row['year'], row['total_emissions'] / 1_000_000),
                    textcoords="offset points", 
                    xytext=(0,5), 
                    ha='center',
                    fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nYearly Inflation Rate Summary:")
    print("=" * 50)
    for _, row in yearly_df.iterrows():
        print(f"Year {int(row['year']):2d}: {row['inflation_rate_pct']:5.1f}% "
              f"(Emissions: {row['total_emissions']/1_000_000:5.1f}M tokens, "
              f"Avg Supply: {row['avg_circulating_supply']/1_000_000:6.1f}M tokens)")
    
    avg_inflation = yearly_df['inflation_rate_pct'].mean()
    print(f"\nAverage inflation rate over {len(yearly_df)} years: {avg_inflation:.1f}%")
    
    return yearly_df

def compare_inflation_scenarios():
    """
    Compare inflation rates between different scenarios or parameter sets.
    """
    print("Comparing inflation rates for different scenarios...")
    
    # Base scenario
    print("\n--- Base Scenario ---")
    base_results = sim2(seed=42)
    base_yearly = plot_yearly_inflation_rate(base_results, 
                                           title="Base Scenario - Yearly Inflation Rate",
                                           save_path="base_inflation_rate.png")
    
    # Higher growth scenario
    print("\n--- Higher Growth Scenario ---")
    high_growth_results = sim2(
        seed=42,
        ent_arrival_rate=2.0,  # Double entity arrival rate
        linear_start_emission=15_000_000,  # Higher initial emissions
        linear_end_emission=3_000_000,     # Higher final emissions
    )
    high_growth_yearly = plot_yearly_inflation_rate(high_growth_results,
                                                   title="Higher Growth Scenario - Yearly Inflation Rate", 
                                                   save_path="high_growth_inflation_rate.png")
    
    # Compare side by side
    plt.figure(figsize=(14, 6))
    plt.plot(base_yearly['year'], base_yearly['inflation_rate_pct'], 
             marker='o', linewidth=2, label='Base Scenario', color='#2E86AB')
    plt.plot(high_growth_yearly['year'], high_growth_yearly['inflation_rate_pct'], 
             marker='s', linewidth=2, label='Higher Growth Scenario', color='#A23B72')
    
    plt.title('Inflation Rate Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_xlim(left=0.5)
    
    plt.tight_layout()
    plt.savefig("inflation_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return base_yearly, high_growth_yearly

if __name__ == "__main__":
    # Run sim2 and create inflation plot
    print("Running sim2 simulation...")
    results = sim2(seed=42)
    print(f"Simulation completed with {len(results)} months of data")
    
    # Create inflation rate plot
    yearly_inflation = plot_yearly_inflation_rate(results, 
                                                 title="BitRobot Network - Yearly Inflation Rate",
                                                 save_path="yearly_inflation_rate.png")
    
    # Optional: Run comparison scenarios
    print("\nWould you like to run comparison scenarios? (uncomment the line below)")
    # compare_inflation_scenarios()
