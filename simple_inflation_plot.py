#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_inflation_rate(df, save_path=None):
    """
    Simple function to plot yearly inflation rate from sim2 dataframe output.
    
    Args:
        df: DataFrame from sim2() with columns 'month', 'total_emissions', 'circulating_supply'
        save_path: Optional path to save the plot (e.g., 'inflation.png')
    
    Returns:
        DataFrame with yearly inflation data
    """
    # Calculate yearly inflation rates
    df_copy = df.copy()
    df_copy['year'] = (df_copy['month'] // 12) + 1
    
    yearly_data = []
    for year in sorted(df_copy['year'].unique()):
        year_data = df_copy[df_copy['year'] == year]
        total_yearly_emissions = year_data['total_emissions'].sum()
        avg_circulating_supply = year_data['circulating_supply'].mean()
        
        if avg_circulating_supply > 0:
            inflation_rate = (total_yearly_emissions / avg_circulating_supply) * 100
        else:
            inflation_rate = 0
        
        yearly_data.append({
            'year': year,
            'inflation_rate_pct': inflation_rate,
            'total_emissions_millions': total_yearly_emissions / 1_000_000,
            'avg_supply_millions': avg_circulating_supply / 1_000_000
        })
    
    yearly_df = pd.DataFrame(yearly_data)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_df['year'], yearly_df['inflation_rate_pct'], 
             marker='o', linewidth=2.5, markersize=8, color='#E74C3C')
    
    plt.title('Yearly Inflation Rate', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Inflation Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().set_xlim(left=0.5)
    
    # Add value labels
    for _, row in yearly_df.iterrows():
        plt.annotate(f'{row["inflation_rate_pct"]:.1f}%', 
                    (row['year'], row['inflation_rate_pct']),
                    textcoords="offset points", 
                    xytext=(0,12), 
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print("\nYearly Inflation Summary:")
    print("-" * 40)
    for _, row in yearly_df.iterrows():
        print(f"Year {int(row['year']):2d}: {row['inflation_rate_pct']:6.1f}%")
    print(f"\nAverage: {yearly_df['inflation_rate_pct'].mean():6.1f}%")
    
    return yearly_df

# Example usage:
if __name__ == "__main__":
    from driver import sim2
    
    # Run simulation
    results = sim2(seed=42)
    
    # Plot inflation rate
    yearly_inflation = plot_inflation_rate(results, save_path="simple_inflation.png")
