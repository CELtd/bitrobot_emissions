"""
Quick script to check fee calculations
"""
import pandas as pd
import driver
import robo_supply_model

# Run the same simulation as in the notebook
emissions_start = 5_000_000
emissions_end = 500_000
emissions_total_target = 197_000_000
subnet_maintenance_fee_pct = 0.20

ent_arrival_rate = 100.0
num_ents_start_desired = 500
num_ents_start_config = num_ents_start_desired - ent_arrival_rate

df_baseline = driver.sim2(
    initial_ents=int(num_ents_start_config),
    initial_subnets=10,
    target_subnets_month_48=30,
    post_48_subnets_per_year=1,
    ent_arrival_rate=ent_arrival_rate,
    ent_lifetime_months=24,
    subnet_lifetime_months=36,
    initial_subnet_revenue=0,
    max_subnet_revenue=100000,
    revenue_growth_months=48,
    revenue_burn_pct=0.5,
    initial_token_price=1.0,
    annual_price_growth_rate=0.0,
    initial_target_staking_apy=0.08,
    final_target_staking_apy=0.04,
    staking_apy_transition_months=48,
    linear_start_emission=emissions_start,
    linear_end_emission=emissions_end,
    linear_total_emissions=emissions_total_target,
    dynamic_staking_fees=False,
    airdrop_allocation=50_000_000,
    community_round_allocation=50_000_000,
    airdrop_vesting_months=6,
    community_round_vesting_months=12,
    subnet_maintenance_fee_pct=subnet_maintenance_fee_pct,
)

# Look at Year 1 (months 0-11)
year1_df = df_baseline[df_baseline['month'] < 12].copy()

print("=" * 100)
print("YEAR 1 DETAILED BREAKDOWN (Months 0-11)")
print("=" * 100)
print()

# Print month-by-month breakdown
print(f"{'Month':<8} {'Subnet Rewards':<18} {'Maint Fee':<18} {'Reg Fees':<18} {'Total Fees':<18} {'Revenue Burn':<18}")
print("-" * 100)
for idx, row in year1_df.iterrows():
    print(f"{row['month']:<8} {row['base_emissions']:>15,.0f}   {row['subnet_maintenance_fees']:>15,.0f}   "
          f"{(row['ent_registration_fees'] + row['subnet_registration_fees']):>15,.0f}   "
          f"{row['total_fees']:>15,.0f}   {row['revenue_burn_tokens']:>15,.0f}")

print("-" * 100)

# Calculate totals
total_subnet_rewards = year1_df['base_emissions'].sum()
total_maintenance_fees = year1_df['subnet_maintenance_fees'].sum()
total_ent_reg = year1_df['ent_registration_fees'].sum()
total_subnet_reg = year1_df['subnet_registration_fees'].sum()
total_reg_fees = total_ent_reg + total_subnet_reg
total_fees = year1_df['total_fees'].sum()
total_revenue_burn = year1_df['revenue_burn_tokens'].sum()

print(f"{'TOTAL':<8} {total_subnet_rewards:>15,.0f}   {total_maintenance_fees:>15,.0f}   "
      f"{total_reg_fees:>15,.0f}   {total_fees:>15,.0f}   {total_revenue_burn:>15,.0f}")
print()

# Verify maintenance fee calculation
expected_maintenance = total_subnet_rewards * subnet_maintenance_fee_pct
print(f"\nVERIFICATION:")
print(f"Total Subnet Rewards (base_emissions): {total_subnet_rewards:,.0f} ROBO")
print(f"Expected Maintenance Fees (20% of rewards): {expected_maintenance:,.0f} ROBO")
print(f"Actual Maintenance Fees: {total_maintenance_fees:,.0f} ROBO")
print(f"Match: {abs(expected_maintenance - total_maintenance_fees) < 1.0}")
print()

# Break down total fees
print(f"\nFEE BREAKDOWN:")
print(f"  Maintenance Fees: {total_maintenance_fees:>15,.0f} ROBO ({total_maintenance_fees/total_fees*100:.1f}%)")
print(f"  ENT Registration: {total_ent_reg:>15,.0f} ROBO ({total_ent_reg/total_fees*100:.1f}%)")
print(f"  Subnet Registration: {total_subnet_reg:>15,.0f} ROBO ({total_subnet_reg/total_fees*100:.1f}%)")
print(f"  {'─'*40}")
print(f"  Total Fees: {total_fees:>15,.0f} ROBO")
print()

# Show what the table prints
print(f"\nWHAT THE TABLE SHOWS FOR YEAR 1:")
print(f"  Fees (column in table): {total_fees:,.0f} ROBO")
print(f"  Revenue Burn (column in table): {total_revenue_burn:,.0f} ROBO")
print(f"  Total Capture (shown in table): {total_fees + total_revenue_burn:,.0f} ROBO")
print()

# Check new entities and subnets
total_new_ents = year1_df['new_ents'].sum()
total_new_subnets = year1_df['new_subnets'].sum()
avg_subnets = year1_df['active_subnets'].mean()

print(f"\nNETWORK ACTIVITY:")
print(f"  Total New ENTs: {total_new_ents:,.0f}")
print(f"  Total New Subnets: {total_new_subnets:,.0f}")
print(f"  Avg Active Subnets: {avg_subnets:.1f}")
print(f"  ENT Registration Fee: 100 ROBO per ENT")
print(f"  Subnet Registration Fee: 5000 ROBO per subnet")
print()

# Verify registration fees
expected_ent_reg = total_new_ents * 100
expected_subnet_reg = total_new_subnets * 5000
print(f"  Expected ENT Reg Fees: {expected_ent_reg:,.0f} ROBO")
print(f"  Actual ENT Reg Fees: {total_ent_reg:,.0f} ROBO")
print(f"  Expected Subnet Reg Fees: {expected_subnet_reg:,.0f} ROBO")
print(f"  Actual Subnet Reg Fees: {total_subnet_reg:,.0f} ROBO")

