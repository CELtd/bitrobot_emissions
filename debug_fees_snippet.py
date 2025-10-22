# PASTE THIS INTO A NEW CELL IN THE NOTEBOOK TO DEBUG THE FEE CALCULATION

# Year 1 is months 0-11
year1 = df_baseline[df_baseline['month'] < 12].copy()

print("="*100)
print("YEAR 1 (Months 0-11) - DETAILED VERIFICATION")
print("="*100)
print()

# Sum up all the relevant columns
base_emissions_sum = year1['base_emissions'].sum()
subnet_rewards_sum = year1['subnet_rewards'].sum()
maintenance_fees_sum = year1['subnet_maintenance_fees'].sum()
ent_reg_fees_sum = year1['ent_registration_fees'].sum()
subnet_reg_fees_sum = year1['subnet_registration_fees'].sum()
total_fees_sum = year1['total_fees'].sum()
net_base_emissions_sum = year1['net_base_emissions'].sum()

print("COLUMN SUMS:")
print(f"  base_emissions:              {base_emissions_sum:>20,.0f} ROBO")
print(f"  subnet_rewards:              {subnet_rewards_sum:>20,.0f} ROBO")  
print(f"  subnet_maintenance_fees:     {maintenance_fees_sum:>20,.0f} ROBO")
print(f"  ent_registration_fees:       {ent_reg_fees_sum:>20,.0f} ROBO")
print(f"  subnet_registration_fees:    {subnet_reg_fees_sum:>20,.0f} ROBO")
print(f"  total_fees:                  {total_fees_sum:>20,.0f} ROBO")
print(f"  net_base_emissions:          {net_base_emissions_sum:>20,.0f} ROBO")
print()

print("VERIFICATION CHECKS:")
print("-"*100)

# Check 1: Are base_emissions and subnet_rewards the same?
print(f"1. base_emissions == subnet_rewards?")
print(f"   base_emissions:    {base_emissions_sum:>20,.0f}")
print(f"   subnet_rewards:    {subnet_rewards_sum:>20,.0f}")
print(f"   Match: {abs(base_emissions_sum - subnet_rewards_sum) < 1.0}")
print()

# Check 2: Is maintenance fee 20% of subnet rewards?
expected_maintenance = subnet_rewards_sum * 0.20
print(f"2. Maintenance fees = 20% of subnet_rewards?")
print(f"   Expected (20%):    {expected_maintenance:>20,.0f}")
print(f"   Actual:            {maintenance_fees_sum:>20,.0f}")
print(f"   Difference:        {abs(expected_maintenance - maintenance_fees_sum):>20,.0f}")
print(f"   Match: {abs(expected_maintenance - maintenance_fees_sum) < 100}")
print()

# Check 3: Do fees add up correctly?
expected_total_fees = maintenance_fees_sum + ent_reg_fees_sum + subnet_reg_fees_sum
print(f"3. total_fees = maintenance + ent_reg + subnet_reg?")
print(f"   Expected:          {expected_total_fees:>20,.0f}")
print(f"   Actual:            {total_fees_sum:>20,.0f}")
print(f"   Difference:        {abs(expected_total_fees - total_fees_sum):>20,.0f}")
print(f"   Match: {abs(expected_total_fees - total_fees_sum) < 1.0}")
print()

# Check 4: Does net = gross - maintenance?
expected_net = subnet_rewards_sum - maintenance_fees_sum
print(f"4. net_base_emissions = subnet_rewards - maintenance_fees?")
print(f"   Expected:          {expected_net:>20,.0f}")
print(f"   Actual:            {net_base_emissions_sum:>20,.0f}")
print(f"   Difference:        {abs(expected_net - net_base_emissions_sum):>20,.0f}")
print(f"   Match: {abs(expected_net - net_base_emissions_sum) < 1.0}")
print()

print("="*100)
print("FEE BREAKDOWN AS PERCENTAGES")
print("="*100)
print(f"  Maintenance fees:       {maintenance_fees_sum:>15,.0f} ROBO ({maintenance_fees_sum/total_fees_sum*100:>5.1f}% of total fees)")
print(f"  ENT registration:       {ent_reg_fees_sum:>15,.0f} ROBO ({ent_reg_fees_sum/total_fees_sum*100:>5.1f}% of total fees)")
print(f"  Subnet registration:    {subnet_reg_fees_sum:>15,.0f} ROBO ({subnet_reg_fees_sum/total_fees_sum*100:>5.1f}% of total fees)")
print(f"  {'-'*80}")
print(f"  Total fees:             {total_fees_sum:>15,.0f} ROBO (100.0% of total fees)")
print()
print(f"  As % of gross rewards:  {total_fees_sum/subnet_rewards_sum*100:.2f}%")
print(f"  Maintenance as % of gross: {maintenance_fees_sum/subnet_rewards_sum*100:.2f}%")
print()

# Show month-by-month to see if there's any anomaly
print("="*100)
print("MONTH-BY-MONTH BREAKDOWN")
print("="*100)
print(f"{'Month':<8} {'Subnet Rewards':<18} {'Maint Fee (20%)':<18} {'Actual Maint':<18} {'Total Fees':<18}")
print("-"*100)
for idx, row in year1.iterrows():
    expected_maint = row['subnet_rewards'] * 0.20
    print(f"{row['month']:<8} {row['subnet_rewards']:>15,.0f}   {expected_maint:>15,.0f}   "
          f"{row['subnet_maintenance_fees']:>15,.0f}   {row['total_fees']:>15,.0f}")

print("="*100)

