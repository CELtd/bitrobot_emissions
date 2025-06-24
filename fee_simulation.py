import numpy as np
import pandas as pd

def run_fee_simulation(
    epochs=48,
    emission_schedule=None,
    lambda_ent=20,
    lambda_subnet=10,
    starting_ents=0,
    starting_subnets=0,
    ent_lifetime=12,
    subnet_lifetime=24,
    lookback_window=12,
    F_base=10.0,
    alpha=0.5,
    eta=0.5,
    gamma=1.0,
    delta=1.0,
    kappa=0.05
):
    # Default emission schedule if not provided
    if emission_schedule is None:
        emission_schedule = np.full(epochs, 1000.0)

    # Initialize registries with starting numbers
    # For starting entities, we assume they were registered at time -1 (just before simulation starts)
    ent_registry = [-1] * starting_ents
    subnet_registry = [-1] * starting_subnets

    results = {
        "epoch": [],
        "active_ENTs": [],
        "active_subnets": [],
        "fees_ENT_reg": [],
        "fees_subnet_reg": [],
        "fees_subnet_maint": [],
        "total_fees": []
    }

    def compute_R_avg(t):
        start = max(0, t - lookback_window + 1)
        return np.mean(emission_schedule[start:t+1])

    for t in range(epochs):
        # Expire old registrations
        ent_registry = [e for e in ent_registry if t - e < ent_lifetime]
        subnet_registry = [s for s in subnet_registry if t - s < subnet_lifetime]

        # New arrivals
        new_ents = np.random.poisson(lambda_ent)
        new_subnets = np.random.poisson(lambda_subnet)
        ent_registry += [t] * new_ents
        subnet_registry += [t] * new_subnets

        # Compute rolling average reward
        R_avg = compute_R_avg(t)

        # Fees for new ENTs
        # fee_ent = sum(max(F_base, min(alpha * R_avg, eta * 1.0)) for _ in range(new_ents))
        fee_ent = sum(F_base for _ in range(new_ents))
        # Fees for new subnets
        # fee_subnet_reg = sum(max(F_base, min(alpha * R_avg, eta * 1.0)) for _ in range(new_subnets))
        fee_subnet_reg = sum(F_base for _ in range(new_subnets))

        # Maintenance fees for subnets
        R_e = emission_schedule[t] / max(len(subnet_registry), 1)
        # fee_subnet_maint = sum(
        #     max(F_base, alpha * gamma * R_e + (1 - alpha) * delta * 1.0)
        #     for _ in subnet_registry
        # )
        fee_subnet_maint = sum(
            F_base
            for _ in subnet_registry
        )

        total_fees = fee_ent + fee_subnet_reg + fee_subnet_maint

        # Record results
        results["epoch"].append(t)
        results["active_ENTs"].append(len(ent_registry))
        results["active_subnets"].append(len(subnet_registry))
        results["fees_ENT_reg"].append(fee_ent)
        results["fees_subnet_reg"].append(fee_subnet_reg)
        results["fees_subnet_maint"].append(fee_subnet_maint)
        results["total_fees"].append(total_fees)

    return pd.DataFrame(results)