"""Generate a reproducible synthetic test-case dataset for development/demo."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_test_cases.csv"

def main(n: int = 1200) -> None:
    rng = np.random.default_rng(42)
    modules = np.array(["Authentication", "Payments", "Search", "Profile", "Inventory", "Notifications", "Reporting"])
    rows = []
    for i in range(n):
        impact = int(rng.integers(1, 6))
        failure_rate = float(np.round(rng.beta(1.5, 5), 3))
        severity = int(rng.integers(1, 6))
        execution_time = float(np.round(rng.lognormal(2.2, 0.65), 2))
        recent_failures = int(rng.poisson(1.1 + failure_rate * 7))
        code_churn = int(rng.gamma(2.0, 13.0))
        dependency_count = int(rng.integers(0, 11))
        last_result = "Fail" if rng.random() < min(0.08 + failure_rate * 0.75, 0.8) else "Pass"

        # Rule-generated labels intentionally create a learnable demonstration target.
        risk = (
            impact * 0.22 + severity * 0.24 + failure_rate * 2.1
            + min(recent_failures, 8) * 0.09
            + min(code_churn, 100) * 0.004
            + dependency_count * 0.035
            + (0.28 if last_result == "Fail" else 0)
        )
        # Add modest label noise to avoid a perfectly deterministic toy dataset.
        risk += float(rng.normal(0, 0.18))
        label = "Critical" if risk >= 2.85 else "High" if risk >= 2.25 else "Medium" if risk >= 1.55 else "Low"
        rows.append({
            "test_case_id": f"TC-{i+1:04d}",
            "test_name": f"{modules[i % len(modules)]} regression scenario {i+1}",
            "module": modules[i % len(modules)],
            "change_impact": impact,
            "failure_rate": failure_rate,
            "severity": severity,
            "execution_time": execution_time,
            "recent_failures": recent_failures,
            "code_churn": code_churn,
            "dependency_count": dependency_count,
            "last_result": last_result,
            "label": label,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Generated {len(rows)} rows: {OUT}")

if __name__ == "__main__":
    main()
