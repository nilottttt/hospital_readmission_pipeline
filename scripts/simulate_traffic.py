"""Send realistic requests to the API to demonstrate monitoring.

    # normal traffic (recent encounters, no drift)
    docker compose exec airflow-scheduler python /opt/airflow/scripts/simulate_traffic.py --n 300

    # drifted traffic: older patients, longer stays, more medications and prior visits
    docker compose exec airflow-scheduler python /opt/airflow/scripts/simulate_traffic.py --n 300 --drift

Then trigger the readmission_model_monitoring DAG and look at the
"Model monitoring" tab of the dashboard.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etl import db  # noqa: E402
from ml import config  # noqa: E402
from ml.features import FEATURES, coerce_types  # noqa: E402


def apply_drift(row: dict, rng: random.Random) -> dict:
    r = dict(row)
    r["age_midpoint"] = min(95.0, r["age_midpoint"] + 10)
    r["length_of_stay_days"] = min(14.0, r["length_of_stay_days"] + 3)
    r["num_medications"] = min(100.0, round(r["num_medications"] * 1.5))
    r["number_inpatient"] = min(100.0, r["number_inpatient"] + rng.choice([1, 2, 3]))
    r["prior_admissions"] = r["number_outpatient"] + r["number_emergency"] + r["number_inpatient"]
    if rng.random() < 0.5:
        r["a1c_result"] = ">8"
    if rng.random() < 0.4:
        r["admission_type"] = "Emergency"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    df = coerce_types(db.fetch_df(
        "SELECT * FROM mart.v_ml_features ORDER BY encounter_order DESC LIMIT %s", (args.n * 5,)))
    df = df.sample(min(args.n, len(df)), random_state=args.seed)
    ok = fail = 0
    t0 = time.time()
    for _, row in df.iterrows():
        enc = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row[FEATURES].items()}
        if args.drift:
            enc = apply_drift(enc, rng)
        resp = requests.post(f"{config.API_URL}/predict",
                             json={"encounter": enc, "explain": False, "source": "simulation"}, timeout=30)
        ok += resp.status_code == 200
        fail += resp.status_code != 200
    print(f"Sent {ok + fail} requests ({'drifted' if args.drift else 'normal'}) in {time.time() - t0:.1f}s: "
          f"{ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
