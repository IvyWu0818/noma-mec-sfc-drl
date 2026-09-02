"""
splice_ga300_into_final_compare_v20.py
Replace the GA entry in results/final_compare_v20.json (currently the
generations=15 GA used by eval_final_compare_v20.py) with the
generations=300 GA already computed into results/baseline_eval_v17.json by
rerun_ga300_v17.py -- same env (IIoTEnvV17), same NUM_TASKS=100, same 20
episodes seeds 0-19, so it's a direct swap-in with no re-evaluation needed.
SAC/TD3/PPO/Greedy entries are left untouched.

Run: python -m experiments.splice_ga300_into_final_compare_v20
"""

import json

FINAL_COMPARE_FILE = "results/final_compare_v20.json"
BASELINE_FILE       = "results/baseline_eval_v17.json"
BACKUP_FILE          = "results/final_compare_v20_ga15_backup.json"


def main():
    with open(FINAL_COMPARE_FILE) as f:
        final_compare = json.load(f)
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)

    with open(BACKUP_FILE, "w") as f:
        json.dump(final_compare, f, indent=2)
    print(f"backed up previous (15-gen GA) results to {BACKUP_FILE}")

    missing = set(final_compare["SAC"].keys()) - set(baseline["GA"].keys())
    if missing:
        raise SystemExit(f"baseline_eval_v17.json GA is missing keys: {missing}")

    final_compare["GA"] = {k: baseline["GA"][k] for k in final_compare["SAC"].keys()}

    with open(FINAL_COMPARE_FILE, "w") as f:
        json.dump(final_compare, f, indent=2)
    print(f"spliced 300-gen GA into {FINAL_COMPARE_FILE}")

    import numpy as np
    print(f"  GA reward:  {np.mean(final_compare['GA']['episode_rewards']):.3f} "
          f"(was {np.mean(json.load(open(BACKUP_FILE))['GA']['episode_rewards']):.3f})")
    print(f"  GA runtime: {np.mean(final_compare['GA']['episode_runtime_sec']):.3f}s "
          f"(was {np.mean(json.load(open(BACKUP_FILE))['GA']['episode_runtime_sec']):.3f}s)")


if __name__ == "__main__":
    main()
