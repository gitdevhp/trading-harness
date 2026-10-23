"""Record golden fingerprints. Run ONLY at a commit whose baseline behaviour
is the reference: python -m ace_harness.tests.make_golden
"""
import json
import os
import subprocess

from ace_harness.tests.golden_util import GOLDEN_SYSTEMS, prompt_hashes, run_golden_system

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden.json")


def main():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    golden = {
        "recorded_at_commit": commit,
        "prompts": prompt_hashes(),
        "systems": {s: run_golden_system(s) for s in GOLDEN_SYSTEMS},
    }
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2, sort_keys=True)
    print(json.dumps(golden, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
