#!/usr/bin/env python3
"""check_arm_params.py -- guard a baseline/panoptex nav2 params PAIR.

Each study arm comes as two params files that are meant to be identical in
every respect EXCEPT the handful of Panoptex-only additions (the risk costmap
layers, the PredictedRisk* controller critic, and the speed-limit topic
controller_server listens on). This script deep-diffs one such pair and fails
loudly if anything else has drifted apart -- e.g. someone tunes a critic
weight in one file and forgets the other.

Two pairs exist and both run under `colcon test`:

    DWB   config/nav2_x3_baseline.yaml       config/nav2_x3_panoptex.yaml
    MPPI  config/nav2_x3_baseline_mppi.yaml  config/nav2_x3_panoptex_mppi.yaml

Usage:
    python3 tools/check_arm_params.py                      # the DWB pair
    python3 tools/check_arm_params.py --baseline A --arm B
    python3 tools/check_arm_params.py A B                  # legacy positional

Exits 0 if the only differences are on the allow-list below, non-zero
(with the offending diff printed) otherwise.
"""
import argparse
import sys
from pathlib import Path

import yaml

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE = THIS_DIR.parent / "config" / "nav2_x3_baseline.yaml"
DEFAULT_PANOPTEX = THIS_DIR.parent / "config" / "nav2_x3_panoptex.yaml"

# Each entry is a tuple of dict keys (a "path") that is allowed to differ
# between the two files. A path also covers everything nested under it
# (e.g. the risk_layer entry covers risk_layer.topic, risk_layer.max_cost, ...).
ALLOWED_PATHS = [
    ("global_costmap", "global_costmap", "ros__parameters", "plugins"),
    ("global_costmap", "global_costmap", "ros__parameters", "risk_layer"),
    # WP-C: lane_layer -- a second RiskLayer instance reading the learned
    # AMR-lane spatial prior (see nav2_x3_panoptex.yaml's own comment and
    # panoptex_nav/README.md's "Lane layer" section). Panoptex-only, same
    # as risk_layer above.
    ("global_costmap", "global_costmap", "ros__parameters", "lane_layer"),
    ("controller_server", "ros__parameters", "FollowPath", "critics"),
    ("controller_server", "ros__parameters", "speed_limit_topic"),
]

# FollowPath.PredictedRisk.* are stored as flat dotted-string keys (nav2's
# own convention for namespacing plugin params in YAML), not a nested dict,
# so they need a prefix-startswith check rather than an exact path.
# The MPPI arm's critic is listed as `PredictedRiskCritic` (nav2_mppi_controller
# resolves every `critics:` entry as "mppi::critics::" + name, so the short name
# is fixed by the plugin's lookup name), the DWB arm's as `PredictedRisk` -- one
# prefix covers both.
PREDICTED_RISK_PREFIX = ("controller_server", "ros__parameters", "FollowPath")
PREDICTED_RISK_KEY_PREFIX = "PredictedRisk"


def is_allowed(path):
    for allowed in ALLOWED_PATHS:
        if path[: len(allowed)] == allowed:
            return True
    if (
        len(path) >= len(PREDICTED_RISK_PREFIX) + 1
        and path[: len(PREDICTED_RISK_PREFIX)] == PREDICTED_RISK_PREFIX
        and isinstance(path[len(PREDICTED_RISK_PREFIX)], str)
        and path[len(PREDICTED_RISK_PREFIX)].startswith(PREDICTED_RISK_KEY_PREFIX)
    ):
        return True
    return False


def diff(a, b, path=()):
    """Yield (path_tuple, kind, value_a, value_b) for every leaf difference.

    kind is one of "added" (only in b), "removed" (only in a), "changed"
    (present in both, different value). Dicts are recursed into; any other
    type (including lists, so critics/plugins compare as whole values) is
    treated as a leaf.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub_path = path + (key,)
            if key not in a:
                yield (sub_path, "added", None, b[key])
            elif key not in b:
                yield (sub_path, "removed", a[key], None)
            else:
                yield from diff(a[key], b[key], sub_path)
    else:
        if a != b:
            yield (path, "changed", a, b)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=None,
                        help="baseline arm params yaml (default: the DWB baseline)")
    parser.add_argument("--arm", type=Path, default=None,
                        help="panoptex arm params yaml (default: the DWB panoptex arm)")
    parser.add_argument("positional", nargs="*", type=Path,
                        help="legacy: [baseline.yaml] [panoptex.yaml]")
    args = parser.parse_args(argv)

    baseline = args.baseline
    arm = args.arm
    if baseline is None and len(args.positional) > 0:
        baseline = args.positional[0]
    if arm is None and len(args.positional) > 1:
        arm = args.positional[1]
    return (baseline or DEFAULT_BASELINE), (arm or DEFAULT_PANOPTEX)


def main():
    baseline_path, panoptex_path = parse_args(sys.argv[1:])

    with open(baseline_path) as f:
        baseline = yaml.safe_load(f)
    with open(panoptex_path) as f:
        panoptex = yaml.safe_load(f)

    diffs = list(diff(baseline, panoptex))

    if not diffs:
        print("No differences at all between {} and {} -- that's suspicious "
              "for supposedly-different study arms.".format(baseline_path, panoptex_path))
        return 1

    violations = []
    print("Full diff between {} (a) and {} (b):".format(baseline_path, panoptex_path))
    for path, kind, va, vb in diffs:
        allowed = is_allowed(path)
        marker = "OK " if allowed else "BAD"
        dotted = ".".join(str(p) for p in path)
        print("  [{}] {} :: {} :: a={!r} b={!r}".format(marker, dotted, kind, va, vb))
        if not allowed:
            violations.append((path, kind, va, vb))

    if violations:
        print()
        print("FAIL: {} disallowed difference(s) found between the two arms' "
              "params files. Only the risk_layer/lane_layer/plugins/critics/"
              "PredictedRisk*/speed_limit_topic keys listed in ALLOWED_PATHS "
              "may differ."
              .format(len(violations)))
        return 1

    print()
    print("OK: every difference between the two files is on the allow-list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
