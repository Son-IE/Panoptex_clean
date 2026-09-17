#!/usr/bin/env python3
"""
prior_report.py  --  turn the prior nodes' research CSVs into one readable report.

Run a session with the CSVs enabled, e.g.:

    ros2 launch risk_perception panoptex_sim.launch.py \
        enable_predictive_costmap:=true enable_spatial_prior:=true \
        prior_log_dir:=~/.panoptex/logs tracker_log_unconfirmed:=true

then point this at whatever landed in that directory:

    python tools/prior_report.py --log-dir ~/.panoptex/logs
    # or name files explicitly:
    python tools/prior_report.py --tracker object_tracker_*.csv \
        --predictive predictive_costmap_*.csv --spatial spatial_prior_*.csv

It prints, per research prior (semantic / behavioral / spatial-flow / relation /
predictive-costmap output), the input and output distributions -- percentile
tables and per-label breakdowns -- and, if matplotlib is present, writes
histograms and per-track timelines next to the CSVs (or --out-dir).

Pure stdlib + numpy; matplotlib optional (percentiles still print without it).
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:       # ImportError, or a NumPy 1.x/2.x ABI AttributeError
    HAVE_MPL = False

PCTS = (50, 90, 95, 99, 99.9)


# --------------------------------------------------------------------------- io

def _read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row, key, default=math.nan):
    v = row.get(key, "")
    if v in ("", "inf", "-inf", "nan", None):
        return math.inf if v == "inf" else (-math.inf if v == "-inf" else default)
    try:
        return float(v)
    except ValueError:
        return default


def _newest(pattern):
    hits = sorted(glob.glob(os.path.expanduser(pattern)), key=os.path.getmtime)
    return hits[-1] if hits else None


def _resolve(arg, log_dir, stem):
    if arg:
        return _newest(arg) or (arg if os.path.exists(arg) else None)
    if log_dir:
        return _newest(os.path.join(os.path.expanduser(log_dir), f"{stem}_*.csv"))
    return None


# --------------------------------------------------------------------- helpers

def _pct_line(name, arr):
    arr = np.asarray([a for a in arr if np.isfinite(a)], dtype=float)
    if arr.size == 0:
        print(f"  {name:<22} (no data)")
        return
    cells = "  ".join(f"p{p}={np.percentile(arr, p):.3f}" for p in PCTS)
    print(f"  {name:<22} n={arr.size:<6} mean={arr.mean():.3f}  {cells}")


def _hist(values, title, xlabel, out_path, vline=None, vlabel=None, overlay=None):
    if not HAVE_MPL:
        return
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=50, color="#378ADD", edgecolor="white",
             label="all" if overlay is not None else None)
    if overlay is not None:
        ov = np.asarray([v for v in overlay if np.isfinite(v)], dtype=float)
        if ov.size:
            plt.hist(ov, bins=50, color="#E24B4A", alpha=0.55,
                     edgecolor="white", label="moving-subset")
        plt.legend()
    if vline is not None:
        plt.axvline(vline, color="#111", linestyle="--",
                    label=vlabel or f"{vline:.2f}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path}")


def _section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# --------------------------------------------------------------------- reports

def report_tracker(rows, out_dir, static_labels):
    if not rows:
        return
    by_label = defaultdict(list)
    for r in rows:
        by_label[r.get("label", "?")].append(r)

    _section("SEMANTIC prior  (label -> category -> movability seed, consequence)")
    print(f"  {'label':<16}{'category':<12}{'p_mov_prior':>12}"
          f"{'consequence':>13}{'tracks':>9}{'rows':>8}")
    for label, rs in sorted(by_label.items()):
        cat = rs[0].get("category", "?")
        pmp = np.nanmean([_f(r, "p_movable_prior") for r in rs])
        cons = np.nanmean([_f(r, "consequence") for r in rs])
        ntr = len({r.get("track_id") for r in rs})
        print(f"  {label:<16}{cat:<12}{pmp:>12.2f}{cons:>13.2f}{ntr:>9}{len(rs):>8}")

    _section("BEHAVIORAL prior  (motion_score -> p_motion latch, p_movable memory)")
    thr = _f(rows[0], "motion_threshold")
    print(f"  configured motion_threshold = {thr:.2f}\n")
    print("  motion_score percentiles:")
    _pct_line("ALL", [_f(r, "motion_score") for r in rows])
    static_scores = []
    for label, rs in sorted(by_label.items()):
        s = [_f(r, "motion_score") for r in rs]
        _pct_line(label, s)
        if static_labels and label in static_labels:
            static_scores += s
    null = static_scores or [_f(r, "motion_score") for r in rows]
    p99 = np.percentile([x for x in null if np.isfinite(x)], 99)
    tag = "static-label subset" if static_scores else "ALL rows (no --static-labels)"
    print(f"\n  suggested motion_threshold = {p99 * 1.1:.2f}  "
          f"(99th pct x1.1 of {tag})")

    print("\n  p_motion>0.5 / moving==1 fraction, and p_movable vs its prior:")
    print(f"  {'label':<16}{'pmot>0.5':>10}{'moving':>9}"
          f"{'pmov_mean':>11}{'pmov_min':>10}{'pmov_max':>10}")
    for label, rs in sorted(by_label.items()):
        pm = np.mean([_f(r, "p_motion") > 0.5 for r in rs])
        mv = np.mean([_f(r, "moving") > 0.5 for r in rs])
        pmov = np.array([_f(r, "p_movable") for r in rs])
        print(f"  {label:<16}{pm:>10.2f}{mv:>9.2f}{np.nanmean(pmov):>11.2f}"
              f"{np.nanmin(pmov):>10.2f}{np.nanmax(pmov):>10.2f}")

    conf_all = [_f(r, "confidence") for r in rows]
    n_unconf = sum(1 for r in rows if _f(r, "confirmed") < 0.5)
    n_notalive = sum(1 for r in rows if _f(r, "alive") < 0.5)
    _section("CONFIRM / EVICT gates  (min_hits, min_confidence effect)")
    _pct_line("confidence ALL", conf_all)
    print(f"  rows with confirmed=0 : {n_unconf} / {len(rows)}"
          f"   (need log_unconfirmed:=true to see these)")
    print(f"  rows with alive=0     : {n_notalive} / {len(rows)}")
    for label, rs in sorted(by_label.items()):
        unc = sum(1 for r in rs if _f(r, "confirmed") < 0.5)
        if unc:
            hits = [_f(r, "hits") for r in rs if _f(r, "confirmed") < 0.5]
            print(f"    {label:<16} {unc} unconfirmed rows, hits "
                  f"{np.nanmin(hits):.0f}-{np.nanmax(hits):.0f}, "
                  f"conf p50={np.nanpercentile([_f(r,'confidence') for r in rs], 50):.3f}")

    _section("RELATION prior  (relconf events -> relation_bonus)")
    ev = [r for r in rows if _f(r, "relconf_in") > 0.0]
    print(f"  relation events (relconf_in>0): {len(ev)} rows across "
          f"{len({r.get('track_id') for r in ev})} tracks")
    if ev:
        _pct_line("relconf_in", [_f(r, "relconf_in") for r in ev])
    _pct_line("relation_bonus (all)", [_f(r, "relation_bonus") for r in rows])

    if out_dir:
        _section("BEHAVIORAL plots")
        _hist([_f(r, "motion_score") for r in rows],
              "motion_score (null vs moving)", "motion_score  (Mahalanobis v)",
              os.path.join(out_dir, "tracker_motion_score_hist.png"),
              vline=thr, vlabel=f"threshold {thr:.2f}",
              overlay=[_f(r, "motion_score") for r in rows
                       if _f(r, "moving") > 0.5])
        _timelines(rows, out_dir)


def _timelines(rows, out_dir, top_n=6):
    if not HAVE_MPL:
        return
    by_id = defaultdict(list)
    for r in rows:
        by_id[r.get("track_id", "?")].append(r)
    top = sorted(by_id.items(), key=lambda kv: -len(kv[1]))[:top_n]
    for tid, rs in top:
        rs = sorted(rs, key=lambda r: _f(r, "t"))
        t0 = _f(rs[0], "t")
        t = [_f(r, "t") - t0 for r in rs]
        label = rs[0].get("label", "?")
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].plot(t, [_f(r, "motion_score") for r in rs], label="motion_score")
        ax[0].axhline(_f(rs[0], "motion_threshold"), color="#888", ls="--",
                      label="threshold")
        ax[0].set_ylabel("motion_score")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].set_title(f"track {tid} ({label})")
        ax[1].plot(t, [_f(r, "p_motion") for r in rs], label="p_motion")
        ax[1].plot(t, [_f(r, "p_movable") for r in rs], label="p_movable")
        ax[1].plot(t, [_f(r, "relation_bonus") for r in rs], label="relation_bonus")
        ax[1].plot(t, [_f(r, "confidence") for r in rs], label="confidence")
        ax[1].set_ylim(-0.05, 1.05)
        ax[1].set_xlabel("t since first row (s)")
        ax[1].legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        p = os.path.join(out_dir, f"tracker_timeline_track{tid}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  saved {p}")


def report_spatial(rows, out_dir):
    if not rows:
        return
    dep = [r for r in rows if r.get("kind") == "deposit"]
    grid = [r for r in rows if r.get("kind") == "grid"]
    _section("SPATIAL-FLOW prior  (two Behavioral gates -> per-category deposit)")
    if dep:
        g1 = np.mean([_f(r, "gate1_pass") > 0.5 for r in dep])
        g2 = np.mean([_f(r, "gate2_pass") > 0.5 for r in dep])
        depd = np.mean([_f(r, "deposited") > 0.5 for r in dep])
        print(f"  deposit evaluations: {len(dep)}")
        print(f"  gate1 (p_movable>=min) pass rate : {g1:.2f}")
        print(f"  gate2 (p_motion & conf)  pass rate : {g2:.2f}")
        print(f"  actually deposited                 : {depd:.2f}")
        # why gate2 fails, among gate1-pass rows
        cand = [r for r in dep if _f(r, "gate1_pass") > 0.5]
        if cand:
            pm_low = sum(1 for r in cand
                         if _f(r, "p_motion") < _f(r, "p_motion_min"))
            cf_low = sum(1 for r in cand
                         if _f(r, "confidence") < _f(r, "min_confidence"))
            print(f"  of {len(cand)} gate1-pass rows: p_motion too low {pm_low}, "
                  f"confidence too low {cf_low}")
        print()
        print(f"  {'category':<12}{'evals':>8}{'g1':>7}{'g2':>7}{'dep':>7}"
              f"{'k_mean':>10}{'radius_mean':>13}")
        by_cat = defaultdict(list)
        for r in dep:
            by_cat[r.get("category", "?")].append(r)
        for cat, rs in sorted(by_cat.items()):
            print(f"  {cat:<12}{len(rs):>8}"
                  f"{np.mean([_f(r,'gate1_pass')>0.5 for r in rs]):>7.2f}"
                  f"{np.mean([_f(r,'gate2_pass')>0.5 for r in rs]):>7.2f}"
                  f"{np.mean([_f(r,'deposited')>0.5 for r in rs]):>7.2f}"
                  f"{np.nanmean([_f(r,'k') for r in rs]):>10.4f}"
                  f"{np.nanmean([_f(r,'radius') for r in rs]):>13.3f}")
    if grid:
        print("\n  learned grid state (last row per category):")
        last = {}
        for r in grid:
            last[r.get("category", "?")] = r
        print(f"  {'category':<12}{'s_max':>9}{'s_nnz':>9}"
              f"{'f_speed_max':>13}{'f_speed_mean':>14}")
        for cat, r in sorted(last.items()):
            print(f"  {cat:<12}{_f(r,'s_max'):>9.3f}{_f(r,'s_nnz'):>9.0f}"
                  f"{_f(r,'f_speed_max'):>13.3f}{_f(r,'f_speed_mean'):>14.3f}")
    if out_dir and dep:
        _hist([_f(r, "p_motion") for r in dep],
              "spatial_prior: p_motion at gate2", "p_motion",
              os.path.join(out_dir, "spatial_gate2_pmotion_hist.png"),
              vline=_f(dep[0], "p_motion_min"), vlabel="p_motion_min")


def report_predictive(rows, out_dir):
    if not rows:
        return
    _section("PREDICTIVE costmap output  (w_stat vs w_move -> forward smear?)")
    by_label = defaultdict(list)
    for r in rows:
        by_label[r.get("label", "?")].append(r)
    print(f"  {'label':<16}{'rows':>7}{'w_stat':>9}{'w_move0':>10}"
          f"{'steps':>8}{'smeared%':>10}{'roll_m':>9}{'cpa_x':>8}")
    for label, rs in sorted(by_label.items()):
        ws = np.nanmean([_f(r, "w_stat") for r in rs])
        wm = np.nanmean([_f(r, "w_move_step0") for r in rs])
        st = np.nanmean([_f(r, "n_move_steps") for r in rs])
        sm = np.mean([_f(r, "n_move_steps") > 0 for r in rs]) * 100.0
        rl = np.nanmean([math.hypot(_f(r, "rollout_dx", 0.0),
                                    _f(r, "rollout_dy", 0.0)) for r in rs])
        cx = np.nanmean([_f(r, "cpa_factor") for r in rs])
        print(f"  {label:<16}{len(rs):>7}{ws:>9.3f}{wm:>10.3f}"
              f"{st:>8.1f}{sm:>10.1f}{rl:>9.2f}{cx:>8.2f}")
    tot = len(rows)
    smeared = sum(1 for r in rows if _f(r, "n_move_steps") > 0)
    far = sum(1 for r in rows if math.hypot(_f(r, "rollout_dx", 0.0),
                                            _f(r, "rollout_dy", 0.0)) > 0.10)
    print(f"\n  track-ticks with a moving hypothesis splatted : {smeared}/{tot}")
    print(f"  track-ticks whose rollout moved > 1 cell (0.1 m): {far}/{tot}")
    print(f"  use_motion_mixture={rows[0].get('use_motion_mixture')}  "
          f"flow_blend_weight={rows[0].get('flow_blend_weight')}  "
          f"spatial_prior_weight={rows[0].get('spatial_prior_weight')}")
    if smeared == 0:
        print("  --> NO forward smear anywhere: every track had w_move<=0.001 "
              "on step 0. In 'pmot' gate check the tracker CSV's p_motion; in "
              "'speed' gate check that any track has |v| > move_min_speed.")

    # RELATION prior -- relbonus feeds C additively (predictive fusion step).
    on = [r for r in rows if _f(r, "relbonus") > 0.0]
    _section("RELATION prior fusion  (relbonus -> C, in the predictive CSV)")
    print(f"  ticks with relbonus>0 : {len(on)} / {tot}")
    if on:
        _pct_line("relbonus (when >0)", [_f(r, "relbonus") for r in on])
        c_on = np.nanmean([_f(r, "C") for r in on])
        c_off = np.nanmean([_f(r, "C") for r in rows if _f(r, "relbonus") == 0.0])
        cons_on = np.nanmean([_f(r, "consequence") for r in on])
        print(f"  mean C  with relbonus : {c_on:.3f}   without : {c_off:.3f}"
              f"   (mean consequence {cons_on:.3f}, so the lift is the "
              f"relation term)")
        if out_dir:
            _hist([_f(r, "relbonus") for r in rows],
                  "predictive: relbonus per tick (scenario_publisher --rel-pulse)",
                  "relbonus", os.path.join(out_dir, "predictive_relbonus_hist.png"))
            # ONE track at a time -- sorting ALL rows (every track, every
            # tick) by t alone and plotting them as one line zig-zags
            # between DIFFERENT tracks' consequence at each timestamp
            # (e.g. a person's 0.75 next to a door's 0.15 next to a
            # forklift's 0.5, all sharing one ~5 Hz publish tick) and reads
            # as wild high-frequency noise that isn't really there for any
            # single object. Plot the track(s) that actually carried
            # relbonus, each on its own axes.
            on_ids = sorted(
                {r.get("track_id", "?") for r in on},
                key=lambda tid: -sum(1 for r in on if r.get("track_id") == tid))
            for tid in on_ids[:3]:
                trs = sorted(
                    [r for r in rows if r.get("track_id") == tid],
                    key=lambda r: _f(r, "t"))
                if len(trs) <= 2 or not HAVE_MPL:
                    continue
                t0 = _f(trs[0], "t")
                tt = [_f(r, "t") - t0 for r in trs]
                plt.figure(figsize=(9, 3.5))
                plt.plot(tt, [_f(r, "relbonus") for r in trs], label="relbonus")
                plt.plot(tt, [_f(r, "C") for r in trs], label="C")
                plt.plot(tt, [_f(r, "consequence") for r in trs],
                         label="consequence", ls="--")
                plt.xlabel("t (s)")
                plt.legend()
                plt.title(f"relation-prior fusion over time -- track {tid} "
                          f"({trs[0].get('label', '?')})")
                plt.tight_layout()
                p = os.path.join(out_dir, f"predictive_relation_timeline_track{tid}.png")
                plt.savefig(p, dpi=130)
                plt.close()
                print(f"  saved {p}")
    else:
        print("  (no relbonus seen -- relation_prompt is off and nothing "
              "published relconf; run tools/scenario_publisher.py "
              "--scenario relation_test --rel-pulse against this node)")
    if out_dir:
        _hist([_f(r, "w_move_step0") for r in rows],
              "predictive: w_move at step 0 (0 => only a round blob)",
              "w_move_step0", os.path.join(out_dir, "predictive_wmove_hist.png"),
              vline=0.001, vlabel="splat cutoff")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="",
                    help="directory the nodes wrote <stem>_<UTC>.csv into; "
                         "the newest of each is used")
    ap.add_argument("--tracker", default="", help="object_tracker CSV (glob ok)")
    ap.add_argument("--spatial", default="", help="spatial_prior CSV (glob ok)")
    ap.add_argument("--predictive", default="", help="predictive_costmap CSV (glob ok)")
    ap.add_argument("--static-labels", default="",
                    help="comma list of labels that should be motionless "
                         "(chair,table,pallet,...): used as the motion_score "
                         "null distribution for the threshold suggestion")
    ap.add_argument("--out-dir", default="",
                    help="where to write PNGs (default: next to the CSVs; "
                         "'-' disables plots)")
    a = ap.parse_args()

    tracker = _resolve(a.tracker, a.log_dir, "object_tracker")
    spatial = _resolve(a.spatial, a.log_dir, "spatial_prior")
    predictive = _resolve(a.predictive, a.log_dir, "predictive_costmap")
    print(f"tracker    : {tracker or '(none)'}")
    print(f"spatial    : {spatial or '(none)'}")
    print(f"predictive : {predictive or '(none)'}")
    if not any((tracker, spatial, predictive)):
        raise SystemExit("no CSVs found -- pass --log-dir or --tracker/--spatial/--predictive")

    if a.out_dir == "-":
        out_dir = ""
    elif a.out_dir:
        out_dir = os.path.expanduser(a.out_dir)
        os.makedirs(out_dir, exist_ok=True)
    else:
        anchor = tracker or spatial or predictive
        out_dir = os.path.dirname(os.path.abspath(anchor))
    if out_dir and not HAVE_MPL:
        print("\n(matplotlib not installed -- tables only, no PNGs)")

    static = {s.strip() for s in a.static_labels.split(",") if s.strip()}
    report_tracker(_read_csv(tracker), out_dir, static)
    report_spatial(_read_csv(spatial), out_dir)
    report_predictive(_read_csv(predictive), out_dir)


if __name__ == "__main__":
    main()
