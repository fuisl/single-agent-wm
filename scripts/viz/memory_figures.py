"""Memory figures for the R-LeWM ladder, into runs/kin_v1/viz/figures/ (inputs that do not exist are skipped).

    source env.sh
    $PY_MAIN scripts/viz/memory_figures.py                       # every runs/kin_v1/*/memtest.json
    $PY_MAIN scripts/viz/memory_figures.py --run_dirs <smoke dirs> --out /tmp/x   # test

  mem_retention.png  E1 memory effect (novel - seen re-entry error, ± s.e.) vs time W is out of view
  mem_surprise.png   E2 surprise (removed - absent, ± s.e.) and violation-of-expectation accuracy vs gap
  mem_lesions.png    E3 belief models: memory effect vs the effect of swapping in the novel twin's belief / of
                     resetting it at the turn-back
  mem_traces.png     error around the turn-back for seen / novel / removed / absent (memtest_traces.npz), per model
  mem_probe_age.png  probe AP on remembered cells vs age: LeWM, M0-M4 and the DreamerV3 reference
"""
import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vizlib import model_label

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = argparse.ArgumentParser()
p.add_argument("--run", default=os.path.join(ROOT, "runs/kin_v1"))
p.add_argument("--run_dirs", nargs="*", default=None, help="default: every <run>/*/memtest.json")
p.add_argument("--trace_gap", type=int, default=40)
p.add_argument("--out", default=None)
args = p.parse_args()
OUT = args.out or os.path.join(args.run, "viz/figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13})
ORDER = ["lewm", "m0", "m1", "m2", "m3", "m4"]
COL = {"lewm": "#000000", "m0": "#56B4E9", "m1": "#0072B2", "m2": "#E69F00", "m3": "#D55E00", "m4": "#CC79A7",
       "wm": "#7f7f7f"}
VCOL = {"seen": "#0072B2", "novel": "#E69F00", "removed": "#D55E00", "absent": "#009E73", "swap": "#CC79A7",
        "reset": "#7f7f7f"}
TURN_S = 1.5  # the turn-away (and turn-back) takes 30 steps


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name), flush=True)


def short(rd):
    return os.path.basename(os.path.normpath(rd)).replace("smoke_", "")


dirs = args.run_dirs if args.run_dirs is not None else \
    [os.path.dirname(f) for f in glob.glob(os.path.join(args.run, "*", "memtest.json"))]
dirs = sorted([d for d in dirs if os.path.exists(os.path.join(d, "memtest.json"))],
              key=lambda d: ORDER.index(short(d)) if short(d) in ORDER else 99)
mt = {short(d): json.load(open(os.path.join(d, "memtest.json"))) for d in dirs}
tr = {short(d): dict(np.load(os.path.join(d, "memtest_traces.npz"))) for d in dirs
      if os.path.exists(os.path.join(d, "memtest_traces.npz"))}
print("memtest results:", list(mt) or "none", flush=True)


def has(m, key):
    return all(key in g for g in mt[m]["gaps"].values())


def series(m, key):
    gaps = sorted(int(g) for g in mt[m]["gaps"])
    g = [mt[m]["gaps"][str(x)] for x in gaps]
    return np.array(gaps) / 20 + TURN_S, np.array([x[key] for x in g]), np.array([x.get(key + "_se", 0) for x in g])


def retention():
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in mt:
        x, y, se = series(m, "memory_effect")
        ax.errorbar(x, y, se, marker="o", capsize=3, color=COL.get(m), label=model_label(m),
                    ls="--" if mt[m]["kind"] != "belief" else "-")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("time the wall is out of view (s): gap + 1.5 s of turning")
    ax.set_ylabel("memory effect: novel - seen\nre-entry error (norm.)")
    ax.set_title("Does the model expect the wall it saw before looking away?")
    ax.text(0.99, 0.02, "window models (dashed) are exactly 0 by construction: their context holds no frame of W",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555555")
    ax.legend(frameon=False, fontsize=10)
    save(fig, "mem_retention.png")


def surprise():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for m in [m for m in mt if has(m, "surprise")]:
        x, y, se = series(m, "surprise")
        ls = "--" if mt[m]["kind"] != "belief" else "-"
        ax[0].errorbar(x, y, se, marker="o", capsize=3, color=COL.get(m), ls=ls, label=model_label(m))
        x, acc, _ = series(m, "voe_acc")
        ax[1].plot(x, acc, marker="o", color=COL.get(m), ls=ls, label=model_label(m))
    ax[0].axhline(0, color="black", lw=0.8); ax[1].axhline(0.5, color="black", lw=0.8, ls=":")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("time the wall is out of view (s)")
    ax[0].set_ylabel("surprise: removed - absent\nre-entry error (norm.)")
    ax[0].set_title("Surprise when a remembered wall is gone")
    ax[1].set_ylabel("P(error removed > absent)"); ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Violation-of-expectation accuracy (chance 0.5)")
    ax[1].legend(frameon=False, fontsize=9)
    save(fig, "mem_surprise.png")


def lesions():
    bel = [m for m in mt if mt[m]["kind"] == "belief" and has(m, "swap_effect")]
    if not bel:
        print("note: no belief-model memtest, skipping mem_lesions.png"); return
    fig, ax = plt.subplots(1, len(bel), figsize=(5.5 * len(bel), 4.6), squeeze=False)
    for a, m in zip(ax[0], bel):
        gaps = sorted(int(g) for g in mt[m]["gaps"])
        x = np.arange(len(gaps))
        for i, (key, lab, col) in enumerate([("memory_effect", "memory effect (novel - seen)", VCOL["novel"]),
                                              ("swap_effect", "belief swapped with novel twin's", VCOL["swap"]),
                                              ("reset_effect", "belief reset at turn-back", VCOL["reset"])]):
            a.bar(x + (i - 1) * 0.27, [mt[m]["gaps"][str(g)].get(key, np.nan) for g in gaps], 0.27, color=col, label=lab)
        a.axhline(0, color="black", lw=0.8)
        a.set_xticks(x); a.set_xticklabels([f"{g / 20:g}" for g in gaps]); a.set_xlabel("gap (s)")
        a.set_title(model_label(m)); a.set_ylabel("change in re-entry error (norm.)")
    ax[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Lesions: is the memory in the belief?  (swap ≈ memory effect = the belief carries it)", fontsize=13)
    fig.tight_layout()
    save(fig, "mem_lesions.png")


def traces():
    if not tr:
        print("note: no memtest_traces.npz, skipping mem_traces.png"); return
    ms = list(tr)
    fig, ax = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 4.2), squeeze=False)
    for a, m in zip(ax[0], ms):
        gaps = sorted({int(k[3:6]) for k in tr[m]})
        g = min(gaps, key=lambda x: abs(x - args.trace_gap))
        for v in ["seen", "novel", "removed", "absent", "swap", "reset"]:
            key = f"gap{g:03d}_{v}"
            if key in tr[m]:
                y = tr[m][key]
                a.plot((np.arange(len(y)) - 6) * 0.25, y, color=VCOL[v], label=v,
                       lw=2 if v in ("seen", "novel") else 1.2, ls="-" if v in ("seen", "novel") else "--")
        a.axvline(0, color="black", lw=0.8, ls=":")
        a.set_title(f"{model_label(m)}\ngap {g / 20:g} s", fontsize=11)
        a.set_xlabel("time from the start of the turn-back (s)")
    ax[0, 0].set_ylabel("one-step latent error (norm.)")
    hl = {}
    for a in ax[0]:
        for h, lab in zip(*a.get_legend_handles_labels()):
            hl.setdefault(lab, h)
    fig.legend(hl.values(), hl.keys(), frameon=False, ncol=len(hl), loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "mem_traces.png")


def probe_age():
    res = {}
    for f in sorted(glob.glob(os.path.join(args.run, "probe*", "probe_results.json"))):
        for k, v in json.load(open(f)).items():
            res.setdefault(k, v)
    pick = [("wm", "wm"), ("lewm", "lewm"), ("frame", "frame")] + \
           [(m, f"{m}_{part}") for m in ORDER[1:] for part in ("belief", "pred") if f"{m}_{part}" in res]
    pick = [(m, k) for m, k in pick if k in res]
    if len(pick) < 2:
        print("note: fewer than 2 probe results, skipping mem_probe_age.png"); return
    bins = list(res[pick[0][1]]["mem_by_age"])
    lab = [f"{int(b.split('-')[0]) / 20:g}-{int(b.split('-')[1]) / 20:g}" for b in bins]
    lab[-1] = f">{int(bins[-1].split('-')[0]) / 20:g}"
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, k in pick:
        ax.plot(range(len(bins)), [res[k]["mem_by_age"][b]["ap"] for b in bins], marker="o",
                color=COL.get(m, "#E69F00" if m == "frame" else None), lw=2.2 if m == "wm" else 1.6,
                ls="--" if m == "wm" else ":" if m == "frame" else "-",
                label="current frame only (CNN)" if m == "frame" else f"{model_label(m)} [{k}]")
    ax.plot(range(len(bins)), [res[pick[0][1]]["mem_by_age"][b]["base_rate"] for b in bins], color="black",
            lw=0.8, ls=":", label="chance")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(lab)
    ax.set_xlabel("time since the obstacle cell was last visible (s)")
    ax.set_ylabel("obstacle AP on remembered cells")
    ax.set_title("Probe: memory of out-of-view obstacles, LeWM ladder vs DreamerV3 reference")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    save(fig, "mem_probe_age.png")


if mt:
    retention(); surprise(); lesions(); traces()
else:
    print("note: no memtest.json found, skipping memtest figures")
probe_age()
