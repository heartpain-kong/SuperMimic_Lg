"""Figures and the results table of the README, computed from the evaluation files in results/.

    python tools/make_figures.py            # writes figures/*.png (light and dark) and prints the markdown table

Time to target follows BlinkMimic's tools/time_to_target.py:
  first   the first checkpoint with normalized evaluation return >= 0.82;
  held x3 the first checkpoint from which three consecutive checkpoints (100 iterations apart) all have return >= 0.82
          and 0 of 40 failed clips.
Training time excludes compilation for every method (see results/README.md).
"""

from __future__ import annotations

import csv
import os
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS, FIGURES = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
TARGET, HELD, CKPT_EVERY = 0.82, 3, 100

# Colors: the palette of the combined XMimic + XSoma paper. XMimic in teal, one shade per GPU count (light theme: 1 GPU light
# to 4 GPUs dark; dark theme the other way round, so more GPUs always stand out more), the baselines in grays outside the
# teal ramp's lightness range with BlinkMimic, the reference, the most prominent, so every gray and teal stay apart under
# color blindness too. The surfaces match GitHub's light and dark page.
THEMES = {
    "light": dict(surface="#ffffff", ink="#1d1d1f", ink2="#48484e", muted="#8e8e93", grid="#e5e5ea", axis="#c7c7cc",
                  ring="#ffffff", gpus={1: "#0db3c1", 2: "#009aa8", 3: "#00818f", 4: "#006977"}, blinkmimic="#3d3d40",
                  instanttrack="#c3c3c8"),
    "dark": dict(surface="#0d1117", ink="#f0f6fc", ink2="#c9d1d9", muted="#8b949e", grid="#21262d", axis="#30363d",
                 ring="#0d1117", gpus={1: "#008492", 2: "#009aa8", 3: "#00b0be", 4: "#36c6d4"}, blinkmimic="#f1f1f6",
                 instanttrack="#525256"),
}
GPUS = {"xmimic_1gpu": 1, "xmimic_fp1": 1, "xmimic_fp": 2, "xmimic_al_mps": 2, "xmimic_alp": 3, "xmimic_awl_mps": 3, "xmimic_awlp": 4}
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

# the README's main comparison: the baselines and XMimic on one to four GPUs (one GPU is the like-for-like comparison)
MAIN = ["blinkmimic", "instanttrack", "xmimic_1gpu", "xmimic_al_mps", "xmimic_awl_mps", "xmimic_awlp"]
ALL = ["blinkmimic", "instanttrack", "xmimic_1gpu", "xmimic_fp1", "xmimic_fp", "xmimic_al_mps", "xmimic_alp",
       "xmimic_awl_mps", "xmimic_awlp"]
# every layout by the GPUs it uses and its --layout name (README: "Every layout")
SHORT = {"blinkmimic": "BlinkMimic", "instanttrack": "InstantTrack v1.3", "xmimic_1gpu": "XMimic, 1 GPU",
         "xmimic_fp1": "1 GPU + probe process (FP1)", "xmimic_fp": "2 GPUs: trainer | prober (FP)",
         "xmimic_al_mps": "2 GPUs: actor | learner (AL)", "xmimic_alp": "3 GPUs: actor | learner | prober (ALP)",
         "xmimic_awl_mps": "3 GPUs: actor | world model | policy (AWL)",
         "xmimic_awlp": "4 GPUs: actor | world model | policy | prober (AWLP)"}
MAIN_LABEL = {"blinkmimic": "BlinkMimic", "instanttrack": "InstantTrack v1.3", "xmimic_1gpu": "XMimic · 1 GPU",
              "xmimic_al_mps": "XMimic · 2 GPUs", "xmimic_awl_mps": "XMimic · 3 GPUs", "xmimic_awlp": "XMimic · 4 GPUs"}


# ---------------------------------------------------------------- data
def load_eval(path):
    """[(iteration, training minutes, normalized return, failed clips)] sorted by iteration."""
    rows = [(int(r["iteration"]), float(r["train_wall_h"]) * 60, float(r["norm_return"]), float(r["n_failed"]))
            for r in csv.DictReader(open(path))]  # fmt: skip
    return sorted(rows)


def time_to_target(rows):
    """(first, held) as (iteration, minutes) or None."""
    first = next(((it, m) for it, m, ret, _ in rows if ret >= TARGET), None)
    ok = {it: ret >= TARGET and nf == 0 for it, _, ret, nf in rows}
    held = next(((it, m) for it, m, _, _ in rows
                 if it % CKPT_EVERY == 0 and all(ok.get(it + j * CKPT_EVERY, False) for j in range(HELD))), None)  # fmt: skip
    return first, held


def load_results():
    """{method: {"label", "gpus", "runs": [rows per run], "seeds": [...]}} from results/runs.csv."""
    out = defaultdict(lambda: {"runs": [], "seeds": []})
    for r in csv.DictReader(open(os.path.join(RESULTS, "runs.csv"))):
        m = out[r["method"]]
        m["label"], m["gpus"] = r["label"], int(r["gpus"])
        m["runs"].append(load_eval(os.path.join(RESULTS, r["file"])))
        m["seeds"].append(int(r["seed"]))
    return dict(out)


def summarize(res):
    """Per method: per-run first / held minutes and iterations, means, it/s over the whole run, speedup vs BlinkMimic."""
    summ = {}
    for key, m in res.items():
        tt = [time_to_target(r) for r in m["runs"]]
        assert all(f and h for f, h in tt), f"{key}: a run never reached the target"
        summ[key] = {
            "first": [f[1] for f, _ in tt], "held": [h[1] for _, h in tt],
            "first_it": [f[0] for f, _ in tt], "held_it": [h[0] for _, h in tt],
            "ips": [r[-1][0] / (r[-1][1] * 60) for r in m["runs"]],
            "final_ret": [r[-1][2] for r in m["runs"]],
        }
    base = np.mean(summ["blinkmimic"]["held"])
    for key, s in summ.items():
        s["speedup"] = base / np.mean(s["held"])
        s["gpu_min"] = res[key]["gpus"] * np.mean(s["held"])
    return summ


# ---------------------------------------------------------------- drawing helpers
def setup(theme):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 11, "axes.titlesize": 12,
                         "svg.fonttype": "none"})
    return THEMES[theme]


def color_of(key, t):
    return t[key] if key in ("blinkmimic", "instanttrack") else t["gpus"][GPUS[key]]


def style_axes(ax, t, grid_axis="x"):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=t["ink2"], labelsize=10.5, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], lw=1.0)
    ax.set_axisbelow(True)


def rounded_bar(ax, y, value, height, color, radius_px=4.0):
    """A horizontal bar from 0 to ``value``: square at the baseline, corners rounded at the data end (radius in pixels)."""
    to_px = ax.transData.transform
    (x0p, y0p), (x1p, y1p) = to_px((0, 0)), to_px((1, 1))
    rx = radius_px / max(abs(x1p - x0p), 1e-9)  # data units per radius, x and y
    ry = radius_px / max(abs(y1p - y0p), 1e-9)
    rx = min(rx, value / 2)
    ry = min(ry, height / 2)
    k = 0.5523  # cubic Bezier quarter circle
    yb, yt, xe = y - height / 2, y + height / 2, value
    verts = [(0, yb), (xe - rx, yb), (xe - rx + k * rx, yb), (xe, yb + ry - k * ry), (xe, yb + ry),
             (xe, yt - ry), (xe, yt - ry + k * ry), (xe - rx + k * rx, yt), (xe - rx, yt), (0, yt), (0, yb)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def save(fig, name, theme, t):
    suffix = "" if theme == "light" else "_dark"
    fig.savefig(os.path.join(FIGURES, f"{name}{suffix}.png"), dpi=200, facecolor=t["surface"])
    plt.close(fig)


# ---------------------------------------------------------------- figures
def bars(res, summ, keys, labels, name, theme, title, subtitle, left_in=2.0):
    """Time to target (held x3): one row per method, bar = mean over runs, rings = individual runs, values in a column."""
    t = setup(theme)
    n = len(keys)
    fig_h = 1.3 + 0.42 * n
    fig = plt.figure(figsize=(9.0, fig_h), facecolor=t["surface"])
    left, right = left_in / 9.0, 1.85 / 9.0
    bottom, top = 0.62 / fig_h, 1.0 / fig_h
    ax = fig.add_axes([left, bottom, 1 - left - right, 1 - bottom - top])
    style_axes(ax, t)
    xmax = max(max(summ[k]["held"]) for k in keys) * 1.04
    ax.set_xlim(0, xmax)
    ax.set_ylim(n - 0.5, -0.5)
    col = blended_transform_factory(ax.transAxes, ax.transData)
    axw = (1 - left - right) * 9.0  # axes width in inches: the value columns sit at fixed distances right of the axes
    x_mean, x_speed = 1 + 0.17 / axw, 1 + 1.12 / axw
    for i, k in enumerate(keys):
        s = summ[k]
        mean = float(np.mean(s["held"]))
        rounded_bar(ax, i, mean, 0.52, color_of(k, t))
        ax.scatter(s["held"], [i] * len(s["held"]), s=30, facecolor=t["surface"], edgecolors=t["ink"], linewidths=1.3, zorder=3)
        sp = s["speedup"]
        ax.text(x_mean, i, f"{mean:.2f} min", transform=col, va="center", ha="left", color=t["ink"], fontsize=11,
                fontweight="bold", clip_on=False)
        ax.text(x_speed, i, f"{sp:.1f}×" if abs(sp - 1) >= 0.05 else f"{sp:.2f}×", transform=col, va="center", ha="left",
                color=t["ink2"], fontsize=11, clip_on=False)
    ax.text(x_mean, -0.85, "mean", transform=col, va="center", ha="left", color=t["muted"], fontsize=9.5, clip_on=False)
    ax.text(x_speed, -0.85, "speedup", transform=col, va="center", ha="left", color=t["muted"], fontsize=9.5, clip_on=False)
    ax.set_yticks(range(n), [labels[k] for k in keys], color=t["ink"], fontsize=11.5)
    ax.tick_params(axis="y", pad=8)
    ax.set_xlabel("training time to the target (minutes, compilation excluded)", color=t["ink2"], fontsize=10.5)
    fig.text(0.012, 1 - 0.30 / fig_h, title, ha="left", va="center", color=t["ink"], fontsize=13.5, fontweight="bold")
    fig.text(0.012, 1 - 0.58 / fig_h, subtitle, ha="left", va="center", color=t["ink2"], fontsize=10.5)
    save(fig, name, theme, t)


def curves(res, keys, labels, name, theme):
    """Evaluation return and failed clips against training time: mean over runs, band = min..max over runs."""
    t = setup(theme)
    fig = plt.figure(figsize=(9.0, 4.2), facecolor=t["surface"])
    axr = fig.add_axes([0.075, 0.14, 0.50, 0.62])
    axf = fig.add_axes([0.665, 0.14, 0.315, 0.62])
    tmax = float(np.floor(min(r[-1][1] for k in keys for r in res[k]["runs"])))  # whole minutes that every run covers
    grid = np.linspace(0, tmax, 600)
    for k in keys:  # baselines first, XMimic on top
        col = color_of(k, t)
        runs = res[k]["runs"]
        t_end = min(r[-1][1] for r in runs)  # average only where every run has data
        g = grid[grid <= t_end]
        for ax, j in ((axr, 2), (axf, 3)):
            c = []
            for r in runs:
                tt, vv = np.array([x[1] for x in r]), np.array([x[j] for x in r])
                c.append(np.where(g >= tt[0], np.interp(g, tt, vv), np.nan))
            c = np.array(c)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # grid points before a method's first evaluation
                mean, lo, hi = np.nanmean(c, 0), np.nanmin(c, 0), np.nanmax(c, 0)
            ax.fill_between(g, lo, hi, color=col, alpha=0.14, lw=0, zorder=2)
            ax.plot(g, mean, color=col, lw=2.0, solid_capstyle="round", solid_joinstyle="round", zorder=3,
                    label=labels[k] if ax is axr else None)
    for ax in (axr, axf):
        style_axes(ax, t, grid_axis="both")
        ax.set_xlim(0, tmax)
        ax.set_xlabel("training time (minutes)", color=t["ink2"], fontsize=10.5)
    axr.set_ylim(0, 0.9)
    axr.axhline(TARGET, color=t["muted"], lw=1.0, zorder=1)
    axr.text(tmax * 0.995, TARGET - 0.015, f"target {TARGET}", ha="right", va="top", color=t["ink2"], fontsize=9.5)
    axf.set_ylim(-0.6, 40.6)
    axf.set_yticks([0, 10, 20, 30, 40])
    axr.set_title("Evaluation return (normalized)", loc="left", color=t["ink"], fontsize=11.5, pad=8)
    axf.set_title("Failed clips (of 40)", loc="left", color=t["ink"], fontsize=11.5, pad=8)
    handles, labs = axr.get_legend_handles_labels()
    want = [labels[k] for k in keys]
    hl = sorted(zip(handles, labs), key=lambda x: want.index(x[1]))
    axr.legend([h for h, _ in hl], [lb for _, lb in hl], frameon=False, fontsize=10, labelcolor=t["ink"], loc="lower right",
               handlelength=1.6, borderaxespad=0.3)
    fig.text(0.012, 0.945, "Learning curves on H100", ha="left", va="center", color=t["ink"], fontsize=13.5,
             fontweight="bold")
    fig.text(0.012, 0.890, "every checkpoint evaluated on all 40 clips × 4 replicas · line = mean over runs, band = min to max",
             ha="left", va="center", color=t["ink2"], fontsize=10.5)
    save(fig, name, theme, t)


# ---------------------------------------------------------------- table
def table(res, summ, keys, labels):
    lines = ["| method | GPUs | runs | it/s | first ≥ 0.82 (min) | held ×3 (min) | speedup | GPU-min to target |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        s, m = summ[k], res[k]
        bold = k == "xmimic_awlp"
        held = f"{np.mean(s['held']):.2f} ({min(s['held']):.2f}–{max(s['held']):.2f})"
        first = f"{np.mean(s['first']):.2f} ({min(s['first']):.2f}–{max(s['first']):.2f})"
        sp = f"{s['speedup']:.2f}×"
        row = [labels[k].replace("|", "\\|"), str(m["gpus"]), str(len(m["runs"])), f"{np.mean(s['ips']):.1f}", first,
               f"**{held}**" if bold else held, f"**{sp}**" if bold else sp, f"{s['gpu_min']:.1f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    res = load_results()
    summ = summarize(res)
    for theme in ("light", "dark"):
        bars(res, summ, MAIN, MAIN_LABEL, "time_to_target", theme, "Time to target on H100",
             "return ≥ 0.82 with 0 of 40 clips failed, held for 3 checkpoints · bar = mean, dots = runs")
        bars(res, summ, ALL, SHORT, "time_to_target_all", theme, "Every layout, time to target on H100",
             "bar = mean, dots = runs", left_in=4.1)
        curves(res, MAIN, MAIN_LABEL, "learning_curves", theme)
    print(table(res, summ, MAIN, MAIN_LABEL))
    print()
    print(table(res, summ, ALL, SHORT))
    for k in ALL:
        s = summ[k]
        print(f"{k:16s} held {np.mean(s['held']):.3f} first {np.mean(s['first']):.3f} it/s {np.mean(s['ips']):.2f} "
              f"speedup {s['speedup']:.3f} final return {np.mean(s['final_ret']):.3f}")


if __name__ == "__main__":
    main()
