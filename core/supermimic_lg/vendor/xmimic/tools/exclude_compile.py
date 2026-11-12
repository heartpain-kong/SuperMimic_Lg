"""Remove BlinkMimic's (and InstantTrack's) one-time compilation and CUDA-graph capture from its checkpoint wall-clock.

BlinkMimic compiles (torch.compile) and captures its CUDA graphs inside the training loop, so its checkpoints' training
time includes them; XMimic compiles ahead of time and starts its clock afterwards. From the run's train_log.csv (one row
per 50 iterations) this estimates the excess time of the two intervals that contain one-time work:
  - the first interval (compilation and capture of the collection, world-model and policy graphs), against the median
    interval of iterations 100-950 (before probe sweeps start);
  - the interval in which the first probe sweep starts (capture of the sweep graph), against the median of the later
    intervals with a sweep running.
Every checkpoint after such an interval gets its excess subtracted.

    python tools/exclude_compile.py <train_log.csv> <eval.csv> <out.csv>

results/blinkmimic/seed*.csv and results/instanttrack/seed*.csv were made this way from the files in their raw/ folders.
"""

import argparse
import csv

ap = argparse.ArgumentParser()
ap.add_argument("train_log")
ap.add_argument("eval_csv")
ap.add_argument("out")
ap.add_argument("--sweep_start", type=int, default=1000)
a = ap.parse_args()
rows = list(csv.DictReader(open(a.train_log)))
its = [int(r["iteration"]) for r in rows]
t = [float(r["wall_h"]) * 3600 for r in rows]
d = [t[0]] + [t[i] - t[i - 1] for i in range(1, len(t))]
med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
steady = med([d[i] for i in range(1, len(d)) if 100 <= its[i] <= 950])
events = [(its[0], d[0] - steady)]
k = next((i for i in range(len(its)) if its[i] > a.sweep_start), None)
if k is not None:
    later = [d[i] for i in range(k + 1, len(d)) if d[i] > 1.3 * steady]  # later intervals with a sweep running
    if later:
        events.append((its[k], max(0.0, d[k] - med(later))))
ev = list(csv.DictReader(open(a.eval_csv)))
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ev[0].keys()), lineterminator="\n")
    w.writeheader()
    for r in ev:
        excess = sum(e for it_e, e in events if it_e <= int(r["iteration"]))
        r = dict(r)
        r["train_wall_h"] = f"{float(r['train_wall_h']) - excess / 3600:.6f}"
        w.writerow(r)
print(f"steady {steady:.2f} s per 50 iterations; one-time excess removed: " + ", ".join(f"{e:.1f} s at it {i}" for i, e in events) + f" -> {a.out}")
