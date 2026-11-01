"""Merge co-detections: count one discharge once, not once per channel (P-29, S15).

Run AFTER mock_ds007118.py. Then fit the merged list with the UNCHANGED fitting
code, and compare:

    python merge_codetections.py sub-018
    python mock_ds007118_fit.py sub-018 --arm channel --tag merged --workers 2
    python plot_merged_comparison.py sub-027 sub-025 sub-018 sub-011

Rule: walk the spikes in time order. The earliest not-yet-used spike opens an
event; every spike on ANY other channel within WINDOW ms of it (default 20 ms,
about one 19.5-ms model bin) belongs to the same event and is dropped. Only the
opening spike is kept, so the event is credited to the channel that peaked
first. The window is anchored on that first spike (no chaining), so a spike
25 ms later starts a new event and can still support a lead-lag edge.

Why 20 ms: below one bin the model cannot tell propagation from the same spike
reaching two channels (P-29). A second spike on the SAME channel inside the
window is dropped too: despite the 40-ms refractory period the detector
occasionally reports both phases of one biphasic spike (P-34).

--near (the lenient version, Q32): only merge spikes on channels no farther
apart than this patient's two most distant NEIGHBOURING bipolar channels (pairs
sharing a contact, midpoint to midpoint). A spike farther away than that is
not the same discharge seen next door, so it stays as its own event.

    python merge_codetections.py sub-018 --near      # -> derivatives/merged_near/

Writes derivatives/<tag>/spikes.tsv (same columns, plus n_channels = how many
channels saw that event), derivatives/<tag>/merge_config.json and
results/<sub>/merge_check[_near].txt, with tag = merged or merged_near.
"""
import csv, json, os
import numpy as np

from mock_common import Recording, Tee, git_sha, parser

ap = parser(__doc__)
ap.add_argument("--window-ms", type=float, default=20.0)
ap.add_argument("--near", action="store_true", help="merge only channels within the max neighbour distance")
args = ap.parse_args()
R = Recording(args)
W = args.window_ms / 1000
TAG = "merged_near" if args.near else "merged"

outdir = os.path.join(R.deriv, TAG)
os.makedirs(outdir, exist_ok=True)
out = Tee(os.path.join(R.results, f"merge_check{'_near' if args.near else ''}.txt"))

rows = list(csv.DictReader(open(R.spikes_tsv, encoding="utf-8"), delimiter="\t"))
rows.sort(key=lambda r: float(r["time_s"]))
xyz = {r["name"]: np.array([float(r["x"]), float(r["y"]), float(r["z"])])
       for r in csv.DictReader(open(R.electrodes_tsv, encoding="utf-8"), delimiter="\t")
       if r["x"] not in ("n/a", "")}
montage = [l.strip() for l in open(R.montage_txt, encoding="utf-8") if l.strip()]


def mid(ch):
    a, b = ch.split("-")
    return (xyz[a] + xyz[b]) / 2 if a in xyz and b in xyz else None


def dist(c1, c2):
    p, q = mid(c1), mid(c2)
    return None if p is None or q is None else float(np.linalg.norm(p - q))


if args.near:
    nb = [dist(a, b) for k, a in enumerate(montage) for b in montage[k + 1:] if set(a.split("-")) & set(b.split("-"))]
    MAX_MM = max(d for d in nb if d is not None)
else:
    MAX_MM = None

# walk in time order; the earliest unused spike opens an event and absorbs every
# unused spike within the window (and, with --near, within MAX_MM of its channel)
used = [False] * len(rows)
kept, dists, same_ch = [], [], 0
for i, first in enumerate(rows):
    if used[i]:
        continue
    t0, c0 = float(first["time_s"]), first["channel"]
    chans = {c0}
    j = i + 1
    while j < len(rows) and float(rows[j]["time_s"]) - t0 <= W:
        r = rows[j]
        d = dist(c0, r["channel"]) if r["channel"] != c0 else 0.0
        if not used[j] and (MAX_MM is None or (d is not None and d <= MAX_MM)):
            used[j] = True
            chans.add(r["channel"])
            same_ch += r["channel"] == c0
            if d is not None and r["channel"] != c0:
                dists.append(d)
        j += 1
    kept.append({**first, "n_channels": len(chans)})

cols = list(rows[0].keys()) + ["n_channels"]
with open(os.path.join(outdir, "spikes.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
    w.writeheader()
    w.writerows(kept)
json.dump({"recording": R.stem, "source": os.path.relpath(R.spikes_tsv, outdir), "window_ms": args.window_ms,
           "rule": "earliest spike in each window kept; window anchored on it; "
                   + (f"only channels within {MAX_MM:.1f} mm (max neighbour distance)" if args.near else "any channel"),
           "max_mm": MAX_MM, "n_in": len(rows), "n_out": len(kept), "git_commit": git_sha()},
          open(os.path.join(outdir, "merge_config.json"), "w", encoding="utf-8"), indent=1)

n_multi = sum(k["n_channels"] > 1 for k in kept)
before = {}
for r in rows:
    before[r["channel"]] = before.get(r["channel"], 0) + 1
after = {}
for k in kept:
    after[k["channel"]] = after.get(k["channel"], 0) + 1
out(f"recording: {R.label}   window {args.window_ms:.0f} ms   "
    + (f"channels within {MAX_MM:.1f} mm only (max neighbour distance)" if args.near else "any channels"))
out(f"spikes: {len(rows)} -> {len(kept)} events ({len(rows) - len(kept)} co-detections removed, "
    f"{100 * (len(rows) - len(kept)) / len(rows):.0f}%)")
out(f"events seen on more than one channel: {n_multi} of {len(kept)}")
out(f"same-channel duplicates inside the window (both phases of one spike, P-34): {same_ch}")
out(f"active channels: {len(before)} -> {len(after)}")
if dists:
    d = np.array(dists)
    out(f"distance from the kept channel to each dropped one (fsaverage mm): median {np.median(d):.0f}, "
        f"<= 25 mm: {np.mean(d <= 25) * 100:.0f}%, > 50 mm: {np.mean(d > 50) * 100:.0f}%")
out("\nbusiest channels, spikes before -> after:")
for c in sorted(before, key=lambda c: -before[c])[:10]:
    out(f"  {c:10s} {before[c]:5d} -> {after.get(c, 0):5d}")
