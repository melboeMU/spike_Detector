"""Join one patient's runs into a single spike list, and check the join (plan.md step 3, S12).

Run AFTER mock_ds007118.py for every run of the patient (and parcels_ds007118.py
on run-01 for the parcel version).

    python join_runs.py sub-017

Why spike lists, not signals: each run is detected on its own and the spike
times are shifted by the total length of the runs before it. Gluing the raw
recordings end to end would put a step in the signal at every join, which the
detector's filters can turn into false spikes.

Writes derivatives_mock/acq-<A>_run-joined/ (spikes.tsv with `run` and
`time_in_run_s` columns, bipolar_channels.txt, parcels.tsv, detect_config.json
listing each run's offset and the join times) and results/<sub>/join_check.txt:
  1. per run: length, candidates, spikes, rate;
  2. montage identical across runs (required to join channel for channel);
  3. run order and gaps, from the EDF header start times;
  4. consistency: do the same channels dominate in every run (Spearman of
     per-channel rate between runs)?
  5. detector vs clinical labels, per run and joined (P-33);
  6. noisy minutes: minutes with many times the median spike count, spread
     over many channels at once (S13);
  7. spikes in the first/last second of a run (filter edge effects);
  8. gain: events per model parameter on the densest channel, one run vs joined.

Model note: the fitting script does not yet treat joins as breaks in the
history; until it does, the last 100 ms of one run can "predict" the first
100 ms of the next (negligible: a handful of bins per join).
"""
import csv, glob, json, os, re, shutil
from collections import Counter
import numpy as np
from scipy.stats import spearmanr

from mock_common import Recording, Tee, git_sha, parser
import mne
mne.set_log_level("ERROR")

ap = parser(__doc__)
ap.add_argument("--exclude", nargs="*", default=[], metavar="RUN",
                help="runs to leave out, e.g. --exclude 02 (sub-031 run-02 duplicates run-01)")
args = ap.parse_args()
acq = args.acq or "A"
R1 = Recording(type(args)(sub=args.sub, acq=acq, run="01"))
base = os.path.dirname(R1.deriv)
runs = sorted(re.search(r"run-(\d+)$", d)[1] for d in glob.glob(os.path.join(base, f"acq-{acq}_run-*"))
              if re.search(r"run-\d+$", d) and os.path.exists(os.path.join(d, "spikes.tsv")))
runs = [r for r in runs if r not in args.exclude]
J = Recording(type(args)(sub=args.sub, acq=acq, run="joined"))
os.makedirs(J.deriv, exist_ok=True); os.makedirs(J.results, exist_ok=True)
out = Tee(os.path.join(J.results, "join_check.txt"))
out(f"{args.sub} acq-{acq}: runs {runs}" + (f"   (excluded: {args.exclude})" if args.exclude else ""))

lab = {r["name"]: r for r in csv.DictReader(open(R1.channels_tsv, encoding="utf-8"), delimiter="\t")}


def label(ch):
    s = {lab[c]["status_description"] for c in ch.split("-")}
    return "SOZ" if "seizure onset zone" in s else "spiking" if "spiking" in s else "normal"


info = []
for r in runs:
    R = Recording(type(args)(sub=args.sub, acq=acq, run=r))
    cfg = json.load(open(R.detect_json, encoding="utf-8"))
    mont = [l.strip() for l in open(R.montage_txt, encoding="utf-8") if l.strip()]
    sp = list(csv.DictReader(open(R.spikes_tsv, encoding="utf-8"), delimiter="\t"))
    start = mne.io.read_raw_edf(R.edf, preload=False).info["meas_date"]
    info.append(dict(run=r, R=R, cfg=cfg, mont=mont, spikes=sp, dur=cfg["duration_s"], start=start))

# 1. per run
out("\n1. runs")
for x in info:
    out(f"  run-{x['run']}: {x['dur'] / 60:5.1f} min, {x['cfg']['n_candidates']:6d} candidates, "
        f"{len(x['spikes']):5d} spikes, {len(x['spikes']) / (x['dur'] / 60):5.1f} spikes/min")

# 2. montage
mont = info[0]["mont"]
same = all(x["mont"] == mont for x in info)
out(f"\n2. montage identical across runs: {same} ({len(mont)} bipolar channels)")
if not same:
    common = [c for c in mont if all(c in x["mont"] for x in info)]
    out(f"   ! using the {len(common)} channels present in every run")
    mont = common

# 3. order and gaps
out("\n3. order and gaps (EDF header start times; may be shifted by de-identification)")
known = all(x["start"] is not None for x in info)
order_ok = known and all(info[k]["start"] <= info[k + 1]["start"] for k in range(len(info) - 1))
for k, x in enumerate(info):
    gap = "" if k == 0 or not known else \
        f"   gap after previous run: {(x['start'] - info[k - 1]['start']).total_seconds() - info[k - 1]['dur']:+.0f} s"
    out(f"  run-{x['run']}: starts {x['start']}{gap}")
out(f"  runs in time order: {order_ok}")

# join
offset, rows, joins = 0.0, [], []
for x in info:
    x["offset"] = offset
    for s in x["spikes"]:
        if s["channel"] in mont:
            rows.append({**s, "time_s": f"{float(s['time_s']) + offset:.4f}", "run": x["run"],
                         "time_in_run_s": s["time_s"]})
    offset += x["dur"]
    joins.append(offset)
joins = joins[:-1]
rows.sort(key=lambda s: float(s["time_s"]))
cols = list(info[0]["spikes"][0].keys()) + ["run", "time_in_run_s"]
with open(J.spikes_tsv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(rows)
open(J.montage_txt, "w", encoding="utf-8").write("\n".join(mont) + "\n")
p1 = os.path.join(info[0]["R"].deriv, "parcels.tsv")
if os.path.exists(p1):
    shutil.copy2(p1, os.path.join(J.deriv, "parcels.tsv"))
cfg = {k: v for k, v in info[0]["cfg"].items() if k not in ("recording", "edf", "duration_s", "n_samples",
                                                               "n_candidates", "n_accepted", "git_commit")}
json.dump({"recording": f"{args.sub}_acq-{acq}_runs-{'+'.join(runs)}_joined", "duration_s": offset,
           "runs": [{"run": x["run"], "edf": x["cfg"]["edf"], "duration_s": x["dur"], "offset_s": x["offset"],
                     "start": str(x["start"]), "n_candidates": x["cfg"]["n_candidates"], "n_accepted": len(x["spikes"])}
                    for x in info],
           "joins_s": joins, "excluded_runs": args.exclude, "montage_identical": same,
           "header_times_note": "EDF start times look overwritten by de-identification (overlapping runs); gaps unknown",
           "join_rule": "each run detected separately; spike times shifted by the summed length of earlier runs",
           **cfg, "git_commit": git_sha()},
          open(J.detect_json, "w", encoding="utf-8"), indent=1)

# 4. consistency between runs
out("\n4. do the same channels dominate in every run? (Spearman of per-channel spike rate)")
rates = {x["run"]: np.array([sum(s["channel"] == c for s in x["spikes"]) / (x["dur"] / 60) for c in mont]) for x in info}
for a in range(len(runs)):
    for b in range(a + 1, len(runs)):
        ra, rb = rates[runs[a]], rates[runs[b]]
        rho = spearmanr(ra, rb)[0] if ra.any() and rb.any() else float("nan")
        out(f"  run-{runs[a]} vs run-{runs[b]}: rho = {rho:.2f}")
for x in info:
    top = [mont[i] for i in np.argsort(-rates[x["run"]])[:3]]
    out(f"  run-{x['run']} busiest: {', '.join(f'{c} ({label(c)})' for c in top)}")

# 5. detector vs labels
out("\n5. detector vs clinical labels: median spikes/min per channel label")
labs = [label(c) for c in mont]
for name, rate in list(rates.items()) + [("joined", np.array([sum(s['channel'] == c for s in rows) / (offset / 60) for c in mont]))]:
    med = {L: np.median([rate[i] for i in range(len(mont)) if labs[i] == L]) for L in ("SOZ", "spiking", "normal") if L in labs}
    top = [labs[i] for i in np.argsort(-rate)[:5]]
    out(f"  {'run-' + name if name != 'joined' else 'joined':8s} " + "  ".join(f"{L} {v:.2f}" for L, v in med.items())
        + f"   top-5 labels: {dict(Counter(top))}")
if "SOZ" not in labs:
    out("  ! no seizure-onset contacts labelled (no seizure captured); check against 'spiking' instead")

# 6. bursts: seconds in which a large share of all channels spike at once
out("\n6. burst seconds (> 25% of active channels spike within the same second) - artifact or seizure, not spikes")
t = np.array([float(s["time_s"]) for s in rows]); chs = np.array([s["channel"] for s in rows])
active = len(set(chs)); sec = t.astype(int)
nch = {k: len(set(chs[sec == k])) for k in np.unique(sec)}
burst = sorted(k for k, n in nch.items() if n > 0.25 * active)
in_b = np.isin(sec, burst)
out(f"  {len(burst)} burst seconds holding {in_b.sum()} of {len(t)} spikes ({in_b.mean():.0%})")
for k in burst[:15]:
    run = next(x["run"] for x in reversed(info) if k >= x["offset"])
    out(f"    {k:6d} s (run-{run}, {k - next(x['offset'] for x in info if x['run'] == run):.0f} s into it): "
        f"{(sec == k).sum():4d} spikes on {nch[k]} channels")
if len(burst) > 15:
    out(f"    ... {len(burst) - 15} more")

# 7. edges
edge = sum(1 for x in info for s in x["spikes"] if float(s["time_s"]) < 1 or float(s["time_s"]) > x["dur"] - 1)
exp = len(rows) * 2 * len(info) / offset
out(f"\n7. spikes within 1 s of a run's start or end: {edge} (expected by chance ~{exp:.1f})")

# 8. gain
params = lambda n: 1 + 5 + 5 * (n - 1)
n_act = len(set(chs))
best1 = max((max(Counter(s["channel"] for s in x["spikes"]).values(), default=0), x["run"]) for x in info)
dens_j = max(Counter(chs).values())
out(f"\n8. densest channel, events per parameter ({params(n_act)} parameters with {n_act} active channels): "
    f"best single run {best1[0] / params(n_act):.2f} (run-{best1[1]}) -> joined {dens_j / params(n_act):.2f}")
out(f"\njoined: {offset / 60:.1f} min, {len(rows)} spikes on {n_act} of {len(mont)} channels; joins at {[round(j) for j in joins]} s")
