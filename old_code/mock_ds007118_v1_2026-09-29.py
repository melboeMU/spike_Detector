"""Mock test on one ds007118 recording: detection -> event density.

Subject sub-027, acquisition B: 36 contacts, 34 min, 1000 Hz, ILAE 1 with the
SOZ completely resected. Chosen because it is small (147 MB) and carries
per-channel labels (SOZ / spiking / normal / artifact, resected / preserved,
lobe) to check the detector against.

    python mock_ds007118.py

Stages
  1. Load the EDF, strip the "-AV" suffix so names join channels.tsv
     (a naive join on the raw EDF names matches ZERO channels).
  2. Build a bipolar montage from PHYSICALLY adjacent contacts, using
     electrodes.tsv, so no pair spans a grid-row break.
  3. Run spikedet.py.
  4. Sanity check: do detections concentrate in clinician-labelled
     SOZ / spiking channels?
  5. Event density per node for the channel arm and the lobe (parcel) arm,
     in the same events-per-parameter terms as problems.md P-09.

Read-only on the data. Writes detections to data/.../derivatives_mock/.
"""
import csv, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SPIKE anirudh"))
from spikedet import SpikeParams, detect_spikes          # noqa: E402

import mne                                               # noqa: E402
mne.set_log_level("ERROR")

SUB = os.path.normpath(os.path.join(HERE, "..", "..", "data", "ds007118", "sub-027"))
STEM = "sub-027_ses-01_task-sleep_acq-B_run-01"
EDF = os.path.join(SUB, STEM + "_ieeg.edf")
CHT = os.path.join(SUB, STEM + "_channels.tsv")
ELT = os.path.join(SUB, STEM + "_electrodes.tsv")
OUT = os.path.join(SUB, "derivatives_mock")
BIN_S = 5 / 256          # Group 11's clinical bin, ~19.5 ms
LAGS = 5                 # self and cross lags, as last year

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. load
raw = mne.io.read_raw_edf(EDF, preload=False)
fs = raw.info["sfreq"]
names = [re.sub(r"-AV$", "", c) for c in raw.ch_names]
lab = {r["name"]: r for r in csv.DictReader(open(CHT, encoding="utf-8"), delimiter="\t")}
assert names == list(lab), "EDF names (minus -AV) must match channels.tsv exactly"
xyz = {r["name"]: np.array([float(r["x"]), float(r["y"]), float(r["z"])])
       for r in csv.DictReader(open(ELT, encoding="utf-8"), delimiter="\t")}

t0 = time.time()
X = raw.get_data().astype(np.float32) * 1e6        # volts -> microvolts
dur = X.shape[1] / fs
print(f"loaded {X.shape[0]} ch x {dur/60:.1f} min at {fs:.0f} Hz  ({time.time()-t0:.1f}s)")

# ------------------------------------------------- 2. bipolar, adjacent only
def num(n):
    m = re.match(r"([A-Za-z]+)(\d+)$", n)
    return (m.group(1), int(m.group(2))) if m else (n, None)

nocoord = [n for n in names if n not in xyz]    # e.g. sub-027 B16: artifact, lobe n/a
cand = []
for i in range(len(names) - 1):
    (a_s, a_n), (b_s, b_n) = num(names[i]), num(names[i + 1])
    if names[i] in nocoord or names[i + 1] in nocoord:
        continue                                 # adjacency cannot be verified
    if a_s == b_s and a_n is not None and b_n == a_n + 1:
        cand.append((i, i + 1, float(np.linalg.norm(xyz[names[i]] - xyz[names[i + 1]]))))
spacing = np.median([d for *_, d in cand])
pairs = [(i, j, d) for i, j, d in cand if d <= 1.5 * spacing]
skipped = [(names[i], names[j], round(d, 1)) for i, j, d in cand if d > 1.5 * spacing]

# drop pairs touching an artifact-labelled contact
bad = {n for n, r in lab.items() if r["status_description"] == "artifact"}
pairs = [(i, j, d) for i, j, d in pairs if names[i] not in bad and names[j] not in bad]

B = np.stack([X[i] - X[j] for i, j, _ in pairs])
bnames = [f"{names[i]}-{names[j]}" for i, j, _ in pairs]
print(f"bipolar: {len(pairs)} pairs  (median contact spacing {spacing:.1f} mm; "
      f"{len(skipped)} non-adjacent jumps skipped, {len(bad)} artifact contacts excluded, "
      f"no coordinates: {nocoord})")
if skipped:
    print(f"  skipped jumps: {skipped[:6]}{' ...' if len(skipped) > 6 else ''}")

# -------------------------------------------------------------- 3. detect
p = SpikeParams(fs=fs, notch_freqs=(60.0, 120.0, 180.0))   # US mains, notch not applied upstream
t0 = time.time()
ev = detect_spikes(B, p, ch_names=bnames, return_rejected=True)
acc = ev["accepted"].astype(bool)
print(f"\ndetection: {acc.size} candidates, {acc.sum()} accepted  ({time.time()-t0:.1f}s)")
from collections import Counter
print("  rejected by:", dict(Counter(ev["reject_reason"][~acc])))

# the full montage, including channels with zero detections (still predictors)
with open(os.path.join(OUT, STEM + "_bipolar_channels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(bnames) + "\n")

with open(os.path.join(OUT, STEM + "_spikes.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["channel", "time_s", "amplitude_uv", "det_z", "scale_ms", "halfwidth_ms"])
    for k in np.flatnonzero(acc):
        w.writerow([ev["channel_name"][k], f"{ev['time_s'][k]:.4f}", f"{ev['amplitude_uv'][k]:.1f}",
                    f"{ev['det_z'][k]:.2f}", f"{ev['scale_ms'][k]:.0f}", f"{ev['halfwidth_ms'][k]:.1f}"])

# ------------------------------------------------ 4. sanity vs clinical labels
def pair_label(bn):
    a, b = bn.split("-")
    s = {lab[a]["status_description"], lab[b]["status_description"]}
    return "SOZ" if "seizure onset zone" in s else "spiking" if "spiking" in s else "normal"

counts = Counter(ev["channel_name"][acc])
rate = {bn: counts.get(bn, 0) / (dur / 60) for bn in bnames}
print("\nspike rate per bipolar channel, by clinical label of its contacts (spikes/min):")
for g in ("SOZ", "spiking", "normal"):
    r = [rate[b] for b in bnames if pair_label(b) == g]
    if r:
        print(f"  {g:8s} n={len(r):2d}  median {np.median(r):6.2f}  mean {np.mean(r):6.2f}  max {max(r):6.2f}")
top = sorted(bnames, key=lambda b: -rate[b])[:8]
print("  top 8:", ", ".join(f"{b}({rate[b]:.1f},{pair_label(b)})" for b in top))

# ------------------------------------------------------- 5. event density
nbins = int(np.ceil(dur / BIN_S))
def occupied(times):
    return len(np.unique(np.floor(np.asarray(times) / BIN_S).astype(int)))

times_by_ch = {b: ev["time_s"][acc & (ev["channel_name"] == b)] for b in bnames}

def report(nodes, title):
    n = len(nodes); params = LAGS * n
    occ = {k: occupied(v) if len(v) else 0 for k, v in nodes.items()}
    epp = {k: o / params for k, o in occ.items()}
    vals = np.array(list(epp.values()))
    print(f"\n{title}: {n} nodes, {params} history parameters per target model, {nbins:,} bins")
    print(f"  events/parameter: median {np.median(vals):.1f}, min {vals.min():.1f}, max {vals.max():.1f}")
    print(f"  nodes below 10 events/parameter: {int((vals < 10).sum())} of {n}   (P-09 floor is 10-20)")
    return epp

report(times_by_ch, "CHANNEL arm")

lobe_of = {}
for b in bnames:
    a, c = b.split("-")
    la, lc = lab[a]["lobe"], lab[c]["lobe"]
    lobe_of[b] = la if la == lc else f"{la}/{lc}"
lobes = {}
for b, t in times_by_ch.items():
    lobes.setdefault(lobe_of[b], []).extend(t)
lob_epp = report({k: np.array(v) for k, v in lobes.items()}, "LOBE (parcel) arm")
for k, v in sorted(lob_epp.items(), key=lambda kv: -kv[1]):
    print(f"    {k:22s} {v:8.1f} events/param   ({len(lobes[k])} spikes)")

print(f"\ndetections written to {os.path.relpath(OUT, HERE)}")
