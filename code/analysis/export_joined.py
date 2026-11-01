"""Export joined-run spike lists by channel and by parcel (after join_runs.py).

    python export_joined.py sub-015 sub-017 ...

Writes Current work/joined_runs/:
  by_channel/<sub>_channel_spikes.tsv   one row per spike, times on the joined clock
  by_channel/<sub>_bipolar_channels.txt every channel, including silent ones
  by_parcel/<sub>_region_spikes.tsv     the same spikes with their Desikan-Killiany region
  by_parcel/<sub>_channel_regions.tsv   channel -> region, including silent channels
  runs.tsv                              each run's length and offset on the joined clock
"""
import argparse, csv, json, os

from mock_common import RESULTS, Recording

ap = argparse.ArgumentParser()
ap.add_argument("subs", nargs="+")
args = ap.parse_args()
OUT = os.path.normpath(os.path.join(RESULTS, "..", "joined_runs"))
for d in ("by_channel", "by_parcel"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

table = []
for sub in args.subs:
    J = Recording(argparse.Namespace(sub=sub, acq="A", run="joined"))
    cfg = json.load(open(J.detect_json, encoding="utf-8"))
    rows = list(csv.DictReader(open(J.spikes_tsv, encoding="utf-8"), delimiter="\t"))
    mont = [l.strip() for l in open(J.montage_txt, encoding="utf-8") if l.strip()]
    keep = ["channel", "time_s", "run", "time_in_run_s", "amplitude_uv", "det_z", "scale_ms", "halfwidth_ms"]
    with open(os.path.join(OUT, "by_channel", f"{sub}_channel_spikes.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keep, delimiter="\t", extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    open(os.path.join(OUT, "by_channel", f"{sub}_bipolar_channels.txt"), "w", encoding="utf-8").write("\n".join(mont) + "\n")
    par = {r["channel"]: r["parcel"] for r in csv.DictReader(open(os.path.join(J.deriv, "parcels.tsv"), encoding="utf-8"), delimiter="\t")}
    with open(os.path.join(OUT, "by_parcel", f"{sub}_region_spikes.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["region", "time_s", "run", "time_in_run_s", "channel"])
        for r in rows:
            w.writerow([par[r["channel"]], r["time_s"], r["run"], r["time_in_run_s"], r["channel"]])
    with open(os.path.join(OUT, "by_parcel", f"{sub}_channel_regions.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["channel", "region"])
        for c in mont:
            w.writerow([c, par[c]])
    for x in cfg["runs"]:
        table.append([sub, x["run"], f"{x['duration_s']:.0f}", f"{x['offset_s']:.0f}", x["n_accepted"], f"{cfg['duration_s']:.0f}"])
    print(f"{sub}: {len(rows)} spikes, {len(mont)} channels, {len(set(par.values()))} regions, {cfg['duration_s'] / 60:.0f} min")

with open(os.path.join(OUT, "runs.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["subject", "run", "run_duration_s", "offset_s", "spikes_in_run", "joined_duration_s"])
    w.writerows(table)
print("wrote", OUT)
