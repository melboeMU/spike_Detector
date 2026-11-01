"""Mock test, part 2: detections -> design matrices -> group-lasso fit -> network.

Run AFTER mock_ds007118.py, which writes the detections this reads.

    python mock_ds007118_fit.py                          # sub-027 acq-B, both arms
    python mock_ds007118_fit.py sub-025 --arm channel --workers 3
    python mock_ds007118_fit.py sub-027 --arm channel --tag planted   # see check_mock_planted.py

Builds inputs in exactly the layout build_inputs.m produces (verified
bit-for-bit against Group 11's shipped files): per target node,
    X = [5 self lags, 5 lags per other node]   (oldest lag first)
    y = occupancy of the current bin (>=1 spike -> 1)
with the first 5 rows dropped. Bin = 5/256 s, as last year.

Then fits with mock_fit.R (identical model to FittingGroupPoisson.R) and reads
the network with the rule from Patient1_Clinical_Data_plots.m: an edge j->i
exists if the L2 norm of j's coefficient block in target i's selected model
exceeds 1e-8. Blocks are found by their group label (= node number), not by
column position.

Nodes with ZERO events are skipped as targets (a Poisson fit on an all-zero
response is degenerate) and left out as predictors: their history columns are
all zero, which grpreg drops anyway, so the fit is unchanged (verified on
sub-027: identical selections and edges) while the matrices get much smaller.

Targets are independent fits, so they are split across --workers R processes.
Each R process holds one design matrix at a time (~0.3-1 GB for 50-100 active
channels), so choose --workers from free memory, not from core count.
"""
import csv, json, os, subprocess, sys, time
import numpy as np
import scipy.io as sio

from mock_common import BIN_S, LAGS, HERE, RSCRIPT, Recording, Tee, git_sha, parser

TOL = 1e-8

ap = parser(__doc__)
ap.add_argument("--arm", choices=["channel", "lobe", "parcel", "both"], default="both",
                help="both = channel + lobe; parcel needs parcels_ds007118.py first")
ap.add_argument("--workers", type=int, default=3)
ap.add_argument("--tag", default=None,
                help="fit an alternative spike list instead: reads derivatives/<tag>/spikes.tsv and "
                     "writes every output there (montage and duration still come from detection)")
args = ap.parse_args()
R = Recording(args)
R.mkdirs()
if not os.path.exists(R.montage_txt):
    sys.exit("run mock_ds007118.py first: it writes the detections and montage this needs")
DUR_S = R.duration_s()
if args.tag:
    R.deriv = os.path.join(R.deriv, args.tag)
    R.spikes_tsv = os.path.join(R.deriv, "spikes.tsv")
out = Tee(os.path.join(R.results, f"modelling_{args.tag + '_' if args.tag else ''}{args.arm}.txt"))
out(f"recording: {R.label}   duration {DUR_S:.0f} s   workers {args.workers}")


def historical(vec, k):
    """MATLAB historicalVec.m: column c holds lag (k-c); oldest lag first."""
    n = vec.size
    H = np.zeros((n, k), dtype=np.uint8)
    for i in range(1, k + 1):
        H[i:, i - 1] = vec[: n - i]
    return H[:, ::-1]


# ------------------------------------------------------------ nodes per arm
det = list(csv.DictReader(open(R.spikes_tsv, encoding="utf-8"), delimiter="\t"))
lab = {r["name"]: r for r in csv.DictReader(open(R.channels_tsv, encoding="utf-8"), delimiter="\t")}
nb = int(np.ceil(DUR_S / BIN_S))

chan_times = {}
for r in det:
    chan_times.setdefault(r["channel"], []).append(float(r["time_s"]))

# the full montage, including zero-event channels
ALL_CH = [l.strip() for l in open(R.montage_txt, encoding="utf-8") if l.strip()]


def lobe_of(bn):
    a, b = bn.split("-")
    la, lb = lab[a]["lobe"], lab[b]["lobe"]
    return la if la == lb else f"{la}/{lb}"


def parcel_map():
    """channel -> Desikan-Killiany region, written by parcels_ds007118.py beside the montage."""
    path = os.path.join(os.path.dirname(R.montage_txt), "parcels.tsv")
    if not os.path.exists(path):
        sys.exit("run parcels_ds007118.py first: the parcel arm needs parcels.tsv")
    return {r["channel"]: r["parcel"] for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t")}


def arm_nodes(arm, all_channels):
    if arm == "channel":
        return {c: np.array(chan_times.get(c, [])) for c in all_channels}
    group_of = lobe_of if arm == "lobe" else parcel_map().get
    nodes = {}
    for c in all_channels:
        nodes.setdefault(group_of(c), []).extend(chan_times.get(c, []))
    return {k: np.array(v) for k, v in nodes.items()}


def occupancy(times):
    y = np.zeros(nb, dtype=np.uint8)
    if len(times):
        y[np.minimum(np.floor(times / BIN_S).astype(int), nb - 1)] = 1
    return y


def fit_folder(outdir, workers):
    """Run mock_fit.R on every input in outdir, split across worker processes."""
    procs = []
    for w in range(workers):
        log = open(os.path.join(outdir, f"fit_log_{w + 1}.txt"), "w", encoding="utf-8")
        procs.append((subprocess.Popen([RSCRIPT, os.path.join(HERE, "mock_fit.R"), outdir, "BIC",
                                        str(w + 1), str(workers)], stdout=log, stderr=subprocess.STDOUT),
                      log))
    for p, log in procs:
        p.wait(); log.close()
    lines = []
    for w in range(workers):
        lines += open(os.path.join(outdir, f"fit_log_{w + 1}.txt"), encoding="utf-8").read().splitlines()
    for ln in sorted(l for l in lines if l.startswith("glm_input_")):
        out(ln)
    errs = [l for l in lines if l.startswith("Error") or "Execution halted" in l]
    if errs or any(p.returncode for p, _ in procs):
        sys.exit("R fitting failed:\n" + "\n".join(errs or lines[-10:]))


def run(arm):
    nodes = arm_nodes(arm, ALL_CH)
    keys = list(nodes)
    Y = np.stack([occupancy(nodes[k]) for k in keys])
    n = len(keys)
    active = [j for j in range(n) if Y[j].sum() > 0]
    outdir = os.path.join(R.deriv, f"fit_{arm}")
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))

    out(f"\n=== {arm.upper()} arm: {n} nodes ({len(active)} with events), "
        f"{LAGS * len(active)} history columns, {nb:,} bins ===")
    for ti in active:
        others = [j for j in active if j != ti]
        X = np.hstack([historical(Y[ti], LAGS)] + [historical(Y[j], LAGS) for j in others])
        group = np.concatenate([np.full(LAGS, ti + 1)] + [np.full(LAGS, j + 1) for j in others])
        X, y = X[LAGS:], Y[ti][LAGS:]
        sio.savemat(os.path.join(outdir, f"glm_input_{arm}_node{ti + 1:03d}.mat"),
                    {"X": X, "y": y.astype(np.uint8), "group": group.astype(float),
                     "region": ti + 1, "numDelays_self": LAGS, "numDelays_cross": LAGS},
                    do_compression=True)
    targets = active
    out(f"targets fitted: {len(targets)} of {n}  (skipped {n - len(targets)} with zero events)")

    t0 = time.time()
    fit_folder(outdir, min(args.workers, len(targets)))
    fit_s = time.time() - t0
    out(f"R fitting: {fit_s:.0f}s")

    # ---------------------------------------------------------- read network
    edges, summary, failed = [], [], []
    for ti in targets:
        g = sio.loadmat(os.path.join(outdir, f"grpreg_{arm}_node{ti + 1:03d}.mat"))
        if "fit_failed" in g:
            failed.append((keys[ti], int(Y[ti].sum()), str(np.ravel(g["message"])[0])))
            continue
        best = int(np.ravel(g["best_idx"])[0]); nl = np.ravel(g["Lambda"]).size
        B = np.asarray(g["Beta"], dtype=float)[1:, best - 1]      # drop intercept row
        grp = np.ravel(g["group"]).astype(int)
        inc = []
        for j in active:
            if j == ti:
                continue
            w = float(np.linalg.norm(B[grp == j + 1]))
            if w > TOL:
                inc.append((keys[j], w)); edges.append((keys[j], keys[ti], w))
        self_w = float(np.linalg.norm(B[grp == ti + 1]))
        summary.append((keys[ti], int(Y[ti].sum()), best, nl, self_w, len(inc)))

    out(f"\n{'target':22s} {'events':>6s} {'idx':>7s} {'self':>6s} {'in-edges':>8s}")
    for k, e, b, nl, s, ni in summary:
        flag = "  intercept-only" if (s <= TOL and ni == 0) else ""
        out(f"{k:22s} {e:6d} {b:3d}/{nl:<3d} {s:6.2f} {ni:8d}{flag}")
    for k, e, msg in failed:
        out(f"{k:22s} {e:6d}  FIT FAILED: {msg}")
    io_ = sum(1 for *_, s, ni in summary if s <= TOL and ni == 0)
    out(f"\nedges selected: {len(edges)}   intercept-only targets: {io_} of {len(summary)}"
        f"   fits failed: {len(failed)}   skipped (zero events): {n - len(targets)}")
    for a, b, w in sorted(edges, key=lambda e: -e[2])[:25]:
        out(f"   {a:18s} -> {b:18s}  {w:.3f}")
    if len(edges) > 25:
        out(f"   ... {len(edges) - 25} more in network_{arm}.tsv")

    # save the network and per-target outcome so results can be read and plotted
    with open(os.path.join(R.deriv, f"network_{arm}.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["source", "target", "weight_L2"])
        for a, b, wt in sorted(edges, key=lambda e: -e[2]):
            w.writerow([a, b, f"{wt:.4f}"])
    with open(os.path.join(R.deriv, f"targets_{arm}.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["node", "events", "outcome", "best_idx", "self_weight", "incoming_edges"])
        for k, e, b, nl, s, ni in summary:
            w.writerow([k, e, "intercept-only" if (s <= TOL and ni == 0) else "fitted",
                        b, f"{s:.3f}", ni])
        for k, e, _ in failed:
            w.writerow([k, e, "fit failed", "", "", 0])
        for ti, k in enumerate(keys):
            if Y[ti].sum() == 0:
                w.writerow([k, 0, "no events", "", "", 0])
    json.dump({"recording": R.stem, "arm": arm, "tag": args.tag, "git_commit": git_sha(), "bin_s": BIN_S,
               "lags_self": LAGS, "lags_cross": LAGS, "criterion": "BIC", "penalty": "grLasso",
               "family": "poisson", "nlambda": 100, "edge_tol": TOL, "n_nodes": n,
               "n_active": len(active), "predictors": "active nodes only (zero-event nodes dropped)",
               "workers": args.workers, "fit_seconds": round(fit_s, 1), "n_edges": len(edges)},
              open(os.path.join(R.deriv, f"fit_config_{arm}.json"), "w", encoding="utf-8"), indent=1)
    return summary, edges


for arm in (["channel", "lobe"] if args.arm == "both" else [args.arm]):
    run(arm)
