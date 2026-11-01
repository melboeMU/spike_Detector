"""Shared paths and arguments for the ds007118 mock scripts.

Every mock script takes the same recording arguments and defaults to the first
mock recording, so the original commands still work:

    python mock_ds007118.py                     # sub-027, acq-B, run-01
    python mock_ds007118.py sub-025             # acq-A, run-01
    python mock_ds007118.py sub-031 --run 02

Per recording, everything derived goes in
    Current work/data/ds007118/<sub>/derivatives_mock/acq-<A>_run-<01>/
and the figure and text outputs in Current work/results/<sub>/.
"""
import argparse, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data", "ds007118"))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "..", "results"))
RSCRIPT = r"C:/Program Files/R/R-4.6.1/bin/Rscript.exe"

BIN_S = 5 / 256          # Group 11's clinical bin, ~19.5 ms
LAGS = 5                 # self and cross lags, as last year


def parser(doc=None):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sub", nargs="?", default="sub-027")
    ap.add_argument("--acq", default=None, help="default: B for sub-027, otherwise A")
    ap.add_argument("--run", default="01")
    return ap


class Recording:
    def __init__(self, args):
        acq = args.acq or ("B" if args.sub == "sub-027" else "A")
        self.sub = args.sub
        self.stem = f"{args.sub}_ses-01_task-sleep_acq-{acq}_run-{args.run}"
        self.subdir = os.path.join(DATA, args.sub)
        # joined runs (join_runs.py) have no sidecars of their own; every run of a
        # patient has identical channels, labels and coordinates, so use run-01's
        side = os.path.join(self.subdir, self.stem.replace("run-joined", "run-01"))
        self.edf = side + "_ieeg.edf"
        self.channels_tsv = side + "_channels.tsv"
        self.electrodes_tsv = side + "_electrodes.tsv"
        self.ieeg_json = side + "_ieeg.json"
        self.deriv = os.path.join(self.subdir, "derivatives_mock", f"acq-{acq}_run-{args.run}")
        self.results = os.path.join(RESULTS, args.sub)
        self.spikes_tsv = os.path.join(self.deriv, "spikes.tsv")
        self.montage_txt = os.path.join(self.deriv, "bipolar_channels.txt")
        self.detect_json = os.path.join(self.deriv, "detect_config.json")
        self.label = f"{args.sub} acq-{acq} run-{args.run}"

    def mkdirs(self):
        os.makedirs(self.deriv, exist_ok=True)
        os.makedirs(self.results, exist_ok=True)

    def duration_s(self):
        """Recording length, written by mock_ds007118.py at detection time."""
        return json.load(open(self.detect_json, encoding="utf-8"))["duration_s"]


def git_sha():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class Tee:
    """Print to the terminal and to a results file at once."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a, **k):
        print(*a, **k)
        print(*a, **k, file=self.f)
        self.f.flush()
