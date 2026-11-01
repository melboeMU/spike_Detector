"""The same spikes modelled at three node scales: channels, atlas parcels, lobes.

Run AFTER mock_ds007118.py, parcels_ds007118.py and
mock_ds007118_fit.py --arm channel / parcel / lobe for each recording.

    python plot_node_scales.py sub-027 sub-025 sub-018

One row per patient, three columns from fine to coarse. Group nodes (parcels,
lobes) sit at the centre of their channels, sized by spike count, with a red
edge when they hold a seizure-onset channel. Same drawing as
channels_vs_lobes.png, with the parcel column added between.

Writes Current work/results/figures/channels_parcels_lobes.png
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mock_common import RESULTS, Recording

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CONTACT, SOZ_RING, EDGE = "#52514e", "#e34948", "#2a78d6"
FILL = (0.165, 0.471, 0.839, 0.22)
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

ap = argparse.ArgumentParser()
ap.add_argument("subs", nargs="+")
args = ap.parse_args()


class P:
    def __init__(self, sub):
        R = Recording(argparse.Namespace(sub=sub, acq=None, run="01"))
        self.sub, self.R = sub, R
        self.lab = {r["name"]: r for r in csv.DictReader(open(R.channels_tsv, encoding="utf-8"), delimiter="\t")}
        xyz3 = {r["name"]: np.array([float(r["x"]), float(r["y"]), float(r["z"])])
                for r in csv.DictReader(open(R.electrodes_tsv, encoding="utf-8"), delimiter="\t")
                if r["x"] not in ("n/a", "")}
        ax = np.sort(np.argsort([-np.ptp([v[k] for v in xyz3.values()]) for k in range(3)])[:2])
        self.xyz = {n: v[ax] for n, v in xyz3.items()}
        self.montage = [l.strip() for l in open(R.montage_txt, encoding="utf-8") if l.strip()]
        self.n = {c: 0 for c in self.montage}
        for r in csv.DictReader(open(R.spikes_tsv, encoding="utf-8"), delimiter="\t"):
            self.n[r["channel"]] += 1
        self.parcel = {r["channel"]: r["parcel"] for r in csv.DictReader(
            open(os.path.join(R.deriv, "parcels.tsv"), encoding="utf-8"), delimiter="\t")}

    def edges(self, arm):
        return [(e["source"], e["target"]) for e in csv.DictReader(
            open(os.path.join(self.R.deriv, f"network_{arm}.tsv"), encoding="utf-8"), delimiter="\t")]

    def soz(self, ch):
        return any(self.lab[c]["status_description"] == "seizure onset zone" for c in ch.split("-"))

    def mid(self, ch):
        a, b = ch.split("-")
        return (self.xyz[a] + self.xyz[b]) / 2

    def lobe_of(self, ch):
        a, b = ch.split("-")
        la, lb = self.lab[a]["lobe"], self.lab[b]["lobe"]
        return la if la == lb else f"{la}/{lb}"

    def groups(self, arm):
        g = {}
        for c in self.montage:
            g.setdefault(self.lobe_of(c) if arm == "lobe" else self.parcel[c], []).append(c)
        return g


def draw_contacts(ax, p):
    for n, (x, y) in p.xyz.items():
        res = p.lab[n]["resection"] == "resected"
        ax.scatter(x, y, s=16, facecolor=CONTACT if res else SURFACE, edgecolor=CONTACT, linewidth=0.5, zorder=3)
        if p.lab[n]["status_description"] == "seizure onset zone":
            ax.scatter(x, y, s=75, facecolor="none", edgecolor=SOZ_RING, linewidth=1.3, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def arrow(ax, p0, p1, lw=1.6, z=6):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=EDGE, lw=lw,
                                shrinkA=5, shrinkB=5, connectionstyle="arc3,rad=0.12"))


def short(name):
    return name.rsplit("-", 1)[0] if name.endswith(("-lh", "-rh")) else name


pts = [P(s) for s in args.subs]
fig, axs = plt.subplots(len(pts), 3, figsize=(15.5, 4.6 * len(pts)))
axs = np.atleast_2d(axs)
for row, p in enumerate(pts):
    a1, a2, a3 = axs[row]
    draw_contacts(a1, p)
    ch_edges = p.edges("channel")
    for s, t in ch_edges:
        arrow(a1, p.mid(s), p.mid(t), lw=1.4)
    a1.set_title(f"{p.sub} — channels: {len(ch_edges)} connections\n"
                 f"({sum(v > 0 for v in p.n.values())} channels with spikes)", loc="left", fontsize=10, color=INK)
    for ax, arm, name in ((a2, "parcel", "atlas parcels (Desikan–Killiany)"), (a3, "lobe", "lobes")):
        draw_contacts(ax, p)
        g = p.groups(arm)
        pos = {k: np.mean([p.mid(c) for c in v], axis=0) for k, v in g.items()}
        spikes = {k: sum(p.n[c] for c in v) for k, v in g.items()}
        smax = max(spikes.values()) or 1
        for k, (x, y) in pos.items():
            has_soz = any(p.soz(c) for c in g[k])
            ax.scatter(x, y, s=50 + 1300 * spikes[k] / smax, facecolor=FILL,
                       edgecolor=SOZ_RING if has_soz else EDGE, linewidth=2.2 if has_soz else 0.8, zorder=5)
        ed = p.edges(arm)
        for s, t in ed:
            arrow(ax, pos[s], pos[t], lw=2)
        involved = {k for e in ed for k in e}
        span = np.ptp(np.array(list(p.xyz.values())), axis=0).max()
        placed = []                                   # label anchors so far, data units
        for k in sorted(pos, key=lambda k: -spikes[k]):   # biggest groups keep their spot
            if spikes[k] < 0.15 * smax and k not in involved:
                continue
            lx, ly = pos[k]
            while any(abs(lx - qx) < 0.12 * span and abs(ly - qy) < 0.035 * span for qx, qy in placed):
                ly -= 0.035 * span                    # nudge down until it clears earlier labels
            placed.append((lx, ly))
            ax.annotate(short(k).replace("unknown", "medial wall"), (lx, ly), ha="center", va="top",
                        fontsize=7, color=INK, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.1", fc=SURFACE, ec="none", alpha=0.8))
        ax.set_title(f"{name}: {len(ed) or 'no'} connections ({len(g)} groups)", loc="left",
                     fontsize=10, color=INK)

fig.legend(handles=[
    Line2D([], [], color=EDGE, lw=1.8, label="connection found by the model"),
    Line2D([], [], marker="o", ls="", ms=12, mfc=FILL, mec=SOZ_RING, mew=2,
           label="group (size = spikes; red edge = holds a seizure-onset channel)"),
    Line2D([], [], marker="o", ls="", ms=5, mfc=CONTACT, mec=CONTACT, label="resected contact"),
    Line2D([], [], marker="o", ls="", ms=5, mfc=SURFACE, mec=CONTACT, label="preserved contact"),
    Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=SOZ_RING, mew=1.4, label="seizure-onset contact")],
    loc="lower center", ncol=5, frameon=False, fontsize=8.5)
fig.suptitle("The same spikes at three node scales: channels are too fine, lobes too coarse; atlas parcels sit "
             "in between and split the temporal lobe", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0.03, 1, 0.975))
out = os.path.join(RESULTS, "figures", "channels_parcels_lobes.png")
fig.savefig(out, dpi=140)
print("wrote", out)
