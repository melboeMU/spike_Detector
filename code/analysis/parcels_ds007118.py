"""Label each bipolar channel with an atlas region (Desikan-Killiany), for the parcel arm.

Run AFTER mock_ds007118.py (uses its montage).

    python parcels_ds007118.py sub-018

ds007118 gives every contact's position on FreeSurfer's average brain
(fsaverage, mm; coordsystem.json: "derived from FreeSurfer recon-all and
projected to fsaverage template"). So each position can be matched to the
nearest vertex of fsaverage's pial (cortical) surface, and that vertex's
label in FreeSurfer's default atlas, Desikan-Killiany ("aparc", 34 regions per
hemisphere), taken as the region. Hemisphere from channels.tsv.

A bipolar channel gets the region at the MIDPOINT of its two contacts: the
signal is a difference between them, so it reflects the tissue in between.
Unlike the lobe arm, this gives every channel exactly one region (no mixed
groups).

Two checks are printed, because a wrong coordinate frame would give
plausible-looking but meaningless labels:
  1. distance from each contact to the nearest surface vertex (should be a few
     mm for contacts lying on the cortex);
  2. agreement between the atlas region's lobe and the dataset's own `lobe`
     label for each contact.

Needs Current work/data/fsaverage/{surf/?h.pial,label/?h.aparc.annot}
(from MNE's fsaverage archive, osf.io 3bxqt, md5 5133fe92b7b8f03ae19219d5f46e4177).
Writes derivatives .../parcels.tsv and results/<sub>/parcels.txt.
"""
import csv, os
from collections import Counter
import numpy as np
import mne
import nibabel as nib
from scipy.spatial import cKDTree

from mock_common import DATA, Recording, Tee, parser

FS = os.path.normpath(os.path.join(DATA, "..", "fsaverage"))
# Desikan-Killiany regions -> the dataset's lobe vocabulary (used only for check 2)
LOBE = {**{r: "frontal" for r in ("superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal",
                                    "parsopercularis", "parstriangularis", "parsorbitalis",
                                    "lateralorbitofrontal", "medialorbitofrontal", "frontalpole",
                                    "paracentral")},
        **{r: "central" for r in ("precentral", "postcentral")},
        **{r: "temporal" for r in ("superiortemporal", "middletemporal", "inferiortemporal", "bankssts",
                                     "fusiform", "transversetemporal", "entorhinal", "temporalpole",
                                     "parahippocampal")},
        **{r: "parietal" for r in ("superiorparietal", "inferiorparietal", "supramarginal", "precuneus")},
        **{r: "occipital" for r in ("lateraloccipital", "lingual", "cuneus", "pericalcarine")},
        "insula": "insula"}

R = Recording(parser(__doc__).parse_args())
out = Tee(os.path.join(R.results, "parcels.txt"))
lab = {r["name"]: r for r in csv.DictReader(open(R.channels_tsv, encoding="utf-8"), delimiter="\t")}
xyz = {r["name"]: np.array([float(r["x"]), float(r["y"]), float(r["z"])])
       for r in csv.DictReader(open(R.electrodes_tsv, encoding="utf-8"), delimiter="\t")
       if r["x"] not in ("n/a", "")}
montage = [l.strip() for l in open(R.montage_txt, encoding="utf-8") if l.strip()]

hemi = {}
for h in ("lh", "rh"):
    rr, _ = mne.read_surface(os.path.join(FS, "surf", f"{h}.pial"))
    labels, _, names = nib.freesurfer.read_annot(os.path.join(FS, "label", f"{h}.aparc.annot"))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    hemi[h] = (cKDTree(rr), labels, names)


def region(p):
    """Region at the nearest vertex of whichever hemisphere's surface the point lies on.

    Not the labelled hemisphere: some recordings are MIRRORED onto the other
    side of fsaverage (sub-025: all 112 contacts labelled right, all lying on
    the left surface). Desikan-Killiany regions are mirror images, so the
    region is right; the hemisphere NAME comes from channels.tsv (hemi_of)."""
    best = None
    for h, (tree, labels, names) in hemi.items():
        d, v = tree.query(p)
        if best is None or d < best[1]:
            k = labels[v]
            best = ((names[k] if k >= 0 else "unknown"), d, h)
    return best


def hemi_of(contact):
    return "lh" if lab[contact]["hemisphere"].upper().startswith("L") else "rh"


out(f"recording: {R.label}   atlas: Desikan-Killiany (fsaverage aparc)")

# check 1 + 2 on contacts
dist, agree, compared, mirrored = [], 0, 0, 0
for n, p in xyz.items():
    reg, d, surf = region(p)
    mirrored += surf != hemi_of(n)
    dist.append(d)
    dl = LOBE.get(reg)
    if dl and lab[n]["lobe"] not in ("n/a", ""):
        compared += 1
        agree += dl == lab[n]["lobe"]
dist = np.array(dist)
out(f"check 1 - contact to nearest cortical surface vertex: median {np.median(dist):.1f} mm, "
    f"90th percentile {np.percentile(dist, 90):.1f} mm, max {dist.max():.1f} mm")
out(f"check 2 - atlas lobe matches the dataset's lobe label: {agree}/{compared} contacts "
    f"({100 * agree / max(compared, 1):.0f}%)")
if mirrored:
    out(f"note - {mirrored}/{len(xyz)} contacts lie on the opposite hemisphere's surface to their "
        f"label (mirrored onto one side of fsaverage); region taken from the surface they lie on")

# channels: region at the midpoint
rows = []
for c in montage:
    a, b = c.split("-")
    mid = (xyz[a] + xyz[b]) / 2
    reg, d, _ = region(mid)
    rows.append((c, f"{reg}-{hemi_of(a)}", d))
with open(os.path.join(R.deriv, "parcels.tsv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["channel", "parcel", "midpoint_to_surface_mm"])
    for c, reg, d in rows:
        w.writerow([c, reg, f"{d:.1f}"])

cnt = Counter(reg for _, reg, _ in rows)
out(f"\n{len(montage)} channels -> {len(cnt)} parcels:")
for reg, n in cnt.most_common():
    out(f"  {reg:32s} {n:3d} channels")
out(f"\nwritten {os.path.relpath(os.path.join(R.deriv, 'parcels.tsv'))}")
