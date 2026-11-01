"""Why does the detector find (or miss) spikes on each kind of channel?

Run AFTER mock_ds007118.py (uses its montage).

    python check_mock_detection.py sub-011

Re-runs spikedet.py with the rejected candidates kept, and breaks the
candidates down by the clinicians' channel label (SOZ / spiking / normal):
how many candidates each kind of channel produces per minute (stage 1: the
adaptive threshold), what share pass the gates, which gate rejects the rest,
and each kind's background amplitude (robust SD). Built to explain sub-011,
where the SOZ channels got almost no spikes.

Writes results/<sub>/detection_check.txt. Needs ~2 GB of memory for a
128-contact, 66-minute recording.
"""
import csv, os, re, sys
from collections import Counter, defaultdict
import numpy as np

from mock_common import HERE, Recording, Tee, parser

sys.path.insert(0, os.path.join(HERE, "..", "SPIKE anirudh"))
from spikedet import SpikeParams, detect_spikes          # noqa: E402
import mne                                               # noqa: E402
mne.set_log_level("ERROR")

R = Recording(parser(__doc__).parse_args())
out = Tee(os.path.join(R.results, "detection_check.txt"))
raw=mne.io.read_raw_edf(R.edf,preload=False); fs=raw.info["sfreq"]
names=[re.sub(r"-AV$","",c) for c in raw.ch_names]; idx={n:i for i,n in enumerate(names)}
lab={r["name"]:r for r in csv.DictReader(open(R.channels_tsv,encoding="utf-8"),delimiter="\t")}
bn=[l.strip() for l in open(R.montage_txt) if l.strip()]
ii=np.array([idx[b.split("-")[0]] for b in bn]); jj=np.array([idx[b.split("-")[1]] for b in bn])
n=raw.n_times; B=np.empty((len(bn),n),np.float32); step=300000
for s in range(0,n,step):
    X=raw.get_data(start=s,stop=min(s+step,n)).astype(np.float32)*1e6; B[:,s:s+X.shape[1]]=X[ii]-X[jj]
del X
import json
mains=float(json.load(open(R.ieeg_json,encoding="utf-8"))["PowerLineFrequency"])
p=SpikeParams(fs=fs,notch_freqs=tuple(mains*k for k in (1,2,3)))
ev=detect_spikes(B,p,ch_names=bn,return_rejected=True)
rms={b:float(np.median(np.abs(B[k])))*1.4826 for k,b in enumerate(bn)}
del B
def grp(b):
    s={lab[c]["status_description"] for c in b.split("-")}
    return "SOZ" if "seizure onset zone" in s else "spiking" if "spiking" in s else "normal"
dur=n/fs/60
by=defaultdict(Counter); amp=defaultdict(list)
for c,a,r,am in zip(ev["channel_name"],ev["accepted"],ev["reject_reason"],ev["amplitude_uv"]):
    by[grp(c)]["candidates"]+=1; by[grp(c)]["accepted" if a else r]+=1
    if not a and r=="not_broadband": amp[grp(c)].append(am)
nch=Counter(grp(b) for b in bn)
out(f"{R.label}: {dur:.0f} min")
out(f"{'label':8s} {'chans':>5s} {'cand/ch/min':>11s} {'accepted%':>9s}  reasons (share of candidates)")
for g in ("SOZ","spiking","normal"):
    c=by[g]; tot=c["candidates"] or 1
    rs=", ".join(f"{k} {100*v/tot:.0f}%" for k,v in c.most_common() if k not in ("candidates","accepted"))
    out(f"{g:8s} {nch[g]:5d} {c['candidates']/nch[g]/dur:11.2f} {100*c['accepted']/tot:8.1f}%  {rs}")
out("median background amplitude (robust SD, uV):", {g: round(float(np.median([rms[b] for b in bn if grp(b)==g])),1) for g in ("SOZ","spiking","normal")})
