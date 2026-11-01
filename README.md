# Spike detection and parcel-wise analysis — handover

From Cezar (Spike model role), 25 October 2026. MRP Project 2, *Finding the Brain
Area Worth Removing*. Dataset: OpenNeuro **ds007118** (65 subjects, ~30-min
sleep iEEG, 1000 Hz, subdural grids/strips, per-channel surgery labels).

This package holds everything needed to take over two parts of the pipeline:

1. **Spike detection** — raw recording → list of spike times per channel.
2. **Parcel-wise analysis** — assigning channels to Desikan–Killiany brain
   regions, and the network at region level.

Everything here also lives in the team repository
(https://github.com/cezar03/Epileptic-Spikes-Networks, folder `Current work/`),
except the data, which git does not track.

---

## 1. What is in the zip

```
code/
  SPIKE anirudh/spikedet.py      the detector itself (from A. Wodeyar; unchanged)
  SPIKE anirudh/demo_spikedet.py its synthetic demo
  analysis/mock_common.py        shared paths and arguments for every script
  analysis/download_ds007118.py  download one recording from OpenNeuro, checksum-verified
  analysis/mock_ds007118.py      DETECTION: recording -> bipolar channels -> spikes
  analysis/check_mock_detection.py  diagnostic: why spikes are found/missed per channel type
  analysis/merge_codetections.py    experimental: count one discharge seen on several channels once
  analysis/join_runs.py          join a patient's runs into one spike list + quality checks (bursts, duplicates)
  analysis/export_joined.py      export joined spike lists by channel and by parcel
  analysis/parcels_ds007118.py   PARCELS: label each channel with its Desikan-Killiany region
  analysis/mock_ds007118_fit.py  network model (channel / parcel / lobe); needs R
  analysis/mock_fit.R            the group-lasso fit called by the line above
  analysis/plot_node_scales.py   draws the channels vs parcels vs lobes figure
old_code/
  mock_ds007118_v1_2026-09-29.py the first version of the detection script
  changes_v1_to_current.diff     line-by-line changes since then
  group11_legacy/                last year's scripts (they never detected spikes; see §3)
results/
  figures/channels_parcels_lobes.png   the node-scale comparison (§5)
  figures/merge_versions.png           the co-detection merge comparison (§4)
  mock_patients.md                     full write-up of all results so far
  sub-XXX/detection.txt                detection report per patient
  sub-XXX/detection_check.txt          detector diagnostics (sub-011, sub-018)
  sub-XXX/parcels.txt                  parcel labelling report + sanity checks
  sub-XXX/modelling_parcel.txt         parcel-level model output
  sub-XXX/merge_check*.txt             co-detection merge reports
data/
  ds007118/sub-XXX/                    channel labels, coordinates, sidecars (no recording)
  ds007118/sub-XXX/derivatives_mock/<acq_run>/
      spikes.tsv            THE DETECTED SPIKES (one row per spike)
      bipolar_channels.txt  every channel, including ones with no spikes
      detect_config.json    every setting used, recording length, git commit
      parcels.tsv           channel -> Desikan-Killiany region
      network_*.tsv         connections found (channel / parcel / lobe)
      targets_*.tsv         per-node fit outcome
      merged_near/          spikes after the lenient co-detection merge (§4)
  fsaverage/                atlas surface + labels the parcel script needs (14 MB)
```

The folder layout matches the repository's `Current work/`, so the scripts find
their inputs without edits once you unzip.

---

## 2. How to run it

**Python** 3.13 with numpy 2.4, scipy 1.15, mne 1.13, nibabel 5.4, matplotlib
3.10 (older versions are probably fine). **R** 4.6 with `grpreg` and
`R.matlab` — only for the network fit. Set the path to `Rscript` in
`code/analysis/mock_common.py` (`RSCRIPT`, currently a Windows path).

**One-time:** clone the dataset's metadata (small, no recordings) next to
`code/`:

```bash
git clone https://github.com/OpenNeuroDatasets/ds007118.git
```

**Per patient** (from `code/analysis/`; sub-027 defaults to acquisition B,
everyone else to A):

```bash
python download_ds007118.py sub-018              # ~570 MB recording, verified
python mock_ds007118.py sub-018                  # detection   (~40 s)
python parcels_ds007118.py sub-018               # region labels (seconds)
python mock_ds007118_fit.py sub-018 --arm parcel --workers 2   # region network (R)
python plot_node_scales.py sub-027 sub-025 sub-018             # the figure
```

The recordings are **not** in this zip (150 MB–1 GB each). Everything you need
for the parcel work *without* re-running detection is included: the spike
lists, channel lists, labels and coordinates. `parcels_ds007118.py` and
`plot_node_scales.py` run on those alone.

---

## 3. How detection works, and what changed

### The pipeline (`mock_ds007118.py`)

1. **Load** the EDF in 5-minute chunks. Strip the `-AV` suffix from channel
   names so they match `channels.tsv`, and **assert** an exact match — a naive
   join matches zero channels silently (problem P-27).
2. **Bipolar montage:** each channel is contact *n* minus contact *n+1* on the
   same lead (`A14-A15` = A14 − A15, µV), but only if the two are physically
   adjacent: ≤ 1.5 × the median neighbour spacing. That stops pairs jumping
   between strips or grid rows. Contacts labelled `artifact` are dropped.
3. **Detect** with `spikedet.py`, default settings, notch at the mains
   frequency read from the recording's sidecar (60 Hz for this dataset).
4. **Write** `spikes.tsv`, `bipolar_channels.txt` and `detect_config.json`,
   and a report to `results/<sub>/detection.txt`, including a check of spike
   rate against the clinicians' labels (seizure-onset / spiking / normal).

### The detector (`spikedet.py`, unchanged)

- **Stage 1:** a multi-scale matched filter (10–66 ms), an adaptive threshold
  (6 × robust SD, per 60 s block) and a 40 ms refractory period → candidates.
- **Stage 2:** four gates reject non-spikes — oscillatory bursts (cycle count),
  no broadband power, no 2–8 Hz slow wave, wrong width (half-width 10–80 ms).
- **Stage 3:** events on more than half the channels within 20 ms are rejected
  as artifacts.
- Bands: 1–80 Hz overall, 20–70 Hz for the sharp component, 2–8 Hz slow wave.
  All values are saved in each `detect_config.json`.

### Old code

- **Group 11 (last year) never detected spikes.** They received spike times
  from MUMC+'s own detector (a spike/gamma-burst model plus an AI classifier
  trained on epileptologist marks), and that code was never handed over.
  `old_code/group11_legacy/` shows what they did with it: `Spike_trains.asv`
  reads the exported spreadsheet into spike trains, `density_detection.m`
  measures event density. Context only — nothing there detects spikes.
- **Our first version** (`old_code/mock_ds007118_v1_2026-09-29.py`) ran
  `spikedet.py` on one recording (sub-027, acquisition B) with everything
  hard-coded.

### What we changed since the first version, and why

| Change | Why |
|---|---|
| Any patient / acquisition / run via arguments (shared in `mock_common.py`) | It only ran sub-027 |
| Recording read in 5-minute float32 chunks instead of all at once | Loading a 128-contact file at once needed ~2 GB of memory |
| Notch frequency read from the recording's sidecar instead of a fixed 60 Hz | Correct for any site, not just US mains |
| Outputs in one folder per recording; `detect_config.json` with every setting, the recording length and the git commit | Any result can be traced to the exact settings that made it |
| Report also saved to `results/<sub>/detection.txt` | It was printed only |
| Contacts without coordinates skipped safely | Some patients have `n/a` coordinates |

**Not changed:** the detector, its settings, the montage rule and the
name check. After the rewrite, sub-027's spikes and montage were verified
**byte-identical** to the first version's output.

---

## 4. Detection results and known problems

| Patient | Outcome | Length | Bipolar channels (with spikes) | Candidates → spikes |
|---|---|---|---|---|
| sub-027 (acq-B) | seizure-free (ILAE 1) | 34 min | 22 (16) | 6,063 → 1,103 |
| sub-025 | seizure-free (ILAE 1) | 32 min | 64 (53) | 11,953 → 692 |
| sub-018 | seizure-free (ILAE 1) | 37 min | 88 (79) | 26,376 → 2,078 |
| sub-011 | seizures continued (ILAE 5) | 66 min | 92 (62) | 26,251 → 686 |

**What works:** on the three seizure-free patients the busiest channels are
seizure-onset channels — the detector agrees with the clinicians.

**Known problems** (IDs refer to `problems.md` in the repository):

- **P-33 — sub-011's seizure-onset channels are nearly silent.** Its busiest
  channels are clinically *normal*. `check_mock_detection.py` shows the
  shortfall is at stage 1: seizure-onset channels produce 8× fewer candidates.
  A louder background was tested and ruled out. Cause unknown — rare spiking
  between seizures, or a spike shape the detector misses. **Check detector vs
  clinical labels on every patient before using it.**
- **P-29 — one discharge counted on several channels.** Neighbouring channels
  catch the same spike a few ms apart; the model then reads it as one channel
  driving another. `merge_codetections.py` counts it once:
  - strict (spikes within 20 ms on *any* channels): networks almost vanish,
    including plausible ones;
  - lenient, `--near` (only channels within the patient's largest
    neighbour distance, 12–19 mm): removes the obvious double counts, keeps
    sub-018's lead–lag pattern. See `results/figures/merge_versions.png`.
  - **Which rule is right is an open question for the supervisors (Q32).**
- **P-28 — bipolar channels share contacts** (`B8-B9` and `B9-B10` both use
  B9), so one discharge appears on both by construction.
- **P-30 — the adjacency rule drops real neighbours** on large grids (37–39
  pairs skipped), because the coordinates are on a template brain and
  distances get stretched.
- **P-34 — some spikes are counted twice on the same channel** (both peaks of
  one two-phase spike, 19–40 ms apart), despite the 40 ms refractory period.
  4–23 per patient; small. Likely cause in the code: see weak point 4 below.

### Known weak points of the detector itself

From reading `spikedet.py` and from what the runs showed. None of them is fixed;
they are what to keep in mind when reading its output.

1. **Each channel is judged against its own background.** A candidate needs a
   score 6 robust standard deviations above that channel's typical level,
   recalculated every 60 s. So a noisy channel needs a bigger event to count, and
   a quiet channel picks up smaller ones; spike counts are not comparable as
   absolute amounts between channels. On sub-011 the seizure-onset channels
   produced 8× fewer candidates (P-33). A louder background alone did not explain
   it, but the relative threshold is part of the picture.
2. **Each channel is detected on its own.** One discharge reaching several
   neighbouring channels is reported once per channel, a few ms apart (P-29). The
   detector has no notion of "the same event"; merging is our post-step
   (`merge_codetections.py`).
3. **The cross-channel artifact check is narrow.** It rejects an event only if
   more than **half of all channels** spike within the **same 20 ms**. Bursts that
   are spread over a second or so, or cover a third of the channels, pass. On the
   joined patients, 54% of sub-015's and 33% of sub-062's spikes fell in seconds
   where more than a quarter of the channels spiked at once (P-36) — likely
   artifact or seizure activity on sub-015. Check with the burst test in
   `join_runs.py` (§6 of its report) before modelling a patient.
4. **The 40 ms gap is enforced before the peak is refined.** Candidates are
   thinned so no two are closer than 40 ms, and *then* each one is moved by up
   to 12 ms onto the largest point of the signal. Two candidates 40 ms apart can
   end up as little as 16 ms apart — which fits the same-channel pairs of
   opposite sign we found (P-34). Fix to try: re-apply the 40 ms gap after
   refining. A reading of the code, not yet confirmed.
5. **One check does almost all the filtering.** On sub-018, of 24,298 rejected
   candidates, 22,747 failed the broadband check (energy in at least 3 of 4
   bands, 8–100 Hz, above the background); the width check rejected 1,105, the
   oscillation check 442 and the slow-wave check 4. So the detector's
   sensitivity rests mostly on that one check's settings (`band_z` = 2,
   `min_bands` = 3), not only on `thresh_mad` — worth including in any
   threshold sweep.
6. **Validated only at channel level.** The detector was tested on synthetic
   data by its author, and on ds007118 we only know that the busiest channels
   are usually the clinically labelled ones. Without clinician-marked spike
   times we cannot measure its sensitivity or false alarms per minute (Q16).
7. **Default settings, not tuned for this dataset.** Every run used the same
   defaults (saved in each `detect_config.json`). Do **not** tune them per
   patient to match the clinical labels: the labels are also what results are
   scored against, so that would make the results circular.

---

## 5. Parcel-wise analysis

### How channels get a region (`parcels_ds007118.py`)

- ds007118 gives each contact's position on **fsaverage**, FreeSurfer's
  standard brain. Each position is matched to the nearest point of
  fsaverage's cortical surface, and that point's label in the
  **Desikan–Killiany** atlas (34 cortical regions per hemisphere) is taken.
- A bipolar channel gets the region at the **midpoint** of its two contacts,
  so every channel belongs to exactly one region.
- Two sanity checks are printed to `results/<sub>/parcels.txt`: distance to
  the surface, and agreement with the dataset's own lobe labels.

### How spikes are pooled (`mock_ds007118_fit.py --arm parcel`)

All spike times of a region's channels go into one list; which channel fired
is dropped. The list becomes a 0/1 series in 19.5 ms bins (1 = at least one
spike). Then the model treats each region like a channel. Two spikes in the
same region *and* the same bin become one — the only merging that happens.

### Results (`results/figures/channels_parcels_lobes.png`)

| Patient | Channels | Parcels | Lobes |
|---|---|---|---|
| sub-027 | 22 nodes, 6 connections | 8 nodes, 6 connections | 4 nodes, 0 |
| sub-025 | 64 nodes, 4 connections | 12 nodes, 2 connections | 6 nodes, 0 |
| sub-018 | 88 nodes, 18 connections | 23 nodes, 16 connections | 12 nodes, 6 |

- **Parcels split the temporal lobe**, which lobes cannot: all three patients
  are temporal-lobe cases, so "temporal" is always the lobe answer. Seizure-onset
  channels fall mostly in entorhinal cortex, plus superior temporal on sub-018.
- **sub-018:** one cluster inside the seizure-onset area (fusiform,
  entorhinal, temporal pole, middle/superior temporal), one in preserved
  frontal/central/parietal cortex. Top hub: fusiform, a seizure-onset region.
- **sub-027:** all 6 connections among removed temporal regions; top hub
  temporal pole, which is also simply the busiest region.
- sub-011 has no parcels yet (its network is empty at every scale).

### Caveats you inherit

- **Co-detections cross region borders.** Of the spike pairs < 20 ms apart,
  only ~10–20% fall in the same region and bin and get merged; about **three
  quarters are in different regions** and can still create false
  region-to-region connections. Fix: merge first, then pool —
  `mock_ds007118_fit.py sub-018 --arm parcel --tag merged_near`.
- **sub-025's coordinates are mirrored** onto the wrong hemisphere (P-35); the
  script takes the region from the surface the contact actually lies on. Only
  60% of its contacts agree with the dataset's lobe labels (vs 99–100% on the
  others), and some land on the unlabelled medial wall (the "medial wall" node
  in the figure).
- **Template brain, not the patient's own,** so contacts near a border may get
  the neighbouring region.
- **Cortex only:** Desikan–Killiany has no hippocampus or amygdala. Acceptable
  for these surface electrodes, but worth stating.
- **Atlas choice is open** (questions Q11/Q25 for Wodeyar, Q23 for the clinical
  supervisors): keep Desikan–Killiany, switch to AAL (what Group 11 used), or a
  finer atlas.

---

## 6. Suggested next steps

1. Run the detector–label check on every new patient before modelling it
   (P-33); log excluded patients with a reason.
2. Re-fit the parcel networks on the lenient-merged spikes (`--tag
   merged_near`) and compare with the current ones.
3. Run `parcels_ds007118.py` on more patients; flag any with low lobe
   agreement like sub-025.
4. Patients with several recordings of the same electrodes (12 in ds007118,
   **sub-017** first: 145 min, seizures continued) give the model more data —
   list in the repository's `plan.md`, step 3.

Questions: Cezar. Full background in the repository: `results/mock_patients.md`
(results), `problems.md` (P-numbers), `questions.md` and
`supervisor_meeting.md` (open questions).
