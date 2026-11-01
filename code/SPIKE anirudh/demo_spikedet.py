import numpy as np, time
from spikedet import SpikeParams, detect_spikes

fs = 1000.0
dur = 120.0
n = int(fs * dur)
rng = np.random.default_rng(1)

def pink(n, rng):
    w = rng.normal(size=n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1/fs); f[0] = f[1]
    W /= f**0.7
    x = np.fft.irfft(W, n)
    return 30 * x / x.std()

n_ch = 8
data = np.stack([pink(n, rng) for _ in range(n_ch)])

# --- inject spikes on channels 0-3
spike_times = []
t = 3.0
while t < dur - 3:
    spike_times.append(t)
    if rng.random() < 0.35:          # doublet 200 ms later
        spike_times.append(t + 0.2)
    t += rng.uniform(1.5, 4.0)
spike_times = np.array(spike_times)

tt = np.arange(-0.15, 0.15, 1/fs)
for ch in range(4):
    for st in spike_times:
        i = int(st * fs)
        tau = 0.012                                     # cusped apex -> broadband
        spk = -np.exp(-np.abs(tt)/tau)
        spk += 0.35*np.exp(-(tt-0.09)**2/(2*0.05**2))   # aftergoing slow wave
        spk = spk/np.abs(spk).max()
        seg = slice(i-len(tt)//2, i-len(tt)//2+len(tt))
        data[ch, seg] += 250 * spk

# --- inject gamma/beta bursts on ALL channels (the distractor)
n_burst = 0
t = 2.0
while t < dur - 2:
    f0 = rng.uniform(22, 65)
    ncyc = rng.integers(5, 10)
    L = int(ncyc/f0*fs)
    tb = np.arange(L)/fs
    env = np.hanning(L)
    for ch in range(n_ch):
        b = 150*env*np.sin(2*np.pi*f0*tb + rng.uniform(0, 6))
        i = int(t*fs)
        data[ch, i:i+L] += b
    n_burst += 1
    t += rng.uniform(1.2, 2.5)

p = SpikeParams(fs=fs, notch_freqs=(60., 120.))
p.channel_block = 8
t0 = time.time()
ev = detect_spikes(data, p, return_rejected=True)
el = time.time() - t0
print(f"runtime {el:.2f}s for {n_ch} ch x {dur}s "
      f"({n_ch*dur/el:.0f} channel-seconds/s)")

acc = ev["accepted"].astype(bool)
print("candidates:", acc.size, " accepted:", acc.sum())
import collections
print(collections.Counter(ev["reject_reason"][~acc]))

# scoring on spike channels
for ch in [0, 4]:
    m = (ev["channel"] == ch) & acc
    det = ev["time_s"][m]
    if ch < 4:
        hit = np.array([np.any(np.abs(det - s) < 0.05) for s in spike_times])
        fp = np.array([not np.any(np.abs(spike_times - d) < 0.05) for d in det])
        print(f"ch{ch}: sens {hit.mean():.3f} ({hit.sum()}/{len(spike_times)}), "
              f"FP {fp.sum()} ({fp.sum()/(dur/60):.2f}/min)")
    else:
        print(f"ch{ch} (no spikes, {n_burst} bursts): {det.size} detections "
              f"= {det.size/(dur/60):.2f} FP/min")

# doublet check
d0 = np.sort(ev["time_s"][(ev["channel"] == 0) & acc])
if d0.size > 1:
    print("min inter-detection interval:", np.round(np.diff(d0).min()*1000, 1), "ms")
