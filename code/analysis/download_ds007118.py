"""Download one ds007118 recording without git-annex, and verify it.

    python download_ds007118.py sub-025              # acq-A, run-01
    python download_ds007118.py sub-027 --acq B
    python download_ds007118.py sub-031 --run 02

Follows data_download_handoff.md exactly. The EDF in the metadata clone is an
annex pointer whose key encodes the size and SHA-256. The clone's git-annex
branch records the file's S3 path and VERSION ID for the public remote, so the
URL is built from recorded values, not guessed. The download is checked against
the key's SHA-256 and deleted if it does not match.

The recording and its sidecars are copied to Current work/data/ds007118/<sub>/,
outside the clone, so git-annex's view of the clone is untouched.
"""
import argparse, glob, hashlib, os, re, shutil, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CLONE = os.path.normpath(os.path.join(HERE, "..", "..", "ds007118"))
DATA = os.path.normpath(os.path.join(HERE, "..", "..", "data", "ds007118"))
S3 = "https://s3.amazonaws.com/openneuro.org"
S3_PUBLIC_UUID = "6291264d"          # remote "s3-PUBLIC" in origin/git-annex:remote.log

ap = argparse.ArgumentParser()
ap.add_argument("sub")
ap.add_argument("--acq", default="A")
ap.add_argument("--run", default="01")
a = ap.parse_args()

ieeg = os.path.join(CLONE, a.sub, "ses-01", "ieeg")
stem = f"{a.sub}_ses-01_task-sleep_acq-{a.acq}_run-{a.run}"
pointer = os.path.join(ieeg, stem + "_ieeg.edf")
key = os.path.basename(open(pointer, encoding="utf-8").read().strip())
m = re.match(r"SHA256E-s(\d+)--([0-9a-f]{64})\.edf$", key)
if not m:
    sys.exit(f"not an annex pointer: {pointer}")
size, sha = int(m.group(1)), m.group(2)

# git-annex "hashdirlower": first 3 + next 3 hex digits of md5(key)
h = hashlib.md5(key.encode()).hexdigest()
rmet = subprocess.run(["git", "-C", CLONE, "show", f"origin/git-annex:{h[:3]}/{h[3:6]}/{key}.log.rmet"],
                      capture_output=True, text=True, check=True).stdout
# lines look like: <time>s <uuid>:V +<versionId>#ds007118/<path>
hits = [ln for ln in rmet.splitlines() if f" {S3_PUBLIC_UUID}" in ln and ":V +" in ln]
if not hits:
    sys.exit("no s3-PUBLIC version record for this key")
version, path = re.search(r":V \+([^#]+)#(.+)$", hits[-1]).groups()
url = f"{S3}/{path}?versionId={version}"

outdir = os.path.join(DATA, a.sub)
os.makedirs(outdir, exist_ok=True)
for f in glob.glob(os.path.join(ieeg, f"{a.sub}_ses-01_task-sleep_acq-{a.acq}_*")):
    if not f.endswith(".edf"):
        shutil.copy2(f, outdir)                        # sidecars are plain text in the clone
dest = os.path.join(outdir, stem + "_ieeg.edf")

if os.path.exists(dest) and os.path.getsize(dest) == size:
    print(f"{stem}: already present, re-verifying")
else:
    print(f"{stem}: downloading {size / 1e6:.0f} MB\n  {url}")
    t0 = time.time()
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=8 << 20)
    os.replace(tmp, dest)
    print(f"  done in {time.time() - t0:.0f}s")

d = hashlib.sha256()
with open(dest, "rb") as f:
    for blk in iter(lambda: f.read(8 << 20), b""):
        d.update(blk)
if os.path.getsize(dest) != size or d.hexdigest() != sha:
    os.remove(dest)
    sys.exit(f"  VERIFY FAILED (size or SHA-256 mismatch) -- file deleted")
print(f"  SHA-256 verified ({sha[:8]}...{sha[-4:]}) -> {os.path.relpath(dest, HERE)}")
