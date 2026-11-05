"""MultiVENT-G frame extraction: sample frames from the source videos (MVG_VIDEO_DIR) at the timestamps the item and
twin files refer to, writing data/multiventG/frames*. Videos are obtained from the MultiVENT-G release; they are never
redistributed here."""
import json, os, subprocess, sys
VID_DIR = os.environ.get("MVG_VIDEO_DIR", "data/multiventG/videos")
OUT = "data/multiventG/frames"; os.makedirs(OUT, exist_ok=True)
index = {os.path.splitext(f)[0]: os.path.join(VID_DIR, f) for f in os.listdir(VID_DIR)}
seen, missing, bad, done = set(), set(), set(), 0
manifest = open("data/multiventG/frames_manifest.jsonl", "w")
for l in open("data/multiventG/claims.jsonl"):
    c = json.loads(l); vid, ev = c["video_id"], c.get("evidence", {})
    if vid not in index:
        missing.add(vid); continue
    ts = []
    if ev.get("time"):
        t0, t1 = ev["time"]; ts = [t0 + (t1 - t0) * f for f in (0.15, 0.5, 0.85)]
    elif ev.get("frame") is not None:
        ts = [ev["frame"] / 30.0]
    for t in ts:
        key = f"{vid}_{t:.2f}"
        if key in seen: continue
        seen.add(key)
        out = f"{OUT}/{key}.jpg"
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}",
                            "-i", index[vid], "-frames:v", "1", "-q:v", "3", out, "-y"],
                           timeout=60)
        if r.returncode == 0 and os.path.exists(out):
            manifest.write(json.dumps({"video_id": vid, "t": round(t, 2), "path": out}) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"[{done}] extracted", flush=True)
        else:
            bad.add(key)
manifest.close()
open("data/multiventG/missing_videos.txt", "w").write("\n".join(sorted(missing)))
print(f"DONE frames: {done} | missing videos: {len(missing)} | failed: {len(bad)}", flush=True)
