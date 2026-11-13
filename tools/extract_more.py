"""보조 영상에서 구간별로 가장 선명한 프레임을 뽑아 frames_multi/ 에 모은다 (메인 24장 + 보조).

실행: uv run --python 3.13 --with opencv-python-headless --with numpy python extract_more.py
"""
import json, shutil, zipfile
from pathlib import Path
import cv2

HERE = Path(__file__).parent
OUT = HERE / "frames_multi"
JOBS = [("20261113_135200.mp4", "low", 20), ("20261113_135233.mp4", "top", 20)]


def sharpest_per_segment(video, n):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    best = [(-1.0, None, -1)] * n
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        seg = min(n - 1, i * n // total)
        g = cv2.cvtColor(cv2.resize(fr, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        s = cv2.Laplacian(g, cv2.CV_64F).var()
        if s > best[seg][0]: best[seg] = (s, fr, i)
        i += 1
    cap.release()
    return [(s, fr, idx, idx / fps) for s, fr, idx in best if fr is not None]


def main():
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir()
    meta = []
    for f in sorted((HERE / "frames").glob("frame_*.jpg")):          # 메인 24장
        shutil.copy(f, OUT / f"main_{f.stem[-2:]}.jpg")
    for name, tag, n in JOBS:
        picks = sharpest_per_segment(HERE / name, n)
        for k, (s, fr, idx, t) in enumerate(picks, 1):
            fn = f"{tag}_{k:02d}.jpg"
            cv2.imwrite(str(OUT / fn), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.append(dict(file=fn, video=name, frame_index=idx, time_s=round(t, 2), sharpness=round(s, 1)))
        print(f"{name}: {len(picks)}장, 선명도 최소 {min(p[0] for p in picks):.1f} / 중앙 {sorted(p[0] for p in picks)[len(picks)//2]:.1f}")
    json.dump(meta, open(OUT / "frames_multi.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    files = sorted(OUT.glob("*.jpg"))
    with zipfile.ZipFile(HERE / "colab" / "frames_multi.zip", "w", zipfile.ZIP_STORED) as z:
        for f in files: z.write(f, f.name)
    print(f"모두 {len(files)}장 → colab/frames_multi.zip {(HERE / 'colab' / 'frames_multi.zip').stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
