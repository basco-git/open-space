"""코랩에서 받은 points.npz 로 스플랫 PLY 를 만들고 세 형식으로 변환해 수치를 잰다.

실행: uv run --python 3.13 --with numpy --with matplotlib python build.py [--K 1.4] [--n-keep 0]
결과: out/scene.ply, out/scene.compressed.ply, out/scene.sog, out/scene.spz, out/results.json, out/cameras.png
"""
import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
SH_C0 = 0.28209479177387814
TOOL = "@playcanvas/splat-transform@3.10.3"


def to_splat_ply(xyz, rgb, path, K=1.4, opacity=0.9):
    """INRIA 형식 이진 PLY (f_rest 없음, 알갱이당 17 float = 68바이트)."""
    n = len(xyz)
    extent = float((xyz.max(0) - xyz.min(0)).max())
    r = K * extent / np.sqrt(n)
    p = xyz.astype(np.float32).copy()
    p[:, 1] *= -1.0  # 모델은 Y 아래, three.js 는 Y 위
    p[:, 2] *= -1.0
    out = np.zeros((n, 17), dtype=np.float32)
    out[:, 0:3] = p
    out[:, 6:9] = (rgb.astype(np.float32) / 255.0 - 0.5) / SH_C0
    out[:, 9] = np.log(opacity / (1.0 - opacity))
    out[:, 10:13] = np.log(r)
    out[:, 13] = 1.0
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
           + "".join(f"property float {k}\n" for k in props) + "end_header\n").encode()
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(out.tobytes())
    return r, extent, len(hdr)


def look_target(cams):
    """카메라 광축(+Z)들에 가장 가까운 점 = 모두가 바라본 대상의 위치 (최소제곱)."""
    P = cams[:, :3, 3].astype(np.float64)
    F = cams[:, :3, 2].astype(np.float64)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, f in zip(P, F):
        M = np.eye(3) - np.outer(f, f); A += M; b += M @ p
    return np.linalg.solve(A, b)


def fit_table_plane(xyz, target, ring=0.45, tol=0.015, iters=400):
    """대상 둘레 바깥 점들로 RANSAC 평면 맞추기. 법선은 모델 위쪽(-y)을 향하게."""
    q = xyz.astype(np.float64)
    cand = q[np.hypot(q[:, 0] - target[0], q[:, 2] - target[2]) > ring]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        s = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; k = (np.abs((cand - s[0]) @ n) < tol).sum()
        if best is None or k > best[0]: best = (k, n, s[0])
    _, n, p0 = best
    inl = cand[np.abs((cand - p0) @ n) < tol]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    return (-n if n[1] > 0 else n), c


def rot_to(a, b):
    """단위 벡터 a 를 b 로 보내는 회전 행렬 (로드리게스)."""
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


def count_from_header(path):
    with open(path, "rb") as f:
        head = f.read(4096).split(b"end_header")[0].decode("latin1")
    for line in head.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=float, default=1.4)
    ap.add_argument("--n-keep", type=int, default=0, help="0 이면 전체, 숫자면 고르게 뽑기")
    ap.add_argument("--opacity", type=float, default=0.9)
    ap.add_argument("--crop", type=float, default=0.0,
                    help="0 보다 크면 카메라 시선이 모이는 점(대상) 둘레 이 반경 안의 점만 남김")
    ap.add_argument("--level", action="store_true", help="탁자 면을 찾아 수평으로 돌리고 대상을 원점에 둠")
    ap.add_argument("--clean-r", type=float, default=0.0, help="--level 과 함께: 이 반경 밖의")
    ap.add_argument("--clean-h", type=float, default=0.03, help="탁자 위 이 높이를 넘는 점을 허공 점으로 보고 지움")
    ap.add_argument("--npz", default=str(HERE / "points.npz"))
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()

    d = np.load(a.npz)
    xyz, rgb, cams = d["xyz"], d["rgb"], d["cams"]
    target = look_target(cams)
    print(f"카메라 시선이 모이는 점(모델 좌표) {np.round(target, 3)} / y·z 뒤집으면 {np.round(target * [1, -1, -1], 3)}")
    if a.crop > 0:
        m = np.linalg.norm(xyz - target, axis=1) < a.crop
        print(f"반경 {a.crop} 자르기: {len(xyz):,} -> {m.sum():,}점")
        xyz, rgb = xyz[m], rgb[m]
    home = None
    if a.level:
        n, c = fit_table_plane(xyz, target)
        foot = target - ((target - c) @ n) * n          # 대상 중심을 탁자 면에 내린 점
        R = rot_to(n, np.array([0.0, -1.0, 0.0]))       # 탁자 법선 → 모델의 위쪽(-y)
        tilt = np.degrees(np.arccos(abs(n[1])))
        xyz = ((xyz.astype(np.float64) - foot) @ R.T).astype(np.float32)
        height = -xyz[:, 1]
        radial = np.hypot(xyz[:, 0], xyz[:, 2])
        if a.clean_r > 0:
            m = ~((radial > a.clean_r) & (height > a.clean_h))
            print(f"탁자 기울기 {tilt:.1f}° 바로잡음 / 받침 바깥(반경>{a.clean_r}) 탁자 위 {a.clean_h} 넘는 허공 점 "
                  f"{(~m).sum():,}개 제거 -> {m.sum():,}점")
            xyz, rgb = xyz[m], rgb[m]
        flip = np.array([1, -1, -1])
        cam0 = R @ (cams[0, :3, 3] - foot); tgt = R @ (target - foot)
        home = dict(pos=(cam0 * flip).round(3).tolist(), target=(tgt * flip).round(3).tolist(), tilt_deg=round(tilt, 1))
        print(f"뷰어 HOME_VIEW = {{ pos: {home['pos']}, target: {home['target']} }}")
    if a.n_keep and a.n_keep < len(xyz):
        idx = np.linspace(0, len(xyz) - 1, a.n_keep).astype(np.int64)
        xyz, rgb = xyz[idx], rgb[idx]

    out = Path(a.out); out.mkdir(exist_ok=True)
    t = time.time()
    r, extent, hdr = to_splat_ply(xyz, rgb, out / "scene.ply", a.K, a.opacity)
    t_ply = time.time() - t
    ply_size = (out / "scene.ply").stat().st_size
    n = len(xyz)
    print(f"알갱이 {n:,} / 반지름 {r:.5f} / 장면 긴 변 {extent:.3f} / scene.ply {ply_size:,}B / "
          f"알갱이당 {(ply_size - hdr) / n:.1f}B / 저장 {t_ply:.2f}초")

    rows = [dict(fmt="ply", file="scene.ply", bytes=ply_size, seconds=round(t_ply, 2), count=n)]
    for ext in ["compressed.ply", "sog", "spz"]:
        dst = out / f"scene.{ext}"
        t = time.time()
        subprocess.run(f'npx -y {TOOL} -w "{out / "scene.ply"}" "{dst}"', shell=True, check=True)
        sec = time.time() - t
        rows.append(dict(fmt=ext, file=dst.name, bytes=dst.stat().st_size, seconds=round(sec, 1), count=None))
    # 알갱이 수: compressed.ply 는 헤더에서, sog/spz 는 다시 ply 로 풀어서 센다
    rows[1]["count"] = count_from_header(out / "scene.compressed.ply")
    for row in rows[2:]:
        back = out / f"_back_{row['fmt']}.ply"
        subprocess.run(f'npx -y {TOOL} -w "{out / row["file"]}" "{back}"', shell=True, check=True)
        row["count"] = count_from_header(back)
        back.unlink()
    for row in rows:
        row["bytes_per_splat"] = round(row["bytes"] / row["count"], 2)
        row["ratio_vs_ply"] = round(ply_size / row["bytes"], 1)

    cpos = cams[:, :3, 3]
    spread = float(np.linalg.norm(cpos.max(0) - cpos.min(0)))
    diag = float(np.linalg.norm(d["xyz"].max(0) - d["xyz"].min(0)))
    res = dict(K=a.K, n_keep=a.n_keep, crop=a.crop, target=target.tolist(), level=a.level,
               clean_r=a.clean_r, clean_h=a.clean_h, home_view=home, opacity=a.opacity, radius=r, extent=extent, splats=n,
               spread=spread, diag=diag, spread_ratio=spread / diag, formats=rows)
    json.dump(res, open(out / "results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\n카메라 퍼짐 {spread:.3f} / 장면 대각선 {diag:.3f} = {spread / diag:.4f}")
    print(f"\n| 형식 | 크기(바이트) | MB | 변환(초) | 알갱이 | 바이트/알갱이 | PLY 대비 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for x in rows:
        print(f"| {x['fmt']} | {x['bytes']:,} | {x['bytes'] / 1e6:.2f} | {x['seconds']} | {x['count']:,} | "
              f"{x['bytes_per_splat']} | {x['ratio_vs_ply']}배 |")

    plot_cameras(d["xyz"], d["rgb"], cams, out / "cameras.png")


def plot_cameras(xyz, rgb, cams, path):
    """위에서 본 평면도 + 옆에서 본 그림: 점(일부)과 카메라 위치·바라보는 방향."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xyz), min(60000, len(xyz)), replace=False)
    P, C = xyz[idx], rgb[idx] / 255.0
    cpos = cams[:, :3, 3]
    fwd = cams[:, :3, 2]  # OpenCV 규약: 카메라 +Z 가 바라보는 방향
    L = 0.08 * float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (i, j, name) in zip(axs, [(0, 2, "위에서 본 모습 (x-z)"), (0, 1, "옆에서 본 모습 (x-y)")]):
        ax.scatter(P[:, i], P[:, j], c=C, s=0.3, linewidths=0)
        ax.plot(cpos[:, i], cpos[:, j], "-", color="#ef4444", lw=1)
        ax.scatter(cpos[:, i], cpos[:, j], c=np.arange(len(cpos)), cmap="autumn", s=25, zorder=3)
        for k in range(len(cpos)):
            ax.arrow(cpos[k, i], cpos[k, j], fwd[k, i] * L, fwd[k, j] * L, color="#0ea5e9", width=L * 0.02)
        ax.text(cpos[0, i], cpos[0, j], " 시작", fontsize=9)
        ax.set_aspect("equal"); ax.set_title(name)
        if j == 1: ax.invert_yaxis()  # 모델 좌표는 Y 가 아래
    fig.suptitle(f"카메라 {len(cpos)}곳의 위치(점)와 바라보는 방향(파란 화살표)")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print("카메라 그림:", path)


if __name__ == "__main__":
    main()
