#!/usr/bin/env python3
"""Render a CloSe-Di 3D scan (.npz) into a plain RGB image, so it can be fed
into s1_infer.py the same way a real photo would be.

CloSe-Di ships 3D scan geometry (points/faces/per-point colors) and SMPL
parameters, but no photographs -- this script is the piece that turns a
scan into something HMR2 can actually look at.

Why a hand-rolled software rasterizer instead of a real renderer: PyTorch3D
was the obvious choice (the official CloSe repo already uses it, see
lib/utils/viz.py's render_p3d()), but the school server's PyTorch (2.5.1)
is newer than PyTorch3D's officially supported range (2.1.0-2.4.1), and
precompiled wheels for unsupported PyTorch versions are a known way to get
an unrecoverable `ImportError: undefined symbol` (see
https://github.com/facebookresearch/pytorch3d/issues/2013 -- someone hit
exactly that on an out-of-range PyTorch version, unresolved). Rather than
gamble a source build against an untested PyTorch version, this reuses the
same painter's-algorithm approach tools/visualize.py already has working
for SMPL meshes (see render_mesh_overlay_solid() there) -- pure numpy +
OpenCV, no compiled renderer dependency at all. The one real difference is
shading each triangle with the *scan's own* per-vertex colors instead of a
fixed mannequin color.

Status: written and validated against synthetic data (see --self-test).
NOT yet run against a real CloSe-Di .npz -- field names/shapes and the
color encoding (0-1 float vs 0-255? RGB vs BGR?) are assumed from the
pipeline plan doc's section 3.2 and need confirming against a real file
(`np.load(path).files`, check `colors.max()`) before trusting the output.

This only produces a clean, unoccluded render. Turning a set of these into
the "no/partial/severe occlusion" groups EXP-002 needs is a separate,
later step (e.g. masking a region of the output image) -- not handled
here.

Usage (self-test, no real data needed):
    python data_prep/render_close_di_scan.py --self-test

Usage (a single scan):
    python data_prep/render_close_di_scan.py --scan path/to/scan.npz --out path/to/render.jpg

Usage (batch, a folder of scans):
    python data_prep/render_close_di_scan.py --scan_dir path/to/CloSe-Di --out_dir results/close_di_renders
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rotate around the Y axis (azimuth: turning left/right) then the X
    axis (elevation: tilting up/down). Returns a (3,3) matrix."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cos_a, sin_a = np.cos(az), np.sin(az)
    cos_e, sin_e = np.cos(el), np.sin(el)
    rot_y = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a],
    ])
    rot_x = np.array([
        [1, 0, 0],
        [0, cos_e, -sin_e],
        [0, sin_e, cos_e],
    ])
    return rot_x @ rot_y


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    """CloSe-Di's exact color encoding hasn't been confirmed against a real
    file yet -- handle both plausible conventions (0-1 float, or 0-255).
    Returns float RGB in [0, 1]."""
    colors = colors.astype(np.float64)
    if colors.max() > 1.0 + 1e-6:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def render_scan(points: np.ndarray, faces: np.ndarray, colors: np.ndarray,
                 resolution: int = 512, azimuth: float = 0.0, elevation: float = 0.0,
                 background: tuple[int, int, int] = (255, 255, 255),
                 margin: float = 0.88) -> np.ndarray:
    """Software-rasterize a colored 3D point cloud + its triangulation into
    a single BGR image (OpenCV convention). Orthographic projection,
    auto-fit to the canvas from the scan's own bounding box -- no camera
    parameters to hand-tune per scan. No external renderer needed.

    points:  (N,3) scan vertices
    faces:   (F,3) triangle indices into points
    colors:  (N,3) per-vertex RGB, either 0-1 or 0-255 range
    azimuth / elevation: degrees, rotate the scan before projecting (e.g.
        for rendering several viewpoints of the same scan)
    margin: fraction of the canvas the scan's bounding box should fill
    """
    import cv2

    colors01 = _normalize_colors(colors)
    rot = _rotation_matrix(azimuth, elevation)
    rotated = points @ rot.T  # (N,3)

    # Orthographic: screen position only depends on rotated X/Y; rotated Z
    # is kept purely as a depth value for painter's-algorithm sorting, not
    # used for scale (that's what makes this orthographic, not perspective).
    xy = rotated[:, :2]
    span = (xy.max(axis=0) - xy.min(axis=0)).max()
    if span <= 0:
        span = 1.0
    scale = (resolution * margin) / span
    center_xy = (xy.max(axis=0) + xy.min(axis=0)) / 2.0

    screen = (xy - center_xy[None, :]) * scale
    screen[:, 0] += resolution / 2.0
    # Image row 0 is the top, but the scan's Y axis conventionally points
    # up, so flip to match.
    screen[:, 1] = resolution / 2.0 - screen[:, 1]

    tri_xyz = rotated[faces]      # (F,3,3), for normals/depth
    tri_screen = screen[faces]    # (F,3,2), for drawing
    tri_colors = colors01[faces]  # (F,3,3)

    normals = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    # Orthographic camera looking down +Z from Z=-infinity: unlike a
    # perspective camera, the to-camera direction is the same constant
    # vector for every point (that's the defining property of parallel
    # projection), so no per-face view-direction computation is needed.
    intensity = -normals[:, 2]
    depth = tri_xyz.mean(axis=1)[:, 2]
    face_color = tri_colors.mean(axis=1)  # (F,3), flat-shaded per triangle

    # Backface culling: keep only triangles facing the camera, otherwise
    # the inside of the scan (back of the body) bleeds through.
    front_mask = intensity > 0.05
    depth, tri_screen, face_color, intensity = (
        depth[front_mask], tri_screen[front_mask],
        face_color[front_mask], intensity[front_mask],
    )

    canvas = np.full((resolution, resolution, 3), background[::-1], dtype=np.uint8)  # BGR
    if len(depth) == 0:
        return canvas

    order = np.argsort(-depth)  # far to near (painter's algorithm)
    tri_screen_int = np.round(tri_screen).astype(np.int32)
    shade = np.clip(0.4 + 0.6 * intensity, 0.0, 1.0)  # keep a floor so nothing goes pure black

    for idx in order:
        color_bgr = tuple(float(c) * 255.0 * shade[idx] for c in face_color[idx][::-1])  # RGB -> BGR
        cv2.fillConvexPoly(canvas, tri_screen_int[idx], color_bgr)

    return canvas


def render_scan_npz(npz_path: Path, resolution: int = 512, azimuth: float = 0.0,
                     elevation: float = 0.0) -> np.ndarray:
    """Load a CloSe-Di scan .npz and render it. Field names per the
    pipeline plan doc's section 3.2 -- not yet verified against a real
    file, see this module's docstring."""
    data = np.load(npz_path)
    return render_scan(data["points"], data["faces"], data["colors"],
                        resolution=resolution, azimuth=azimuth, elevation=elevation)


def self_test() -> None:
    """Validate the rendering math against a synthetic colored cube -- no
    real CloSe-Di data needed."""
    import cv2

    points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # back  (z=-1)
        [4, 6, 5], [4, 7, 6],  # front (z=+1)
        [0, 4, 5], [0, 5, 1],  # bottom
        [3, 2, 6], [3, 6, 7],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ])
    # A distinct color per vertex, so each visible face averages to
    # something recognizably different -- lets us sanity-check by eye that
    # rotating the camera actually shows different faces/colors.
    colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
        [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.2, 0.2, 0.2],
    ], dtype=np.float64)

    img_front = render_scan(points, faces, colors, resolution=256, azimuth=0, elevation=0)
    img_side = render_scan(points, faces, colors, resolution=256, azimuth=90, elevation=0)

    assert img_front.shape == (256, 256, 3)
    assert img_front.std() > 5, "rendered image has almost no variation; projection/scaling is probably broken"
    assert not np.array_equal(img_front, img_side), "rotating the camera produced an identical image"

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "render_self_test_front.png"), img_front)
    cv2.imwrite(str(out_dir / "render_self_test_side.png"), img_side)
    print("[self-test] all checks passed.")
    print(f"[self-test] sample renders written to {out_dir}/render_self_test_front.png "
          f"and {out_dir}/render_self_test_side.png -- open them and confirm you see "
          f"a colored cube from two different angles.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                     help="validate the rendering math against a synthetic cube; needs no real data")
    ap.add_argument("--scan", type=str, help="path to a single CloSe-Di .npz scan")
    ap.add_argument("--scan_dir", type=str, help="batch mode: render every *.npz in this folder")
    ap.add_argument("--out", type=str, help="output image path (single-scan mode)")
    ap.add_argument("--out_dir", type=str, help="output folder (batch mode)")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--azimuth", type=float, default=0.0, help="degrees, rotate around the vertical axis")
    ap.add_argument("--elevation", type=float, default=0.0, help="degrees, tilt up/down")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    import cv2

    if args.scan:
        if not args.out:
            raise SystemExit("--scan needs --out")
        img = render_scan_npz(Path(args.scan), resolution=args.resolution,
                               azimuth=args.azimuth, elevation=args.elevation)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path}")
    elif args.scan_dir:
        if not args.out_dir:
            raise SystemExit("--scan_dir needs --out_dir")
        scan_dir = Path(args.scan_dir)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scan_paths = sorted(scan_dir.glob("*.npz"))
        print(f"Rendering {len(scan_paths)} scan(s)...")
        for scan_path in scan_paths:
            img = render_scan_npz(scan_path, resolution=args.resolution,
                                   azimuth=args.azimuth, elevation=args.elevation)
            cv2.imwrite(str(out_dir / f"{scan_path.stem}.jpg"), img)
        print(f"Done. Output written to {out_dir}")
    else:
        raise SystemExit("Provide --scan, --scan_dir, or --self-test")


if __name__ == "__main__":
    main()
