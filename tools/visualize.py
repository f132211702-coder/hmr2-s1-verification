#!/usr/bin/env python3
"""Debug visualizations for S1 output: draws detection boxes, projects the
predicted SMPL mesh back onto the image, and exports .obj/.mtl files viewable
in any 3D viewer.

This reads the .npz files s1_infer.py already produced — it does not reload
the HMR2 model or run the detector again, so it stays fast even over a large
saved result set. It does load the SMPL mesh topology (faces), which is a
fixed constant (independent of pose/shape/the transformer network) and cheap
to load on its own.

Usage:
    python tools/visualize.py --img path/to/image.jpg --pred_dir results/s1_raw --out results/s1_viz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # smplx's SMPL loader goes through pickle under the hood; keep the same
    # weights_only=False override used everywhere else in this repo so it
    # behaves consistently regardless of the installed pytorch-lightning version.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)  # same body color as the official demo.py


def get_smpl_faces() -> np.ndarray:
    """The SMPL triangle topology (6890 vertices), identical for every body
    shape/pose. Loading just this via smplx is cheap and does not touch the
    HMR2 transformer network."""
    import smplx
    from hmr2.configs import CACHE_DIR_4DHUMANS
    smpl_path = Path(CACHE_DIR_4DHUMANS) / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    return smplx.SMPLLayer(model_path=str(smpl_path), num_betas=10).faces


def load_predictions(pred_dir: Path, image_stem: str) -> list[dict]:
    """Load every `<image_stem>_<person_id>_smpl_params.npz` s1_infer.py
    produced for this image."""
    people = []
    for npz_path in sorted(pred_dir.glob(f"{image_stem}_*_smpl_params.npz")):
        data = np.load(npz_path)
        person_id = int(npz_path.stem.rsplit("_", 3)[-3])
        people.append({
            "person_id": person_id,
            "pred_vertices": data["pred_vertices"],
            "cam_t": data["cam_t"],
            "scaled_focal_length": float(data["scaled_focal_length"]),
            "bbox": data["bbox"],
        })
    return people


def project_vertices_to_image(vertices: np.ndarray, cam_t: np.ndarray,
                               focal_length: float, img_w: int, img_h: int) -> np.ndarray:
    """Project SMPL vertices (model space, camera at the origin looking +Z)
    to full-image 2D pixel coordinates. Pure numpy, mirrors
    hmr2/utils/geometry.py's perspective_projection() (identity rotation,
    camera_center = image center) without needing pyrender/OpenGL."""
    cam_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float64)
    pts = vertices.astype(np.float64) + cam_t.astype(np.float64)[None, :]
    projected = pts[:, :2] / pts[:, 2:3]
    projected = projected * focal_length + cam_center[None, :]
    return projected


def draw_boxes(img_bgr: np.ndarray, people: list[dict]) -> np.ndarray:
    import cv2
    viz = img_bgr.copy()
    for p in people:
        x1, y1, x2, y2 = [int(round(v)) for v in p["bbox"]]
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(viz, str(p["person_id"]), (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return viz


def draw_mesh_overlay(img_bgr: np.ndarray, people: list[dict]) -> np.ndarray:
    """Project each person's SMPL vertices back onto the image as a point
    cloud — a quick visual check of whether the estimated pose/shape lines
    up with the photo."""
    img_h, img_w = img_bgr.shape[:2]
    viz = img_bgr.copy()
    palette = [
        (0, 255, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (255, 0, 0), (128, 0, 255), (0, 200, 100),
    ]
    for p in people:
        color = palette[p["person_id"] % len(palette)]
        pts = project_vertices_to_image(
            p["pred_vertices"], p["cam_t"], p["scaled_focal_length"], img_w, img_h
        )
        x = np.clip(np.round(pts[:, 0]).astype(int), 0, img_w - 1)
        y = np.clip(np.round(pts[:, 1]).astype(int), 0, img_h - 1)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx = np.clip(x + dx, 0, img_w - 1)
                yy = np.clip(y + dy, 0, img_h - 1)
                viz[yy, xx] = color
    return viz


def render_mesh_overlay_solid(img_bgr: np.ndarray, people: list[dict], faces: np.ndarray) -> np.ndarray:
    """Software-rasterize each person's mesh as a solid pale mannequin over
    the original image (same visual effect as the official demo.py's overlay
    render) via a painter's-algorithm triangle sort — no pyrender/OpenGL
    needed. All people's triangles are depth-sorted together so overlapping
    people occlude each other correctly."""
    import cv2

    img_h, img_w = img_bgr.shape[:2]
    viz = img_bgr.copy()
    base_color_bgr = np.array([235.0, 233.0, 230.0])

    all_depths, all_screen_tris, all_intensity = [], [], []
    cam_center = np.array([img_w / 2.0, img_h / 2.0])
    for p in people:
        cam_verts = p["pred_vertices"].astype(np.float64) + p["cam_t"].astype(np.float64)[None, :]
        projected = cam_verts[:, :2] / cam_verts[:, 2:3] * p["scaled_focal_length"] + cam_center[None, :]

        tri_cam = cam_verts[faces]        # (F,3,3) camera space, for normals/depth
        tri_screen = projected[faces]     # (F,3,2) screen pixels, for drawing

        normals = np.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0])
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        normals = normals / norm_len

        centroid = tri_cam.mean(axis=1)
        centroid_len = np.linalg.norm(centroid, axis=1, keepdims=True)
        centroid_len[centroid_len == 0] = 1.0
        view_dir = -centroid / centroid_len  # face-centroid-to-camera direction

        intensity = np.sum(normals * view_dir, axis=1)
        depth = centroid[:, 2]

        # Backface culling: keep only front-facing triangles, otherwise the
        # solid mannequin would look like a translucent wireframe.
        front_mask = intensity > 0.05
        all_depths.append(depth[front_mask])
        all_screen_tris.append(tri_screen[front_mask])
        all_intensity.append(np.clip(intensity[front_mask], 0.0, 1.0))

    if not all_depths:
        return viz

    depths = np.concatenate(all_depths)
    screen_tris = np.concatenate(all_screen_tris)
    intensities = np.concatenate(all_intensity)

    # Painter's algorithm: draw far-to-near so nearer triangles correctly
    # cover farther ones.
    order = np.argsort(-depths)
    screen_tris_int = np.round(screen_tris).astype(np.int32)

    for idx in order:
        shade = 0.35 + 0.65 * intensities[idx]
        color = tuple(float(c) * shade for c in base_color_bgr)
        cv2.fillConvexPoly(viz, screen_tris_int[idx], color)

    return viz


def render_mesh_shaded(vertices: np.ndarray, faces: np.ndarray, out_path: Path,
                        elev: float = -90.0, azim: float = -90.0) -> None:
    """Render the mesh in the same pale-blue, flat-shaded style as the
    official demo.py, using matplotlib's Poly3DCollection instead of
    pyrender/OpenGL — works on machines without a GPU/display too. The
    light source is a headlight fixed to the camera direction, so whichever
    face points at the viewer is always lit, at any elev/azim."""
    tris = vertices[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    elev_r, azim_r = np.radians(elev), np.radians(azim)
    cam_dir = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ])
    intensity = np.clip(normals @ cam_dir, 0.2, 1.0)
    colors = np.clip(np.array(LIGHT_BLUE)[None, :] * intensity[:, None], 0, 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(4, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.add_collection3d(Poly3DCollection(tris, facecolor=colors, edgecolor=None, linewidths=0))

    center = vertices.mean(0)
    span = (vertices.max(0) - vertices.min(0)).max() / 2 * 1.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, facecolor="white")
    plt.close(fig)


def write_obj(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    """Write vertices + the fixed SMPL triangle topology as a plain
    Wavefront .obj (+ matching .mtl), viewable in Blender/MeshLab/any OS's
    built-in 3D viewer. .obj indices are 1-based; SMPL's faces are 0-based."""
    mtl_name = out_path.stem + ".mtl"

    with open(out_path, "w") as f:
        f.write("# exported by tools/visualize.py (S1 pipeline)\n")
        f.write(f"mtllib {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("usemtl smpl_body\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    mtl_path = out_path.parent / mtl_name
    with open(mtl_path, "w") as f:
        f.write("# exported by tools/visualize.py (S1 pipeline)\n")
        f.write("newmtl smpl_body\n")
        f.write(f"Kd {LIGHT_BLUE[0]:.6f} {LIGHT_BLUE[1]:.6f} {LIGHT_BLUE[2]:.6f}\n")
        f.write("Ka 0.0 0.0 0.0\n")
        f.write("Ks 0.1 0.1 0.1\n")
        f.write("Ns 10.0\n")
        f.write("d 1.0\n")


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True, help="s1_infer.py's --out folder")
    ap.add_argument("--id-prefix", type=str, default="",
                     help="must match whatever --id-prefix was passed to s1_infer.py when producing "
                          "--pred_dir (e.g. '<sequence_name>__'), so this can find the right npz files.")
    ap.add_argument("--out", type=str, default="results/s1_viz")
    ap.add_argument("--no-boxes", action="store_true")
    ap.add_argument("--no-mesh-overlay", action="store_true")
    ap.add_argument("--no-mesh-solid", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-obj", action="store_true")
    args = ap.parse_args()

    img_path = Path(args.img)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Must match the image_id s1_infer.py actually saved its npz files
    # under — if that run used --id-prefix (e.g. batching a dataset one
    # sequence at a time), this has to be given the same prefix, or
    # load_predictions() won't find anything.
    image_id = args.id_prefix + img_path.stem

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"Could not read image {img_path}")

    people = load_predictions(pred_dir, image_id)
    if not people:
        raise SystemExit(f"No predictions found in {pred_dir} for {image_id}")
    print(f"{img_path.name}: {len(people)} prediction(s) loaded from {pred_dir}")

    if not args.no_boxes:
        cv2.imwrite(str(out_dir / f"{image_id}_boxes.jpg"), draw_boxes(img_bgr, people))

    if not args.no_mesh_overlay:
        cv2.imwrite(str(out_dir / f"{image_id}_mesh_overlay.jpg"), draw_mesh_overlay(img_bgr, people))

    faces = None
    if not args.no_mesh_solid or not args.no_render or not args.no_obj:
        faces = get_smpl_faces()

    if not args.no_mesh_solid:
        solid = render_mesh_overlay_solid(img_bgr, people, faces)
        divider = np.full((img_bgr.shape[0], 4, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(out_dir / f"{image_id}_mesh_solid.jpg"), cv2.hconcat([img_bgr, divider, solid]))

    for p in people:
        if not args.no_render:
            render_mesh_shaded(p["pred_vertices"], faces,
                                out_dir / f"{image_id}_{p['person_id']}_render.png")
        if not args.no_obj:
            write_obj(p["pred_vertices"], faces,
                      out_dir / f"{image_id}_{p['person_id']}_smpl_params.obj")

    print(f"Done. Output written to {out_dir}")


if __name__ == "__main__":
    main()
