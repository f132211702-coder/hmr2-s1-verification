# S1: Body Geometry Estimation (HMR 2.0)

## What this is

This is stage **S1** of a 3D virtual try-on (VTO) pipeline being built as a
course project: given a photo of a person, recover their 3D body shape and
pose as SMPL parameters. Stages S2–S4 (garment fitting, draping, and
fit-quality scoring) are a teammate's responsibility and out of scope for
this repo; this repo's output is exactly the input those stages need.

## What problem this solves

Turning a flat photo into a usable 3D body is the first hard step of any
image-based virtual try-on system — without it, there's no body geometry to
fit a garment mesh to. Rather than training a new model, this stage wraps
the existing state-of-the-art pretrained **HMR 2.0** model
([4D-Humans](https://github.com/shubham-goel/4D-Humans), ICCV 2023) behind
a clean, minimal CLI/library interface, and adds the quantitative
evaluation (does HMR2.0's output actually hold up against ground-truth SMPL
from public datasets?) and pose-correction tooling needed to trust that
output downstream.

## Input / Output / Model

| | |
|---|---|
| **Input** | one RGB image (a photo of a person) |
| **Output** | per detected person: `betas` (10,), `body_pose` (23,3,3), `global_orient` (1,3,3), `cam_t` (3,) — the SMPL shape/pose/camera parameters |
| **Model** | the official pretrained **HMR 2.0** checkpoint ([4D-Humans](https://github.com/shubham-goel/4D-Humans), ICCV 2023) — used as-is, **not retrained**. See [NOTICE.md](NOTICE.md) |

Everything else in this repo exists to serve that one input→output contract.
You should not need to read past this section unless you're debugging a
specific estimate or extending the pipeline.

## Features

- **`s1_infer.py`** — single-image or batch-folder SMPL estimation, as a CLI or an importable `HMR2Estimator` class
- **`tools/visualize.py`** — render detection boxes / mesh overlays / `.obj` exports from a saved result, without reloading the model
- **`tools/refine_leg_pose.py`** — optional ViTPose-based post-processing that corrects a known HMR2 leg-pose failure mode
- **`eval/eval_against_gt.py`** — MPJPE / PA-MPJPE / beta-error evaluation against a dataset's ground-truth SMPL (3DPW, CloSe-Di)
- **`scripts/download_checkpoint.sh`** — one-line fetch of the pretrained checkpoint

## Install

```bash
conda env create -f configs/environment.yml
conda activate 4D-humans
pip install -e .
bash scripts/download_checkpoint.sh
```

The SMPL neutral body model can't be redistributed here (license
restriction): register at https://smplify.is.tue.mpg.de/ and place it at
`~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl`.

### Known install issue: `chumpy`

`conda`'s pip section doesn't reliably install `git+https://...` dependencies
(it can silently skip them), so `chumpy` — needed by `smplx` to read
`SMPL_NEUTRAL.pkl` — sometimes ends up missing after `conda env create`.
`pip install`ing it also fails on its own with a `ModuleNotFoundError: No
module named 'pip'` error, because chumpy's old-style `setup.py` does
`import pip` and pip's build isolation doesn't include pip itself. Fix:

```bash
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | optional HuggingFace access token; avoids the "unauthenticated requests" rate limit when downloading the RT-DETR/ViTPose models | unset (unauthenticated, rate-limited) |

The checkpoint/SMPL cache directory (`~/.cache/4DHumans`) is **not**
configurable via an environment variable — it's hardcoded in the vendored
package (`src/hmr2/configs/__init__.py`), so `scripts/download_checkpoint.sh`
matches that path exactly rather than exposing a variable that the Python
code wouldn't actually honor.

## Usage

Single image:

```bash
python s1_infer.py --img path/to/image.jpg --out results/s1_raw
```

Batch mode, for running a whole dataset (e.g. as input to a quantitative evaluation):

```bash
python s1_infer.py --img_folder /path/to/images --out results/s1_raw
```

As a library:

```python
from s1_infer import HMR2Estimator
est = HMR2Estimator()
results = est.estimate(image_bgr)  # BGR numpy array, e.g. cv2.imread()
```

Debugging a single estimate — `s1_infer.py` only writes `.npz` files, no
images. To visually sanity-check a result, run the separate visualization
tool on top of its output (this does not reload the model):

```bash
python tools/visualize.py --img path/to/image.jpg --pred_dir results/s1_raw --out results/s1_viz
```

If a leg pose looks off (a known HMR2 failure mode on unusual poses — see
`tools/refine_leg_pose.py`'s module docstring), there's an optional
ViTPose-based post-processing step:

```bash
python tools/refine_leg_pose.py --img path/to/image.jpg --s1_out results/s1_raw --out results/s1_refined
```

Quantitative evaluation against a dataset's ground-truth SMPL:

```bash
# validates the error math against synthetic data, no dataset needed
python eval/eval_against_gt.py --self-test

# once a dataset is downloaded and s1_infer.py has produced --pred_dir
python eval/eval_against_gt.py --dataset 3dpw \
    --pred_dir results/s1_raw_3dpw --gt_dir /path/to/3DPW/sequenceFiles/test \
    --out results/eval_3dpw.csv
```

The MPJPE/PA-MPJPE/beta-error math is implemented and self-tested; the
per-dataset ground-truth loaders are still a skeleton pending real data —
see the TODOs in `eval/eval_against_gt.py`'s module docstring.

## Layout

```
s1_infer.py              the CLI / library entry point (input → output, above)
tools/
  visualize.py            optional: render/export a saved result for inspection
  refine_leg_pose.py       optional: post-hoc leg-pose correction
eval/
  eval_against_gt.py       our MPJPE/PA-MPJPE/beta-error evaluation
  official_eval.py         upstream's own eval.py, kept as a cross-check
scripts/
  download_checkpoint.sh   one-line checkpoint fetch
configs/
  environment.yml          conda environment (training-only deps removed)
src/
  hmr2/                    vendored model package — internal, see NOTICE.md
  vendor_demos/            upstream's demo.py / track.py / gradio_app.py, kept but not part of this pipeline
```

## License

MIT — see [LICENSE.md](LICENSE.md). This repo's own code
(`s1_infer.py`/`tools/`/`eval/`/`scripts/`) and the vendored 4D-Humans code
under `src/` are each MIT-licensed under separate copyright; see
[NOTICE.md](NOTICE.md) for the vendored code's provenance and exact terms.
