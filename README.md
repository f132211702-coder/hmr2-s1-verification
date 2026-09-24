# S1: Body Geometry Estimation (HMR 2.0)

Stage S1 of a 3D virtual try-on pipeline: estimate SMPL body parameters from
an RGB image. S2–S4 (garment fitting/draping/scoring) are a teammate's
responsibility and out of scope for this repo.

## Input / Output / Model

| | |
|---|---|
| **Input** | one RGB image (a photo of a person) |
| **Output** | per detected person: `betas` (10,), `body_pose` (23,3,3), `global_orient` (1,3,3), `cam_t` (3,) — the SMPL shape/pose/camera parameters |
| **Model** | the official pretrained **HMR 2.0** checkpoint ([4D-Humans](https://github.com/shubham-goel/4D-Humans), ICCV 2023) — used as-is, **not retrained**. See [NOTICE.md](NOTICE.md) |

Everything else in this repo exists to serve that one input→output contract.
You should not need to read past this section unless you're debugging a
specific estimate or extending the pipeline.

## Quickstart

```bash
conda env create -f configs/environment.yml
conda activate 4D-humans
pip install -e .
bash scripts/download_checkpoint.sh
```

The SMPL neutral body model can't be redistributed here (license
restriction): register at https://smplify.is.tue.mpg.de/ and place it at
`~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl`.

```bash
python s1_infer.py --img path/to/image.jpg --out results/s1_raw
```

Or as a library:

```python
from s1_infer import HMR2Estimator
est = HMR2Estimator()
results = est.estimate(image_bgr)  # BGR numpy array, e.g. cv2.imread()
```

Batch mode, for running a whole dataset (e.g. as input to a quantitative
evaluation):

```bash
python s1_infer.py --img_folder /path/to/images --out results/s1_raw
```

## Quantitative evaluation (3DPW / CloSe-Di)

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

## Debugging a single estimate

`s1_infer.py` only writes `.npz` files — no images. To visually sanity-check
a result, run the separate visualization tool on top of its output (this
does not reload the model):

```bash
python tools/visualize.py --img path/to/image.jpg --pred_dir results/s1_raw --out results/s1_viz
```

If a leg pose looks off (a known HMR2 failure mode on unusual poses — see
`tools/refine_leg_pose.py`'s module docstring), there's an optional
ViTPose-based post-processing step:

```bash
python tools/refine_leg_pose.py --img path/to/image.jpg --s1_out results/s1_raw --out results/s1_refined
```

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
