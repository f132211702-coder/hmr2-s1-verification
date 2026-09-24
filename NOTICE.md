# Notice

`src/hmr2/` and `src/vendor_demos/` vendor the official **4D-Humans / HMR2.0**
release, unmodified in structure (only `setup.py`'s dependency list was
edited — the `chumpy` git dependency was moved out of `install_requires`):

- Source: https://github.com/shubham-goel/4D-Humans
- Commit: `efe18deff163b29dff87ddbd575fa29b716a356c` (branch `main`)
- Paper: Goel, Pavlakos, Rajasegaran, Kanazawa, Malik. *"Humans in 4D:
  Reconstructing and Tracking Humans with Transformers."* ICCV 2023.
- License: see `LICENSE.md`

The pretrained model is used as-is for inference (S1 of this pipeline: RGB
image → SMPL parameters); it is **not retrained** here. Vendor's own
documentation is kept at `src/hmr2/UPSTREAM_README.md` for reference.

`src/hmr2/` still contains training-only code (`models/discriminator.py`,
`models/losses.py`, `datasets/dataset.py`, `datasets/image_dataset.py`,
`datasets/mocap_dataset.py`, ...) — these could not be safely deleted:
`HMR2.__init__` in `models/hmr2.py` unconditionally constructs a
`Discriminator` and the loss modules, and `datasets/__init__.py`
unconditionally imports the training dataset classes, so removing those
files breaks `import hmr2` entirely, including plain inference. What *was*
removed because nothing in the inference path imports it: the top-level
`train.py` / `fetch_training_data.sh` entry points and the Hydra training
configs (`hmr2/configs_hydra/`).
