from setuptools import setup, find_packages

# Package source lives under src/ (vendored 4D-Humans / HMR2.0), so
# find_packages() must look there instead of the repo root.
setup(
    description='HMR2 as a package',
    name='hmr2',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'gdown',
        'numpy',
        'torch',
        'torchvision',
        'pytorch-lightning',
        'smplx==0.1.28',
        'pyrender',
        'opencv-python',
        'yacs',
        'omegaconf',  # needed to unpickle the official checkpoint's hparams, even without detectron2
        'scikit-image',
        'einops',
        'timm',
        'webdataset',
        'dill',
        'pandas',
    ],
    extras_require={
        'all': [
            'detectron2 @ git+https://github.com/facebookresearch/detectron2',
        ],
        # s1_infer.py's optional --gender-aware flag. Not in install_requires
        # since it pulls in TensorFlow, which nothing else here needs.
        'gender': [
            'deepface',
            'tf-keras',  # deepface refuses to import under tensorflow>=2.16 (Keras 3) without it
        ],
    },
)
