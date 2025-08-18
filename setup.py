from setuptools import setup, find_packages

setup(
    name="geneformer",
    version="0.1.0",
    author="Christina Theodoris",
    author_email="christina.theodoris@gladstone.ucsf.edu",
    description="Geneformer is a transformer model pretrained \
                 on a large-scale corpus of single \
                 cell transcriptomes to enable context-aware \
                 predictions in settings with limited data in \
                 network biology.",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=[
        "anndata",
        "bitsandbytes",
        "datasets",
        "loompy",
        "matplotlib",
        "numpy",
        "optuna",
        "optuna-integration",
        "packaging",
        "pandas",
        "peft",
        "pyarrow",
        "pytz",
        "ray",
        "scanpy",
        "scikit-learn",
        "scipy",
        "seaborn",
        "setuptools",
        "statsmodels",
        "tdigest",
        "tensorboard",
        "torch",
        "tqdm",
        "transformers",
    ],
)
