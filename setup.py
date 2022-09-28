from setuptools import setup

setup(
    name="geneformer",
    version="0.0.1",
    author="Christina Theodoris",
    author_email="christina.theodoris@gladstone.ucsf.edu",
    description="Geneformer is a transformer model pretrained \
                 on a large-scale corpus of ~30 million single \
                 cell transcriptomes to enable context-aware \
                 predictions in settings with limited data in \
                 network biology.",
    packages=["geneformer"],
    include_package_data=True,
    install_requires=[
        "datasets",
        "loompy",
        "numpy",
        "transformers",
    ],
)
