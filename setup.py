from setuptools import setup

setup(
    name="acdc",
    version="0.1.0",
    packages=["acdc", "subnetwork_probing"],
    license="LICENSE",
    description="ACDC: built on top of TransformerLens: an implementation of transformers tailored for mechanistic interpretability.",
    long_description=open("README.md").read(),
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
        "huggingface_hub",
        "cmapy",
        "graphviz", 
        "matplotlib",
        "kaleido",
        "plotly",
        "torchtyping",
        "cmapy",
        "networkx",
        "rich",
        # "git+https://github.com/deepmind/tracr.git", # sad does not work; pip deprecated
    ],
    extras_require={
        "dev": ["pytest", "mypy", "pytest-cov"],
    },
)
