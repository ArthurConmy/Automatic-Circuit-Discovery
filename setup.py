from setuptools import setup, find_packages

setup(
    name="acdc",
    version="0.1.0",
    packages=find_packages(),
    license="LICENSE",
    description="ACDC: Automatic Circuit DisCovery implementation on top of TransformerLens",
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
        "torchtyping", # TODO doesn't install?
        "huggingface_hub",
        "cmapy", # TODO doesn't install?
        "graphviz", 
        "networkx",
        "rich",
        # "git+https://github.com/deepmind/tracr.git", # sad does not work; from setup.py : (
        "rich",
        "accelerate",
        "typing-extensions",
        "pydot",
    ],
    extras_require={
        "dev": ["pytest", "mypy", "pytest-cov"],
    },
)
