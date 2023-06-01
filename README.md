<!-- :warning: This codebase is still under construction :warning: -->

This is the accompanying code to the arXiv paper "Towards Automated Circuit Discovery for Mechanistic Interpretability".

To run ACDC, see `acdc/main.py`. To see how we edit connections, see `notebooks/evaluating_subgraphs.py`. This repo is still under construction.

# Automatic Circuit Discovery 

## Installation:

```bash
git clone https://github.com/ArthurConmy/Automatic-Circuit-Discovery
cd Automatic-Circuit-Discovery
pip install -e .
pip install -r requirement.txt
```

Installing `requirements.txt` include installing DeepMind's `tracr` (from <a href="https://github.com/deepmind/tracr/commit/e75ecdaec12bf2d831a60e54d4270e8fa31fb537">this commit</a>). This seems to be bugged on Windows - you may need to delete this from `requirements.txt` and install again

You may also need do this
```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 graphviz
```
in order to install graphics dependencies on linux.

## Tests (not currently mantained!)

From the root directory, run 

```bash
pytest tests/acdc -vv
```

## TODO

[ ] Neuron-level experiments
[ ] Position-level experiments