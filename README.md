<!-- :warning: This codebase is still under construction :warning: -->

This is the accompanying code to the arXiv paper "Towards Automated Circuit Discovery for Mechanistic Interpretability".

To run ACDC, see `acdc/main.py`. To see how we edit connections, see `notebooks/evaluating_subgraphs.py`. This repo is still under construction.

# Automatic Circuit Discovery 

## Installation:

```bash
git clone https://github.com/ArthurConmy/Automatic-Circuit-Discovery
cd Automatic-Circuit-Discovery
pip install -e .
```

You may need to install DeepMind's `tracr` if you're dealing with that and you may also need do this

```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6
```

in order to install graphics dependencies on linux.

## Tests (not currently mantained!)

From the root directory, run 

```bash
pytest tests/acdc -vv
```
