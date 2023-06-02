<!-- :warning: This codebase is still under construction :warning: -->

This is the accompanying code to the arXiv paper "Towards Automated Circuit Discovery for Mechanistic Interpretability".

To run ACDC, see `transformer_lens/main.py`. To see how we edit connections, see `notebooks/evaluating_subgraphs.py`. <b>This repo is still under construction.</b>

# Automatic Circuit Discovery 

## Installation:

```bash
git clone https://github.com/ArthurConmy/Automatic-Circuit-Discovery
cd Automatic-Circuit-Discovery
pip install -e .
```

You may need to install DeepMind's `tracr` if you're dealing with that (e.g <a href="https://github.com/deepmind/tracr/commit/e75ecdaec12bf2d831a60e54d4270e8fa31fb537">this commit</a>). This seems to be bugged on Windows. 

You may also need do this

```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 graphviz
```

in order to install graphics dependencies on linux.

## Tests

From the root directory, run 

```bash
pytest tests/acdc -vvv --ignore=subnetwork_probing/ --ignore=tests/subnetwork_probing/
```

(currently working on getting these passing)

## TODO

[ ] Neuron-level experiments

[ ] Position-level experiments

[ ] `tracr` and other dependencies better managed

[ ] Make SP tests work

[ ] Add 16H back