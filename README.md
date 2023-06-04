:warning: This codebase is still under construction :warning:

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]() [![Open Pull Requests](https://img.shields.io/github/issues-pr/ArthurConmy/Automatic-Circuit-Discovery.svg)](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/pulls)

# Automated Circuit DisCovery 

This is the accompanying code to the paper "Towards Automated Circuit Discovery for Mechanistic Interpretability".

* :zap: To run ACDC, see `acdc/main.py`, or <a href="https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/arthur-try-merge-tl/notebooks/colabs/ACDC_Main_Demo.ipynb#scrollTo=njv8l86QSPka">this Colab notebook</a>
* :wrench: To see how edit edges in computational graphs in models, see `notebooks/editing_edges.py` or <a href="https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/arthur-try-merge-tl/notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb">this Colab notebook</a>

## Installation:

```bash
pip install git@https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@arthur-acdc-feature-prepend git@https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git@arthur-try-merge-tl 
```

### Dev installation:

```bash
git clone git@github.com:ArthurConmy/Automatic-Circuit-Discovery.git
cd Automatic-Circuit-Discovery
pip install -e .
```

### Details

This codebase currently works on top of the TransformerLens main branch (as of 4th June 2023), so probably `transformer_lens==1.2.3` or later (for now, install TransformerLens from source).

You may need to install DeepMind's `tracr` if you're dealing with that (e.g <a href="https://github.com/deepmind/tracr/commit/e75ecdaec12bf2d831a60e54d4270e8fa31fb537">this commit</a>). This seems to be bugged on Windows. 

You may also need do this

```bash
sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6 graphviz
```

in order to install graphics dependencies on linux.

## Tests

From the root directory, run 

```bash
pytest -vvv
```

## Citing ACDC

If you use ACDC, please reach out! You can reference the work as follows:

```
@misc{conmy2023automated,
      title={Towards Automated Circuit Discovery for Mechanistic Interpretability}, 
      author={Arthur Conmy and Augustine N. Mavor-Parker and Aengus Lynch and Stefan Heimersheim and Adrià Garriga-Alonso},
      year={2023},
      eprint={2304.14997},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## TODO

[ ] Make `TransformerLens` install be Neel's code not my PR

[ ] Delete `arthur-try-merge-tl` references from the repo

[ ] Neuron-level experiments

[ ] Position-level experiments

[ ] `tracr` and other dependencies better managed

[ ] Make SP tests work (lots outdated so skipped) - and check they install (no __init__.pys !!!)

[ ] Make the 9 tests also failing on TransformerLens-main pass