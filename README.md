:warning: This codebase is still under construction :warning:

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]() [![Open Pull Requests](https://img.shields.io/github/issues-pr/ArthurConmy/Automatic-Circuit-Discovery.svg)](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/pulls)

# Automated Circuit DisCovery 

![](assets/acdc_finds_subgraph.png)

This is the accompanying code to the paper "Towards Automated Circuit Discovery for Mechanistic Interpretability".

* :zap: To run ACDC, see `acdc/main.py`, or <a href="https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/main/notebooks/colabs/ACDC_Main_Demo.ipynb">this Colab notebook</a>
* :wrench: To see how edit edges in computational graphs in models, see `notebooks/editing_edges.py` or <a href="https://colab.research.google.com/github/ArthurConmy/Automatic-Circuit-Discovery/blob/main/notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb">this Colab notebook</a>

This library builds upon the abstractions (`HookPoint`s and standardised `HookedTransformer`s) from [TransformerLens](https://github.com/neelnanda-io/TransformerLens) :mag_right:

## Installation:

First, install the system dependencies for either [Mac](#apple-mac-os-x) or [Linux](#penguin-ubuntu-linux).

Then, you need Python 3.8+ and [Poetry](https://python-poetry.org/docs/) to install ACDC, like so

```bash
git clone git+https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git
cd Automatic-Circuit-Discovery
poetry env use 3.10 # Python 3.10 is recommended but use any Python version >= 3.8
poetry install
```

### System Dependencies

#### :penguin: Ubuntu Linux

```bash
sudo apt-get update && sudo apt-get install libgl1-mesa-glx graphviz build-essential graphviz-dev
```

#### :apple: Mac OS X

On Mac, you need to let pip (inside poetry) know about the path to the Graphviz libraries.

```
brew install graphviz
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
```


## Tests

From the root directory, run 

```bash
pytest -vvv -m "not slow"
```

This will only select tests not marked as `slow`. These tests take a _long_ time, and are good to run occasionally, but
not every time.

You can run the slow tests with

``` bash
pytest -s -m slow
```

## Contributing 

We welcome issues where the code is unclear!

If you make a PR, make sure you run 
```bash
chmod +x experiments/make_notebooks.sh
./experiments/make_notebooks.sh
```
And check that no errors arise. It is essential that the notebooks converted here consist only of `#%% [markdown]` markdown-only cells, and `#%%` cells with code.

## Citing ACDC

If you use ACDC, please reach out! You can reference the work as follows:

```
@misc{conmy2023automated,
      title={Towards Automated Circuit Discovery for Mechanistic Interpretability}, 
      author={Arthur Conmy and Augustine N. Mavor-Parker and Aengus Lynch and Stefan Heimersheim and Adri{\`a} Garriga-Alonso},
      year={2023},
      eprint={2304.14997},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## TODO

[ x ] Make `TransformerLens` install be Neel's code not my PR

[ x ] Add `hook_mlp_in` to `TransformerLens` and delete `hook_resid_mid` (and test to ensure no bad things?)

[ x ] Delete `arthur-try-merge-tl` references from the repo

[ ] Fix huge edge sizes in Induction Main example

[ ] Find a better way to deal with the versioning on the Colabs installs...

[ ] Neuron-level experiments

[ ] Position-level experiments

[ ] `tracr` and other dependencies better managed

[ ] Make SP tests work (lots outdated so skipped) - and check SubnetworkProbing installs properly (no __init__.pys !!!)

[ ] Make the 9 tests also failing on TransformerLens-main pass

[ ] Remove Codebase under construction
