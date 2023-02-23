Public rust circuit version of ACDC.

# Installation instructions:

1. Install `rust_circuit_public` following closely the instructions in https://github.com/ArthurConmy/rust_circuit_public (this will take ~20GB of space on a fresh machine. Initially, ignore warnings. Use ChatGPT to resolve errors)

2. Install libraries needed here: `pip install -r requirements.txt`

3. Install the directory `~/tensors_by_hash_cache` from the instructions here: https://github.com/redwoodresearch/remix_public/blob/45c291dbd4b62af8e907fe5a44852a3e865728ee/README.md

# TODO

[ ] make this into a pip installable, so we can just `pip install -e .` and then import much more easily.
[ ] check that the docstring circuit, which uses unconventional tensors, works for us.