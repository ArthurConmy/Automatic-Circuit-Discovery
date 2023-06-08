# Setup

This implementation of Subnetwork Probing should install by default when installing the ACDC code.

It hosts a fork of `transformer_lens` as a submodule. This should probably be changed in the future.

The fork introduces the class `MaskedHookPoint`, which masks some of its values with either zero or stored activations.
That's the only crucial difference with mainstream `transformer_lens`. Most of the complexity is in `train.py`.

# Subnetwork Probing

[Low-Complexity Probing via Finding Subnetworks](https://github.com/stevenxcao/subnetwork-probing)  
Steven Cao, Victor Sanh, Alexander M. Rush  
NAACL-HLT 2021  

# HISP 

[Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) Michel et al 2019
