Fork of the TransformerLens repo for Callum and Arthur's research sprint.

See `transformer_lens/rs/arthurs_notebooks/example_notebook.py` for example usage.

## Setup:

This setup relies on using an SSH key to access Github. See [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and the associated links on that page (if you don't have an SSH key to begin with)

```
git clone git@github.com:ArthurConmy/TransformerLens.git
cd TransformerLens
git checkout researchsprint
pip install -e .
```

## Difference from [the main branch of TransformerLens](https://github.com/neelnanda-io/TransformerLens)

1. We set the `ACCELERATE_DISABLE_RICH` environment variable in `transformer_lens/__init__.py` to `"1"` to stop an annoying reformatting of notebook error messages
2. We add the `qkv_normalized_input` hooks that can be optionally added to models

## [See the main TransformerLens README here](https://github.com/neelnanda-io/TransformerLens)
