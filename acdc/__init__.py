def check_transformer_lens_version():
    """Test that your TransformerLens version is up-to-date for ACDC
    by checking that `hook_mlp_in`s exist"""

    from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": 1, 
            "d_model": 1,
            "n_ctx": 1,
            "d_head": 1,
            "act_fn": "gelu",
            "d_vocab": 0,
        }
    )

    from transformer_lens.HookedTransformer import HookedTransformer
    mini_trans = HookedTransformer(cfg)

    mini_trans.blocks[0].hook_mlp_in # try and access the hook_mlp_in: if this fails, your TL is not sufficiently up-to-date

check_transformer_lens_version()