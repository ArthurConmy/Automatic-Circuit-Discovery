import logging

try:
    from transformer_lens.utils import select_compatible_kwargs
except ImportError as e:
    logging.warning(
        f"Are you sure you have an up-to-date TransformerLens installed? As of 6th June 2023, `select_compatible_kwargs` should be importable (this was added at the same time as the functionality needed for ACDC) but there is an error {e=}"
    )