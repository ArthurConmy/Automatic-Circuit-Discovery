import torch

def logit_diff(
    logits_or_model,
    dataset,
    logits=None,
    mean=True,
    item=True,
):
    if "HookedTransformer" in str(type(logits_or_model)):
        logits = model(dataset.toks.long()).detach()
    else:
        logits = logits_or_model

    logits_on_end = logits[torch.arange(dataset.N), dataset.word_idx["end"]]
    
    logits_on_correct = logits_on_end[torch.arange(dataset.N), dataset.io_tokenIDs]
    logits_on_incorrect = logits_on_end[torch.arange(dataset.N), dataset.s_tokenIDs]

    logit_diff = logits_on_correct - logits_on_incorrect
    if mean:
        logit_diff = logit_diff.mean()
    if item:
        logit_diff = logit_diff.item()

    return logit_diff
