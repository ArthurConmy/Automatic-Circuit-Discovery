from transformer_lens.cautils.utils import *
import torch

def get_loss_from_end_state(
    model,
    end_state,
    targets,
    logits=None,
    return_logits=False,
):
    # end state has shape batch, seq_len, hidden_size
    # targets has shape batch, seq_len

    if logits is None:
        assert list(end_state.shape) == list(targets.shape) + [
            model.cfg.d_model
        ], f"end_state.shape: {end_state.shape}, targets.shape: {targets.shape}"
        assert len(end_state.shape) == 3, "We stricter now"
        post_layer_norm = model.ln_final(end_state)
        logits = model.unembed(post_layer_norm)
    else:
        assert end_state is None
        assert logits.shape == targets.shape + (model.cfg.d_vocab,)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    if len(targets.shape) == 2:
        loss = -log_probs[
            torch.arange(targets.shape[0]).unsqueeze(1),
            torch.arange(targets.shape[1]).unsqueeze(0),
            targets,
        ]
    elif len(targets.shape) == 1:
        assert loss.shape[0]==1, loss.shape
        loss = -log_probs[
            :, torch.arange(targets.shape[0]), targets
        ]

    if return_logits:
        return loss, logits
    return loss

def get_filtered_webtext(model, batch_size=30, seed: int = 1729, device="cuda", max_seq_len=1024):
    """
    Returns webtext that is all equal to length max token length. Ah.
    """
    dataset = get_webtext(seed=seed)
    filtered_tokens = []
    targets = []  # targets for prediction

    print("Not rapid, but not THAT slow :-) ")
    _idx = -1
    while len(filtered_tokens) < batch_size:
        _idx += 1
        cur_tokens = model.to_tokens(dataset[_idx], truncate=False).tolist()[0]
        if (
            len(cur_tokens) >= max_seq_len
        ):  # so we're not biasing towards early sequence positions...
            filtered_tokens.append(cur_tokens[:max_seq_len])
            targets.append(cur_tokens[1 : max_seq_len + 1])

    mybatch = torch.LongTensor(filtered_tokens).to(device)
    mytargets = torch.LongTensor(targets).to(device)
    return mybatch, mytargets

def set_to_value(
    z, 
    hook,
    head_idx,
    new_value,
    seq_indices=None,
):
    if seq_indices is None:
        assert z[:, :, head_idx].shape == new_value.shape
        z[:, :, head_idx] = new_value
    else:
        assert len(seq_indices)==len(z)
        z[torch.arange(len(z)), seq_indices, head_idx] = new_value

    return z

    #                 1.5583579540252686,
# for 11 3