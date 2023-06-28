from transformer_lens.cautils.utils import *
from transformer_lens import FactoredMatrix

def get_neg_copying_score(
    model: HookedTransformer,
    W_EE: Float[Tensor, "d_vocab d_model"],
    head: Tuple[int, int],
    n_batches: int = 10,
    sample_size: Optional[int] = None,
    return_prob: bool = False,
    return_frac_on_topk: bool = False,
    truncate_negative: bool = False,
):
    '''
    Gets neg copying scores (either as a sample, or as avg diff).
    '''
    assert not(return_prob and return_frac_on_topk)

    layer, head_idx = head
    W_V = model.W_V[layer, head_idx]
    W_O = model.W_O[layer, head_idx]

    W_U = model.W_U

    full_OV_circuit = FactoredMatrix(W_EE @ W_V, W_O @ W_U)

    results = []

    for batch in range(n_batches):

        random_sample = t.randint(low=0, high=model.cfg.d_vocab, size=(sample_size,))

        sample_OV_circuit_negated = - full_OV_circuit.A[random_sample, :] @ full_OV_circuit.B[:, random_sample]

        if return_prob:
            probs = sample_OV_circuit_negated.softmax(dim=-1)
            results.append(probs.diag().mean().item())
        elif return_frac_on_topk:
            topk = sample_OV_circuit_negated.topk(k=sample_size, dim=-1).indices
            results.append((topk == t.arange(sample_size).unsqueeze(0)).float().mean().item())
        else:
            diag_sum = sample_OV_circuit_negated.trace()
            offdiag_sum = sample_OV_circuit_negated.sum() - diag_sum
            diag_avg = diag_sum / sample_size
            offdiag_avg = offdiag_sum / (sample_size * (sample_size - 1))
            diff = diag_avg - offdiag_avg
            results.append(diff.item())
    
    results = t.tensor(results).mean()
    if truncate_negative:
        return results * (results > 0)
    else:
        return results
    



def split_range(total, size):
    numbers = list(range(total))
    chunks = [numbers[i:i+size] for i in range(0, len(numbers), size)]
    return chunks



def get_neg_copying_score_full(
    model: HookedTransformer,
    W_EE: Float[Tensor, "d_vocab d_model"],
    head: Tuple[int, int],
    block_size: int = 5000,
    return_prob: bool = False,
    topk: Optional[int] = None,
    truncate_negative: bool = False,
):
    '''
    Gets neg copying scores for the entire OV matrix.

    There are 3 options:
        if return_prob, it returns average diag probs.
        if else:
            if topk is not None, it returns the fraction of time topk includes diag.
            if topk is None, it returns the average diff between diag and offdiag.
    '''
    assert not(return_prob and topk)

    d_vocab = model.cfg.d_vocab

    layer, head_idx = head
    W_V = model.W_V[layer, head_idx]
    W_O = model.W_O[layer, head_idx]

    W_U = model.W_U

    full_OV_circuit = FactoredMatrix(W_EE @ W_V, W_O @ W_U)

    results = []

    split_rng = split_range(d_vocab, block_size)

    for rng in split_rng:
        zero_rng = range(len(rng))
        sample_OV_circuit_negated = - full_OV_circuit.A[rng, :] @ full_OV_circuit.B

        if return_prob:
            diag_probs = sample_OV_circuit_negated.softmax(dim=-1)[zero_rng, rng]
            results.append(diag_probs.mean().item())

        elif topk is not None:
            topk_indices = sample_OV_circuit_negated.topk(k=topk, dim=-1).indices
            results.append((topk_indices == t.tensor(rng).unsqueeze(1).to(device)).float().mean().item())

        else:
            diag_sum = sample_OV_circuit_negated[zero_rng, rng].sum()
            offdiag_sum = sample_OV_circuit_negated.sum() - diag_sum
            diag_avg = diag_sum / d_vocab
            offdiag_avg = offdiag_sum / (d_vocab * (d_vocab - 1))
            diff = diag_avg - offdiag_avg
            results.append(diff.item())
    
    results = t.tensor(results).mean()
    if truncate_negative:
        return results * (results > 0)
    else:
        return results