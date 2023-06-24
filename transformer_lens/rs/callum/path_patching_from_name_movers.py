from transformer_lens.cautils.utils import *

def get_io_vs_s_attn_for_nmh(
    patched_cache: ActivationCache,
    orig_dataset: IOIDataset,
    orig_cache: ActivationCache,
    neg_nmh: Tuple[int, int],
) -> Float[Tensor, "batch"]:
    '''
    Returns the difference between patterns[END, IO] and patterns[END, S1], where patterns
    are the attention patterns for the negative name mover head.

    This is returned in the form of a tuple of 2 tensors: one for the patched distribution
    (calculated using `patched_cache` which is returned by the path patching algorithm), and
    one for the clean IOI distribution (which is just calculated directly from that cache).
    '''
    layer, head = neg_nmh
    attn_pattern_patched = patched_cache["pattern", layer][:, head]
    attn_pattern_clean = orig_cache["pattern", layer][:, head]
    # both are (batch, seq_Q, seq_K), and I want all the "end -> IO" attention probs

    N = orig_dataset.toks.size(0)
    io_seq_pos = orig_dataset.word_idx["IO"]
    s1_seq_pos = orig_dataset.word_idx["S1"]
    end_seq_pos = orig_dataset.word_idx["end"]

    return (
        attn_pattern_patched[range(N), end_seq_pos, io_seq_pos] - attn_pattern_patched[range(N), end_seq_pos, s1_seq_pos],
        attn_pattern_clean[range(N), end_seq_pos, io_seq_pos] - attn_pattern_clean[range(N), end_seq_pos, s1_seq_pos],
    )


def get_nnmh_patching_patterns(num_batches, neg_nmh, nmhs, model, orig_is_ioi=True):
    results_patched = t.empty(size=(0,)).to(device)
    results_clean = t.empty(size=(0,)).to(device)

    for seed in tqdm(range(num_batches)):

        ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric = generate_data_and_caches(20, model=model, seed=seed)

        if orig_is_ioi: 
            orig_dataset, new_dataset, orig_cache, new_cache = ioi_dataset, abc_dataset, ioi_cache, abc_cache
        else:
            orig_dataset, new_dataset, orig_cache, new_cache = abc_dataset, ioi_dataset, abc_cache, ioi_cache

        new_results_patched, new_results_clean = path_patch(
            model,
            orig_input=orig_dataset.toks,
            new_input=new_dataset.toks,
            orig_cache=orig_cache,
            new_cache=new_cache,
            sender_nodes=[Node("z", layer=layer, head=head) for layer, head in nmhs], # Output of all name mover heads
            receiver_nodes=Node("q", neg_nmh[0], head=neg_nmh[1]), # To query input of negative name mover head
            patching_metric=partial(get_io_vs_s_attn_for_nmh, orig_dataset=orig_dataset, orig_cache=orig_cache, neg_nmh=neg_nmh),
            apply_metric_to_cache=True,
            direct_includes_mlps=not(model.cfg.use_split_qkv_input),
        )
        results_patched = t.concat([results_patched, new_results_patched])
        results_clean = t.concat([results_clean, new_results_clean])

        t.cuda.empty_cache()

    return results_patched, results_clean