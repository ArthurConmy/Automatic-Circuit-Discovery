    """
    This indents a notebook so that it can be run entirely as a test!
    """

def test_notebook():
    # %%

    from IPython import get_ipython

    if get_ipython() is not None:
        get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
        get_ipython().run_line_magic("autoreload", "2")  # type: ignore

    from tqdm import tqdm
    import warnings
    import acdc
    from copy import deepcopy
    from acdc import HookedTransformer
    from acdc.induction.utils import get_all_induction_things
    import torch
    from acdc.acdc_utils import TorchIndex, Edge, EdgeType, OrderedDefaultdict, make_nd_dict

    RECOMPUTE = True # turn this off to just get the ordered heads, with no need for a backward pass!
    ZERO_ABLATION = False

    if not RECOMPUTE:
        torch.set_grad_enabled(False)

    # %%

    num_examples = 40
    seq_len = 300
    
    all_induction_things = get_all_induction_things(
        kl_return_tensor=True,
        num_examples=num_examples,
        device="cuda",
        seq_len=seq_len,
        return_one_element=False,
        sixteen_heads=True,
    )

    tl_model = all_induction_things.tl_model
    toks_int_values = all_induction_things.validation_data
    toks_int_values_other = all_induction_things.validation_patch_data
    metric = all_induction_things.validation_metric
    mask_rep = all_induction_things.validation_mask

    assert tl_model.cfg.use_attn_result, "Set this to True"
    if ZERO_ABLATION:
        tl_model.global_cache.sixteen_heads_config.zero_ablation = True

    # %%

    assert not tl_model.global_cache.sixteen_heads_config.forward_pass_enabled

    with torch.no_grad():
        _, corrupted_cache = tl_model.run_with_cache(
            toks_int_values_other,
        )
    tl_model.zero_grad()
    tl_model.global_cache.second_cache = corrupted_cache

    # %%

    tl_model.global_cache.sixteen_heads_config.forward_pass_enabled = True

    clean_cache = tl_model.add_caching_hooks(
        # toks_int_values,
        incl_bwd=True,
    )

    clean_logits = tl_model(toks_int_values)
    kl_result = metric(clean_logits)
    assert list(kl_result.shape) == [num_examples, seq_len], kl_result.shape
    kl_result = (kl_result * mask_rep).sum() / mask_rep.int().sum().item()

    if RECOMPUTE:
        kl_result.backward(retain_graph=True)

    # %%

    shap = list(clean_cache["blocks.0.attn.hook_result"].shape)
    assert len(shap) == 4, shap
    assert shap[2] == 8, shap  # not num_heads ???

    # %%

    assert RECOMPUTE

    keys = []
    for layer_idx in range(2):
        for head_idx in range(8):
            keys.append((layer_idx, head_idx))

    results = {
        (layer_idx, head_idx): torch.zeros(size=(num_examples, seq_len))
        for layer_idx, head_idx in keys
    }

    for i in tqdm(range(num_examples)):
        for j in tqdm(range(seq_len)):
            if mask_rep[i, j] == 0:
                continue

            tl_model.zero_grad()
            tl_model.reset_hooks()
            clean_cache = tl_model.add_caching_hooks(incl_bwd=True)
            clean_logits = tl_model(toks_int_values)
            kl_result = metric(clean_logits)[i, j]
            kl_result.backward(retain_graph=True)

            for layer_idx in range(2):
                fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"
                bwd_hook_name = f"blocks.{layer_idx}.attn.hook_result_grad"

                cur_results = torch.abs(
                    torch.einsum(
                        "bshd,bshd->bh",
                        clean_cache[bwd_hook_name], # gradient
                        clean_cache[fwd_hook_name]- (0.0 if ZERO_ABLATION else corrupted_cache[fwd_hook_name]), 
                    )
                )

                for head_idx in range(8):
                    results_entry = cur_results[i, head_idx].item()
                    results[(layer_idx, head_idx)][(i, j)] = results_entry

    for k in results:
        results[k].to("cpu")

    # %%

    assert RECOMPUTE

    kls = {
        (layer_idx, head_idx): torch.zeros(size=(num_examples, seq_len))
        for layer_idx, head_idx in results.keys()
    }

    from tqdm import tqdm

    for i in tqdm(range(num_examples)):
        for j in tqdm(range(seq_len)):
            if mask_rep[i, j] == 0:
                continue  # lolololol

            tl_model.zero_grad()
            tl_model.reset_hooks()
            clean_cache = tl_model.add_caching_hooks(incl_bwd=True)
            clean_logits = tl_model(toks_int_values)
            kl_result = metric(clean_logits)[i, j]
            # print(f"{kl_result=}")
            kl_result.backward(retain_graph=True)

            for layer_idx in range(2):
                fwd_hook_name = f"blocks.{layer_idx}.attn.hook_result"

                for head_idx in range(8):
                    g = (
                        tl_model.hook_dict[fwd_hook_name]
                        .xi.grad[0, 0, head_idx, 0]
                        .norm()
                        .item()
                    )
                    kls[(layer_idx, head_idx)][i, j] = g

    for k in kls:
        kls[k].to("cpu")

    # %%

    assert RECOMPUTE

    for k in results:
        print(k, results[k].norm().item(), kls[k].norm().item())  # should all be close!!!
        assert torch.allclose(results[k], kls[k])

    import gc; gc.collect()
    torch.cuda.empty_cache()

    # note the shape of these is 40*300, to get expectation take the sum and divide by mask_reps.sum().int()

    #%%

    # some intersection with Aug's works!
    # goal 1: get order of heads
    # goal 2: edit xis 
    # goal 3: compute num edges

    # %%

    assert RECOMPUTE

    # def compute_scores(kl_dict):
    kl_dict = deepcopy(results)
    if True:
        scores_list = torch.zeros(size=(2, 8))
        mask_list = []
        for layer_idx in range(2):
            for head_idx in range(8):
                score = kl_dict[(layer_idx, head_idx)].sum() / mask_rep.sum().int().item()
                scores_list[layer_idx, head_idx] = score
        
        # normalize by L2 of the layers
        l2_norms = scores_list.norm(dim=0).unsqueeze(0)
        scores_list = scores_list / l2_norms

        all_heads = []
        for layer_idx in range(2):
            for head_idx in range(8):
                all_heads.append((layer_idx, head_idx))

        # sort both lists by scores
        sorted_indices = sorted(all_heads, key=lambda x: scores_list[x])

        # mask_list = list(mask_list)
        # mask_list.reverse()
        # scores_list = list(scores_list)
        # scores_list.reverse()
        # # return mask_list, scores_list

    # %%

    def mask_head(model, head_to_mask_tuple, unmask=False):
        layer = head_to_mask_tuple[0]
        head = head_to_mask_tuple[1]
        for layer_index, layer_object in enumerate(model.blocks):
            for head_index in range(8):
                if layer_index == layer and head_index == head:
                    layer_object.attn.hook_result.xi.data[:, :, head, :] = (1.0 if unmask else 0.0)

    def remove_node(corr, node):
        children = list(node.children)
        print(children)
        for child in children:
            print("Removing edge", child.name, child.index, node.name, node.index)
            try: corr.remove_edge(child.name, child.index, node.name, node.index)
            except: pass
        parents = list(node.parents)
        print(parents)
        for parent in parents:
            print("Removing edge", node.name, node.index, parent.name, parent.index)
            try: corr.remove_edge(node.name, node.index, parent.name, parent.index)
            except: pass
        return corr

    def mask_head_in_correspondence(corr, head_to_mask_tuple):
        layer_to_mask = head_to_mask_tuple[0]
        head_to_mask = head_to_mask_tuple[1]

        found = False

        print("Masking...", head_to_mask_tuple)
        for node in corr.nodes():
            head = node.index.as_index[-1]
            layer = int(node.name.split(".")[1])

            if layer == layer_to_mask and head == head_to_mask:
                found=True
                corr = remove_node(corr, node)


        assert found, head_to_mask_tuple
        return corr

    def count_no_edges(corr, verbose=False):
        """TODO Arthur move this elsewhere"""

        cnt = 0

        for key, edge in corr.all_edges().items():
            if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                if verbose:
                    print(key)
                cnt += 1

        if verbose:
            print("No edge", cnt)

        return cnt

    from acdc.TLACDCCorrespondence import TLACDCCorrespondence
    corr = TLACDCCorrespondence.setup_from_model(tl_model)

    # %%

    if not RECOMPUTE:
        sorted_indices = [ # precomputed
            (1, 2),
            (1, 4),
            (1, 7),
            (1, 1),
            (0, 5),
            (1, 3),
            (1, 0),
            (0, 6),
            (1, 6),
            (0, 0),
            (0, 3),
            (1, 5),
            (0, 1),
            (0, 7),
            (0, 4),
            (0, 2),
        ]

        sorted_indices = [ # FOR ZERO
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (0, 3),
            (1, 7),
            (0, 1),
            (0, 2),
            (0, 5),
            (0, 7),
            (0, 4),
            (1, 0),
            (0, 6),
            (1, 6),
            (0, 0),
            (1, 5),
        ]

    # mask_list, scores_list = compute_scores(kls)
    kl_div_list = []
    edge_list = []

    print(metric(tl_model(toks_int_values)), "should be rough 0")

    for index in sorted_indices:
        mask_head(tl_model, index)
        # mask_head(corr, index)
        corr = mask_head_in_correspondence(corr, index)
        edges = count_no_edges(corr)
        kl_div = metric(tl_model(toks_int_values))
        kl_div_list.append(kl_div.sum().item() / mask_rep.sum().int().item())
        edge_list.append(edges)

    # %%