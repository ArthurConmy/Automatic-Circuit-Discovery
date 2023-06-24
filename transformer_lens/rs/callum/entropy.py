from transformer_lens.cautils.utils import *

def entropy_measure(
    model: HookedTransformer,
    batch: Union[
        Int[Tensor, "batch_size seq_len"],
        List[Int[Tensor, "batch_size seq_len"]]
    ] # if it's a list then we iterate through it
):
    batch_list = [batch] if isinstance(batch, Tensor) else batch
    results = []

    progress_bar = tqdm(total = model.cfg.n_layers * (model.cfg.n_heads + 1) * len(batch_list))

    for batch in batch_list:
        batch_size, seq_len = batch.shape

        logits, cache = model.run_with_cache(
            batch,
            names_filter = lambda name: any([name.endswith(x) for x in ["resid_pre", "resid_mid", "z", "scale", "mlp_out"]])
        )
        W_U = model.W_U

        resid_entropies = t.zeros(2 * model.cfg.n_layers + 1, batch_size, seq_len)
        # Contains the entropies at each resid_pre or resid_mid layer (and the entropy for final layer)
        entropy_diffs = t.zeros(model.cfg.n_layers, model.cfg.n_heads + 1, batch_size, seq_len)
        # Contains the difference in entropy before and after the component's contribution
        entropy_marginals = t.zeros(model.cfg.n_layers, model.cfg.n_heads + 1, batch_size, seq_len)
        # Contains the marginal contribution to entropy of this component (i.e. compared to the final value)

        entropy_final = Categorical(logits = logits).entropy()
        resid_entropies[-1] = entropy_final

        scale = cache["scale"]

        for layer in range(model.cfg.n_layers):

            resid_pre = cache["resid_pre", layer] / scale
            resid_mid = cache["resid_mid", layer] / scale

            resid_pre_logits = einops.einsum(resid_pre, W_U, "batch seq d_model, d_model d_vocab -> batch seq d_vocab")
            resid_mid_logits = einops.einsum(resid_mid, W_U, "batch seq d_model, d_model d_vocab -> batch seq d_vocab")
            
            resid_pre_entropy = Categorical(logits = resid_pre_logits).entropy()
            resid_mid_entropy = Categorical(logits = resid_mid_logits).entropy()

            resid_entropies[2 * layer, :, :] = resid_pre_entropy
            resid_entropies[2 * layer + 1, :, :] = resid_mid_entropy


            for head in range(model.cfg.n_heads):
                
                # Calculate contribution of this head to the final value of residual stream
                head_contribution = einops.einsum(cache["z", layer][:, :, head], model.W_O[layer, head], "batch seq d_head, d_head d_model -> batch seq d_model") / scale
                head_contribution_logits = einops.einsum(head_contribution, W_U, "batch seq d_model, d_model d_vocab -> batch seq d_vocab")

                # Calculate the entropy diff from this head's contribution
                new_entropy = Categorical(logits = resid_pre_logits + head_contribution_logits).entropy()
                entropy_diff = new_entropy - resid_pre_entropy
                entropy_diffs[layer, head, :, :] = entropy_diff

                # Calculate the marginal contribution to entropy of this head
                ablated_entropy = Categorical(logits = logits - head_contribution_logits).entropy()
                entropy_marginals[layer, head, :, :] = entropy_final - ablated_entropy

                # Update progress bar
                progress_bar.update(1)
                t.cuda.empty_cache()

            # Calculate contribution of this head to the final value of residual stream
            mlp_contribution = cache["mlp_out", layer] / scale
            mlp_contribution_logits = einops.einsum(mlp_contribution, W_U, "batch seq d_model, d_model d_vocab -> batch seq d_vocab")
            
            # Calculate the entropy diff from this head's contribution
            new_entropy = Categorical(logits = resid_mid_logits + mlp_contribution_logits).entropy()
            entropy_diff = new_entropy - resid_mid_entropy
            entropy_diffs[layer, -1, :, :] = entropy_diff

            # Calculate the marginal contribution to entropy of this head
            ablated_entropy = Categorical(logits = logits - mlp_contribution_logits).entropy()
            entropy_marginals[layer, -1, :, :] = entropy_final - ablated_entropy
            
            # Update progress bar
            progress_bar.update(1)
            t.cuda.empty_cache()

        entropy_diffs_mean = einops.reduce(
            entropy_diffs,
            "layers heads batch seq -> layers heads",
            reduction = "mean"
        )
        entropy_marginals_mean = einops.reduce(
            entropy_marginals,
            "layers heads batch seq -> layers heads",
            reduction = "mean"
        )
        resid_entropies_mean = einops.reduce(
            resid_entropies, 
            "resid_position batch seq -> resid_position", 
            reduction = "mean"
        )

        results.append((resid_entropies_mean, entropy_diffs_mean, entropy_marginals_mean))
        t.cuda.empty_cache()

    resid_entropies_mean, entropy_diffs_mean, entropy_marginals_mean = list(zip(*results))
    resid_entropies_mean = sum(resid_entropies_mean) / len(resid_entropies_mean)
    entropy_diffs_mean = sum(entropy_diffs_mean) / len(entropy_diffs_mean)
    entropy_marginals_mean = sum(entropy_marginals_mean) / len(entropy_marginals_mean)

    return resid_entropies_mean, entropy_diffs_mean, entropy_marginals_mean



def concat_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]



def make_entropy_resid_plots(
    resid_entropies: Float[Tensor, "resid_position"],
    static: bool = False,
):
    resid_entropies_diffs = resid_entropies[1:] - resid_entropies[:-1]
    resid_entropies_attn_diffs, resid_entropies_mlp_diffs = resid_entropies_diffs[::2].tolist(), resid_entropies_diffs[1::2].tolist()

    # labels = concat_lists([[f"Attn {i}", f"MLP {i}"] for i in range(model.cfg.n_layers)])
    # print(labels)
    line(
        [resid_entropies_attn_diffs, resid_entropies_mlp_diffs], 
        width=600, 
        title="Increase in entropy at each layer & component (logit lens)", 
        labels={"value": "Entropy diff", "index": "Layer", "variable": "Component"},
        template="simple_white",
        names=["Attention", "MLPs"],
        static=static
    )



def make_entropy_plots(
    entropy_diffs: Float[Tensor, "layers heads_and_mlps"],
    entropy_marginals: Float[Tensor, "layers heads_and_mlps"],
    model: HookedTransformer,
    title: Optional[str] = None,
    static: bool = False
):
    (layers, heads_and_mlps) = entropy_marginals.shape

    assert layers == model.cfg.n_layers
    assert heads_and_mlps == model.cfg.n_heads + 1
    assert entropy_diffs.shape == entropy_marginals.shape

    entropy = t.stack([entropy_diffs, entropy_marginals], dim=0)

    title = f" ({title})" if (title is not None) else ""
    fig_list = []
    fig_list.append(imshow(
        entropy, 
        facet_col=0,
        facet_labels=["Diff (pre/post)", "Marginal (wrt final logits)"],
        width=1000,
        title="Reduction in entropy as a consequence of each head" + title,
        border=True,
        labels={"x": "Heads (+ MLP)", "y": "Layer"},
        draw=True,
        return_fig=True
    ))

    zmax = entropy[..., :-1].abs().max().item()
    fig_list.append(imshow(
        entropy[..., :-1],
        facet_col=0,
        facet_labels=["Diff (pre/post)", "Marginal (wrt final logits)"],
        width=1000, 
        title="Remove MLPs" + title, 
        border=True, 
        zmin=-zmax, 
        zmax=zmax, 
        labels={"x": "Heads", "y": "Layer"},
        draw=True,
        return_fig=True
    ))

    entropy_increases = entropy[..., :-1] * (entropy[..., :-1] > 0)
    zmax = entropy_increases.max().item()
    fig_list.append(imshow(
        entropy_increases, 
        facet_col=0,
        facet_labels=["Diff (pre/post)", "Marginal (wrt final logits)"],
        width=1000, 
        title="Only showing entropy increases" + title, 
        border=True, 
        zmin=-zmax, 
        zmax=zmax, 
        labels={"x": "Heads", "y": "Layer"},
        draw=True,
        return_fig=True
    ))

    return fig_list