# def hook_fn_cache_new_result_projections(
#     v: Float[Tensor, "batch seq n_heads d_head"],
#     hook: HookPoint,
#     toks: Int[Tensor, "batch seq"],
#     model: HookedTransformer,
#     head_idx: int,
# ):
#     '''
#     Hook function to compute the projections of the vectors which will be moved from each source position.

#     This doesn't change the values; it just stores some stuff in context.
#     '''
#     # Calculate the result (pre taking weighted average from attn) by multiplying the attn scores by the values
#     result_pre_attn = einops.einsum(v[:, :, head_idx], model.W_O[hook.layer(), head_idx], "batch seq d_head, d_head d_model -> batch seq d_model")
#     token_unembeddings = model.W_U.T[toks]

#     # The result_projected tensor tells us, for each source token, what is the vector that will get moved from source 
#     # to destination (after projecting it onto the source unembedding)
#     result_pre_attn_projected, result_pre_attn_perpendicular, projection_coeffs = project(result_pre_attn, token_unembeddings, return_type="both")

#     result_hook = model.hook_dict[utils.get_act_name("result", hook.layer())]
#     result_hook.ctx["result_pre_attn_projected"] = result_pre_attn_projected
#     result_hook.ctx["projection_coeffs"] = projection_coeffs


# def hook_fn_take_weighted_avg(
#     result: Float[Tensor, "batch seq n_heads d_head"],
#     hook: HookPoint,
#     head_idx: int,
#     only_use_neg_projections: bool,
#     return_new_loss: bool,
# ):
#     '''
#     Now that `hook_fn_cache_new_result_projections` has cached the (pre-attention) results, we can now use the actual
#     attention pattern to take linear combination of new results.
#     '''
#     result_hook = hook

#     result_pre_attn_projected: Float[Tensor, "batch seqK d_model"] = result_hook.ctx.pop("result_pre_attn_projected")
#     pattern: Float[Tensor, "batch seqQ seqK"] = result_hook.ctx["pattern"]
#     projection_coeffs: Float[Tensor, "batch seqK 1"] = result_hook.ctx["projection_coeffs"]

#     # The new result_projected tensor tells us, for each source token AND destination token, what is the vector that actually 
#     # gets moved from source to destination (after projecting it onto the source unembedding)
#     result_post_attn_projected = einops.einsum(
#         result_pre_attn_projected, pattern,
#         "batch seqK d_model, batch seqQ seqK -> batch seqQ d_model",
#     )
#     result_post_attn_projected_neg = einops.einsum(
#         result_pre_attn_projected * (projection_coeffs < 0).float(), pattern,
#         "batch seqK d_model, batch seqQ seqK -> batch seqQ d_model",
#     )
#     result_hook.ctx["result_post_attn_projected"] = result_post_attn_projected_neg if only_use_neg_projections else result_post_attn_projected

#     if return_new_loss:
#         # ! In this case, we actually want the new projected results at each destination position (so we can convert them into new logits, and then new losses)
#         # We don't causally intervene (because we're looking at direct effects only), although we do cache the results at each destination token
#         result_hook["result"] = result[:, :, head_idx].clone()
#     else:
#         result_raw_norms = result[:, :, head_idx].norm(dim=1)
#         result_projected_norms = result_post_attn_projected.norm(dim=1)
#         result_projected_norms_neg = result_post_attn_projected_neg.norm(dim=1)

#         norm_fraction = result_projected_norms / result_raw_norms
#         norm_fraction_neg = result_projected_norms_neg / result_raw_norms

#         result_hook.ctx["norms"] = {
#             "projection": norm_fraction, 
#             "neg_projection": norm_fraction_neg
#         }


# def hook_fn_calculate_logit_diff(
#     scale: Float[Tensor, "batch seq 1"],
#     hook: HookPoint,
#     toks: Int[Tensor, "batch seq"],
#     model: HookedTransformer,
#     layer: int,
# ):
#     '''
#     Note on how we deal with BOS, because it's too long to fit in a comment.

#     We don't care about the predictions made by BOS tokens, so we want to zero attention whenever destination = BOS.
#     We also don't care about attending to BOS tokens, so we want to zero attention whenever source = BOS.
    
#     The thing we divide each of the (batch, seqQ) token scores by is the sum of attention over keys.
#     The thing we divide each of the (batch,) sequence scores by is the sum of attention over keys and queries.
#     '''
#     assert isinstance(toks, Int[Tensor, "batch seq"])
#     batch_size, seq_len = toks.shape

#     # Check BOS is at start (this matters!)
#     assert t.all(toks[:, 0] == model.tokenizer.bos_token_id)

#     # Get change in attn result from the context of the result hook
#     result_hook = model.hook_dict[utils.get_act_name("result", layer)]

#     result_post_attn_projected = result_hook.ctx["result_post_attn_projected"]
#     pattern = result_hook.ctx.pop("pattern")
    
#     # Scale the change
#     result_post_attn_projected_scaled = (result_post_attn_projected - result_post_attn_projected.mean(-1, keepdims=True)) / scale

#     # Calculate weighted average of how much each source token is suppressed
#     token_unembeddings = model.W_U.T[toks]
#     all_logit_suppression = einops.einsum(
#         result_post_attn_projected_scaled, token_unembeddings,
#         "batch seqQ d_model, batch seqK d_model -> batch seqQ seqK",
#     )
    
#     # The [i, q, k]-th elem is how much (in sequence i, destination position j) each of the source tokens' (k) logits are directly affected
#     # I now take a weighted average of this, over the attention paid to source tokens
#     is_not_bos_2d_mask = (toks != model.tokenizer.bos_token_id).float()
#     is_not_bos_3d_mask = einops.einsum(is_not_bos_2d_mask, is_not_bos_2d_mask, "batch seqQ, batch seqK -> batch seqQ seqK")
#     is_not_self_3d_mask = 1.0 - einops.repeat(t.eye(seq_len).to(device), "seqQ seqK -> batch seqQ seqK", batch=batch_size).float()
#     full_mask = is_not_self_3d_mask * is_not_bos_3d_mask
#     pattern_non_bos = pattern * full_mask # is_not_bos_3d_mask

#     # We don't want to divide by anything per dest. If a dest token only attends to BOS, we don't want to reweight - we don't care about that dest!
#     weighted_avg_logit_suppression_per_dest = einops.einsum(
#         all_logit_suppression, pattern_non_bos,
#         "batch seqQ seqK, batch seqQ seqK -> batch seqQ",
#     )

#     # We do want to renormalize per sequence, by dividing by the total attention we're summing over.
#     non_bos_attn_per_seq = pattern_non_bos.sum((-1, -2))
#     weighted_avg_logit_suppression_per_seq = einops.einsum(
#         all_logit_suppression, pattern_non_bos,
#         "batch seqQ seqK, batch seqQ seqK -> batch",
#     ) / non_bos_attn_per_seq

#     hook.ctx["weighted_avg_logit_suppression"] = {
#         "per_position": weighted_avg_logit_suppression_per_dest,
#         "per_sequence": weighted_avg_logit_suppression_per_seq,
#         "pattern_non_bos": pattern_non_bos
#     }


# def compute_weighted_avg_logit_suppression(
#     model: HookedTransformer,
#     toks: Int[Tensor, "batch seq"],
#     head: Tuple[int, int],
#     only_use_neg_projections: bool = False,
#     return_new_loss = False,
# ):
#     layer, head_idx = head

#     model.reset_hooks(including_permanent=True)
#     model.clear_contexts()

#     scale_hook = model.hook_dict[utils.get_act_name("scale")]
#     result_hook = model.hook_dict[utils.get_act_name("result", layer)]

#     # Define fwd_hooks
#     fwd_hooks = [
#         (utils.get_act_name("v", layer), partial(hook_fn_cache_new_result_projections, toks=toks, model=model, head_idx=head_idx)),
#         (utils.get_act_name("pattern", layer), partial(hook_fn_cache_attn, model=model, head_idx=head_idx)),
#         (utils.get_act_name("result", layer), partial(hook_fn_take_weighted_avg, head_idx=head_idx, only_use_neg_projections=only_use_neg_projections, return_new_loss=return_new_loss)),
#         (utils.get_act_name("scale"), partial(hook_fn_calculate_logit_diff, toks=toks, model=model, layer=layer)),
#     ]
#     # Forward pass which stores things in hook context (but doesn't actually change the logits)
#     logits = model.run_with_hooks(
#         toks,
#         return_type = "logits",
#         fwd_hooks = fwd_hooks
#     )

#     if return_new_loss:
#         # * TODO - this is artificial because it doesn't renormalize via layernorm, or include non-direct effects of 10.7. Is this a problem, or is this actually the best way?
#         orig_result = result_hook.ctx.pop("result")
#         new_result = result_hook.ctx.pop("result_post_attn_projected")
#         change_in_logits = einops.einsum(new_result - orig_result, model.W_U, "batch seq d_model, d_model d_vocab -> batch seq d_vocab")
#         edited_logits = logits + change_in_logits
#         orig_loss = model.loss_fn(logits, toks, per_token=True)
#         new_loss = model.loss_fn(edited_logits, toks, per_token=True)
#         return orig_loss, new_loss
#     else:
#         weighted_avg_logit_suppression = scale_hook.ctx.pop("weighted_avg_logit_suppression")
#         norm_fractions = result_hook.ctx["norms"]
#         return weighted_avg_logit_suppression, norm_fractions