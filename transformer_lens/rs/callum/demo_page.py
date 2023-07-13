# %%

from transformer_lens.cautils.notebook import *

import markdown


from transformer_lens.rs.callum.demo_page_backend import (
    parse_str,
    parse_str_tok_for_printing,
    HeadResults,
    LogitResults,
    ModelResults,
    get_data_dict,
    topk_of_Nd_tensor,
    generate_4_html_plots,
)

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cpu"
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)

# %%

BATCH_SIZE = 50
SEQ_LEN = 60 # 1024

DATA_STR = get_webtext(seed=6)[:BATCH_SIZE]
DATA_STR = [parse_str(s) for s in DATA_STR]

DATA_TOKS = model.to_tokens(DATA_STR)
DATA_STR_TOKS = model.to_str_tokens(DATA_STR)

if SEQ_LEN < 1024:
    DATA_TOKS = DATA_TOKS[:, :SEQ_LEN]
    DATA_STR_TOKS = [str_toks[:SEQ_LEN] for str_toks in DATA_STR_TOKS]

DATA_STR_TOKS_PARSED = [[parse_str_tok_for_printing(tok) for tok in toks] for toks in DATA_STR_TOKS]

NEGATIVE_HEADS = [(10, 7), (11, 10)]

print(DATA_TOKS.shape, "\n")

print(DATA_STR_TOKS[0])


# %%


generate_4_html_plots(
    model,
    DATA_TOKS[35:40],
    DATA_STR_TOKS_PARSED[35:40],
    display_plot_for_batch_idx=36-35,
)

# %%

# loss_diffs_mean_direct_107 = loss_diffs[2, 0]

# most_useful_positions = topk_of_Nd_tensor(loss_diffs_mean_direct_107, 5)

# for batch_idx, seq_pos in most_useful_positions:
#     print("\n".join([
#         f"Batch = {batch_idx}",
#         f"Seq pos = {seq_pos}",
#         f"Loss increase from ablation = {loss_diffs_mean_direct_107[batch_idx, seq_pos]}",
#         f"Text = {''.join(DATA_STR_TOKS_PARSED[batch_idx][seq_pos-10: seq_pos+1])}",
#         ""
#     ]))
