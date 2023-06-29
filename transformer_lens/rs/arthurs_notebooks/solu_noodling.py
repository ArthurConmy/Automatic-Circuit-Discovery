# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_loss_from_end_state
import argparse

#%%

fully_trained_model = HookedTransformer.from_pretrained(
    "solu-4l",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)

#%%

CKPTS=[262144, 2621440, 4718592, 7077888, 9175040, 11272192, 13631488, 15728640, 18087936, 20185088, 22282240, 33292288, 44302336, 55312384, 66322432, 77332480, 88342528, 99352576, 110362624, 121372672, 132382720, 143392768, 154402816, 165412864, 176422912, 187432960, 198443008, 209453056, 220463104, 264503296, 308281344, 352321536, 396361728, 440401920, 484442112, 528482304, 572522496, 616300544, 660340736, 704380928, 748421120, 792461312, 836501504, 880279552, 924319744, 968359936, 1012400128, 1056440320, 1100480512, 1144520704, 1188298752, 1232338944, 1276379136, 1320419328, 1364459520, 1408499712, 1452277760, 1496317952, 1540358144, 1584398336, 1628438528, 1672478720, 1716518912, 1760296960, 1804337152, 1848377344, 1892417536, 1936457728, 1980497920, 2024275968, 2068316160, 2112356352, 2156396544, 2200436736, 2420375552, 2640314368, 2860515328, 3080454144, 3300392960, 3520331776, 3740270592, 3960471552, 4180410368, 4400349184, 4620288000, 4840488960, 5060427776, 5280366592, 5500305408, 5720506368, 5940445184, 6160384000, 6380322816, 6600523776, 6820462592, 7040401408, 7260340224, 7480279040, 7700480000, 7920418816, 8140357632, 8360296448, 8580497408, 8800436224, 9020375040, 9240313856, 9460514816, 9680453632, 9900392448, 10120331264, 10340270080, 10560471040, 10780409856, 11000348672, 11220287488, 11440488448, 11660427264, 11880366080, 12100304896, 12320505856, 12540444672, 12760383488, 12980322304, 13200523264, 13420462080, 13640400896, 13860339712, 14080278528, 14300479488, 14520418304, 14740357120, 14960295936, 15180496896, 15400435712, 15620374528, 15840313344, 16060514304, 16280453120, 16500391936, 16720330752, 16940269568, 17160470528, 17380409344, 17600348160, 17820286976, 18040487936, 18260426752, 18480365568, 18700304384, 18920505344, 19140444160, 19360382976, 19580321792, 19800522752, 20020461568, 20240400384, 20460339200, 20680278016, 20900478976, 21120417792, 21340356608, 21560295424, 21780496384]
dataset=get_webtext(dataset="NeelNanda/c4-code-20k")
IDX = 78
CUTOFF = 17
words = fully_trained_model.to_str_tokens(dataset[IDX])[:CUTOFF]
toks = fully_trained_model.to_tokens(dataset[IDX])[0, :CUTOFF]

#%%

ans = defaultdict(dict)
for ckpt_idx in tqdm(list(range(20, len(CKPTS), 5))):
    torch.cuda.empty_cache()
    ckpt = CKPTS[ckpt_idx]
    print(ckpt)
    model = HookedTransformer.from_pretrained(
        "solu-4l",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=False,
        checkpoint_value=ckpt,
    )

    logits = model(toks.unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)

    for i in range(len(words)):
        if words[i] == "import":
            # print(i)
            # print(words[:i+2], "<- including correct token")
            top_logits = logits[0, i, :].topk(10)
            # print(probs[0, i, :].topk(10).values)
            # print(model.to_str_tokens(top_logits.indices))
            probs_on_incorrect = probs[0, i, list(set(toks[:i+1]))].sum()
            print(probs_on_incorrect)
            ans[ckpt][i] = probs_on_incorrect.item()

#%%

fig = go.Figure()

for ckpt, ans in ans.items():
    for i, prob in ans.items():
        fig.add_trace(
            go.Scatter(
                x=[ckpt],
                y=[prob],
                mode="markers",
                marker=dict(size=10),
                name=f"{i}",
            )
        )