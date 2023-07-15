# %% [markdown] [4]:

"""
SoLU Experiments1
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project
from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_loss_from_end_state
import argparse

#%%

# just a quick check of something...
model = HookedTransformer.from_pretrained("gpt2")
tokens = torch.arange(5).long().unsqueeze(0).cuda()

#%%

logits = model(tokens)

#%%

assert model.unembed.W_U.mean(0).abs().max() < 1e-5
assert model.unembed.W_U.mean(1).abs().max() < 1e-5

#%%

model.unembed.W_U -= model.unembed.W_U.mean(0, keepdim=True)

#%%

assert model.unembed.W_U.mean(0).abs().max() < 1e-5
assert model.unembed.W_U.mean(1).abs().max() < 1e-5

#%%

new_logits = model(tokens)

#%%

torch.testing.assert_allclose(logits, new_logits)

#%%

MODEL_NAME = "solu-10l"

fully_trained_model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)

#%%

CKPTS={"solu-4l": [262144, 2621440, 4718592, 7077888, 9175040, 11272192, 13631488, 15728640, 18087936, 20185088, 22282240, 33292288, 44302336, 55312384, 66322432, 77332480, 88342528, 99352576, 110362624, 121372672, 132382720, 143392768, 154402816, 165412864, 176422912, 187432960, 198443008, 209453056, 220463104, 264503296, 308281344, 352321536, 396361728, 440401920, 484442112, 528482304, 572522496, 616300544, 660340736, 704380928, 748421120, 792461312, 836501504, 880279552, 924319744, 968359936, 1012400128, 1056440320, 1100480512, 1144520704, 1188298752, 1232338944, 1276379136, 1320419328, 1364459520, 1408499712, 1452277760, 1496317952, 1540358144, 1584398336, 1628438528, 1672478720, 1716518912, 1760296960, 1804337152, 1848377344, 1892417536, 1936457728, 1980497920, 2024275968, 2068316160, 2112356352, 2156396544, 2200436736, 2420375552, 2640314368, 2860515328, 3080454144, 3300392960, 3520331776, 3740270592, 3960471552, 4180410368, 4400349184, 4620288000, 4840488960, 5060427776, 5280366592, 5500305408, 5720506368, 5940445184, 6160384000, 6380322816, 6600523776, 6820462592, 7040401408, 7260340224, 7480279040, 7700480000, 7920418816, 8140357632, 8360296448, 8580497408, 8800436224, 9020375040, 9240313856, 9460514816, 9680453632, 9900392448, 10120331264, 10340270080, 10560471040, 10780409856, 11000348672, 11220287488, 11440488448, 11660427264, 11880366080, 12100304896, 12320505856, 12540444672, 12760383488, 12980322304, 13200523264, 13420462080, 13640400896, 13860339712, 14080278528, 14300479488, 14520418304, 14740357120, 14960295936, 15180496896, 15400435712, 15620374528, 15840313344, 16060514304, 16280453120, 16500391936, 16720330752, 16940269568, 17160470528, 17380409344, 17600348160, 17820286976, 18040487936, 18260426752, 18480365568, 18700304384, 18920505344, 19140444160, 19360382976, 19580321792, 19800522752, 20020461568, 20240400384, 20460339200, 20680278016, 20900478976, 21120417792, 21340356608, 21560295424, 21780496384],
"solu-10l": [196608, 3342336, 6291456, 9240576, 12386304, 15335424, 18284544, 21233664, 24379392, 27328512, 30277632, 45219840, 60358656, 75300864, 90243072, 105381888, 120324096, 135266304, 150208512, 165347328, 180289536, 195231744, 210370560, 225312768, 240254976, 255197184, 270336000, 285278208, 300220416, 360382464, 420347904, 480313344, 540278784, 600244224, 660209664, 720371712, 780337152, 840302592, 900268032, 960233472, 1020198912, 1080360960, 1140326400, 1200291840, 1260257280, 1320222720, 1380384768, 1440350208, 1500315648, 1560281088, 1620246528, 1680211968, 1740374016, 1800339456, 1860304896, 1920270336, 1980235776, 2040201216, 2100363264, 2160328704, 2220294144, 2280259584, 2340225024, 2400387072, 2460352512, 2520317952, 2580283392, 2640248832, 2700214272, 2760376320, 2820341760, 2880307200, 2940272640, 3000238080, 3300261888, 3600285696, 3900309504, 4200333312, 4500357120, 4800380928, 5100208128, 5400231936, 5700255744, 6000279552, 6300303360, 6600327168, 6900350976, 7200374784, 7500201984, 7800225792, 8100249600, 8400273408, 8700297216, 9000321024, 9300344832, 9600368640, 9900392448, 10200219648, 10500243456, 10800267264, 11100291072, 11400314880, 11700338688, 12000362496, 12300386304, 12600213504, 12900237312, 13200261120, 13500284928, 13800308736, 14100332544, 14400356352, 14700380160, 15000207360, 15300231168, 15600254976, 15900278784, 16200302592, 16500326400, 16800350208, 17100374016, 17400201216, 17700225024, 18000248832, 18300272640, 18600296448, 18900320256, 19200344064, 19500367872, 19800391680, 20100218880, 20400242688, 20700266496, 21000290304, 21300314112, 21600337920, 21900361728, 22200385536, 22500212736, 22800236544, 23100260352, 23400284160, 23700307968, 24000331776, 24300355584, 24600379392, 24900206592, 25200230400, 25500254208, 25800278016, 26100301824, 26400325632, 26700349440, 27000373248, 27300200448, 27600224256, 27900248064, 28200271872, 28500295680, 28800319488, 29100343296, 29400367104, 29700390912]}[MODEL_NAME]
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
        MODEL_NAME,
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

#%%

INC = 1 # incrementation; set to 1 for control
bars = {
    j: {i: None for i in range(20)} for j in (torch.tensor([7, 10, 14])-int(INC)).tolist()
}
fully_trained_model.set_use_attn_result(True)
for head_idx in [18, 19] + list(range(17, -1, -1)):
    print(head_idx)
    fully_trained_model.reset_hooks()
    def zer(z, hook, head):
        z[:, :, head] *= 0
        return z
    fully_trained_model.add_hook(
        "blocks.9.attn.hook_result",
        partial(zer, head=head_idx),
    )
    logits = fully_trained_model(toks.unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)

    for i in range(len(words)-int(INC)):
        if words[i+int(INC)] == "import":
            print(i)
            # print(words[:i+2], "<- including correct token")
            top_logits = logits[0, i, :].topk(10)
            # print(probs[0, i, :].topk(10).values)
            # print(model.to_str_tokens(top_logits.indices))
            probs_on_incorrect = probs[0, i, list(set(toks[:i+1].tolist()))].sum()
            print(probs_on_incorrect)
            # ans[ckpt][i] = probs_on_incorrect.item()
            bars[i][head_idx]=probs_on_incorrect.item()

# %%

fig = go.Figure()

for idx in bars:

    fig.add_trace(
        go.Scatter(
            x=list(range(20)),
            y=torch.tensor([bars[idx][j] for j in range(20)]).log(),
            mode="markers",
            marker=dict(size=10),
            name="Prompt "+str(idx),
        )
    )
    fig.update_layout(
        xaxis_title="Layer 9 Head Index",
        yaxis_title="Probability on all current tokens in context",
    )

if INC: 
    fig.update_layout(
        title="Comparison to baseline"
    )

else:
    fig.update_layout(
        title="Effect of zero ablating a Layer 9 Head"
    )

fig.show()

    # %%

### AND THE OTHER THINGS

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
        # checkpoint_value=ckpt,
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

eyes = [7, 10, 14]
fully_trained_model_answers = [0.0001, 0.0231, 0.0570]

for eye, answer in zip(eyes, fully_trained_model_answers, strict=True):
    ans[max(ans)][eye] = answer

#%%

fig = go.Figure()
for eye in eyes:
    # for i, prob in ans.items():
    fig.add_trace(
        go.Scatter(
            x=list(ans.keys()),
            y=[ans[key][eye] for key in ans.keys()],
            mode="markers",
            marker=dict(size=10),
            name=f"{eye}",
        )
    )

fig.show()
# %%
