#%%

from transformer_lens.cautils.notebook import *  # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload
DEVICE = "cuda"

#%%

MODEL_NAME = "gpt2-small"
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)

#%%

N=100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=True,
    seed=35795,
    device=DEVICE,
)
abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->CDE, BAB->CDE")

# %%

def get_cache(model, dataset):
    _, cache = model.run_with_cache(
        dataset.toks,
        names_filter=lambda name: name in ["blocks.9.attn.hook_result", "blocks.10.hook_resid_pre", "blocks.10.attn.hook_pattern"],
    )
    return cache

cache = get_cache(model, ioi_dataset)
abc_cache = get_cache(model, abc_dataset)

#%%

def editor(z, hook, idx, noise=False):

    z[:, ioi_dataset.word_idx["end"], 7] -= cache["blocks.9.attn.hook_result"][:, ioi_dataset.word_idx["end"], idx]

    if noise:
        z_slice = cache["blocks.9.attn.hook_result"][:, ioi_dataset.word_idx["end"], idx]
        print(z_slice.shape, z_slice.norm())
        nz = z_slice.norm()
        z[:, ioi_dataset.word_idx["end"], 7] += torch.normal(mean = 0.0, std=(0.0*z_slice + nz*0.00004))

    else:
        z[:, ioi_dataset.word_idx["end"], 7] += abc_cache["blocks.9.attn.hook_result"][:, ioi_dataset.word_idx["end"], idx]

    print("EDITED!")
    return z

#%%

anss=[]
for head_idx in range(12):
    model.reset_hooks()
    model.add_hook("blocks.10.hook_q_input", partial(editor, idx=head_idx, noise=True), level=1)
    anss.append(get_cache(model, ioi_dataset))

#%%

def get_mean_from_cache(cache):
    return cache["blocks.10.attn.hook_pattern"][torch.arange(len(ioi_dataset)), 7, ioi_dataset.word_idx["end"], ioi_dataset.word_idx["IO"]].mean().cpu()


#%%

# for noise, 
px.bar(
    x=[str(x) for x in range(12)],
    y=[get_mean_from_cache(ans) - get_mean_from_cache(cache) for ans in anss],
    title="Change in IO attention, when we replace 9.x -> 10.7 connection with noise with standard deviation 0.00004*(norm of 9.x output)",
).show()

# %%
