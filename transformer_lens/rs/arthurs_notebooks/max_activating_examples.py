#%% [markdown]
# <h1> Max activating examples for 10.7 (by norm projection to logits)</h1>
# <p> This is mostly stolen from the `arthur_experiment.py` notebook from the hackathon</p>
# <p> Since TransformerLens normalizes the </p>

from transformer_lens.cautils.notebook import * # use from transformer_lens.cautils.utils import * instead for the same effect without autoreload
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

#%%

model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
model.set_use_attn_result(True)

#%%

data = get_webtext()

#%%

LAYER_IDX = 10
HEAD_IDX = 7
unembedding = model.W_U.clone()
HEAD_HOOK_NAME = f"blocks.{LAYER_IDX}.attn.hook_result"

#%%

# In this cell I look at the sequence positions where the
# NORM of the 10.7 output (divided by the layer norm scale)
# is very large across several documents
# 
# we find that 
# i) just like in IOI, the top tokens are not interpretable and the bottom tokens repress certain tokens in prompt
# ii) unlike in IOI it seems that it is helpfully blocks the wrong tokens from prompt from being activated - example:
# 
# ' blacks', ' are', ' arrested', ' for', ' marijuana', ' possession', ' between', ' four', ' and', ' twelve', ' times', ' more', ' than'] -> 10.7 REPRESSES " blacks"

contributions = []

NORM_CUTOFF = 90
DOCUMENT_PREFIX_LENGTH = 256 # use this so the prompts we pass to the model aren't too big
PRINTING_CONTEXT_PREFIX = 70 # need to be long for prompt three
NUM_PROMPTS = 10
NUM_TOP_TOKENS = 10

for i in tqdm(range(NUM_PROMPTS)):
    tokens = model.tokenizer(
        data[i], 
        return_tensors="pt", 
        truncation=True, 
        padding=True
    )["input_ids"].to(DEVICE)
    
    tokens = tokens[0:1, :256]

    model.reset_hooks()
    logits, cache = model.run_with_cache(
        tokens,
        names_filter = lambda name: name in [HEAD_HOOK_NAME, "ln_final.hook_scale"],
    )
    output = cache[HEAD_HOOK_NAME][0, :, HEAD_IDX] / cache["ln_final.hook_scale"][0, :, 0].unsqueeze(dim=-1) # account for layer norm scaling
    
    contribution = einops.einsum(
        output,
        unembedding,
        "s d, d V -> s V",
    )

    for j in range(tokens.shape[-1]):
        if contribution[j].norm().item() > NORM_CUTOFF:
            print("-"*50, "\nPREFIX TO MAX ACTIVATING EXAMPLE: ...", "|"+"|".join(model.to_str_tokens(tokens[0, max(0,j-PRINTING_CONTEXT_PREFIX): j+1]))+"|")
            print("CORRECT COMPLETION:", "|"+model.tokenizer.decode(tokens[0, j+1])+"|")
            print()

            top_tokens = t.topk(contribution[j], NUM_TOP_TOKENS).indices
            bottom_tokens = t.topk(-contribution[j], NUM_TOP_TOKENS).indices

            print("TOP TOKENS")
            for tok in top_tokens:
                print(model.tokenizer.decode(tok))
            print()
            print("BOTTOM TOKENS")
            for tok in bottom_tokens:
                print(model.tokenizer.decode(tok))

# %% [markdown]

# <p> Arthur's rough explanation of the three prompts:</p>
# <p> |It|�|�|s| a| great| way| to| get| your| friends|,| -> | friends| repressed
# <p> | Gender| in| Agriculture| Partnership| (|G|AP|)| is| �|�|trans|forming| agriculture|...| security|.|�|�| G| -> |AP| repressed
# <p> | a| new| drug| called| se|l|um|et|in|ib| increases|...|Although| the| effects| of| -> | se| repressed

#%%