#%%
 
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

from transformer_lens import HookedTransformer
import torch

# %%

sentence = "Hello world!"
    
# %%

model = HookedTransformer.from_pretrained("gpt2")

#%%

prompt = """Your task is to add calls to a Question Answering API to a piece of text. The questions should help you get information required to complete the text. You can call the API by writing "[QA(question)]" where "question" is the question you want to ask. Here are some examples of API calls:\nInput: The New England Journal of Medicine is a registered trademark of the MMS.\nOutput: The New England Journal of Medicine is a registered trademark of [QA("Who is the publisher of The New England Journal of Medicine?")] the MMS.\n\nInput: Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company.\nOutput: Coca-Cola, or Coke, is a carbonated soft drink manufactured by the Coca-Cola Company."""
tokens = model.tokenizer.encode(prompt)
logits = model(prompt, prepend_bos=False, return_type="logits")

#%%

unprocessed_model = HookedTransformer.from_pretrained(
    "redwood_attn_2l",
    center_writing_weights=False, 
    center_unembed=False,
    fold_ln=False,
)

# %%

processed_model = HookedTransformer.from_pretrained( # comment out the assertion to load this in...
    "redwood_attn_2l",
    center_writing_weights=True,
    center_unembed=True,
    fold_ln=False,
)

# %%

tokens_processed = processed_model.to_tokens(sentence)
tokens_unprocessed = unprocessed_model.to_tokens(sentence)
assert torch.allclose(tokens_processed, tokens_unprocessed)

# %%

logits_processed = processed_model(sentence, return_type="logits")
logits_unprocessed = unprocessed_model(sentence, return_type="logits")

#%%

probs_processed = torch.nn.functional.softmax(logits_processed, dim=-1)
probs_unprocessed = torch.nn.functional.softmax(logits_unprocessed, dim=-1)

#%%

assert torch.allclose(probs_processed, probs_unprocessed, atol=1e-2, rtol=1e-2) # not even the norms are close!

# %%
