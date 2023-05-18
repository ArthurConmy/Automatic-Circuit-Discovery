#%%

import warnings
from functools import partial
from copy import deepcopy
import torch.nn.functional as F
from typing import List
import click
import IPython
from acdc.acdc_utils import kl_divergence
import torch
from acdc.ioi.ioi_dataset import IOIDataset  # NOTE: we now import this LOCALLY so it is deterministic
from tqdm import tqdm
import wandb
from acdc.HookedTransformer import HookedTransformer
from acdc.ioi.utils import get_gpt2_small

NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", 
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance", 
    "rule", "sanctions", "shift", "siege", "slump", 
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour", 
    "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work",
]

#%% 

model = get_gpt2_small()

# %%

YEARS = []
YEARS_BY_CENTURY = {}

for century in range(11, 18):
    all_success = []
    for year in range(century * 100 + 2, (century * 100) + 99):
        a = model.tokenizer.encode(f" {year}")
        if a == [model.tokenizer.encode(f" {str(year)[:2]}")[0], model.tokenizer.encode(str(year)[2:])[0]]:
            all_success.append(str(year))
            continue
    YEARS.extend(all_success[1:-1])
    YEARS_BY_CENTURY[century] = all_success[1:-1]

TOKENS = {
    i: torch.tensor(model.tokenizer.encode(f"{'0' if i<=9 else ''}{i}")).long().item() for i in range(1, 100) # I hope!
}
INV_TOKENS = {v: k for k, v in TOKENS.items()}

def greaterthan_metric(logits, tokens):
    probs = F.softmax(logits[:, -1], dim=-1) # last elem???
    ans = 0.0
    for i in range(len(probs)):
        yearend = INV_TOKENS[tokens[i][7].item()]
        for year_suff in range(yearend+1, 99):
            ans += probs[i, TOKENS[year_suff]]
        for year_pref in range(0, yearend):
            ans -= probs[i, TOKENS[year_pref]]
    return - float(ans / len(probs))

def get_year_data(num_examples, model):
    template = "The {noun} lasted from the year {year1} to "

    # set some random seed
    torch.random.manual_seed(54)
    nouns_perm = torch.randint(0, len(NOUNS), (num_examples,))
    years_perm = torch.randint(0, len(YEARS), (num_examples,))

    prompts = []
    prompts_tokenized = []
    for i in range(num_examples):
        year = YEARS[years_perm[i]]
        prompts.append(
            template.format(
                noun=NOUNS[nouns_perm[i]],
                year1=year,
            ) + year[:2]
        )
        prompts_tokenized.append(model.tokenizer.encode(prompts[-1], return_tensors="pt").to(model.cfg.device))
        assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (prompts_tokenized[-1].shape, prompts_tokenized[0].shape)
    prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
    assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape

    return prompts_tokenized, prompts

def get_all_greaterthan_things(num_examples, device="cuda"):
    model = get_gpt2_small(device=device)
    data, prompts = get_year_data(num_examples, model)
    return model, data, prompts, partial(greaterthan_metric, tokens=data)