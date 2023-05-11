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
"voyage", "warfare", "work"
]

#%% 

model = get_gpt2_small()

# %%

YEARS = []
cnt=0
for year in range(1100, 1800):
    a = model.tokenizer.encode(f" {year}")
    if len(a) == 1:
        cnt += 1
        continue
    if not (98 >= year % 100 >= 2):
        continue
    YEARS.append(year)
    
print(cnt)

# %%

def get_year_data(num_examples):
    template = "The {noun} lasted from {year1} to "

    # set some random seed 

def get_all_greaterthan_things():
    raise NotImplementedError()