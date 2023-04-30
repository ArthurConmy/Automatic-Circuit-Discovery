#%%

import torch 
import acdc

#%%

model = acdc.HookedTransformer.from_pretrained(
    model_name="redwood_attn_2l",
    # fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)

#%%
            
def shuffle_tensor(tens):
    return tens[torch.randperm(tens.shape[0])]

#%%

lis = [shuffle_tensor(torch.arange(max(i, 100))).cuda() for i in range(10, 1000)]
    
def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

      
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=100_000),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
    ) as p:
        for thing in lis:
            model(thing) # shuffle_tensor(torch.arange(iter)).cuda())
            p.step()
# %%
