import torch
from typing import Union, Tuple, Literal, Dict
from collections import OrderedDict


class GlobalCache: # this dict stores the activations from the forward pass
    """Class for managing several caches for passing activations around
    
    Also has flags that are relevant for whether we're doing 16 Heads things or not"""

    def __init__(self, device: Union[str, Tuple[str, str]] = "cuda"):
        # TODO find a way to make the device propagate when we to .to on the p
        # TODO make it essential first key is a str, second a TorchIndex, third a str

        if isinstance(device, str):
            device = (device, device)

        self.cache = OrderedDict() 
        self.second_cache = OrderedDict()
        self.device: Tuple[str, str] = (device, device)


    def clear(self, just_first_cache=False):
        
        if not just_first_cache:
            self.cache = OrderedDict()
        else:
            raise NotImplementedError()
            self.__init__(self.device[0], self.device[1]) # lol

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def to(self, device, which_caches: Literal["first", "second", "all"]="all"): # 

        caches = []
        if which_caches != "second":
            self.device = (device, self.device[1])
            caches.append(self.cache)
        if which_caches != "first":
            self.device = (self.device[0], device)
            caches.append(self.second_cache)

        # move all the parameters
        for cache in caches: # mutable means this works..
            for name in cache:
                cache_keys = list(cache.keys())
                for k in cache_keys:
                    cache[k].to(device) #  = cache[name].to(device)

        return self