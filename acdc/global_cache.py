import torch
from typing import Union, Tuple, Literal, Dict
from collections import OrderedDict


class GlobalCache: # this dict stores the activations from the forward pass
    """Class for managing several caches for passing activations around"""

    def __init__(self, device: Union[str, Tuple[str, str]] = "cuda"):
        # TODO find a way to make the device propagate when we to .to on the p
        # TODO make it essential first key is a str, second a TorchIndex, third a str

        if isinstance(device, str):
            device = (device, device)

        self.online_cache = OrderedDict() 
        self.corrupted_cache = OrderedDict()
        self.device: Tuple[str, str] = (device, device)


    def clear(self, just_first_cache=False):
        
        if not just_first_cache:
            self.online_cache = OrderedDict()
        else:
            raise NotImplementedError()
            self.__init__(self.device[0], self.device[1]) # lol

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def to(self, device, which_caches: Literal["online", "corrupted", "all"]="all"): # 

        caches = []
        if which_caches != "online":
            self.device = (device, self.device[1])
            caches.append(self.online_cache)
        if which_caches != "corrupted":
            self.device = (self.device[0], device)
            caches.append(self.corrupted_cache)

        # move all the parameters
        for cache in caches: # mutable means this works..
            for name in cache:
                cache_keys = list(cache.keys())
                for k in cache_keys:
                    cache[k].to(device) #  = cache[name].to(device)

        return self