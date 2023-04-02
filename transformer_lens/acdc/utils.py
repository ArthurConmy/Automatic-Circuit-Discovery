import wandb
from functools import partial
from copy import deepcopy
import warnings
import collections
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Literal, Dict, Tuple, Union, List, Optional, Callable, TypeVar, Generic, Iterable, Set, Type, cast, Sequence, Mapping, overload
import torch
import time
from transformer_lens.HookedTransformer import HookedTransformer
from collections import OrderedDict

class OrderedDefaultdict(collections.OrderedDict):
    """ A defaultdict with OrderedDict as its base class. 
    Thanks to https://stackoverflow.com/a/6190500/1090562"""

    def __init__(self, default_factory=None, *args, **kwargs):
        if not (default_factory is None or callable(default_factory)):
            raise TypeError('first argument must be callable or None')
        super(OrderedDefaultdict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory  # called by __missing__()

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key,)
        self[key] = value = self.default_factory()
        return value

    def __repr__(self):  # Optional.
        return '%s(%r, %r)' % (self.__class__.__name__, self.default_factory, self.items())

class EdgeType(Enum):
    """Property of edges in the computational graph - either 
    
    1. The parent adds to the child (e.g residual stream)
    2. The *single* child is a function of and only of the parent (e.g the value hooked by hook_q is a function of what hook_q_input saves)
    3. Generally like 2. but where there are generally multiple parents, so it's hard to separate effects. E.g hook_result is a function of hook_q, _k and _v but once we've computed hook_result it's difficult to edit one input"""

    ADDITION = 0
    DIRECT_COMPUTATION = 1

    def __eq__(self, other):
        # TODO WTF? Why do I need this?? To busy to look into now, check the commit where we add this later
        return self.value == other.value

class Edge:
    def __init__(
        self,
        edge_type: EdgeType,
        present: bool = True,
    ):
        self.edge_type = edge_type
        self.present = present

    def __repr__(self) -> str:
        return f"Edge({self.edge_type}, {self.present})"

# TODO attrs.frozen???
class TorchIndex:
    """There is not a clean bijection between things we 
    want in the computational graph, and things that are hooked
    (e.g hook_result covers all heads in a layer)
    
    `HookReference`s are essentially indices that say which part of the tensor is being affected. 
    
    E.g (slice(None), slice(None), 3) means index [:, :, 3]"""

    def __init__(
        self, 
        list_of_things_in_tuple
    ):
        for arg in list_of_things_in_tuple: # TODO write this less verbosely. Just typehint + check typeguard saves us??
            if type(arg) in [type(None), int]:
                continue
            else:
                assert isinstance(arg, list)
                assert all([type(x) == int for x in arg])

        self.as_index = tuple([slice(None) if x is None else x for x in list_of_things_in_tuple])
        self.hashable_tuple = tuple(list_of_things_in_tuple)

    def __hash__(self):
        return hash(self.hashable_tuple)

    def __eq__(self, other):
        return self.hashable_tuple == other.hashable_tuple

    def __repr__(self) -> str:
        ret = "["
        for idx, x in enumerate(self.hashable_tuple):
            if idx > 0:
                ret += ", "
            if x is None:
                ret += "COL"
            elif type(x) == int:
                ret += str(x)
            else:
                raise NotImplementedError(x)
        ret += "]"
        return ret

def make_nd_dict(end_type, n = 3) -> Any:
    """Make biiig default dicts : ) : )"""

    if n not in [3, 4]:
        raise NotImplementedError("Only implemented for 3/4")
        
    if n == 3:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(end_type)))

    if n == 4:
        return OrderedDefaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(end_type))))

def ct():
    return time.ctime().replace(" ", "_").replace(":", "_").replace("__", "_")