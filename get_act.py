import torch
def get_act_hook(fn, alt_act=None, idx=None, dim=None, name=None, message=None):
    """Return an hook that modify the activation on the fly. alt_act (Alternative activations) is a tensor of the same shape of the z.
    E.g. It can be the mean activation or the activations on other dataset."""
    if alt_act is not None:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name

            if message is not None:
                print(message)

            if (
                dim is None
            ):  # mean and z have the same shape, the mean is constant along the batch dimension
                return fn(z, alt_act, hook)
            if dim == 0:
                z[idx] = fn(z[idx], alt_act[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], alt_act[:, idx], hook)
            elif dim == 2:
                z[:, :, idx] = fn(z[:, :, idx], alt_act[:, :, idx], hook)
            return z

    else:

        def custom_hook(z, hook):
            hook.ctx["idx"] = idx
            hook.ctx["dim"] = dim
            hook.ctx["name"] = name

            if message is not None:
                print(message)

            if dim is None:
                return fn(z, hook)
            if dim == 0:
                z[idx] = fn(z[idx], hook)
            elif dim == 1:
                z[:, idx] = fn(z[:, idx], hook)
            elif dim == 2:
                print("weurhed", idx)
                assert torch.allclose(z[:, :, idx], z[:, :, idx+1])
                old_z = z[:, :, idx].clone()
                print(torch.norm(old_z))
                z[:, :, idx] = 0.0 # -= fn(old_z, hook)
                print(torch.norm(z[:, :, idx]), torch.norm(z[:, :, idx+1]))
                assert not torch.allclose(z[:, :, idx], z[:, :, idx+1])
                # print(fn)
            return z

    return custom_hook