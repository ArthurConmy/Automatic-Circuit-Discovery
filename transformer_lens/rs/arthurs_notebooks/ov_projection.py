#%% [markdown] [4]:

from transformer_lens.cautils.notebook import *

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
LAYER_IDX, HEAD_IDX = NEG_HEADS[model.cfg.model_name]


#%%

dataset = {
    " station": """"Astronomy Picture of the Day Discover the cosmos! Each day a different image or photograph of our 
fascinating universe is featured, along with a brief explanation written by a professional astronomer. 2009 January
16ISS: Reflections of EarthCredit & Copyright: Ralf VandeberghExplanation: Remarkable details are visible in this 
view of the orbiting International Space Station (ISS), recorded with a small telescope on planet Earth through a 
clear twilight sky. Seen on December 27th at about 75 degrees elevation and some 350 kilometers above the planet's 
surface, parts of the station, including the Kibo and Columbus science modules, even seem to reflect the Earth's 
lovely bluish colors. The image also shows off large power generating solar arrays on the station's 90 meter long 
integrated truss structure Just put your cursor over the picture to identify some of the major parts of the ISS""",
    " Roger": """Senators send letter to Roger GoodellSteve Delsohn reports the details surrounding the domestic 
violence charge against Panthers""",
    " overhead": """The missile as seen from the overlook appears to be at 
almost the exact same angle and position in its arc as this shot, presumably taken several seconds into the flight 
from below.Here are the two shots Kim Jong Un is supposed to be looking at, enlarged, with a zoomed-in version of 
the above shot, as well.A side by side comparison looks even more similar. The first in this series is the overhead
shot, with the following two being the Kim Jong Un shots. Obviously the overhead is significantly better 
resolution. If these are in fact edited, then the lower resolution helps to not only fit it better into the image, 
but to obscure artifacts and obvious tells that it is the same image.In addition, I attempted to lower the 
resolution of the overhead shot to fit the other two better. Again, the first image in this series is the overhead,
the second two are from the Kim""",
    " exploitation": """Thus a husband
has a clear incentive to appropriate the wife��s future quasi-rents, by divorcing her unilaterally after having 
extracted most of his quasi-rent from the marriage. This is called quasi-rent destruction.While the example is 
provably the exception, it still is helpful in illustrating the concept. Clearly if a man was able to get away with
this he would be rewarded materially for betraying his wife. But divorce theft isn��t the only option available to 
the spouse which has the other one over a barrel. They could also use this change in fortunes to renegotiate the 
terms of the marriage in their favor under threat of divorce, which economists call exploitation:Brinig and Allen 
(2000) argue that there are two different types of quasi""",
}
#%%

def normalize(tens):
    assert len(list(tens.shape)) == 1
    return tens / tens.norm()

V_HOOK_NAME = f"blocks.{LAYER_IDX}.attn.hook_v"
PATTERN_HOOK_NAME = f"blocks.{LAYER_IDX}.attn.hook_pattern"
HOOK_ATTN_OUT = f"blocks.{LAYER_IDX}.hook_attn_out"
O_MATRIX = model.W_O[LAYER_IDX, HEAD_IDX] # [d_head, d_model]
O_BIAS = model.b_O[LAYER_IDX] # [d_model]
HOOK_ATTN_RESULT = f"blocks.{LAYER_IDX}.attn.hook_result"
assert list(O_BIAS.shape) == [model.cfg.d_model]
# ! no bias added

scale_factors = [-2.0, -1.0, 0.0, 1.0, 2.0]

for suppressed_word, text in dataset.items():
    all_results = {}
    for component_to_scale in ["unembedding", "orthogonal", "full"]:
        tokens = model.to_tokens(text).tolist()[0]
        suppressed_token = int(model.to_tokens(suppressed_word)[0][1:])
        suppressed_indices = [i for i, t in enumerate(tokens) if t == suppressed_token]
        assert len(tokens)-1 not in suppressed_indices, "Should not suppress the true token"
        results = []

        for scale_factor in scale_factors:
            # cache the O and V on the forward pass
            model.reset_hooks()
            logits, cache = model.run_with_cache(
                torch.LongTensor(tokens),
                names_filter = lambda name: name in [PATTERN_HOOK_NAME, V_HOOK_NAME, HOOK_ATTN_OUT, HOOK_ATTN_RESULT],
            )
            v_act = cache[V_HOOK_NAME][0, :, HEAD_IDX] # [batch, pos, head_index, d_head]
                                                    # this has bias added
            pattern = cache[PATTERN_HOOK_NAME][0, HEAD_IDX, -2]
            normal_probs = torch.softmax(logits[0, -2], dim=-1)
            normal_loss = -torch.log(normal_probs[tokens[-1]]).item()
            print(normal_loss, model.to_string(tokens[-3:]))

            assert list(v_act.shape) == [len(tokens), model.cfg.d_head], f"{v_act.shape} != {[len(tokens), model.cfg.d_head]}"
            assert list(pattern.shape) == [len(tokens)], f"{pattern.shape} != {[len(tokens)]}"

            all_v_contributions = v_act[suppressed_indices] # shape [indices, d_head]
            all_pattern_contributions = pattern[suppressed_indices] # shape [indices]

            v_contribution = einops.einsum(
                all_v_contributions,
                all_pattern_contributions,
                "indices d, indices-> d",
            )

            ov_contribution = einops.einsum(
                v_contribution,
                O_MATRIX,
                "d_head, d_head d_model-> d_model",
            )

            # compute the two components
            unembedding_unit_vector = normalize(model.W_U[:, suppressed_token])
            unembedding_component = einops.einsum(
                unembedding_unit_vector,
                ov_contribution,
                "d_model, d_model->",
            )

            orthogonal_vector = normalize(ov_contribution - unembedding_unit_vector * unembedding_component)
            orthogonal_component = einops.einsum(
                orthogonal_vector,
                ov_contribution,
                "d_model, d_model->",
            )

            recomputed_component = unembedding_component * unembedding_unit_vector + orthogonal_component * orthogonal_vector
            assert torch.allclose(
                ov_contribution, 
                recomputed_component,
                atol=1e-5,
                rtol=1e-5,
            )

            full_contribution = cache[HOOK_ATTN_RESULT][0, -2] # ???

            def editor(
                z, 
                hook, 
                cached_contribution, 
                unembedding_component,
                unembedding_unit_vector,
                orthogonal_component,
                orthogonal_vector,
            ):
                assert z[0, -2].shape == cached_contribution.shape, f"{z[0, -2].shape} != {cached_contribution.shape}"
                new_z = z[0, -2].clone()

                new_z -= cached_contribution
                new_z += unembedding_component * unembedding_unit_vector
                new_z += orthogonal_component * orthogonal_vector

                z[0, -2] = new_z
                return z

            model.reset_hooks()

            # EDITING_IN_HOOK = f"blocks.{LAYER_IDX}.hook_resid_mid"
            EDITING_IN_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"

            if component_to_scale == "unembedding" or component_to_scale == "orthogonal":
                fwd_hooks = [(EDITING_IN_HOOK, partial(
                    editor,
                    cached_contribution = ov_contribution,
                    unembedding_component = unembedding_component * (scale_factor if component_to_scale == "unembedding" else 1.0),
                    unembedding_unit_vector = unembedding_unit_vector,
                    orthogonal_component = orthogonal_component * (scale_factor if component_to_scale == "orthogonal" else 1.0),
                    orthogonal_vector = orthogonal_vector,
                ))]

            else:
                fwd_hooks = [
                    (EDITING_IN_HOOK, partial(
                        editor,
                        cached_contribution = cache[HOOK_ATTN_RESULT][0, -2, HEAD_IDX],
                        unembedding_component = scale_factor,
                        unembedding_unit_vector = cache[HOOK_ATTN_RESULT][0, -2, HEAD_IDX],
                        orthogonal_component = 0.0,
                        orthogonal_vector = 0.0,
                    )),
                ]

            new_logits = model.run_with_hooks(
                torch.LongTensor(tokens),
                fwd_hooks=fwd_hooks,
            )

            new_probs = torch.softmax(new_logits[0, -2], dim=-1)
            assert list(new_probs.shape) == [model.cfg.d_vocab], f"{new_probs.shape} != {[model.cfg.d_vocab]}"
            results.append(-new_probs[tokens[-1]].log().detach().cpu()) # Losses
        all_results[component_to_scale] = deepcopy(results)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot(scale_factors, all_results["unembedding"], label="unembedding")
    ax.plot(scale_factors, all_results["orthogonal"], label="orthogonal")
    ax.plot(scale_factors, all_results["full"], label="full")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Loss")
    ax.set_title(f"{suppressed_word=} {int(unembedding_component.item())=} {int(orthogonal_component.item())=} {EDITING_IN_HOOK=}") #  {model.to_string(suppressed_word)}")
    ax.legend()

    plt.show()

# %%
