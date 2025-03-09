# import os
# import openai
# from dotenv import load_dotenv
# load_dotenv()
# openai.api_key=os.getenv('OPENAI_KEY')
import torch
from functools import partial
# from scaling_feature_discovery.scaling_feature_discovery.gemmascope1 import JumpReluSae
from typing import Any
# from simple_parsing import Serializable
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from dataclasses import dataclass, field
import einops
# call openai skeleton
# import openai
# from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# def call_openai(prompt):
#     # openai.api_key = api_key
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.7,
#         max_tokens=500
#     )
#     return response["choices"][0]["message"]["content"]

def get_gradients(model, prompt, direction, downstream_layer=15):
    gradients = {layer: None for layer in range(downstream_layer)}
    # gradients=defaultdict(torch.Tensor)
    with torch.enable_grad():
        with model.trace(prompt) as tracer:
            for layer in range(downstream_layer):
                gradients[layer] = model.model.layers[layer].output[0].grad.save()  # ctx d_model
            activation = model.model.layers[downstream_layer].output[0]
            projection = einops.einsum(direction.half(), activation, 'dim, batch ctx dim -> batch ctx')[0, -1]
            projection.backward() 
    return gradients

def hook(module, input, output, ablation_vector,features,layer):
    if isinstance(output, tuple):
        activation = output[0]
        if len(output)>1:
            print(output[1])
    else:
        activation = output

    if activation.shape[1]>1:
        ablation_vector=einsum( ablation_vector[:,:,features[layer]],model.model.layers[layer].ae.ae.W_dec[features[layer],:],'b c i, i d -> b c d')
        activation= activation - ablation_vector
    
    if isinstance(output, tuple):
        return (activation, *output[1:])
    else:
        return activation

def feature_ablate(model_name,model, prompt,features):
    ablations={}   
    with model.trace() as tracer:
        with tracer.invoke(prompt) as invoker:
            for layer,indices in features.items():
                ablations[layer]=model.model.layers[layer].ae.output.save()           
                
    # ablations[layer]=torch.mul(activations[:,:,indices],model.model.layers[layer].ae.ae.W_dec[indices,:]).save()
    
    auto_model=AutoModelForCausalLM.from_pretrained(model_name).cuda()
    auto_tokenizer=AutoTokenizer.from_pretrained(model_name)
    hook_handles = []
   
    for name, module in auto_model.named_modules():
        if "model.layers." == name[:-1]:  # Modify all transformer layers
            if int(name.replace("model.layers.","")) not in features.keys():
                continue
            handle = module.register_forward_hook(partial(hook, ablation_vector=ablations[int(name.replace("model.layers.",""))],features=features,layer=layer))
            hook_handles.append(handle)    
    with torch.no_grad():
        inputs=auto_tokenizer(prompt, return_tensors="pt").to("cuda")
        output_ids = auto_model.generate(**inputs, max_new_tokens=100, do_sample=False)
    for handle in hook_handles:
        handle.remove()
    generated_text = auto_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text 

    
# def ablate_direction(model, prompt, direction, downstream_layer=15):
#     direction=direction/ direction.norm()
#     with torch.no_grad():
#         with model.generate(prompt, max_new_tokens=100, num_return_sequences=1, do_sample=False, scan=True, validate=True) as gen:
#             for layer in range(downstream_layer):
#                 out,cache=model.model.layers[layer].output
#                 proj= einops.einsum(direction.half(), out, 'dim, batch ctx dim -> batch ctx')
#                 # model.model.layers[layer].output = (out - proj.unsqueeze(-1) * direction,cache)
#             tokens = gen.generator.output.save() 
#         return model.tokenizer.batch_decode(tokens[0], skip_special_tokens=True)    
        
# @dataclass
# class AutoencoderConfig(Serializable):
#     model_name_or_path: str = "model"
#     device: Optional[str] = None
#     kwargs: Dict[str, Any] = field(default_factory=dict)

# class AutoencoderLatents(torch.nn.Module):
#     """
#     Unified wrapper for different types of autoencoders, compatible with nnsight.
#     """
#     def __init__(
#         self,
#         autoencoder: Any,
#         forward_function: Callable,
#         width: int,
#     ) -> None:
#         super().__init__()
#         self.ae = autoencoder
#         self._forward = forward_function
#         self.width = width
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self._forward(x)

#     @classmethod
#     def from_pretrained(
#         cls,
#         config: AutoencoderConfig,
#         hookpoint: str,
#         **kwargs
#     ):
#         device = config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
#         model_name_or_path = config.model_name_or_path

#         sae = JumpReluSae.from_pretrained(model_name_or_path,hookpoint,device)
#         forward_function = lambda x: sae.pre_acts(x)
#         width = sae.W_enc.data.shape[1]
#         return cls(sae, forward_function, width, hookpoint)
    
#     @classmethod
#     def random(cls, config: AutoencoderConfig, hookpoint: str, **kwargs):
#         pass
    
# def load_gemma_autoencoders(
#     model: Any,
#     ae_layers: list[int],
#     average_l0s: dict[int, int],
#     size: str,
#     type: str,
#     dtype: torch.dtype = torch.bfloat16,
# ):
#     submodules = {}

#     for layer in ae_layers:
#         path = f"layer_{layer}/width_{size}/average_l0_{average_l0s[layer]}"
#         sae = JumpReluSae.from_pretrained(
#             f"google/gemma-scope-2b-pt-{type}", path, "cuda"
#         )

#         sae.to(dtype)

#         def _forward(sae, x):
#             encoded = sae.encode(x)
#             return encoded

#         assert type in [
#             "res",
#             "mlp",
#         ], "Only res and mlp are supported for gemma autoencoders"
#         hookpoint = (
#             f"layers.{layer}"
#             if type == "res"
#             else f"layers.{layer}.post_feedforward_layernorm"
#         )
#         submodule=model.model.layers[layer]
#         submodule.ae = AutoencoderLatents(
#             sae, partial(_forward, sae), width=sae.W_enc.shape[1]
#         )
#         submodules[hookpoint] = submodule
        
#     with model.edit("") as edited:
#         for path, submodule in submodules.items():
#             if "embed" not in path and "mlp" not in path:
#                 acts = submodule.output[0]
#             else:
#                 acts = submodule.output
#             submodule.ae(acts, hook=True)

#     return submodules,model
