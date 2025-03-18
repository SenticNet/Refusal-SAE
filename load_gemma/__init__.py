
from functools import partial
from .gemmascope import JumpReLUSAE,JumpReLUSAE_Base
from typing import List, Dict
import torch
from .wrapper import AutoencoderLatents
DEVICE = "cuda:0"




def load_gemma_autoencoders(model, ae_layers: list[int],average_l0s: Dict[int,int],size:str,type:str,device="cuda",train_encoder_only=False):
    submodules = {}

    for layer in ae_layers:
        path=f"model.layers.{layer}"
        sae = JumpReLUSAE.from_pretrained(
            "nirmalendu01/gemma-2b-it-jumprelu-saes-enc-dec" if not train_encoder_only else "nirmalendu01/gemma-2b-it-jumprelu-saes-encoderonly", path, device
        )
        
        sae.half()
        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded
        if type == "res":
            submodule = model.model.layers[layer]
        elif type == "mlp":
            submodule = model.model.layers[layer].post_feedforward_layernorm
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=sae.W_enc.shape[1]
        )

        submodules[submodule.path] = submodule

    with model.edit(" ") as edited:
        for _, submodule in submodules.items():
            if type == "res":
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited


def load_gemma_autoencoders_base(model, ae_layers: list[int],average_l0s: Dict[int,int],size:str,type:str,device="cuda"):
    submodules = {}

    for layer in ae_layers:
    
        path = f"layer_{layer}/width_{size}/average_l0_{average_l0s[layer]}"
        sae = JumpReLUSAE_Base.from_pretrained(
            f"google/gemma-scope-2b-pt-{type}", path, device
        )

        sae.half()
        def _forward(sae, x):
            encoded = sae.encode(x)
            return encoded
        if type == "res":
            submodule = model.model.layers[layer]
        elif type == "mlp":
            submodule = model.model.layers[layer].post_feedforward_layernorm
        submodule.ae = AutoencoderLatents(
            sae, partial(_forward, sae), width=sae.W_enc.shape[1]
        )

        submodules[submodule.path] = submodule

    with model.edit(" ") as edited:
        for _, submodule in submodules.items():
            if type == "res":
                acts = submodule.output[0]
            else:
                acts = submodule.output
            submodule.ae(acts, hook=True)

    return submodules, edited