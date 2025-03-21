
from functools import partial
from .gemmascope import JumpReLUSAE,JumpReLUSAE_Base
from typing import List, Dict
import torch
from .wrapper import AutoencoderLatents
DEVICE = "cuda:0"
from huggingface_hub import list_repo_files


def get_optimal_file(repo_id,layer,width):
    directory_path = f"layer_{layer}/width_{width}"
    files_with_l0s = [
            (f, int(f.split("_")[-1].split("/")[0]))
            for f in list_repo_files(repo_id, repo_type="model", revision="main")
            if f.startswith(directory_path) and f.endswith("params.npz")
        ]

    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return optimal_file


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


def load_gemma_autoencoders_base(model, ae_layers: list[int],size:str,types,device="cuda"):
    submodules = {}
    for layer in ae_layers:
        for type in types:
            repo_id = f"google/gemma-scope-2b-pt-{type}"
            path = get_optimal_file(repo_id,layer,size)
            sae = JumpReLUSAE_Base.from_pretrained(
                repo_id, path, device
            )

            sae.half()
            def _forward(sae, x):
                encoded = sae.encode(x)
                return encoded
            if type == "res":
                submodule = model.model.layers[layer]
            elif type == "mlp":
                submodule = model.model.layers[layer].post_feedforward_layernorm
            elif type == 'att':
                submodule = model.model.layers[layer].self_attn.o_proj
            submodule.ae = AutoencoderLatents(
                sae, partial(_forward, sae), width=sae.W_enc.shape[1]
            )
            submodules[submodule.path] = submodule

    with model.edit(" ") as edited:
        for submodule_path, submodule in submodules.items():
            if 'post_feedforward_layernorm' in submodule_path:
                acts = submodule.output
            elif 'attn' in submodule_path:
                acts = submodule.input
            else:
                acts = submodule.output[0]
            submodule.ae(acts, hook=True)
    return submodules, edited
