from transformer_lens import HookedTransformer
from huggingface_hub import hf_hub_download,list_repo_files
import torch
from sae_lens import SAE
from utils.gemmascope import JumpReLUSAE_Base

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


model_path = {
    'gemma-2b': "google/gemma-2-2b-it",
    'gemma-9b': "google/gemma-2-9b-it",
    'llama': "meta-llama/Llama-3.1-8B-Instruct"
}
sae_naming = {
    'res': 'blocks.{l}.hook_resid_post',
    'mlp': 'blocks.{l}.hook_mlp_post',
    'attn': 'blocks.{l}.attn.hook_z',
}

model_sizes = {
    'gemma-2b':'65k',
    'gemma-9b':'16k',
    'llama':'32k'

}
sae_repo_ids = {
    'gemma-2b': "google/gemma-scope-2b-pt-res",
    'gemma-9b': "google/gemma-scope-9b-pt-res",
    'llama': "llama_scope_lxr_8x"
}




def load_tl_model(model_name,torch_dtype = torch.bfloat16,device = 'cuda'):
    m_path = model_path[model_name]
    model = HookedTransformer.from_pretrained(
            m_path,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=True,
            refactor_factored_attn_matrices=False,
            default_padding_side = 'left',
            default_prepend_bos = False,
            torch_dtype = torch_dtype,
            device = device
        ) 
    model.tokenizer.add_bos_token=False # chat models already have 
    return model

def load_sae(model_name,num_layers,device = 'cuda',torch_dtype = torch.bfloat16,split_device = False):
    saes  = {}
    repo_id = sae_repo_ids[model_name]
    size = model_sizes[model_name]
    sae_key_fn = sae_naming['res']
    if split_device:
        total_devices = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(1,total_devices)] # assume device: 0 is the main device
        device_per_layer = num_layers // (total_devices-1)
        layer_device = [list(range(device_per_layer*i,device_per_layer*(i+1))) for i in range(total_devices-1)]
    for layer in range(num_layers):
        if split_device:
            for i in range(len(layer_device)):
                if layer in set(layer_device[i]):
                    device = devices[i]
                    break
        if 'llama' in model_name:
            sae_id = f"l{layer}r_8x"
            sae,_,_ = SAE.from_pretrained(release=repo_id,sae_id = sae_id,device=device)
        else:
            sae_id = get_optimal_file(repo_id, layer,size)
            sae = JumpReLUSAE_Base.from_pretrained(repo_id, sae_id, device).to(torch_dtype).to(device)
        
        saes[sae_key_fn.format(l=layer)] = sae.to(torch_dtype)
    return saes
    

    