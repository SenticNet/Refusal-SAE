from utils import *
from data_utils import *
from eval_refusal import *
from tqdm import tqdm
import os
from transformer_lens import utils, HookedTransformer
import torch
from load_gemma import get_optimal_file
from load_gemma.gemmascope import JumpReLUSAE,JumpReLUSAE_Base
from argparse import ArgumentParser
from sae_lens import SAE
import json


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name',type=str,required=True)
    parser.add_argument('--datasets',nargs='+',type=str,required=False,default=['jailbreakbench','harmbench_test','strongreject'])
    parser.add_argument('--bz',type=int,default=32)
    parser.add_argument('--K',nargs='+',type=int,required=True)
    parser.add_argument('--clamp_val',type=int,required=False,default = 0.)
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False) # set grads to False

    model_path = "google/gemma-2-2b-it" if args.model_name == 'gemma' else "meta-llama/Llama-3.1-8B-Instruct"
    size = '65k' if 'gemma' in args.model_name else '32k'
    device = 'cuda'
    torch_dtype = torch.float16

    ## Model 
    model = HookedTransformer.from_pretrained(
    model_path,
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    default_padding_side = 'left',
    default_prepend_bos = False,
    torch_dtype = torch_dtype,
    device = device
    )  
    ## SAE
    sae_layers = model.cfg.n_layers
    saes = {}
    comps = ['res']

    sae_naming = {
        'res': 'blocks.{l}.hook_resid_post',
        'mlp': 'blocks.{l}.hook_mlp_post',
        'attn': 'blocks.{l}.attn.hook_z',
    }
    for comp in comps:
        sae_key_fn = sae_naming[comp]
        for layer in range(sae_layers):
            if 'gemma' in args.model_name:
                repo_id = f"google/gemma-scope-2b-pt-res"
                sae_path = get_optimal_file(repo_id, layer,size)
                saes[sae_key_fn.format(l=layer)] = JumpReLUSAE_Base.from_pretrained(repo_id, sae_path, device).to(torch_dtype).to(device)
            else:
                sae,_,_=SAE.from_pretrained(release="llama_scope_lxr_8x", sae_id=f"l{layer}r_8x", device=device)
                saes[sae_key_fn.format(l=layer)] = sae.to(torch_dtype)
    
    ## Load training data get steering vec. Filter samples that base model dont refuse/refuse on harmful/harmless
    harmful_train, harmless_train, harmful_val, harmless_val = load_refusal_datasets()
    is_base_refusal,_ = eval_ds(model,harmful_train,None,steering_fn = None,average_samples=False,bz=args.bz,use_tqdm=True)
    is_base_harmless,_ = eval_ds(model,harmless_train,None,steering_fn = None,average_samples=False,bz=args.bz,use_tqdm=True)

    harmless_train = [x for x,y in zip(harmless_train,is_base_harmless) if not y]
    harmful_train = [x for x,y in zip(harmful_train,is_base_refusal) if y]

    steering_vec = get_steering_vec(harmful_train,harmless_train,model)
    
    gen_kwargs = {'max_new_tokens':512,'do_sample':False,'generate':True}
    feat_layer = 14 if 'gemma' in args.model_name else 11 # which steering vec layer to use.

    minimal_circuit = torch.load(f'circuit/{"gemma" if "gemma" in args.model_name.lower() else "llama"}_general.pt')['circuit'] # load minimal circuit

    eval_datasets = {}
    for dataset_name in args.datasets:
        loaded_ds = load_all_dataset(dataset_name,instructions_only=True)
        if 'strongreject' in dataset_name:
            loaded_ds = loaded_ds[:150]
        eval_datasets[dataset_name] = loaded_ds
    
    ## Path
    main_dir = 'completion'
    completion_path = {
        'base':'{main_dir}/{ds_name}/{model_name}_base.jsonl',
        'sae':'{main_dir}/{ds_name}/{model_name}_sae_{K}_clamp{clamp_val}.jsonl',
        'vec':'{main_dir}/{ds_name}/{model_name}_vec.jsonl'
    }
    
    
    for ds_name,ds in tqdm(eval_datasets.items(),total = len(eval_datasets), desc= 'Base and Vec Completion'):
        os.makedirs(f'{main_dir}/{ds_name}',exist_ok=True)
        for gen_type in ['base','vec']:
            curr_path = completion_path[gen_type].format(ds_name = ds_name,model_name = args.model_name,main_dir = main_dir)
            if not os.path.exists(curr_path):
                if gen_type == 'base':
                    completion = eval_ds(model,ds,saes,steering_fn = None,bz = args.bz,**gen_kwargs)
                elif gen_type == 'vec':
                    completion = eval_ds(model,ds,saes,steering_vec[feat_layer],steering_fn = 'vec',bz = args.bz,**gen_kwargs)
                # save jsonl
                saved_completion = []
                for c,p in zip(completion,ds):
                    saved_completion.append({'instruction':p,'completion':c})
                with open(curr_path,'w') as f:
                    for c in saved_completion:
                        f.write(json.dumps(c, ensure_ascii=False)+'\n')

    # SAE different K size
    for ds_name,ds in tqdm(eval_datasets.items(),total = len(eval_datasets),desc = 'SAE Completion'):
        os.makedirs(f'{main_dir}/{ds_name}',exist_ok=True)
        for K in args.K:
            if K == 0:
                K = len(minimal_circuit)
            K_circuit = sort_back(minimal_circuit[:int(K)])
            curr_path = completion_path['sae'].format(ds_name = ds_name,model_name = args.model_name,K = K,main_dir = main_dir,clamp_val=args.clamp_val)

            if not os.path.exists(curr_path):
                completion = eval_ds(model,ds,saes,feats = K_circuit,steering_fn = 'sae',bz = args.bz,clamp_value=args.clamp_val,**gen_kwargs)
                saved_completion = []
                for c,p in zip(completion,ds):
                    saved_completion.append({'instruction':p,'completion':c})
                with open(curr_path,'w') as f:
                    for c in saved_completion:
                        f.write(json.dumps(c, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
        

    
