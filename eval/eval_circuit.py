import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.utils import *
from utils.plot_utils import *
from utils.data_utils import *
from utils.eval_refusal import *
from utils.attribution_utils import *
from tqdm import tqdm
from collections import defaultdict,Counter
from utils.gemmascope import JumpReLUSAE_Base,get_optimal_file
from sae_lens import SAE
from transformer_lens import utils, HookedTransformer
import numpy as np
import torch.nn.functional as F
import pickle
from argparse import ArgumentParser

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_grad_enabled(False)

model_names = {
    'gemma-2b': "google/gemma-2-2b-it",
    'gemma-9b': "google/gemma-9-9b-it",
    'llama': "meta-llama/Llama-3.1-8B-Instruct"
}
model_best_layers = {
    'gemma-2b': 15,
    'llama':11,
}
harm_cats = ['Illegal Activity','Child Abuse','Hate/Harass/Violence','Physical Harm','Economic Harm','Fraud/Deception','Adult Content'] 

def main():
    argparser = ArgumentParser()
    argparser.add_argument('--model', type=str, default='gemma-2b',choices = ['gemma-2b','gemma-9b','llama'])
    argparser.add_argument('--bz',type = int,default = 64)
    argparser.add_argument('--topk_common',type = int,nargs='+',default = [30])
    argparser.add_argument('--threshold',type = int,nargs='+',default = [0.05])
    argparser.add_argument('--clamp_val',type = int,nargs='+',default = -1)
    argparser.add_argument('--filter_cat',type = int,nargs='+',default = [])
    argparser.add_argument('--topk_feat_layer',type = int,default = 20)
    args = argparser.parse_args()

    device = 'cuda:1' # should use 2 devices 
    torch_dtype = torch.bfloat16
    model_name = model_names[args.model]

    ## Load model and SAE
    model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    default_padding_side = 'left',
    default_prepend_bos = False,
    torch_dtype = torch_dtype,
    device = device
    )  
    model.tokenizer.add_bos_token=False

    size = '65k' if 'gemma' in model_name else '32k'
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
            if 'gemma' in model_name:
                repo_id = f"google/gemma-scope-2b-pt-res"
                sae_path = get_optimal_file(repo_id, layer,size)
                saes[sae_key_fn.format(l=layer)] = JumpReLUSAE_Base.from_pretrained(repo_id, sae_path, device).to(torch_dtype).to(device)
            else:
                sae,_,_=SAE.from_pretrained(release="llama_scope_lxr_8x", sae_id=f"l{layer}r_8x", device=device)
                saes[sae_key_fn.format(l=layer)] = sae.to(torch_dtype)
    

    # load harmless dataset
    _, harmless_train, _, _ = load_refusal_datasets()
    is_base_harmless,_ = batch_single(harmless_train,model,eval_refusal=True,avg_samples=False)
    harmless_train = [x for x,y in zip(harmless_train,is_base_harmless) if not y]

    ## load cat harmful dataset
    cat_harmful_dataset = load_dataset("declare-lab/CategoricalHarmfulQA",split = 'en').to_list()
    cat_harm_ds = defaultdict(list)
    for d in cat_harmful_dataset: 
        if d['Category'] not in harm_cats:
            continue
        if len(args.filter_cat): # to only filter for this
            if d['Category'] not in args.filter_cat:
                continue
        cat_harm_ds[d['Category']].append(d['Question']) # all have 50 size
    harmless_train = harmless_train[:50]

    # Get steering vector
    steering_vec = {k: get_steering_vec(ds,harmless_train,model) for k,ds in cat_harm_ds.items()}
    # load harmbench classifier 
    hb_model = load_harmbench_classifier()

    # Perform IG 10 steps (ind_jailbreak uses sample-wise corrupt logit)
    circuit_bz = 5 # too high OOM
    best_layer = model_best_layers[args.model]
    cat_attr = {}
    for cat in tqdm(cat_harm_ds.keys(),total = len(cat_harm_ds)):
        all_attr = []
        for i in range(0,len(cat_harm_ds[cat]),circuit_bz):
            attr,_,_ = linear_attribution(model,saes,cat_harm_ds[cat][i:i+circuit_bz],steering_vec = steering_vec[cat][best_layer],interpolate_steps=10,ind_jailbreak=True)
            all_attr.append(attr)
        all_attr = pad_sequence_3d(*all_attr) # left pad them all to same seq length 
        cat_attr[cat] = all_attr
        clear_mem()

    ## Eval ## 
    responses_cache = defaultdict(defaultdict)
    sae_scores = defaultdict(list)
    base_scores = defaultdict(list)
    vec_scores = defaultdict(list)

    to_iter = tqdm(cat_harm_ds.items(),total = len(cat_harm_ds)) if len(cat_harm_ds) > 1 else cat_harm_ds.items()
    threshold_iter = tqdm(args.threshold,total = len(args.threshold)) if len(args.threshold) > 1 else [args.threshold]

    for cat,ds in to_iter:
        base_gen = batch_generate(ds,model,args.bz)
        base_safety = harmbench_judge(ds,base_gen,hb_model)
        base_refusal = np.mean([substring_matching_judge_fn(x) for x in base_gen])
        print (f'Cat: {cat},Base safety: {base_safety:.2f}, string: {base_refusal:.2f}')
        responses_cache[cat]['base'] = base_gen
        base_scores['safety'].append(base_safety)
        base_scores['refusal'].append(base_refusal)

        # Vec Steer
        steer_gen = batch_generate(ds,model,args.bz,steering_fn = 'vec',steering_vec = steering_vec[cat][best_layer]) # only steer input (to compare against sae clamping which is only done on the input space.)   
        steer_safety = harmbench_judge(ds,steer_gen,hb_model)
        steer_refusal = np.mean([substring_matching_judge_fn(x) for x in steer_gen])
        print (f'Cat: {cat}, Vec Ablation safety: {steer_safety:.2f}, string: {steer_refusal:.2f}')
        responses_cache[cat]['vec'] = steer_gen
        vec_scores['safety'].append(steer_safety)
        vec_scores['refusal'].append(steer_refusal)
        if len(args.threshold) > 1 and len(args.topk_common) > 1:
            sae_safety_store,sae_refusal_scores,response_store = {},{},{}
        else:
            sae_safety_store,sae_refusal_scores,response_store = [],[],[]
        topk_feats = topk_feat_sim(saes,steering_vec[cat][best_layer],args.topk_feat_layer)
        for threshold in threshold_iter:
            for topk_com in args.topk_common:
                    feature_set_layer,_,_ = topk_feature(model,ds,cat_attr[cat],threshold,topk_feat = topk_feats,topk = topk_com)
                    clamp_fn = partial(clamp_sae,saes=saes,circuit=feature_set_layer,multiply = True,val = args.clamp_val,ind=True)
                    sae_gen = batch_generate(ds,model,args.bz,saes = saes,steering_fn = 'custom',custom_fn =clamp_fn)
                    sae_safety = harmbench_judge(ds,sae_gen,hb_model)
                    sae_refusal = np.mean([substring_matching_judge_fn(x) for x in sae_gen])
                    if isinstance(sae_safety_store,dict):
                        sae_safety_store[(threshold,topk_com)] = sae_safety
                        sae_refusal_scores[(threshold,topk_com)] = sae_refusal
                        response_store[(threshold,topk_com)] = sae_gen
                    else:
                        sae_safety_store.append(sae_safety)
                        sae_refusal_scores.append(sae_refusal)
                        response_store.append(sae_gen)

        if isinstance(sae_safety_store,dict):
            responses_cache[cat]['sae'] = response_store
            sae_scores = {'safety':sae_safety_store,'refusal':sae_refusal_scores}
        else:
            responses_cache[cat]['sae'] = response_store
            sae_scores['safety'].append(sae_safety_store)
            sae_scores['refusal'].append(sae_refusal_scores)
    
    ## Save results
    cache_dir = 'cache'
    result_dir = 'results'
    os.makedirs(cache_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)

    cache_path = os.path.join(cache_dir,f'{"gemma" if "gemma" in model_name else "llama"}_cat_harm.pkl')
    result_path = os.path.join(result_dir,f'{"gemma" if "gemma" in model_name else "llama"}_cat_harm_results.pkl')

    responses_cache = recursive_to_dict(responses_cache)
    score_cache = recursive_to_dict(score_cache)

    with open(cache_path,'wb') as f:
        pickle.dump(responses_cache,f)
    with open(result_path,'wb') as f:
        pickle.dump(score_cache,f)



if __name__ == '__main__':
    main()
