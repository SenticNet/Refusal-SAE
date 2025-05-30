from utils.utils import *
from utils.plot_utils import *
from utils.data_utils import *
from utils.eval_refusal import *
from utils.attribution_utils import *
from utils.model_utils import *
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
import pickle
from utils.eval_capability import *


model_best_layer = {'gemma-2b':15,
              'llama':11}

## HP
topk_feat_layer = 10
topk = 20

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-2b", help="Base model name")
    parser.add_argument("--bz", type=int, default=64, help="batch size for normal inference")
    parser.add_argument("--la_bz", type=int, default=3, help="batch size for doing LA-IG , need to set much smaller")
    parser.add_argument("--dataset", type=str, default="benchmark", choices=['benchmark','cat_harm'])
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False) # rmb set to true for grads

    feature_cache = f'cache/{args.dataset}_{args.model}_feats.pkl'
    torch_dtype = torch.bfloat16
    device = "cuda:0" # set to 0 for main device, because 1 is used for the harmbench classifier

    model_name = args.model
    model = load_tl_model(model_name,device = device, torch_dtype = torch_dtype)
    num_sae_layer = model.cfg.n_layers
    saes = load_sae(model_name,num_sae_layer,device=device, torch_dtype=torch_dtype,split_device = False)
    model.model_name = model_name

    best_layer = model_best_layer[model_name]
    
    # load harmless
    _, harmless_train, _, _ = load_refusal_datasets() 
    is_base_harmless,_ = batch_single(harmless_train,model,eval_refusal=True,avg_samples=False)
    harmless_train = [x for x,y in zip(harmless_train,is_base_harmless) if not y]

    if args.dataset == 'benchmark': ## load jb bench/harmbench/advbench
        N_test = 100
        harm_ds_names = ['harmbench_test','jailbreakbench','advbench']
        harm_ds = {name:load_all_dataset(name,instructions_only=True)[:N_test] for name in harm_ds_names}

    else:
        harm_cats = ['Illegal Activity','Child Abuse','Hate/Harass/Violence','Physical Harm','Economic Harm','Fraud/Deception','Adult Content'] 
        cat_harmful_dataset = load_dataset("declare-lab/CategoricalHarmfulQA",split = 'en').to_list()
        harm_ds = defaultdict(list)
        for d in cat_harmful_dataset: 
            if d['Category'] not in harm_cats:
                continue
            harm_ds[d['Category']].append(d['Question']) # all have 50 size
    
    steering_vec = {k: get_steering_vec(ds,harmless_train[:len(ds)],model) for k,ds in harm_ds.items()}

    harm_attr =  {}
    for harm_name,ds in tqdm(harm_ds.items(),total = len(harm_ds)):
        all_attr = []
        for i in range(0,len(ds),args.la_bz):
            attr,_,_ = linear_attribution(model,saes,ds[i:i+args.la_bz],steering_vec = steering_vec[harm_name][best_layer],interpolate_steps=10,ind_jailbreak=True)
            all_attr.append(attr)
        harm_attr[harm_name] = pad_sequence_3d(*all_attr)

    
    if args.dataset == 'benchmark':
        baseline_feats = defaultdict(dict)
        global_feat_dict = defaultdict(dict)

        for harm_name,attr in harm_attr.items():
            ## (LA-IG)
            la_feats = defaultdict(list)
            avg_attr = torch.stack([v.mean(1) for k,v in sorted(attr.items(),key= lambda x: x[0])]).permute(1, 0, 2) # [bz layer feat] 
            for sample_idx in range(len(avg_attr)):
                sample_feats = defaultdict(list)
                topk_layers,topk_feats = topk2d(avg_attr[sample_idx],topk)
                for l,f in zip(topk_layers,topk_feats):
                    sample_feats[l.item()].append(f.item())
                for l,feats in sample_feats.items():
                    la_feats[l].append(feats) # is gonna be a list of lists of len dataset

            global_attr = avg_attr.mean(0) # [layer,feat]
            topk_layers,topk_feats = topk2d(global_attr,topk)
                
            global_la_f =  [(l,f) for l,f in zip(topk_layers.tolist(),topk_feats.tolist())]
            global_la_feat_dict = defaultdict(list)
            for l,f in global_la_f:
                global_la_feat_dict[l].append(f)

            baseline_feats['la'][harm_name] = la_feats
            global_feat_dict['la'][harm_name] = global_la_feat_dict
            
            # Our approach
            topk_feats_dict = topk_feat_sim(saes,steering_vec[harm_name][best_layer],topk_feat_layer)
            feature_set_stuff = find_features(model,harm_ds[harm_name],attr,topk_feats_dict,topk=topk)
            global_ours_feats = feature_set_stuff['global_feat_list']
            global_ours_feat_dict = defaultdict(list)
            for l,f in global_ours_feats:
                global_ours_feat_dict[l].append(f)

            baseline_feats['our'][harm_name] = feature_set_stuff['feat_dict']
            global_feat_dict['our'][harm_name] = global_ours_feat_dict

        with open(feature_cache, 'wb') as f:
            pickle.dump({'local':baseline_feats,'global':global_feat_dict},f)
    else:
        all_harm_feats = []
        cat_harm_feats = {}
        for cat in harm_attr.keys():
            topk_feats = topk_feat_sim(saes,steering_vec[cat][best_layer],topk_feat_layer)
            global_ds_feats = find_features(model,harm_ds[cat],harm_attr[cat],topk_feats,topk=topk)['global_feat_list'] # global F^* for each cat
            cat_harm_feats[cat] = global_ds_feats
            all_harm_feats.append(global_ds_feats) 

        all_harm_feats = set(all_harm_feats[0]).intersection(*map(set,all_harm_feats[1:]))

        with open(feature_cache, 'wb') as f:
            pickle.dump({'specific':cat_harm_feats,'common':all_harm_feats},f)

if __name__ == "__main__":
    main()