from utils.utils import *
from utils.plot_utils import *
from utils.data_utils import *
from utils.eval_refusal import *
from utils.attribution_utils import *
from utils.model_utils import *
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import os
import numpy as np
import torch.nn.functional as F
import pickle
from utils.eval_capability import *



model_best_layer = {'gemma-2b':15,
              'llama':11}

## HP
topk_feat_layer = 10
topk = 20
model_clamp_val = {'gemma-2b': -3,
                   'llama':-1}

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-2b", help="Base model name")
    parser.add_argument("--bz", type=int, default=64, help="batch size for normal inference")
    parser.add_argument("--la_bz", type=int, default=3, help="batch size for doing LA-IG , need to set much smaller")
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False) # rmb set to true for grads

    # Load the model
    torch_dtype = torch.bfloat16
    device = "cuda:1" # set to 1 for main device, because 0 is used for the harmbench classifier
    model_name = args.model
    model = load_tl_model(model_name,device = device, torch_dtype = torch_dtype)
    num_sae_layer = model.cfg.n_layers
    saes = load_sae(model_name,num_sae_layer,device=device, torch_dtype=torch_dtype,split_device = False)
    size = model_sizes[model_name]
    model.model_name = model_name

    best_layer = model_best_layer[model_name]
    clamp_val = model_clamp_val[model_name]


    # Load hb classifier
    hb_model = load_harmbench_classifier() # this is loaded on "cuda:1"


    N_test = 100 # number of samples to test on
    harmful_train, harmless_train, _, harmless_val = load_refusal_datasets() 

    is_base_harmless,_ = batch_single(harmless_train,model,eval_refusal=True,avg_samples=False)

    is_base_harmful,_ = batch_single(harmful_train,model,eval_refusal=True,avg_samples=False)
    harmful_train = [x for x,y in zip(harmful_train,is_base_harmful) if y]
    harmless_train = [x for x,y in zip(harmless_train,is_base_harmless) if not y]


    ## load jb bench/harmbench/advbench
    harm_ds_names = ['harmbench_test','jailbreakbench','advbench']
    harm_ds = {name:load_all_dataset(name,instructions_only=True)[:N_test] for name in harm_ds_names}

    bm_dir = f'cache/benchmarks' # cache results
    os.makedirs(bm_dir,exist_ok=True)
    base_path = os.path.join(bm_dir,f'{model_name}_base.pkl')

    ## Get base results
    if not os.path.exists(base_path):
        base_harm_safety = {}
        base_harm_refusal = {}
        base_resps = {}
        for harm_name,ds in tqdm(harm_ds.items(),total = len(harm_ds)):
            base_resp = batch_generate(ds,model,bz = args.bz)
            base_resps[harm_name] = base_resp
            base_jb = harmbench_judge(ds,base_resp,hb_model)
            base_refusal = np.mean([substring_matching_judge_fn(x) for x in base_resp])
            print (f'Base refusal on {harm_name} jb: {base_jb:.2f}, refusal: {base_refusal:.2f}')
            base_harm_safety[harm_name] = base_jb
            base_harm_refusal[harm_name] = base_refusal

        with open(base_path,'wb') as f:
            pickle.dump({'jb':base_harm_safety,'refusal':base_harm_refusal,'base_resps':base_resps},f)
    else:
        with open(base_path,'rb') as f:
            data = pickle.load(f)
            base_harm_safety = data['jb']
            base_harm_refusal = data['refusal']
            base_resps = data['base_resps']
    

    ## Activation Steering
    steer_path = os.path.join(bm_dir,f'{model_name}_steer.pkl')
    steering_vec = {k: get_steering_vec(ds,harmless_train[:len(ds)],model) for k,ds in harm_ds.items()}

    if not os.path.exists(steer_path):
        steer_harm_jb = {}
        steer_harm_refusal = {}
        steer_resps = {}
        for k,ds in harm_ds.items():
            steer_resp = batch_generate(ds,model,bz = args.bz,steering_fn = 'vec',steering_vec=steering_vec[k][best_layer])
            steer_resps[k] = steer_resp
            steer_harm_jb[k] = harmbench_judge(ds,steer_resp,hb_model)
            steer_harm_refusal[k] = np.mean([substring_matching_judge_fn(x) for x in steer_resp])
            print (f'Steering vec on {k} jb: {steer_harm_jb[k]:.2f}, refusal: {steer_harm_refusal[k]:.2f}')

        with open(steer_path,'wb') as f:
            pickle.dump({'jb':steer_harm_jb,'refusal':steer_harm_refusal,'steer_resps':steer_resps},f)
    else:
        with open(steer_path,'rb') as f:
            data = pickle.load(f)
            steer_harm_jb = data['jb']
            steer_harm_refusal = data['refusal']
            steer_resps = data['steer_resps']

    ## Do LA-IG
    harm_attr =  {}
    for harm_name,ds in tqdm(harm_ds.items(),total = len(harm_ds)):
        all_attr = []
        for i in range(0,len(ds),args.la_bz):
            attr,_,_ = linear_attribution(model,saes,ds[i:i+args.la_bz],steering_vec = steering_vec[harm_name][best_layer],interpolate_steps=10,ind_jailbreak=True)
            all_attr.append(attr)
        harm_attr[harm_name] = pad_sequence_3d(*all_attr)


    ## Get the feature sets for the baselines
    baseline_feats = defaultdict(dict)
    global_feat_dict = defaultdict(dict) # only for LA-IG and ours

    for harm_name,ds in tqdm(harm_ds.items(),total = len(harm_ds)):
        # 1) Baseline 1 (topk feature with direction using cosine similarity)
        cs_feats = topk_feat_from_cosine(steering_vec[harm_name][best_layer],saes,topk) 
        baseline_feats['cs'][harm_name] = cs_feats

        # baseline 2 (topk feature using act diff between harmful and harmless)
        act_feats = topk_feat_from_act_diff(ds,harmless_train,model,saes,topk,avg_seq='max')
        baseline_feats['act'][harm_name] = act_feats
    
    for harm_name,attr in harm_attr.items():
        ## baseline 3 (LA-IG)
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

    ## Get harmful dataset scores ##
    bm_path = os.path.join(bm_dir,f'{model_name}_baseline.pkl')
    if not os.path.exists(bm_path):
        fn_kwargs = {'val':clamp_val,'multiply':True}
        baseline_jb_scores = defaultdict(dict)
        baseline_refusal_scores = defaultdict(dict)
        for harm_name,ds in tqdm(harm_ds.items(),total = len(harm_ds)): # over all harmful datasets
            for baseline_name in baseline_feats.keys(): # over all baselines
                b_feats = baseline_feats[baseline_name][harm_name]
                if baseline_name in ['cs','act']:
                    fn_kwargs['ind']=False # is shared across samples
                else:
                    fn_kwargs['ind']=True
                baseline_resp = batch_generate(ds,model,bz = args.bz,saes=saes,steering_fn = 'sae',circuit = b_feats,fn_kwargs = fn_kwargs)
                baseline_jb = harmbench_judge(ds,baseline_resp,hb_model)
                baseline_jb_scores[harm_name][baseline_name] = baseline_jb
                baseline_refusal_scores[harm_name][baseline_name]= np.mean([substring_matching_judge_fn(x) for x in baseline_resp])
                print (f'Baseline ({baseline_name}) on {harm_name} JB: {baseline_jb_scores[harm_name][baseline_name]:.2f}')

        # save it 
        bm_path = os.path.join(bm_dir,f'{model_name}_baseline.pkl')
        with open(bm_path,'wb') as f:
            pickle.dump({'jb':baseline_jb_scores,'refusal':baseline_refusal_scores},f)


    ## CE loss on Alpaca and the Pile and refusal scores on the Alpaca

    ce_test_size = 1000 # take 1000 samples
    pile_bz = 20
    _, _, _, alpaca_ds = load_refusal_datasets(val_size=ce_test_size)  # take 500 for CE loss
    num_pile_iter = ce_test_size//pile_bz
    pile_iterator = load_pile_iterator(pile_bz,model.tokenizer,device=model.cfg.device)

    ## Get on-policy rollouts
    base_alpaca_outputs = batch_generate(alpaca_ds,model,bz = args.bz,saes=saes,steering_fn = None,max_new_tokens=256,use_tqdm=True)


    ## Eval CE loss on the Alpaca/The Pile
    base_alpaca_loss = get_ce_loss(alpaca_ds,base_alpaca_outputs,args.bz,model,use_tqdm=True)
    print (f'Base Alpaca CE Loss: {base_alpaca_loss:.3f}')

    base_pile_loss = []
    batch_no = 0
    for pile_inputs,loss_mask in pile_iterator:
        if batch_no >= num_pile_iter:
            break
        base_pile_loss.append(get_input_ce_loss(pile_inputs,loss_mask,model))
        batch_no += 1
    base_pile_loss = np.mean(base_pile_loss)
    print (f'Base Pile CE Loss: {base_pile_loss:.3f}')

    # do it for all the baselines
    all_baseline_names = ['steer']+ list(baseline_feats.keys())  # add steer
    baseline_alpaca_loss = {}
    baseline_pile_loss = {}
    for baseline_name in tqdm(all_baseline_names,total = len(all_baseline_names)):
        baseline_avg_loss = defaultdict(list) # just avg the loss over features found over different harm datasets (make it simpler)
        for harm_name in harm_ds.keys(): 
            model.reset_hooks()  # always reset
            if baseline_name == 'steer':
                model.add_hook(resid_name_filter,partial(ablate_hook,vec = steering_vec[harm_name][best_layer]))
            else:
                fn_kwargs = {'ind':False,'val':clamp_val,'multiply':True}
                if baseline_name in ['cs','act']:
                    feat_set = baseline_feats[baseline_name][harm_name]
                else:
                    feat_set = global_feat_dict[baseline_name][harm_name] # take global feats
                model.add_hook(resid_name_filter,partial(clamp_sae,saes = saes, circuit = feat_set,**fn_kwargs))
            a_loss = get_ce_loss(alpaca_ds,base_alpaca_outputs,args.bz,model)

            p_loss = []
            batch_no = 0
            for pile_inputs,loss_mask in pile_iterator:
                if batch_no >= num_pile_iter:
                    break
                p_loss.append(get_input_ce_loss(pile_inputs,loss_mask,model))
                batch_no += 1
            p_loss = np.mean(p_loss)
            baseline_avg_loss['alpaca'].append(a_loss)
            baseline_avg_loss['pile'].append(p_loss)
        
        baseline_alpaca_loss[baseline_name] = np.mean(baseline_avg_loss['alpaca'])
        baseline_pile_loss[baseline_name] = np.mean(baseline_avg_loss['pile'])

        print (f'Baseline ({baseline_name}) Alpaca CE Loss: {baseline_alpaca_loss[baseline_name]:.3f}')
        print (f'Baseline ({baseline_name}) Pile CE Loss: {baseline_pile_loss[baseline_name]:.3f}')
        
    model.reset_hooks()


    ## Refusal scores on the Alpaca

    refusal_alpaca_ds = alpaca_ds[:100]
    max_tokens = 100
    sae_clamp_range = range(10,35,5) if model_name == 'gemma-2b' else range(1,6)

    refusal_cache = os.path.join(bm_dir,f'{model_name}_refusal.pkl')

    if not os.path.exists(refusal_cache):
        all_base_refusal = []
        all_steer_refusal = []
        sae_refusal_scores = defaultdict(lambda: defaultdict(list))


        for dataset_name in tqdm(['harmbench_test','jailbreakbench','advbench'],total = 3):
            ## Base refusal score
            base_alpaca_gen = batch_generate(refusal_alpaca_ds,model,bz = args.bz,saes=saes,steering_fn = None,max_new_tokens=max_tokens)
            base_refusal_score = np.mean([substring_matching_judge_fn(x) for x in base_alpaca_gen])
            all_base_refusal.append(base_refusal_score)
            print (f'Base refusal score on {dataset_name}: {base_refusal_score:.2f}')

            ## Vec refusal score
            custom_steer_fn = partial(steer_hook,vec = steering_vec[dataset_name][best_layer]) # steer_hook adds the vec
            custom_steer_filter = lambda x: x == 'blocks.{l}.hook_resid_post'.format(l=best_layer) # only add in this layer.
            steer_alpaca_gen = batch_generate(refusal_alpaca_ds,model,bz = args.bz,steering_fn = 'custom',custom_fn =custom_steer_fn, max_new_tokens=max_tokens,custom_filter = custom_steer_filter)
            steer_refusal_score = np.mean([substring_matching_judge_fn(x) for x in steer_alpaca_gen])
            all_steer_refusal.append(steer_refusal_score)
            print (f'Steer refusal score on {dataset_name}: {steer_refusal_score:.2f}')

            for sae_baseline in baseline_feats.keys():
                if sae_baseline in ['cs','act']:
                    feat_set = baseline_feats[sae_baseline][dataset_name]
                else:
                    feat_set = global_feat_dict[sae_baseline][dataset_name]
                for clamp_val in sae_clamp_range:
                    sae_alpaca_gen = batch_generate(refusal_alpaca_ds,model,bz = bz,saes=saes,steering_fn = 'sae',circuit = feat_set,fn_kwargs = {'val': clamp_val,'ind':False,'multiply':False},max_new_tokens=100)
                    feat_refusal_score = np.mean([substring_matching_judge_fn(x) for x in sae_alpaca_gen])
                    sae_refusal_scores[sae_baseline][clamp_val].append(feat_refusal_score)
                print (f'SAE ({sae_baseline}), : {sae_refusal_scores[sae_baseline]}')

        avg_sae_refusal_scores = defaultdict(list)
        for baseline_name,clamp_items in sae_refusal_scores.items():
            for clamp_value,scores in sorted(clamp_items.items(),key=lambda x: x[0]): # sort by clamp value
                avg_sae_refusal_scores[baseline_name].append(np.mean(scores))
                    
        all_base_refusal = np.mean(all_base_refusal)
        all_steer_refusal = np.mean(all_steer_refusal)

        all_alpaca_refusal_scores = {
            'base': all_base_refusal,
            'steer': all_steer_refusal,
            **avg_sae_refusal_scores
        }
        
        with open(refusal_cache,'wb') as f:
            pickle.dump(all_alpaca_refusal_scores,f)
    
    ## Eval GSM8K
    gsm8k_ds = load_gsm8k()

    gsm8k_bz = args.bz//4 # set to lower for gsm8k - uses 8shot
    gsm8k_string = {}

    ## Base gsm8k
    model.reset_hooks()
    base_string = eval_gsm8k(gsm8k_ds,model,gsm8k_bz,use_tqdm=True)
    gsm8k_string['base'] = base_string
    print (f'Base GSM8K string acc: {base_string:.2f}')

    # baselines
    all_baseline_names = ['steer']+ list(baseline_feats.keys())  # add steer
    for baseline_name in tqdm(all_baseline_names,total = len(all_baseline_names)):
        all_gsm8k_string = []
        for harm_name in harm_ds.keys(): 
            model.reset_hooks()
            if baseline_name == 'steer':
                model.add_hook(resid_name_filter,partial(ablate_hook,vec = steering_vec[harm_name][best_layer]))
            else:
                fn_kwargs = {'ind':False,'val':clamp_val,'multiply':True}
                if baseline_name in ['cs','act']:
                    feat_set = baseline_feats[baseline_name][harm_name]
                else:
                    feat_set = global_feat_dict[baseline_name][harm_name] # take global feats
                model.add_hook(resid_name_filter,partial(clamp_sae,saes = saes, circuit = feat_set,**fn_kwargs))
            
            baseline_string = eval_gsm8k(gsm8k_ds,model,gsm8k_bz,use_tqdm=True)
            all_gsm8k_string.append(baseline_string)
        gsm8k_string[baseline_name] = np.mean(all_gsm8k_string)
    model.reset_hooks()

    for k,v in gsm8k_string.items():
        print (f'{k} Acc: {v:.2f}')


    ## Eval ARC
    arc_ds = load_arc()
    model.reset_hooks()
    arc_scores = {}
    arc_scores['base'] = eval_arc(arc_ds,model,args.bz,use_tqdm=True)

    all_baseline_names = ['steer']+ list(baseline_feats.keys())  # add steer
    for baseline_name in tqdm(all_baseline_names,total = len(all_baseline_names)):
        all_arc_scores = []
        for harm_name in harm_ds.keys(): 
            model.reset_hooks()
            if baseline_name == 'steer':
                model.add_hook(resid_name_filter,partial(ablate_hook,vec = steering_vec[harm_name][best_layer]))
            else:
                fn_kwargs = {'ind':False,'val':clamp_val,'multiply':True}
                if baseline_name in ['cs','act']:
                    feat_set = baseline_feats[baseline_name][harm_name]
                else:
                    feat_set = global_feat_dict[baseline_name][harm_name] # take global feats
                model.add_hook(resid_name_filter,partial(clamp_sae,saes = saes, circuit = feat_set,**fn_kwargs))
            all_arc_scores.append(eval_arc(arc_ds,model,args.bz,use_tqdm=False))
        arc_scores[baseline_name] = np.mean(all_arc_scores)
    model.reset_hooks()

    for k,v in arc_scores.items():
        print (f'{k} ARC acc: {v:.2f}')


if __name__ == "__main__":
    main()