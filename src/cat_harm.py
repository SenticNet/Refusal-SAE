from utils.utils import *
from utils.plot_utils import *
from utils.data_utils import *
from utils.eval_refusal import *
from utils.attribution_utils import *
from utils.model_utils import *
from utils.neuronpedia import *
from tqdm import tqdm
from collections import defaultdict,Counter
import numpy as np
import torch.nn.functional as F
from einops import einsum
from copy import deepcopy
from argparse import ArgumentParser

model_best_layer = {'gemma-2b':15,
              'llama':11}

## HP
topk_feat_layer = 10
topk_common = 20
clamp_val = -3 # set to this because we are gonna use this to ablate smaller sets


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-2b", help="Base model name")
    parser.add_argument("--bz", type=int, default=64, help="batch size for normal inference")
    parser.add_argument("--la_bz", type=int, default=3, help="batch size for doing LA-IG , need to set much smaller")
    parser.add_argument("--print_all", action = 'store_true', help="print all the features found for each analysis, will be alot")
    parser.add_argument("--use_vllm", type=bool, default = False, help="use vllm for harmbench")
    parser.add_argument("--eval_jb", type=bool, default = False, help="eval jailbreak scores")

    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False) # rmb set to true for grads

    feature_cache = f'cache/cat_harm_{args.model}_feats.pkl'

    # Load the model
    torch_dtype = torch.bfloat16
    device = "cuda:0" # set to 0 for main device, because 1 is used for the harmbench classifier
    if not os.path.exists(feature_cache):
        if len(torch.cuda.device_count()) < 2:
            print ('Make sure you have 2 GPUs, else will run into OOM!')
        else:
            device = "cuda:1" # set to 1 for main device, because 0 is used for the harmbench classifier
    model_name = args.model
    model = load_tl_model(model_name,device = device, torch_dtype = torch_dtype)
    num_sae_layer = model.cfg.n_layers
    saes = load_sae(model_name,num_sae_layer,device=device, torch_dtype=torch_dtype,split_device = False)
    size = model_sizes[model_name]
    model.model_name = model_name

    if args.print_all: # load the explanation df to retrieve the descriptions
        interp_ds = get_explanation_df(model_name,model.cfg.n_layers)

    # Load hb classifier
    if args.eval_jb:
        hb_model = load_harmbench_classifier(use_vllm=args.use_vllm) 
    ## load harmless/CATQA ds - use the harmless to get V^*
    _, harmless_train, _, harmless_val = load_refusal_datasets()
    is_base_harmless,_ = batch_single(harmless_train,model,eval_refusal=True,avg_samples=False)
    is_val_refusal,_ = batch_single(harmless_val,model,eval_refusal=True,avg_samples=False)
    # filter out
    harmless_train = [x for x,y in zip(harmless_train,is_base_harmless) if not y]
    harmless_val = [x for x,y in zip(harmless_val,is_val_refusal) if not y]

    # only these have perf drop
    harm_cats = ['Illegal Activity','Child Abuse','Hate/Harass/Violence','Physical Harm','Economic Harm','Fraud/Deception','Adult Content'] 
    cat_harmful_dataset = load_dataset("declare-lab/CategoricalHarmfulQA",split = 'en').to_list()
    cat_harm_ds = defaultdict(list)
    for d in cat_harmful_dataset: 
        if d['Category'] not in harm_cats:
            continue
        cat_harm_ds[d['Category']].append(d['Question']) # all have 50 size
    
    steering_vec = {k: get_steering_vec(ds,harmless_train[:50],model) for k,ds in cat_harm_ds.items()} # Get refusal direction

    best_layer = model_best_layer[model_name] # best layer to get V^*

    ### Do Linear attribution-IG
    
    if not os.path.exists(feature_cache):
        cat_attr = {}
        for cat in tqdm(cat_harm_ds.keys(),total = len(cat_harm_ds)):
            all_attr= []
            for i in range(0,len(cat_harm_ds[cat]),args.la_bz):
                attr,_,_ = linear_attribution(model,saes,cat_harm_ds[cat][i:i+args.la_bz],steering_vec = steering_vec[cat][best_layer],interpolate_steps=10,ind_jailbreak=True)
                all_attr.append(attr)
            
            all_attr = pad_sequence_3d(*all_attr) # left pad them all to same seq length 
            cat_attr[cat] = all_attr

        ###############################################
        ####### Transfer experiment, section 4.2 ######
        ###############################################

        all_harm_feats = []
        cat_harm_feats = {}
        for cat in cat_attr.keys():
            topk_feats = topk_feat_sim(saes,steering_vec[cat][best_layer],topk_feat_layer)
            global_ds_feats = find_features(model,cat_harm_ds[cat],cat_attr[cat],topk_feats,topk=topk_common)['global_feat_list'] # global F^* for each cat
            cat_harm_feats[cat] = global_ds_feats
            all_harm_feats.append(global_ds_feats) 

        all_harm_feats = set(all_harm_feats[0]).intersection(*map(set,all_harm_feats[1:])) # common feats
    else:
        feature_dict = pickle.load(open(feature_cache,'rb'))
        all_harm_feats = feature_dict['common']
        cat_harm_feats = feature_dict['specific']
    if args.print_all:
        print ('Common Features across all categories:')
        for (l,f) in all_harm_feats:
            print (f'Layer {l}, Feature {f}: {get_feat_description(model_name,interp_ds,f,l)}')
        print ('--'*80)

    num_common_feats = len(all_harm_feats)
    specific_harm_feats = defaultdict(list) # specific feats
    
    for cat,harm_feats in cat_harm_feats.items():
        if args.print_all:
            print (f'Category {cat} Specific Features:')
        for (l,f) in harm_feats:
            if (l,f) in all_harm_feats:
                continue
            if args.print_all:
                print (f'Layer {l}, Feature {f}: {get_feat_description(model_name,interp_ds,f,l)}')
            if len(specific_harm_feats[cat]) < num_common_feats: # only compare between same number of feats
                specific_harm_feats[cat].append((l,f))
        if args.print_all:
            print ('--'*80)

    """
    Experiment 1) Compare similar feat effect vs dissimilar (both sets have no overlap). Clamp the other features to original val (only in input space).
    """
    ## cache it 
    cache_path = f'cache/{"gemma" if "gemma" in model_name else "llama"}_cat_harm_trf.pkl'
    response_path = f'cache/{"gemma" if "gemma" in model_name else "llama"}_cat_harm_responses.pkl'

    if not os.path.exists(cache_path):
        experiment_1_scores = defaultdict(dict)
        common_feat_dict = defaultdict(list) # common features
        for l,f in all_harm_feats:
            common_feat_dict[l].append(f)

        all_specific_feats = {} # store the specific features for each category so we can use them later
        all_responses = defaultdict(dict) # cache the responses
        for cat,ds in tqdm(cat_harm_ds.items(),total = len(cat_harm_ds),desc = 'Jailbreak between similar vs dissimilar feat'):

            data_feat_vals = get_sae_feat_val(model,saes,ds) # get the feat values of the dataset to retain
            # mean_feat_vals = {k:v[:,1:].mean(dim=1) for k,v in data_feat_vals.items()} # mean over seq (use this for the output tokens) ignore bos token.
            specific_feat_dict = defaultdict(list) # specific features
            global_ds_feats = defaultdict(list)

            for l,f in cat_harm_feats[cat]:
                global_ds_feats[l].append(f) # add the common features to the specific feat dict

            for l,f in specific_harm_feats[cat]:
                specific_feat_dict[l].append(f)

            all_specific_feats[cat] = specific_feat_dict

            # Clamp fns
            specific_fn = partial(clamp_sae,saes=saes,circuit=specific_feat_dict,multiply = True,val = clamp_val,ind=False,retain_feats = {'idx':common_feat_dict,'val':data_feat_vals}) # feature set for each sample
            common_fn = partial(clamp_sae,saes=saes,circuit=common_feat_dict,multiply = True,val = clamp_val,ind=False,retain_feats = {'idx':specific_feat_dict,'val':data_feat_vals}) # feature set for all samples
            all_fn = partial(clamp_sae,saes=saes,circuit=global_ds_feats,multiply = True,val = -3 if 'gemma-2b' in model_name else -1,ind=False) # we increase the val for refusal/harm since smaller set

            ## Specific
            specific_gen = batch_generate(ds,model,args.bz,saes = saes,steering_fn = 'custom',custom_fn =specific_fn)
            all_responses[cat]['specific'] = specific_gen # cache the responses
            experiment_1_scores[cat]['specific'] = harmbench_judge(ds,specific_gen,hb_model,bz = args.bz) if args.eval_jb else -1

            common_gen = batch_generate(ds,model,args.bz,saes = saes,steering_fn = 'custom',custom_fn =common_fn)
            all_responses[cat]['common'] = common_gen # cache the responses
            experiment_1_scores[cat]['common'] = harmbench_judge(ds,common_gen,hb_model,bz = args.bz) if args.eval_jb else -1
            
            # # Own (all feats within the target category)
            all_gen = batch_generate(ds,model,args.bz,saes = saes,steering_fn = 'custom',custom_fn =all_fn)
            all_responses[cat]['all'] = all_gen # cache the responses
            experiment_1_scores[cat]['all'] = harmbench_judge(ds,all_gen,hb_model,bz = args.bz) if args.eval_jb else -1

            if args.eval_jb:
                print (f'Cat: {cat}, Specific/Common: {experiment_1_scores[cat]["specific"]:.2f}/{experiment_1_scores[cat]["common"]:.2f}, All: {experiment_1_scores[cat]["all"]:.2f}')

        all_cat_scores = defaultdict(dict)
        for cat in experiment_1_scores.keys():
            specific_score = experiment_1_scores[cat]['specific']
            common_score = experiment_1_scores[cat]['common']
            all_score = experiment_1_scores[cat]['all']
            all_cat_scores[cat]['specific'] = specific_score/ all_score
            all_cat_scores[cat]['common'] = common_score/ all_score
        

        experiment_2_scores = defaultdict(lambda: defaultdict(dict))
        for cat,ds in tqdm(cat_harm_ds.items(),total = len(cat_harm_ds),desc = 'Jailbreak between similar vs dissimilar feat'):
            for other_cat in cat_harm_ds.keys(): # Measure on other datasets
                if other_cat == cat:
                    continue
                other_specific_feats = all_specific_feats[other_cat]
                data_feat_vals = get_sae_feat_val(model,saes,ds) # get the feat values of the dataset to retain
                # Clamp fns
                # mean_feat_vals = {k:v.mean(dim=1) for k,v in data_feat_vals.items()}
                specific_fn = partial(clamp_sae,saes=saes,circuit=other_specific_feats,multiply = True,val = clamp_val,ind=False,retain_feats = {'idx':common_feat_dict,'val':data_feat_vals}) # feature set for each sample

                ## Specific
                specific_gen = batch_generate(ds,model,args.bz,saes = saes,steering_fn = 'custom',custom_fn =specific_fn)
                all_responses[cat][other_cat] = specific_gen # cache the responses
                experiment_2_scores[cat][other_cat]['specific'] = harmbench_judge(ds,specific_gen,hb_model,bz = args.bz) if args.eval_jb else -1


        for cat in experiment_2_scores.keys():
            curr_cat_specific = []
            for other_cat in experiment_2_scores[cat].keys():
                norm_score = experiment_1_scores[cat]['all']
                specific_score = experiment_2_scores[cat][other_cat]['specific']
                curr_cat_specific.append(specific_score/norm_score)
            all_cat_scores[cat]['transfer'] = np.mean(curr_cat_specific)
            print (f'Cat: {cat}, Transfer: {all_cat_scores[cat]["transfer"]:.2f}')


        with open(cache_path,'wb') as f:
            pickle.dump(recursive_to_dict(all_cat_scores),f)

        with open(response_path,'wb') as f:
            pickle.dump(recursive_to_dict(all_responses),f)
    else:
        with open(cache_path,'rb') as f:
            all_cat_scores = pickle.load(f)
        for cat in all_cat_scores.keys():
            print (f'Cat: {cat}, Specific/Common: {all_cat_scores[cat]["specific"]:.2f}/{all_cat_scores[cat]["common"]:.2f}, Transfer: {all_cat_scores[cat]["transfer"]:.2f}')

    ###########
    ## Plot ###
    ###########

    save_path = 'images/cat_harm.png' # change here
    os.makedirs(os.path.dirname(save_path),exist_ok=True)

    cat_group_mapping = {
    'Illegal Activity': 'Ill.Act',
    'Child Abuse': 'Child.A',
    'Hate/Harass/Violence': 'H/H/V',
    'Physical Harm': 'Phys.H',
    'Economic Harm': 'Econ.H',
    'Fraud/Deception': 'Fra/Dec',
    'Adult Content': 'Adult.C'
    }
    labels_mapping = {
        'specific': 'Specific',
        'common': 'Common',
        'transfer': 'Specific (Transfer)'
    }

    categories = list(all_cat_scores.keys())
    subcats    = list(next(iter(all_cat_scores.values())).keys())
    x          = np.arange(len(categories))
    bar_w      = 0.25
    offsets    = np.linspace(-bar_w, bar_w, len(subcats))
    fig, ax    = plt.subplots(1, 1, figsize=(7, 5))
    colors     = sns.color_palette("deep", len(subcats))  # seaborn colors

    # Draw bars for each subcat
    for j, sub in enumerate(subcats):
        heights = [all_cat_scores[cat][sub] for cat in categories]
        ax.bar(x + offsets[j], heights, bar_w, label=labels_mapping[sub], color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels([cat_group_mapping[x] for x in categories], rotation=25, ha='right', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=13)

    # Y-axis label and legend
    ax.set_ylabel('Normalized Jailbreak', fontsize=14)
    ax.legend(loc='upper center', frameon=False, fontsize=12, ncol=3)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.15)

    plt.savefig(save_path,dpi=300)



if __name__ == '__main__':
    main()
    



        
        