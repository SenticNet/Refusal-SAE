from utils.eval_refusal import *
from utils.data_utils import *
from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser
import os
import torch
import pickle


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-2b", help="Base model name")
    parser.add_argument("--bz", type=int, default=16, help="batch size for normal inference")
    parser.add_argument("--use_vllm", type=bool, default = False, help="use vllm for harmbench")
    parser.add_argument("--dataset", type=str, default="benchmark", choices=['benchmark', 'cat_harm'])
    args = parser.parse_args()


    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_grad_enabled(False) # rmb set to true for grads

    hb_model = load_harmbench_classifier(use_vllm= args.use_vllm)

    if args.dataset == 'benchmark':
        bm_dir = f'cache/benchmarks'
        base_dict = pickle.load(open(os.path.join(bm_dir,f'{args.model}_base.pkl'), 'rb'))
        steer_dict = pickle.load(open(os.path.join(bm_dir,f'{args.model}_steer.pkl'), 'rb'))
        baseline_dict = pickle.load(open(os.path.join(bm_dir,f'{args.model}_baseline.pkl'), 'rb'))
        base_resp = base_dict['base_resps']
        steer_resp = steer_dict['steer_resps']
        baseline_resp = baseline_dict['baseline_resps']



        # load dataset
        N_test = 100 # number of samples to test on
        harm_ds_names = ['harmbench_test','jailbreakbench','advbench']
        harm_ds = {name:load_all_dataset(name,instructions_only=True)[:N_test] for name in harm_ds_names}

        base_jb = {}
        steer_jb = {}
        baseline_jb = defaultdict(dict)
        for harm_name,ds in harm_ds.items():
            base_jb[harm_name] = harmbench_judge(ds,base_resp[harm_name],hb_model,bz = args.bz)
            steer_jb[harm_name] = harmbench_judge(ds,steer_resp[harm_name],hb_model,bz = args.bz)
            for baseline_name,resp in baseline_resp[harm_name].items():
                baseline_jb[harm_name][baseline_name] = harmbench_judge(ds,resp,hb_model,bz = args.bz)
        
        bm_path = os.path.join(bm_dir,f'{args.model}_baseline.pkl')
        pickle.dump({'jb':baseline_jb,'baseline_resps':baseline_resp}, open(bm_path, 'wb'))

        bm_path = os.path.join(bm_dir,f'{args.model}_base.pkl')
        pickle.dump({'jb':base_jb,'base_resps':base_resp}, open(bm_path, 'wb'))

        bm_path = os.path.join(bm_dir,f'{args.model}_steer.pkl')
        pickle.dump({'jb':steer_jb,'steer_resps':steer_resp}, open(bm_path, 'wb'))
    else:
        response_path = f'cache/{"gemma" if "gemma" in args.model else "llama"}_cat_harm_responses.pkl'
        with open(response_path, 'rb') as f:
            all_responses = pickle.load(f)

        ## load dataset
        harm_cats = ['Illegal Activity','Child Abuse','Hate/Harass/Violence','Physical Harm','Economic Harm','Fraud/Deception','Adult Content'] 
        cat_harmful_dataset = load_dataset("declare-lab/CategoricalHarmfulQA",split = 'en').to_list()
        cat_harm_ds = defaultdict(list)
        for d in cat_harmful_dataset: 
            if d['Category'] not in harm_cats:
                continue
            cat_harm_ds[d['Category']].append(d['Question'])
        
        cat_harm_scores = defaultdict(dict)
        experiment_1_scores = defaultdict(dict)

        for cat,ds in tqdm(cat_harm_ds.items(), desc="Evaluating Harmbench Scores",total = len(cat_harm_ds)):
            specific_gen = all_responses[cat]['specific']
            experiment_1_scores[cat]['specific'] = harmbench_judge(ds,specific_gen,hb_model,bz = args.bz)
            common_gen = all_responses[cat]['common']
            experiment_1_scores[cat]['common'] = harmbench_judge(ds,common_gen,hb_model,bz = args.bz)
            all_gen = all_responses[cat]['all']
            experiment_1_scores[cat]['all'] = harmbench_judge(ds,all_gen,hb_model,bz = args.bz)


        for cat in experiment_1_scores.keys():
            specific_score = experiment_1_scores[cat]['specific']
            common_score = experiment_1_scores[cat]['common']
            all_score = experiment_1_scores[cat]['all']
            cat_harm_scores[cat]['specific'] = specific_score/ all_score
            cat_harm_scores[cat]['common'] = common_score/ all_score
        
        experiment_2_scores = defaultdict(lambda: defaultdict(dict))
        for cat,ds in cat_harm_ds.items():
            for other_cat in cat_harm_ds.keys():
                if other_cat == cat:
                    continue
                transfer_gen = all_responses[cat][other_cat]
                experiment_2_scores[cat][other_cat] = harmbench_judge(ds,transfer_gen,hb_model,bz = args.bz)
        
        for cat in experiment_2_scores.keys():
            curr_cat_specific = []
            for other_cat in experiment_2_scores[cat].keys():
                norm_score = experiment_1_scores[cat]['all']
                specific_score = experiment_2_scores[cat][other_cat]
                curr_cat_specific.append(specific_score/norm_score)
            cat_harm_scores[cat]['transfer'] = np.mean(curr_cat_specific)
        
        cache_path = f'cache/{"gemma" if "gemma" in args.model else "llama"}_cat_harm_trf.pkl'
        with open(cache_path,'wb') as f:
            pickle.dump(recursive_to_dict(cat_harm_scores),f)


if __name__ == "__main__":
    main()