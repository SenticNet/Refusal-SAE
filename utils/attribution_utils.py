from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from functools import partial
from collections import Counter
from utils.utils import *

def linear_attribution(model,saes,ds,steering_vec,interpolate_steps = 1,ind_jailbreak = False):
    """
    Use linear attribution to get the features for each token
    if ind_jailbreak, we instead get a jailbreak logit for each example.
    """
    refusal_id = torch.tensor(model.tokenizer.encode('I',add_special_tokens=False)[0]).to(model.cfg.device)
    if not ind_jailbreak:
        jailbreak_id = torch.tensor(model.tokenizer.encode('Here',add_special_tokens=False)[0]).to(model.cfg.device)

    def metric_fn(x):
        refusal_logit = x[:,-1,refusal_id]
        if not ind_jailbreak:
            jailbreak_logit = x[:,-1,jailbreak_id]
        else:
            jailbreak_logit = x[torch.arange(x.shape[0]),-1,jailbreak_id]
        return jailbreak_logit - refusal_logit
    
    encoded_inps = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model)
    
    patch_cache = defaultdict(dict)
    clean_cache = {}
    with torch.no_grad():
        # Get patch = steered states
        model.reset_hooks()
        model.add_hook(resid_name_filter,partial(ablate_hook,vec =steering_vec,saes = saes,cache = patch_cache,store = True,store_error=True))
        patched_logits = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask)
        model.reset_hooks()
        if ind_jailbreak:
            jailbreak_id = patched_logits[:,-1].argmax(-1) # use 1st token of the steered as jailbreak (customize for each sample rather than using "Here" for all)

        # get clean
        model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = clean_cache))
        _ = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask)
        model.reset_hooks()
    
    torch.set_grad_enabled(True) # allow grads
    # get grads (interpolate average across steps)
    all_grads = defaultdict(list)
    for step in range(interpolate_steps):
        model.reset_hooks()
        alpha = step/interpolate_steps
        grad_cache = {}
        model.add_hook(resid_name_filter,partial(sae_grad_patch_IG,grad_cache = grad_cache,saes=saes,patch_cache = patch_cache,alpha=alpha))
        model.add_hook(resid_name_filter,sae_bwd_hook,'bwd')
        logits = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask)
        loss = metric_fn(logits).sum()
        logits.grad = None
        for v in grad_cache.values():
            v.grad = None
        loss.backward()
        with torch.no_grad():
            for l,acts in grad_cache.items():
                all_grads[l].append(acts.grad.detach())
        del grad_cache
        torch.cuda.empty_cache()
    all_grads = {k:torch.stack(v).mean(0) for k,v in all_grads.items()}
    attrs = {}
    delta = {}
    for l,clean_acts in clean_cache.items():
        delt = patch_cache['feat'][l] - clean_acts
        attrs[l] = (delt * all_grads[l]).detach().cpu()
        delta[l] = delt.detach().cpu()
    
    all_grads = {k:v.detach().cpu() for k,v in all_grads.items()}
    model.reset_hooks()
    del patch_cache
    del clean_cache
    torch.set_grad_enabled(False)
    clear_mem()
    return attrs,all_grads,delta

def create_circuit_mask(circuit,threshold,topk_feat=None,clamp_val = None,device = None,ind_topk=False):
    """
    Create a mask for the circuit, using threshold, if topk_feat is given, further threshold using tha
    Threshold or topk - threshold select all nodes > threshold
    topk_feat is selecting only features that exist in this topk_feat per layer (2nd stage filtering.)
    """
    circuit_mask = {}
    num_feats = 0
    bz = circuit[0].shape[0]
    for l, feats in circuit.items():
        if device is not None:
            feats = feats.to(device)
        mask = feats.clone() > threshold
        if topk_feat is not None:
            if ind_topk: # topk_feat is per sample
                per_row_mask = torch.zeros_like(mask, dtype=torch.bool,device = feats.device)
                for i in range(bz):
                    per_row_mask[i,:,topk_feat[l][i]] = True
                mask *= per_row_mask
                del per_row_mask
            else: # topk_feat is fixed for all samples
                topk_mask = torch.zeros(feats.shape[-1], dtype=torch.bool,device = feats.device)
                topk_mask[topk_feat[l]] = True
                mask_indices = (~topk_mask).nonzero(as_tuple=True)[0]
                mask[:, :, mask_indices] = False
                del topk_mask
        mask = ~mask # invert so those false are now 1, those true are 0
        mask = mask.to(feats.dtype)
        num_feats += (mask == 0).sum().item()
        if clamp_val is not None:
            mask = torch.where(mask == 0, clamp_val, mask)
        circuit_mask[l] = mask.to('cpu') # switch back to cpu to save ram
    return circuit_mask,num_feats/bz

def threshold_mask(circuit,threshold,device='cuda'): # simple thresholding (pos = 1 instead of 0 in create_circuit_mask)
    circuit_mask = {}
    num_feats = 0
    bz = circuit[0].shape[0]
    for l, feats in circuit.items():
        if device is not None:
            feats = feats.to(device)
        mask = feats.clone() > threshold
        circuit_mask[l] = mask.to('cpu')
        num_feats += mask.sum().item()
    return circuit_mask,num_feats/bz




def topk_match_mask(circuit, attribution_vals,clamp_val = 0,device='cuda'):
    """
    Used to replicate the feature circuit A from another circuit of attribution values
    For each sample (row), look at how many feat values that are highlighted = clamp_val = K, then take corresponding K in the corresponding
    row of the score tensor.
    Set the highlighted ones to clamp val and rest to 1, since we want to use this as a multiplication mask on the features
    """
    out_circuit = {}
    for l, feats in circuit.items():
        feats = feats.to(device)
        attr = attribution_vals[l].to(feats.device)
        B, D, F = feats.shape
        flat_feats = feats.view(B, -1)
        flat_attr = attr.view(B, -1)
        out_mask = torch.ones_like(flat_attr, dtype=torch.float,device = feats.device)

        # Count number of positions to clamp per row
        num_topk = (flat_feats == clamp_val).sum(dim=1)

        for i in range(B):
            k = num_topk[i].item()
            if k == 0:
                continue
            # Top-k on the row directly
            _, topk_indices = torch.topk(flat_attr[i], k=k, largest=True)
            out_mask[i].index_fill_(0, topk_indices, clamp_val)  # in-place efficient

        out_circuit[l] = out_mask.view(B, D, F).to('cpu')  # switch back to cpu to save ram

    return out_circuit


def topk_feature(model,ds,attribution,threshold,topk_feat,topk):
    """
    1) Filter the attribution values using topk features set -> sparse feature node circuit
    2) For each sample, keep a count of the number of features higlighted across tokens and pick top K
    Ignore bos and pad tokens (bos have high activations and is biased)
    Outputs the topk features and inputs without pad/bos
    """
    circuit,n_feats = create_circuit_mask(attribution,threshold,topk_feat = topk_feat,device = model.cfg.device)
    if n_feats < topk:
        print (f'Warning: Not enough features {n_feats} to select from {topk}')
    encoded_inputs = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model).input_ids 
    circuit_list,non_padded_encoded_inputs = circuit_tolist(circuit,encoded_inputs,True,True)
    feat_set_dict = defaultdict(list) # each layer is a nested ragged list, where each outer list is for samples and inner is features for that layer.
    all_sample_feats = [] # each  list contains list of tuples (l,f) for each sample

    all_feat_token_activated = []
    for i,sample_circuit in enumerate(circuit_list):
        common_feats = []
        feat_token_activated = defaultdict(list) # keep track for each sample, which tokens are activated for which feature.
        for j,seq_feats in enumerate(sample_circuit):
            common_feats.extend(seq_feats)
        
        counter_feat = Counter(common_feats)
        topk_feats = counter_feat.most_common(topk)
        topk_feats = [feat[0] for feat in topk_feats]
        for j,seq_feats in enumerate(sample_circuit): # for each seq, see if the feat is in the topk_feats
            for fe in seq_feats:
                if fe in set(topk_feats):
                    feat_token_activated[fe].append(j) # for each feature, keep track of which tokens are activated
        all_feat_token_activated.append(feat_token_activated)
        all_sample_feats.append(topk_feats)
        sample_feat_set = defaultdict(list)
        for layer,feat in topk_feats:
            sample_feat_set[layer].append(feat)
        for layer in range(model.cfg.n_layers): # even when there is no feats, we add to every layer to keep the index
            feats = sample_feat_set.get(layer,[])
            feat_set_dict[layer].append(feats) 
    return {'feat_dict':feat_set_dict,'feat_list':all_sample_feats,'input':non_padded_encoded_inputs,'feat_token':all_feat_token_activated}


def clamp_circuit_to_value(circuit,true_val = 0,clamp_val = -1):
    clamped_circuit = {}
    for l,v in circuit.items():
        clamped_circuit[l] = v.clone()
        clamped_circuit[l][v == true_val] = clamp_val
    return clamped_circuit

def nested_defaultdict(depth):
    if depth == 1:
        return defaultdict()
    return defaultdict(lambda: nested_defaultdict(depth - 1))


def get_edges(model,saes,ds,circuit,grads,deltas,prior_layers = 1,clamped_val = 0.):
    """
    edges is a dict, keys -> layers, values -> 3D tensor (batch,seq,feat)
    the output is batch: dl: d_seq: df: ul: u_seq: uf and take note that seq is referring to pad positions (for d_seq and u_seq)
    """
    torch.set_grad_enabled(True)
    encoded_inps = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model)
    all_edges = nested_defaultdict(7) # 7 levels deep
    all_layers = sorted(list(circuit.keys()),reverse=True) # from highest to lowest layer
    all_layers = [dl for dl in all_layers if len((circuit[dl] == clamped_val).nonzero()) > 0] # check for important feats identified via clamped_val
    for dl_index,downstream_layer in tqdm(enumerate(all_layers),total = len(all_layers)): # from last layer to first layer
        if dl_index == len(all_layers) - 1: # no more upstream
            break
        upstream_mod = all_layers[dl_index+1:dl_index+prior_layers+1] 
        for ul in upstream_mod: # per upstream layers (run per upstream cuz need to stop grad the intermediate nodes)
            intermediate_nodes = []
            if downstream_layer - ul > 2:
                intermediate_nodes.extend(list(range(ul+1,downstream_layer)))

            model.reset_hooks()
            ul_grad_cache = {}
            dl_feat_cache = {}
            model.add_hook(f'blocks.{ul}.hook_resid_post',partial(sae_grad_hook,saes=saes,grad_cache = ul_grad_cache))
            model.add_hook(f'blocks.{downstream_layer}.hook_resid_post',partial(store_sae_feat,saes=saes,cache = dl_feat_cache))
            model.add_hook(resid_name_filter,sae_bwd_hook,'bwd')
            # stop grad for the intermediate layers
            for il in intermediate_nodes:
                model.add_hook(f'blocks.{il}.hook_resid_post',stop_grad)
            
            _ = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask) # run the model to get the gradients
            model.reset_hooks()
            to_backprop = grads[downstream_layer].to(model.cfg.device) * dl_feat_cache[downstream_layer] # dl grad * dl feat

            downstream_feat_idx = (circuit[downstream_layer] == clamped_val).nonzero()

            for b,s,f in downstream_feat_idx: # find each feature index
                to_backprop[b,s,f].backward(retain_graph=True)
                dl_to_ul = ul_grad_cache[ul].grad * deltas[ul].to(model.cfg.device) # ul grad * (patch-clean)
                for us in range(circuit[ul][b].shape): # over seq 
                    if us <= s: # only tokens at or before can influence the downstream seq
                        uf_indexes = circuit[ul][b][us].nonzero(as_tuple=True)
                        for uf in uf_indexes:
                            all_edges[b][downstream_layer][s][f][ul][us][uf] = dl_to_ul[b,us,uf].clone().detach().cpu()
            # for b in circuit[downstream_layer].keys(): # find each feature index
            #     for s in circuit[downstream_layer][b].keys():
            #         for f in circuit[downstream_layer][b][s]:
            #             to_backprop[b,s,f].backward(retain_graph=True)
            #             dl_to_ul = ul_grad_cache[ul].grad * deltas[ul].to(model.cfg.device) # ul grad * (patch-clean)

            #             for us in circuit[ul][b].keys(): # only look at same sample
            #                 if us <= s: # only tokens at or before can influence the downstream seq
            #                     for uf in circuit[ul][b][us]:
            #                         all_edges[b][downstream_layer][s][f][ul][us][uf] = dl_to_ul[b,us,uf].clone().detach().cpu() # batch: dl: seq: df: ul: us: uf
            
            del ul_grad_cache
            del dl_feat_cache
            clear_mem()

    model.reset_hooks()
    clear_mem()
    torch.set_grad_enabled(False)
    return all_edges # keys are batch: dl: dl_seq: dl.feat: ul: ul_seq: ul.feat
