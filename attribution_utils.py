from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from utils import *


# Minimality
def minimize_circuit(model,saes,ds,circuit,greedy=True,approx = True,interpolate_steps=1,N=-1,bz=-1,max_seq=True):
    """
    Greedy means each iteration we pick the best feature and add to a set and compare the next remaining feat effect with the set
    Approx means we approximate the effect of each feature by using linear attribution
    interpolate is IG, approximate gradient with steps
    """
    refusal_id = model.tokenizer.encode('I',add_special_tokens=False)[0]
    flatten_circuit = []

    ## Convert circuit feats to list instead of tensor (cant use set later)
    circuit = {l:v.tolist() for l,v in circuit.items()}

    for l,feats in circuit.items():
        for feat in feats:
            flatten_circuit.append((l,feat)) 
    C_K = []
    if N == -1:
        N = len(flatten_circuit) if greedy else 1
    else:
        N = min(N,len(flatten_circuit)) # choose at most the number of features
    if approx:
        torch.set_grad_enabled(True) # allow grads
    attr_vals = []

    for iter in tqdm(range(N),total = N): # iter over N times to retrieve N best features
        model.reset_hooks() 
        current_cir_wo_K = list(set(flatten_circuit) - set(C_K))
        if approx:
            if bz == -1:
                bz = len(ds)
            
            curr_best = defaultdict(list) # sum over all batches
            for batch_i in range(0,len(ds),bz):
                ds_batch = ds[batch_i:batch_i+bz]   
                encoded_inps = encode_fn([format_prompt(model.tokenizer,x) for x in ds_batch],model)

                with torch.no_grad(): # get the current activation
                    feat_cache = {}
                    model.add_hook(resid_name_filter,partial(clamp_sae,saes = saes,circuit = sort_back(C_K)))
                    model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = feat_cache))
                    _ = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask)

                all_grads = defaultdict(list)
                for step in range(interpolate_steps):
                    grad_cache = {}
                    model.reset_hooks()
                    alpha = step/interpolate_steps
                    model.add_hook(resid_name_filter,partial(sae_grad_hook,saes = saes,grad_cache = grad_cache,clamp_circuit = sort_back(C_K),alpha = 1- alpha))
                    model.add_hook(resid_name_filter,sae_bwd_hook,'bwd')
                    logits = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask)
                    loss = -logits[:,-1,refusal_id].mean() # ablate causes logit to reduce
                    logits.grad = None
                    for k,v in grad_cache.items():
                        v.grad = None
                    loss.backward()
                    with torch.no_grad():
                        for l,acts in grad_cache.items():
                            all_grads[l].append(acts.grad.detach())
                all_grads = {k:torch.stack(v).mean(0) for k,v in all_grads.items()} # mean over steps
                for l,f in current_cir_wo_K:
                    curr_attr = (-feat_cache[l] * all_grads[l])[:,:,f]
                    if max_seq:
                        curr_attr = curr_attr.max(1).values.sum().item() # take max instead
                    else:
                        curr_attr = curr_attr.sum().item()
                    curr_best[(l,f)].append(curr_attr)
                del all_grads
                torch.cuda.empty_cache()
            curr_best = {k:np.sum(v) for k,v in curr_best.items()}
            
            if greedy:
                best_t = max(curr_best,key = curr_best.get)
                C_K.append(best_t)
                attr_vals.append(curr_best[best_t])
            else: # sort them
                curr_best = sorted(curr_best.items(),key = lambda x: x[1],reverse = True)
                C_K = [x[0] for x in curr_best]
                attr_vals = [x[1] for x in curr_best]
        else:
            curr_best = []
            for l,f in tqdm(current_cir_wo_K,total = len(current_cir_wo_K)):
                C_K_V = C_K + [(l,f)]
                C_K_circuit = current_cir_wo_K
                C_K_V_circuit = list(set(flatten_circuit) - set(C_K_V))
                _,F_K = eval_ds(model,ds,saes,feats = sort_back(C_K_circuit))
                _,F_K_V = eval_ds(model,ds,saes,feats = sort_back(C_K_V_circuit)) # shld be higher if impt
                curr_best.append((l,f,F_K_V - F_K))
            if greedy:
                best_feat = max(curr_best,key = lambda x:x[2])
                C_K.append((best_feat[0],best_feat[1]))
                attr_vals.append(best_feat[2])
            else:
                curr_best = sorted(curr_best,key = lambda x: x[2],reverse = True)
                C_K = [(l,f) for l,f,_ in curr_best]
                attr_vals = [attr for _,_,attr in curr_best]
        
    model.reset_hooks()
    torch.set_grad_enabled(False)
    model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    return C_K,attr_vals


def minimize_circuit_sample(model,saes,ds,circuit,N=-1,patch=False,steering_vec=None,cal_joint=True,store_grad = False,interpolate_steps = 1):
    """
    Minimize per sample, per token instead
    each token for each sample have a ranked list of features
    cal_joint dont do joint effect, just one forward pass.
    interpolate for IG,

    """
    refusal_id = torch.tensor(model.tokenizer.encode('I',add_special_tokens=False)[0]).to(model.cfg.device)
    jailbreak_id = torch.tensor(model.tokenizer.encode('Here',add_special_tokens=False)[0]).to(model.cfg.device)
    if not patch:
        metric_fn = lambda x: -x[:,-1,refusal_id]
    else:
        def metric_fn(x):
            jailbreak_logit = x[:,-1,jailbreak_id]
            refusal_logit = x[:,-1,refusal_id]
            return jailbreak_logit - refusal_logit


    ## Convert circuit feats to list instead of tensor (cant use set later)
    circuit = {l:v.tolist() for l,v in sorted(circuit.items(),key = lambda x: x[0])}
    total_feat = sum([len(v) for v in circuit.values()])

    if N == -1:
        N = total_feat
    if not cal_joint: # just 1 iteration.
        N = 1
    N = min(N,total_feat) # choose at most the number of features
    
    if N > 1:
        to_iter = tqdm(range(N),total = N)
    else:
        to_iter = range(N)

    if isinstance(ds,list):
        encoded_inps = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model)
    else:
        encoded_inps = ds
        
    B,S = encoded_inps['input_ids'].shape
    ## If patch, we collect the patch states once -> collect all feature vals after ablating the direction from each layer -> this push the logit to be jailbreak_id
    patch_cache = defaultdict(dict)
    if patch:
        with torch.no_grad():
            model.reset_hooks()
            model.add_hook(resid_name_filter,partial(ablate_hook,vec =steering_vec,saes = saes,cache = patch_cache,store = True,store_error=True))
            _ = model(encoded_inps['input_ids'],attention_mask = encoded_inps['attention_mask'])
            model.reset_hooks()

    torch.set_grad_enabled(True) # allow grads
    # pad token ids (ignore these pad tokens later)
    ignore_positions = []
    for row in encoded_inps['input_ids']:
        row_ignore_positions = [pos for pos, token_id in enumerate(row) if token_id == model.tokenizer.pad_token_id]
        ignore_positions.append(row_ignore_positions)

    # messy, a triple nested dict (layer,batch,token_ids) -> list of ranked features.
    starting_feats = {l : defaultdict(lambda: defaultdict()) for l in range(model.cfg.n_layers)}
    for l,feats in circuit.items():
        for b in range(B):
            for s in range(S):
                starting_feats[l][b][s] = feats

    C_K = {l: defaultdict(lambda: defaultdict(list)) for l in range(model.cfg.n_layers)}
    C_V = {l: defaultdict(lambda: defaultdict(list)) for l in range(model.cfg.n_layers)} 
    sorted_C_K = [[[] for _ in range(S)] for _ in range(B)]
    sorted_C_V = [[[] for _ in range(S)] for _ in range(B)]
    
    if interpolate_steps > 1:
        assert not cal_joint, "should only use IG if not doing joint effect else too expensive"

    for iter in to_iter: # iter over N times to retrieve N best features
        model.reset_hooks() 
        if iter  > 0:
            current_cir_wo_K = remove_overlap_nested_dict(starting_feats,C_K) # remove feats in starting from ck
        else:
            current_cir_wo_K = starting_feats

        with torch.no_grad(): # get the clean acts, simulate joint effect.
            clean_cache = defaultdict(dict)
            model.add_hook(resid_name_filter,partial(clamp_individual,saes = saes,circuit = C_K))
            model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = clean_cache,store_error=True))
            _ = model(encoded_inps['input_ids'],attention_mask = encoded_inps['attention_mask'])

        
        # add grad hooks
        all_grads = defaultdict(list)
        for step in range(interpolate_steps):
            model.reset_hooks()
            alpha = step/interpolate_steps
            grad_cache = {}
            if cal_joint or interpolate_steps == 1:
                model.add_hook(resid_name_filter,partial(sae_grad_hook,saes = saes,grad_cache = grad_cache,clamp_circuit = C_K if cal_joint else {},ind = True,alpha = 1-alpha))
            else:
                model.add_hook(resid_name_filter,partial(sae_grad_patch_IG,grad_cache = grad_cache,saes=saes,patch_cache = patch_cache,alpha=alpha))
            model.add_hook(resid_name_filter,sae_bwd_hook,'bwd')

            # get grads
            logits = model(encoded_inps['input_ids'],attention_mask = encoded_inps['attention_mask'])
            loss = metric_fn(logits).sum() # ablate causes logit to reduce
            logits.grad = None
            for v in grad_cache.values():
                v.grad = None
            loss.backward()
            with torch.no_grad():
                for l,acts in grad_cache.items():
                    all_grads[l].append(acts.grad.detach())
            del grad_cache
            clear_mem()

        all_grads = {k:torch.stack(v).mean(0) for k,v in all_grads.items()} # mean over steps

        # for each feat, get best across all feats across all layers.
        all_attrs = {}
        for l in clean_cache['feat'].keys():
            if patch:
                all_attrs[l] = (patch_cache['feat'][l] - clean_cache['feat'][l]) * all_grads[l]
            else:
                all_attrs[l] = -clean_cache['feat'][l] * all_grads[l] # (B,S,feats)
        if store_grad:
            out_grad ={l: (all_grads[l].detach().cpu(),(patch_cache['feat'][l] - clean_cache['feat'][l]).detach().cpu()) for l in range(model.cfg.n_layers)} # store grad and delta for edge computation

        attr_store = [[[] for _ in range (S)] for _ in range(B)] # each inner most list store layer x feat values
        ids_store = [[[] for _ in range (S)] for _ in range(B)] # to reference above.

        for l,nested_dict in current_cir_wo_K.items():
            for b in range(B):
                for s in range(S):
                    curr_avail_feats = nested_dict[b][s]
                    if len(curr_avail_feats):
                        for cf in curr_avail_feats:
                            attr_store[b][s].append(all_attrs[l][b,s,cf].item())
                            ids_store[b][s].append((l,cf)) # store layer, cf since we gonna max over all layer, feats

        for b in range(B):
            for s in range(S):
                if len(attr_store[b][s]):
                    if s in set(ignore_positions[b]): # ignore if the current seq_pos is pad
                        continue
                    if cal_joint:
                        best_ids = np.argmax(attr_store[b][s])
                        best_attr = attr_store[b][s][best_ids]
                        best_layer,best_feat = ids_store[b][s][best_ids]
                        C_K[best_layer][b][s].append(best_feat) # add to the list of feats
                        C_V[best_layer][b][s].append(best_attr) # add to the list of feats
                        sorted_C_K[b][s].append((best_layer,best_feat))
                        sorted_C_V[b][s].append(best_attr)
                    else:
                        sorted_ids = np.argsort(attr_store[b][s])[::-1]
                        sorted_C_K[b][s] = [ids_store[b][s][j] for j in sorted_ids]
                        sorted_C_V[b][s] = [attr_store[b][s][j] for j in sorted_ids]

        del all_grads
        del clean_cache
        model.zero_grad()
        clear_mem()

    del patch_cache
    model.reset_hooks()
    torch.set_grad_enabled(False)
    clear_mem()

    for b in range(B):
        sorted_C_K[b] = [x for x in sorted_C_K[b] if len(x)]
        sorted_C_V[b] = [x for x in sorted_C_V[b] if len(x)]

    if not store_grad:
        return sorted_C_K,sorted_C_V
    else:
        return sorted_C_K,sorted_C_V,out_grad

def linear_attribution(model,saes,ds,steering_vec,interpolate_steps = 1):
    """
    Use linear attribution to get the features for each token
    """
    refusal_id = torch.tensor(model.tokenizer.encode('I',add_special_tokens=False)[0]).to(model.cfg.device)
    jailbreak_id = torch.tensor(model.tokenizer.encode('Here',add_special_tokens=False)[0]).to(model.cfg.device)
    def metric_fn(x):
        jailbreak_logit = x[:,-1,jailbreak_id]
        refusal_logit = x[:,-1,refusal_id]
        return jailbreak_logit - refusal_logit
    
    encoded_inps = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model)
    
    patch_cache = defaultdict(dict)
    clean_cache = {}
    with torch.no_grad():
        # Get patch = steered states
        model.reset_hooks()
        model.add_hook(resid_name_filter,partial(ablate_hook,vec =steering_vec,saes = saes,cache = patch_cache,store = True,store_error=True))
        _ = model(encoded_inps.input_ids,attention_mask = encoded_inps.attention_mask)
        model.reset_hooks()

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
        loss = metric_fn(logits).mean()
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

def create_circuit_mask(circuit,threshold,topk_feat=None,clamp_val = None,device = None):
    """
    Create a mask for the circuit, using threshold, if topk_feat is given, further threshold using tha
    """
    circuit_mask = {}
    num_feats = 0
    bz = circuit[0].shape[0]
    for l, feats in circuit.items():
        if device is not None:
            feats = feats.to(device)
        mask = feats.clone() > threshold
        if topk_feat is not None:
            topk_mask = torch.zeros(feats.shape[-1], dtype=torch.bool)
            topk_mask[topk_feat[l]] = True
            mask_indices = (~topk_mask).nonzero(as_tuple=True)[0]
            mask[:, :, mask_indices] = False
        mask = ~mask # invert so those false are now 1, those true are 0
        mask = mask.to(feats.dtype)
        num_feats += (mask == 0).sum().item()
        if clamp_val is not None:
            mask = torch.where(mask == 0, clamp_val, mask)
        circuit_mask[l] = mask.to('cpu') # switch back to cpu to save ram
    return circuit_mask,num_feats/bz

def topk_match_mask(circuit, attribution_vals,clamp_val = 0):
    """
    Used to replicate the feature circuit A from another circuit of attribution values
    For each sample (row), look at how many feat values that are highlighted = clamp_val = K, then take corresponding K in the corresponding
    row of the score tensor.
    Set the highlighted ones to clamp val and rest to 1, since we want to use this as a multiplication mask on the features
    """
    out_circuit = {}
    for l, feats in circuit.items():
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

        out_circuit[l] = out_mask.view(B, D, F)

    return out_circuit

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
