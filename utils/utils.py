
import torch
import torch.nn.functional as F
from einops import einsum
from functools import partial
from collections import defaultdict
import gc
from copy import deepcopy
import numpy as np

retrieve_layer_fn = lambda x: int(x.split('.')[1])
resid_name_filter = lambda x: 'resid_post' in x
num_bos_token = {
    'gemma-2b': 1,
    'llama': 26
}

def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()

def format_prompt(tokenizer,prompt):
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=False)

def encode_fn(prompt,model):
    return model.tokenizer(prompt,padding='longest',truncation =False,return_tensors='pt').to(model.cfg.device)

def track_grad(tensor: torch.Tensor):
    tensor.requires_grad_(True)
    tensor.retain_grad()

def stop_grad(act,hook):
    act.grad = torch.zeros_like(act)

def sae_grad_hook(act,hook,saes,grad_cache,clamp_circuit={},alpha=1,ind = False):  
    """
    Hook to replace activation with sae activation and store the gradients of the sae feats
    some saes might be on separate device.
    """
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = (act - x_hat).detach().clone()
    if not ind:
        clamp_feat = clamp_circuit.get(retrieve_layer_fn(hook.name),[])
        if len(clamp_feat):
            f = f.clone()
            f[:,:,clamp_feat] = 0
    else:
        f = f.clone()
        to_clamp = clamp_circuit.get(retrieve_layer_fn(hook.name),[])
        if len(to_clamp):
            f = clamp_feat_ind(f,to_clamp,0,multiply=False) # clamp individual features for each token.
    f = f * alpha
    res = res * alpha
    ablated_x_hat = saes[hook.name].decode(f).to(act.device)
    f = f.to(act.device) # make sure f is on the same device as act
    track_grad(f)
    grad_cache[retrieve_layer_fn(hook.name)] = f
    return ablated_x_hat + res

def sae_grad_patch_IG(act,hook,saes,grad_cache,patch_cache,alpha):
    """
    Hook to directly replace the feat of the encoder.
    patch_cache contains the f and res of the patched states
    """
    if saes.get(hook.name,None) is None:
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].W_dec.device))
    clean_x_hat = saes[hook.name].decode(f).to(act.device)
    clean_res = act- clean_x_hat
    f = (1-alpha) * f + (alpha * patch_cache['feat'][retrieve_layer_fn(hook.name)].to(f.device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    track_grad(f)
    grad_cache[retrieve_layer_fn(hook.name)] = f
    res = (1-alpha) * clean_res + (alpha * patch_cache['res'][retrieve_layer_fn(hook.name)])
    return x_hat + res

def sae_bwd_hook(output_grads: torch.Tensor, hook):  
    return (output_grads,)

def clamp_sae(act,hook,saes,circuit,pos = 'all',val = 0,multiply=False,only_input=False,ind=False,retain_feats = {}):
    """
    Hook to replace activation with sae activation while ablating certain features
    val is the val to either clamp/multiply
    pos is the position to clamp/multiply
    only_input only steers the input during seq generation
    ind means features for each sample
    retain_feats contains the set of features, we want to retain the original val. (prevent upstream feature changes from affecting it.)
    """
    if (only_input and act.shape[1] == 1) or saes.get(hook.name,None) is None: # only input space or if sae layer dont exists
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[])
    if len(feats):
        if pos == 'all':
            token_pos = slice(None)
        else:
            if len(pos) > 1:
                token_pos,feats = torch.meshgrid(pos,torch.tensor(feats), indexing='ij')
            else:
                token_pos = pos

        if not ind:
            if multiply:
                f[:,token_pos,feats] *= val
            else:
                f[:,token_pos,feats] = val
        else:
            for i,sample_feat in enumerate(feats):
                if multiply:
                    f[i,token_pos,sample_feat] *= val
                else:
                    f[i,token_pos,sample_feat] = val
    if len(retain_feats): # only input space.
        if len(retain_feats['idx'].get(retrieve_layer_fn(hook.name),[])):
            retain_idx = retain_feats['idx'][retrieve_layer_fn(hook.name)]
            if act.shape[1] >1: # input space
                retain_vals = retain_feats['val'][retrieve_layer_fn(hook.name)]
                if not isinstance(retain_idx[0],list):
                    f[:,:,retain_idx] = retain_vals[:,:,retain_idx].to(f.device)
                else: # the features to retain is individual samples
                    for i,sample_feat in enumerate(retain_idx):
                        f[i,:,sample_feat] = retain_vals[i,:,sample_feat].to(f.device)
            else:
                if retain_feats.get('mean_idx',None) is not None:
                    retain_vals = retain_feats['mean_val'][retrieve_layer_fn(hook.name)] # avg over seq
                    if not isinstance(retain_idx[0],list):
                        f[:,:,retain_idx] = retain_vals[:,retain_idx].unsqueeze(1).to(f.device)
                    else:
                        for i,sample_feat in enumerate(retain_idx):
                            f[i,:,sample_feat] = retain_vals[i,sample_feat].to(f.device)
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res

def clamp_feat_by_pos(act,hook,saes,circuit,circuit_pos,val = 0): ## Only for single samples!
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[]) # feat indx
    feat_pos = circuit_pos[retrieve_layer_fn(hook.name)] # pos for each indx to clamp
    for feat,pos in zip(feats,feat_pos):
        f[:,pos:,feat] *= val
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res


def clamp_sae_mask(act,hook,saes,circuit):
    if act.shape[1] == 1 or saes.get(hook.name,None) is None: # only input space.
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[]).to(act.device)
    if len(feats):
        f *= feats.to(act.device)
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res

    
def clamp_sae_to_val(act,hook,saes,circuit,circuit_val,retain_feats = {}):
    """
    clamp only feats in the circuit to the circuit_val, should have same shape
    """
    if saes.get(hook.name,None) is None:
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[])
    max_feats = circuit_val.get(retrieve_layer_fn(hook.name),[])
    if len(feats):
        f[:,:,feats] = max_feats.expand_as(f[:,:,feats])
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    if len(retain_feats) and act.shape[1] >1: # only input space.
        if len(retain_feats['idx'].get(retrieve_layer_fn(hook.name),[])):
            retain_idx = retain_feats['idx'][retrieve_layer_fn(hook.name)]
            retain_vals = retain_feats['val'][retrieve_layer_fn(hook.name)]
            f[:,:,retain_idx] = retain_vals[:,:,retain_idx].to(f.device)
    return clamped_x_hat + res

def store_sae_feat(act,hook,saes,cache,store_error=False,detach=False):
    """
    Hook to store the sae features
    """
    if saes.get(hook.name,None) is None:
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    if store_error:
        xhat = saes[hook.name].decode(f).to(act.device)
        res = act - xhat
        cache['res'][retrieve_layer_fn(hook.name)] = res
        cache['feat'][retrieve_layer_fn(hook.name)] = f
    else:
        cache[retrieve_layer_fn(hook.name)] = f
    
    if detach:
        for k,v in cache.items():
            if isinstance(v,torch.Tensor):
                cache[k] = v.detach().cpu()
            else:
                for kk,vv in v.items():
                    cache[k][kk] = vv.detach().cpu()
    
    return act

def steer_hook(act,hook,vec):
    """
    Hook to steer the activations
    """
    return act + vec

def ablate_hook(act,hook,vec,saes=None,cache= {},store=False,store_error=False,only_input = False): ## Check
    """
    Hook to ablate the direction
    """
    if only_input and act.shape[1] == 1:
        return act
    norm_dir = F.normalize(vec,dim = -1)
    proj = einsum(act, norm_dir.unsqueeze(-1), 'b c d, d s-> b c s')
    ablated_act =  act - (proj * norm_dir)
    if store and saes.get(hook.name,None) is not None:
        f = saes[hook.name].encode(ablated_act.to(saes[hook.name].device))
        xhat = saes[hook.name].decode(f).to(act.device)
        res = act - xhat
        if store_error:
            cache['feat'][retrieve_layer_fn(hook.name)] = f
            cache['res'][retrieve_layer_fn(hook.name)] = res # store the error also for IG if using
        else:
            cache[retrieve_layer_fn(hook.name)] = f.to(act.device)
    return ablated_act

def clamp_to_circuit(act,hook,saes,circuit,circuit_val):
    """
    Given a circuit_mask and circuit, set the feat val to circuit for pos in the mask = 1
    """
    if act.shape[1] == 1:
        return act
    f = saes[hook.name].encode(act)
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    curr_mask = circuit.get(retrieve_layer_fn(hook.name),[]).to(f.device)
    if len(curr_mask):
        curr_circuit = circuit_val['feat'][retrieve_layer_fn(hook.name)].to(f.device)
        # curr_res = circuit_val['res'][retrieve_layer_fn(hook.name)].to(f.device) # set the res to the patched res as well.
        f[curr_mask] = curr_circuit[curr_mask]
    clamped_x_hat = saes[hook.name].decode(f)
    return clamped_x_hat + res
    

def get_steering_vec(harmful,harmless,model,return_separate_vectors = False): # if return_separate_vectors, return harmful_vec and harmless vec as well.
    harmful_inps = encode_fn([format_prompt(model.tokenizer,x) for x in harmful],model)
    harmless_inps = encode_fn([format_prompt(model.tokenizer,x) for x in harmless],model)

    _,harmful_cache = model.run_with_cache(harmful_inps.input_ids,attention_mask = harmful_inps.attention_mask,names_filter = resid_name_filter)
    _,harmless_cache = model.run_with_cache(harmless_inps.input_ids,attention_mask = harmless_inps.attention_mask,names_filter = resid_name_filter)

    harmful_acts = {retrieve_layer_fn(k):harmful_cache[k][:,-1] for k in harmful_cache.keys()}
    harmless_acts = {retrieve_layer_fn(k):harmless_cache[k][:,-1] for k in harmless_cache.keys()}

    steering_vec = {k: harmful_acts[k].mean(0)- harmless_acts[k].mean(0) for k in harmful_acts.keys()}
    del harmful_cache,harmless_cache
    torch.cuda.empty_cache()
    if not return_separate_vectors:
        return steering_vec
    else:
        return steering_vec, (harmful_acts,harmless_acts)


def topk_feat_sim(saes,steer_vec,topk=5,return_val = False):# diff from above, rather than looking at act value, we look at similarity with steering vec
    norm_steer_vec = F.normalize(steer_vec,dim = -1)
    sim_act = {} 
    all_sim = {}
    for layer,sae in saes.items():
        feat_vec = F.normalize(sae.W_dec,dim = -1).to(norm_steer_vec.device)
        sim = einsum(feat_vec, norm_steer_vec , 'n d, d -> n')
        sim_act[retrieve_layer_fn(layer)] = sim.topk(topk).indices
        all_sim[layer]= sim
    if return_val:
        return sim_act,all_sim
    return sim_act

def get_feat_rank_vals(model,saes,ds,feats,avg_over_dim = (0,1),select_token = None,max_seq=False): # get rank of features across samples and tokens of specific circuit
    if isinstance(feats,list):
        feats = sort_back(feats)
    layer_acts = {}
    prompts = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model)
    model.reset_hooks()
    model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = layer_acts))
    _ = model(prompts.input_ids,attention_mask = prompts.attention_mask)
    
    feat_ranks = defaultdict(list)
    feat_vals = {}
    for l,feat in feats.items():
        if avg_over_dim is not None: # else no averaging
            if select_token is None and not max_seq:
                act = layer_acts[l].mean(avg_over_dim)
            elif select_token is not None:
                act = layer_acts[l][:,select_token].mean(0)
            elif max_seq:
                act = layer_acts[l].max(dim=1).values.mean(0) # take max over seq
        else:
            act = layer_acts[l]
        if act.ndim == 1: # rank of feat in act
            sorted_act = torch.argsort(act,dim = -1,descending = True)
            ranks = torch.argsort(sorted_act,dim = -1)
            for f in feat: # get rank of f in act
                rank = ranks[f].item()
                feat_ranks[l].append(rank)
        feat_vals[l] = act[...,feat]
    model.reset_hooks()
    return feat_ranks,feat_vals

def get_feat_val(model,saes,input_ids,feats,ops = 'max',ignore_bos=True,clamp_fn = None): # clamp_fn to clamp upstream feats
    if not isinstance(input_ids,torch.Tensor):
        input_ids = encode_fn(format_prompt(model.tokenizer,input_ids),model).input_ids
    feat_cache = {}
    model.reset_hooks()
    if clamp_fn is not None:
        model.add_hook(resid_name_filter,clamp_fn)
    model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = feat_cache,detach=True))
    _ = model(input_ids)
    feat_vals = []
    for l,f in feats:
        if ignore_bos:
            bos_n = num_bos_token[model.model_name] + 1 if 'llama' in model.model_name else num_bos_token[model.model_name]
            val = feat_cache[l][0,bos_n:,f] # ignore the bos (when taking max,mean it can get skewed due to large act from bos)
        else:
            val = feat_cache[l][0,:,f]

        if ops == 'max':
            val = feat_cache[l][0,:,f].max(dim=0).values.item()
        elif ops == 'mean':
            val = feat_cache[l][0,:,f].mean(dim=0).item()
        feat_vals.append(val)
    if not isinstance(feat_vals[0],torch.Tensor):
        feat_vals = torch.tensor(feat_vals)
    else:
        feat_vals = torch.stack(feat_vals,dim = 1)
    model.reset_hooks()
    return feat_vals

def get_sae_feat_val(model,saes,ds,detach=True,is_chat=True): # just get feat val
    layer_acts = {}
    prompts = encode_fn([format_prompt(model.tokenizer,x) if is_chat else x for x in ds],model)
    model.reset_hooks()
    model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = layer_acts,detach=detach))
    _ = model(prompts.input_ids,attention_mask = prompts.attention_mask)
    model.reset_hooks()
    return layer_acts

def sort_back(circuit):
    out = defaultdict(list)
    for k,v in circuit:
        out[k].append(v)
    return out

def get_circuit_act_diff(x,y,abs=False,average_over_feats=True):
    """
    x,y is a dict of list, each list is the act value for the feature.
    set x as the higher val circuit if not abs
    average_over_feats= True if average across feats within each layer
    if both are 0, then return 0
    """
    avg_diff = {}
    for l in sorted(list(x.keys())):
        diff = x[l] - y[l]
        if abs:
            diff = diff.abs()
        avg_dist = diff/((x[l]+ y[l])/2)
        # set nan to 0 (divide by 0 case)
        avg_dist[torch.isnan(avg_dist)] = 0
        if average_over_feats:
            avg_diff[l] = avg_dist.mean().item()
        else:
            avg_diff[l] = avg_dist.tolist()
    return avg_diff

def get_circuit_overlap(circuit1,circuit2,interval=10):
    """
    Given two circuits, calculate the overlap of the circuits at each interval
    """
    if isinstance(circuit1[0][1],torch.Tensor):
        circuit1 = [(x[0],x[1].item()) for x in circuit1]
        circuit2 = [(x[0],x[1].item()) for x in circuit2]
    overlap = []
    intervals = list(range(interval,len(circuit1),interval))
    for K in intervals:
        curr_circuit1 = set(circuit1[:K])
        curr_circuit2 = set(circuit2[:K])
        overlap.append(len(curr_circuit1.intersection(curr_circuit2))/len(curr_circuit1))
    return overlap,intervals

def get_max_token_for_feat(model,saes,circuit,ds,topk=3,ignore_bos=True):
    """
    Given a circuit and list of samples, find the topk tokens for each sample, averaged over the circuit activations
    ignore_bos since bos token act tends to be extremely high.
    returns a list of topk_tokens for each sample and the tokenized ids
    """
    if isinstance(circuit,list):
        circuit = sort_back(circuit)
    ds_circuit_vals = []
    for sample in ds:
        all_feat_vals = get_sae_feat_val(model,saes,[sample])
        circuit_vals = []
        for l,feat_idx in circuit.items():
            circuit_vals.append(all_feat_vals[l][0,:,feat_idx])
        circuit_vals = torch.cat(circuit_vals,dim=-1).mean(-1) # average across feats
        ds_circuit_vals.append(circuit_vals)
    sample_topk_tokens = []
    tokenized_ids = [encode_fn(format_prompt(model.tokenizer,x),model).input_ids[0] for x in ds]
    for i in range(len((ds))):
        bos_pos = [j for j,token in enumerate(tokenized_ids[i]) if token == model.tokenizer.bos_token_id]
        sorted_token_idx = ds_circuit_vals[i].argsort(descending = True).tolist()
        if ignore_bos:
            sorted_token_idx = [x for x in sorted_token_idx if x not in bos_pos]
        sample_topk_tokens.append(sorted_token_idx[:topk])
    return sample_topk_tokens,tokenized_ids


def get_random_circuit(circuit,feat_size):
    """
    given a circuit {l:[f1,f2,...]}, return a random circuit of the same size per layer. Random sample feats from feat_size
    """
    random_circuit = {}
    for l,feats in circuit.items():
        random_circuit[l] = torch.randint(0,feat_size,(len(feats),)).tolist()
    return random_circuit

def remove_overlap_nested_dict(x,y):
    for k in y:
        for outer_key in y[k]:
            for inner_key in y[k][outer_key]:
                curr_list = y[k][outer_key][inner_key]
                if len(curr_list):
                    x[k][outer_key][inner_key] = list(set(x[k][outer_key][inner_key]) - set(curr_list))
    return x

def threshold_circuit(feats,attrs,threshold):
    """
    Filter out feats whose attribution values are below the threshold
    outputs a nested list (used circuit_list_to_dict to convert to dict form for hooking) and number of valid nodes
    """
    threshold_feats = [[] for _ in range(len(feats))]
    valid_nodes = []
    for i,sample_ in enumerate(attrs):
        valid_nodes.append(sum([(np.array(x) > threshold).sum().item() for x in sample_]))
        threshold_feats[i] = [[] for _ in range(len(sample_))]
        for j,pos_feat in enumerate(sample_):
            valid_pos = (torch.tensor(pos_feat) > threshold).nonzero(as_tuple=True)[0].tolist()
            for k,(l,f) in enumerate(feats[i][j]):
                if k in valid_pos:
                    threshold_feats[i][j].append((l,f))
    return threshold_feats,np.mean(valid_nodes)


def patchscope(model,saes,feat):
    l,f = feat
    vec = saes[f'blocks.{l}.hook_resid_post'].W_dec[f]
    target_prompt = "cat -> cat\n1135 -> 1135\nhello -> hello\n? -> "
    encoded = model.to_tokens(target_prompt)
    def set_vec(act,hook,vec):
        act[:,-1] = vec
        return act
    model.reset_hooks()
    model.add_hook(f'blocks.{model.cfg.n_layers-1}.hook_resid_post',partial(set_vec,vec = vec)) # set last layer only (others have no effect)
    logit = model(encoded)[:,-1]
    pred_tokens = model.tokenizer.batch_decode(logit.argmax(-1))
    return pred_tokens[0]

def get_pad_token_offsets(model,ds):
    sample_inp = encode_fn([format_prompt(model.tokenizer, x) for x in ds], model)
    pad_token_pos = [] 
    for i in range(sample_inp.input_ids.shape[0]):
        pad_token_pos.append((sample_inp.input_ids[i] == model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0].tolist())
    return pad_token_pos

def recursive_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: recursive_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        return {k: recursive_to_dict(v) for k, v in d.items()}
    else:
        return d
    

def pad_sequence_3d(*dicts):
    """
    Pads a list of dictionaries of 3D tensors to the maximum sequence length across all inputs.
    Each dictionary must have the same keys and tensors of shape [batch, seq_len, feature_dim].
    Returns a list of padded dictionaries in the same order.
    concat at the end too
    """
    max_seq_len = max(d[k].shape[1] for d in dicts for k in d)

    padded_dicts = []
    for d in dicts:
        padded = {}
        for k, v in d.items():
            seq_len = v.shape[1]
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                v = torch.nn.functional.pad(v, (0, 0, pad_len, 0), 'constant', 0)
            padded[k] = v
        padded_dicts.append(padded)
    
    concat_dict = defaultdict(list)
    for d in padded_dicts:
        for k,v in d.items():
            concat_dict[k].append(v)
    
    concat_dict = {key: torch.cat(value, dim=0) for key, value in concat_dict.items()}

    return concat_dict

def concat_batch_feat_dicts(dicts):
    """
    Given a dictionary containing a list of defaultdict(list), inner dict keys are the layer, values are 3D tensors, concat across the lists and push it to the inner dict key
    """
    concatenated = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            concatenated[k].append(v)
    
    concatenated = {k: torch.cat(v, dim=0) for k, v in concatenated.items()}
    return concatenated

def circuit_tolist(circuit,model,input_ids=None,remove_padding=False,ignore_bos=False):
    """
    For each sample and each sequence position, extract a list of (layer, feature_index)
    for all features where the value is 0 (i.e., important).
    convert into nested list: batch :seq:tuples (l,feat)
    if input_ids is given and remove_padding is True, we find for each seq the pad positions and remove them, also applies to the circuit. This allows easy accessing of individual token features
    """
    layers = list(circuit.keys())
    B, S, F = next(iter(circuit.values())).shape
    circuit_list = [[[] for _ in range(S)] for _ in range(B)]  # nested list: [B][S]

    for layer in layers:
        tensor = circuit[layer]  # shape: [B, S, F]
        important_pos = (tensor == 0).nonzero(as_tuple=False)  # shape: [N, 3]

        for b, s, f in important_pos:
            circuit_list[b][s].append((layer, f.item()))
    
    if remove_padding and input_ids is not None:
        new_circuit_list = []
        new_input_ids = []
        for b in range(B):
            pad_pos = (input_ids[b] == model.tokenizer.bos_token_id).nonzero(as_tuple=True)[0][-1].item() 
            if ignore_bos:
                pad_pos += num_bos_token[model.model_name]
            new_circuit_list.append(circuit_list[b][pad_pos:])
            new_input_ids.append(input_ids[b][pad_pos:])
        circuit_list = new_circuit_list
        input_ids = new_input_ids
        return circuit_list, input_ids
    else:
        return circuit_list

def topk2d(tensor,topk):
    topk_feat_ind = torch.topk(tensor.flatten(),topk).indices
    topk_layers = topk_feat_ind // tensor.shape[1]
    topk_feats = topk_feat_ind % tensor.shape[1]
    return topk_layers,topk_feats