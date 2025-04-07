
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
    f = saes[hook.name].encode(act)
    clean_x_hat = saes[hook.name].decode(f).to(act.device)
    clean_res = act- clean_x_hat
    f = (1-alpha) * f + (alpha * patch_cache['feat'][retrieve_layer_fn(hook.name)])
    x_hat = saes[hook.name].decode(f).to(act.device)
    track_grad(f)
    grad_cache[retrieve_layer_fn(hook.name)] = f
    res = (1-alpha) * clean_res + (alpha * patch_cache['res'][retrieve_layer_fn(hook.name)])
    return x_hat + res

def sae_bwd_hook(output_grads: torch.Tensor, hook):  
    return (output_grads,)

def clamp_sae(act,hook,saes,circuit,pos = 'all',val = 0,multiply=False,is_mask = False,only_input=False):
    """
    Hook to replace activation with sae activation while ablating certain features
    val is the val to either clamp/multiply
    pos is the position to clamp/multiply
    if is_mask is True, then apply according to feat (the pos are in circuit) -> should be 3D
    only_input only steers the input during seq generation
    """
    if only_input and act.shape[1] == 1:
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[])
    if len(feats):
        if not is_mask:
            if pos == 'all':
                token_pos = slice(None)
            else:
                token_pos = pos
            if multiply:
                f[:,token_pos,feats] *= val
            else:
                f[:,token_pos,feats] = val
        else:
            f *= feats
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res


def clamp_sae_mask(act,hook,saes,circuit):
    if act.shape[1] == 1: # only on input space.
        return act
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[]).to(act.device)
    if len(feats):
        f *= feats.to(act.device)
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res

    
def clamp_sae_to_max(act,hook,saes,circuit,max_circuit):
    """
    max_circuit is the has the max value for each feature in circuit, clamp it to that
    circuit contains the positions
    """
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[])
    max_feats = max_circuit.get(retrieve_layer_fn(hook.name),[])
    if len(feats):
        f[:,:,feats] = max_feats.expand_as(f[:,:,feats])
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res

def store_sae_feat(act,hook,saes,cache,store_error=False):
    """
    Hook to store the sae features
    """
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    if store_error:
        xhat = saes[hook.name].decode(f).to(act.device)
        res = act - xhat
        cache['res'][retrieve_layer_fn(hook.name)] = res
        cache['feat'][retrieve_layer_fn(hook.name)] = f
    else:
        cache[retrieve_layer_fn(hook.name)] = f.to(act.device)
    
    return act

def steer_hook(act,hook,vec):
    """
    Hook to steer the activations
    """
    return act + vec

def ablate_hook(act,hook,vec,saes=None,cache= {},store=False,store_error=False):
    """
    Hook to ablate the direction
    """
    norm_dir = F.normalize(vec,dim = -1)
    proj = einsum(act, norm_dir.unsqueeze(-1), 'b c d, d s-> b c s')
    ablated_act =  act - (proj * norm_dir)
    if store:
        f = saes[hook.name].encode(ablated_act)
        xhat = saes[hook.name].decode(f)
        res = act - xhat
        if store_error:
            cache['feat'][retrieve_layer_fn(hook.name)] = f
            cache['res'][retrieve_layer_fn(hook.name)] = res # store the error also for IG if using
        else:
            cache[retrieve_layer_fn(hook.name)] = f
    return ablated_act

def ablate_and_store_hook(act,hook,vec,saes,pre_cache,post_cache):
    """
    Store feat act of model pre and post vector ablation
    """
    pre_cache[retrieve_layer_fn(hook.name)] = saes[hook.name].encode(act)
    norm_dir = F.normalize(vec,dim = -1)
    proj = einsum(act, norm_dir.unsqueeze(-1), 'b c d, d s-> b c s')
    ablated =  act - (proj * norm_dir)
    f = saes[hook.name].encode(ablated)
    post_cache[retrieve_layer_fn(hook.name)] = f
    return ablated

def clamp_feat_ind(f,circuit,clamp_val=0.,multiply=False):
    for b in circuit:
        for s in circuit[b]:
            pos = circuit[b][s]
            if len(pos):
                if not multiply:
                    f[b,s,pos] = clamp_val
                else:
                    f[b,s,pos] *= clamp_val
    return f

def clamp_individual(act,hook,saes,circuit,clamp_val = 0,multiply = False):
    if act.shape[1] > 1: # clamping at sample + token level, only input
        layer = retrieve_layer_fn(hook.name)
        f = saes[hook.name].encode(act.to(saes[hook.name].device))
        x_hat = saes[hook.name].decode(f).to(act.device)
        res = act - x_hat
        f = clamp_feat_ind(f,circuit[layer],clamp_val,multiply)
        clamped_x_hat = saes[hook.name].decode(f).to(act.device)
        return clamped_x_hat + res
    else:
        return act

def clamp_to_circuit(act,hook,saes,circuit,flag = -1e6):
    """
    Given a circuit, clamp the feats that are not non_flag to the vals in the circuit
    """
    layer = retrieve_layer_fn(hook.name)
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    to_copy = circuit[layer]
    mask = to_copy != flag
    f[mask] = to_copy[mask]
    clamped_x_hat = saes[hook.name].decode(f).to(act.device)
    return clamped_x_hat + res




def get_steering_vec(harmful,harmless,model):
    harmful_inps = encode_fn([format_prompt(model.tokenizer,x) for x in harmful],model)
    harmless_inps = encode_fn([format_prompt(model.tokenizer,x) for x in harmless],model)

    _,harmful_cache = model.run_with_cache(harmful_inps.input_ids,attention_mask = harmful_inps.attention_mask,names_filter = resid_name_filter)
    _,harmless_cache = model.run_with_cache(harmless_inps.input_ids,attention_mask = harmless_inps.attention_mask,names_filter = resid_name_filter)

    steering_vec = {retrieve_layer_fn(k):harmful_cache[k][:,-1].mean(0) - harmless_cache[k][:,-1].mean(0) for k in harmful_cache.keys()}
    del harmful_cache,harmless_cache
    torch.cuda.empty_cache()
    return steering_vec

def get_topk_feat_diff(model,harmful,harmless,saes,topk=5):
    harmful_acts = get_sae_feat_val(model,saes,harmful)
    harmless_acts = get_sae_feat_val(model,saes,harmless)
    return {k:torch.topk(harmful_acts[k][:,-1].mean(0) - harmless_acts[k][:,-1].mean(0),topk).indices for k in harmful_acts.keys()}



def topk_feat_sim(saes,steer_vec,topk=5):# diff from above, rather than looking at act value, we look at similarity with steering vec
    norm_steer_vec = F.normalize(steer_vec,dim = -1)
    sim_act = {} 
    for layer,sae in saes.items():
        feat_vec = F.normalize(sae.W_dec,dim = -1).to(norm_steer_vec.device)
        sim = einsum(feat_vec, norm_steer_vec , 'n d, d -> n')
        sim_act[retrieve_layer_fn(layer)] = sim.topk(topk).indices
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

def get_sae_feat_val(model,saes,ds): # just get feat val
    layer_acts = {}
    prompts = encode_fn([format_prompt(model.tokenizer,x) for x in ds],model)
    model.reset_hooks()
    model.add_hook(resid_name_filter,partial(store_sae_feat,saes = saes,cache = layer_acts))
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


def minimize_misc(x,y,ids_list,x_list,y_list): 
    """
    given x,y which are both B,D tensor and ids which stores the mapping of the value in x to layer, feat
    push the value in the corresponding ids_list to x_list and y to ylist
    """
    B,D = x.shape
    for i in range(B):
        for j in range(D):
            l,f = ids_list[x[i,j]] # tuple
            x_list[l][i][j].append(f) # f is the feat idx
            y_list[i][j].append(y[i,j])
    return x_list,y_list

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

def circuit_list_to_dict(circuit,num_layers,pad_pos=None):
    """
    assume circuit is a nested list (batch:samples:list of tuples (layer,feat)) , pad to offset seq_pos
    Used to convert circuit in nested list form to dict form
    """
    circuit_dict = {l: defaultdict(lambda: defaultdict(list)) for l in range(num_layers)}
    for sample_pos, sample_feats in enumerate(circuit):
        max_pad_pos = max(pad_pos[sample_pos]) + 1 if len(pad_pos[sample_pos]) > 0 else 0
        for seq_pos, seq_feats in enumerate(sample_feats):
            for l,f in seq_feats:
                circuit_dict[l][sample_pos][seq_pos+max_pad_pos].append(f) 
    return circuit_dict

def create_mask_from_circuit(circuit,circuit_vals,flag = -1e6):
    """
    Given a circuit in dict form, create a similar circuit but set pos where non included nodes are -1e6 (just a flag for hook) and included nodes to circuit_val
    circuit is a double nested dict (layer:sample_pos:seq_pos:list of feats)
    circuit_vals is a dict of layer:feat:val where val = [batch,seq,feat_dim]
    output a similar circuit as circuit_vals but with the non-included nodes set to -1e6 and the included nodes set to circuit_vals
    """
    out_circuit = deepcopy(circuit_vals)
    for l in circuit:
        out_circuit[l] = torch.ones_like(out_circuit[l]) * flag
        for i in circuit[l]:
            for j in circuit[l][i]: # a list
                for f in circuit[l][i][j]:
                    out_circuit[l][i,j,f] = circuit_vals[l][i,j,f]
    return out_circuit


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

    return padded_dicts

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

def circuit_tolist(circuit):
    """
    For each sample and each sequence position, extract a list of (layer, feature_index)
    for all features where the value is 0 (i.e., important).
    convert into nested list: batch :seq:tuples (l,feat)
    """
    layers = list(circuit.keys())
    B, S, F = next(iter(circuit.values())).shape
    result = [[[] for _ in range(S)] for _ in range(B)]  # nested list: [B][S]

    for layer in layers:
        tensor = circuit[layer]  # shape: [B, S, F]
        important_pos = (tensor == 0).nonzero(as_tuple=False)  # shape: [N, 3]

        for b, s, f in important_pos:
            result[b][s].append((layer, f.item()))

    return result