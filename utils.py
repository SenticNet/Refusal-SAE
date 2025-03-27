
import torch
import torch.nn.functional as F
from einops import einsum
from functools import partial
from collections import defaultdict

retrieve_layer_fn = lambda x: int(x.split('.')[1])
resid_name_filter = lambda x: 'resid_post' in x

def format_prompt(tokenizer,prompt):
    return tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=False)

def encode_fn(prompt,model):
    return model.tokenizer(prompt,padding='longest',truncation =False,return_tensors='pt').to(model.cfg.device)

def track_grad(tensor: torch.Tensor):
    tensor.requires_grad_(True)
    tensor.retain_grad()

def sae_grad_hook(act,hook,saes,grad_cache,clamp_circuit={}, alpha=1):  
    """
    Hook to replace activation with sae activation and store the gradients of the sae feats
    some saes might be on separate device.
    """
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = (act - x_hat).detach().clone()
    clamp_feat = clamp_circuit.get(retrieve_layer_fn(hook.name),[])
    if len(clamp_feat):
        f = f.clone()
        f[:,:,clamp_feat] = 0
    f = f * alpha
    res = res * alpha
    ablated_x_hat = saes[hook.name].decode(f).to(act.device)
    f = f.to(act.device) # make sure f is on the same device as act
    track_grad(f)
    grad_cache[retrieve_layer_fn(hook.name)] = f
    return ablated_x_hat + res

def sae_bwd_hook(output_grads: torch.Tensor, hook):  
    return (output_grads,)

def clamp_sae(act,hook,saes,circuit,pos = 'all',val = 0,multiply=False):
    """
    Hook to replace activation with sae activation while ablating certain features
    """
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    x_hat = saes[hook.name].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(retrieve_layer_fn(hook.name),[])
    if len(feats):
        if pos == 'all':
            token_pos = slice(None)
        else:
            token_pos = pos
        if multiply:
            f[:,token_pos,feats] *= val
        else:
            f[:,token_pos,feats] = val
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

def store_sae_feat(act,hook,saes,cache):
    """
    Hook to store the sae features
    """
    f = saes[hook.name].encode(act.to(saes[hook.name].device))
    cache[retrieve_layer_fn(hook.name)] = f.to(act.device)
    return act

def steer_hook(act,hook,vec):
    """
    Hook to steer the activations
    """
    return act + vec

def ablate_hook(act,hook,vec):
    """
    Hook to ablate the direction
    """
    norm_dir = F.normalize(vec,dim = -1)
    proj = einsum(act, norm_dir.unsqueeze(-1), 'b c d, d s-> b c s')
    return act - (proj * norm_dir)

def get_steering_vec(harmful,harmless,model):
    harmful_inps = encode_fn([format_prompt(model.tokenizer,x) for x in harmful],model)
    harmless_inps = encode_fn([format_prompt(model.tokenizer,x) for x in harmless],model)

    _,harmful_cache = model.run_with_cache(harmful_inps.input_ids,attention_mask = harmful_inps.attention_mask,names_filter = resid_name_filter)
    _,harmless_cache = model.run_with_cache(harmless_inps.input_ids,attention_mask = harmless_inps.attention_mask,names_filter = resid_name_filter)

    steering_vec = {retrieve_layer_fn(k):harmful_cache[k][:,-1].mean(0) - harmless_cache[k][:,-1].mean(0) for k in harmful_cache.keys()}
    del harmful_cache,harmless_cache
    torch.cuda.empty_cache()
    return steering_vec


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


def get_random_circuit(circuit,feat_size):
    """
    given a circuit {l:[f1,f2,...]}, return a random circuit of the same size per layer. Random sample feats from feat_size
    """
    random_circuit = {}
    for l,feats in circuit.items():
        random_circuit[l] = torch.randint(0,feat_size,(len(feats),)).tolist()
    return random_circuit
