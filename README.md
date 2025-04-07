## Current Experiment

Look at `notebooks/cat_harm.ipynb` on the categorical harm dataset in https://huggingface.co/datasets/declare-lab/CategoricalHarmfulQA

1. Get topk cosine similarity per layer using the refusal direction (find the best layer via sweeping)
2. Perform Linear Attribution from https://arxiv.org/pdf/2403.19647 to compute attribution values for each feature node. Use a threshold to create feature circuits of nodes.
3. Perform an additional step of filtering using the topk cosine features, which means we further filter away features not in these topk after the threshold, the main motivation is to remove noise since the proxy loss (jailbroken token) use in attribution may not be perfect, ie picking up features related to the word itself.
4. Benchmark against the method of 2. and find that the additional step enables jailbreak
5. Perform edge attribution if desired (run `get_edges` function).


Look at code_doc.txt for information on usage of main functions.
