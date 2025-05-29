from IPython.display import IFrame
from collections import defaultdict
import requests
import pandas as pd
import os
import pickle

def get_explanation_df(model_name,num_layers):
    neuropedia_path = f'{model_name}_res_neuropedia.pkl'
    if not os.path.exists(neuropedia_path): # takes 5 min, just cache them for later use.
        saes_descriptions = defaultdict(defaultdict)
        for layer in range(num_layers):
            url = f"https://www.neuronpedia.org/api/explanation/export?modelId=gemma-2-{model_name.split('-')[-1]}&saeId={layer}-gemmascope-res-{size}"
            headers = {"Content-Type": "application/json"}

            response = requests.get(url, headers=headers)
            data = response.json()
            explanations_df = pd.DataFrame(data)
            # # rename index to "feature"
            explanations_df.rename(columns={"index": "feature"}, inplace=True)
            explanations_df["feature"] = explanations_df["feature"].astype(int)
            explanations_df["description"] = explanations_df["description"].apply(
                lambda x: x.lower()
            )
            saes_descriptions[layer]['res'] = explanations_df
        with open(neuropedia_path,'wb') as f:
            pickle.dump(saes_descriptions,f)
    else:
        with open(neuropedia_path,'rb') as f:
            saes_descriptions = pickle.load(f)
    return saes_descriptions

def get_feat_description(model_name,expl_df,feat,layer,comp = 'res',size='32k'): # get the description given feature and layer
    if 'gemma' in model_name:
        df = expl_df[layer][comp]
        try:
            return df[df["feature"] == feat]["description"].iloc[0]
        except:
            return "No description found"
    else:
        api_url = "https://www.neuronpedia.org/api/feature/llama3.1-8b/{l}-llamascope-res-{size}/{f}"
        try:
            data = requests.get(api_url.format(l=layer,f=feat,size= size)).json()
            return data["explanations"][0]["description"]
        except:
            return "No description found"