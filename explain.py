import torch
import simplejson as json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from torch.cuda.amp import autocast
from pathlib import Path
from huggingface_hub import snapshot_download
from decompress import decompress_file
from collections import defaultdict
from datasets import load_dataset
from dotenv import load_dotenv
import os
import argparse

load_dotenv()
openai.api_key=os.getenv('OPENAI_KEY')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
template='''<bos><start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
'''

def get_top_activation_samples_along_direction(explain_data,direction,tokenizer,model, key="text",topk=10,batch_size=32, ):
    features=defaultdict(list)
    with torch.inference_mode(), autocast(dtype=torch.float16):
        for data_indx in tqdm(range(0, len(explain_data), batch_size)):  # Batch processing
            batch_data = explain_data[data_indx:data_indx + batch_size]

            # Tokenize inputs in a batch
            inputs = tokenizer(
                [template.format(prompt=item) for item in batch_data],
                return_tensors="pt",
                max_length=256,
                padding=True,
                truncation=True
            ).to(device)

            # Model inference
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[1:17]

            for layer, hidden_state in enumerate(hidden_states):
                activation=torch.matmul(hidden_state,direction.T)[:,-1] # shape: (batch_size,)
                batch_indices = torch.arange(data_indx, data_indx + len(batch_data), device=activation.device) # Shape: (batch_size,)
                features[layer].extend(list(zip(batch_indices.tolist(),activation.tolist())))
                features[layer]=sorted(features[layer],key=lambda x: x[1],reverse=True)[:topk]
    return features

def explain(top_5_activation_records_for_feature):
    conversation = [
        {"role": "system", "content": '''We are studying features in a large language model. Each feature looks for some particular pattern in a dataset.
        Look at the parts of the dataset the feature activates for, and summarize in a single sentence what the feature is looking for.
        The activation format is prompt<tab>activation. A feature finding what it's looking for is represented by a non-zero activation value.
        The higher the activation value, the stronger the match. If there is no pattern, respond with 'nothing specific'.'''},
    ]

    for record_idx, activation_str in enumerate(top_5_activation_records_for_feature):
        user_message = f"Top Activation Example {record_idx}:\n{activation_str}"
        conversation.append({"role": "user", "content": user_message})

    conversation.append({"role": "user", "content": f"Explain what the feature in a large language model might be doing based on the top 5 activation records."})
    model_engine = "gpt-4o-mini"

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=conversation,
        temperature=0,
        max_tokens=100,
    )
    explanation = response['choices'][0]['message']['content'].strip()
    return explanation
        
def explain_single_feature(layer,samples,data,tokenizer):
    prompts=[]
    if len(set([item[1] for item in samples]))==1:
        return
    explanation=None
    # print([item[0] for item in samples])
    # print(f"layer{layer}:{sum([1 for item in samples if 'access' in data[item[0]].lower()])}")
    # print(f"layer{layer}:{sum([1 for item in samples if 'allow' in data[item[0]].lower()])}")
    print([data[item[0]][:256] for item in samples[:5]])
    for data_idx,act in samples:
        item=data[int(data_idx)]
        item=tokenizer.tokenize(item)[:256]
        item=tokenizer.convert_tokens_to_string(item)
        prompts.append(f"{item}\t{act}")
        
    try:    
        prompts=prompts[:5] 
        explanation=explain(prompts)
    except Exception as e:
        print(f"Error in explaining feature: {layer}, Exception: {e}")
    return explanation  

def explain_features_with_openai(tokenizer,features,data):
    for layer, samples in tqdm(features.items()):
            explanation=explain_single_feature(layer, samples, data, tokenizer)
            print(f"Layer: {layer}, Explanation: {explanation}")
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction_file", type=str, default="./load_gemma/direction.pt", required=False, help="gradient directions file")
    parser.add_argument("--explain_dataset", type=str, required=False,default="EleutherAI/rpj-v2-sample", help="Hugging Face dataset path")
    
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="Pretrained model")
    parser.add_argument("--topk", type=int, default=10, help="Top-K activations to extract")
    parser.add_argument("--key", type=str, default="raw_content", help="Key to extract from dataset")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16).cuda()
    explain_data=load_dataset(args.explain_dataset,split="train[:1%]")
    explain_data=explain_data[args.key]
    print(f"Explain data sample: {explain_data[0]}")
    
    direction=torch.load(args.direction_file).half().to(device)
    # features=get_top_activation_samples_along_direction(explain_data,direction,tokenizer,model,topk=10,batch_size=32, )
    
    # with open("features.json","w") as f:
    #     json.dump(features,f)
    
    with open("features.json","r") as f:
        features=json.load(f)
    explain_features_with_openai(tokenizer,features, explain_data)

if __name__ == "__main__":
    main()                