import os
from datasets import load_dataset
from collections import defaultdict
from langdetect import detect, DetectorFactory
import random
import json

def load_catgorical_harm_ds():
    harmful_dataset = load_dataset("declare-lab/CategoricalHarmfulQA",split = 'en').to_list()
    category_ds = defaultdict(list)
    for d in harmful_dataset:
        category_ds[d['Subcategory']].append(d['Question'])
    return category_ds

def is_english(text: str) -> bool:
    try:
        return detect(text) == 'en'
    except Exception:
        return False
    
def sort_len(prompts,tokenizer): # sort to minimize padding
    prompt_lens = [len(tokenizer.encode(x)) for x in prompts]
    sorted_prompts = [x for _,x in sorted(zip(prompt_lens,prompts),key = lambda x:x[0])]
    return sorted_prompts

def load_wjb_ds(tokenizer,size=100):
    wjb_ds = load_dataset("allenai/wildjailbreak",'eval')['train']
    DetectorFactory.seed = 0 
    wjb_harmful = [d['adversarial'] for d in wjb_ds if d['label'] == 1 and is_english(d['adversarial'])] # suppose to jailbreak and english
    wjb_harmless = [d['adversarial'] for d in wjb_ds if d['label'] == 0 and is_english(d['adversarial'])][:size] # suppose to not jailbreak?
    wjb_harmful_eval = sort_len(wjb_harmful,tokenizer)[:size]
    return wjb_harmful_eval,wjb_harmless


dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

SPLIT_DATASET_FILENAME = os.path.join('dataset', 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca"]


def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_refusal_datasets(train_size=128,val_size=32):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=True), train_size)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=True), train_size)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='val', instructions_only=True), val_size)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='val', instructions_only=True), val_size)
    return harmful_train, harmless_train, harmful_val, harmless_val

def load_all_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join('dataset', 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset

