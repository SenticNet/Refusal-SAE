from utils.eval_refusal import custom_generate
from utils.utils import encode_fn,format_prompt
from tqdm import tqdm
from copy import deepcopy
from functools import partial
import numpy as np
from utils.openai_utils import openai_call,async_process
import re
gsm8k_fewshot = [
    {
        'input': 'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
        'output': 'There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. #### 6'
    },
    {
        'input': 'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
        'output': 'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. #### 5'
    },
    {
        'input': 'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
        'output': 'Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. #### 39'
    },
    {
        'input': 'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
        'output': 'Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. #### 8'
    },
    {
        'input': 'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
        'output': 'Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. #### 9'
    },
    {
        'input': 'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
        'output': 'There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. #### 29'
    },
    {
        'input': 'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
        'output': 'Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. #### 33'
    },
    {
        'input': 'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
        'output': 'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. #### 8'
    }
]

ds_fewshot = {
    'gsm8k': gsm8k_fewshot,
}

def sort_len(examples,tokenizer,key = None): # sort according to length (longest first - to estimate mem size) - samples are close to each other by length to reduce padding
    if key is None:
        example_lens = [len(tokenizer.encode(x)) for x in examples]
    else:
        example_lens = [len(tokenizer.encode(x[key])) for x in examples]
    sorted_examples = [x for _,x in sorted(zip(example_lens,examples),key = lambda x:x[0],reverse = True)]
    return sorted_examples


def format_fewshot(ds_name,qn_formatting_fn=None):
    fewshot = ds_fewshot[ds_name]
    all_fs = []
    for fs in fewshot:
        if qn_formatting_fn is not None:
            all_fs.append({'role':'user','content':qn_formatting_fn(fs['input'])})
        else:
            all_fs.append({'role':'user','content':fs['input']})
        all_fs.append({'role':'assistant','content':fs['output']})
    return all_fs


def eval_gsm8k(ds,model,bz,use_fewshot=False,use_tqdm = False,use_openai = False):

    eval_template = "You are given a question and and the true answer to the question. Please help to judge if the model-generated answer is correct. You must only focus on the numerical values reply with 'yes' if it is an exact match, otherwise reply with 'no'.\n\nQuestion: {question}\nTrue Answer: {answer}\nModel Answer: {pred}\n\nRespond with 'yes' or 'no' only."

    def format_qn(q):
        return f"Question: {q}\nA: Let's think step by step."

    def parse_ans(x):
        return x.split('####')[-1].strip().replace(',','')
    
    def extract_first_number(s):
        match = re.search(r'-?\d+(?:\.\d+)?', s)
        if match:
            return match.group()
        else:
            return None
    
    acc = []
    string_acc = [] # try to see if the string is correct. using re
    ds = sort_len(ds,model.tokenizer,key = 'question')
    if use_fewshot:
        gsm8k_fewshot = format_fewshot('gsm8k',format_qn)
    else:
        gsm8k_fewshot = []
    to_iter = tqdm(range(0,len(ds),bz),total = len(ds)//bz) if use_tqdm else range(0,len(ds),bz)
    for batch_id in to_iter:
        batch = ds[batch_id:batch_id+bz]
        ques = [ex['question'] for ex in batch]
        ans = [parse_ans(ex['answer']) for ex in batch]
        all_formatted = []
        for q in ques:
            temp_fewshot = deepcopy(gsm8k_fewshot)
            temp_fewshot.append({'role':'user','content':format_qn(q)})
            all_formatted.append(temp_fewshot)
        formatted_ques = [model.tokenizer.apply_chat_template(x,tokenize=False,add_generation_prompt=True) for x in all_formatted]
        all_tokenized = encode_fn(formatted_ques,model)
        cot_outputs = custom_generate(model,all_tokenized.input_ids,attention_mask=all_tokenized.attention_mask,max_new_tokens=512)
        cot_gen = model.tokenizer.batch_decode(cot_outputs,skip_special_tokens=True) ## we just treat this as the CoT gen. and prompt it again
        ans_prompt = []
        for inp,cot in zip(formatted_ques,cot_gen): # prompt it again
            ans_prompt.append(inp + cot + '\nSo the final numeric answer is:')
        ans_prompt_tokenized = encode_fn(ans_prompt,model)
        ans_gen = custom_generate(model,ans_prompt_tokenized.input_ids,attention_mask=ans_prompt_tokenized.attention_mask,max_new_tokens=10)
        ans_gen = model.tokenizer.batch_decode(ans_gen,skip_special_tokens=True)
        ans_gen = [x.strip() if x.strip() != '' else 'No Answer' for x in ans_gen ] 
        ## eval with openai
        if use_openai:
            eval_prompts = [eval_template.format(question = q,answer=a,pred=p) for q,a,p in zip(ques,ans,ans_gen)]
            eval_fn = partial(openai_call,max_new_tokens= 5)
            eval_results = async_process(eval_fn,eval_prompts,workers= bz)
            results,cost = zip(*eval_results)
            for p in results:
                if 'yes' in p.lower() and 'no' not in p.lower():
                    acc.append(1)
                elif 'no' in p.lower() and 'yes' not in p.lower():
                    acc.append(0)
                else:
                    continue # ignore
        ## eval string match
        parsed_pred = [extract_first_number(x.replace(',','')) for x in ans_gen]
        for p,a in zip(parsed_pred,ans):
            try:
                p = int(p)
            except:
                try:
                    p = float(p)
                except:
                    p = None
            string_acc.append(1 if str(p) == str(a) else 0)
    if use_openai:
        return sum(acc)/len(acc),sum(string_acc)/len(string_acc)
    else:
        return sum(string_acc)/len(string_acc)


def eval_arc(ds,model,bz,use_tqdm=False):

    def format_qn(q,c):
        choices = '\n'.join([f'({chr(65+i)}) {cc}' for i,cc in enumerate(c)])
        return f'Given the following question and four candidate answers (A, B, C and D), choose the best answer.\nQuestion: {q.strip()}\nChoices:\n{choices}\nYour response should end with "The best answer is ([the_answer_letter])" where the [the_answer_letter] is one of A, B, C or D.'
    
    ans_prefix = 'The best answer is ('

    acc = []
    to_iter = tqdm(range(0,len(ds),bz),total = len(ds)//bz) if use_tqdm else range(0,len(ds),bz)
    for batch_id in to_iter:
        batch = ds[batch_id:batch_id+bz]
        ques = [d['question'] for d in batch]
        choices = [d['choices']['text'] for d in batch]
        ans = [d['answerKey'] for d in batch]
        formatted_ques = [format_qn(q,c) for q,c in zip(ques,choices)]
        
        chat_formatted = [format_prompt(model.tokenizer,x) + ans_prefix for x in formatted_ques]
        encoded_inp = encode_fn(chat_formatted,model)
        pred_output = custom_generate(model,encoded_inp.input_ids,attention_mask=encoded_inp.attention_mask,max_new_tokens=1)
        pred = model.tokenizer.batch_decode(pred_output,skip_special_tokens=True)
        acc.extend([1 if p.lower().strip() == a.lower() else 0 for p,a in zip(pred,ans)])
    return np.mean(acc)




