from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
from functools import partial
import time

def async_process(fn,inps,workers=10,msg=''):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        if len(msg):
            out = list(tqdm(executor.map(fn,inps),total = len(inps),desc = msg))
        else:
            out = list(executor.map(fn,inps))
    return out

def openai_call(message,model='gpt-4o',max_new_tokens=100,temperature=0.,n=1):
    client = OpenAI()
    if not isinstance(message,list):
        message = [{'role':'user','content':message}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=message,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n,
            )
        cost = cal_cost(model,response.usage.prompt_tokens,response.usage.completion_tokens)
        if n > 1:
            return [r.message.content for r in response.choices],cost
        else:
            return response.choices[0].message.content,cost
    except Exception as e:
        print ('Error:',e)

def cal_cost(model_name,in_tokens,out_tokens):
    if model_name in ['gpt-4','GPT4']:
        cost = in_tokens * (10/1e6) + (out_tokens * (30/1e6))
    elif model_name in ['gpt-4o','GPT4o']:
        cost = in_tokens * (2.5/1e6) + (out_tokens * (10/1e6))
    elif model_name == 'gpt-3.5-turbo-0125':
        cost = in_tokens * (0.5/1e6) + (out_tokens * (1.5/1e6))
    elif model_name == 'gpt-3.5-turbo-instruct':
        cost = in_tokens * (1.5/1e6) + (out_tokens * (2/1e6))
    else:
        raise NotImplementedError
    return cost