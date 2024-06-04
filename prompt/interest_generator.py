#!/usr/bin/env python
# coding: utf-8

# # Load model and tokenizer

# In[1]:


import sys
sys.path.append("../")

import os
from utils.args import get_args

args = get_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)


# In[3]:


from utils.model import *

model = load_model()
tokenizer = load_tokenizer()


# # Load dataset

# In[5]:


DATASET = "amazon_toy"


# In[6]:


from pathlib import Path

root = Path("./")
dataset_folder = root.joinpath("dataset", DATASET)


# In[10]:


from datasets import load_dataset

print("Loading dataset ...")
# load the dataset created in Part 1
dataset = load_dataset("csv", data_files=dataset_folder.joinpath("history.csv").__str__(), split="train")


# In[11]:


import json

# Drop user that is not in the list
with open(dataset_folder.joinpath("user_id2token_id.json"), "r") as fp:
    user_id2token_id = json.load(fp)
    user_ids = set(user_id2token_id)

dataset = dataset.select(
    (
        i for i in range(len(dataset)) 
        if str(dataset[i]["user_id"]) in user_ids 
    )
)


# In[16]:


import csv

with open(dataset_folder.joinpath("item_id2token_id.json"), "r") as fp:
    item_id2token_id = json.load(fp)
    item_ids = set(item_id2token_id)

id2name, name2id = {}, {}
name_list = []
with open(dataset_folder.joinpath("item_id2name.csv"), "r") as fp:
    reader = csv.reader(fp)
    for row in reader:
        id, name = row
        if id not in item_ids:
            continue

        id2name[id] = name
        name2id[name] = id
        name_list.append(name)


# # Run zero shot testing

# In[19]:


SYSTEM_TEXT = """You are a professional sales staff and are required to find out the user's interest based on products the user bought before. The following is an example of input:
Products which the user bought before:
NYX Cosmetics LONG &amp; FULL LASHES MASCARA SERUM - JET BLACK - AYD04
NEW! Keratin Express Trio! Shampoo 10 oz, Conditioner 10 oz, Daily Keratin 2 oz. Best Value

The example of output:
- Makeup and skincare bundles
- New launches and limited-edition products
- Discounts and deals
- Subscription boxes
- Gift sets and travel-sized products

You should return five sales types the user might be interested in and not return any explanation. The format should be as follows:
- xxx
- yyy
- zzz
- aaa
- bbb

Now the task begins, please think, analyze as concisely as possible.
"""

USER_TEXT = """Products which the user bought before:
{history_items}"""


# In[43]:


def extract_answer(text):
    lines = text.split("\n")
    return [line[2:] for line in lines[:5]]


# In[14]:


from utils.result import *

result_folder = set_up_result_dir(dataset_folder, "interest/llama3_v1")


# In[ ]:


import ast
from tqdm import tqdm


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

for i, row in enumerate(tqdm(dataset)):
    if i % args.n_p != args.pid:
        continue

    user_id = row["user_id"]
    if user_is_predicted(result_folder, user_id):
        continue
    
    history_items = ast.literal_eval(row["history_items"])
    history_items = [item for item in history_items if item in name_list][:10]

    input_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "system", 
                "content": SYSTEM_TEXT
            },
            {
                "role": "user", 
                "content": USER_TEXT.format(history_items="\n".join(history_items))
            },
        ],
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)

    interests = extract_answer(result)

    with open(result_folder.joinpath(get_result_filename(user_id)), "w") as fp:
        fp.write("\n".join(interests))
