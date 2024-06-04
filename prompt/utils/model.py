import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import PartialState


# MODELPATH = "meta-llama/Llama-2-7b-chat-hf"
MODELPATH = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_model():
    print("Loading llama ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODELPATH,    
        device_map={"": PartialState().process_index},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,      
    )
    return model


def load_tokenizer():
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODELPATH,
        trust_remote_code=True,
        padding_side="left",
    )      
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def is_valid_length(tokenizer, prompt):
    prompt_tokenized = tokenizer(prompt, return_tensors="pt")
    return len(prompt_tokenized["input_ids"][0]) < 2048


def forward_decode(model, tokenizer, prompt):
    prompt_tokenized = tokenizer(prompt, return_tensors="pt")
    inputs = prompt_tokenized.to("cuda")
    outputs = model.generate(
        **inputs, 
        max_length=4096, 
        pad_token_id=tokenizer.pad_token_id
    )
    text = tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    return text