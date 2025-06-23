import os

import transformers
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else None
    if device:
        print(f"Using {torch.cuda.get_device_name(0)}")
    bitsquant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    sharded_model = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        sharded_model,
        trust_remote_code=True,
        quantization_config=bitsquant_config,
        device_map="auto",
        token=True,
    )

    text_gen_pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.85,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=256,
        device_map="auto"
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    quick_test = "Who is MistralAI? Please write an essay"
    ans = mistral_llm.invoke(quick_test)
    print(ans)

if __name__ == "__main__":
    main()