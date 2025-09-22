from __future__ import annotations
# Avoid vision deps & tokenizer warnings
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from typing import Any, Dict

def _lazy_import_st():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _lazy_import_transformers():
    import transformers
    return transformers

def _lazy_import_torch():
    import torch
    return torch

def _lazy_import_openai():
    from openai import OpenAI
    return OpenAI

def build_embedder(model_name: str, normalize: bool=True, batch_size: int=64):
    ST = _lazy_import_st()
    model = ST(model_name)
    def embed(texts: list[str]) -> np.ndarray:
        vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize)
        if normalize and vecs.ndim == 2:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs
    return embed

class GenHandle:
    def __init__(self, backend, generate, tokenizer=None, model=None):
        self.backend = backend
        self.generate = generate
        self.tokenizer = tokenizer
        self.model = model

def load_generator(cfg: Dict[str, Any]) -> GenHandle:
    backend = cfg["backend"]
    if backend == "transformers":
        return _load_transformers(cfg)
    elif backend == "transformers-4bit":
        return _load_transformers_4bit(cfg)
    elif backend == "vllm-openai":
        return _load_vllm_openai(cfg)
    else:
        raise ValueError(f"Unknown backend: {backend}")

def _prep_inputs_with_chat_template(tokenizer, prompt: str, device):
    # Returns (inputs_dict, prompt_len_tokens)
    import torch
    messages=[{"role":"user","content":prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = {"input_ids": input_ids.to(device), "attention_mask": torch.ones_like(input_ids).to(device)}
    return inputs, int(input_ids.shape[-1])

def _load_transformers(cfg: Dict[str, Any]) -> GenHandle:
    transformers = _lazy_import_transformers()
    torch = _lazy_import_torch()
    model_id = cfg["model_id"]
    dtype = cfg.get("dtype","auto")
    device_map = cfg.get("device_map","auto")
    tok = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(torch.float16 if dtype in ("float16","half") else torch.bfloat16 if dtype=="bfloat16" else "auto"),
        device_map=device_map,
        trust_remote_code=True
    )
    model.eval()
    def _gen(prompt: str, gen_kwargs: Dict[str,Any]) -> str:
        # Ensure we pass a MAPPING to generate(**inputs)
        used_chat_template = hasattr(tok, "apply_chat_template")
        if used_chat_template:
            inputs, prompt_len = _prep_inputs_with_chat_template(tok, prompt, model.device)
        else:
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k,v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens",512),
                do_sample=gen_kwargs.get("temperature",0)>0,
                temperature=gen_kwargs.get("temperature",0.35),
                top_p=gen_kwargs.get("top_p",0.9),
                top_k=gen_kwargs.get("top_k",40),
                repetition_penalty=gen_kwargs.get("repetition_penalty",1.05),
            )
        # Strip the prompt (prevents echo)
        gen_tokens = out_ids[0, prompt_len:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()
    return GenHandle("transformers", _gen, tok, model)

def _load_transformers_4bit(cfg: Dict[str, Any]) -> GenHandle:
    transformers = _lazy_import_transformers()
    torch = _lazy_import_torch()
    quant_cfg = transformers.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], device_map=cfg.get("device_map","auto"), quantization_config=quant_cfg, trust_remote_code=True
    )
    model.eval()
    def _gen(prompt: str, gen_kwargs: Dict[str,Any]) -> str:
        used_chat_template = hasattr(tok, "apply_chat_template")
        if used_chat_template:
            inputs, prompt_len = _prep_inputs_with_chat_template(tok, prompt, model.device)
        else:
            inputs = tok(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k,v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens",512),
                do_sample=gen_kwargs.get("temperature",0)>0,
                temperature=gen_kwargs.get("temperature",0.35),
                top_p=gen_kwargs.get("top_p",0.9),
                top_k=gen_kwargs.get("top_k",40),
                repetition_penalty=gen_kwargs.get("repetition_penalty",1.05),
            )
        gen_tokens = out_ids[0, prompt_len:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()
    return GenHandle("transformers-4bit", _gen, tok, model)

def _load_vllm_openai(cfg: Dict[str, Any]) -> GenHandle:
    OpenAI = _lazy_import_openai()
    client = OpenAI(base_url=cfg.get("openai_base_url","http://localhost:8000/v1"),
                     api_key=cfg.get("openai_api_key","sk-local"))
    model = cfg.get("openai_model","qwen2.5-7b-instruct")
    def _gen(prompt: str, gen_kwargs: Dict[str,Any]) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=gen_kwargs.get("max_new_tokens",512),
            temperature=gen_kwargs.get("temperature",0.35),
            top_p=gen_kwargs.get("top_p",0.9)
        )
        
        # Check if response has choices
        if not resp.choices:
            raise ValueError("OpenAI API response contains no choices")
        
        # Check if first choice has a message
        if not resp.choices[0].message:
            raise ValueError("OpenAI API response choice has no message")
        
        # Check if message content is not None
        if resp.choices[0].message.content is None:
            raise ValueError("OpenAI API response message content is None")
            
        return resp.choices[0].message.content
    return GenHandle("vllm-openai", _gen)
