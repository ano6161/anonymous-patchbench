import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
from tqdm import tqdm
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage, AssistantMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import os
import glob

def get_tokenizer(model_name, snapshot_path):
    return AutoTokenizer.from_pretrained(snapshot_path, use_fast=True, local_files_only=True,)

def get_model(model_name, snapshot_path, dtype="auto", device_map="auto"):
    if model_name == "mistralai/Ministral-3-14B-Instruct-2512":
        return AutoModelForImageTextToText.from_pretrained(
            snapshot_path,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            snapshot_path,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        )

def repo_to_latest_snapshot_dir(repo_id: str) -> str:
    repo_dir = os.path.join(
        os.environ["HF_HOME"],
        "hub",
        "models--" + repo_id.replace("/", "--"),
        "snapshots",
    )
    snaps = sorted(glob.glob(os.path.join(repo_dir, "*")))
    if not snaps:
        raise FileNotFoundError(f"No local snapshot found for {repo_id} in {repo_dir}")
    return snaps[-1]

MODEL_FILENAME_MAP = {
    "Qwen/Qwen2.5-3B-Instruct":                    "Qwen__Qwen2.5-3B-it",
    "Qwen/Qwen2.5-14B-Instruct":                   "Qwen__Qwen2.5-14B-it",
    "mistralai/Mistral-7B-Instruct-v0.3":           "mistralai__Mistral-7B-it",
    "mistralai/Ministral-3-14B-Instruct-2512":      "mistralai__Ministral-3-14B-2512-it",
    "google/gemma-3-4b-it":                         "google__gemma-3-4b-it",
    "google/gemma-3-12b-it":                        "google__gemma-3-12b-it",
    "meta-llama/Llama-3.1-8B-Instruct":             "meta-llama__Llama-3.1-8B-it",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct":    "llama4_scout-it",
}


def model_filename(model_id: str) -> str:
    """Return the canonical filename stem for a given HuggingFace model ID."""
    if model_id not in MODEL_FILENAME_MAP:
        raise ValueError(f"No filename mapping for model '{model_id}'. Add it to MODEL_FILENAME_MAP.")
    return MODEL_FILENAME_MAP[model_id]


def safe_name(model_id: str, dir: bool = True, remove_first_caps: bool = False) -> str:
    name = model_id.replace(" ", "_")
    
    if not dir:
        name = name.split("/", 1)[-1]

    name = name.replace("/", "__")
    
    if remove_first_caps:
        name = name[0].lower() + name[1:]
    return name

def add_padding_token(tokenizer, repo_id: str):
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer