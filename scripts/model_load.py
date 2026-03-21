from __future__ import annotations

from huggingface_hub import get_token
import torch


def maybe_hf_token_kwargs() -> dict[str, object]:
    token = get_token()
    if token:
        return {"token": token}
    return {}


def requires_trust_remote_code(model_name: str) -> bool:
    lowered = model_name.lower()
    return lowered.startswith("thudm/chatglm")


def preferred_dtype(model_name: str):
    lowered = model_name.lower()
    if lowered.startswith("google/gemma"):
        return torch.bfloat16
    return torch.float16


def language_model_kwargs(model_name: str) -> dict[str, object]:
    kwargs: dict[str, object] = {"device_map": "auto", "torch_dtype": preferred_dtype(model_name)}
    kwargs.update(maybe_hf_token_kwargs())
    if requires_trust_remote_code(model_name):
        kwargs["trust_remote_code"] = True
        kwargs["tokenizer_kwargs"] = {"trust_remote_code": True, **maybe_hf_token_kwargs()}
    elif maybe_hf_token_kwargs():
        kwargs["tokenizer_kwargs"] = maybe_hf_token_kwargs()
    return kwargs


def tokenizer_kwargs(model_name: str) -> dict[str, object]:
    kwargs: dict[str, object] = maybe_hf_token_kwargs()
    if requires_trust_remote_code(model_name):
        kwargs["trust_remote_code"] = True
    return kwargs


def automodel_kwargs(model_name: str) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "torch_dtype": preferred_dtype(model_name),
        "device_map": "auto",
    }
    kwargs.update(maybe_hf_token_kwargs())
    if requires_trust_remote_code(model_name):
        kwargs["trust_remote_code"] = True
    return kwargs
