from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelLoadConfig:
    model_id: str
    load_in_4bit: bool = True
    torch_dtype: str = 'auto'
    device_map: str = 'auto'
    trust_remote_code: bool = False


def load_model_and_tokenizer(config: ModelLoadConfig) -> Tuple[object, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError('transformers is required to load models') from exc

    # Try AutoProcessor first (for multimodal models like Gemma 4), fall back to AutoTokenizer
    tokenizer = None
    try:
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
    except Exception:
        pass
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
    if getattr(tokenizer, 'pad_token', None) is None and getattr(tokenizer, 'eos_token', None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        'trust_remote_code': config.trust_remote_code,
        'device_map': config.device_map,
    }
    resolved_dtype = resolve_torch_dtype(config.torch_dtype)
    if resolved_dtype is not None:
        kwargs['torch_dtype'] = resolved_dtype
    if config.load_in_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    model.eval()
    return model, tokenizer


def resolve_torch_dtype(torch_dtype: str) -> Optional[object]:
    if torch_dtype == 'auto':
        return None

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required to resolve torch_dtype') from exc

    if not hasattr(torch, torch_dtype):
        raise ValueError(f'Unsupported torch dtype: {torch_dtype}')
    return getattr(torch, torch_dtype)


def resolve_model_device(model: object) -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required to resolve the model device') from exc

    if hasattr(model, 'device'):
        return model.device

    for parameter in model.parameters():
        if parameter.device.type != 'meta':
            return parameter.device

    return torch.device('cpu')
