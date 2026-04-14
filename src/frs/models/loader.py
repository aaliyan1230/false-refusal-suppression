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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('transformers is required to load models') from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
    kwargs = {
        'trust_remote_code': config.trust_remote_code,
        'device_map': config.device_map,
    }
    if config.torch_dtype != 'auto':
        kwargs['torch_dtype'] = config.torch_dtype
    if config.load_in_4bit:
        kwargs['load_in_4bit'] = True
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    return model, tokenizer
