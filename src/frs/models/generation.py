from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TextGenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False


def generate_text(model: object, tokenizer: object, prompt: str, config: Optional[TextGenerationConfig] = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required for generation') from exc

    generation_config = config or TextGenerationConfig()
    encoded = tokenizer(prompt, return_tensors='pt')
    if hasattr(model, 'device'):
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            do_sample=generation_config.do_sample,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
