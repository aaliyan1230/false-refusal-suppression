from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from frs.models.loader import resolve_model_device


@dataclass(frozen=True)
class TextGenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False
    max_input_length: Optional[int] = None


def generate_text(model: object, tokenizer: object, prompt: str, config: Optional[TextGenerationConfig] = None) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required for generation') from exc

    generation_config = config or TextGenerationConfig()
    encoded = tokenize_prompt(model, tokenizer, prompt, max_input_length=generation_config.max_input_length)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            do_sample=generation_config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_texts(
    model: object,
    tokenizer: object,
    prompts: Sequence[str],
    config: Optional[TextGenerationConfig] = None,
) -> List[str]:
    generation_config = config or TextGenerationConfig()
    return [generate_text(model, tokenizer, prompt, generation_config) for prompt in prompts]


def next_token_distribution(
    model: object,
    tokenizer: object,
    prompt: str,
    max_input_length: Optional[int] = None,
) -> List[float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required to compute next-token distributions') from exc

    encoded = tokenize_prompt(model, tokenizer, prompt, max_input_length=max_input_length)
    with torch.no_grad():
        logits = model(**encoded, use_cache=False).logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)[0]
    return probabilities.detach().float().cpu().tolist()


def tokenize_prompt(
    model: object,
    tokenizer: object,
    prompt: str,
    max_input_length: Optional[int] = None,
) -> dict:
    encoded = tokenizer(
        text=prompt,
        return_tensors='pt',
        truncation=max_input_length is not None,
        max_length=max_input_length,
    )
    device = resolve_model_device(model)
    return {key: value.to(device) for key, value in encoded.items()}
