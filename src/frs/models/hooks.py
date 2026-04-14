from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ActivationRecorder:
    module_names: List[str]
    outputs: Dict[str, list] = field(default_factory=dict)
    _hooks: List[object] = field(default_factory=list)

    def attach(self, model: object) -> "ActivationRecorder":
        modules = dict(model.named_modules())
        for module_name in self.module_names:
            if module_name not in modules:
                raise KeyError(f'Module not found: {module_name}')
            module = modules[module_name]

            def hook(_module, _inputs, output, name=module_name):
                self.outputs.setdefault(name, []).append(_detach_output(output))

            self._hooks.append(module.register_forward_hook(hook))
        return self

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def extract_last_token_vector(output: Any) -> List[float]:
    tensor = _unwrap_output(output)
    if not hasattr(tensor, 'ndim'):
        raise TypeError('Expected a tensor-like output from the hooked module')
    if tensor.ndim < 2:
        raise ValueError('Expected a batched sequence activation tensor')
    return tensor[0, -1, :].detach().float().cpu().tolist()


def _detach_output(output: Any) -> Any:
    tensor = _unwrap_output(output)
    if hasattr(tensor, 'detach'):
        return tensor.detach()
    return tensor


def _unwrap_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output
