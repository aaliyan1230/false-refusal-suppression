from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


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
                self.outputs.setdefault(name, []).append(output)

            self._hooks.append(module.register_forward_hook(hook))
        return self

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
