from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass(frozen=True)
class GeminiClient:
    api_key: str
    model_name: str = 'gemini-2.5-flash'
    timeout_seconds: int = 60

    def generate_json(self, prompt: str, temperature: float = 0.7) -> Any:
        endpoint = (
            'https://generativelanguage.googleapis.com/v1beta/models/'
            f'{self.model_name}:generateContent?key={parse.quote(self.api_key)}'
        )
        payload = {
            'contents': [
                {
                    'parts': [
                        {
                            'text': prompt,
                        }
                    ]
                }
            ],
            'generationConfig': {
                'temperature': temperature,
                'responseMimeType': 'application/json',
            },
        }
        body = json.dumps(payload).encode('utf-8')
        http_request = request.Request(
            endpoint,
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:  # pragma: no cover - network path
            details = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'Gemini API request failed with {exc.code}: {details}') from exc

        text = _extract_text_response(response_payload)
        return json.loads(text)


def _extract_text_response(payload: dict[str, Any]) -> str:
    candidates = payload.get('candidates') or []
    if not candidates:
        raise RuntimeError('Gemini API returned no candidates')
    parts = ((candidates[0].get('content') or {}).get('parts')) or []
    chunks = [part.get('text', '') for part in parts if isinstance(part, dict)]
    text = ''.join(chunks).strip()
    if not text:
        raise RuntimeError('Gemini API returned an empty response')
    return text