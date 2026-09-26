"""Explicit provider settings; keys are read only when a live run starts."""
from types import SimpleNamespace
import os
import time
from .data import ROOT, read_json


class Client:
    def __init__(self, system, family='qa'):
        from openai import OpenAI
        spec = read_json(ROOT / 'configs/systems.json')[system]
        cfg = dict(spec['default'])
        cfg.update(spec.get(family, {}))
        self.settings = cfg
        self.config = SimpleNamespace(model=spec['model'])
        key = os.environ.get(spec['api_key_env'])
        if not key:
            raise ValueError(f"Set {spec['api_key_env']} in the environment before --execute")
        self.client = OpenAI(api_key=key, base_url=spec['base_url'], timeout=cfg.get('timeout', 2400), max_retries=2)

    def request(self, messages):
        cfg = self.settings
        kwargs = {'model': self.config.model, 'messages': messages}
        for key in ('temperature', 'top_p', 'max_tokens', 'seed', 'reasoning_effort'):
            if cfg.get(key) is not None:
                kwargs[key] = cfg[key]
        if cfg.get('extra_body'):
            kwargs['extra_body'] = cfg['extra_body']
        if cfg.get('stream'):
            kwargs.update(stream=True, stream_options={'include_usage': True})
        return kwargs

    def chat(self, messages, transport_retries=None):
        retries = self.settings.get('transport_retries', 4) if transport_retries is None else transport_retries
        started = time.perf_counter()
        for attempt in range(retries + 1):
            try:
                result = self.client.chat.completions.create(**self.request(messages))
                if self.settings.get('stream'):
                    text, usage = [], None
                    for chunk in result:
                        if chunk.choices and chunk.choices[0].delta.content:
                            text.append(chunk.choices[0].delta.content)
                        if chunk.usage is not None:
                            usage = chunk.usage
                    content = ''.join(text)
                else:
                    content = result.choices[0].message.content or ''
                    usage = result.usage
                totals = {out: int(getattr(usage, inp, 0) or 0) for out, inp in
                          [('input_tokens','prompt_tokens'), ('output_tokens','completion_tokens'), ('total_tokens','total_tokens')]}
                return content, totals, attempt, time.perf_counter() - started
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                if attempt >= retries or status in {400, 401, 402, 403, 404}:
                    raise
                time.sleep(min(20, 2 ** (attempt + 1)))


class MockClient:
    """Offline testing only; never represents an experimental model."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.config = SimpleNamespace(model='offline-test')
        self.calls = []

    def chat(self, messages, transport_retries=None):
        self.calls.append([dict(m) for m in messages])
        return next(self.responses), {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}, 0, 0.0
