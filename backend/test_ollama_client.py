import json
import unittest
from unittest.mock import patch

import httpx

from backend.ollama_client import OllamaClient


class OllamaTransportTests(unittest.TestCase):
    def test_native_request_has_schema_context_and_deadline(self):
        seen = []

        def handle(request):
            seen.append(request)
            return httpx.Response(200, json={'message': {'content': '{"ok":true}'},
                                             'done_reason': 'stop', 'prompt_eval_count': 300})

        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            client = OllamaClient(context_length=16384, http_client=http)
            response = client.chat.completions.create(model='gemma3:4b', temperature=.1,
                timeout=12, max_tokens=2200, messages=[{'role': 'user', 'content': '相談'}],
                response_format={'json_schema': {'schema': {'type': 'object'}}})
        self.assertEqual(str(seen[0].url), 'http://localhost:11434/api/chat')
        payload = json.loads(seen[0].content)
        self.assertEqual(payload['options']['num_ctx'], 16384)
        self.assertEqual(payload['format'], {'type': 'object'})
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['keep_alive'], '15m')
        self.assertEqual(response.choices[0].message.content, '{"ok":true}')
        self.assertEqual(seen[0].extensions['timeout']['read'], 12)

    def test_truncated_output_or_context_cannot_be_accepted(self):
        for payload in ({'done_reason': 'length'}, {'prompt_eval_count': 15000}):
            with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as http:
                client = OllamaClient(context_length=16384, http_client=http)
                with self.assertRaises(ValueError):
                    client.chat.completions.create(model='model', temperature=.1,
                        timeout=12, max_tokens=2200, messages=[],
                        response_format={'json_schema': {'schema': {'type': 'object'}}})

    def test_cpu_override_is_sent_only_when_configured(self):
        bodies = []

        def handle(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={'message': {'content': '{}'}})

        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            client = OllamaClient(http_client=http)
            with patch.dict('os.environ', {'OLLAMA_NUM_GPU': '0', 'OLLAMA_THINK': 'false'}):
                client.chat.completions.create(model='model', temperature=.1,
                    timeout=12, max_tokens=2200, messages=[],
                    response_format={'json_schema': {'schema': {'type': 'object'}}})
        self.assertEqual(bodies[0]['options']['num_gpu'], 0)
        self.assertFalse(bodies[0]['think'])

    def test_qwen_reasoning_default_and_explicit_override(self):
        bodies = []

        def handle(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={'message': {'content': '{}'}})

        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            client = OllamaClient(http_client=http)
            with patch.dict('os.environ', {}, clear=True):
                for model in ('qwen3:8b', 'gemma3:4b', 'qwen3:14b'):
                    client.create(model=model, temperature=.7, timeout=12, max_tokens=2200,
                        messages=[], response_format={'json_schema': {'schema': {'type': 'object'}}})
            with patch.dict('os.environ', {'OLLAMA_THINK': 'true'}):
                client.create(model='qwen3:8b', temperature=.7, timeout=12, max_tokens=2200,
                    messages=[], response_format={'json_schema': {'schema': {'type': 'object'}}})
        self.assertFalse(bodies[0]['think'])
        self.assertEqual(bodies[0]['options']['temperature'], .7)
        self.assertEqual(bodies[0]['options']['top_p'], .8)
        self.assertNotIn('think', bodies[1])
        self.assertFalse(bodies[2]['think'])
        self.assertEqual(bodies[2]['options']['top_p'], .8)
        self.assertTrue(bodies[3]['think'])


if __name__ == '__main__':
    unittest.main()
