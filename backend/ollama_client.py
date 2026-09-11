"""既存SDKの通信・例外処理を使い、Ollama固有の文脈長を指定する。"""
import os
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = 'qwen3:4b-instruct-2507-q4_K_M'
DEFAULT_CONTEXT_LENGTH = 8192

class OllamaClient:
    def __init__(self, context_length=None, http_client=None):
        self.context_length = context_length or int(os.environ.get('OLLAMA_NUM_CTX', str(DEFAULT_CONTEXT_LENGTH)))
        if self.context_length < 4096:
            raise ValueError('OLLAMA_NUM_CTXは4096以上にしてください')
        self._sdk = OpenAI(base_url='http://localhost:11434/api', api_key='ollama',
                           timeout=110, max_retries=0, http_client=http_client)
        # 既存コードと同じ呼び出し口。通信のみnative APIへ変換する。
        self.chat = SimpleNamespace(completions=self)

    def create(self, *, model, temperature, timeout, max_tokens, messages, response_format):
        options = {'temperature': temperature, 'num_predict': max_tokens,
                   'num_ctx': self.context_length}
        if 'OLLAMA_NUM_GPU' in os.environ:
            options['num_gpu'] = int(os.environ['OLLAMA_NUM_GPU'])
        request_options = {}
        if model.startswith('qwen3:'):
            # 通常の会話では速度を優先。厳格な審査実験では環境変数で変更できる。
            request_options['think'] = False
        if 'OLLAMA_THINK' in os.environ:
            # thinking対応モデルでは、構造化出力前の長い推論を環境設定で制御できる。
            request_options['think'] = os.environ['OLLAMA_THINK'].lower() == 'true'
        if model.startswith('qwen3:'):
            options.update(temperature=0.6 if request_options['think'] else temperature,
                           top_p=0.95 if request_options['think'] else 0.8, top_k=20, min_p=0)
        payload = self._sdk.post('/chat', cast_to=dict[str, Any], options={'timeout': timeout}, body={
            'model': model, 'messages': messages, 'stream': False,
            'keep_alive': '15m',
            'format': response_format['json_schema']['schema'],
            'options': options,
            **request_options,
        })
        if payload.get('done_reason') == 'length':
            raise ValueError('モデルの出力が長すぎます。各項目を簡潔にしてください')
        if payload.get('prompt_eval_count', 0) >= self.context_length - max_tokens:
            raise ValueError('相談と履歴が長いため文脈の余裕が不足しています。新しい相談で内容を短くしてください')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=payload.get('message', {}).get('content', '')))])
