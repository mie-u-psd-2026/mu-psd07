import json
import os
from pathlib import Path
from time import monotonic

from flask import Flask, request, jsonify, send_from_directory, Response
from openai import APIError, APITimeoutError

if __package__:
    from .decision import decide, DecisionQualityError, DEFAULT_TIMEOUT
    from .ollama_client import OllamaClient, DEFAULT_MODEL
else:
    from decision import decide, DecisionQualityError, DEFAULT_TIMEOUT
    from ollama_client import OllamaClient, DEFAULT_MODEL

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / 'frontend'
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='/static')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', DEFAULT_MODEL)
STRICT_REVIEW = os.environ.get('DECISION_STRICT_REVIEW', 'false').lower() == 'true'
MAX_QUESTIONS = 15
DECISION_TIMEOUT = int(os.environ.get('DECISION_TIMEOUT_SECONDS', str(DEFAULT_TIMEOUT)))
if not 30 <= DECISION_TIMEOUT <= 600:
    raise ValueError('DECISION_TIMEOUT_SECONDSは30〜600秒で指定してください')
SKIP_LOG_PATH = BACKEND_DIR / 'skip_log.json'


def load_skipped_questions():
    try:
        data = json.loads(SKIP_LOG_PATH.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except (OSError, ValueError):
        pass
    return []


def save_skipped_questions(items):
    try:
        SKIP_LOG_PATH.write_text(json.dumps(items, ensure_ascii=False), encoding='utf-8')
    except OSError:
        pass


def learn_skipped_question(question):
    items = load_skipped_questions()
    if question not in items:
        items.append(question)
        save_skipped_questions(items)


client = OllamaClient()


@app.after_request
def add_header(response):
    if request.endpoint in ('index', 'static', 'runtime_config'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/runtime-config.js')
def runtime_config():
    # 画面がサーバーより先に中断しないよう、公開してよい期限だけを配信する。
    return Response(f'globalThis.APP_REQUEST_TIMEOUT_MS = {(DECISION_TIMEOUT + 10) * 1000};',
                    mimetype='application/javascript')


@app.route('/send_api', methods=['POST'])
def send_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='相談内容をJSON形式で送信してください。'), 400
    consultation = data.get('consultation')
    history = data.get('history', [])
    force_result = data.get('force_result', False)
    skipped_question = data.get('skip_question')
    skipped = data.get('skipped_questions', [])
    if not isinstance(consultation, str) or not consultation.strip():
        return jsonify(error='相談内容を入力してください。'), 400
    if not isinstance(force_result, bool):
        return jsonify(error='途中終了の指定が不正です。'), 400
    if (not isinstance(history, list) or len(history) > MAX_QUESTIONS or any(
        not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ('question', 'answer')
        ) or ('condition' in item and not isinstance(item['condition'], str))
        for item in history
    )):
        return jsonify(error='質問と回答の履歴が不正です。'), 400
    if (not isinstance(skipped, list) or len(skipped) > 50
            or any(not isinstance(q, str) or not q.strip() for q in skipped)
            or (skipped_question is not None and
                (not isinstance(skipped_question, str) or not skipped_question.strip()))):
        return jsonify(error='スキップ対象の質問が不正です。'), 400
    skipped = list(dict.fromkeys(skipped))
    if skipped_question:
        learn_skipped_question(skipped_question.strip())
        if skipped_question.strip() not in skipped:
            skipped.append(skipped_question.strip())
    # 計画は履歴から再構築し、クライアントから受け取らない。
    history = [{k: item[k] for k in ('question', 'answer', 'condition') if k in item}
               for item in history]
    try:
        started = monotonic()
        result = decide(client, OLLAMA_MODEL, consultation.strip(), history,
                        force_result=force_result or len(history) >= MAX_QUESTIONS,
                        skipped=skipped, timeout=DECISION_TIMEOUT, strict_review=STRICT_REVIEW)
        response = jsonify(result)
        response.headers['Server-Timing'] = f'decision;dur={(monotonic() - started) * 1000:.1f}'
        return response
    except (APITimeoutError, TimeoutError):
        return jsonify(error='AIの応答がタイムアウトしました。入力は保持されています。再試行してください。'), 504
    except DecisionQualityError as error:
        return jsonify(error=str(error)), 502
    except (ValueError, TypeError):
        return jsonify(error='AIの応答形式が不正でした。再試行してください。'), 502
    except APIError:
        app.logger.warning('Ollama API call failed')
        return jsonify(error='AIとの通信に失敗しました。Ollamaの起動とモデルの取得を確認してください。'), 502


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, use_reloader=True,
            extra_files=[str(BACKEND_DIR / name) for name in
                         ('prompt.txt', 'result_prompt.txt', 'planning_prompt.txt', 'review_prompt.txt',
                          'question_review_prompt.txt', 'result_review_prompt.txt',
                          'practical_prompt.txt', 'question_examples.json')])
