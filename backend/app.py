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


def is_vague_recommendation(recommendation):
    text = re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', recommendation)).casefold()
    return text in ('おすすめ', 'おすすめです', 'おすすめします', 'おすすめだ') or len(text) <= 1


def is_conditional_recommendation(recommendation):
    text = unicodedata.normalize('NFKC', recommendation).casefold()
    return any(marker in text for marker in ('の場合', '場合、', '場合に', 'なら', 'ならば', 'でいうと', 'に限れば'))


def normalize_text(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text)).casefold()


FILLER_QUESTION_TEXT = {
    normalize_text(text) for text in
    ('わからない', 'わからないです', 'わからない・決められない', '分からない', '決められない',
     'まだわからない', 'どちらともいえない', '未定', 'その他', 'はい', 'いいえ', 'どちらでも')
}


FALLBACK_QUESTIONS = [
    {
        'status': 'question',
        'question': '予算はいくらまで使えますか？',
        'answer_type': 'number',
        'unit': '円',
        'confidence': 30,
    },
    {
        'status': 'question',
        'question': '誰と一緒に行きたいですか？',
        'answer_type': 'choice',
        'options': ['一人で', '友人や家族と', 'わからない・決められない'],
        'confidence': 30,
    },
    {
        'status': 'question',
        'question': '今の疲れ具合はいかがですか？',
        'answer_type': 'choice',
        'options': ['とても疲れている', '少し疲れている', '疲れていない', 'わからない・決められない'],
        'confidence': 30,
    },
]


def is_filler_question(question):
    normalized = normalize_text(question)
    return len(normalized) <= 3 or normalized in FILLER_QUESTION_TEXT


def is_malformed_question(question):
    if '\n' in question or '選択肢' in question or '選択・' in question:
        return True
    if len(normalize_text(question)) > 60:
        return True
    return bool(re.search(r'\d{1,2}[\.、．)）]', question))


def asks_direct_choice(question, consultation):
    raw = unicodedata.normalize('NFKC', question).casefold()
    if not re.search(r'どちら|どっち', raw):
        return False
    needle = normalize_text(consultation)
    parts = re.split(r'[と・、,。か\s]', raw)
    return sum(1 for part in parts if len(normalize_text(part)) > 1 and normalize_text(part) in needle) >= 2


def reuses_last_answer(question, history):
    if not history:
        return False
    candidate = normalize_text(question)
    last_answer = normalize_text(history[-1]['answer'])
    return last_answer and (candidate == last_answer or candidate in last_answer or last_answer in candidate)


def is_repeated_question(question, history):
    candidate = normalize_text(question)
    return any(
        candidate == normalize_text(item['question'])
        or SequenceMatcher(None, candidate, normalize_text(item['question'])).ratio() >= 0.8
        for item in history
    )


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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": consultation.strip()},
    ]
    for item in history:
        messages.append({"role": "assistant", "content": item['question']})
        messages.append({"role": "user", "content": item['answer']})
    messages[0]['content'] += '\n回答済み質問数: ' + str(len(history))
    if history:
        messages[0]['content'] += '\n以下は回答済みの履歴です。「わからない」も有効な回答です。同じ判断条件を聞き直さず、予算・時間・場所・利用頻度など未確認の別の条件へ進んでください。\n' + json.dumps(history, ensure_ascii=False)

    try:
        deadline = monotonic() + 100
        confidence_reached = False
        for attempt in range(3):
            remaining = deadline - monotonic()
            if remaining <= 0:
                return jsonify(error='AIの応答がタイムアウトしました。再試行してください。'), 504
            # Force a final result on user request, when confidence hit 100,
            # when the question cap is reached, or after rejected duplicates.
            require_result = force_result or confidence_reached or len(history) >= MAX_QUESTIONS or attempt == 2
            if require_result:
                messages[0]['content'] += (
                    '\n質問は終了です。不明な条件は不明と明記し、既知の条件で最終結果を生成してください。'
                    '\nrecommendationには比較中の候補名（例: 温泉旅行）を必ず含め、「おすすめ」のような一般名詞だけで返さないでください。'
                )
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=OLLAMA_MODEL,
                temperature=0.2,
                timeout=remaining,
                response_format={
                    'type': 'json_schema',
                    'json_schema': {
                        'name': 'decision_response',
                        'schema': DecisionResult.model_json_schema() if require_result else RESPONSE_ADAPTER.json_schema(),
                    },
                },
            )
            if not chat_completion.choices:
                raise ValueError('Empty model response')
            result = RESPONSE_ADAPTER.validate_json(chat_completion.choices[0].message.content or '').model_dump()
            if require_result and result['status'] != 'result':
                raise ValueError('Expected final result')
            if result['status'] == 'question' and is_repeated_question(result['question'], history):
                messages[0]['content'] += '\n直前の生成候補「' + result['question'] + '」は回答済み質問の繰り返しなので却下しました。この質問の言い換えも禁止です。別の判断条件を質問してください。'
                continue
            if result['status'] == 'question' and (is_filler_question(result['question']) or reuses_last_answer(result['question'], history) or is_malformed_question(result['question']) or asks_direct_choice(result['question'], consultation)):
                messages[0]['content'] += '\n直前の生成候補の質問文「' + result['question'] + '」は定型句、直前の回答の使い回し、選択肢の埋め込み、または相談の結論をそのまま聞き返す直接比較なので却下しました。判断材料となる具体的な条件（時間・予算・疲れ具合・同行者など）を1つだけ質問してください。'
                continue
            if result['status'] == 'result' and not require_result and len(history) < MIN_QUESTIONS_FOR_RESULT:
                messages[0]['content'] += '\n回答数が' + str(len(history)) + 'と少なく情報が不足しています。最終結果は出さず、不足している判断条件を質問してください。最低' + str(MIN_QUESTIONS_FOR_RESULT) + '問の回答が揃うまで質問を続けてください。'
                continue
            if result['status'] == 'question' and result['confidence'] >= 100 and len(history) < MIN_QUESTIONS_FOR_RESULT:
                messages[0]['content'] += '\nまだ回答が少なく情報が不足しています。confidenceを100にせず、質問を続けてください。'
                result['confidence'] = 50
            elif result['status'] == 'question' and result['confidence'] >= 100:
                messages[0]['content'] += '\nconfidenceが100になりました。追加の質問は不要です。既知の条件で最終結果を生成してください。'
                confidence_reached = True
                continue
            if result['status'] == 'result' and require_result and attempt < 2 and (is_vague_recommendation(result['recommendation']) or is_conditional_recommendation(result['recommendation'])):
                messages[0]['content'] += '\n「' + result['recommendation'] + '」は曖昧または条件文（「の場合」「なら」など）なので却下しました。比較中の候補名を1つだけ明記して最終結果を再生成してください。'
                continue
            if result['status'] == 'result' and len(history) < MIN_QUESTIONS_FOR_RESULT and not force_result:
                asked = {normalize_text(item['question']) for item in history}
                fallback = next(
                    (q for q in FALLBACK_QUESTIONS if normalize_text(q['question']) not in asked),
                    FALLBACK_QUESTIONS[0],
                )
                return jsonify(fallback)
            return jsonify(result)
    except APITimeoutError:
        return jsonify(error='AIの応答がタイムアウトしました。少し待って再試行してください。'), 504
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
