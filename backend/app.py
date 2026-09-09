import json
import os
import re
import unicodedata
from time import monotonic
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Literal

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI, APIError, APITimeoutError
from pydantic import BaseModel, Field, TypeAdapter


NonemptyText = Annotated[str, Field(min_length=1)]
TextList = Annotated[list[NonemptyText], Field(min_length=1)]


class ChoiceQuestion(BaseModel):
    status: Literal['question']
    question: NonemptyText
    answer_type: Literal['choice']
    options: list[NonemptyText] = Field(min_length=2, max_length=5)
    condition: str = ''
    confidence: int = Field(ge=0, le=100)


class NumberQuestion(BaseModel):
    status: Literal['question']
    question: NonemptyText
    answer_type: Literal['number']
    unit: str
    condition: str = ''
    confidence: int = Field(ge=0, le=100)


class TextQuestion(BaseModel):
    status: Literal['question']
    question: NonemptyText
    answer_type: Literal['text']
    condition: str = ''
    confidence: int = Field(ge=0, le=100)


class DecisionResult(BaseModel):
    status: Literal['result']
    recommendation: NonemptyText
    conditions: TextList
    reason: NonemptyText
    pros: TextList
    cons: TextList
    alternative: NonemptyText
    confidence: int = Field(ge=0, le=100)


RESPONSE_ADAPTER = TypeAdapter(ChoiceQuestion | NumberQuestion | TextQuestion | DecisionResult)

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / 'frontend'
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='/static')
SYSTEM_PROMPT = (BACKEND_DIR / 'prompt.txt').read_text(encoding='utf-8')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'gemma3:4b')
MAX_QUESTIONS = 15
MIN_QUESTIONS_FOR_RESULT = 3


client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=100.0,
    max_retries=0,
)


@app.after_request
def add_header(response):
    if request.endpoint in ('index', 'static'):
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


NUMBER_QUESTION_MARKERS = (
    'どのくらい', 'どれくらい', 'いくら', 'いくつ', '何円', '何分', '何時間', '何日',
    '何歳', '何人', '何回', '金額', '予算', '距離', 'km', 'キロ',
)


def numeric_unit_for(question):
    text = unicodedata.normalize('NFKC', question).casefold()
    if '頻度' in text or '回数' in text or '程度' in text:
        return None
    if not any(marker in text for marker in NUMBER_QUESTION_MARKERS):
        return None
    if any(marker in text for marker in ('いくら', '円', '金額', '予算', '金まで')):
        return '円'
    if any(marker in text for marker in ('km', '距離', 'キロ')):
        return 'km'
    if '時間' in text or '分' in text:
        return '時間'
    if '日' in text or '泊' in text:
        return '日'
    if '歳' in text:
        return '歳'
    if '人' in text:
        return '人'
    return '回'


def is_malformed_question(question):
    if '\n' in question or '選択肢' in question or '選択・' in question:
        return True
    if len(normalize_text(question)) > 60:
        return True
    return bool(re.search(r'\d{1,2}[\.、．)）]', question))


UNRELATED_CHOICE_MARKERS = ('搭載', 'スペック', 'メモリ', 'cpu', 'gpu', '処理速度', '容量', '筐体', '発熱', 'os', 'モジュール', 'インターフェース')


def has_unrelated_choice(question, options):
    text = unicodedata.normalize('NFKC', question).casefold()
    if not any(marker in text for marker in ('場所', 'どこで', 'どこに', 'どこか', 'どこでも', 'どこででも')):
        return False
    return any(
        any(marker in unicodedata.normalize('NFKC', option).casefold() for marker in UNRELATED_CHOICE_MARKERS)
        for option in options
    )


def asks_direct_choice(question, consultation):
    raw = unicodedata.normalize('NFKC', question).casefold()
    if not re.search(r'どちら|どっち', raw):
        return False
    needle = normalize_text(consultation)
    parts = re.split(r'[と・、,。か\s]', raw)
    matches = 0
    for part in parts:
        norm = normalize_text(part)
        if norm and norm in needle:
            matches += 1
    return matches >= 2


def presumes_single_candidate(question, consultation):
    raw = unicodedata.normalize('NFKC', question).casefold()
    m = re.search(r'([^、,，。?？\s]{2,15})(場合|なら|のとき|だったら|ならば|でしたら)', raw)
    if not m:
        return False
    premise = normalize_text(m.group(1))
    if len(premise) < 2:
        return False
    candidates = extract_candidates(consultation)
    if len(candidates) < 2:
        return False
    for candidate in candidates:
        cn = normalize_text(candidate)
        if premise in cn or cn in premise:
            return True
        for start in range(len(cn) - 1):
            if cn[start:start + 2] in premise:
                return True
    return False


def clean_conditional_recommendation(recommendation):
    markers = ('の場合', 'ならば', 'なら', 'だったら', 'のとき', 'でしたら', '場合、', '場合に', '場合でいうと', 'でいうと', 'に限れば')
    head = recommendation
    for marker in markers:
        idx = recommendation.find(marker)
        if idx != -1:
            head = recommendation[:idx].strip('、,，。 「」')
            break
    if not head:
        return recommendation
    return head


def asks_comparison_result(question, consultation):
    raw = unicodedata.normalize('NFKC', question).casefold()
    if not re.search(r'どちらが|どっちが|どれが|どちらの方が|どっちの方が', raw):
        return False
    candidates = extract_candidates(consultation)
    return len(candidates) >= 2


def reuses_last_answer(question, history):
    if not history:
        return False
    candidate = normalize_text(question)
    last_answer = normalize_text(history[-1]['answer'])
    return last_answer and (candidate == last_answer or candidate in last_answer or last_answer in candidate)


RESTATEMENT_MARKERS = ('とのことですが', 'とのことでしたが', 'と伺いました', 'ということですね',
                      '迷っていますが', '迷っているのですが', 'ようですね', 'なんですね', 'ですが')


def question_restates_consultation(question, consultation):
    raw = unicodedata.normalize('NFKC', question).casefold()
    if not any(marker in raw for marker in RESTATEMENT_MARKERS):
        return False
    needle = normalize_text(consultation)
    head = normalize_text(re.split(r'[、,，。\n]', question)[0])
    if len(needle) < 6 or len(head) < 4:
        return False
    common = 0
    for a, b in zip(head, needle):
        if a != b:
            break
        common += 1
    return common >= 6


TIME_TOPIC_MARKERS = ('時間', '所要', '通勤', '移動', '確保', '合計', '平均', '毎日')
MONEY_MARKERS = ('費', '予算', 'お金', '円')


def has_time_topic(question):
    return any(marker in unicodedata.normalize('NFKC', question).casefold() for marker in TIME_TOPIC_MARKERS)


def has_money_marker(question):
    return any(marker in question for marker in MONEY_MARKERS)


def is_repeated_question(question, history):
    candidate = normalize_text(question)
    for item in history:
        prev = normalize_text(item['question'])
        if candidate == prev or SequenceMatcher(None, candidate, prev).ratio() >= 0.8:
            return True
        if (has_time_topic(question) and has_time_topic(item['question'])
                and not has_money_marker(question) and not has_money_marker(item['question'])
                and SequenceMatcher(None, candidate, prev).ratio() >= 0.45):
            return True
    return False


def normalize_condition(condition):
    text = unicodedata.normalize('NFKC', condition).casefold()
    return re.sub(r'[\s「」『』（）()、。・:：?？!！,.]+', '', text)


def condition_repeated(condition, history):
    candidate = normalize_condition(condition)
    if not candidate:
        return False
    for item in history:
        prev = item.get('condition')
        if not prev:
            continue
        prev = normalize_condition(prev)
        if not prev:
            continue
        if candidate == prev or candidate in prev or prev in candidate:
            return True
        if SequenceMatcher(None, candidate, prev).ratio() >= 0.8:
            return True
    return False


CANDIDATE_STOPWORDS = {
    '迷っています', '迷って', '迷っ', '悩んで', '悩んでいます', '考えています', '検討しています',
    '選びたい', '決めかねています', '思っています', '希望します', '希望です', 'したいです', 'です', 'ます',
}

MULTI_DAY_PATTERN = re.compile(r'\d+\s*[〜~－\-–]\s*\d+\s*日|\d+\s*泊|\d+\s*日間|数日間|複数日|まる\d+日')


def stem_candidate(bit):
    return re.sub(r'(に行く|に行き|にしようか|にしよう|にする|を選ぶ|を選びたい|で行く|で過ごす|で楽しむ|で休む|で買う|を買う|しますか|します|したい|です|ます)$', '', bit)


def extract_candidates(consultation):
    text = unicodedata.normalize('NFKC', consultation).casefold()
    candidates = []
    for bit in re.split(r'[、。\s?？と、か|それともまたは]', text):
        bit = stem_candidate(bit.strip())
        if len(bit) < 2 or bit in CANDIDATE_STOPWORDS:
            continue
        stemmed = normalize_text(bit)
        if stemmed and stemmed not in candidates:
            candidates.append(stemmed)
    return candidates


def result_text(result):
    return normalize_text(' '.join([
        result.get('recommendation', ''),
        result.get('reason', ''),
        result.get('alternative', ''),
        *result.get('conditions', []),
        *result.get('pros', []),
        *result.get('cons', []),
    ]))


def answer_implies_multiple_days(answer):
    a = unicodedata.normalize('NFKC', answer).casefold()
    return bool(MULTI_DAY_PATTERN.search(a)) and '日帰り' not in a


def answer_is_single_day(answer):
    return '日帰り' in unicodedata.normalize('NFKC', answer).casefold()


def result_contradicts_history(result, history):
    text = result_text(result)
    for item in history:
        if answer_implies_multiple_days(item['answer']) and any(w in text for w in ('日帰り', '一泊', '0泊')):
            return '回答で複数日に渡る日程と答えたのに、結果で「日帰り」が適しているとしています'
        if answer_is_single_day(item['answer']) and '泊' in text and '日帰り' not in text:
            return '回答で「日帰り」と答えたのに、結果に宿泊が含まれています'
    return ''


def result_addresses_all_candidates(result, consultation):
    candidates = extract_candidates(consultation)
    if len(candidates) < 2:
        return True
    text = result_text(result)
    return all(candidate in text for candidate in candidates)


def result_reason_is_grounded(result, history):
    if not history:
        return True
    base = normalize_text(result.get('reason', '') + result.get('recommendation', ''))
    if not base:
        return False
    return any(
        len(normalize_text(item['answer'])) >= 2 and normalize_text(item['answer']) in base
        for item in history
    )


def result_gate_issue(result, consultation, history):
    contradiction = result_contradicts_history(result, history)
    if contradiction:
        return contradiction
    if not result_addresses_all_candidates(result, consultation):
        return '相談で比較している複数の候補のうち片方しか取り上げず、候補同士の比較がないまま結論を出しています'
    if not result_reason_is_grounded(result, history):
        return 'recommendationとreasonがユーザーの回答内容と結びついておらず、根拠が示されていません'
    return ''


def asks_user_to_decide(question, consultation):
    raw = unicodedata.normalize('NFKC', question).casefold()
    if re.search(r'[方か]が?[いい良い]|にする[方かが]?[いい良い]|にし[ようよ]う|を選び|がいいと|にしま[すせんか]|を決め|決めなさい|決められます|どちら[がは].*[いい良い]|どっち[がは].*[いい良い]', raw):
        return True
    if re.search(r'[がは]おすすめ|おすすめ[はかが]', raw):
        return True
    needle = normalize_text(consultation)
    parts = re.split(r'[と・、,。か\s]', raw)
    matched = sum(1 for part in parts if len(normalize_text(part)) > 1 and normalize_text(part) in needle)
    if matched >= 2 and re.search(r'[方か]が?[いい良い]|にする|を選び', raw):
        return True
    return False


@app.route('/send_api', methods=['POST'])
def send_api():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='相談内容をJSON形式で送信してください。'), 400
    consultation = data.get('consultation')
    history = data.get('history', [])
    force_result = bool(data.get('force_result'))
    if not isinstance(consultation, str) or not consultation.strip():
        return jsonify(error='相談内容を入力してください。'), 400
    if not isinstance(history, list) or len(history) > MAX_QUESTIONS or any(
        not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ('question', 'answer')
        ) for item in history
    ):
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
            if result['status'] == 'question' and (
                is_repeated_question(result['question'], history)
                or (result.get('condition') and condition_repeated(result['condition'], history))
            ):
                messages[0]['content'] += '\n直前の生成候補「' + result['question'] + '」（condition: ' + (result.get('condition') or 'なし') + '）は回答済みの同じ判断条件の繰り返しなので却下しました。この質問の言い換えも禁止です。別の判断条件を質問してください。'
                continue
            if result['status'] == 'question' and (is_filler_question(result['question']) or reuses_last_answer(result['question'], history) or is_malformed_question(result['question']) or asks_direct_choice(result['question'], consultation) or asks_user_to_decide(result['question'], consultation) or has_unrelated_choice(result['question'], result.get('options', [])) or question_restates_consultation(result['question'], consultation) or asks_comparison_result(result['question'], consultation) or presumes_single_candidate(result['question'], consultation)):
                messages[0]['content'] += '\n直前の生成候補の質問文「' + result['question'] + '」は定型句、直前の回答の使い回し、選択肢の埋め込み、相談の結論をそのまま聞き返す直接比較、ユーザーに結論を直接選ばせる質問、片方の候補だけを前提にした質問、場所などの質問トピックと無関係な語句を含む選択肢、または相談内容の引き写し・前置き（「〜とのことですが」など）だったため却下しました。質問トピックに直結する自然な選択肢、または判断材料となる具体的な条件（時間・予算・疲れ具合・同行者など）を1つだけ、前置きなしで質問してください。'
                continue
            if result['status'] == 'question' and result['answer_type'] == 'text':
                unit = numeric_unit_for(result['question'])
                if unit:
                    result['answer_type'] = 'number'
                    result['unit'] = unit
            if result['status'] == 'result' and not require_result and len(history) < MIN_QUESTIONS_FOR_RESULT:
                messages[0]['content'] += '\n回答数が' + str(len(history)) + 'と少なく情報が不足しています。最終結果は出さず、不足している判断条件を質問してください。最低' + str(MIN_QUESTIONS_FOR_RESULT) + '問の回答が揃うまで質問を続けてください。'
                continue
            if result['status'] == 'result' and not require_result:
                gate_issue = result_gate_issue(result, consultation, history)
                if gate_issue:
                    messages[0]['content'] += '\n直前の生成候補の最終結果は却下しました。理由: ' + gate_issue + ' 候補を比較し、ユーザーの回答内容を根拠に、回答と矛盾しない最終結果を再生成してください。'
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
            if result['status'] == 'result' and is_conditional_recommendation(result['recommendation']):
                result['recommendation'] = clean_conditional_recommendation(result['recommendation'])
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
    except APIError as error:
        app.logger.error('Ollama API call failed: %s', error)
        return jsonify(error='AIとの通信に失敗しました。Ollamaの起動とモデルの取得を確認してください。'), 502


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        use_reloader=True,
        extra_files=[str(BACKEND_DIR / 'prompt.txt')],
    )
