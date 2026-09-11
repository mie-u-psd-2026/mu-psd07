"""相談と回答を根拠に計画・生成・審査する。状態は毎回履歴から再構築する。"""
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model, model_validator


Text = Annotated[str, Field(min_length=1, max_length=1000)]
ShortText = Annotated[str, Field(min_length=1, max_length=80)]
UNKNOWN_STREAK_LIMIT = 4
DEFAULT_TIMEOUT = 180


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class ChoiceQuestion(StrictModel):
    status: Literal['question']
    question: Text
    answer_type: Literal['choice']
    options: list[Annotated[str, Field(min_length=1, max_length=24)]] = Field(min_length=2, max_length=5)
    condition: ShortText
    confidence: int = Field(ge=0, le=99)


class NumberQuestion(StrictModel):
    status: Literal['question']
    question: Text
    answer_type: Literal['number']
    unit: ShortText
    condition: ShortText
    confidence: int = Field(ge=0, le=99)


class TextQuestion(StrictModel):
    status: Literal['question']
    question: Text
    answer_type: Literal['text']
    condition: ShortText
    confidence: int = Field(ge=0, le=99)


class DecisionResult(StrictModel):
    status: Literal['result']
    recommendation: ShortText
    conditions: list[Text] = Field(min_length=1, max_length=10)
    reason: Text
    pros: list[Text] = Field(min_length=1, max_length=5)
    cons: list[Text] = Field(min_length=1, max_length=5)
    alternative: Text
    confidence: int = Field(ge=0, le=100)


RESPONSE_ADAPTER = TypeAdapter(ChoiceQuestion | NumberQuestion | TextQuestion | DecisionResult)


class Criterion(StrictModel):
    id: ShortText
    label: ShortText
    state: Literal['unasked', 'known', 'unknown'] = Field(description='unasked=未質問、known=具体的回答あり、unknown=本人が不明と回答済み')
    value: str = Field(max_length=200, description='未質問なら空文字')
    evidence: str = Field(max_length=300, description='本人の発言の原文引用。未質問なら空文字')
    impact: Text
    priority: int = Field(ge=1, le=5)
    input_reason: str = Field(default='', max_length=300)


class DecisionPlan(StrictModel):
    candidates: list[ShortText] = Field(min_length=2, max_length=5)
    criteria: list[Criterion] = Field(max_length=36)
    ready: bool = Field(description='追加質問が必要ならfalse。推薦できる、または途中終了ならtrue')
    readiness_reason: Text
    next_condition: str = Field(max_length=80)

    @model_validator(mode='after')
    def consistent(self):
        ids = [c.id for c in self.criteria]
        if len(set(ids)) != len(ids) or len(set(self.candidates)) != len(self.candidates):
            raise ValueError('候補と条件IDは重複させないでください')
        if not self.ready:
            selected = next((c for c in self.criteria if c.id == self.next_condition), None)
            if selected is None or selected.state != 'unasked':
                available = [c.id for c in self.criteria if c.state == 'unasked']
                raise ValueError(f'next_condition={self.next_condition}は無効です。未確認のID {available} から選んでください。まだ聞いていない条件はstate=unaskedです')
        return self


class SupportedFact(StrictModel):
    label: ShortText
    value: ShortText
    source: int = Field(ge=0, description='0=相談文、1以降=履歴の回答番号')
    unknown: bool


class PendingCondition(StrictModel):
    label: ShortText
    impact: Text
    priority: int = Field(ge=1, le=5)
    input_reason: str = Field(default='', max_length=300,
        description='通常は空文字。選択肢や範囲で答えられず、固有の文字列や正確な値の入力が不可欠な場合だけ理由を記述')


class PlanningOutline(StrictModel):
    candidates: list[ShortText] = Field(min_length=2, max_length=5,
        description='consultationで比較している候補。相談に候補があるなら新しい商品名や下位分類にすり替えない')
    readiness_reason: Text
    ready: bool = Field(description='本人の事情から候補を推奨できるならtrue。商品の実売価格など未確認の外部情報は結論に留保を添える')
    pending: list[PendingCondition] = Field(max_length=6,
        description='ready=trueなら空配列。falseなら、本人が自分の事情から答えられて推薦を変えうる未回答条件だけ')


class PlanningDraft(PlanningOutline):
    facts: list[SupportedFact] = Field(default_factory=list, max_length=15)


class QuestionDraft(StrictModel):
    # 短い判断上の目的だけを保持し、詳細な推論や内部計画は画面へ返さない。
    purpose: Text
    input_reason: str = Field(max_length=300)
    response: ChoiceQuestion | NumberQuestion | TextQuestion


class ChoiceDraft(QuestionDraft):
    response: ChoiceQuestion


class PracticalPlan(StrictModel):
    candidates: list[ShortText] = Field(min_length=2, max_length=5)
    purpose: Annotated[str, Field(min_length=1, max_length=160)] = Field(
        description='本人のどの事情を確認すると候補の判断が変わるか。結論なら既知の判断材料')
    ready: bool
    next_condition: str = Field(max_length=80, description='これから本人に確認する条件名。結論なら空文字')


class PracticalTurn(StrictModel):
    # 計画を先に出力し、その条件に対応する質問を後から作る。通信は1回。
    plan: PracticalPlan
    response: ChoiceQuestion | DecisionResult


class PracticalResultPlan(PracticalPlan):
    ready: Literal[True]
    next_condition: Literal['']


class PracticalResultTurn(PracticalTurn):
    plan: PracticalResultPlan
    response: DecisionResult


class ExampleQuestion(StrictModel):
    status: Literal['example']
    example_id: int = Field(ge=0, le=1)
    confidence: int = Field(ge=0, le=99)


class PracticalExampleTurn(PracticalTurn):
    response: ExampleQuestion | ChoiceQuestion | DecisionResult


class ReviewBase(StrictModel):
    @property
    def approved(self):
        return all(value for key, value in self.model_dump().items() if key != 'issues') and not self.issues


class QuestionReview(ReviewBase):
    relevant: bool = Field(description='質問が選択対象の条件に対応して判断に役立てばtrue')
    new_information: bool = Field(description='実際の相談・回答履歴で未回答の内容ならtrue。計画に書いてあるだけでは回答済みではない')
    answerable: bool = Field(description='初心者が自分の事情から1項目について答えられればtrue')
    neutral: bool = Field(description='最終候補の比較判断を本人に丸投げしていなければtrue')
    options_fit: bool = Field(description='選択肢が質問への自然で異なる回答ならtrue')
    choice_priority: bool = Field(description='answer_type=choiceならtrue。number/textは選択肢では足りない理由がある場合だけtrue')
    issues: list[Text] = Field(max_length=8, description='falseの項目の理由だけ。すべてtrueなら空配列')


class ResultReview(ReviewBase):
    plan_grounded: bool
    result_grounded: bool
    candidates_compared: bool
    uncertainty_honest: bool
    issues: list[Text] = Field(max_length=8, description='falseの項目の理由だけ。すべてtrueなら空配列')


class QuestionIssue(StrictModel):
    criterion: Literal['relevant', 'new_information', 'answerable',
                       'neutral', 'options_fit', 'choice_priority']
    reason: Text = Field(description='今回の案に存在する具体的な欠点と、相談・回答・案のどこが根拠か')


class ResultIssue(StrictModel):
    criterion: Literal['plan_grounded', 'result_grounded', 'candidates_compared', 'uncertainty_honest']
    reason: Text = Field(description='今回の結果に存在する具体的な欠点と、その根拠')


class QuestionInspection(StrictModel):
    issues: list[QuestionIssue] = Field(max_length=7, description='実在する欠点だけ。問題がなければ空配列[]')

    def review(self):
        return QuestionReview(**{key: not any(i.criterion == key for i in self.issues)
            for key in QuestionReview.model_fields if key != 'issues'},
            issues=[i.reason for i in self.issues])


class ResultInspection(StrictModel):
    issues: list[ResultIssue] = Field(max_length=4, description='実在する欠点だけ。問題がなければ空配列[]')

    def review(self):
        return ResultReview(**{key: not any(i.criterion == key for i in self.issues)
            for key in ResultReview.model_fields if key != 'issues'},
            issues=[i.reason for i in self.issues])


class DecisionQualityError(ValueError):
    pass


def normalized(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text)).casefold()


def question_examples(consultation, condition, excluded, answered_conditions=()):
    """確認済みの聞き方を参考にする。回答済み・スキップ済みの文面は例から外す。"""
    catalog = json.loads(Path(__file__).with_name('question_examples.json').read_text(encoding='utf-8'))
    excluded = {normalized(q) for q in excluded}
    answered = {normalized(label) for label in answered_conditions}
    usable = [item for item in catalog if normalized(item['question']) not in excluded
              and normalized(item['condition']) not in answered]
    def score(item):
        return sum(2 * (normalized(word) in normalized(consultation)) +
                   (normalized(word) in normalized(condition)) for word in item['keywords'])
    return [{key: value for key, value in item.items() if key != 'keywords'}
            for item in sorted((item for item in usable if score(item) > 0), key=score, reverse=True)[:2]]


def is_unknown_answer(answer):
    return normalized(answer) in {normalized(value) for value in (
        'わからない・決められない', 'わからない', '不明', '未定')}


def materialize_plan(draft, consultation, history):
    """原文・ID・状態・優先順位の対応はモデルに再生成させない。"""
    sources = [consultation, *(item['answer'] for item in history)]
    criteria = []
    # 画面から回答済み条件が分かる場合、モデルによる要約・根拠番号の再生成は不要。
    recorded_sources = set()
    for source, item in enumerate(history, 1):
        if not item.get('condition'):
            continue
        recorded_sources.add(source)
        answer = item['answer']
        criteria.append(Criterion(id=f'answer_{source}', label=item['condition'],
            state='unknown' if is_unknown_answer(answer) else 'known',
            value=answer[:200], evidence=answer[:300],
            impact='本人の回答をそのまま保持した判断条件', priority=1))
    for index, fact in enumerate(draft.facts):
        if fact.source >= len(sources):
            raise DecisionQualityError('存在する回答番号だけを根拠にしてください')
        if fact.source in recorded_sources or any(normalized(c.label) == normalized(fact.label) for c in criteria):
            continue
        criteria.append(Criterion(id=f'known_{index}', label=fact.label,
            state='unknown' if fact.unknown else 'known', value=fact.value,
            evidence=sources[fact.source][:300], impact='本人の発言に基づく判断条件', priority=1))
    answered = {normalized(c.label) for c in criteria}
    answered.update(normalized(item['condition']) for item in history if item.get('condition'))
    for index, pending in enumerate(draft.pending):
        if normalized(pending.label) in answered:
            continue
        criteria.append(Criterion(id=f'ask_{index}', label=pending.label, state='unasked',
            value='', evidence='', impact=pending.impact, priority=pending.priority,
            input_reason=pending.input_reason))
    unasked = [c for c in criteria if c.state == 'unasked']
    if not draft.ready and not unasked:
        raise DecisionQualityError('回答済み以外の必要な条件をpendingに入れるか、情報が十分ならreadyをtrueにしてください')
    return DecisionPlan(candidates=draft.candidates, criteria=criteria, ready=draft.ready,
        readiness_reason=draft.readiness_reason,
        next_condition='' if draft.ready else max(unasked, key=lambda c: c.priority).id)


def structured_call(client, model, schema, system, payload, deadline, max_tokens=2200, compact=False):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError('相談の処理時間を超過しました')
    completion = client.chat.completions.create(
        model=model, temperature=0.7, timeout=remaining, max_tokens=max_tokens,
        messages=[{'role': 'system', 'content': system if compact else system + '\nKeep reasoning brief: at most 200 words. Do not restate the input or schema.\n出力スキーマ:\n' + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
                  {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}],
        response_format={'type': 'json_schema', 'json_schema': {
            'name': schema.__name__, 'schema': schema.model_json_schema(),
        }},
    )
    if monotonic() > deadline:
        raise TimeoutError('相談の処理時間を超過しました')
    if not completion.choices:
        raise ValueError('Empty model response')
    return schema.model_validate_json(completion.choices[0].message.content or '')


def validate_plan_sources(plan, consultation, history):
    sources = [consultation, *(item['answer'] for item in history)]
    for condition in plan.criteria:
        if condition.state in ('known', 'unknown'):
            if not condition.evidence or not any(condition.evidence in source for source in sources):
                raise DecisionQualityError('既知・不明の条件には相談または回答からの原文引用が必要です')
        elif condition.value or condition.evidence:
            raise DecisionQualityError('未確認の条件に値や根拠を補ってはいけません')


def validate_draft(response, plan, history, skipped, draft=None):
    if response.status == 'result':
        if response.recommendation not in plan.candidates:
            raise DecisionQualityError('おすすめは比較中の候補名を1つ、そのまま指定してください')
        return
    selected = next(c for c in plan.criteria if c.id == plan.next_condition)
    # 条件名はサーバーが選んだ内部情報。モデルによる表記揺れで再生成しない。
    # 質問そのものがこの条件に対応しているかは、別の意味の審査で確認する。
    response.condition = selected.label
    question = normalized(response.question)
    if len(question) < 4 or '\n' in response.question or len(response.question) > 100:
        raise DecisionQualityError('質問は短い1文にしてください')
    if any(question == normalized(item['question']) for item in history):
        raise DecisionQualityError('回答済みの質問です。別の未確認条件を選んでください')
    if any(question == normalized(item) for item in skipped):
        raise DecisionQualityError('スキップされた質問の文面を再利用しないでください')
    if (response.answer_type == 'choice' and '静音' in response.question
            and not any(word in response.question for word in ('ファン', '動作音', '冷却'))
            and any('pc' in normalized(c) or 'パソコン' in c for c in plan.candidates)):
        # 意図を保ち、専門語だけの聞き方を表示前に具体化する。AIの再生成は不要。
        response.question = 'パソコンを使うとき、ファンなどの音はどのくらい気になりますか？'
        response.options = ['できるだけ静かな方がよい', '多少の音は気にならない', '特に気にしない']
        if normalized(response.question) in {normalized(q) for q in [
                *(h['question'] for h in history), *skipped]}:
            raise DecisionQualityError('音については回答済みまたはスキップ済みです。別の条件へ進んでください')
    if response.answer_type == 'choice':
        options = [normalized(option) for option in response.options]
        if len(options) < 2 or len(set(options)) != len(options) or any(not option for option in options):
            raise DecisionQualityError('選択肢は空でない異なる回答候補にしてください')
        if any(re.search(r'[?？]|ますか|ですか|でしょうか', o) or len(o) > 24 for o in response.options):
            raise DecisionQualityError('選択肢は質問文ではなく24文字以内の短い回答にしてください')
    elif not draft or not draft.input_reason.strip():
        raise DecisionQualityError('選択式が優先です。数値・自由入力が不可欠な理由を示してください')


def validate_practical_question(response, plan):
    if response.status != 'question':
        return
    # 飼う前の犬・猫の比較で、動物の一般知識を利用者に回答させない。
    if any('犬' in c for c in plan.candidates) and any('猫' in c for c in plan.candidates):
        if re.search(r'^(?:犬|猫)(?:は|って|の習性|の性格|の寿命).*(?:鳴|吠|夜行性|習性|性格|寿命|生き)', response.question):
            raise DecisionQualityError('動物の習性はAIが考慮する知識です。本人の住居・留守時間・世話に使える時間を聞いてください')
    # 軽量モードでも、結論をそのまま本人に選ばせる明白な質問は出さない。
    if re.search(r'どちら.*(?:適して|好き|良い|よい|選びたい|希望します|考えて|思い)', response.question):
        raise DecisionQualityError('結論を聞き返さず、本人の生活・目的・制約を質問してください')
    if response.answer_type == 'choice':
        candidates = {normalized(c) for c in plan.candidates}
        if len(candidates.intersection(normalized(o) for o in response.options)) >= 2:
            raise DecisionQualityError('最終候補を選択肢にせず、判断に使う本人の事情を聞いてください')


def practical_decide(client, model, consultation, history, force_result, skipped, deadline, trace, unknown_stop):
    # 質問は軽量モデル、結論は比較・根拠の説明が安定する大きいモデルを使う。
    result_model = os.environ.get('OLLAMA_RESULT_MODEL', 'qwen3:8b') if model == 'qwen3:4b-instruct-2507-q4_K_M' else model
    prompt = Path(__file__).with_name('practical_prompt.txt').read_text(encoding='utf-8')
    payload = {'consultation': consultation, 'history': history, 'force_result': force_result,
               'skipped_questions': skipped or [], 'question_examples': question_examples(
                   consultation, '', [*(h['question'] for h in history), *(skipped or [])],
                   [h.get('condition', '') for h in history])}
    corrections = []
    examples = payload['question_examples']
    if force_result:
        examples = payload['question_examples'] = []
    schema = PracticalResultTurn if force_result else (PracticalExampleTurn if examples else PracticalTurn)
    for attempt in range(2):
        record = {'attempt': attempt + 1, 'unknown_stop': unknown_stop, 'strict_review': False}
        if trace is not None:
            trace.append(record)
        start = monotonic()
        try:
            selected_model = result_model if force_result or attempt else model
            record['model'] = selected_model
            generated = structured_call(client, selected_model, schema,
                prompt, {**payload, 'corrections': corrections}, deadline, max_tokens=800, compact=True)
            record['model_seconds'] = round(monotonic() - start, 3)
            record['draft'] = generated.model_dump()
            response = generated.response
            if response.status == 'result' and not force_result and selected_model != result_model:
                return practical_decide(client, model, consultation, history, True, skipped,
                                        deadline, trace, unknown_stop)
            if isinstance(response, ExampleQuestion):
                if response.example_id >= len(examples):
                    raise DecisionQualityError('存在するquestion_examplesの番号を選んでください')
                example = examples[response.example_id]
                # 参照番号で条件が一意に決まる。モデルの重複記入の不一致で再生成しない。
                generated.plan.next_condition = example['condition']
                generated.plan.purpose = example['purpose']
                response = ChoiceQuestion(status='question', answer_type='choice',
                    question=example['question'], options=example['options'],
                    condition=example['condition'], confidence=response.confidence)
                record['question_source'] = 'example'
            generated.plan.ready = response.status == 'result'
            if isinstance(response, ChoiceQuestion):
                response.question = ' '.join(response.question.splitlines()).strip()
                unique = {}
                for option in response.options:
                    unique.setdefault(normalized(option), option)
                response.options = list(unique.values())
            if isinstance(response, ChoiceQuestion) and not generated.plan.next_condition:
                generated.plan.next_condition = response.condition
            pending = [] if generated.plan.ready else [PendingCondition(
                label=generated.plan.next_condition, impact=generated.plan.purpose, priority=5)]
            plan = materialize_plan(PlanningDraft(candidates=generated.plan.candidates, pending=pending,
                ready=generated.plan.ready, readiness_reason=generated.plan.purpose), consultation, history)
            record['plan'] = plan.model_dump()
            validate_plan_sources(plan, consultation, history)
            validate_draft(response, plan, history, skipped or [])
            validate_practical_question(response, plan)
            if response.status == 'result':
                response.conditions = [f"{h.get('condition') or h['question']}：{h['answer']}"
                                       for h in history[-10:]] if history else [
                                           f'相談：{consultation}', '追加の条件は未確認です']
            record['review_mode'] = 'basic_checks'
            return response.model_dump()
        except ValueError as error:
            record['error'] = str(error)
            logging.getLogger(__name__).warning('Decision validation failed: attempt=%s type=%s',
                                               attempt + 1, type(error).__name__)
            corrections.append(str(error)[:500])
    fallback = pc_recovery_question(consultation, history, skipped or []) if not force_result else None
    if fallback is None:
        if __package__:
            from .pc_recovery import recover
        else:
            from pc_recovery import recover
        fallback = recover(consultation, history, skipped or [], force_result)
    if fallback is not None:
        if trace is not None:
            trace.append({'recovery': 'pc_guided', 'status': fallback['status']})
        return RESPONSE_ADAPTER.validate_python(fallback).model_dump()
    raise DecisionQualityError('質問を整えられませんでした。入力は保持されています。再試行してください。')


def pc_recovery_question(consultation, history, skipped):
    """PC形態の比較で適用範囲が明確な未回答質問へ復帰する。用途を推測しない。"""
    text = normalized(consultation)
    if not (any(word in text for word in ('ノートpc', 'ノートパソコン'))
            and any(word in text for word in ('デスクトップpc', 'デスクトップパソコン'))):
        return None
    # 相談文に既に用途・持ち運びが書かれている可能性があれば、自動補充しない。
    known = {
        '主な用途': ('用途', '使', 'ゲーム', '動画', '閲覧', '文書', '仕事', '学習', '勉強', '制作', '開発', 'プログラミング'),
        '持ち運び': ('持ち運', '持ち出', '外出', '自宅', '家で', '通学', '通勤', '毎日'),
    }
    for item in question_examples(consultation, '',
            [*(h['question'] for h in history), *skipped],
            [h.get('condition', '') for h in history]):
        label = item['condition']
        if label not in known or any(word in text for word in known[label]):
            continue
        # 古い履歴など条件名がない場合も、別文面で同じ条件を聞かない。
        if any(any(word in normalized(h['question']) for word in known[label]) for h in history):
            continue
        return ChoiceQuestion(status='question', answer_type='choice', question=item['question'],
            condition=label, options=item['options'], confidence=0).model_dump()
    return None


def decide(client, model, consultation, history, force_result=False, skipped=None,
           timeout=DEFAULT_TIMEOUT, trace=None, strict_review=False, fast_templates=True):
    """通常は計画・生成と形式検証。別AIによる審査は厳格検証時だけ行う。"""
    deadline = monotonic() + timeout
    # 別の条件を確認する余地を残し、不明が4回続いたら暫定結論へ進む。
    unknown_stop = len(history) >= UNKNOWN_STREAK_LIMIT and all(
        is_unknown_answer(item['answer']) for item in history[-UNKNOWN_STREAK_LIMIT:])
    force_result = force_result or unknown_stop
    if fast_templates and not strict_review and not force_result:
        if __package__:
            from .fast_questions import template_step
        else:
            from fast_questions import template_step
        question, complete = template_step(consultation, history, skipped or [])
        if question:
            if trace is not None:
                trace.append({'question_source': 'template', 'model_calls': 0})
            return RESPONSE_ADAPTER.validate_python(question).model_dump()
        force_result = complete
    if not strict_review:
        return practical_decide(client, model, consultation, history, force_result, skipped,
                                deadline, trace, unknown_stop)
    prompt_dir = Path(__file__).resolve().parent
    common = (prompt_dir / 'prompt.txt').read_text(encoding='utf-8')
    planning = (prompt_dir / 'planning_prompt.txt').read_text(encoding='utf-8')
    reviewing = (prompt_dir / 'review_prompt.txt').read_text(encoding='utf-8')
    payload = {'consultation': consultation, 'history': history,
               'skipped_questions': skipped or [], 'force_result': force_result}
    feedback = []
    plan = None
    attempted_conditions = set()
    for attempt in range(3):
        record = {'attempt': attempt + 1, 'unknown_stop': unknown_stop, 'strict_review': strict_review}
        if trace is not None:
            trace.append(record)
        try:
            if plan is None:
                outline = structured_call(client, model, PlanningOutline, planning,
                    {**payload, 'corrections': feedback, 'question_examples': question_examples(
                        consultation, '', [*(item['question'] for item in history), *(skipped or [])])}, deadline)
                # 既知情報は相談原文と実際の回答に限定し、モデルに再要約させない。
                planning_draft = PlanningDraft(**outline.model_dump())
                if force_result:
                    planning_draft.ready = True
                plan = materialize_plan(planning_draft, consultation, history)
                attempted_conditions.clear()
            record['plan'] = plan.model_dump()
            validate_plan_sources(plan, consultation, history)
            require_result = force_result or plan.ready
            generation_payload = {**payload, 'plan': plan.model_dump(), 'corrections': feedback}
            if not require_result:
                generation_payload['selected_condition'] = next(
                    c.model_dump() for c in plan.criteria if c.id == plan.next_condition)
                generation_payload['plan']['criteria'] = [c for c in generation_payload['plan']['criteria']
                    if c['state'] != 'unasked' or c['id'] == plan.next_condition]
                generation_payload['question_examples'] = question_examples(consultation,
                    generation_payload['selected_condition']['label'],
                    [*(item['question'] for item in history), *(skipped or [])])
            schema = create_model('DecisionResult', __base__=DecisionResult,
                recommendation=(Literal[tuple(plan.candidates)], ...)) if require_result else (
                QuestionDraft if generation_payload['selected_condition']['input_reason'] else ChoiceDraft)
            generation_prompt = (prompt_dir / 'result_prompt.txt').read_text(encoding='utf-8') if require_result else common
            generated = structured_call(client, model, schema, generation_prompt, generation_payload, deadline)
            draft = None if require_result else generated
            response = generated if require_result else generated.response
            record['draft'] = generated.model_dump()
            validate_draft(response, plan, history, skipped or [], draft)
            review_schema = ResultInspection if require_result else QuestionInspection
            rules = (prompt_dir / ('result_review_prompt.txt' if require_result else 'question_review_prompt.txt')).read_text(encoding='utf-8')
            review = structured_call(client, model, review_schema, reviewing + '\n' + rules, {
                # 未質問の計画を回答済みの事実と混同させない。審査に必要な情報だけ渡す。
                **payload, 'candidates': plan.candidates,
                'known_facts': [c.model_dump() for c in plan.criteria if c.state != 'unasked'],
                'selected_condition': generation_payload.get('selected_condition'),
                'draft': generated.model_dump(),
            }, deadline).review()
            record['review'] = review.model_dump()
            if not review.approved:
                failed = [k for k, v in review.model_dump().items() if v is False]
                raise DecisionQualityError(' / '.join([*failed, *review.issues]))
            return response.model_dump()
        except ValueError as error:
            # JSONの不備も審査不合格と同じ修正経路へ。ネットワーク例外は上位で扱う。
            record['error'] = str(error)
            feedback.append(str(error)[:1000])
            if plan is not None and not (force_result or plan.ready):
                # 同じ条件への修正が堂々巡りにならないよう、計画内の次の有用な条件を試す。
                # 変更後も形式検証・意味の審査はすべて通す。
                attempted_conditions.add(plan.next_condition)
                remaining = [c for c in plan.criteria if c.state == 'unasked'
                             and c.id not in attempted_conditions]
                if remaining:
                    plan.next_condition = max(remaining, key=lambda c: c.priority).id
                else:
                    plan = None
            else:
                plan = None
    raise DecisionQualityError('質問または結果の品質を確認できませんでした。入力を保持して再試行してください。')
