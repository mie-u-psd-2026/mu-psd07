"""単純なノート/デスクトップ比較の復帰経路。自由な相談内容は推測しない。"""
import re
import json
from pathlib import Path


def matches(consultation):
    compact = re.sub(r'[\s。．、,！？!?]', '', consultation)
    pattern = r'(?:今回の相談)?ノート(?:PC|パソコン)(?:にする)?かデスクトップ(?:PC|パソコン)(?:にする)?か(?:で)?(?:迷っています|迷う)'
    return bool(re.fullmatch(pattern, compact, re.IGNORECASE))


def gaming_result(consultation, history):
    """追加回答で優先条件が明確なゲーム用途だけ、回答に根拠を限定して比較する。"""
    if not matches(consultation):
        return None
    answers = {h.get('condition'): h['answer'] for h in history}
    if not {'持ち運び', '主な用途', '設置場所', '購入予算', '遊びたいゲーム', '持ち運びの必要性'} <= answers.keys():
        return None
    if answers['主な用途'] != 'ゲーム' or answers['遊びたいゲーム'] not in ('映像が細かい3Dゲーム', '対戦ゲームを滑らかに遊びたい'):
        return None
    if answers['持ち運びの必要性'] == '同じパソコンを外でも使う必要がある':
        choice = 'ノートPC'
        reason = 'ゲーム用途でも、同じPCを外出先で使う必要があるという条件を優先しました。希望するゲームの動作を満たす機種か確認が必要です。'
        pros = ['自宅と外出先で同じPCを使える']
        alternative = '外出先で同じPCを使う必要がなくなり、常設できるならデスクトップPCも比較してください。'
    elif answers['設置場所'] == '常設できる' and answers['持ち運びの必要性'] in ('持ち運べると便利だが必須ではない', '自宅で使えればよい'):
        choice = 'デスクトップPC'
        reason = '持ち運びは必須ではなく常設もできるため、3Dゲームや対戦ゲームの動作を優先し、デスクトップPCを中心に比較することを勧めます。'
        pros = ['持ち運びに合わせる必要がなく、ゲーム用の本体や画面を選べる']
        alternative = '同じPCを外出先で使う必要があるなら、ゲームに対応したノートPCが候補になります。'
    else:
        return None
    return dict(status='result', recommendation=choice,
                conditions=[f'{h.get("condition") or h["question"]}：{h["answer"]}' for h in history[-10:]],
                reason=reason, pros=pros,
                cons=['予算内で希望のゲームが動くか、本体と周辺機器の合計額を機種ごとに確認する必要がある',
                      '動作音・発熱・ゲーム性能は機種によって異なる'],
                alternative=alternative, confidence=70)


def template_step(consultation, history, skipped):
    """基本4条件と、用途によって結論を変える追加条件を即時表示する。"""
    if not matches(consultation):
        return None, False
    catalog = json.loads(Path(__file__).with_name('question_examples.json').read_text(encoding='utf-8'))
    conditions = ['持ち運び', '主な用途', '設置場所', '購入予算']
    answers = {h.get('condition'): h['answer'] for h in history}
    if answers.get('主な用途') == 'ゲーム':
        catalog.extend([
            dict(condition='遊びたいゲーム', question='主にどんなゲームを遊びたいですか？',
                 options=['軽めのゲームが中心', '映像が細かい3Dゲーム', '対戦ゲームを滑らかに遊びたい', 'まだ決めていない']),
            dict(condition='持ち運びの必要性', question='外出先でも、このパソコンを使う必要がありますか？',
                 options=['同じパソコンを外でも使う必要がある', '持ち運べると便利だが必須ではない', '自宅で使えればよい', 'まだ決めていない']),
        ])
        conditions.append('遊びたいゲーム')
        if answers.get('持ち運び') != 'ほとんど持ち運ばない':
            conditions.append('持ち運びの必要性')
    answered = {h.get('condition') for h in history}
    excluded = {re.sub(r'\W', '', q) for q in [*(h['question'] for h in history), *skipped]}
    for index, condition in enumerate(conditions):
        item = next(e for e in catalog if e['condition'] == condition)
        if condition in answered or re.sub(r'\W', '', item['question']) in excluded:
            continue
        return dict(status='question', question=item['question'], answer_type='choice',
                    options=item['options'], condition=condition, confidence=min(index * 15, 75)), False
    return None, True


QUESTIONS = [
    ('設置場所', '机にパソコンとモニターを常設できる場所はありますか？',
     ['常設できる', '使うたびに片付けたい', 'まだ決めていない']),
    ('重視する点', '使い勝手で、特に重視したいことは何ですか？',
     ['場所を変えて使えること', '画面や機器の構成を選べること', '特にこだわりはない']),
]


def recover(consultation, history, skipped, force_result=False):
    # 追加の予算・用途・制約が相談文にある場合は、この単純なルールを適用しない。
    if not matches(consultation):
        return None
    answers = {h.get('condition'): h['answer'] for h in history if h.get('condition')}
    if not force_result and not {'持ち運び', '主な用途'} <= answers.keys():
        return None
    recorded = {h['question'] for h in history} | set(skipped)
    if not force_result:
        for condition, question, options in QUESTIONS:
            if condition not in answers and question not in recorded:
                return dict(status='question', question=question, answer_type='choice',
                            condition=condition, options=options, confidence=0)
    carry = answers.get('持ち運び', '')
    space = answers.get('設置場所', '')
    priority = answers.get('重視する点', '')
    if (answers.get('持ち運びの必要性') == '同じパソコンを外でも使う必要がある'
            or ('持ち運びの必要性' not in answers and carry in ('ほぼ毎日', '週に数回'))):
        choice, reason = 'ノートPC', f'持ち運びについて「{carry}」と回答しているため、持ち運べるノートPCを暫定的に勧めます。'
    elif space == '使うたびに片付けたい' or priority == '場所を変えて使えること':
        choice, reason = 'ノートPC', '片付けや場所を変えて使う希望に合わせ、ノートPCを暫定的に勧めます。'
    elif (space == '常設できる' and answers.get('主な用途') == 'ゲーム'
          and answers.get('持ち運びの必要性') in ('持ち運べると便利だが必須ではない', '自宅で使えればよい')
          and answers.get('遊びたいゲーム') in ('映像が細かい3Dゲーム', '対戦ゲームを滑らかに遊びたい')):
        choice, reason = 'デスクトップPC', 'ゲームの動作を重視し、常設できて持ち運びは必須ではないため、デスクトップPCを暫定的に勧めます。性能は機種ごとに確認が必要です。'
    elif carry == 'ほとんど持ち運ばない' and space == '常設できる' and priority == '画面や機器の構成を選べること':
        choice, reason = 'デスクトップPC', '持ち運びがほぼ不要で常設でき、画面や機器の構成を選びたいという回答からの暫定案です。'
    else:
        return dict(status='result', recommendation='現時点では保留',
                    conditions=[f'{k}：{v}' for k, v in answers.items()][-10:] or ['判断条件は未確認です'],
                    reason='詳しい比較を生成できず、確認済みの条件だけでは一方を勧められませんでした。',
                    pros=['確認した条件を購入前の比較に使える'], cons=['おすすめはまだ確定していない'],
                    alternative='持ち運びや収納を重視するならノートPC、常設して機器の構成を選びたいならデスクトップPCを比較してください。', confidence=0)
    return dict(status='result', recommendation=choice,
                conditions=[f'{k}：{v}' for k, v in answers.items()][-10:], reason=reason,
                pros=['回答した使い方や設置方法に合う'],
                cons=['予算と用途に必要な性能は、購入候補の機種で確認が必要'],
                alternative=('持ち運ばず常設できるならデスクトップPCも比較してください。' if choice == 'ノートPC'
                             else '持ち運びや片付けを優先するならノートPCが候補になります。'), confidence=40)
