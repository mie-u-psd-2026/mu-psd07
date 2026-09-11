"""画面の相談例に対応する質問を返す。追加条件のある相談はAIに任せる。"""
import json
import re
import unicodedata
from pathlib import Path

if __package__:
    from .pc_recovery import template_step as pc_step, gaming_result
else:
    from pc_recovery import template_step as pc_step, gaming_result


def normalize(text):
    return re.sub(r'[^\w]', '', unicodedata.normalize('NFKC', text)).casefold()


def template_step(consultation, history, skipped):
    flows = json.loads(Path(__file__).with_name('template_flows.json').read_text(encoding='utf-8'))
    flow = next((f for f in flows if normalize(consultation) in
                 {normalize(c) for c in f['consultations']}), None)
    if flow is None:
        question, complete = pc_step(consultation, history, skipped)
        if complete:
            result = gaming_result(consultation, history)
            if result is not None:
                return result, False
        return question, complete
    answered = {normalize(h.get('condition', '')) for h in history}
    excluded = {normalize(q) for q in [*(h['question'] for h in history), *skipped]}
    for index, item in enumerate(flow['questions']):
        if normalize(item['condition']) in answered or normalize(item['question']) in excluded:
            continue
        return dict(status='question', answer_type='choice', question=item['question'],
                    options=item['options'], condition=item['condition'], confidence=index * 15), False
    return None, True
