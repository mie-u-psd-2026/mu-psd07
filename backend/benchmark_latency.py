"""実モデルの待ち時間を測る。CPU指定もノートPC実測とは区別する。"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
from time import monotonic

from .decision import decide
from .ollama_client import DEFAULT_MODEL, OllamaClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('evaluation-results/latency.json'))
    args = parser.parse_args()
    if args.cpu:
        os.environ['OLLAMA_NUM_GPU'] = '0'
    client = OllamaClient()
    pet = '一人暮らしで犬を飼うか猫を飼うか迷っています。'
    history = [
        {'question': '住まいでは飼えますか？', 'condition': '住居の飼育制約', 'answer': '犬も猫も飼える'},
        {'question': '留守の時間は？', 'condition': '留守の時間', 'answer': '4時間以上8時間未満'},
    ]
    cases = [
        ('pet_first', pet, [], False, []),
        ('pet_followup', pet, history, False, []),
        ('pet_skip', pet, [], False, ['今の住まいでは、犬や猫を飼えますか？']),
        ('unseen_hobby', '休日に陶芸か写真を始めたいです。どちらが合うか迷っています。', [], False, []),
        ('provisional_result', pet, history, True, []),
    ]
    report = {'model': args.model, 'cpu_only': args.cpu,
              'note': '実行した機器の測定。初回ロード・修正・失敗の時間も含む。品質は別途確認が必要。',
              'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, consultation, answers, force, skipped in cases:
        trace = []
        start = monotonic()
        row = {'name': name}
        try:
            row['response'] = decide(client, args.model, consultation, answers,
                force_result=force, skipped=skipped, trace=trace, timeout=120)
        except Exception as error:
            row['error'] = str(error)
        row.update(seconds=round(monotonic() - start, 3), trace=trace)
        report['cases'].append(row)
        report['average_seconds'] = round(mean(r['seconds'] for r in report['cases']), 3)
        report['failures'] = sum('error' in r for r in report['cases'])
        report['complete'] = len(report['cases']) == len(cases)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k: v for k, v in row.items() if k != 'trace'}, ensure_ascii=False), flush=True)
    print(f"Average: {report['average_seconds']}s; failures: {report['failures']}")


if __name__ == '__main__':
    main()
