import unittest
from unittest.mock import Mock
from backend.decision import decide
from backend.pc_recovery import recover, template_step


class GamingTests(unittest.TestCase):
    consultation = 'ノートPCにするかデスクトップPCにするか迷っています。'

    def history(self):
        return [dict(question=k + 'は？', condition=k, answer=v) for k, v in [
            ('持ち運び', 'たまに'), ('主な用途', 'ゲーム'), ('設置場所', '常設できる'),
            ('購入予算', '15万円以上20万円未満')]]

    def test_reported_four_answers_need_game_and_portability_details_without_ai(self):
        history = self.history()
        client = Mock()
        for condition, answer in [('遊びたいゲーム', '映像が細かい3Dゲーム'),
                                  ('持ち運びの必要性', '持ち運べると便利だが必須ではない')]:
            response = decide(client, 'test-model', self.consultation, history)
            self.assertEqual(response['condition'], condition)
            history.append(dict(question=response['question'], condition=condition, answer=answer))
        client.chat.completions.create.assert_not_called()
        self.assertEqual(template_step(self.consultation, history, []), (None, True))
        self.assertEqual(decide(client, 'test-model', self.consultation, history)['recommendation'], 'デスクトップPC')
        client.chat.completions.create.assert_not_called()
        self.assertEqual(recover(self.consultation, history, [], True)['recommendation'], 'デスクトップPC')
        history[-1]['answer'] = '同じパソコンを外でも使う必要がある'
        self.assertEqual(decide(client, 'test-model', self.consultation, history)['recommendation'], 'ノートPC')
        self.assertEqual(recover(self.consultation, history, [], True)['recommendation'], 'ノートPC')

    def test_unknown_and_skipped_details_are_not_repeated(self):
        history = self.history()
        game, _ = template_step(self.consultation, history, [])
        history.append(dict(question=game['question'], condition=game['condition'], answer='わからない'))
        portability, _ = template_step(self.consultation, history, [])
        self.assertEqual(template_step(self.consultation, history, [portability['question']]), (None, True))
        self.assertEqual(recover(self.consultation, self.history(), [], True)['recommendation'], '現時点では保留')
