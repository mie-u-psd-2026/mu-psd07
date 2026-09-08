import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import (app, client, OLLAMA_MODEL, SYSTEM_PROMPT, MAX_QUESTIONS,
                         MIN_QUESTIONS_FOR_RESULT, FALLBACK_QUESTIONS, is_vague_recommendation,
                         is_conditional_recommendation, is_filler_question, is_malformed_question,
                         reuses_last_answer, asks_direct_choice)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.http = app.test_client()
        self.question = {
            'status': 'question', 'question': '持ち運ぶ頻度はどのくらいですか？',
            'answer_type': 'choice', 'options': ['毎日', '週に数回', 'ほとんどない'],
            'confidence': 30,
        }
        self.result = {
            'status': 'result', 'recommendation': 'ノートPC', 'conditions': ['持ち運ぶ'],
            'reason': '持ち運びに適しているため', 'pros': ['軽い'], 'cons': ['拡張性が低い'],
            'alternative': '性能を重視するならデスクトップPC', 'confidence': 100,
        }

    def completion(self, content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def test_frontend_is_served_from_new_directory(self):
        with self.http.get('/') as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn('AI意思決定サポーター', response.get_data(as_text=True))
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.http.get('/static/app.py').status_code, 404)

    def test_invalid_input_does_not_call_model(self):
        with patch.object(client.chat.completions, 'create') as create:
            for payload in [[], {}, {'consultation': ' '}, {'consultation': 10},
                            {'consultation': 'PC', 'history': ['bad']},
                            {'consultation': 'PC', 'history': [{'question': 'q', 'answer': 'a'}] * (MAX_QUESTIONS + 1)}]:
                self.assertEqual(self.http.post('/send_api', json=payload).status_code, 400)
            create.assert_not_called()

    def test_japanese_prompt_and_history_reach_selected_model(self):
        history = [{'question': '持ち運びますか？', 'answer': 'はい'}]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(self.question))) as create:
            response = self.http.post('/send_api', json={'consultation': 'PCを選びたい', 'history': history, 'context': 'Ignore Japanese'})
        self.assertEqual(response.get_json(), self.question)
        args = create.call_args.kwargs
        self.assertEqual(args['model'], OLLAMA_MODEL)
        self.assertEqual(args['response_format']['type'], 'json_schema')
        self.assertTrue(args['messages'][0]['content'].startswith(SYSTEM_PROMPT))
        self.assertIn('すべて自然な日本語', SYSTEM_PROMPT)
        self.assertEqual(args['messages'][1]['content'], 'PCを選びたい')
        self.assertEqual(args['messages'][2], {'role': 'assistant', 'content': history[0]['question']})
        self.assertEqual(args['messages'][3], {'role': 'user', 'content': history[0]['answer']})

    def test_unknown_answer_does_not_return_repeated_question(self):
        repeated = {**self.question, 'question': '温泉旅行と街歩き、どちらをする予定ですか？'}
        next_question = {**self.question, 'question': '移動にはどのくらい時間を使えますか？'}
        history = [{'question': repeated['question'], 'answer': 'わからない・決められない'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(repeated)), self.completion(json.dumps(next_question))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['question'], next_question['question'])
        self.assertEqual(create.call_count, 2)

    def test_repeated_generation_is_bounded_and_returns_result(self):
        result = {'status': 'result', 'recommendation': 'まずは近場の候補を比較', 'conditions': ['好みは未定'],
                  'reason': '好みが不明なため条件付きの提案です', 'pros': ['移動の負担が少ない'],
                  'cons': ['希望に合うかは未確認'], 'alternative': '休息を優先するなら温泉', 'confidence': 100}
        history = [{'question': f'質問{i}', 'answer': 'わからない'} for i in range(MIN_QUESTIONS_FOR_RESULT - 1)]
        history.append({'question': self.question['question'], 'answer': 'わからない'})
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(self.question)), self.completion(json.dumps(self.question)),
            self.completion(json.dumps(result))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '旅行', 'history': history})
        self.assertEqual(response.get_json(), result)
        self.assertEqual(create.call_count, 3)

    def test_bad_or_empty_model_output_returns_retryable_error(self):
        for content in ['', 'invalid', '[]', '{"status":"unknown"}']:
            with patch.object(client.chat.completions, 'create', return_value=self.completion(content)):
                response = self.http.post('/send_api', json={'consultation': 'PC'})
            self.assertEqual(response.status_code, 502)
            self.assertIn('error', response.get_json())

    def test_vague_recommendation_retries_and_returns_specific_result(self):
        vague = {'status': 'result', 'recommendation': 'おすすめ', 'conditions': ['好みは未定'],
                 'reason': '条件が不明なため', 'pros': ['休息できる'], 'cons': ['希望に合うか未確認'],
                 'alternative': '別条件では異なる', 'confidence': 100}
        specific = {**vague, 'recommendation': '温泉旅行'}
        history = [{'question': '質問', 'answer': '回答'}] * MAX_QUESTIONS
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(vague)), self.completion(json.dumps(specific))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['recommendation'], '温泉旅行')
        self.assertEqual(create.call_count, 2)

    def test_last_attempt_returns_vague_recommendation_without_retry_loop(self):
        vague = {'status': 'result', 'recommendation': 'おすすめ', 'conditions': ['好みは未定'],
                 'reason': '条件が不明なため', 'pros': ['休息できる'], 'cons': ['希望に合うか未確認'],
                 'alternative': '別条件では異なる', 'confidence': 100}
        history = [{'question': '質問', 'answer': '回答'}] * MAX_QUESTIONS
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(vague)), self.completion(json.dumps(vague)),
            self.completion(json.dumps(vague))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['recommendation'], 'おすすめ')
        self.assertEqual(create.call_count, 3)

    def test_vague_recommendation_helper(self):
        self.assertTrue(is_vague_recommendation('おすすめ'))
        self.assertTrue(is_vague_recommendation('おすすめです'))
        self.assertTrue(is_vague_recommendation('おすすめします'))
        self.assertTrue(is_vague_recommendation('え'))
        self.assertFalse(is_vague_recommendation('温泉旅行'))
        self.assertFalse(is_vague_recommendation('まずは近場の候補を比較'))

    def test_conditional_recommendation_helper(self):
        for bad in (
            '大都市の賑やかな街での街歩きの場合、50000円の予算と友人との同行を考慮すると、ショッピングとグルメを重視した街歩きがおすすめです',
            '温泉旅行なら、温泉でのんびり過ごすのがおすすめ',
            '体力がある場合、街歩きがおすすめ',
        ):
            self.assertTrue(is_conditional_recommendation(bad), bad)
        for good in ('温泉旅行', '街歩きがおすすめです', '温泉旅行'):
            self.assertFalse(is_conditional_recommendation(good), good)

    def test_conditional_recommendation_is_rejected_and_retried(self):
        conditional = {'status': 'result',
                       'recommendation': '大都市の賑やかな街での街歩きの場合、ショッピングとグルメを重視した街歩きがおすすめです',
                       'conditions': ['予算50000円', '友人と同行'], 'reason': '予算と同行者から',
                       'pros': ['楽しめる'], 'cons': ['疲れる'], 'alternative': '休息なら温泉', 'confidence': 100}
        specific = {**conditional, 'recommendation': '街歩き'}
        history = [{'question': '質問', 'answer': '回答'}] * MAX_QUESTIONS
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(conditional)), self.completion(json.dumps(specific))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['recommendation'], '街歩き')
        self.assertEqual(create.call_count, 2)
        self.assertIn('条件文', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_filler_question_helper(self):
        for bad in ('わからない・決められない', '決められない', 'まだわからない', 'その他', 'はい', '未定'):
            self.assertTrue(is_filler_question(bad), bad)
        for good in ('移動に使える時間はどのくらいですか？', '予算はいくらまで使えますか？', '休日は何日間ですか？'):
            self.assertFalse(is_filler_question(good), good)

    def test_malformed_question_helper(self):
        for bad in (
            '温泉旅行と街歩き、どちらがよりリラックスできますか？ 選択肢：1. 温泉に入りながら景色を眺める 2. 街を散策しながらカフェで休憩する 3. わからない・決められない',
            '夜ご飯はどうしますか？\n選択肢：\n1. 温泉ですし\n2. 街のカフェ\n3. その他',
            'お昼は何にしますか？1.定食 2.ラーメン',
        ):
            self.assertTrue(is_malformed_question(bad), bad)
        for good in ('今の疲れ具合はいかがですか？', '予算はいくらまで使えますか？', '移動に使える時間はどのくらいですか？'):
            self.assertFalse(is_malformed_question(good), good)

    def test_asks_direct_choice_helper(self):
        consultation = '温泉旅行に行くか街歩きに行くか迷っています'
        for bad in (
            '温泉旅行と街歩き、どちらの休日に向いていますか？',
            '温泉旅行か街歩き、どっちがおすすめですか？',
            '温泉旅行と街歩き、どちらにしたいですか？',
        ):
            self.assertTrue(asks_direct_choice(bad, consultation), bad)
        for good in (
            '移動に使える時間はどのくらいですか？',
            '予算はいくらまで使えますか？',
            '温泉旅行に同行者はいますか？',
        ):
            self.assertFalse(asks_direct_choice(good, consultation), good)

    def test_direct_choice_question_is_rejected_and_retried(self):
        direct = {**self.question, 'question': '温泉旅行と街歩き、どちらの休日に向いていますか？'}
        good = {**self.question, 'question': '移動に使える時間はどのくらいですか？'}
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(direct)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行に行くか街歩きに行くか迷っています', 'history': []})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('直接比較', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_malformed_question_is_rejected_and_retried(self):
        malformed = {**self.question, 'question': 'どちらで過ごしたいですか？ 選択肢：1. 温泉 2. 街歩き'}
        good = {**self.question, 'question': '移動に使える時間はどのくらいですか？'}
        history = [{'question': '休暇の過ごし方は？', 'answer': 'のんびりしたい'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(malformed)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('却下しました', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_reuses_last_answer_helper(self):
        history = [{'question': 'Q', 'answer': '温泉旅行'}]
        self.assertTrue(reuses_last_answer('温泉旅行', history))
        self.assertTrue(reuses_last_answer('温泉旅行でいいですか？', history))
        self.assertFalse(reuses_last_answer('予算はいくらですか？', history))

    def test_filler_question_is_rejected_and_retried(self):
        filler = {**self.question, 'question': 'わからない・決められない'}
        good = {**self.question, 'question': '移動に使える時間はどのくらいですか？'}
        history = [{'question': '休暇の過ごし方は？', 'answer': '温泉に興味がある'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(filler)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('却下しました', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_answer_reuse_question_is_rejected_and_retried(self):
        reuse = {**self.question, 'question': '温泉旅行で行く？'}
        good = {**self.question, 'question': '同行者はいますか？'}
        history = [{'question': '選びたいのはどちらですか？', 'answer': '温泉旅行'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(reuse)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)

    def test_max_answers_require_result(self):
        history = [{'question': '質問', 'answer': '回答'}] * MAX_QUESTIONS
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(self.question))):
            self.assertEqual(self.http.post('/send_api', json={'consultation': 'PC', 'history': history}).status_code, 502)
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(self.result))):
            response = self.http.post('/send_api', json={'consultation': 'PC', 'history': history})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), self.result)

    def test_confidence_100_question_triggers_result(self):
        ready = {**self.question, 'question': '予算はどれくらいですか？', 'confidence': 100}
        history = [{'question': f'質問{i}', 'answer': '回答'} for i in range(MIN_QUESTIONS_FOR_RESULT)]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(ready)), self.completion(json.dumps(self.result))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json(), self.result)
        self.assertEqual(create.call_count, 2)
        self.assertIn('confidenceが100', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_result_before_min_answers_is_rejected_and_kept_questioning(self):
        history = [{'question': '質問1', 'answer': '回答1'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(self.result)), self.completion(json.dumps(self.question))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['question'], self.question['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('最終結果', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_result_on_last_attempt_still_rejected_before_min_answers(self):
        history = [{'question': '質問1', 'answer': '回答1'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(self.result)), self.completion(json.dumps(self.result)),
            self.completion(json.dumps(self.result))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['status'], 'question')
        self.assertIn(payload['question'], [q['question'] for q in FALLBACK_QUESTIONS])
        self.assertEqual(payload['confidence'], 30)
        self.assertEqual(create.call_count, 3)

    def test_fallback_question_avoids_already_asked_questions(self):
        history = [{'question': '予算はいくらまで使えますか？', 'answer': '50000円'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(self.result)), self.completion(json.dumps(self.result)),
            self.completion(json.dumps(self.result))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        payload = response.get_json()
        self.assertNotEqual(payload['question'], '予算はいくらまで使えますか？')
        self.assertEqual(create.call_count, 3)

    def test_early_confidence_100_question_is_returned_with_capped_confidence(self):
        ready = {**self.question, 'question': '予算はどれくらいですか？', 'confidence': 100}
        history = [{'question': '質問1', 'answer': '回答1'}]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(ready))) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['confidence'], 50)
        self.assertEqual(create.call_count, 1)

    def test_force_result_skips_questions(self):
        history = [{'question': '移動時間は？', 'answer': '半日'}]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(self.result))) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history, 'force_result': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['recommendation'], 'ノートPC')
        self.assertIn('質問は終了です', create.call_args.kwargs['messages'][0]['content'])

    def test_force_result_question_returns_error(self):
        history = [{'question': '移動時間は？', 'answer': '半日'}]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(self.question))):
            response = self.http.post('/send_api', json={'consultation': 'PC', 'history': history, 'force_result': True})
        self.assertEqual(response.status_code, 502)


if __name__ == '__main__':
    unittest.main()
