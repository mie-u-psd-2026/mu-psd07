import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import (app, client, OLLAMA_MODEL, SYSTEM_PROMPT, MAX_QUESTIONS,
                         MIN_QUESTIONS_FOR_RESULT, FALLBACK_QUESTIONS, is_vague_recommendation,
                         is_conditional_recommendation, is_filler_question, is_malformed_question,
                         reuses_last_answer, asks_direct_choice, asks_user_to_decide, numeric_unit_for,
                         has_unrelated_choice, is_repeated_question, has_time_topic, condition_repeated,
                         result_gate_issue, result_contradicts_history, result_reason_is_grounded,
                         result_addresses_all_candidates)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.http = app.test_client()
        self.question = {
            'status': 'question', 'question': '持ち運ぶ頻度はどのくらいですか？',
            'answer_type': 'choice', 'options': ['毎日', '週に数回', 'ほとんどない'],
            'condition': '頻度', 'confidence': 30,
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

    def test_has_unrelated_choice(self):
        self.assertTrue(has_unrelated_choice(
            '普段、お仕事や勉強をする場所はいかがなところですか？',
            ['自宅の机や部屋', 'カフェやコワーキングスペース', 'パソコンを搭載したPC'],
        ))
        self.assertFalse(has_unrelated_choice(
            '普段、お仕事や勉強をする場所はいかがなところですか？',
            ['自宅の机や部屋', 'カフェやコワーキングスペース', '図書館'],
        ))
        self.assertFalse(has_unrelated_choice(
            '必要なPCの性能はどのレベルですか？',
            ['パソコンを搭載したPC', 'ノートPC', 'デスクトップPC'],
        ))

    def test_unrelated_choice_question_is_rejected_and_retried(self):
        bad = {**self.question, 'question': '普段、お仕事や勉強をする場所はいかがなところですか？',
               'options': ['自宅の机や部屋', 'カフェやコワーキングスペース', 'パソコンを搭載したPC']}
        good = {**self.question, 'question': '移動にはどのくらい時間を使えますか？'}
        history = [{'question': 'PCの利用目的は？', 'answer': 'オンライン授業'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(bad)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '大学用PCをノートかデスクトップか迷っています', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('却下しました', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_prompt_requires_explicit_target_for_number_questions(self):
        self.assertIn('何に対する値を聞くのかを質問文に必ず明記', SYSTEM_PROMPT)
        self.assertIn('通勤にかかる時間はどのくらいですか？', SYSTEM_PROMPT)
        self.assertIn('対象（通勤・移動・仕事など）が不明で答えられない曖昧な質問は禁止', SYSTEM_PROMPT)

    def test_numeric_unit_for(self):
        self.assertEqual(numeric_unit_for('普段、通勤時間はどのくらいかかりますか？'), '時間')
        self.assertEqual(numeric_unit_for('移動にはどのくらい時間を使えますか？'), '時間')
        self.assertEqual(numeric_unit_for('PCに使える予算はいくらですか？'), '円')
        self.assertEqual(numeric_unit_for('カーシェアに月いくらまで使えますか？'), '円')
        self.assertEqual(numeric_unit_for('会社までの距離はどのくらいですか？'), 'km')
        self.assertEqual(numeric_unit_for('休暇は何日間取れますか？'), '日')
        self.assertEqual(numeric_unit_for('利用頻度はどのくらいですか？'), None)
        self.assertEqual(numeric_unit_for('気になっている点があれば教えてください。'), None)


    def test_text_time_question_is_converted_to_number(self):
        text_question = {**self.question, 'question': '普段、通勤時間はどのくらいかかりますか？', 'answer_type': 'text'}
        history = [{'question': 'カークラブの利用頻度は？', 'answer': '毎日'}]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(text_question))):
            response = self.http.post('/send_api', json={'consultation': '車を買うかカーシェアか', 'history': history})
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload['status'], 'question')
        self.assertEqual(payload['answer_type'], 'number')
        self.assertEqual(payload['unit'], '時間')
        self.assertEqual(payload['question'], text_question['question'])


    def test_text_input_question_stays_text(self):
        textual = {**self.question, 'question': '気になっている点があれば教えてください。', 'answer_type': 'text'}
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(textual))):
            response = self.http.post('/send_api', json={'consultation': '転職するか今の仕事を続けるか', 'history': [{'question': '今の仕事の満足度は？', 'answer': '普通'}]})
        payload = response.get_json()
        self.assertEqual(payload['answer_type'], 'text')


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

    def test_condition_repeated_helper(self):
        history = [{'question': '通学に何分かかりますか？', 'answer': '30分', 'condition': '通学時間'}]
        self.assertTrue(condition_repeated('通学時間', history))
        self.assertTrue(condition_repeated('通学時間です', history))
        self.assertFalse(condition_repeated('疲労', history))
        self.assertFalse(condition_repeated('予算', [{'question': 'q', 'answer': 'a', 'condition': '通学時間'}]))
        self.assertFalse(condition_repeated('', history))
        self.assertFalse(condition_repeated('疲労', [{'question': 'q', 'answer': 'a'}]))

    def test_same_condition_question_is_rejected_and_retried(self):
        dup = {**self.question, 'question': '通学の負担を感じることはありますか？', 'condition': '通学時間',
               'options': ['感じやすい', 'ふつう', 'ほとんど感じない']}
        good = {**self.question, 'question': '今の疲れ具合はいかがですか？', 'condition': '疲労',
                'options': ['元気', '少し疲れている', 'かなり疲れている']}
        history = [{'question': '通学に何分かかりますか？', 'answer': '30分', 'condition': '通学時間'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(dup)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '通学かリモートか迷っています', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(response.get_json()['condition'], '疲労')
        self.assertEqual(create.call_count, 2)
        self.assertIn('同じ判断条件の繰り返し', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_different_condition_question_is_accepted(self):
        question = {**self.question, 'question': '通学の負担を感じることはありますか？', 'condition': '通学の負担',
                    'options': ['感じやすい', 'ふつう', 'ほとんど感じない']}
        history = [{'question': '通学に何分かかりますか？', 'answer': '30分', 'condition': '通学時間'}]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(question))) as create:
            response = self.http.post('/send_api', json={'consultation': '通学かリモートか迷っています', 'history': history})
        self.assertEqual(response.get_json()['question'], question['question'])
        self.assertEqual(response.get_json()['condition'], '通学の負担')
        self.assertEqual(create.call_count, 1)

    def test_text_dedup_still_applies_when_condition_labels_differ(self):
        dup = {**self.question, 'question': '移動に使える時間、1日にどれくらい確保できますか？', 'condition': '移動時間',
               'options': ['1時間', '2時間', '3時間']}
        good = {**self.question, 'question': '同行者は誰と行きますか？', 'condition': '同行者',
                'options': ['ひとり', '家族', '友人']}
        history = [{'question': '1日の所要時間はどのくらいですか？', 'answer': '6時間', 'condition': '通学時間'}]
        self.assertFalse(condition_repeated('移動時間', history))
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(dup)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '通学かリモートか迷っています', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('同じ判断条件の繰り返し', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_time_topic_question_is_repeated_with_paraphrase(self):
        q6 = '1日の所要時間は、通勤時間とリモートワークの時間を合わせて、平均して何時間ほど確保できますか？'
        q7 = 'リモートワークと通勤を合わせた1日の所要時間は、平均して何時間ほどですか？'
        self.assertTrue(has_time_topic(q6))
        self.assertTrue(has_time_topic(q7))
        self.assertTrue(is_repeated_question(q7, [{'question': q6, 'answer': '7時間'}]))
        self.assertTrue(is_repeated_question(q6, [{'question': q7, 'answer': '7時間'}]))

    def test_unrelated_time_questions_are_not_repeated(self):
        self.assertFalse(is_repeated_question(
            '通勤にかかるお金はどのくらいですか？',
            [{'question': '通勤時間はどのくらいですか？', 'answer': '60分'}],
        ))
        self.assertFalse(is_repeated_question(
            '通勤時間はどのくらいですか？',
            [{'question': '移動費の予算はいくらですか？', 'answer': '1万円'}],
        ))

    def test_time_topic_duplicate_is_rejected_and_retried(self):
        q5 = '移動に使える時間、1日にどれくらい確保できますか？'
        q3 = '1日の所要時間はどのくらいですか？'
        dup = {**self.question, 'question': q5}
        good = {**self.question, 'question': '現在の疲れ具合はいかがですか？',
                'options': ['元気', '少し疲れている', 'かなり疲れている']}
        history = [
            {'question': '通勤時間はどのくらいですか？', 'answer': '60分'},
            {'question': q3, 'answer': '6時間'},
        ]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(dup)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': 'リモートワークか通勤か迷っています', 'history': history})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('却下しました', create.call_args_list[1].kwargs['messages'][0]['content'])

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
        for good in ('旅行に使える日数はどのくらいですか？', '予算はいくらまで使えますか？', '休日は何日間ですか？'):
            self.assertFalse(is_filler_question(good), good)

    def test_malformed_question_helper(self):
        for bad in (
            '温泉旅行と街歩き、どちらがよりリラックスできますか？ 選択肢：1. 温泉に入りながら景色を眺める 2. 街を散策しながらカフェで休憩する 3. わからない・決められない',
            '夜ご飯はどうしますか？\n選択肢：\n1. 温泉ですし\n2. 街のカフェ\n3. その他',
            'お昼は何にしますか？1.定食 2.ラーメン',
        ):
            self.assertTrue(is_malformed_question(bad), bad)
        for good in ('今の疲れ具合はいかがですか？', '予算はいくらまで使えますか？', '旅行に使える日数はどのくらいですか？'):
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
            '旅行に使える日数はどのくらいですか？',
            '予算はいくらまで使えますか？',
            '温泉旅行に同行者はいますか？',
        ):
            self.assertFalse(asks_direct_choice(good, consultation), good)

    def test_asks_user_to_decide_helper(self):
        consultation = '車を買うかカーシェアを利用するか迷っています'
        for bad in (
            '車を購入する方が良いと思いますか？',
            '車を購入する方にしますか？',
            '車を購入する方がいいですか？',
            'おすすめはどちらですか？',
            '車を購入するかカーシェアにするか、どちらがいいと思いますか？',
        ):
            self.assertTrue(asks_user_to_decide(bad, consultation), bad)
        for good in (
            '車にかける予算はいくらまで使えますか？',
            '車の利用頻度はどのくらいですか？',
            '家族は何人ですか？',
        ):
            self.assertFalse(asks_user_to_decide(good, consultation), good)

    def test_direct_choice_question_is_rejected_and_retried(self):
        direct = {**self.question, 'question': '温泉旅行と街歩き、どちらの休日に向いていますか？'}
        good = {**self.question, 'question': '旅行に使える日数はどのくらいですか？'}
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(direct)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行に行くか街歩きに行くか迷っています', 'history': []})
        self.assertEqual(response.get_json()['question'], good['question'])
        self.assertEqual(create.call_count, 2)
        self.assertIn('直接比較', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_malformed_question_is_rejected_and_retried(self):
        malformed = {**self.question, 'question': 'どちらで過ごしたいですか？ 選択肢：1. 温泉 2. 街歩き'}
        good = {**self.question, 'question': '旅行に使える日数はどのくらいですか？'}
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
        good = {**self.question, 'question': '旅行に使える日数はどのくらいですか？'}
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

    def test_result_contradicting_day_answer_is_rejected_and_retried(self):
        bad = {'status': 'result', 'recommendation': '街歩き',
               'conditions': ['日数'], 'reason': '2〜3日ですと日帰りで街歩きが適しています',
               'pros': ['気軽に楽しめる'], 'cons': ['暑さがつらい'], 'alternative': '日帰り以外なら温泉旅行', 'confidence': 80}
        good = {'status': 'result', 'recommendation': '温泉旅行',
                'conditions': ['日数', '予算'], 'reason': '2〜3日の日程と50000円の予算なので温泉旅行が合います',
                'pros': ['ゆっくり休める'], 'cons': ['移動に時間がかかる'], 'alternative': '時間のない日は街歩きが向きます', 'confidence': 90}
        history = [
            {'question': '旅行の日数はどのくらいですか？', 'answer': '2〜3日'},
            {'question': '予算はいくらですか？', 'answer': '50000円'},
            {'question': '同行者はいますか？', 'answer': '家族'},
        ]
        self.assertTrue(result_contradicts_history(bad, history))
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(bad)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['recommendation'], '温泉旅行')
        self.assertEqual(create.call_count, 2)
        self.assertIn('矛盾', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_result_without_candidate_comparison_is_rejected_and_retried(self):
        bad = {'status': 'result', 'recommendation': '温泉旅行',
               'conditions': ['重視点'], 'reason': '食べ歩きを重視しているので温泉旅行です',
               'pros': ['食事が楽しめる'], 'cons': ['移動が多い'], 'alternative': '別の条件なら別の選択肢', 'confidence': 70}
        good = {**bad, 'recommendation': '街歩き',
                'reason': '食べ歩きを重視しているので街歩きを選びます',
                'alternative': 'ゆっくり休みたいなら温泉旅行が向きます'}
        history = [
            {'question': '旅の重視ポイントは？', 'answer': '食べ歩きを重視'},
            {'question': '予算は？', 'answer': '3000円'},
            {'question': '同行者は？', 'answer': '友人'},
        ]
        self.assertFalse(result_addresses_all_candidates(bad, '温泉旅行か街歩き'))
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(bad)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['recommendation'], '街歩き')
        self.assertEqual(create.call_count, 2)
        self.assertIn('比較', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_result_before_min_answers_keeps_questioning(self):
        history = [{'question': '旅の重視ポイントは？', 'answer': '食べ歩きを重視'}]
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(self.result)), self.completion(json.dumps(self.question))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        payload = response.get_json()
        self.assertEqual(payload['status'], 'question')
        self.assertEqual(create.call_count, 2)
        self.assertIn('情報が不足', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_sufficient_conditions_return_result_without_extra_questions(self):
        result = {'status': 'result', 'recommendation': '温泉旅行',
                  'conditions': ['重視点', '日数'], 'reason': 'ゆっくり休みたいので温泉旅行が合います',
                  'pros': ['くつろげる'], 'cons': ['移動が必要'], 'alternative': '活動的に遊ぶなら街歩きが向きます', 'confidence': 90}
        history = [
            {'question': '旅の重視ポイントは？', 'answer': 'ゆっくり休みたい'},
            {'question': '日数は？', 'answer': '2泊3日'},
            {'question': '疲れ具合は？', 'answer': 'かなり疲れている'},
        ]
        with patch.object(client.chat.completions, 'create', return_value=self.completion(json.dumps(result))) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['recommendation'], '温泉旅行')
        self.assertEqual(create.call_count, 1)

    def test_result_with_unsubstantiated_reason_is_rejected_and_retried(self):
        bad = {'status': 'result', 'recommendation': '温泉旅行',
               'conditions': ['予算'], 'reason': '温泉でリラックスできるからです',
               'pros': ['癒される'], 'cons': ['遠い'], 'alternative': '近場なら街歩きが向きます', 'confidence': 80}
        good = {**bad, 'reason': '50000円の予算と家族でゆっくりできるので温泉旅行が合います'}
        history = [
            {'question': '予算は？', 'answer': '50000円'},
            {'question': '同行者は？', 'answer': '家族'},
            {'question': '重視点は？', 'answer': 'ゆっくり休む'},
        ]
        self.assertFalse(result_reason_is_grounded(bad, history))
        self.assertTrue(result_reason_is_grounded(good, history))
        with patch.object(client.chat.completions, 'create', side_effect=[
            self.completion(json.dumps(bad)), self.completion(json.dumps(good))
        ]) as create:
            response = self.http.post('/send_api', json={'consultation': '温泉旅行か街歩き', 'history': history})
        self.assertEqual(response.get_json()['recommendation'], '温泉旅行')
        self.assertEqual(create.call_count, 2)
        self.assertIn('根拠', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_result_gate_issue_helper(self):
        result = {'status': 'result', 'recommendation': 'ノートPC', 'conditions': ['移動'], 'reason': '持ち運ぶのでノートPC',
                  'pros': ['軽い'], 'cons': ['性能が低い'], 'alternative': '家で使うならデスクトップ', 'confidence': 80}
        history = [{'question': '質問', 'answer': '持ち運ぶ'}]
        self.assertEqual(result_gate_issue(result, 'ノートPCかデスクトップ', history), '')

    def test_prompt_requires_contradiction_free_result(self):
        self.assertIn('矛盾', SYSTEM_PROMPT)

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
