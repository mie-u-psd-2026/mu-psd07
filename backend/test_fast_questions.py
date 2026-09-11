import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.decision import decide, ChoiceQuestion
from backend.fast_questions import template_step


class FastQuestionTests(unittest.TestCase):
    def setUp(self):
        self.flows = json.loads(Path(__file__).with_name('template_flows.json').read_text(encoding='utf-8'))

    def test_each_example_reaches_result_generation_without_question_model_calls(self):
        for flow in self.flows:
            with self.subTest(flow=flow['id']):
                client = Mock()
                history = []
                consultation = flow['consultations'][0]
                for item in flow['questions']:
                    response = decide(client, 'test-model', consultation, history)
                    ChoiceQuestion.model_validate(response)
                    self.assertEqual(response['condition'], item['condition'])
                    self.assertEqual(len(response['options']), len(set(response['options'])))
                    history.append(dict(question=response['question'], condition=response['condition'],
                                        answer=response['options'][0]))
                client.chat.completions.create.assert_not_called()
                with patch('backend.decision.practical_decide', return_value={'status': 'result'}) as generate:
                    self.assertEqual(decide(client, 'test-model', consultation, history)['status'], 'result')
                    self.assertTrue(generate.call_args.args[4])

    def test_skips_unknown_answers_and_extra_constraints(self):
        for flow in self.flows:
            consultation = flow['consultations'][0]
            first, second = flow['questions'][:2]
            history = [dict(question=first['question'], condition=first['condition'], answer='わからない')]
            question, complete = template_step(consultation, history, [second['question']])
            self.assertEqual(question['condition'], flow['questions'][2]['condition'])
            self.assertFalse(complete)
            self.assertEqual(template_step(consultation, [], [q['question'] for q in flow['questions']]), (None, True))
            self.assertEqual(template_step(consultation + '予算は月1万円です。', [], []), (None, False))

    def test_explicit_early_finish_bypasses_templates(self):
        with patch('backend.decision.practical_decide', return_value={'status': 'result'}) as generate:
            decide(Mock(), 'test-model', self.flows[0]['consultations'][0], [], force_result=True)
            self.assertTrue(generate.call_args.args[4])
