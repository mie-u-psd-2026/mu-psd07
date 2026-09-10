import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.evaluate import (ConversationGrade, evaluate_scenario, write_report,
                               answer_question, SimulatedAnswer)


class EvaluationTests(unittest.TestCase):
    def test_unknown_mode_does_not_invent_profile_answers(self):
        client = Mock()
        self.assertEqual(answer_question(client, 'model', {'answer_mode': 'unknown'}, {}),
                         'わからない・決められない')
        client.chat.completions.create.assert_not_called()

    def test_simulated_choice_must_exist_on_screen(self):
        question = {'answer_type': 'choice', 'options': ['毎日', '週に数回']}
        with patch('backend.evaluate.structured_call', return_value=SimulatedAnswer(answer='架空の回答')):
            with self.assertRaises(ValueError):
                answer_question(Mock(), 'model', {'answer_mode': 'profile', 'profile': '人物設定'}, question)

    def test_simulator_schema_contains_only_displayed_choices_and_unknown(self):
        question = {'answer_type': 'choice', 'options': ['毎日', '週に数回']}
        with patch('backend.evaluate.structured_call', return_value=SimulatedAnswer(answer='毎日')) as call:
            answer_question(Mock(), 'model', {'answer_mode': 'profile', 'profile': '毎日'}, question)
        schema = call.call_args.args[2].model_json_schema()
        self.assertEqual(schema['properties']['answer']['enum'],
                         ['毎日', '週に数回', 'わからない・決められない'])

    def test_no_result_cannot_pass_and_does_not_call_judge(self):
        grade = ConversationGrade(usefulness=5, answerability=5, non_repetition=5,
                                   grounding=5, choice_priority=5, evidence=['根拠'], concerns=[])
        scenario = {'id': 'test', 'split': 'heldout', 'consultation': '相談', 'answer_mode': 'unknown'}
        with patch('backend.evaluate.decide', side_effect=TimeoutError()), \
                patch('backend.evaluate.structured_call', return_value=grade) as judge:
            record = evaluate_scenario(Mock(), 'model', 'judge', scenario, 3)
        judge.assert_not_called()
        self.assertFalse(record['passed'])
        self.assertTrue(record['errors'])
        with tempfile.TemporaryDirectory() as tmp:
            report = {'model': 'model', 'judge_model': 'judge', 'scenarios': [record]}
            target = Path(tmp) / 'report.json'
            write_report(target, report)
            self.assertIn('要確認', target.with_suffix('.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
