import unittest
from unittest.mock import patch
from backend.app import (app, client, MAX_QUESTIONS, load_skipped_questions,
                         save_skipped_questions, learn_skipped_question)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.http = app.test_client()

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

    def test_browser_deadline_follows_server_configuration(self):
        with patch('backend.app.DECISION_TIMEOUT', 240):
            response = self.http.get('/runtime-config.js')
        self.assertEqual(response.status_code, 200)
        self.assertIn('250000', response.get_data(as_text=True))
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(response.mimetype, 'application/javascript')

    def test_skip_log_roundtrip(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'skip_log.json'
            with patch('backend.app.SKIP_LOG_PATH', log):
                self.assertEqual(load_skipped_questions(), [])
                learn_skipped_question('犬を飼う場合、ペットフードの費用はいくらですか？')
                learn_skipped_question('犬を飼う場合、ペットフードの費用はいくらですか？')
                self.assertEqual(load_skipped_questions(), ['犬を飼う場合、ペットフードの費用はいくらですか？'])
                save_skipped_questions([])
                self.assertEqual(load_skipped_questions(), [])

    def test_skip_question_bad_type_returns_400(self):
        with patch.object(client.chat.completions, 'create') as create:
            response = self.http.post('/send_api', json={'consultation': '犬か猫', 'skip_question': 123})
        self.assertEqual(response.status_code, 400)
        create.assert_not_called()
