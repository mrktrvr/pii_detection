import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)


class TestRuleBasedModel(unittest.TestCase):

    def setUp(self):
        self.src_list = [
            ['Dr. John was at the event.', True],
            ['he was at the event.', False],
            ['send to 42 Elm Street.', True],
            ['send to the office.', False],
            ['it is raining today.', False],
            ['it is raining in London today.', True],
            ['it is raining in london today.', False],
        ]

    def tearDown(self):
        del self.src_list

    def test_rule_based(self):
        from models.rule_based import detect_pii
        for src, ans in self.src_list:
            res = detect_pii(src)
            self.assertEqual(res, ans)

    def test_transformer(self):
        from models.transformer_model import TransformerPIIDetector
        detector = TransformerPIIDetector()
        for src, ans in self.src_list:
            res = detector.detect(src)
            self.assertEqual(res, ans)


if __name__ == '__main__':
    unittest.main()
