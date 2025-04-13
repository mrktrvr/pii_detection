import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from utils.data_generator import generate_dataset


class TestDataGenerator(unittest.TestCase):

    def test_dataset_length(self):
        data = generate_dataset(10)
        self.assertEqual(len(data), 10)

    def test_flag_distribution(self):
        n_samples = 100
        data = generate_dataset(n_samples, pii_ratio=0.25)
        flags = [d['flag'] for d in data]
        ratio = sum(flags) / len(flags)
        self.assertEqual(len(data), n_samples)
        self.assertTrue(0.2 < ratio)
        self.assertTrue(ratio < 0.3)

    def test_data_structure(self):
        data = generate_dataset(1)
        sample = data[0]
        self.assertIn('dataset_id', sample)
        self.assertIn('id', sample)
        self.assertIn('value', sample)
        self.assertIn('flag', sample)
        self.assertIsInstance(sample['value'], str)


if __name__ == '__main__':
    unittest.main()
