import unittest
import pandas as pd
from preprocessing import load_raw_data, preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_load_raw_data(self):
        user_surveys, user_interviews, user_logs = load_raw_data('data/')
        self.assertIsInstance(user_surveys, pd.DataFrame)
        self.assertIsInstance(user_interviews, pd.DataFrame)
        self.assertIsInstance(user_logs, pd.DataFrame)

    def test_preprocess_data(self):
        user_surveys = pd.DataFrame({'q1': [1, 2, 3], 'q2': [4, 5, 6]})
        user_interviews = pd.DataFrame({'transcript': ['interview 1', 'interview 2', 'interview 3']})
        user_logs = pd.DataFrame({'action': ['click', 'scroll', 'submit']})
        preprocessed_data = preprocess_data(user_surveys, user_interviews, user_logs)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        self.assertGreater(len(preprocessed_data.columns), 0)

if __name__ == '__main__':
    unittest.main()
