import unittest
import numpy as np
from model import build_seenai_model

class TestModel(unittest.TestCase):
    def test_build_seenai_model(self):
        input_shape = (10,)
        output_shape = 2
        model = build_seenai_model(input_shape, output_shape)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.layers[-1].units, output_shape)

    def test_model_prediction(self):
        input_shape = (10,)
        output_shape = 2
        model = build_seenai_model(input_shape, output_shape)
        X_test = np.random.rand(1, 10)
        y_pred = model.predict(X_test)
        self.assertEqual(y_pred.shape, (1, output_shape))

if __name__ == '__main__':
    unittest.main()
