import unittest
import numpy as np
from sklearn.metrics import f1_score
from main import model, X_scaled, df, DATA_SIZE

class TestISF(unittest.TestCase):
    def test_f1_score_calculation(self):
        true_labels = np.concatenate([
            np.zeros(int(DATA_SIZE * 0.97)),
            np.ones(int(DATA_SIZE * 0.03))
        ]).astype(int)

        predictions = model.predict(X_scaled)
        predictions = (predictions == -1).astype(int)

        f1 = f1_score(true_labels, predictions)
        self.assertGreater(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
    
    def test_anomaly_column_exists(self):
        """Test that anomaly column was created"""
        self.assertIn('anomaly', df.columns)
    
    def test_predictions_binary(self):
        """Test that predictions are binary"""
        self.assertTrue(all(df['anomaly'].isin([0, 1])))

if __name__ == '__main__':
    unittest.main()