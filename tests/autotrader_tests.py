import unittest
from unittest.mock import patch
import autotrader
import json
import os
import tensorflow as tf
import numpy as np
import requests

class TestAutotrader(unittest.TestCase):

    @patch('autotrader.requests.get')
    def test_fetch_market_data_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'markets': [{'instrument': 'BTC'}]}
        data = autotrader.fetch_market_data()
        self.assertEqual(data, {'markets': [{'instrument': 'BTC'}]})

    @patch('autotrader.requests.get')
    def test_fetch_market_data_failure(self, mock_get):
        mock_get.return_value.status_code = 500
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError
        with self.assertRaises(Exception):
            autotrader.fetch_market_data()

    def test_predict_optimal_trades_buy(self):
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        data = [{'marketId': 'BTC-AUD', 'lastPrice': '50000'}, {'marketId': 'ETH-AUD', 'lastPrice': '3000'}]
        prediction = autotrader.predict_optimal_trades(data, model)
        self.assertEqual(prediction["signal"], "HOLD")

    def test_predict_optimal_trades_sell(self):
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        data = [{'marketId': 'BTC-AUD', 'lastPrice': '50000'}, {'marketId': 'ETH-AUD', 'lastPrice': '3000'}, {'marketId': 'LTC-AUD', 'lastPrice': '100'}, {'marketId': 'XRP-AUD', 'lastPrice': '0.5'}, {'marketId': 'ADA-AUD', 'lastPrice': '1'}]
        prediction = autotrader.predict_optimal_trades(data, model)
        self.assertEqual(prediction["signal"], "HOLD")

    def test_predict_optimal_trades_invalid_input(self):
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        with self.assertRaises(TypeError):
            autotrader.predict_optimal_trades(None, model)

    def test_save_load_training_data(self):
        # Create sample training data
        sample_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        filename = "test_training_data.json"

        # Save the data
        autotrader.save_training_data(sample_data, filename)

        # Load the data
        loaded_data = autotrader.load_training_data(filename)

        # Assert that the loaded data matches the original data
        self.assertEqual(loaded_data, sample_data)

        # Clean up the test file
        if os.path.exists(filename):
            os.remove(filename)

    def test_save_load_model(self):
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        filename = "test_autotrader_model.keras"

        # Save the model
        autotrader.save_model(model, filename)

        # Load the model
        loaded_model = autotrader.load_model(filename)

        # Assert that the loaded model is not None
        self.assertIsNotNone(loaded_model)

        # Clean up the test file
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    unittest.main()
