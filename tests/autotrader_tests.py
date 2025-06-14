import unittest
from unittest.mock import patch
import autotrader
import json

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
        with self.assertRaises(Exception):
            autotrader.fetch_market_data()


    def test_predict_optimal_trades_buy(self):
        data = {'markets': [{'instrument': 'BTC'}, {'instrument': 'ETH'}]}
        prediction = autotrader.predict_optimal_trades(data)
        self.assertEqual(prediction, {"signal": "BUY"})

    def test_predict_optimal_trades_sell(self):
        data = {'markets': [{'instrument': 'BTC'}, {'instrument': 'ETH'}, {'instrument': 'LTC'}, {'instrument': 'XRP'}, {'instrument': 'ADA'}]}
        prediction = autotrader.predict_optimal_trades(data)
        self.assertEqual(prediction, {"signal": "SELL"})

    def test_predict_optimal_trades_invalid_input(self):
        with self.assertRaises(ValueError):
            autotrader.predict_optimal_trades(None)

if __name__ == '__main__':
    unittest.main()
