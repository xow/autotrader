# -*- coding: utf-8 -*-
"""
Integration tests for BTCMarkets API interactions.
"""
import pytest
import requests
import time
from unittest.mock import Mock, patch, MagicMock
import json

# Import the autotrader module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader


class TestAPIIntegration:
    """Test cases for BTCMarkets API integration."""
    
    def test_api_endpoint_format(self, isolated_trader):
        """Test that API endpoint is correctly formatted."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = []
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            isolated_trader.fetch_market_data()
            
            # Verify the URL was called correctly
            expected_url = "https://api.btcmarkets.net/v3/markets/tickers?marketId=BTC-AUD"
            mock_get.assert_called_once_with(expected_url, timeout=10)
    
    def test_api_response_handling_success(self, isolated_trader):
        """Test successful API response handling."""
        mock_response_data = [
            {
                "marketId": "BTC-AUD",
                "lastPrice": "50000.00",
                "volume24h": "150.75",
                "bestBid": "49995.00",
                "bestAsk": "50005.00",
                "high24h": "51000.00",
                "low24h": "49000.00"
            }
        ]
        
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = isolated_trader.fetch_market_data()
            
            assert result == mock_response_data
            assert len(result) == 1
            assert result[0]["marketId"] == "BTC-AUD"
    
    def test_api_timeout_handling(self, isolated_trader):
        """Test API timeout handling."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
            
            result = isolated_trader.fetch_market_data()
            
            assert result is None
            # Should retry 3 times
            assert mock_get.call_count == 3
    
    def test_api_connection_error_handling(self, isolated_trader):
        """Test API connection error handling."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            result = isolated_trader.fetch_market_data()
            
            assert result is None
            assert mock_get.call_count == 3  # Should retry 3 times
    
    def test_api_http_error_handling(self, isolated_trader):
        """Test API HTTP error handling."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            result = isolated_trader.fetch_market_data()
            
            assert result is None
            assert mock_get.call_count == 3  # Should retry 3 times
    
    def test_api_invalid_json_handling(self, isolated_trader):
        """Test handling of invalid JSON responses."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = isolated_trader.fetch_market_data()
            
            assert result is None
    
    def test_api_exponential_backoff(self, isolated_trader):
        """Test exponential backoff between retries."""
        with patch("requests.get") as mock_get, \
             patch("time.sleep") as mock_sleep:
            
            mock_get.side_effect = requests.exceptions.RequestException("API Error")
            
            isolated_trader.fetch_market_data()
            
            # Should have called sleep with exponential backoff
            expected_sleep_calls = [
                ((1,),),  # 2^0 = 1
                ((2,),)   # 2^1 = 2
            ]
            assert mock_sleep.call_args_list == expected_sleep_calls
    
    def test_data_extraction_btc_aud_market(self, isolated_trader):
        """Test extraction of BTC-AUD market data from API response."""
        api_response = [
            {
                "marketId": "ETH-AUD",
                "lastPrice": "3000.00",
                "volume24h": "100.00"
            },
            {
                "marketId": "BTC-AUD",
                "lastPrice": "45000.50",
                "volume24h": "123.45",
                "bestBid": "44995.00",
                "bestAsk": "45005.00",
                "high24h": "46000.00",
                "low24h": "44000.00"
            }
        ]
        
        extracted_data = isolated_trader.extract_comprehensive_data(api_response)
        
        assert extracted_data is not None
        assert extracted_data["price"] == 45000.50
        assert extracted_data["volume"] == 123.45
        assert extracted_data["bid"] == 44995.00
        assert extracted_data["ask"] == 45005.00
        assert extracted_data["high24h"] == 46000.00
        assert extracted_data["low24h"] == 44000.00
    
    def test_data_extraction_missing_btc_aud(self, isolated_trader):
        """Test extraction when BTC-AUD market is not in response."""
        api_response = [
            {
                "marketId": "ETH-AUD",
                "lastPrice": "3000.00",
                "volume24h": "100.00"
            }
        ]
        
        extracted_data = isolated_trader.extract_comprehensive_data(api_response)
        
        assert extracted_data is None
    
    def test_data_extraction_invalid_price_format(self, isolated_trader):
        """Test extraction with invalid price format."""
        api_response = [
            {
                "marketId": "BTC-AUD",
                "lastPrice": "invalid_price",
                "volume24h": "123.45"
            }
        ]
        
        extracted_data = isolated_trader.extract_comprehensive_data(api_response)
        
        assert extracted_data is None
    
    def test_data_extraction_missing_fields(self, isolated_trader):
        """Test extraction with missing required fields."""
        api_response = [
            {
                "marketId": "BTC-AUD",
                "lastPrice": "45000.50"
                # Missing other required fields
            }
        ]
        
        extracted_data = isolated_trader.extract_comprehensive_data(api_response)
        
        # Should still work with default values
        assert extracted_data is not None
        assert extracted_data["price"] == 45000.50
        assert extracted_data["volume"] == 0.0  # Default value
    
    def test_rate_limiting_simulation(self, isolated_trader):
        """Test behavior under rate limiting conditions."""
        call_count = 0
        
        def rate_limited_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Simulate rate limiting
                response = Mock()
                response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")
                return response
            else:
                # Successful response after rate limiting
                response = Mock()
                response.json.return_value = [{"marketId": "BTC-AUD", "lastPrice": "45000"}]
                response.raise_for_status.return_value = None
                return response
        
        with patch("requests.get", side_effect=rate_limited_response):
            result = isolated_trader.fetch_market_data()
            
            assert result is None  # Should fail after 3 attempts
            assert call_count == 3
    
    def test_api_data_validation(self, isolated_trader):
        """Test validation of API data before processing."""
        # Test with various edge cases
        test_cases = [
            # Empty response
            [],
            # Null response
            None,
            # Invalid structure
            [{"invalid": "structure"}],
            # Valid but zero price
            [{"marketId": "BTC-AUD", "lastPrice": "0.00", "volume24h": "100"}],
            # Negative price (should be handled)
            [{"marketId": "BTC-AUD", "lastPrice": "-100.00", "volume24h": "100"}]
        ]
        
        for test_data in test_cases:
            extracted = isolated_trader.extract_comprehensive_data(test_data)
            
            if test_data == [] or test_data is None or (test_data and "lastPrice" not in test_data[0]):
                assert extracted is None
    
    def test_concurrent_api_calls_handling(self, isolated_trader):
        """Test handling of concurrent API calls."""
        import threading
        import time
        
        results = []
        
        def make_api_call():
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.json.return_value = [{"marketId": "BTC-AUD", "lastPrice": "45000"}]
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                result = isolated_trader.fetch_market_data()
                results.append(result)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_api_call)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All calls should succeed
        assert len(results) == 3
        assert all(result is not None for result in results)
    
    def test_api_response_caching_behavior(self, isolated_trader):
        """Test that API responses are not inappropriately cached."""
        call_responses = [
            [{"marketId": "BTC-AUD", "lastPrice": "45000"}],
            [{"marketId": "BTC-AUD", "lastPrice": "45100"}]
        ]
        
        with patch("requests.get") as mock_get:
            # Setup different responses for consecutive calls
            mock_responses = []
            for response_data in call_responses:
                mock_response = Mock()
                mock_response.json.return_value = response_data
                mock_response.raise_for_status.return_value = None
                mock_responses.append(mock_response)
            
            mock_get.side_effect = mock_responses
            
            # Make two consecutive calls
            result1 = isolated_trader.fetch_market_data()
            result2 = isolated_trader.fetch_market_data()
            
            # Should get different results (no caching)
            assert result1[0]["lastPrice"] == "45000"
            assert result2[0]["lastPrice"] == "45100"
            assert mock_get.call_count == 2
    
    def test_api_ssl_verification(self, isolated_trader):
        """Test that SSL verification is properly handled."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = [{"marketId": "BTC-AUD", "lastPrice": "45000"}]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            isolated_trader.fetch_market_data()
            
            # Verify SSL is not explicitly disabled
            call_kwargs = mock_get.call_args[1] if mock_get.call_args else {}
            assert call_kwargs.get("verify", True) is not False
