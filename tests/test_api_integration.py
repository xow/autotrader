# -*- coding: utf-8 -*-
"""
Integration tests for BTCMarkets API interactions.
"""
import pytest
import requests
import time
from unittest.mock import Mock, patch, MagicMock
import json
from collections import deque # Import deque

# Import the autotrader module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autotrader import ContinuousAutoTrader # Import from top-level autotrader.py
from autotrader.utils.exceptions import APIError, DataError, NetworkError, NetworkTimeoutError


class TestAPIIntegration:
    """Test cases for BTCMarkets API integration."""
    
    def test_api_endpoint_format(self, isolated_trader):
        """Test that API endpoint is correctly formatted."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = []
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            try:
                isolated_trader.fetch_market_data()
            except Exception as e:
                print(f"API request failed: {e}")
                raise
            
            # Verify the URL and params were called correctly
            expected_url = isolated_trader.settings.api.base_url + "/markets/tickers"
            expected_params = {"marketId": "BTC-AUD"}
            mock_get.assert_called_once_with(
                expected_url,
                params=expected_params,
                timeout=isolated_trader.settings.api.timeout
            )
    
    def test_api_response_handling_success(self, isolated_trader):
        """Test successful API response handling."""
        mock_response_data = [
            {
                "marketId": "BTC-AUD",
                "lastPrice": 50000.00, # Changed to float
                "volume24h": 150.75,  # Changed to float
                "bestBid": 49995.00,  # Changed to float
                "bestAsk": 50005.00,  # Changed to float
                "high24h": 51000.00,  # Changed to float
                "low24h": 49000.00   # Changed to float
            }
        ]
        
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = isolated_trader.fetch_market_data()
            
            print(f"API result: {result}")
            assert result == mock_response_data
            assert len(result) == 1
            assert result[0]["marketId"] == "BTC-AUD"
            # Ensure lastPrice is a float, not a string
            assert isinstance(result[0]["lastPrice"], float)
    
    def test_api_timeout_handling(self, isolated_trader):
        """Test API timeout handling."""
        with patch("requests.get") as mock_get, \
             pytest.raises(NetworkTimeoutError) as exc_info:
            mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
            
            isolated_trader.fetch_market_data()
            
        # Verify the error message
        assert "Market data request timed out" in str(exc_info.value)
        # Should only try once since we're raising NetworkTimeoutError on first timeout
        assert mock_get.call_count == 1
    
    def test_api_connection_error_handling(self, isolated_trader):
        """Test API connection error handling."""
        with patch("requests.get") as mock_get, \
             pytest.raises(NetworkError) as exc_info:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            isolated_trader.fetch_market_data()
            
        # Verify the error message
        assert "Failed to connect to the exchange" in str(exc_info.value)
        # Should only try once since we're raising NetworkError on first connection error
        assert mock_get.call_count == 1
    
    def test_api_http_error_handling(self, isolated_trader):
        """Test API HTTP error handling."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.text = "Not Found"
            # Ensure the mock_response passed to HTTPError has a status_code
            mock_response_for_http_error = Mock(status_code=404, text="Not Found")
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found", response=mock_response_for_http_error)
            mock_get.return_value = mock_response
            
            with pytest.raises(APIError) as exc_info:
                isolated_trader.fetch_market_data()
            
            # Verify the error details
            assert exc_info.value.status_code == 404
            assert "404" in str(exc_info.value)
            assert mock_get.call_count == 1
    
    def test_api_invalid_json_handling(self, isolated_trader):
        """Test handling of invalid JSON responses."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.raise_for_status.return_value = None
            mock_response.text = "{invalid json}"
            mock_get.return_value = mock_response
            
            with pytest.raises(DataError) as exc_info:
                isolated_trader.fetch_market_data()
            
            # Verify the error details
            assert "Invalid JSON response" in str(exc_info.value)
            assert exc_info.value.data_type == "market_data"
            assert "json_decode_error" in str(exc_info.value.context["validation_errors"]) # Access via context
            assert mock_get.call_count == 1
    
    def test_api_exponential_backoff(self, isolated_trader):
        """Test exponential backoff between retries."""
        # This test is designed for a retry mechanism *outside* fetch_market_data.
        # Since fetch_market_data now raises immediately, this test needs to be adapted.
        # We'll simulate multiple failures and ensure the exception is raised.
        
        # Create a list of side effects: two HTTP errors, then a successful response
        mock_responses = []
        # First mock response: HTTP 500 error
        mock_response_500_1 = Mock(status_code=500, text="Internal Server Error") # Set status_code directly
        mock_response_500_1.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error", response=mock_response_500_1)
        mock_responses.append(mock_response_500_1)

        # Second mock response: HTTP 500 error
        mock_response_500_2 = Mock(status_code=500, text="Internal Server Error") # Set status_code directly
        mock_response_500_2.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Error", response=mock_response_500_2)
        mock_responses.append(mock_response_500_2)
        
        # Third mock response: successful
        mock_response_success = Mock()
        mock_response_success.json.return_value = [{"marketId": "BTC-AUD", "lastPrice": 45000.0}]
        mock_response_success.raise_for_status.return_value = None
        mock_responses.append(mock_response_success)
        
        with patch("requests.get", side_effect=mock_responses) as mock_get:
            # The first call should raise an APIError immediately
            with pytest.raises(APIError) as exc_info:
                isolated_trader.fetch_market_data()
            
            assert exc_info.value.status_code == 500
            assert mock_get.call_count == 1
            
            # To test exponential backoff, we would need to simulate the retry loop.
            # For now, we confirm the immediate exception on the first call.
    
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
                response = Mock(status_code=429, text="Too Many Requests") # Set status_code directly
                response.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests", response=response)
                return response
            else:
                # Successful response after rate limiting
                response = Mock()
                response.json.return_value = [{"marketId": "BTC-AUD", "lastPrice": 45000.0}] # Return float
                response.raise_for_status.return_value = None
                return response
        
        with patch("requests.get", side_effect=rate_limited_response) as mock_get:
            # The rate limiting simulation should now raise an APIError immediately
            with pytest.raises(APIError) as exc_info:
                isolated_trader.fetch_market_data()
            
            assert exc_info.value.status_code == 429
            assert mock_get.call_count == 1 # Should be called once before raising
    
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
            
            # The extract_comprehensive_data method now returns None for invalid data
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
            [{"marketId": "BTC-AUD", "lastPrice": 45000.0}], # Use float
            [{"marketId": "BTC-AUD", "lastPrice": 45100.0}]  # Use float
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
            assert result1[0]["lastPrice"] == 45000.0
            assert result2[0]["lastPrice"] == 45100.0
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
