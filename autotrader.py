import requests
from datetime import datetime
import tensorflow as tf
import numpy as np
import json
import os
import time

model_filename = "autotrader_model.keras"
training_data_filename = "training_data.json"

def fetch_market_data():
    # Function to retrieve live market data from BTCMarkets using v3 API
    base_url = 'https://api.btcmarkets.net/v3'
    endpoint = '/markets/tickers?marketId=BTC-AUD' # Using a specific market for demonstration
    url = f"{base_url}{endpoint}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch market data from {url}: {str(e)}")

def collect_training_data(num_samples=10):
    """Collects market data and prepares it for training."""
    training_data = []
    for _ in range(num_samples):
        try:
            data = fetch_market_data()
            if isinstance(data, list) and data:
                btc_aud_data = next((item for item in data if item.get('marketId') == 'BTC-AUD'), None)
                if btc_aud_data and 'lastPrice' in btc_aud_data:
                    try:
                        price = float(btc_aud_data['lastPrice'])
                        # In a real scenario, you'd also collect other features and labels
                        training_data.append(price)
                    except ValueError:
                        print("Warning: Could not convert 'lastPrice' to float.")
                else:
                    print("Warning: 'BTC-AUD' market data or 'lastPrice' not found in the response.")
            else:
                print("Warning: Invalid market data format.")
        except Exception as e:
            print(f"Error collecting data: {e}")
        time.sleep(1)  # Wait a second to avoid rate limiting
    return training_data

def save_training_data(data, filename="training_data.json"):
    """Saves training data to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Training data saved to {filename}")
    except Exception as e:
        print(f"Error saving training data: {e}")

def load_training_data(filename="training_data.json"):
    """Loads training data from a JSON file."""
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        print(f"Training data loaded from {filename}")
        return data
    except FileNotFoundError:
        print(f"No training data file found at {filename}. Starting with an empty dataset.")
        return []
    except Exception as e:
        print(f"Error loading training data: {e}")
        return []

def save_model(model, filename="autotrader_model.keras"):
    """Saves the TensorFlow model to a file."""
    try:
        model.save(filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename="autotrader_model.keras"):
    """Loads the TensorFlow model from a file."""
    try:
        model = tf.keras.models.load_model(filename)
        print(f"Loaded model from {filename}")
        return model
    except FileNotFoundError:
        print(f"No model file found at {filename}. Creating a new model.")
        return None  # Indicate that the model needs to be created
    except Exception as e:
        print(f"Error loading model: {e}. Creating a new model.")
        return None

def predict_optimal_trades(data, model):
    """
    Implement a placeholder machine learning algorithm using TensorFlow to predict optimal trades.
    This is a simplified example and would require proper model training and data preprocessing
    in a real-world scenario.
    """
    if data is None:
        raise TypeError("Input data cannot be None")
    if model is None or not isinstance(data, list) or not data:
        return {"signal": "HOLD", "prediction_score": 0.5}  # Default if no data

    # Extract a numerical feature from the market data for prediction
    # For this example, we expect a list of market objects and will try to find BTC-AUD
    input_feature = 0.0
    btc_aud_data = None
    for market_info in data:
        if market_info.get('marketId') == 'BTC-AUD':
            btc_aud_data = market_info
            break

    if btc_aud_data and 'lastPrice' in btc_aud_data:
        try:
            input_feature = float(btc_aud_data['lastPrice'])
        except ValueError:
            print("Warning: Could not convert 'lastPrice' to float.")
            input_feature = 0.0
    else:
        print("Warning: 'BTC-AUD' market data or 'lastPrice' not found in the response.")
        # Fallback if specific market data isn't found, or handle as an error
        # For now, we'll proceed with 0.0, but a real app might raise an error or log more.

    # Normalize or scale the input feature if necessary for the model
    # For this dummy model, a simple scaling. Adjust based on expected price range.
    scaled_input = np.array([[input_feature / 50000.0]])  # Example scaling, assuming max price around 50000

    # Make a prediction
    try:
        prediction_raw = model.predict(scaled_input, verbose=0)[0][0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"signal": "HOLD", "prediction_score": 0.5}  # Default if prediction fails

    # Convert prediction to a trading signal
    if prediction_raw > 0.6:
        signal = "BUY"
    elif prediction_raw < 0.4:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {"signal": signal, "prediction_score": float(prediction_raw)}

def simulate_trades(prediction):
    """
    Function to perform simulated trades based on predictions.
    Should include logic for handling trade execution without real market impact.
    """
    # Basic simulation example - format as needed by task
    if prediction["signal"] == "BUY":
        print("Simulation: Buy order executed")
    elif prediction["signal"] == "SELL":
        print("Simulation: Sell order executed")
    else:
        print("No trade action taken")

# Main entry point
if __name__ == "__main__":
    save_interval_seconds = 3600  # Save every hour
    last_save_time = 0

    # Load existing training data and model
    training_data = load_training_data(training_data_filename)
    model = load_model(model_filename)

    if model is None:
        print(f"Creating a new model.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

    while True:
        try:
            # Collect new training data
            new_data = collect_training_data(num_samples=10)
            training_data.extend(new_data)

            # Prepare data for training (very basic example)
            if training_data:
                prices = np.array(training_data)
                labels = np.array([1 if i > 0 and prices[i] > prices[i-1] else 0 for i in range(1, len(prices))])
                inputs = np.array([[price / 50000.0] for price in prices[1:]])  # Scale the prices
                if len(inputs) > 0 and len(labels) > 0:
                    model.fit(inputs, labels, epochs=2, verbose=0)  # Reduced epochs for demonstration
                    print("Model trained.")
                else:
                    print("Not enough data to train the model.")
            else:
                print("No training data available.")

            # Fetch market data for prediction
            market_data = fetch_market_data()
            if model:
                trade_prediction = predict_optimal_trades(market_data, model)
                simulate_trades(trade_prediction)
            else:
                print("Model not loaded or created. Cannot make predictions.")

            # Save model and training data at regular intervals
            current_time = time.time()
            if current_time - last_save_time >= save_interval_seconds:
                save_training_data(training_data, training_data_filename)
                save_model(model, model_filename)
                last_save_time = current_time

            time.sleep(60)  # Wait for 60 seconds before the next iteration

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(60)  # Wait before retrying
