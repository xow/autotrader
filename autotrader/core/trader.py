def prepare_features(self, data_point: dict) -> list:
        """Prepare feature vector from a single data point.
        
        Args:
            data_point: Dictionary containing price and indicator data
            
        Returns:
            list: Feature vector for model input
        """
        try:
            if not data_point:
                raise ValueError("Empty data point provided")
            
            # Extract features in the exact order expected by the model
            features = [
                float(data_point.get('price', 0)),
                float(data_point.get('volume', 0)),
                float(data_point.get('spread', 0)),
                float(data_point.get('sma_5', 0)),
                float(data_point.get('sma_20', 0)),
                float(data_point.get('ema_12', 0)),
                float(data_point.get('ema_26', 0)),
                float(data_point.get('rsi', 0)),
                float(data_point.get('macd', 0)),
                float(data_point.get('macd_signal', 0)),
                float(data_point.get('bb_upper', 0)),
                float(data_point.get('bb_lower', 0))
            ]
            
            # Log the first few features for debugging (without sensitive data)
            if len(features) > 0:
                logger.debug(f"Prepared features: price={features[0]:.2f}, volume={features[1]:.2f}, "
                            f"spread={features[2]:.2f}, rsi={features[7]:.2f}")
            
            return features
        
        except Exception as e:
            logger.exception(f"Error preparing features: {e}")
            # Return a zero vector of expected length on error
            features = [0.0] * 12  # 12 features in total
            return features
