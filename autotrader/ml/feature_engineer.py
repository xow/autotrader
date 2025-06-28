"""
Feature Engineering Pipeline for autotrader bot.

Provides comprehensive feature engineering capabilities including market data preprocessing,
technical indicators integration, and data normalization for ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

from .indicators import TechnicalIndicators

logger = logging.getLogger("autotrader.ml.feature_engineer")


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    use_sma: bool = True
    sma_periods: List[int] = None
    use_ema: bool = True
    ema_periods: List[int] = None
    use_rsi: bool = True
    rsi_period: int = 14
    use_macd: bool = True
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    use_bollinger: bool = True
    bb_period: int = 20
    bb_std: int = 2
    use_volume_indicators: bool = True
    
    # Price features
    use_price_ratios: bool = True
    use_price_differences: bool = True
    use_log_returns: bool = True
    use_volatility: bool = True
    volatility_window: int = 10
    
    # Time-based features
    use_time_features: bool = True
    use_cyclical_encoding: bool = True
    
    # Lag features
    use_lag_features: bool = True
    lag_periods: List[int] = None
    
    # Rolling statistics
    use_rolling_stats: bool = True
    rolling_windows: List[int] = None
    
    # Scaling
    scaling_method: str = "standard"  # standard, minmax, robust, quantile
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering pipeline for market data.
    
    Transforms raw market data into ML-ready features including technical indicators,
    price transformations, time-based features, and proper normalization.
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureConfig()
        self.indicators = TechnicalIndicators()
        self.scaler = None
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False
        
        # Initialize scaler based on config
        self._init_scaler()
        
        logger.info("Feature engineer initialized")
    
    def _init_scaler(self):
        """Initialize scaler based on configuration."""
        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        elif self.config.scaling_method == "quantile":
            self.scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
        else:
            logger.warning(f"Unknown scaling method: {self.config.scaling_method}, using standard")
            self.scaler = StandardScaler()
    
    def fit(self, X: Union[List[Dict], pd.DataFrame], y=None) -> 'FeatureEngineer':
        """
        Fit the feature engineer to training data.
        
        Args:
            X: Raw market data (list of dicts or DataFrame)
            y: Target values (ignored)
        
        Returns:
            Self
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(X, list):
                df = pd.DataFrame(X)
            else:
                df = X.copy()
            
            # Generate all features
            features_df = self._generate_all_features(df)
            
            # Store feature names
            self.feature_names_ = list(features_df.columns)
            
            # Fit scaler
            self.scaler.fit(features_df.values)
            
            self.is_fitted_ = True
            logger.info(f"Feature engineer fitted with {len(self.feature_names_)} features")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting feature engineer: {e}")
            raise
    
    def transform(self, X: Union[List[Dict], pd.DataFrame]) -> np.ndarray:
        """
        Transform raw market data into engineered features.
        
        Args:
            X: Raw market data
        
        Returns:
            Transformed features array
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        try:
            # Convert to DataFrame if needed
            if isinstance(X, list):
                df = pd.DataFrame(X)
            else:
                df = X.copy()
            
            # Generate features
            features_df = self._generate_all_features(df)
            
            # Ensure same feature order
            if list(features_df.columns) != self.feature_names_:
                features_df = features_df.reindex(columns=self.feature_names_, fill_value=0)
            
            # Scale features
            scaled_features = self.scaler.transform(features_df.values)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            raise
    
    def fit_transform(self, X: Union[List[Dict], pd.DataFrame], y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all configured features."""
        features_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['price', 'volume', 'timestamp']
        for col in required_cols:
            if col not in features_df.columns:
                if col == 'price':
                    # Try common price column names
                    price_cols = ['close', 'last', 'price_close', 'close_price']
                    for pcol in price_cols:
                        if pcol in features_df.columns:
                            features_df['price'] = features_df[pcol]
                            break
                    else:
                        raise ValueError(f"No price column found. Expected one of: {price_cols}")
                elif col == 'volume':
                    if 'vol' in features_df.columns:
                        features_df['volume'] = features_df['vol']
                    else:
                        logger.warning("No volume column found, using zeros")
                        features_df['volume'] = 0
                elif col == 'timestamp':
                    if 'time' in features_df.columns:
                        features_df['timestamp'] = features_df['time']
                    else:
                        logger.warning("No timestamp column, using index")
                        features_df['timestamp'] = pd.to_datetime(features_df.index, unit='s')
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(features_df['timestamp']):
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], unit='s', errors='coerce')
        
        # Generate feature groups
        if self.config.use_sma or self.config.use_ema:
            features_df = self._add_moving_averages(features_df)
        
        if self.config.use_rsi:
            features_df = self._add_rsi_features(features_df)
        
        if self.config.use_macd:
            features_df = self._add_macd_features(features_df)
        
        if self.config.use_bollinger:
            features_df = self._add_bollinger_features(features_df)
        
        if self.config.use_volume_indicators:
            features_df = self._add_volume_features(features_df)
        
        if self.config.use_price_ratios:
            features_df = self._add_price_ratios(features_df)
        
        if self.config.use_price_differences:
            features_df = self._add_price_differences(features_df)
        
        if self.config.use_log_returns:
            features_df = self._add_log_returns(features_df)
        
        if self.config.use_volatility:
            features_df = self._add_volatility_features(features_df)
        
        if self.config.use_time_features:
            features_df = self._add_time_features(features_df)
        
        if self.config.use_lag_features:
            features_df = self._add_lag_features(features_df)
        
        if self.config.use_rolling_stats:
            features_df = self._add_rolling_statistics(features_df)

        # Add bid/ask related features if available
        features_df = self._add_bid_ask_features(features_df)

        # Select only numeric columns for the final feature set
        # This ensures that non-numeric columns like 'marketId' are excluded before scaling.
        numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
        result_df = features_df[numeric_cols].copy()
        
        # Handle missing values
        result_df = result_df.ffill().bfill().fillna(0)
        
        return result_df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple and exponential moving averages."""
        prices = df['price'].values
        
        if self.config.use_sma:
            for period in self.config.sma_periods:
                df[f'sma_{period}'] = df['price'].rolling(window=period, min_periods=1).mean()
        
        if self.config.use_ema:
            for period in self.config.ema_periods:
                df[f'ema_{period}'] = df['price'].ewm(span=period, adjust=False).mean()
        
        return df
    
    def _add_rsi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and related features."""
        prices = df['price'].values
        
        # Basic RSI
        rsi_values = []
        for i in range(len(prices)):
            if i < self.config.rsi_period:
                rsi_values.append(50.0)  # Neutral
            else:
                window_prices = prices[max(0, i-self.config.rsi_period):i+1]
                rsi = self.indicators.relative_strength_index(window_prices, self.config.rsi_period)
                rsi_values.append(rsi)
        
        df['rsi'] = rsi_values
        
        # RSI-based features
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
        
        return df
    
    def _add_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD and related features."""
        prices = df['price'].values
        
        # Calculate MACD for each row
        macd_values = []
        signal_values = []
        histogram_values = []
        
        for i in range(len(prices)):
            if i < max(self.config.macd_slow, self.config.macd_signal):
                macd_values.append(0.0)
                signal_values.append(0.0)
                histogram_values.append(0.0)
            else:
                window_prices = prices[:i+1]
                macd_data = self.indicators.macd(
                    window_prices, 
                    self.config.macd_fast, 
                    self.config.macd_slow, 
                    self.config.macd_signal
                )
                macd_values.append(macd_data['macd'])
                signal_values.append(macd_data['signal'])
                histogram_values.append(macd_data['histogram'])
        
        df['macd'] = macd_values
        df['macd_signal'] = signal_values
        df['macd_histogram'] = histogram_values
        
        # MACD-based features
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(int)
        
        return df
    
    def _add_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        prices = df['price'].values
        
        # Calculate Bollinger Bands
        bb_upper = []
        bb_middle = []
        bb_lower = []
        
        for i in range(len(prices)):
            if i < self.config.bb_period:
                price = prices[i] if i < len(prices) else 0
                bb_upper.append(price)
                bb_middle.append(price)
                bb_lower.append(price)
            else:
                window_prices = prices[max(0, i-self.config.bb_period+1):i+1]
                bb_data = self.indicators.bollinger_bands(
                    window_prices, 
                    self.config.bb_period, 
                    self.config.bb_std
                )
                bb_upper.append(bb_data['upper'])
                bb_middle.append(bb_data['middle'])
                bb_lower.append(bb_data['lower'])
        
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Bollinger Bands features
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20, min_periods=1).quantile(0.1)).astype(int)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume price trend
        df['vpt'] = (df['volume'] * ((df['price'] - df['price'].shift(1)) / df['price'].shift(1))).cumsum()
        
        # On-balance volume
        df['obv'] = (df['volume'] * np.where(df['price'] > df['price'].shift(1), 1, 
                                           np.where(df['price'] < df['price'].shift(1), -1, 0))).cumsum()
        
        return df
    
    def _add_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price ratio features."""
        for period in [5, 10, 20]:
            df[f'price_ratio_{period}'] = df['price'] / df['price'].rolling(window=period, min_periods=1).mean()
        
        return df
    
    def _add_price_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price difference features."""
        df['price_change'] = df['price'].diff()
        df['price_change_pct'] = df['price'].pct_change()
        
        for period in [1, 2, 5, 10]:
            df[f'price_change_{period}'] = df['price'].diff(period)
            df[f'price_change_pct_{period}'] = df['price'].pct_change(period)
        
        return df
    
    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log return features."""
        df['log_return'] = np.log(df['price'] / df['price'].shift(1))
        
        for period in [1, 2, 5, 10]:
            df[f'log_return_{period}'] = np.log(df['price'] / df['price'].shift(period))
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Price volatility (rolling standard deviation of returns)
        df['volatility'] = df['price'].pct_change().rolling(
            window=self.config.volatility_window, min_periods=1
        ).std()
        
        # High-low volatility
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_volatility'] = (df['high'] - df['low']) / df['price']
        
        # Average True Range (ATR) approximation
        df['atr'] = df['price'].diff().abs().rolling(window=14, min_periods=1).mean()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            return df
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        if self.config.use_cyclical_encoding:
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        for lag in self.config.lag_periods:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        return df
    
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        for window in self.config.rolling_windows:
            # Price statistics
            df[f'price_mean_{window}'] = df['price'].rolling(window=window, min_periods=1).mean()
            df[f'price_std_{window}'] = df['price'].rolling(window=window, min_periods=1).std()
            df[f'price_min_{window}'] = df['price'].rolling(window=window, min_periods=1).min()
            df[f'price_max_{window}'] = df['price'].rolling(window=window, min_periods=1).max()
            df[f'price_skew_{window}'] = df['price'].rolling(window=window, min_periods=1).skew()
            
            # Volume statistics
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window, min_periods=1).std()
        
        return df
    
    def get_feature_importance(self, model=None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            model: Trained model with feature_importances_ attribute
        
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_fitted_:
            return {}
        
        if model is None or not hasattr(model, 'feature_importances_'):
            # Return uniform importance
            return {name: 1.0 / len(self.feature_names_) for name in self.feature_names_}
        
        importances = model.feature_importances_
        return dict(zip(self.feature_names_, importances))

    def _add_bid_ask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features derived from bid and ask prices."""
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = df['ask'] - df['bid']
            df['mid_price'] = (df['bid'] + df['ask']) / 2
        return df

    def get_feature_names(self) -> List[str]:
        """Get names of engineered features."""
        return self.feature_names_.copy()

    def inverse_transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original space.
        
        Args:
            X: Scaled features
        
        Returns:
            Unscaled features
        """
        if not self.is_fitted_:
            raise ValueError("FeatureEngineer must be fitted before inverse transform")
        
        return self.scaler.inverse_transform(X)
    
    def prepare_sequences(
        self, 
        X: Union[List[Dict], pd.DataFrame], 
        sequence_length: int = 20
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare sequential data for time-series models.
        
        Args:
            X: Raw market data
            sequence_length: Length of sequences to create
        
        Returns:
            Tuple of (sequences, sample_timestamps) 
        """
        # Transform features
        features = self.transform(X)
        
        if len(features) < sequence_length:
            logger.warning(f"Insufficient data for sequences. Need at least {sequence_length}, got {len(features)}")
            return np.array([]), []
        
        # Create sequences
        sequences = []
        timestamps = []
        
        for i in range(sequence_length, len(features)):
            seq = features[i-sequence_length:i]
            sequences.append(seq)
            
            # Track timestamp if available
            if isinstance(X, pd.DataFrame) and 'timestamp' in X.columns:
                timestamps.append(str(X.iloc[i]['timestamp']))
            elif isinstance(X, list) and len(X) > i and 'timestamp' in X[i]:
                timestamps.append(str(X[i]['timestamp']))
            else:
                timestamps.append(f"sample_{i}")
        
        return np.array(sequences), timestamps
