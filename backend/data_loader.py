# data pipeline
#!/usr/bin/env python3
"""
Data loader module for anomaly detection dashboard.
Handles loading, cleaning, and basic analysis of datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


class DataLoader:
    """Load and analyze time-series data for anomaly detection."""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.loaded_data = {}
    
    def load_csv(self, filename: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.
        
        Args:
            filename: Name of the CSV file
            parse_dates: List of columns to parse as dates
        
        Returns:
            pandas DataFrame
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Default to parsing 'timestamp' column as datetime
        if parse_dates is None:
            parse_dates = ['timestamp']
        
        df = pd.read_csv(filepath, parse_dates=parse_dates)
        self.loaded_data[filename] = df
        
        print(f"✓ Loaded {filename}: {len(df)} rows")
        return df
    
    def describe_data(self, df: pd.DataFrame, name: str = "") -> Dict:
        """
        Get statistical description of the data.
        
        Args:
            df: DataFrame to describe
            name: Name for display purposes
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max()),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
            } if 'timestamp' in df.columns else None,
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict()
        }
        
        return stats
    
    def clean_data(self, df: pd.DataFrame, remove_duplicates: bool = True,
                   handle_missing: str = 'forward_fill') -> pd.DataFrame:
        """
        Clean the data (remove duplicates, handle missing values).
        
        Args:
            df: DataFrame to clean
            remove_duplicates: Remove duplicate rows
            handle_missing: How to handle missing values ('forward_fill', 'drop', 'interpolate')
        
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates()
            print(f"  Removed {len(df) - len(df_clean)} duplicate rows")
        
        # Handle missing values
        if handle_missing == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
        elif handle_missing == 'interpolate':
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_cols] = df_clean[numeric_cols].interpolate()
        elif handle_missing == 'drop':
            df_clean = df_clean.dropna()
        
        missing_after = df_clean.isnull().sum().sum()
        if missing_after > 0:
            print(f"  Remaining missing values: {missing_after}")
        else:
            print(f"  No missing values")
        
        return df_clean
    
    def normalize_data(self, df: pd.DataFrame, 
                      numeric_cols: Optional[list] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numeric columns to 0-1 range.
        
        Args:
            df: DataFrame to normalize
            numeric_cols: List of columns to normalize. If None, normalize all numeric columns
        
        Returns:
            Tuple of (normalized DataFrame, normalization parameters for denormalization)
        """
        df_norm = df.copy()
        norm_params = {}
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Avoid division by zero
            if max_val == min_val:
                df_norm[col] = 0
                norm_params[col] = {'min': min_val, 'max': min_val, 'range': 0}
            else:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                norm_params[col] = {'min': min_val, 'max': max_val, 'range': max_val - min_val}
        
        print(f"  Normalized {len(numeric_cols)} columns")
        return df_norm, norm_params
    
    def get_time_series_features(self, df: pd.DataFrame, 
                                metric_col: str, 
                                window_sizes: list = [24, 168]) -> pd.DataFrame:
        """
        Calculate time-series features (moving average, rolling std, etc.).
        
        Args:
            df: DataFrame with 'timestamp' column
            metric_col: Name of the metric column to analyze
            window_sizes: List of window sizes for rolling calculations (hours)
        
        Returns:
            DataFrame with additional feature columns
        """
        df_features = df.copy()
        
        for window in window_sizes:
            # Moving average
            df_features[f'{metric_col}_ma_{window}'] = df_features[metric_col].rolling(
                window=window, center=False
            ).mean()
            
            # Rolling standard deviation
            df_features[f'{metric_col}_std_{window}'] = df_features[metric_col].rolling(
                window=window, center=False
            ).std()
            
            # Moving min/max
            df_features[f'{metric_col}_min_{window}'] = df_features[metric_col].rolling(
                window=window, center=False
            ).min()
            df_features[f'{metric_col}_max_{window}'] = df_features[metric_col].rolling(
                window=window, center=False
            ).max()
        
        print(f"  Added {len(window_sizes) * 4} time-series features")
        return df_features
    
    def split_train_test(self, df: pd.DataFrame, 
                        test_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame to split
            test_split: Proportion of data for testing (0.0-1.0)
        
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - test_split))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        return train_df, test_df
    
    def print_summary(self):
        """Print summary of all loaded data."""
        print("\n" + "="*70)
        print("LOADED DATA SUMMARY")
        print("="*70)
        
        for filename, df in self.loaded_data.items():
            print(f"\n{filename}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {', '.join(df.columns)}")
            if 'timestamp' in df.columns:
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


# Example usage
if __name__ == '__main__':
    # Initialize loader
    loader = DataLoader('data')
    
    # Load temperature data
    print("\n1. Loading temperature data...")
    temp_df = loader.load_csv('temperature_logs.csv')
    
    # Get statistics
    print("\n2. Analyzing data...")
    stats = loader.describe_data(temp_df, 'Temperature Data')
    print(f"  Shape: {stats['shape']}")
    print(f"  Columns: {', '.join(stats['columns'])}")
    
    # Clean data
    print("\n3. Cleaning data...")
    temp_clean = loader.clean_data(temp_df)
    
    # Calculate features
    print("\n4. Adding time-series features...")
    temp_features = loader.get_time_series_features(temp_clean, 'temperature', window_sizes=[6, 24])
    
    # Normalize
    print("\n5. Normalizing data...")
    temp_norm, norm_params = loader.normalize_data(temp_features, ['temperature'])
    
    # Split data
    print("\n6. Splitting data...")
    train, test = loader.split_train_test(temp_norm, test_split=0.2)
    
    print(f"\nFirst 5 rows of processed data:")
    print(train.head(5).to_string())