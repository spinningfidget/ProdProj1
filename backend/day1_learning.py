#!/usr/bin/env python3
"""
DAY 1: Pandas & Time-Series Concepts
Learning objectives:
1. Load and explore data with pandas
2. Understand time-series data
3. Calculate basic statistics
4. Visualize trends and patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("\n" + "="*70)
print("DAY 1: PANDAS & TIME-SERIES FUNDAMENTALS")
print("="*70)

# ============================================================================
# PART 1: PANDAS BASICS - MUST MEMORIZE
# ============================================================================
print("\n[PART 1] PANDAS BASICS")
print("-" * 70)

# Create a simple DataFrame manually (to understand structure)
data = {
    'date': pd.date_range('2024-01-01', periods=5),
    'temperature': [20.1, 21.3, 19.8, 22.5, 20.9],
    'humidity': [45, 50, 48, 52, 46]
}
df_simple = pd.DataFrame(data)

print("\n1. Create a DataFrame:")
print(df_simple)
print(f"\nDataFrame shape: {df_simple.shape}  # (rows, columns)")
print(f"Column names: {list(df_simple.columns)}")
print(f"Data types:\n{df_simple.dtypes}")

# ============================================================================
# PART 2: DESCRIBING DATA - KEY PANDAS METHODS
# ============================================================================
print("\n\n[PART 2] DESCRIBING DATA - MEMORIZE THESE!")
print("-" * 70)

# First, let's load real data (generate it on the fly)
print("\n2. Generate time-series data:")
dates = pd.date_range('2024-01-01', periods=100, freq='D')
np.random.seed(42)
temperature_data = 20 + 5 * np.sin(np.arange(100) / 15) + np.random.normal(0, 1, 100)

df = pd.DataFrame({
    'date': dates,
    'temperature': temperature_data,
    'sensor_id': 'SENSOR_001'
})

# .describe() - MEMORIZE THIS!
print("\ndf.describe()  # Statistical summary (count, mean, std, min, max, etc.)")
print(df.describe())

print("\nKey stats from df.describe():")
print(f"  Mean temperature: {df['temperature'].mean():.2f}°C")
print(f"  Std deviation: {df['temperature'].std():.2f}°C")
print(f"  Min: {df['temperature'].min():.2f}°C")
print(f"  Max: {df['temperature'].max():.2f}°C")

# ============================================================================
# PART 3: TIME-SERIES CONCEPTS
# ============================================================================
print("\n\n[PART 3] TIME-SERIES CONCEPTS - CRITICAL FOR ANOMALY DETECTION")
print("-" * 70)

print("\n3. Understanding Time-Series Data:")
print("""
A time-series is a sequence of data points indexed by time.
Examples:
  - Temperature readings every hour
  - Stock prices every minute
  - Server CPU usage every second
  - Website traffic every day

Key characteristics:
  - TREND: Long-term direction (up, down, or flat)
  - SEASONALITY: Repeating patterns (daily, weekly, yearly)
  - NOISE: Random fluctuations
  - ANOMALIES: Unusual points that don't fit the pattern
""")

# Visualize the concept
print("\n4. Decompose the time-series:")
print(f"   Your data: {df['temperature'].min():.2f} to {df['temperature'].max():.2f}°C")
print(f"   Mean: {df['temperature'].mean():.2f}°C")
print(f"   Std Dev: {df['temperature'].std():.2f}°C")
print(f"   Range (Max - Min): {df['temperature'].max() - df['temperature'].min():.2f}°C")

# ============================================================================
# PART 4: INDEXING - MEMORIZE THESE PATTERNS
# ============================================================================
print("\n\n[PART 4] INDEXING - THESE 5 PATTERNS ARE ESSENTIAL")
print("-" * 70)

print("\n5. Five essential indexing patterns:")

# Pattern 1: .loc[] - by label
print("\n  Pattern 1: df.loc[] - index by label/name")
print(f"    df.loc[0, 'temperature'] = {df.loc[0, 'temperature']:.2f}")

# Pattern 2: .iloc[] - by position
print("\n  Pattern 2: df.iloc[] - index by position (0, 1, 2, ...)")
print(f"    df.iloc[0, 1] = {df.iloc[0, 1]:.2f}  # First row, second column")

# Pattern 3: Column selection
print("\n  Pattern 3: df['column'] - select a column")
print(f"    df['temperature'].head(3) = \n{df['temperature'].head(3).values}")

# Pattern 4: Boolean indexing
print("\n  Pattern 4: Boolean indexing - filter by condition")
hot_days = df[df['temperature'] > df['temperature'].mean() + 1]
print(f"    Days hotter than mean+1std: {len(hot_days)} days")

# Pattern 5: Slicing
print("\n  Pattern 5: Slicing - df[start:end]")
print(f"    df[0:3] = {len(df[0:3])} rows")

# ============================================================================
# PART 5: ROLLING WINDOWS - CRITICAL FOR TIME-SERIES
# ============================================================================
print("\n\n[PART 5] ROLLING WINDOWS - DETECT TRENDS & ANOMALIES")
print("-" * 70)

print("\n6. Calculate rolling statistics:")

# 7-day moving average
df['ma_7'] = df['temperature'].rolling(window=7, center=False).mean()

# 7-day rolling std (helps detect anomalies)
df['std_7'] = df['temperature'].rolling(window=7).std()

print(f"\n  7-day moving average (ma_7):")
print(df[['date', 'temperature', 'ma_7']].head(10).to_string(index=False))

print(f"\n\n  7-day rolling std (std_7) - helps find anomalies:")
print(df[['date', 'temperature', 'std_7']].tail(10).to_string(index=False))

# ============================================================================
# PART 6: DETECTING OUTLIERS WITH STATISTICS
# ============================================================================
print("\n\n[PART 6] SIMPLE ANOMALY DETECTION METHODS")
print("-" * 70)

print("\n7. Method 1: Z-Score (values > 3 std away from mean = anomaly)")

# Calculate Z-score
mean = df['temperature'].mean()
std = df['temperature'].std()
df['z_score'] = (df['temperature'] - mean) / std

# Find anomalies
anomalies_z = df[df['z_score'].abs() > 3]
print(f"   Found {len(anomalies_z)} anomalies using Z-score")
if len(anomalies_z) > 0:
    print(f"   Anomalies:\n{anomalies_z[['date', 'temperature', 'z_score']].to_string(index=False)}")

print("\n8. Method 2: IQR (Interquartile Range - values beyond Q1-1.5*IQR or Q3+1.5*IQR)")

Q1 = df['temperature'].quantile(0.25)
Q3 = df['temperature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies_iqr = df[(df['temperature'] < lower_bound) | (df['temperature'] > upper_bound)]
print(f"   Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"   Found {len(anomalies_iqr)} anomalies using IQR")

# ============================================================================
# PART 7: DATA RESAMPLE - TIME-SERIES AGGREGATION
# ============================================================================
print("\n\n[PART 7] RESAMPLING - CHANGE TIME FREQUENCY")
print("-" * 70)

print("\n9. Resample data to different frequencies:")

# Daily average
daily_avg = df.set_index('date')['temperature'].resample('D').mean()
print(f"   Daily average temperature:\n{daily_avg.head()}")

# Weekly average
weekly_avg = df.set_index('date')['temperature'].resample('W').mean()
print(f"\n   Weekly average temperature:\n{weekly_avg.head()}")

# ============================================================================
# PART 8: PRACTICAL EXERCISE
# ============================================================================
print("\n\n[PART 8] YOUR TURN - PRACTICE EXERCISE")
print("-" * 70)

print("""
EXERCISE: Analyze the temperature data and answer these questions:

1. What's the average temperature over the entire dataset?
2. How many days had temperatures above 25°C?
3. What's the 7-day moving average on day 50?
4. How many anomalies did you find using Z-score method (threshold = 2)?
5. Which sensor_id appears in the data?
""")

# Show the data you'll analyze
print("\nData to analyze:")
print(df.head(10).to_string(index=False))

# ANSWERS (uncomment to check your work)
print("\n--- ANSWERS (for checking) ---")
print(f"1. Average temperature: {df['temperature'].mean():.2f}°C")
print(f"2. Days above 25°C: {len(df[df['temperature'] > 25])}")
print(f"3. 7-day MA on day 50: {df.loc[50, 'ma_7']:.2f}°C")
z_threshold_2 = df[(df['z_score'].abs() > 2)]
print(f"4. Anomalies (Z-score > 2): {len(z_threshold_2)}")
print(f"5. Sensor ID: {df['sensor_id'].unique()}")

# ============================================================================
# PART 9: KEY FORMULAS TO MEMORIZE
# ============================================================================
print("\n\n[PART 9] KEY FORMULAS & CODE PATTERNS TO MEMORIZE")
print("-" * 70)

print("""
FORMULA 1 - Z-SCORE ANOMALY DETECTION:
    z = (x - mean) / std
    If |z| > 3: point is an anomaly
    
    Code:
    >>> mean = df['value'].mean()
    >>> std = df['value'].std()
    >>> z_scores = (df['value'] - mean) / std
    >>> anomalies = df[z_scores.abs() > 3]

FORMULA 2 - IQR ANOMALY DETECTION:
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Lower bound = Q1 - 1.5 * IQR
    Upper bound = Q3 + 1.5 * IQR
    If x < lower or x > upper: anomaly
    
    Code:
    >>> Q1 = df['value'].quantile(0.25)
    >>> Q3 = df['value'].quantile(0.75)
    >>> IQR = Q3 - Q1
    >>> bounds = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

FORMULA 3 - ROLLING AVERAGE:
    MA = mean of last N values
    Used to smooth data and see trends
    
    Code:
    >>> df['ma_7'] = df['value'].rolling(window=7).mean()

FORMULA 4 - ROLLING STANDARD DEVIATION:
    STD = standard deviation of last N values
    Used to detect when variability increases
    
    Code:
    >>> df['std_7'] = df['value'].rolling(window=7).std()
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*70)
print("DAY 1 SUMMARY - WHAT YOU SHOULD KNOW")
print("="*70)

print("""
✓ How to load CSV data with pandas
✓ Key methods: .describe(), .loc[], .iloc[], .head(), .tail()
✓ What time-series data is and its components (trend, seasonality, anomaly)
✓ How to calculate Z-score and IQR for anomaly detection
✓ How to use rolling windows to detect trends
✓ How to resample data to different time frequencies

NEXT STEPS:
1. Run this script and understand every output
2. Modify the code and experiment with different parameters
3. Load your own CSV files and practice the same operations
4. Tomorrow: Implement these detection methods in a separate module
""")

print("\n")