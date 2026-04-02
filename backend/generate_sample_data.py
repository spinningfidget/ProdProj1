#!/usr/bin/env python3
"""
Generate sample datasets for anomaly detection testing.
Creates three types of data: temperature logs, system metrics, and business KPIs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_temperature_data(days=30, anomaly_percentage=0.05):
    """
    Generate realistic temperature sensor data with random anomalies.
    
    Args:
        days: Number of days of data to generate
        anomaly_percentage: Percentage of data points that are anomalies (0.0-1.0)
    
    Returns:
        pandas DataFrame with timestamp and temperature columns
    """
    # Create timestamps (one reading per hour)
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days * 24,
        freq='h'
    )
    
    # Generate base temperature data (20-25°C normal, with daily cycle)
    np.random.seed(42)
    hour_of_day = timestamps.hour
    base_temp = 22.5 + 2 * np.sin(2 * np.pi * hour_of_day / 24)
    noise = np.random.normal(0, 0.5, len(timestamps))
    temperatures = base_temp + noise
    
    # Add random anomalies (spikes to extreme values)
    n_anomalies = int(len(temperatures) * anomaly_percentage)
    anomaly_indices = np.random.choice(len(temperatures), n_anomalies, replace=False)
    
    temperatures = np.array(temperatures)  # Convert to mutable array
    for idx in anomaly_indices:
        # Create spike anomalies (very hot or very cold)
        if np.random.random() > 0.5:
            temperatures[idx] += np.random.uniform(8, 15)  # Hot spike
        else:
            temperatures[idx] -= np.random.uniform(8, 15)  # Cold spike
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.round(temperatures, 2),
        'sensor_id': 'TEMP_001'
    })
    
    return df


def generate_system_metrics(days=30, anomaly_percentage=0.05):
    """
    Generate CPU/Memory usage data with anomalies.
    
    Args:
        days: Number of days of data
        anomaly_percentage: Percentage of anomalies
    
    Returns:
        pandas DataFrame with metrics
    """
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days * 24,
        freq='h'
    )
    
    np.random.seed(43)
    
    # Normal CPU usage (30-60%), with business hour peaks
    hour_of_day = timestamps.hour
    base_cpu = 45 + 15 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    base_cpu = np.clip(base_cpu, 20, 75)  # Keep in realistic range
    cpu_usage = base_cpu + np.random.normal(0, 5, len(timestamps))
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    # Normal Memory usage (50-80%)
    memory_usage = np.random.normal(65, 8, len(timestamps))
    memory_usage = np.clip(memory_usage, 20, 95)
    
    # Add anomalies
    n_anomalies = int(len(timestamps) * anomaly_percentage)
    anomaly_indices = np.random.choice(len(timestamps), n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # CPU or memory spike to near 100%
        if np.random.random() > 0.5:
            cpu_usage[idx] = np.random.uniform(85, 100)
        else:
            memory_usage[idx] = np.random.uniform(85, 100)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.round(cpu_usage, 2),
        'memory_usage': np.round(memory_usage, 2),
        'disk_io': np.round(np.random.normal(30, 10, len(timestamps)), 2)
    })
    
    return df


def generate_business_kpi_data(days=30, anomaly_percentage=0.05):
    """
    Generate business KPI data (sales, user activity, revenue).
    
    Args:
        days: Number of days of data
        anomaly_percentage: Percentage of anomalies
    
    Returns:
        pandas DataFrame with KPI metrics
    """
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days * 24,
        freq='h'
    )
    
    np.random.seed(44)
    
    # Sales per hour (trending up, with weekly cycle)
    day_of_week = timestamps.dayofweek
    base_sales = 100 * (1 + 0.01 * np.arange(len(timestamps)) / len(timestamps))
    weekly_pattern = 0.8 + 0.4 * np.sin(2 * np.pi * day_of_week / 7)
    sales = base_sales * weekly_pattern + np.random.normal(0, 20, len(timestamps))
    sales = np.maximum(sales, 10)  # No negative sales
    
    # User sessions (similar pattern)
    sessions = (sales / 50 + np.random.normal(0, 2, len(timestamps))).astype(int)
    sessions = np.maximum(sessions, 0)
    
    # Revenue (correlated with sales)
    revenue = sales * np.random.uniform(80, 120, len(timestamps)) / 100
    
    # Add anomalies (sudden drops or spikes)
    n_anomalies = int(len(timestamps) * anomaly_percentage)
    anomaly_indices = np.random.choice(len(timestamps), n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        if np.random.random() > 0.5:
            sales[idx] *= np.random.uniform(0.1, 0.3)  # Sudden drop
        else:
            sales[idx] *= np.random.uniform(2.5, 4.0)  # Sudden spike
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'sales': np.round(sales, 2),
        'sessions': sessions,
        'revenue': np.round(revenue, 2)
    })
    
    return df


def save_datasets(output_dir='data'):
    """
    Generate all sample datasets and save them to CSV files.
    
    Args:
        output_dir: Directory to save CSV files
    """
    # Create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating sample datasets...")
    
    # Generate and save temperature data
    temp_data = generate_temperature_data()
    temp_path = os.path.join(output_dir, 'temperature_logs.csv')
    temp_data.to_csv(temp_path, index=False)
    print(f"✓ Temperature data saved to {temp_path}")
    
    # Generate and save system metrics
    system_data = generate_system_metrics()
    system_path = os.path.join(output_dir, 'system_metrics.csv')
    system_data.to_csv(system_path, index=False)
    print(f"✓ System metrics saved to {system_path}")
    
    # Generate and save business KPI data
    kpi_data = generate_business_kpi_data()
    kpi_path = os.path.join(output_dir, 'business_kpi.csv')
    kpi_data.to_csv(kpi_path, index=False)
    print(f"✓ Business KPI data saved to {kpi_path}")
    
    return {
        'temperature': temp_data,
        'system_metrics': system_data,
        'business_kpi': kpi_data
    }


if __name__ == '__main__':
    # Generate datasets
    datasets = save_datasets('data')
    
    # Display summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\n  Columns: {', '.join(df.columns)}")
        print(f"\n  First few rows:")
        print(df.head(3).to_string(index=False))