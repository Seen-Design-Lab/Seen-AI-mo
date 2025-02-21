import os
import pandas as pd

def save_preprocessed_data(data, data_dir, filename='preprocessed_data.csv'):
    """Save the preprocessed data to a CSV file."""
    data_path = os.path.join(data_dir, 'processed', filename)
    pd.DataFrame(data).to_csv(data_path, index=False)
    print(f'Preprocessed data saved to: {data_path}')

def load_preprocessed_data(data_dir, filename='preprocessed_data.csv'):
    """Load the preprocessed data from a CSV file."""
    data_path = os.path.join(data_dir, 'processed', filename)
    return pd.read_csv(data_path)
