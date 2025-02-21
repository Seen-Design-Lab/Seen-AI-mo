import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(data_dir):
    """Load and split the preprocessed data into training, validation, and test sets."""
    preprocessed_data = pd.read_csv(os.path.join(data_dir, 'processed', 'data.csv'))
    
    # Split the data into features and labels
    X = preprocessed_data.drop('target', axis=1)
    y = preprocessed_data['target']
    
    # Split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    
    return X_train, X_val, y_train, y_val
    
def load_test_data(data_dir):
    """Load the test data."""
    test_data = pd.read_csv(os.path.join(data_dir, 'processed', 'test_data.csv'))
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    return X_test, y_test
