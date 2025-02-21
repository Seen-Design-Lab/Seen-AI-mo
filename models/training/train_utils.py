import os
import numpy as np
from tensorflow.keras.optimizers import Adam

def get_optimizer(config):
    """Get the optimizer for model training."""
    return Adam(learning_rate=config.learning_rate)

def save_model(model, model_dir, model_name='seenai_model.h5'):
    """Save the trained model."""
    model_path = os.path.join(model_dir, model_name)
    model.save_weights(model_path)
    print(f'Model saved to: {model_path}')

def load_model(model, model_dir, model_name='seenai_model.h5'):
    """Load the trained model."""
    model_path = os.path.join(model_dir, model_name)
    model.load_weights(model_path)
    print(f'Model loaded from: {model_path}')
    return model
