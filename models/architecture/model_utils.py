import os
import tensorflow as tf

def save_model(model, model_dir, model_name='seenai_model.h5'):
    """Save the trained model."""
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f'Model saved to: {model_path}')

def load_model(model_dir, model_name='seenai_model.h5'):
    """Load the trained model."""
    model_path = os.path.join(model_dir, model_name)
    model = tf.keras.models.load_model(model_path)
    print(f'Model loaded from: {model_path}')
    return model
