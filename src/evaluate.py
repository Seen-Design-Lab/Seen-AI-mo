from model import build_seenai_model
from utils.data_utils import load_test_data

def evaluate_seenai_model(model, config):
    """Evaluate the trained "SeenAI" model."""
    # Load the test data
    X_test, y_test = load_test_data(config.data_dir)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy
