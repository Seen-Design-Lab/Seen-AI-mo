import os
import numpy as np
from seenai.model import build_seenai_model
from seenai.utils.data_utils import load_test_data

def evaluate_seenai_model(model, config):
    """Evaluate the trained "SeenAI" model."""
    # Load the test data
    X_test, y_test = load_test_data(config.data_dir)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return loss, accuracy

def evaluate_and_save_results(model, config):
    """Evaluate the model and save the results."""
    test_loss, test_accuracy = evaluate_seenai_model(model, config)
    
    # Save the evaluation results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    
    results_path = os.path.join(config.model_dir, 'evaluation_results.json')
    np.save(results_path, results)
    
    print(f"Evaluation results saved to: {results_path}")
