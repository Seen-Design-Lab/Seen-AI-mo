import pickle

class SeenAIDeployment:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def save_model(self, model):
        """
        Save SeenAI model to disk
        """
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
            
    def load_model(self):
        """
        Load SeenAI model from disk
        """
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)