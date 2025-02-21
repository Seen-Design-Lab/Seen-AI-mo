from preprocessing import load_raw_data, preprocess_data
from model import build_seenai_model
from train import train_seenai_model
from evaluate import evaluate_seenai_model
from config import Config

def main():
    # Load configuration
    config = Config()
    
    # Load and preprocess the data
    user_surveys, user_interviews, user_logs = load_raw_data(config.data_dir)
    preprocessed_data = preprocess_data(user_surveys, user_interviews, user_logs)
    
    # Train the "SeenAI" model
    model = train_seenai_model(config)
    
    # Evaluate the trained model
    
