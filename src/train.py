from model import build_seenai_model
from utils.data_utils import load_and_split_data

def train_seenai_model(config):
    """Train the "SeenAI" model."""
    # Load and split the data
    X_train, X_val, y_train, y_val = load_and_split_data(config.data_dir)
    
    # Build the model
    model = build_seenai_model(input_shape=X_train.shape[1], output_shape=y_train.shape[1])
    
    # Train the model
    model.fit(X_train, y_train,
              epochs=config.num_epochs,
              batch_size=config.batch_size,
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])
    
    return model
