import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_seenai_model(input_shape, output_shape, config):
    """Build the "SeenAI" model architecture."""
    model = Sequential()
    model.add(Dense(config.hidden_units_1, activation='relu', input_shape=input_shape))
    model.add(Dropout(config.dropout_rate_1))
    model.add(Dense(config.hidden_units_2, activation='relu'))
    model.add(Dropout(config.dropout_rate_2))
    model.add(Dense(output_shape, activation='sigmoid'))
    
    model.compile(optimizer=config.optimizer,
                  loss=config.loss_function,
                  metrics=config.metrics)
    
    return model
