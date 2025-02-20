import tensorflow as tf
from tensorflow import keras
import numpy as np

class SeenAI:
    def __init__(self):
        self.model = None
        
    def build_model(self, input_shape, output_shape):
        """
        Build the basic architecture for SeenAI
        """
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(output_shape, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train SeenAI with your data
        """
        self.model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        return self.model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2)
    
    def predict(self, X):
        """
        Make predictions with SeenAI
        """
        return self.model.predict(X)

# Initialize SeenAI
seen_ai = SeenAI()