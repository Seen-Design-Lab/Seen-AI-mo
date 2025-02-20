# Example training code
import numpy as np
from seen_ai_model import SeenAI

# Create sample data (replace with your actual data)
X_train = np.random.random((1000, 20))  # 1000 samples, 20 features
y_train = np.random.randint(2, size=(1000, 5))  # 5 output classes

# Initialize and train SeenAI
seen_ai = SeenAI()
model = seen_ai.build_model(input_shape=(20,), output_shape=5)
history = seen_ai.train(X_train, y_train, epochs=10)

# Make predictions
sample_data = np.random.random((1, 20))
predictions = seen_ai.predict(sample_data)