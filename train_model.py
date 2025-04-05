import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions/signs we try to detect
actions = ['hello', 'thanks', 'iloveyou']

# Data collection parameters
no_sequences = 30
sequence_length = 30

def load_data():
    print("\nLoading collected data...")
    sequences, labels = [], []
    for action_idx, action in enumerate(actions):
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                try:
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                except Exception as e:
                    print(f"Error loading file for {action}, sequence {sequence}, frame {frame_num}")
                    print(f"Error: {str(e)}")
                    raise
            sequences.append(window)
            labels.append(action_idx)
    
    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(labels).astype(int)
    print("Data loaded successfully!")
    return X, y

def train_model():
    print("\nStarting model training...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    log_dir = os.path.join('Logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=(30,1662)),
        tf.keras.layers.Dropout(0.3),
        LSTM(256, return_sequences=True, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        LSTM(128, return_sequences=False, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])

    # Learning rate scheduling
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['categorical_accuracy'])
    
    print("\nModel architecture:")
    model.summary()
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=20,
        restore_best_weights=True
    )
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[tb_callback, early_stopping]
    )

    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nSaving model...")
    try:
        model.save('model.h5', save_format='h5')
        model.save_weights('model.weights.h5')
        print("Model and weights saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise

    return model, history

if __name__ == "__main__":
    try:
        print("Starting Sign Language Model Training")
        print("-----------------------------------")
        train_model()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Training failed!") 