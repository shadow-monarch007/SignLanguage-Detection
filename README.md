# Sign-Language-Detectiom
This project implements a real-time Sign Language Recognition system using MediaPipe, OpenCV, and TensorFlow. It captures hand and pose keypoints, trains an LSTM model, and performs real-time sign detection.

Features
Data Collection: Uses MediaPipe Holistic to extract keypoints from videos and saves them as NumPy arrays.
Model Training: An LSTM-based deep learning model is trained on the collected data.
Real-Time Detection: Uses OpenCV to detect and classify sign language gestures live via webcam.
Confidence Smoothing: Implements majority voting and confidence thresholding to improve detection accuracy.
Prerequisites
Make sure you have Python 3.10 installed.

Dependencies
The following dependencies are required to run this project:

numpy - For numerical operations.
opencv-python - For video processing and computer vision tasks.
tensorflow - For building and training the machine learning model.
mediapipe - For extracting keypoints from videos.
To install the required dependencies, create a virtual environment (optional but recommended) and run the following command:

bash
Copy
Edit
pip install -r requirements.txt
Alternatively, you can manually install the dependencies using the following commands:

bash
Copy
Edit
pip install numpy opencv-python tensorflow mediapipe
Files Overview
capture_data.py: Captures video sequences, extracts keypoints, and saves them as NumPy arrays for training.
train_model.py: Trains the model using the collected keypoints and saves the trained model to a file.
main.py: Runs the real-time sign language detection using the webcam and the trained model.
utils.py: Contains utility functions for keypoint extraction, drawing landmarks, and model inference.
Usage Instructions
1. Collect Data
Run the following command to start collecting data for training:

bash
Copy
Edit
python capture_data.py
This script will open a webcam feed and guide you through collecting data for predefined signs (e.g., 'hello', 'thanks', 'iloveyou'). You will need to press 'q' to quit the collection at any time.

2. Train the Model
After collecting the data, you can train the model with the following command:

bash
Copy
Edit
python train_model.py
This script will load the collected data, train the LSTM-based model, and save the model as model.h5 and the weights as model.weights.h5.

3. Run Real-Time Detection
Once the model is trained, run the following command to start detecting sign language gestures in real-time:

bash
Copy
Edit
python main.py
The script will start the webcam and display the detected signs along with their confidence scores.

Project Structure
bash
Copy
Edit
├── capture_data.py      # Collect data for training
├── train_model.py       # Train the sign language recognition model
├── main.py              # Real-time sign language detection
├── utils.py             # Helper functions for model inference and drawing
├── model.h5             # Trained model (saved after training)
├── model.weights.h5     # Model weights (saved after training)
└── requirements.txt     # List of dependencies for the project
Future Improvements
Add more sign gestures for expanded recognition.
Optimize model performance for real-world deployment.
Implement a GUI for better user interaction.
