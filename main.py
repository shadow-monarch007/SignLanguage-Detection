import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz
from collections import deque

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Define actions/signs
actions = np.array(['hello', 'thanks', 'iloveyou'])
colors = [(245,117,16), (117,245,16), (16,117,245)]

try:
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('model.h5')
    model.load_weights('model.weights.h5')  # Updated filename to match what train.py saves
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def get_majority_vote(predictions, confidence_scores):
    if not predictions:
        return None, 0
    
    # Count occurrences and sum confidences for each prediction
    counts = {}
    confidences = {}
    for pred, conf in zip(predictions, confidence_scores):
        if pred in counts:
            counts[pred] += 1
            confidences[pred] += conf
        else:
            counts[pred] = 1
            confidences[pred] = conf
    
    # Get the majority prediction
    majority_pred = max(counts.items(), key=lambda x: x[1])[0]
    avg_confidence = confidences[majority_pred] / counts[majority_pred]
    
    return majority_pred, avg_confidence

def run_detection():
    # Detection variables
    sequence = []
    sentence = []
    predictions_queue = deque(maxlen=30)  # Store more predictions for smoothing
    confidence_queue = deque(maxlen=30)  # Store confidence scores
    
    # Detection parameters
    threshold = 0.85  # Higher threshold for more confident predictions
    min_consecutive = 15  # More consecutive predictions required
    min_confidence = 0.75  # Minimum average confidence required
    
    cap = cv2.VideoCapture(0)
    
    # Increase MediaPipe confidence thresholds
    with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=2  # Use more accurate model
    ) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                continue

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                
                # Get prediction and confidence
                pred_idx = np.argmax(res)
                confidence = res[pred_idx]
                
                # Only consider confident predictions
                if confidence > threshold:
                    predictions_queue.append(pred_idx)
                    confidence_queue.append(confidence)
                    
                    # Get majority prediction from recent history
                    majority_pred, avg_confidence = get_majority_vote(
                        list(predictions_queue), 
                        list(confidence_queue)
                    )
                    
                    # Update sentence if we have consistent predictions
                    if (len(predictions_queue) >= min_consecutive and 
                        avg_confidence > min_confidence):
                        if len(sentence) == 0 or actions[majority_pred] != sentence[-1]:
                            sentence.append(actions[majority_pred])
                            # Clear queues after adding prediction
                            predictions_queue.clear()
                            confidence_queue.clear()
                
                # Limit sentence length
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                # Visualization
                image = prob_viz(res, actions, image, colors)
                
                # Display confidence
                conf_txt = f'Confidence: {confidence:.2f}'
                cv2.putText(image, conf_txt, (400,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                # Display average confidence if available
                if len(confidence_queue) > 0:
                    avg_conf_txt = f'Avg Conf: {sum(confidence_queue)/len(confidence_queue):.2f}'
                    cv2.putText(image, avg_conf_txt, (400,60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Display prediction
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('Sign Language Detection', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection() 