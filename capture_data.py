import os
import cv2
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
import numpy as np

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions/signs we try to detect
actions = ['hello', 'thanks', 'iloveyou']

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

def create_folders():
    print("Creating folders for data collection...")
    try:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.exists(action_path):
                os.makedirs(action_path)
                
            for sequence in range(no_sequences):
                sequence_path = os.path.join(action_path, str(sequence))
                if not os.path.exists(sequence_path):
                    os.makedirs(sequence_path)
        print("Folders created successfully!")
    except Exception as e:
        print(f"Error creating folders: {str(e)}")
        raise

def collect_data():
    mp_holistic = mp.solutions.holistic
    
    print("\nStarting data collection...")
    print("Press 'q' to quit at any time")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video capture device")
        
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            print(f"\nCollecting data for action: {action}")
            
            for sequence in range(no_sequences):
                print(f"\nSequence {sequence + 1}/{no_sequences}")
                
                # Countdown before starting
                for countdown in range(5, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        raise Exception("Failed to grab frame")
                        
                    cv2.putText(frame, f"Starting in {countdown}", (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(1000)

                # Collect frames
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        raise Exception("Failed to grab frame")

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    
                    cv2.putText(image, f'Recording {action}', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Sequence {sequence + 1}/{no_sequences}', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, f'Frame {frame_num + 1}/{sequence_length}', (15,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    cv2.imshow('OpenCV Feed', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    try:
        print("Starting Sign Language Data Collection")
        print("-------------------------------------")
        
        # Create folders for data collection
        create_folders()
        
        # Start collecting data
        print("\nPreparing to collect data...")
        print("Available signs:", actions)
        input("Press Enter when ready to start data collection...")
        
        if collect_data():
            print("\nData collection completed successfully!")
        else:
            print("\nData collection was interrupted")
            
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Data collection failed!") 