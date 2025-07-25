import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the trained model
model = joblib.load("sign_language_model.pkl")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

def speak(text):
    """Speak the detected sign using TTS."""
    engine.say(text)
    engine.runAndWait()

def get_hand_landmarks(frame):
    """Detect hand landmarks from a frame and return as a NumPy array."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    landmarks_list = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks_list.append(np.array(landmarks).reshape(1, -1))
    return landmarks_list

def predict_sign(landmarks):
    """Predict the sign from landmarks using the trained model."""
    if landmarks is not None and len(landmarks) > 0:
        prediction = model.predict(landmarks[0])[0]
        return prediction
    return None

def capture_frame_from_camera(camera_index=1):
    """Capture a single frame from the webcam."""
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to access the webcam.")
    return frame

# Example usage (for testing, remove or comment out in production)
if __name__ == "__main__":
    frame = capture_frame_from_camera()
    landmarks = get_hand_landmarks(frame)
    sign = predict_sign(landmarks)
    if sign:
        print(f"Detected Sign: {sign}")
        speak(sign)
    else:
        print("No hand detected.")
