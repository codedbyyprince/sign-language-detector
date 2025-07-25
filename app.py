from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from gui import get_hand_landmarks, predict_sign

app = Flask(__name__)

@app.route('/')
def homes():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'sign': None, 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'sign': None, 'error': 'Empty file uploaded'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    landmarks = get_hand_landmarks(frame)
    sign = predict_sign(landmarks)
    return jsonify({'sign': sign})
