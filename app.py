from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from keras.saving.save import load_model
import numpy as np
import requests
from io import BytesIO
import tensorflow as tf
import base64

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Base route
@app.route('/')
@cross_origin()
def base():
    return 'Face emotion recognition API'


# Prediction API route using image base64
@app.route('/predict_base64', methods=['POST'])
def predict():
    # Get image from url post
    input_json = request.get_json(force=True)
    res = {'image': input_json['image']}

    # prepare res['image'] to be converted to BytesIO
    res['image'] = res['image'].split(',')[1]
    res['image'] = base64.b64decode(res['image'])

    # Convert res['image'] to BytesIO
    imgmodel = BytesIO(res['image'])

    # Classes
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Load model
    model = load_model('facial_recognition_model.h5')

    # Load image
    img = tf.keras.utils.load_img(imgmodel, target_size=(48, 48), color_mode='grayscale')
    img = np.array(img)
    img = img.reshape(1, 48, 48, 1)
    img = img / 255

    # Predict
    prediction = model.predict(img)
    print(classes[np.argmax(prediction)])

    # Return
    return jsonify({'emotion': classes[np.argmax(prediction)]})


# Prediction API route using image URL
@app.route('/predict_url', methods=['POST'])
def predict_emotion():
    # Get image from url post
    input_json = request.get_json(force=True)
    res = {'url': input_json['url']}

    # Load image
    response = requests.get(res['url'])

    # Get io.BytesIO object from response
    imgmodel = BytesIO(response.content)

    # Classes
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Load model
    model = load_model('facial_recognition_model.h5')

    # Load image
    img = tf.keras.utils.load_img(imgmodel, target_size=(48, 48), color_mode='grayscale')
    img = np.array(img)
    img = img.reshape(1, 48, 48, 1)
    img = img / 255

    # Predict
    prediction = model.predict(img)
    print(classes[np.argmax(prediction)])

    # Return
    return jsonify({'emotion': classes[np.argmax(prediction)]})