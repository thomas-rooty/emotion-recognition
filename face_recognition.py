from keras.saving.save import load_model
import numpy as np
import tensorflow as tf

# Classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
model = load_model('facial_recognition_model.h5')

# Load image
img = tf.keras.utils.load_img('samples/img.png', target_size=(48, 48), color_mode='grayscale')
img = np.array(img)
img = img.reshape(1, 48, 48, 1)
img = img / 255

# Predict
prediction = model.predict(img)
print(classes[np.argmax(prediction)])
