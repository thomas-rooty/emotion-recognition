import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.applications import InceptionV3
from keras.layers import Dropout, Flatten, Dense, Input
import tensorflow as tf
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

# Load training data
train_df = pd.read_csv('dataset/train.csv')
print(train_df.shape)

# Preprocessing
# create a 4D array of pixel values
train_df['pixels'] = [np.fromstring(x, dtype=int, sep=' ').reshape(-1, 48, 48, 1) for x in train_df['pixels']]
pixels = np.concatenate(train_df['pixels'])
labels = train_df.emotion.values

print(pixels.shape)
print(labels.shape)

# Label distribution
train_dist = (train_df.emotion.value_counts() / len(train_df)).to_frame().sort_index(ascending=True).T
sns.displot(train_dist, x=train_df.emotion)

# View samples of images
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
plt.close()
plt.rcParams["figure.figsize"] = [16, 16]

row = 0
for emotion in np.unique(labels):

    all_emotion_images = train_df[train_df['emotion'] == emotion]
    for i in range(5):
        img = all_emotion_images.iloc[i,].pixels.reshape(48, 48)
        lab = emotions[emotion]

        plt.subplot(7, 5, row + i + 1)
        plt.imshow(img, cmap='binary_r')
        plt.text(-30, 5, s=str(lab), fontsize=10, color='b')
        plt.axis('off')
    row += 5

plt.show()

# Data split for training and validation
X_train, X_valid, y_train, y_valid = train_test_split(pixels, labels, test_size=0.2, stratify=labels, random_state=1)

print('X_train Shape:', X_train.shape)
print('y_train Shape:', y_train.shape)
print()
print('X_valid Shape:', X_valid.shape)
print('y_valid Shape:', y_valid.shape)
# normalizing
Xs_train = X_train / 255
Xs_valid = X_valid / 255

# Start training with a base model
np.random.seed(1)
tf.random.set_seed(1)

base_model = tf.keras.applications.VGG16(
    input_shape=(48, 48, 1),
    include_top=False,
    weights=None  # 'imagenet'
)

base_model.trainable = False

# tf.keras.utils.plot_model(base_model, show_shapes=True)
base_model.summary()

# Create a new model
np.random.seed(1)
tf.random.set_seed(1)

cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.65),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    BatchNormalization(),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(128, activation='relu'),
    Dropout(0.65),
    Dense(64, activation='relu'),
    Dropout(0.25),
    BatchNormalization(),
    Dense(7, activation='softmax')
])

cnn.summary()

# Train the model
opt = tf.keras.optimizers.Adam(0.001)
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Trainings
# Training n°1
h1 = cnn.fit(
    Xs_train, y_train,
    batch_size=32,
    epochs=100,
    verbose=1,
    validation_data=(Xs_valid, y_valid)
)

history = h1.history
print(history.keys())

epoch_range = range(1, len(history['loss']) + 1)

plt.figure(figsize=[14, 4])
plt.subplot(1, 2, 1)
plt.plot(epoch_range, history['loss'], label='Training')
plt.plot(epoch_range, history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epoch_range, history['accuracy'], label='Training')
plt.plot(epoch_range, history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Training n°2
tf.keras.backend.set_value(cnn.optimizer.learning_rate, 0.0001)

h2 = cnn.fit(
    Xs_train, y_train,
    batch_size=32,
    epochs=40,
    verbose=1,
    validation_data=(Xs_valid, y_valid)
)

for k in history.keys():
    history[k] += h2.history[k]

epoch_range = range(1, len(history['loss']) + 1)

plt.figure(figsize=[14, 4])
plt.subplot(1, 2, 1)
plt.plot(epoch_range, history['loss'], label='Training')
plt.plot(epoch_range, history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epoch_range, history['accuracy'], label='Training')
plt.plot(epoch_range, history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Training n°3
tf.keras.backend.set_value(cnn.optimizer.learning_rate, 0.00001)

h3 = cnn.fit(
    Xs_train, y_train,
    batch_size=32,
    epochs=200,
    verbose=1,
    validation_data=(Xs_valid, y_valid)
)

for k in history.keys():
    history[k] += h3.history[k]

epoch_range = range(1, len(history['loss']) + 1)

plt.figure(figsize=[14, 4])
plt.subplot(1, 2, 1)
plt.plot(epoch_range, history['loss'], label='Training')
plt.plot(epoch_range, history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epoch_range, history['accuracy'], label='Training')
plt.plot(epoch_range, history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Plot a confusion matrix
y_pred = cnn.predict(Xs_valid)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_valid, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=emotions, yticklabels=emotions)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save model
cnn.save('model_export/facial_recognition_model.h5')