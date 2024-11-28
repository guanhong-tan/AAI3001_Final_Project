import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Load data
print("Loading data...")
X_train = np.load('C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/X_train.npy')
X_test = np.load('C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/X_test.npy')
y_train = np.load('C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/Y_train.npy')
y_test = np.load('C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/Y_test.npy')

# Data Augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data_gen = data_gen.flow(X_train, y_train, batch_size=32)

# Load pre-trained MobileNetV2
print("\nBuilding model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(np.unique(y_train)), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model
print("\nTraining model...")
history = model.fit(
    train_data_gen,
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Save the final model
model.save('C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model/final_model.keras')
print("Training completed successfully!")

# Plot training and validation accuracy
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_accuracy) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', linestyle='--')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss', linestyle='--')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()