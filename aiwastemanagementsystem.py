import os  # Provides a way of using operating system dependent functionality like reading or writing to the file system.
import zipfile  # Used to handle ZIP files, including extracting files.
import tensorflow as tf  # Main library for deep learning.
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation and preprocessing.
from tensorflow.keras.applications import MobileNetV2  # Pre-trained MobileNetV2 model.
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Layers to add to the model.
from tensorflow.keras.models import Model  # Model class to create a Keras model.
from tensorflow.keras.optimizers import Adam  # Optimizer for training the model.
import cv2  # OpenCV library for computer vision tasks.
import numpy as np  # Library for numerical operations.
from tensorflow.keras.models import load_model  # Function to load a saved Keras model.

# Path to your dataset directory
# Update the following line with the actual path to your dataset directory.
dataset_path = '/Users/sairaghavganesh/saiProjects/dataset-resized'

# Data preprocessing
# ImageDataGenerator helps with preprocessing the image data, including rescaling the pixel values.
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Rescale the images and set aside a validation subset.

# Create a data generator for the training data
train_generator = datagen.flow_from_directory(
    dataset_path,  # Directory where the dataset is stored.
    target_size=(224, 224),  # Resize all images to 224x224 pixels.
    batch_size=32,  # Number of images to be yielded from the generator per batch.
    class_mode='categorical',  # Classification mode - categorical for multi-class classification.
    subset='training'  # Set this generator to use the training subset.
)

# Create a data generator for the validation data
validation_generator = datagen.flow_from_directory(
    dataset_path,  # Directory where the dataset is stored.
    target_size=(224, 224),  # Resize all images to 224x224 pixels.
    batch_size=32,  # Number of images to be yielded from the generator per batch.
    class_mode='categorical',  # Classification mode - categorical for multi-class classification.
    subset='validation'  # Set this generator to use the validation subset.
)

# Model building
# Load the base model with pre-trained weights
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Use the MobileNetV2 model without the top layer.
x = base_model.output  # Get the output of the base model.
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer to reduce the dimensionality.
x = Dense(1024, activation='relu')(x)  # Add a dense layer with 1024 units and ReLU activation.
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Add the final dense layer with softmax activation for classification.

# Combine the base model and the new layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
# Train the model with the training data and validate with the validation data
model.fit(
    train_generator,  # Training data generator.
    epochs=10,  # Number of epochs to train the model.
    validation_data=validation_generator  # Validation data generator.
)

# Save the trained model using the new format
model.save('smart_waste_sorting_model.keras')  # Save the trained model to a file.

# Labels corresponding to the dataset classes
labels = {v: k for k, v in train_generator.class_indices.items()}  # Get the class labels from the training generator.

# Advice dictionary
advice = {
    "cardboard": "This is cardboard. Please recycle it in the appropriate bin.",
    "glass": "This is glass. Please recycle it in the glass recycling bin.",
    "metal": "This is metal. Please recycle it in the metal recycling bin.",
    "paper": "This is paper. Please recycle it in the paper recycling bin.",
    "plastic": "This is plastic. Please recycle it in the appropriate bin.",
    "trash": "This is general trash. Dispose of it in the general waste bin."
}

# Function to classify waste items and provide advice
def classify_waste(image, threshold=0.6):
    img = cv2.resize(image, (224, 224))  # Resize the image to match the input size of the model.
    img = np.expand_dims(img, axis=0) / 255.0  # Expand dimensions and rescale pixel values.
    prediction = model.predict(img)  # Predict the class of the image.
    max_prob = np.max(prediction)  # Get the highest probability from the prediction.
    if max_prob < threshold:  # Check if the highest probability is below the threshold.
        return None, None  # Return None if the prediction is not confident enough.
    class_id = np.argmax(prediction, axis=1)[0]  # Get the index of the class with the highest probability.
    class_name = labels[class_id]  # Get the class name corresponding to the index.
    return class_name, advice.get(class_name, "No advice available for this item.")  # Return the class name and advice.

# Load the trained model using the new format
model = load_model('smart_waste_sorting_model.keras')  # Load the saved model.

# Initialize the camera
cap = cv2.VideoCapture(0)  # Open the default camera.

while True:
    ret, frame = cap.read()  # Read a frame from the camera.
    if not ret:  # Break the loop if there is an issue reading the frame.
        break

    # Perform classification
    class_name, class_advice = classify_waste(frame)  # Classify the waste item in the frame.

    # Display the result and advice if classification is confident
    if class_name and class_advice:
        cv2.putText(frame, f'Class: {class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Display the class name.
        cv2.putText(frame, f'Advice: {class_advice}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)  # Display the advice.
    
    cv2.imshow('Waste Classification', frame)  # Show the frame with the classification and advice.

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if the 'q' key is pressed.
        break

cap.release()  # Release the camera.
cv2.destroyAllWindows()  # Close all OpenCV windows.
