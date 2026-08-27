AIM

To build and train a Convolutional Neural Network (CNN) using TensorFlow/Keras for classifying images from the CIFAR-10 dataset.

PROCEDURE
Import the required Python libraries such as TensorFlow, Keras, NumPy, and Matplotlib.

Load the CIFAR-10 dataset containing 50,000 training images and 10,000 testing images.

Normalize the image pixel values to the range 0 to 1.

Convert the class labels into one-hot encoded vectors.

Visualize sample images from the training dataset.

Build a CNN model using convolutional, max-pooling, dropout, flatten, and dense layers.

Compile the CNN using the Adam optimizer, categorical cross-entropy loss, and accuracy metric.

Train the model using the training dataset for 30 epochs with a batch size of 64.

Plot the training and validation accuracy and loss to analyze model performance.

Evaluate the trained model using the test dataset.

Predict the classes of test images and compare the predicted labels with the actual labels.

CODE

# Ex.No.5 - Building and Training a CNN




import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np




(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()




X_train = X_train.astype('float32') / 255.0

X_test = X_test.astype('float32') / 255.0


y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)




class_names = [

    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]


plt.figure(figsize=(10, 10))


for i in range(16):

    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[np.argmax(y_train[i])])

plt.show()




model = models.Sequential()


model.add(layers.Conv2D(

    32, (3, 3), activation='relu',
    padding='same', input_shape=(32, 32, 3)
))


model.add(layers.Conv2D(

    32, (3, 3), activation='relu',
    padding='same'
))


model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))


model.add(layers.Conv2D(

    64, (3, 3), activation='relu',
    padding='same'
))


model.add(layers.Conv2D(

    64, (3, 3), activation='relu',
    padding='same'
))


model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))


model.add(layers.Conv2D(

    128, (3, 3), activation='relu',
    padding='same'
))


model.add(layers.Conv2D(

    128, (3, 3), activation='relu',
    padding='same'
))


model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))


model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))




model.compile(

    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()




history = model.fit(

    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)





plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)


plt.plot(

    history.history['accuracy'],
    label='Train Accuracy'
)


plt.plot(

    history.history['val_accuracy'],
    label='Validation Accuracy'
)


plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()


plt.subplot(1, 2, 2)


plt.plot(

    history.history['loss'],
    label='Train Loss'
)


plt.plot(

    history.history['val_loss'],
    label='Validation Loss'
)


plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()


plt.show()




test_loss, test_accuracy = model.evaluate(

    X_test,
    y_test,
    verbose=2
)


print("Test Accuracy:", test_accuracy)





def plot_predictions(index):


    img = X_test[index]

    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)
    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )
    plt.axis('off')
    plt.show()


for i in range(5):

    plot_predictions(i)
RESULT


Thus, a Convolutional Neural Network was successfully built and trained using TensorFlow/Keras on the CIFAR-10 dataset. The model learned to classify images into 10 different categories: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. The training and validation performance were visualized using accuracy and loss graphs, and the trained model was used to predict test images successfully.












