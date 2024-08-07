# -*- coding: utf-8 -*-
"""A Comparative Study of Deep Learning Models for COVID-19 X-ray Classification

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uakMVZc_N_Z5EHJfHBN3x1NoY9JxCz5R

### Importing Libararies
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import EfficientNetB3
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import cv2

from warnings import filterwarnings
filterwarnings('ignore')

"""### Importing the training and test sets"""

from google.colab import drive
drive.mount('/content/drive')

train_dir = "/content/drive/MyDrive/ba/xray_dataset_covid19/train"
test_dir = "/content/drive/MyDrive/ba/xray_dataset_covid19/test"

classes_train = os.listdir(train_dir)
classes_test = os.listdir(test_dir)

"""#### Function that will be used to predict some images from the test dataset"""

def plot_prediction(test_generator, n_images, model):
    i = 1
    images, labels = test_generator.next()
    predictions = np.argmax(model.predict(images), axis=1)
    labels = labels.astype('int32')

    plt.figure(figsize=(12, 16))

    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)

        if predictions[i] == labels[i]:
            title_obj = plt.title(classes[label])
            plt.setp(title_obj, color='g')
            plt.axis('off')
        else:
            title_obj = plt.title(classes[label])
            plt.setp(title_obj, color='r')
            plt.axis('off')
        i += 1

        if i == n_images+1:
            break

    plt.show()

"""### Data Analysis"""

train_dict = {}
test_dict = {}

for c in classes_train:
    train_dict[c] = len(os.listdir(os.path.join(train_dir, c)))

for c in classes_test:
    test_dict[c] = len(os.listdir(os.path.join(test_dir, c)))

classes = ['PNEUMONIA', 'NORMAL']

train_values = [train_dict[class_name] for class_name in classes]
test_values = [test_dict[class_name] for class_name in classes]

plt.bar(range(len(classes)), train_values, width=0.5, label='Training')
plt.bar(range(len(classes)), test_values, width=0.5, bottom=train_values, label='Test')

plt.xlabel('Classes')
plt.ylabel('Number of Images')
plt.title('Images x Classes')
plt.xticks(range(len(classes)), classes)
plt.legend()
plt.show()

image = cv2.imread(f"{train_dir}/PNEUMONIA/CD50BA96-6982-4C80-AE7B-5F67ACDBFA56.jpeg")

plt.figure(figsize=(2, 5))
plt.imshow(image)
plt.title("PNEUMONIA Image")
plt.show()

image = cv2.imread(f"{train_dir}/NORMAL/IM-0007-0001.jpeg")

plt.figure(figsize=(2, 5))
plt.imshow(image)
plt.title("NORMAL Image")
plt.show()

"""Here, we use ImageDataGenerator to make some tweaks to our training dataset, adding some variety to the data so that our model can learn even better.

We will set the number of mini-batches (32), which is the batch size per iteration during model training.
"""

# Training
train_datagen = ImageDataGenerator(
    zoom_range = 0.1,
    horizontal_flip = True,
    rescale = 1.0/255.0,
    width_shift_range = 0.10,
    height_shift_range = 0.10,
    shear_range = 0.1,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode = 'binary',
    color_mode = 'rgb',
    batch_size = 16,
    target_size = (224, 224)
)

batch_images, batch_labels = next(train_generator)

fig, axes = plt.subplots(4, 4, figsize=(16, 8))

axes = axes.flatten()

for i in range(16):
    axes[i].imshow(batch_images[i])
    axes[i].set_title("TRAIN Generated")

for ax in axes:
    ax.axis('off')

plt.show()

# Test
test_datagen = ImageDataGenerator(
    rescale = 1.0/255.0,
)

test_generator = train_datagen.flow_from_directory(
    test_dir,
    class_mode = 'binary',
    color_mode = 'rgb',
    batch_size = 16,
    target_size=(224, 224)
)

batch_images, batch_labels = next(test_generator)

fig, axes = plt.subplots(4, 4, figsize=(16, 8))

axes = axes.flatten()

for i in range(16):
    axes[i].imshow(batch_images[i])
    axes[i].set_title("TEST Generated")

for ax in axes:
    ax.axis('off')

plt.show()
# Here, we just changed the scales of our test images

"""### Callbacks
#### Callbacks are functions we give to our model during training to make our lives easier and use less processing.
- Here, for example, we use EarlyStopping with patience=5 and monitor='val_loss', that means when 5 consecutive epochs have the same val_loss value, our model stops the traing right there.
- We also use ReduceLROnPlateau (Reduce Learning Rate On Plateau), it means that when the model stops improving during 3 consecutive times, the learning rate will decrease 0.5, but not ultrapassing the minimum that is 0.00001.
"""

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_acc',
    patience= 3,
    verbose = 1,
    factor = 0.5,
    min_lr = 0.00001
)

callbacks_list = [early_stop, learning_rate_reduction]

"""### Model 1: Convolutional Neural Networks"""

model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dense(128, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model1.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

test_loss, test_acc = model1.evaluate(test_generator)

print(f'Test accuracy: {test_acc}')
print(f'Test Loss: {test_loss}')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model1)

"""### Model 2: VGG

#### Loading pre-trained VGG model (excluding the top layer)
"""

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

"""#### Freeze the layers in the pre-trained model"""

for layer in vgg_model.layers:
    layer.trainable = False

"""#### Creating a new model on top of the pre-trained model"""

model_vgg = models.Sequential()
model_vgg.add(vgg_model)
model_vgg.add(layers.Flatten())
model_vgg.add(layers.Dense(128, activation='relu'))
model_vgg.add(layers.Dense(1, activation='sigmoid'))

model_vgg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_vgg = model_vgg.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

vgg_test_loss, vgg_test_acc = model_vgg.evaluate(test_generator)

print(f'VGG Test accuracy: {vgg_test_acc}')

plt.plot(history_vgg.history['accuracy'], label='accuracy')
plt.plot(history_vgg.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_vgg)

"""### Model 3: Xception

#### Loading pre-trained Xception model (excluding the top layer)
"""

xception_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in xception_model.layers:
    layer.trainable = False

model_xception = models.Sequential()
model_xception.add(xception_model)
model_xception.add(layers.GlobalAveragePooling2D())
model_xception.add(layers.Dense(128, activation='relu'))
model_xception.add(layers.Dense(1, activation='sigmoid'))

model_xception.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_xception = model_xception.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

xception_test_loss, xception_test_acc = model_xception.evaluate(test_generator)

print(f'Xception Test accuracy: {xception_test_acc}')

plt.plot(history_xception.history['accuracy'], label='accuracy')
plt.plot(history_xception.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_xception)

"""### Model 4: ResNet50"""

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in resnet_model.layers:
    layer.trainable = False

model_resnet = models.Sequential()
model_resnet.add(resnet_model)
model_resnet.add(layers.GlobalAveragePooling2D())
model_resnet.add(layers.Dense(128, activation='relu'))
model_resnet.add(layers.Dense(1, activation='sigmoid'))

model_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_resnet = model_resnet.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

resnet_test_loss, resnet_test_acc = model_resnet.evaluate(test_generator)

print(f'ResNet Test accuracy: {resnet_test_acc}')

plt.plot(history_resnet.history['accuracy'], label='accuracy')
plt.plot(history_resnet.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_resnet)

"""### Model 5: InceptionV3"""

inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in inception_model.layers:
    layer.trainable = False

model_inception = models.Sequential()
model_inception.add(inception_model)
model_inception.add(layers.GlobalAveragePooling2D())
model_inception.add(layers.Dense(128, activation='relu'))
model_inception.add(layers.Dense(1, activation='sigmoid'))

model_inception.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_inception = model_inception.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

inception_test_loss, inception_test_acc = model_inception.evaluate(test_generator)

print(f'InceptionV3 Test accuracy: {inception_test_acc}')

plt.plot(history_inception.history['accuracy'], label='accuracy')
plt.plot(history_inception.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_inception)

plot_prediction(test_generator, 5, model_inception)

"""### Model 6: InceptionResNet"""

inception_resnet_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in inception_resnet_model.layers:
    layer.trainable = False

model_inception_resnet = models.Sequential()
model_inception_resnet.add(inception_resnet_model)
model_inception_resnet.add(layers.GlobalAveragePooling2D())
model_inception_resnet.add(layers.Dense(128, activation='relu'))
model_inception_resnet.add(layers.Dense(1, activation='sigmoid'))

model_inception_resnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_inception_resnet = model_inception_resnet.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

inception_resnet_test_loss, inception_resnet_test_acc = model_inception_resnet.evaluate(test_generator)

print(f'InceptionResNetV2 Test accuracy: {inception_resnet_test_acc}')

plt.plot(history_inception_resnet.history['accuracy'], label='accuracy')
plt.plot(history_inception_resnet.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_inception_resnet)

"""### Model 7: DenseNet"""

densenet_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in densenet_model.layers:
    layer.trainable = False

model_densenet = models.Sequential()
model_densenet.add(densenet_model)
model_densenet.add(layers.GlobalAveragePooling2D())
model_densenet.add(layers.Dense(128, activation='relu'))
model_densenet.add(layers.Dense(1, activation='sigmoid'))

model_densenet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_densenet = model_densenet.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

densenet_test_loss, densenet_test_acc = model_densenet.evaluate(test_generator)

print(f'DenseNet Test accuracy: {densenet_test_acc}')

plt.plot(history_densenet.history['accuracy'], label='accuracy')
plt.plot(history_densenet.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_densenet)

"""### Model 8: EfficientNet"""

efficientnet_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in efficientnet_model.layers:
    layer.trainable = False

model_efficientnet = models.Sequential()
model_efficientnet.add(efficientnet_model)
model_efficientnet.add(layers.GlobalAveragePooling2D())
model_efficientnet.add(layers.Dense(128, activation='relu'))
model_efficientnet.add(layers.Dense(1, activation='sigmoid'))

model_efficientnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_efficientnet = model_efficientnet.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

efficientnet_test_loss, efficientnet_test_acc = model_efficientnet.evaluate(test_generator)

print(f'EfficientNet Test accuracy: {efficientnet_test_acc}')

plt.plot(history_efficientnet.history['accuracy'], label='accuracy')
plt.plot(history_efficientnet.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plot_prediction(test_generator, 5, model_efficientnet)

"""### Model 9: Support Vector Machine"""

X_train_svm = np.array([train_generator[i][0][j] for i in range(len(train_generator)) for j in range(len(train_generator[i][0]))])
y_train_svm = np.array([train_generator[i][1][j] for i in range(len(train_generator)) for j in range(len(train_generator[i][1]))])

X_test_svm = np.array([test_generator[i][0][j] for i in range(len(test_generator)) for j in range(len(test_generator[i][0]))])
y_test_svm = np.array([test_generator[i][1][j] for i in range(len(test_generator)) for j in range(len(test_generator[i][1]))])

X_train_svm_flat = X_train_svm.reshape(X_train_svm.shape[0], -1)
X_test_svm_flat = X_test_svm.reshape(X_test_svm.shape[0], -1)

svm_model = SVC()
svm_model.fit(X_train_svm_flat, y_train_svm)

svm_y_pred = svm_model.predict(X_test_svm_flat)
svm_acc = accuracy_score(y_test_svm, svm_y_pred)

print(f'SVM Test accuracy: {svm_acc}')

"""### Model 10: Random Forest Classifier

#### Flatten the images
"""

X_train_rf_flat = X_train_svm_flat
X_test_rf_flat = X_test_svm_flat

rf_model = RandomForestClassifier()
rf_model.fit(X_train_rf_flat, y_train_svm)

rf_y_pred = rf_model.predict(X_test_rf_flat)
rf_acc = accuracy_score(y_test_svm, rf_y_pred)

print(f'Random Forest Test accuracy: {rf_acc}')

model_names = ['CNN', 'SVM', 'Random Forest', 'VGG', 'Xception', 'ResNet', 'InceptionV3', 'InceptionResNetV2', 'DenseNet', 'EfficientNet']

test_accuracies = [test_acc, svm_acc, rf_acc, vgg_test_acc, xception_test_acc, resnet_test_acc, inception_test_acc, inception_resnet_test_acc, densenet_test_acc, efficientnet_test_acc]

plt.figure(figsize=(15, 8))
plt.bar(model_names, test_accuracies, color='blue')
plt.title('Test Accuracies of Different Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set the y-axis limit to match the accuracy range (0 to 1)
plt.show()