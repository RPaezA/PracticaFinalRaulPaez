# PracticaFinalRaulPaez
Practica: obtener un clasificador de imagenes

****importamos las libreria a utilizar

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow.keras as keras
import os
import random
from keras.models import Sequential
from PIL import Image


****Configuracion del generador de imagenes tanto para test y train

train_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/raulp/Downloads/CarneDataset/train',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(300,300))

  val_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/raulp/Downloads/CarneDataset/train',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(300,300))

  test_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/raulp/Downloads/CarneDataset/test',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(300,300))

  valtest_ds = tf.keras.utils.image_dataset_from_directory(
  '/Users/raulp/Downloads/CarneDataset/test',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(300,300))

  
****asignamos a class_names los nombres de todas las categorias de carnes que constan en el directorio CarneDataset

class_names = train_ds.class_names
print(class_names)


****Importar la libreria de matplolib para graficar

import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

****Validamos el tamaño asignado a la imagen y el batch para train y test

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

for image_batch, labels_batch in test_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


****Creacion del modelo

num_classes = len(class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(300, 300, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])


****Entrenamiento del modelo con 10 epocas
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


****Graficas de exactitud y perdida del modelo entrenado

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

****Selecciono una imagen aleatoria del directroio train para realizar la prediccion

image_path = '/Users/raulp/Downloads/CarneDataset/test/CLASS_04/12-CAPTURE_20220614_124928_394.png'
image = tf.keras.preprocessing.image.load_img(image_path).resize((300,300))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.

****Realizo la prediccion

prediccion = model.predict(input_arr)

****mensaje que indica a que categoria de test pertenece la imagen

score = tf.nn.softmax(prediccion[0])
print(
    "Esta imagen parece ser {} con un {:.2f} % de exactitud."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

****Generacion de matriz de confusion

matc=confusion_matrix(y_real, y_pred)

plot_confusion_matrix(conf_mat=matc, figsize=(9,9), class_names = names, show_normed=False)
plt.tight_layout()
