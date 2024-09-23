# %% [markdown]
# # ***MalarIA***

# %% [markdown]
# # ***Proyecto TIC 2024***

# %% [markdown]
# # **Librerias**

# %%
# Estas líneas de código nos permiten instalar opencv desde Jupyter Notebook, que vamos a necesitar para importar 'cv2'
import tensorflow as tf

devices = tf.config.list_physical_devices('GPU')

print(devices)

if len(devices) == 0:
   raise ValueError()



import sys


# %%
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import seed

import datetime

import os
import PIL

import shutil
import random
SEED = 456
random.seed(SEED)

import math

import pandas as pd

import argparse

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# %%
tf.random.set_seed(456)
from tensorflow import keras


# %% [markdown]
# # **Set de rutas de imágenes**

# %%
print(os.getcwd())

# %%
# Rutas absolutas
path_todas = "imagenes_todas"
path_sanos = "./imagenes_todas/Uninfected"
path_infectados = "./imagenes_todas/Parasitized"



# %%
tf.config.list_physical_devices('GPU')

# %%

# %%
lista_sanos = os.listdir(path_sanos)
lista_infectados = os.listdir(path_infectados)

# %% [markdown]
# # **Creación de listas y revisación de las mismas**

# %%
if not os.path.exists(path_sanos):
    print(f"La carpeta '{path_sanos}' no existe.")
else :
  lista_sanos = os.listdir(path_sanos)

if not os.path.exists(path_infectados):
     print(f"La carpeta '{path_infectados}' no existe.")
else :
  lista_infectados = os.listdir(path_infectados)

# %%
for item in lista_sanos:
  lista_archivo = item.split('.')
  if lista_archivo[1] != 'png':
    print ('Carpeta de sanos:' , item)

# %%
for item in lista_infectados:
  lista_archivo = item.split('.')
  if lista_archivo[1] != 'png':
    print ('Carpeta de infectados:' , item)

# %%
print('Número de imágenes correspondientes a pacientes sanos:', len(lista_sanos))

print('Número de imágenes correspondientes a pacientes infectados:', len(lista_infectados))

# %% [markdown]
# # **Plot de 10 imágenes de cada clase**

# %%
# Crear una figura para las imágenes
plt.figure(figsize=(24, 14))

# Mostrar las imágenes
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Crear una subfigura en la posición i+1
    image_path = os.path.join(path_sanos, lista_sanos[i])  # Ruta completa de la imagen
    image = plt.imread(image_path)  # Leer la imagen
    plt.imshow(image)  # Mostrar la imagen
    plt.axis('off')  # Opcional: ocultar los ejes

# Ajustar el diseño para evitar superposición
plt.tight_layout()
plt.show()

# %%
# Crear una figura para las imágenes
plt.figure(figsize=(24,14))

# Mostrar las imágenes
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Crear una subfigura en la posición i+1
    image_path = os.path.join(path_infectados, lista_infectados[i])  # Ruta completa de la imagen
    image = plt.imread(image_path)  # Leer la imagen
    plt.imshow(image)  # Mostrar la imagen
    plt.axis('off')  # Opcional: ocultar los ejes

# Ajustar el diseño para evitar superposición
plt.tight_layout()
plt.show()

# %% [markdown]
# # **Dataframe**

# %%
df_sano = pd.DataFrame(lista_sanos, columns=['image_id'])
# El "Thumbs.db" elimina cosas que no tienen relevancia en el codigo y que pueden interferir en su función.
df_sano = df_sano[df_sano['image_id'] != 'Thumbs.db']
# Añadimos una columna objetivo o 'target', esto marca con un 0 los sanos y con un 1 los infectados
df_sano['target'] = 0
# Repetimos el proceso para los infectados

df_infectados = pd.DataFrame(lista_infectados, columns=['image_id'])
df_infectados = df_infectados[df_infectados['image_id'] != 'Thumbs.db']
df_infectados['target'] = 1
# Creamos otro dataframe que combine los dos anteriores
df_combinado = pd.concat([df_sano, df_infectados], axis=0).reset_index(drop=True)

df_combinado

# %%
# ver cuantas imagenes hay de cada tipo
df_combinado['target'].value_counts()

# %%
df_combinado.info()

# %% [markdown]
# # **Seteamos DS y variables globales**

# %%
batch_size = 32
img_height = 100
img_width = 100

data_dir = r'/content/drive/MyDrive/imagenes_todas'

# %% [markdown]
# Creamos el dataset de train

# %%
with tf.device('/device:GPU:0'):
  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# %% [markdown]
# Creamos el dataset de validation

# %%
with tf.device('/device:GPU:0'):
  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# %% [markdown]
# Creamos el dataset general

# %%
with tf.device('/device:GPU:0'):
  ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      labels='inferred',
      # label_mode='int',
      # class_names=None,
      color_mode='rgb',
      batch_size=32,
      image_size=(img_height, img_width),
      shuffle=True,
      seed=SEED,
      # validation_split=None,
      # subset=None,
      # interpolation='bilinear',
      # follow_links=False,
      # crop_to_aspect_ratio=False,
      # pad_to_aspect_ratio=False,
      # data_format=None,
  )

# %% [markdown]
# Separación del ds entre los ds de train, validation y test

# %%
def get_dataset_partition_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert train_split + val_split + test_split == 1.0
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=SEED)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size + val_size)
    return train_ds, test_ds, val_ds

# %%
with tf.device('/device:GPU:0'):
  train_ds, test_ds, val_ds = get_dataset_partition_tf(ds, shuffle_size=10000)

# %% [markdown]
# # **Crear el modelo**

# %% [markdown]
# Estructura del modelo

# %%
with tf.device('/device:GPU:0'):
  num_classes = 2

  modelo = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(8, 3, padding='same', name='conv_layer_1', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 3, padding='same', name='conv_layer_2', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
  ])

# %% [markdown]
# Complilar el modelo

# %%
with tf.device('/device:GPU:0'):
  modelo.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

# %% [markdown]
# Resumen del modelo

# %%
modelo.summary()

# %% [markdown]
# Entrenar el modelo

# %%

# %%
with tf.device('/device:GPU:0'):
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# %%
with tf.device('/device:GPU:0'):
  history = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[tensorboard_callback]
  )

# %% [markdown]
# Ver resultados del modelo

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 16))
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

# %%
modelo.save('mi_modelo.h5')

# %%
model = tf.keras.models.load_model('mi_modelo.h5')

# %%
model.summary()

# %%
img, lbl = next(iter(val_ds))

# %%
model.fit(val_ds)


