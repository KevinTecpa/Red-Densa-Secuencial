

# mnist_Dense.py
# Red Densa Secuencial para clasificación de dígitos MNIST

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


# Inicializar experimento en Weights & Biases

wandb.init(
    project="mnist-dense",          # Nombre del proyecto en wandb
    name="experimento_2",    # Nombre de la corrida
    config={
        "epochs": 10,
        "batch_size": 32,
        "optimizer": "sgd",
        "loss": "categorical_crossentropy",
        "architecture": "Dense(30, relu) -> Dense(10, softmax)"
    }
)


# Cargar y preparar los datos MNIST

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar imágenes de 0-255 a 0-1
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# One-hot encoding de las etiquetas
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)


# Definir el modelo secuencial denso

model = keras.Sequential([
    layers.Input(shape=(784,)),       # capa de entrada
    layers.Dense(60, activation="relu"),
    Dropout(0.2),
    layers.Dense(10, activation="softmax") # capa de salida (10 clases)
])

# Compilar el modelo
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),  # SGD como en el código original
    loss="categorical_crossentropy",                    # función de costo
    metrics=["accuracy"]
)


# Entrenamiento del modelo con callbacks de wandb
callbacks = [
    WandbMetricsLogger(log_freq="epoch"),
    WandbModelCheckpoint("models/mnist_dense.keras")  # guarda modelo en formato Keras 3
]


history = model.fit(
    x_train, y_train_cat,
    epochs=wandb.config.epochs,
    batch_size=wandb.config.batch_size,
    validation_data=(x_test, y_test_cat),
    callbacks=callbacks
)


# Evaluar el modelo final
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Guardar el modelo entrenado en wandb
model.save("mnist_dense_final.h5")
wandb.save("mnist_dense_final.h5")


