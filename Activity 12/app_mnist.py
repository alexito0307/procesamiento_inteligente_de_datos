import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# -------------------------------------------------
# 1. Configuracion de logging
# -------------------------------------------------
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# -------------------------------------------------
# 2. Cargar dataset MNIST
# -------------------------------------------------
print("Cargando dataset MNIST...")

dataset, metadata = tfds.load(
    'mnist',
    as_supervised=True,
    with_info=True
)

train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = [
    'Cero', 'Uno', 'Dos', 'Tres', 'Cuatro',
    'Cinco', 'Seis', 'Siete', 'Ocho', 'Nueve'
]

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

# -------------------------------------------------
# 3. Normalizacion
# -------------------------------------------------
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize).cache()
test_dataset = test_dataset.map(normalize).cache()

# -------------------------------------------------
# 4. Modelo
# -------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------------------------
# 5. Preparar batches
# -------------------------------------------------
BATCHSIZE = 32

train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCHSIZE)
test_dataset = test_dataset.batch(BATCHSIZE)

# -------------------------------------------------
# 6. Entrenamiento
# -------------------------------------------------
print("Entrenando modelo...")
model.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=math.ceil(num_train_examples / BATCHSIZE)
)

# -------------------------------------------------
# 7. Evaluacion
# -------------------------------------------------
test_loss, test_accuracy = model.evaluate(
    test_dataset,
    steps=math.ceil(num_test_examples / BATCHSIZE)
)

print("Resultado en las pruebas:", test_accuracy)

# -------------------------------------------------
# 8. Servidor HTTP
# -------------------------------------------------
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        print("Peticion recibida")

        try:
            # Leer datos del body
            content_length = int(self.headers['Content-Length'])
            data = self.rfile.read(content_length)
            data = data.decode()

            # Limpiar "pixeles="
            data = data.replace('pixeles=', '')
            data = parse.unquote(data)

            # Convertir a arreglo
            arr = np.fromstring(data, np.float32, sep=",")

            if arr.size != 784:
                raise ValueError(f"Se esperaban 784 pixeles y llegaron {arr.size}")

            arr = arr.reshape(28, 28)
            arr = np.array(arr, dtype=np.float32)
            arr = arr.reshape(1, 28, 28, 1)

            # Prediccion
            prediction_values = model.predict(arr, batch_size=1, verbose=0)
            prediction_index = int(np.argmax(prediction_values))
            prediction_name = class_names[prediction_index]

            print("Prediccion final:", prediction_index, "-", prediction_name)

            # Respuesta
            response = f"{prediction_index} ({prediction_name})"

            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))

        except Exception as e:
            print("Error:", str(e))
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode("utf-8"))

# -------------------------------------------------
# 9. Iniciar servidor
# -------------------------------------------------
print("Iniciando servidor en http://localhost:8000 ...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()