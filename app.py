from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os
from PIL import Image

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustado a tu carpeta 'models'
MODEL_PATH = os.path.join("models", "mobilenet_model.h5")
class_names = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

# Cargar el modelo al iniciar (asegúrate de que el archivo existe ahí)
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Modelo cargado desde: {MODEL_PATH}")
else:
    print(f"❌ ERROR: No se encuentra el modelo en {MODEL_PATH}")


def prepare_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización idéntica a tu entrenamiento
    return img_array


# Ruta para servir el HTML principal
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay archivo'}), 400

    file = request.files['file']
    img_bytes = file.read()

    # Abrir imagen con PIL
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Preprocesar y predecir
    prepared_img = prepare_image(img)
    predictions = model.predict(prepared_img)[0]

    pred_class = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return jsonify({
        'class': class_names[pred_class],
        'confidence': confidence,
        'all_predictions': predictions.tolist()
    })


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')

    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Servidor corriendo en el puerto {port}")
    app.run(host='0.0.0.0', port=port)