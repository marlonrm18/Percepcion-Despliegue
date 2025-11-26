import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify, render_template
import io
from PIL import Image
import base64

app = Flask(__name__)

# Cargar el modelo
model = None
def load_model():
    try:
        # VERIFICA EL NOMBRE EXACTO DE TU MODELO
        model = tf.keras.models.load_model('modelo_meat_quality_final.h5')
        print("✅ Modelo cargado exitosamente")
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        return None

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar modelo
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 503
        
    try:
        # Verificar imagen
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió imagen'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        # Procesar imagen
        img = Image.open(file.stream)
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predicción
        prediction = model.predict(img_array, verbose=0)
        prob_spoiled = float(prediction[0][0])
        prob_fresh = 1 - prob_spoiled
        
        # Resultado
        if prob_spoiled > 0.5:
            result = "DAÑADA"
            confidence = prob_spoiled * 100
            color = "red"
        else:
            result = "FRESCA"
            confidence = prob_fresh * 100
            color = "green"
        
        # Imagen a base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'result': result,
            'confidence': round(confidence, 2),
            'color': color,
            'image': f"data:image/jpeg;base64,{img_str}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Error procesando imagen: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)