from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import os

# Inisialisasi Flask App
app = Flask(__name__)

# Direktori untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan direktori untuk upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Muat model ML yang telah dilatih
model = load_model('model/best_model_EfficientNetB1.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


def prepare_image(image_path, target_size):
    # Membuka dan memproses gambar menggunakan Pillow
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            "status": {
                "code": 400,
                "message": "No file part"
            }
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            "status": {
                "code": 400,
                "message": "No selected file"
            }
        }), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        processed_image = prepare_image(file_path, target_size=(224, 224))

        # Melakukan prediksi menggunakan model
        predictions = model.predict(processed_image)
        class_idx = np.argmax(predictions, axis=1)
        confidence = np.max(predictions, axis=1)[0]

        # Mendefinisikan label kelas
        labels = {0: 'Human', 1: 'AI'}
        class_name = labels[class_idx[0]]

        # Opsional: Hapus file setelah diproses
        os.remove(file_path)

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting"
            },
            "data": {
                "image_prediction": f"Probably {class_name}",
                "confidence": float(confidence)
            }
        }), 200

    else:
        return jsonify({
            "status": {
                "code": 400,
                "message": "Invalid file format"
            }
        }), 400


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
