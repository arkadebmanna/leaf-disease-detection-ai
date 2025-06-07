from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('../model/leaf_model.h5')

class_names = ['Healthy', 'Leaf Spot', 'Powdery Mildew', 'Rust', 'Blight']  # Update as per your model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    uploaded_file = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            uploaded_file = file.filename

            img = preprocess_image(filepath)
            pred = model.predict(img)
            prediction_idx = np.argmax(pred)
            prediction = class_names[prediction_idx]
            confidence = round(np.max(pred) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence, image=uploaded_file)

if __name__ == '__main__':
    app.run(debug=True)
