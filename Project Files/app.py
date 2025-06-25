from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/rice_model.h5')
class_labels = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Red']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        return render_template('result.html', prediction=predicted_class, img_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)