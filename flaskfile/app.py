from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/muthu/flaskfile/new.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = '/flaskfile/static/uploads/'

db = SQLAlchemy(app)
custom_weights_file = "/home/muthu/flaskfile/best.pt"
model = YOLO(custom_weights_file)
confidence_thresholds = {'0': 0.7, '1': 0.8, '2': 0.7}

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        image_binary = file.read()
        new_image = Image(image_data=image_binary)
        db.session.add(new_image)
        db.session.commit()
        return redirect(url_for('predict'))

from flask import request, redirect, url_for
@app.route('/predict')
def predict():
    annotated_image_path = None
    latest_image = Image.query.order_by(Image.id.desc()).first()
    if latest_image:
        image_array = np.frombuffer(latest_image.image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        results = model(image)
        detections = results[0] if isinstance(results, list) else results

        # Define a dictionary mapping class labels to class names
        class_names = {
            0: "Alternaria",
            1: "Anthracnose",
            2: "Black Mold Rot",
            3: "Healthy",
            4: "Stem End Rot"
        }

        # Modify the part of your code where predictions are generated
        predictions = []
        for i, det in enumerate(detections.boxes.xyxy[0]):
            if len(det.shape) == 0:
                continue

            class_label = int(det[-1]) if det.shape[-1] > 0 else None
            confidence = det[-2] if det.shape[-1] > 1 else None

            if class_label is not None and confidence is not None:
                if confidence > confidence_thresholds.get(str(class_label), 0):
                    # Look up the class name using the dictionary
                    class_name = class_names.get(class_label, "Unknown")
                    predictions.append(class_name)

        
        # Save annotated image
        annotated_image = results[0].plot()
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "annotated_image.jpg")
        cv2.imwrite(annotated_image_path, annotated_image)

    return render_template('result.html', predictions=predictions, annotated_image_path=annotated_image_path)


if __name__ == '__main__':
    app.run(debug=True)
