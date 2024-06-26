
import os
from flask import Flask, request, render_template, send_from_directory,redirect
import numpy as np
from PIL import Image

# Load your deep learning model here
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assuming you're using TensorFlow
model = tf.keras.models.load_model('Double-U-Net.h5')

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def process_image(filepath):
    # Preprocess the image
    input_image = preprocess_image(filepath)
    
    # Make predictions
    processed_image = model.predict(input_image)
    
    # Post-process the output if necessary (e.g., convert to image format)
    processed_image = np.squeeze(processed_image, axis=0)  # Remove batch dimension
    processed_image = (processed_image * 255).astype(np.uint8)  # Convert back to [0, 255]
    processed_image = Image.fromarray(processed_image)

    # Save the processed image
    enhanced_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
    processed_image.save(enhanced_image_path)
    return enhanced_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        processed_image_path = process_image(filepath)
        return send_from_directory(directory=PROCESSED_FOLDER, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)















# from flask import Flask,request, url_for, redirect, render_template, send_file
# import pickle
# import os
# import numpy as np
# from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, concatenate, MaxPooling2D, Cropping2D, Activation, UpSampling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import numpy as np
# import cv2
# import os 
# from PIL import Image
# import io
# import base64

# # UPLOAD_FOLDER = 'uploads/'
# # PROCESSED_FOLDER = 'processed/'

# # if not os.path.exists(UPLOAD_FOLDER):
# #     os.makedirs(UPLOAD_FOLDER)

# # if not os.path.exists(PROCESSED_FOLDER):
# #     os.makedirs(PROCESSED_FOLDER)


# # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# app = Flask(__name__)

# # Load the pre-trained model from file
# model = tf.keras.models.load_model('Double-U-Net.h5')

# # Function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((224, 224))  # Resize image to the input size expected by your model
#     image = np.array(image) / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Function to postprocess the output image
# def postprocess_image(output):
#     output = np.squeeze(output, axis=0)  # Remove batch dimension
#     output = (output * 255).astype(np.uint8)  # Denormalize pixel values
#     output_image = Image.fromarray(output)
#     return output_image

# @app.route('/')
# def hello_world():
#     return render_template("index.html")

# # model=pickle.load(open('LogiModel.pkl','rb'))
# # Loaded the ML Model in the app.py file    

# @app.route('/upload',methods=['POST','GET'])
# def predict():
#     file = request.files['image']
#     image = Image.open(file.stream).convert('RGB')
    
#     # Preprocess the input image
#     input_image = preprocess_image(image)

#     # Perform prediction
#     output_image_array = model.predict(input_image)
    
#     # Postprocess the output image
#     output_image = postprocess_image(output_image_array)

#     # Save the processed image to a bytes buffer
#     buffer = io.BytesIO()
#     output_image.save(buffer, format='PNG')
#     buffer.seek(0)

#     img_str = base64.b64encode(buffer.getvalue()).decode()

#     return render_template('index.html', img_data=img_str)


# if __name__ == '__main__':
#     app.run(debug=True)


