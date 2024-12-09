import os
from flask import Flask, request, render_template, send_from_directory,redirect
import numpy as np
from PIL import Image
import base64

# Load your deep learning model here
import tensorflow as tf

app = Flask(__name__)
if __name__ == "__main__":

    app.run(debug=True)

model = tf.keras.models.load_model('REVIDE.h5')


# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
RESHAPED_FOLDER = 'reshaped'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESHAPED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['RESHAPED_FOLDER'] = RESHAPED_FOLDER


def preprocess_image(image_path):
    print ('preprocess started')
    image = Image.open(image_path).convert('RGB')
    image = image.resize((192, 128))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print ('preprocess ended')

    return image

def process_image(filepath):
    print ('process started')
    
    # Preprocess the image
    input_image = preprocess_image(filepath)
    
    # Make predictions
    print('will predict')
    processed_image = model.predict(input_image)
    print('predicted')

    input_image = np.squeeze(input_image, axis=0)  # Remove batch dimension
    input_image = (input_image * 255).astype(np.uint8)  # Convert back to [0, 255]
    input_image = Image.fromarray(input_image)
    
    # Post-process the output if necessary (e.g., convert to image format)
    processed_image = np.squeeze(processed_image, axis=0)  # Remove batch dimension
    processed_image = (processed_image * 255).astype(np.uint8)  # Convert back to [0, 255]
    processed_image = Image.fromarray(processed_image)
    
    
    reshaped_image_path = os.path.join(RESHAPED_FOLDER, os.path.basename(filepath))
    input_image.save(reshaped_image_path)


    # Save the processed image
    enhanced_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
    processed_image.save(enhanced_image_path)
    print ('process started')

    return [enhanced_image_path,reshaped_image_path]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']

    if file.filename == '':
        print (22)
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # pfilepath=preprocess_image(filepath)
        [processed_image_path,reshaped_image_path] = process_image(filepath)
        print(1021)
        
        # Convert the processed image to base64
        with open(processed_image_path, "rb") as image_file:
            print(1)
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        with open(reshaped_image_path, "rb") as image_file:
            print(1)
            original_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return render_template('index.html', img_data=encoded_string, original_img=original_string)
        # return the image from the image path
        


if __name__ == '__main__':
    app.run(debug=True)


