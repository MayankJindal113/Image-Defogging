import os
from flask import Flask, request, render_template, redirect
import numpy as np
from PIL import Image
import base64
import cv2
import tensorflow as tf
from flask import Flask, send_from_directory
from flask import Response, request, send_file
import re  # Import the re module

# Initialize Flask app
app = Flask(__name__)

# Load deep learning models
model3 = tf.keras.models.load_model('2xUNet.h5')
model2 = tf.keras.models.load_model('REVIDE.h5')
model1 = tf.keras.models.load_model('Double-U-Net.h5')

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
RESHAPED_FOLDER = 'reshaped'
FRAMES_FOLDER = 'frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(RESHAPED_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['RESHAPED_FOLDER'] = RESHAPED_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER

# Helper functions
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        # image = image.resize((192, 128))  # Resize to model input size 256x512
        image = image.resize((512, 256))  # Resize to model input size 256x512

        image = np.array(image) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


def process_image(filepath):
    input_image = preprocess_image(filepath)
    if input_image is None:
        return None, None
    
    try:
        processed_image = model3.predict(input_image)
        reshaped_image = np.squeeze(input_image, axis=0)
        reshaped_image = (reshaped_image * 255).astype(np.uint8)
        reshaped_image = Image.fromarray(reshaped_image)
        reshaped_image_path = os.path.join(RESHAPED_FOLDER, os.path.basename(filepath))
        reshaped_image.save(reshaped_image_path)

        processed_image = np.squeeze(processed_image, axis=0)
        processed_image = (processed_image * 255).astype(np.uint8)
        processed_image = Image.fromarray(processed_image)
        processed_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
        processed_image.save(processed_image_path)

        return processed_image_path, reshaped_image_path
    except Exception as e:
        print(f"Error during image processing: {e}")
        return None, None


def process_video(filepath, frame_rate=15):
    try:
        print('In the try block')
        video_name = os.path.splitext(os.path.basename(filepath))[0]
        frame_folder = os.path.join(FRAMES_FOLDER, video_name)
        os.makedirs(frame_folder, exist_ok=True)

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print('Video couldn\'t be opened\n')
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(int(video_fps / frame_rate), 1)
        frame_count = 0
        frame_paths = []
        print ('I am here\n')
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Havock')
                break
            if frame_count % frame_interval == 0:
                print('something with frames\n')
                frame_filename = f"{video_name}_{frame_count}.jpg"
                frame_path = os.path.join(frame_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            frame_count += 1
        cap.release()

        # Process each frame using the model
        processed_frame_paths = []
        for frame_path in frame_paths:
            processed_image_path, _ = process_image(frame_path)
            if processed_image_path:
                print('Frames were processed\n')
                processed_frame_paths.append(processed_image_path)

        # Combine processed frames back into a video
        height, width, _ = cv2.imread(processed_frame_paths[0]).shape
        output_video_path = os.path.join(PROCESSED_FOLDER, f"{video_name}_processed.mp4")
        video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
        print('entering the loop')
        for frame_path in processed_frame_paths:
            print('working\n')
            video.write(cv2.imread(frame_path))
        video.release()

        return output_video_path
    except Exception as e:
        print(f"Error during video processing: {e}")
        return None


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            processed_image_path, reshaped_image_path = process_image(filepath)
            if processed_image_path is None or reshaped_image_path is None:
                return "Error in processing the image.", 500
            
            with open(processed_image_path, "rb") as image_file:
                img_data = base64.b64encode(image_file.read()).decode('utf-8')
            with open(reshaped_image_path, "rb") as image_file:
                original_img = base64.b64encode(image_file.read()).decode('utf-8')
            
            return render_template('index.html', img_data=img_data, original_img=original_img)

    if 'video' in request.files:
        print('In video')
        file = request.files['video']
        if file and file.filename.lower().endswith(('mp4', 'avi', 'mov', 'mkv')):
            print('I just found a video format')
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print('I saved a video in location', filepath)
            processed_video_path = process_video(filepath)
            if processed_video_path is None:
                print('The vid couldnt be processed')
                return "Error in processing the video.", 500
            print('Video processing done Completely, now will render it\n')
            # return render_template('index.html', video_path=f'/processed/{os.path.basename(processed_video_path)}')
            # return f"Video processed and saved at {processed_video_path}"
            return render_template('index.html',original_video_path=f'/uploads/{os.path.basename(filepath)}',video_path=f'/processed/{os.path.basename(processed_video_path)}')


    return "Invalid file format. Please upload a valid image or video.", 400

# @app.route('/processed/<filename>')
# def serve_processed_file(filename):
#     return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/processed/<filename>')
def serve_processed_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        return Response("File not found", status=404)

    # Check for 'Range' header
    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(file_path)  # Serve the entire file if no Range header

    # Parse the Range header (e.g., "bytes=0-1000")
    range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
    if not range_match:
        return Response("Invalid range", status=416)

    start_byte = int(range_match.group(1))
    end_byte = range_match.group(2)
    file_size = os.path.getsize(file_path)
    end_byte = int(end_byte) if end_byte else file_size - 1

    # Validate range
    if start_byte >= file_size or end_byte >= file_size:
        return Response("Requested range not satisfiable", status=416)

    # Serve the requested byte range
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        data = f.read(end_byte - start_byte + 1)

    headers = {
        'Content-Range': f'bytes {start_byte}-{end_byte}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(len(data)),
        'Content-Type': 'video/mp4',
    }
    return Response(data, status=206, headers=headers)


if __name__ == '__main__':
    app.run(debug=True)




# import os
# from flask import Flask, request, render_template, send_from_directory,redirect
# import numpy as np
# from PIL import Image
# import base64

# # Load your deep learning model here
# import tensorflow as tf

# app = Flask(__name__)
# if __name__ == "__main__":

#     app.run(debug=True)

# model = tf.keras.models.load_model('REVIDE.h5')


# # Configuration
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# RESHAPED_FOLDER = 'reshaped'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# os.makedirs(RESHAPED_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
# app.config['RESHAPED_FOLDER'] = RESHAPED_FOLDER


# def preprocess_image(image_path):
#     print ('preprocess started')
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize((192, 128))  # Resize to the input size expected by the model
#     image = np.array(image) / 255.0  # Normalize to [0, 1]
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     print ('preprocess ended')

#     return image

# def process_image(filepath):
#     print ('process started')
    
#     # Preprocess the image
#     input_image = preprocess_image(filepath)
    
#     # Make predictions
#     print('will predict')
#     processed_image = model.predict(input_image)
#     print('predicted')

#     input_image = np.squeeze(input_image, axis=0)  # Remove batch dimension
#     input_image = (input_image * 255).astype(np.uint8)  # Convert back to [0, 255]
#     input_image = Image.fromarray(input_image)
    
#     # Post-process the output if necessary (e.g., convert to image format)
#     processed_image = np.squeeze(processed_image, axis=0)  # Remove batch dimension
#     processed_image = (processed_image * 255).astype(np.uint8)  # Convert back to [0, 255]
#     processed_image = Image.fromarray(processed_image)
    
    
#     reshaped_image_path = os.path.join(RESHAPED_FOLDER, os.path.basename(filepath))
#     input_image.save(reshaped_image_path)


#     # Save the processed image
#     enhanced_image_path = os.path.join(PROCESSED_FOLDER, os.path.basename(filepath))
#     processed_image.save(enhanced_image_path)
#     print ('process started')

#     return [enhanced_image_path,reshaped_image_path]

# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return redirect(request.url)
#     file = request.files['image']

#     if file.filename == '':
#         print (22)
#         return redirect(request.url)
#     if file:
#         filename = file.filename
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)

#         # pfilepath=preprocess_image(filepath)
#         [processed_image_path,reshaped_image_path] = process_image(filepath)
#         print(1021)
        
#         # Convert the processed image to base64
#         with open(processed_image_path, "rb") as image_file:
#             print(1)
#             encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

#         with open(reshaped_image_path, "rb") as image_file:
#             print(1)
#             original_string = base64.b64encode(image_file.read()).decode('utf-8')
        
#         return render_template('index.html', img_data=encoded_string, original_img=original_string)
#         # return the image from the image path
        


# if __name__ == '__main__':
#     app.run(debug=True)


