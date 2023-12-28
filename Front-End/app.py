from flask import Flask, request, render_template
from flask_cors import CORS 
from datetime import datetime 
from inference import inference
from urllib.parse import urlencode
import base64

import os

app = Flask(__name__)
  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return render_template('error.html', error='No file part')

    file = request.files['file']

    # Check if the file is one of the allowed types/extensions
    # allowed_extensions = {'jpg', 'jpeg', 'png'}
    # if file.filename.split('.')[-1].lower() not in allowed_extensions:
    #     return render_template('error.html', error='Invalid file type')

    # Save the uploaded image to a temporary location
    upload_folder = '/Users/ali/Desktop/AI_PROJECT/Upload_folder'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
     # Generate dynamic folder path based on current timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    
    

    # Generate dynamic image name based on current timestamp
    image_name = f'detected_{timestamp_str}'
    image_path = os.path.join('/Users/ali/Desktop/AI_project/BoundingImage')

    filename=file.filename
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)


    # Run YOLOv5 detection command
    detection_command = f"python /Users/ali/Desktop/AI_project/yolov5/detect.py --weights '/Users/ali/Desktop/AI_project/best (1).pt' --source '{file_path}' --save-txt --project '{image_path}' --name '{image_name}'"
    os.system(detection_command)
    Image_extract=os.path.join(image_path,image_name)
    
    iMAGE_eXTRACT=os.path.join(Image_extract,filename)
    
    if os.path.exists(iMAGE_eXTRACT):
        final_prediction=inference(iMAGE_eXTRACT)
        print(final_prediction)
    else:
        final_prediction=inference(file_path)
        print(final_prediction)
    # print(iMAGE_eXTRACT)
    
    
    print(file_path)
    
    # Convert the image to base64 encoding
    with open(file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    print(final_prediction)

    

    # Return response to frontend (you may customize this based on your needs)
    return render_template('index.html', final_predictions=final_prediction['finalPrediction'],confidence_level=final_prediction['confidenceLevel'])

if __name__ == '__main__':
    app.run(debug=True,port=5001)
