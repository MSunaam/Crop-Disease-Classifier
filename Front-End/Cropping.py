import os

weights_path = "/Users/ali/Desktop/AI project/best (1).pt"
image="/Users/ali/Desktop/AI project/anthracnose2_.jpg"


# Enclose paths in double quotes within the command string
def yolo(image):
    source_path = image
    detection_command = f'python /Users/ali/Desktop/AI\ Project/yolov5/detect.py --weights "{weights_path}" --source "{source_path}" --save-txt'
    os.system(detection_command)


yolo(image)