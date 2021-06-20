import cv2
import numpy as np 
import argparse
import sys
import os
from os import listdir
from os.path import isfile, join
import math
from mtcnn_cv2 import MTCNN


# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-webcam', help="no path", action='store_true')
parser.add_argument('-video', help="requires path")
parser.add_argument('-image', help="requires  path")
parser.add_argument('-folder', help="requires path")
args = parser.parse_args()

###########################################################################

def drawbox(frame, box, boxcolor, confidence, detection) :
    '''
    keypoints = detection['keypoints']
    left_eye = keypoints['left_eye']
    ## left_eye = (x,y)
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    '''
    
    # print(f)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    cv2.rectangle(frame, (x, y), (x+w, y+h), boxcolor, 2)
    
    '''
    ## draw keypoints
    for point in [left_eye,right_eye,nose,mouth_left,mouth_right]:
        cv2.circle(frame, point, radius=4, color=(0, 0, 255), thickness=-1)
    '''
    
    textcolor = (255,155,0)
    text = "{:.4f}".format(confidence)
    cv2.putText(frame, text,(x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,textcolor,2)
    return(frame)

def draw_boxes(frame, detections, detector) :
    number_detections = 0
    detection_color = [((53*x)%255,(79*x)%255,(194*x)%255) for x in range(1,100)]
    non_detection_color = (155,255,0)

    if len(detections) > 0 :
        for f in detections:
            box = f['box']
            confidence = f['confidence']
            keypoints = f['keypoints']
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
            nose = keypoints['nose']
            mouth_left = keypoints['mouth_left']
            mouth_right = keypoints['mouth_right']
            
            if(wink(frame, f)) :
                number_detections += 1
                drawbox(frame, box, detection_color[number_detections], confidence, f)
#            else: 
#                drawbox(frame, box, non_detection_color, confidence,f)
    return(number_detections)


def wink(frame, f) :
    rows, cols, bands = frame.shape
    box = f['box']
    confidence = f['confidence']
    keypoints = f['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    
    ## while smiling: mouse width will be wider and the overall mouth will lift
    ## to simulate there phenomenon, use the distance of eyes and nose as base, measure the distance(ratio) of mouth
    ## mouth_eye_width_ratio - horizontal ratio : bigger --> smiling
    eye_distance = math.dist(left_eye,right_eye)
    mouth_width = math.dist(mouth_left,mouth_right)
    horizontal_ratio = mouth_width / eye_distance
    
    ## vertical ratio: mouse-nose / eye_nose  ## should smaller than 1, smaller --> smiling
    ## use the mid point to present the mouth and eyes
    mouth_mid = ((mouth_left[0]+mouth_right[0])/2 , (mouth_left[1]+mouth_right[1])/2)
    eye_mid =  ((left_eye[0]+right_eye[0])/2 , (left_eye[1]+right_eye[1])/2)
    vertical_ratio = math.dist(mouth_mid, nose) / math.dist(eye_mid,nose)
    
    ## record.append((mouth_eye_width_ratio,vertical_ratio))
    
    standard_horizontal_ratio = 0.90
    standard_vertical_ratio = 0.84
    result = 0
    result += (horizontal_ratio-standard_horizontal_ratio)*3
    result += -(vertical_ratio-standard_vertical_ratio)
    
    if result > 0:
        return True
    else:
        return False
    

def detection_frame(detector, frame) :
        detections = detector.detect_faces(frame)
        number_detections = draw_boxes(frame, detections, detector)
        return(number_detections)


def detect_video(detector, video_source) :
        windowName = "Video"
        showlive = True
        while(showlive):
                ret, frame = video_source.read()
                if not ret:
                    showlive = False;
                else :
                    detection_frame(detector, frame)
                    cv2.imshow(windowName, frame)
                    if cv2.waitKey(30) >= 0:
                        showlive = False
        # outside the while loop
        video_source.release()
        cv2.destroyAllWindows()
        return



###########################################################################

def runon_image(detector, path) :
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections_in_frame = detection_frame(detector, frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("one image", frame)
        cv2.waitKey(0)
        return detections_in_frame

def runon_webcam(detector) :
        video_source = cv2.VideoCapture(0)
        if not video_source.isOpened():
                print("Can't open default video camera!")
                exit()

        detect_video(detector, video_source)
        return

def runon_video(detector, path) :
        video_source = cv2.VideoCapture(path)
        if not video_source.isOpened():
                print("Can't open video ", path)
                exit()
        detect_video(detector, video_source)
        return

def runon_folder(detector, path) :
    if(path[-1] != "/"):
        path = path + "/"
        files = [join(path,f) for f in listdir(path) if isfile(join(path,f))]
    all_detections = 0
    for f in files:
        f_detections = runon_image(detector, f)
        all_detections += f_detections
    return all_detections


## smiling_record = []
## record = []



'''
detector = MTCNN() 

image = "D:\\untitled.png"
runon_image(detector, image)

folder_path = r"D:\\winks\normal"
runon_folder(detector,folder_path)

runon_video(detector, "D:\\funny.mp4")

runon_webcam(detector)


folder_path = r"D:\\winks"
count = runon_folder(detector,folder_path)

cv2.destroyAllWindows()

'''








if __name__ == '__main__':
        webcam = args.webcam
        video = args.video
        image = args.image
        folder = args.folder
        if not webcam and video is None and image is None and folder is None :
                print(
                "one argument from webcam,video,image,folder must be given.",
                "\n",
                "Example)",
                "-webcam or -video clip.avi or -image image.jpg or -folder images")
                sys.exit()

        detector = MTCNN()

        if webcam :
            runon_webcam(detector)
        elif video is not None :
            runon_video(detector,video)
        elif image is not None :
            runon_image(detector,image)
        elif folder is not None :
            all_detections = runon_folder(detector,folder)
            print("total of ", all_detections, " detections")
        else :
            print("impossible")
            sys.exit()

        cv2.destroyAllWindows()


