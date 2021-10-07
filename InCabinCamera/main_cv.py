import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import pygame 
from pygame import mixer 
from imutils import face_utils

import os
import sys
import dlib
import argparse

# helper modules
from drawFace import draw
import reference_world as world
import tensorflow as tf
from striprtf.striprtf import rtf_to_text

CATEGORIES = ['anger','pain','neutral','sad','pain']

# variables 
frame_counter  =0
CEF_COUNTER    =0
TOTAL_BLINKS   =0

counter_right  =0
counter_left   =0
counter_center =0 
counter_HD     =0
counter_dowsi  =0
counter_emer   =0
start_voice    = False
image_list = []

# constants
CLOSED_EYES_FRAME =1
NUM_FACE = 2
FONTS =cv.FONT_HERSHEY_COMPLEX
drowsiness_flag = False
emergency_flag = False
HeadDown_flag  = False

#Settingup MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)

# initialize mixer 
mixer.init()
# loading in the voices/sounds 
voice_left = mixer.Sound('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/LandmarkDetection/Eyes-Position-Estimator-Mediapipe/Eye_Tracking_part4/Voice/left.wav')
voice_right = mixer.Sound('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/LandmarkDetection/Eyes-Position-Estimator-Mediapipe/Eye_Tracking_part4/Voice/Right.wav')
voice_center = mixer.Sound('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/LandmarkDetection/Eyes-Position-Estimator-Mediapipe/Eye_Tracking_part4/Voice/center.wav')
voice_drowsiness = mixer.Sound('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Hackathon/Hackcelerate/Code/voice/DrowisnessAlert.wav')
voice_emergency  = mixer.Sound('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Hackathon/Hackcelerate/Code/voice/EmergencyAlert.wav')

PREDICTOR_PATH = os.path.join("/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/PoseEstimation/HeadPoseEstimation/models", "shape_predictor_68_face_landmarks.dat")
model          = tf.keras.models.load_model("/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/EmotionDetection/EmotionDetection_DeepLearning/Emotion-Detection/emotion_detection_CNN.model")


# Reading HeartRate Module
heartrate_array = []

with open('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Hackathon/Hackcelerate/Code/hr.txt', 'r') as file:
    for line in file:
        heartrate_array.append(int(line))

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

face3Dmodel = world.ref3DModel()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh

# camera object 
#camera = cv.VideoCapture("/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/LandmarkDetection/Eyes-Position-Estimator-Mediapipe/Eye_Tracking_part1/VideoFile.mp4")
camera = cv.VideoCapture("/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Ideation/LandmarkDetection/Eyes-Position-Estimator-Mediapipe/Eye_Tracking_part1/1632999156973.mp4")


# landmark detection function 

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    #cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    #cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            print(f'[ERROR - System]Cannot read from source')
            break # no more frames break
        #  resizing frame
        
        frame = cv.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv.INTER_LINEAR)
        dim = frame.shape
        mask = np.zeros(dim, dtype=np.uint8)

        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)

        ###### GAZE Detection

        h, w, c = rgb_frame.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        rgb_frame.flags.writeable = False
        gaze_results = face_detection.process(rgb_frame)

        # Draw the face detection annotations on the image.
        rgb_frame.flags.writeable = True


        if gaze_results.detections:
            for detection in gaze_results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                x_min = relative_bounding_box.xmin
                y_min = relative_bounding_box.ymin
                widthh = relative_bounding_box.width
                heightt = relative_bounding_box.height

                absx,absy=mp_drawing._normalized_to_pixel_coordinates(x_min,y_min,w,h)
                abswidth,absheight = mp_drawing._normalized_to_pixel_coordinates(x_min+widthh,y_min+heightt,w,h)
                
            newrect = dlib.rectangle(absx,absy,abswidth,absheight)
            cv.rectangle(frame, (absx, absy), (abswidth, absheight),(0, 255, 0), 2)
  
            shape = predictor(cv.cvtColor(frame, cv.COLOR_BGR2RGB), newrect)
            draw(frame, shape) 
            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = frame.shape
            focalLength = 1 * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))
            
            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

            cv.line(frame, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv.LINE_AA)

            # calculating euler angles
            rmat, jac = cv.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])

            rvec_matrix = cv.Rodrigues(rotationVector)[0]
            proj_matrix = np.hstack((rvec_matrix, translationVector))
            eulerAngles = cv.decomposeProjectionMatrix(proj_matrix)[6]

            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(math.asin(math.sin(roll)))
            yaw = math.degrees(math.asin(math.sin(yaw)))

            if pitch < 20 :
                cv.putText(mask, f'Pitch               : {round(pitch,2)}',  (10, 240), FONTS, 0.6, utils.RED, 2)
                utils.colorBackgroundText(frame,  f'Head Down !!! ', FONTS, 1.7, (int(frame_width/4), 200), 2, utils.YELLOW, pad_x=6, pad_y=6, ) 
                HeadDown_flag = True
            elif pitch > 50 :
                cv.putText(mask, f'Pitch               : {round(pitch,2)}',  (10, 240), FONTS, 0.6, utils.RED, 2)
                utils.colorBackgroundText(frame,  f'Head Up !!! ', FONTS, 1.7, (int(frame_width/4), 200), 2, utils.YELLOW, pad_x=6, pad_y=6, ) 
                HeadUp_flag = True
            else:
                cv.putText(mask, f'Pitch               : {round(pitch,2)}',  (10, 240), FONTS, 0.6, utils.GREEN, 2)

        
            cv.putText(mask, f'Roll                : {round(roll,2)}',  (10, 260), FONTS, 0.6, utils.GREEN, 2)
            cv.putText(mask, f'Yaw                 : {round(yaw,2)}',  (10, 280), FONTS, 0.6, utils.GREEN, 2)

            ##### Emotion Detection starts
            img_rows,img_cols = 224,224
            gray = frame[absy:absheight,absx:abswidth]
            pred_img = cv.resize(gray, (img_rows,img_cols))
            gray_resize = pred_img.reshape(-1,img_rows,img_cols, 1)

            
            try:
                prediction = model.predict([gray_resize])
                cv.putText(mask, f'Emotion             :{CATEGORIES[prediction.argmax()]}', (10, 320), FONTS, 0.6, utils.GREEN, 2)
                if CATEGORIES[prediction.argmax()] == 'pain':
                    cv.putText(mask, f'Emotion             :{CATEGORIES[prediction.argmax()]}', (10, 320), FONTS, 0.6, utils.RED, 2)
            except:
                None
            # Emotion Detection ends
        
        ##### Occupancy Detection
        if frame_counter >= 80:
            cv.putText(mask, f'No of Occupancy     : 1', (10, 350), FONTS, 0.6, utils.RED, 2)
        else:
            cv.putText(mask, f'No of Occupancy     : 2', (10, 350), FONTS, 0.6, utils.GREEN, 2)
        
        
        ##### Eye Measurement detection
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(mask, faceLms,mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)

            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

            if ratio >5.5:
                CEF_COUNTER +=1
                if CEF_COUNTER > 7:  
                    utils.colorBackgroundText(frame,  f'Emergency Detected!!! ', FONTS, 1.7, (int(frame_width/4), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, ) 
                    emergency_flag = True
                elif CEF_COUNTER > 3: 
                    utils.colorBackgroundText(frame,  f'Drowsiness Detected!!! ', FONTS, 1.7, (int(frame_width/4), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )
                    drowsiness_flag = True
                
                    #voice_left.play() 
            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:
                    TOTAL_BLINKS +=1
                    CEF_COUNTER =0
            cv.putText(mask, f'Total Blinks        : {TOTAL_BLINKS}', (10, 150), FONTS, 0.6, utils.GREEN, 2)
            
            #cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            #cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            
            eye_position_right, color = positionEstimator(crop_right)
            eye_position_left, color = positionEstimator(crop_left)
            cv.putText(mask, f'Right eye position : {eye_position_right}', (10, 180), FONTS, 0.6, utils.GREEN, 2)
            cv.putText(mask, f'Left eye position  : {eye_position_left}', (10, 200), FONTS, 0.6, utils.GREEN, 2)
            
            try:
                if heartrate_array[frame_counter] > 120:
                    cv.putText(mask, f'Heart Rate  : {heartrate_array[frame_counter]}', (10, 370), FONTS, 0.6, utils.RED, 2)
                else:
                    cv.putText(mask, f'Heart Rate  : {heartrate_array[frame_counter]}', (10, 370), FONTS, 0.6, utils.GREEN, 2)
            except:
                None
            # Starting Voice Indicator 
            
            if HeadDown_flag == True and pygame.mixer.get_busy()==0 and counter_HD < 2:
                counter_HD   += 1 
                counter_emer  = 0
                counter_dowsi = 0
                voice_emergency.play()
                emergency_flag = False
                drowsiness_flag = False
                HeadDown_flag = False
                HeadUp_flag = False

            if emergency_flag == True and pygame.mixer.get_busy()==0 and counter_emer < 2:
                counter_HD    = 0 
                counter_emer += 1
                counter_dowsi = 0
                voice_emergency.play()
                emergency_flag = False
                drowsiness_flag = False
                HeadDown_flag = False
                HeadUp_flag = False

            if drowsiness_flag == True and pygame.mixer.get_busy()==0 and (counter_dowsi < 2 and counter_HD == 0 and counter_emer == 0):
                counter_HD    = 0 
                counter_emer  = 0
                counter_dowsi+= 1
                voice_drowsiness.play()
                drowsiness_flag = False
                emergency_flag = False
                HeadDown_flag = False
                HeadUp_flag = False
            

        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        
        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        # creating mask from gray scale dim
        concated_image =  np.concatenate((frame, mask), axis=1)
        cv.imshow('frame', concated_image)
        image_list.append(concated_image)

        height, width, layers = image_list[0].shape
        size = (width, height)
        out = cv.VideoWriter('drive_0_output_1.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20, size)

        for i in range(len(image_list)):
            out.write(image_list[i])
        
        key = cv.waitKey(1)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    out.release()
    camera.release()
