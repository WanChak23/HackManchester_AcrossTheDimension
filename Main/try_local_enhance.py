#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ultralytics
import webbrowser
from ultralytics import YOLO
import os
import torch
from unittest import result
import cv2
import numpy as np
import json
import time
import socket
import mediapipe as mp
import pyautogui
import copy

from mediapipe.tasks import python




mp_hands = mp.solutions.hands
hands = mp_hands.Hands()



# Yolo detect dictionary
object_code = {0: "person",
      1: "bicycle",
      2: "car",
      3: "motorcycle",
      4: "airplane",
      5: "bus",
      6: "train",
      7: "truck",
      8: "boat",
      9: "traffic light",
      10: "fire hydrant",
      11: "stop sign",
      12: "parking meter",
      13: "bench",
      14: "bird",
      15: "cat",
      16: "dog",
      17: "horse",
      18: "sheep",
      19: "cow",
      20: "elephant",
      21: "bear",
      22: "zebra",
      23: "giraffe",
      24: "backpack",
      25: "umbrella",
      26: "handbag",
      27: "tie",
      28: "suitcase",
      29: "frisbee",
      30: "skis",
      31: "snowboard",
      32: "sports ball",
      33: "kite",
      34: "baseball bat",
      35: "baseball glove",
      36: "skateboard",
      37: "surfboard",
      38: "tennis racket",
      39: "bottle",
      40: "wine glass",
      41: "cup",
      42: "fork",
      43: "knife",
      44: "spoon",
      45: "bowl",
      46: "banana",
      47: "apple",
      48: "sandwich",
      49: "orange",
      50: "broccoli",
      51: "carrot",
      52: "hot dog",
      53: "pizza",
      54: "donut",
      55: "cake",
      56: "chair",
      57: "couch",
      58: "potted plant",
      59: "bed",
      60: "dining table",
      61: "toilet",
      62: "tv",
      63: "laptop",
      64: "mouse",
      65: "remote",
      66: "keyboard",
      67: "cell phone",
      68: "microwave",
      69: "oven",
      70: "toaster",
      71: "sink",
      72: "refrigerator",
      73: "book",
      74: "clock",
      75: "vase",
      76: "scissors",
      77: "teddy bear",
      78: "hair drier",
      79: "toothbrush"}


websiteDict = {"cat": "https://en.wikipedia.org/wiki/Cat", 
               "dog": "https://en.wikipedia.org/wiki/Dog", 
               "bird": "https://en.wikipedia.org/wiki/Bird", 
               "broccoli": "https://en.wikipedia.org/wiki/Broccoli",
               "Carrot": "https://en.wikipedia.org/wiki/Carrot"}

 
def hand_track(last_frame_open_palm):


    open_palm_threshold = 0.24
    #last_frame_open_palm = False


    # Initialize the MediaPipe Hands model
    

    # Initialize the VideoCapture object

    #cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    #def track_hand(last_frame_open_palm):
    out = ""
    #ret = False
    #while ret == False:
    while True:
        ret, frame_cam = cam.read()
        #if not ret:
        #    break

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        #frame_rgb = cv2.flip(frame_rgb,1)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)


        cor_all = []

        if results.multi_hand_landmarks:

            for landmarks in results.multi_hand_landmarks:
                cor_finger = []

                """for i in [8]:
                    point = landmarks.landmark[i]
                    x, y = int(frame.shape[1] * point.x), int(frame.shape[0] * point.y)
                    cor_finger.append(x)
                    cor_finger.append(y)
                """
                for point in landmarks.landmark:
                    x, y = int(frame_cam.shape[1] * point.x), int(frame_cam.shape[0] * point.y)
                    cor_finger.append([frame_cam.shape[1]-x,y])
                    
                    
                    #cv2.circle(frame_cam, (x, y), 5, (0, 0, 255), -1)

                cor_all.append(cor_finger)
            #print(cor_all[8])



            #for i in range(5):
            #    cv2.circle(frame, (cor_all[0][i][0], cor_all[0][i][1]), 5, (0, 0, 255), -1)
        #print(recognition_result)


            #cv2.imshow("Hand Motion Tracking", frame_cam)




            landmarks = results.multi_hand_landmarks[0].landmark

            # Calculate the distance between the fingertips (e.g., index finger and thumb)
            dist_index_thumb = ((landmarks[4].x - landmarks[12].x) ** 2 + (landmarks[4].y - landmarks[12].y) ** 2) ** 0.5

            # Check if the distance is above the threshold
            if dist_index_thumb > open_palm_threshold:
                open_palm_detected = True
            else:
                open_palm_detected = False

            # Print a message when the open palm gesture is detected

            if open_palm_detected and not last_frame_open_palm:
                out = "o"
            #elif open_palm_detected:
            #    print("keep")

            elif open_palm_detected == False and last_frame_open_palm:
                out = "c"

            else: 
                out = "n"
            last_frame_open_palm = open_palm_detected
            
            return cor_all[0][8], out, open_palm_detected
            

        #print(out)
        #cor_all[0][8]

        #cv2.imshow("Finger Tracking", frame)


#        try:
#            send_str = str(cor_all[0][8][0]) + ";" + str(cor_all[0][8][1]) + ";" + out    
#            send_str = send_str.encode()
            #print(send_str.decode())



#            s1.send(send_str)
#        except:
#            pass


        #time.sleep(0.1)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break 
        # Press 'Esc' to exit
def generate(filename, start_frame):
    cap = cv2.VideoCapture('Final_Main/'+filename)
    try: 
      
        # creating a folder named data 
        if not os.path.exists('Final_Main/data'): 
            os.makedirs('Final_Main/data') 
  
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
  

    
    # Loop until the end of the video
    if (cap.isOpened()== False): 
        print("Error opening video file") 
    # Read until video is completed 
    
    current_frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    while(cap.isOpened()): 
      
    # Capture frame-by-frame 
        
        ret, frame = cap.read()
    
        if ret: 
        # Display the resulting frame 
            cv2.imshow('Frame', frame) 
            #cv2.waitKey(250)
        # Press space on keyboard to exit 
            if cv2.waitKey(1) & 0xFF == ord('s'): 
            
                #path = './Final_Main/data/picture.jpg'
                #print ('Creating...' + path) 
                # writing the extracted images 
                #cv2.imwrite(path, frame) 
                
                return frame, current_frame
                
    # Break the loop 
        else: 
            cv2.destroyAllWindows()
            break
        current_frame += 1
        
        
        
    # release the video capture object
    cap.release()
    

def detection(frame_np):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # Images
    #img_path = path  # Specify the path to your image
    #imgs = [img_path]  # batch of images

    # Inference
    img = frame_np
    results = model(img)

    # Load the image using OpenCV
    
    #img = cv2.imread(img_path)

    
    
    # Get the predicted bounding boxes
    pred_boxes = results.pred[0][:, :].numpy()

 
    objects_cor = []
    objects_name = []
    
    # Draw your custom bounding boxes
    for box in pred_boxes:
        x1, y1, x2, y2, probab, code = box.astype(float)
        x1 = int (x1)
        x2 = int (x2)
        y1 = int(y1)
        y2 = int(y2)
        code = int(code)
        objects_cor.append([x1,y1,x2,y2])
        objects_name.append(object_code[code])
        color = (255, 125, 130)  # Green color (BGR)
        thickness = 1  # Line thickness
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        #cv2.imshow('Frame',img)
        #cv2.waitKey(250)
        #objects_name.append(data[code])
        #cv2.destroyAllWindows()
    #while (True):
    #    cv2.imshow('Frame',img)
    #    if cv2.waitKey(1) & 0xFF == ord('s'): 
    #        break

    return objects_cor,objects_name, img


def cor_within_object(hand_cor_lst, object_cor_lst):
    # compare "x" location 
    
    if (hand_cor_lst[0]>=int(object_cor_lst[0])) and (hand_cor_lst[0]<=int(object_cor_lst[2])):
        if (hand_cor_lst[1]>=int(object_cor_lst[1])) and (hand_cor_lst[1]<=int(object_cor_lst[3])):
            return True
        else:
            return False
    else:
        return False



"""
TCP_IP = "172.20.10.4"
TCP_PORT = 5005
BUFFER_SIZE = 1024 


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #set up socket
s.bind((TCP_IP,TCP_PORT))
s.listen(1)


conn, addr = s.accept()
exceptNum = 0
exceptNum1 = 0 

print("connect to edwin")
cv2.imshow('Frame',img)
"""

frame_up_to = 0

frame_np, frame_up_to = generate('test_v1.mp4', frame_up_to)

object_cor_lst,objects_name_lst, img = detection(frame_np)


cam = cv2.VideoCapture(0)
last_frame_open_palm = False
within_one = False
change = False
img_now = np.copy(img)
added_rectangle = False

while True:
    
    while True:


        #global object_cor_lst



    #    data = conn.recv(BUFFER_SIZE)
    #    data = data.decode()
    #    data = data.split(";")

    #    try:
    #        hand_cor = [int(data[0]), int(data[1])]
    #        palm_state = data[2]
    #        print(data)
    #    except :
            #print("running except1" + str(exceptNum1))
            #if exceptNum1 == 100:
            #    exceptNum1 = 0
            #else:
            #    exceptNum1 += 1
    #        pass

        try:
            hand_cor, palm_state, last_frame_open_palm = hand_track(last_frame_open_palm)
        except:
            pass
        
        
        

        for i in range(len(object_cor_lst)):
            test = cor_within_object(hand_cor, object_cor_lst[i])
            if test:
                color = (255, 255, 255)  # Green color (BGR)
                thickness = 4  # Line thickness
                img_now = np.copy(img)
                img_now = cv2.rectangle(img_now,(object_cor_lst[i][0],object_cor_lst[i][1]),(object_cor_lst[i][2],object_cor_lst[i][3]), color, thickness)
                change = True
                if palm_state == 'o':
                    webbrowser.open(websiteDict[objects_name_lst[i]])
                    #print(objects_name_lst[i])
                    #print(objects_name_lst)

                elif palm_state == "c":
                    browserExe= "chrome.exe"
                    #os.system("taskkill /f /im "+browserExe) #on windows OS
                    #os.system("killall -9 'Google Chrome'") #on macOS to kill browser
                    pyautogui.hotkey('command', 'w')#on macOS to kill only one tab

    
        img_now = cv2.circle(img_now, (hand_cor[0], hand_cor[1]), 5, (0, 0, 255), -1)    
        
        cv2.imshow('Frame',img_now)
        
        if change == False:
            img_now = np.copy(img)
        change = False
        
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cam.release()
            break

    frame_np, frame_up_to = generate('test_v1.mp4', frame_up_to)

    object_cor_lst,objects_name_lst, img = detection(frame_np)


    cam = cv2.VideoCapture(0)
    last_frame_open_palm = False
    change = False
    img_now = np.copy(img)


# In[ ]:




