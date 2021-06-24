# import necessary packages
import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
#from PIL import ImageFont, ImageDraw, Image
 
 
model = load_model('model.h5')
model.summary()
 
# open webcam 현재 캠의 최대 해상도가 hd화질급이기 때문에 1280x720가 최대 
webcam = cv2.VideoCapture(0) 
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
 
# loop through frames
while webcam.isOpened():
 
    # read frame from webcam 
    status, frame = webcam.read()
    
    if not status:
        print("Could not read frame")
        exit()
 
    # apply face detection
    face, confidence = cv.detect_face(frame)
 
    # loop through detected faces
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        
        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            
            face_region = frame[startY:endY, startX:endX]
            
            face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)
            
            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            prediction = model.predict(x)
 
            if prediction < 0.5: # 인식된 사람이 아니면 블러,
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "My face ({:.2f}%)".format((1 - prediction[0][0])*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
 
                
            else: # 인식된 사람이면 초록색 
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "who? ({:.2f}%)".format(prediction[0][0]*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                '모자이크 효과 주기: 얼굴 부분을 줄였다가 다시 원크기로 복구시키면 모자이크처럼 됨.'
                face_region = frame[startY:endY, startX:endX]
                
                M = face_region.shape[0]
                N = face_region.shape[1]
         
                face_region = cv2.resize(face_region, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
                face_region = cv2.resize(face_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = face_region
     
                
    # display output
    cv2.imshow("bluuring", frame)
 
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows() 
