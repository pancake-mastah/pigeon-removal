import cv2
from ultralytics import YOLO
import pygame

pygame.mixer.init()
pygame.mixer.music.load("C:/Users/himan/Downloads/female-scream-longer-251068.mp3")

model = YOLO("C:/Users/himan/OneDrive/Desktop/pigeon/best (2).pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame,conf=0.8)

    for result in results:
        for detection in result.boxes.data:
            class_id = int(detection[5]) 
            if class_id == 0:  
                if not pygame.mixer.music.get_busy(): 
                    pygame.mixer.music.play()


    annotated_frame = results[0].plot()  
    cv2.imshow('Pigeon Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
