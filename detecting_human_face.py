import cv2 
import numpy as np

faceCascade = cv2.CascadeClassifier("photos\haarcascade_frontalface_default.xml")




cap = cv2.VideoCapture(0)

cap.set(3, 500)

cap.set(4, 500)

cap.set(10, 0.1)

while True:
    succss, img = cap.read()
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayScale, 1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0), 2)
        cv2.putText(img,"Human face",(x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),1)
        cv2.imshow("cam", img)
    if cv2.waitKey(1) & 0xFF ==ord("d"):
        break
    
cap.release()
cv2.destroyAllWindows()



