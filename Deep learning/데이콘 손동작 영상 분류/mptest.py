
#https://www.computervision.zone/lessons/code-and-files-6/
#https://www.section.io/engineering-education/creating-a-hand-tracking-module/
#https://google.github.io/mediapipe/solutions/hands.html
import cv2
import numpy as np
import os
os.chdir("E:\GoogleDrive\pycv\리모콘 제스쳐")

os.getcwd()
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.8, maxHands=2)

def conv(path):
    cap = cv2.VideoCapture(path)
    out=[]
    while True:

        ret, img_bgr = cap.read()
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        hands, img_bgr = detector.findHands(img_bgr)  # With Draw
        if hands:
            out.append(hands[0]['lmList'])

        if ret == False:
            break
        key = cv2.waitKey(33)
        if key == 1:
            break
        # 무한반복
        # if (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cv2.imshow("Result", img_bgr)
        if (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            break
    cap.release()
    cv2.destroyAllWindows()
    return out

a=conv('test/TEST_008.mp4')

#추출한 좌표 확인해보기
x=[]
y=[]
z=[]
black_canvas = np.zeros((128, 128, 3), dtype="uint8")
for j in range(cv2.CAP_PROP_FRAME_COUNT):
    for i in range(20):
        x,y,z=a[j][i]
        x1,y1,z1=a[j][i+1]

        im=cv2.line(black_canvas,(x,y),(x1,y1),(255,255,255),4)
# js=[0,29]
# for j in js:
#     for i in range(20):
#         x,y,z=a[j][i]
#         x1,y1,z1=a[j][i+1]
#         im=cv2.line(black_canvas,(x,y),(x1,y1),(255,255,255),5)

cv2.imshow("a",im)
cv2.waitKey(0)
cv2.destroyAllWindows()


