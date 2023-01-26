##mediapipe써서
import random
import pandas as pd
import numpy as np
import os
import cv2
import torch

from sklearn.model_selection import train_test_split

import mediapipe as mp
import warnings

os.chdir("E:\GoogleDrive\pycv\리모콘 제스쳐")
warnings.filterwarnings(action='ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':8,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])
black_canvas = np.zeros((128, 128, 3), dtype="uint8")

#마지막프레임만 저장
def trCovertFrameJpg(ids):
    global black_canvas
    for id in ids:
        cap = cv2.VideoCapture('./train/'+id+'.mp4')
        frames = []
        #os.mkdir(f"train/{id}")
        for i in range(CFG['FPS']):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    black_canvas = np.zeros((128, 128, 3), dtype="uint8")
                    mp_drawing.draw_landmarks(black_canvas, res, None,
                                              mp_drawing.DrawingSpec(thickness=2,
                                                                     color=(254, 254, 254)))
            frame = cv2.resize(black_canvas, (128,128))
            #frames.append(frame)

            if i == 26:   cv2.imwrite(f"./tr/{id}.jpg", frame)
        cap.release()

def teCovertFrameJpg(ids):
    global black_canvas
    for id in ids:
        cap = cv2.VideoCapture('./test/'+id+'.mp4')
        frames = []
        #os.mkdir(f"train/{id}")
        for i in range(CFG['FPS']):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    black_canvas = np.zeros((128, 128, 3), dtype="uint8")
                    mp_drawing.draw_landmarks(black_canvas, res, None,
                                              mp_drawing.DrawingSpec(thickness=2,
                                                                     color=(254, 254, 254)))
            frame = cv2.resize(black_canvas, (128,128))
            #frames.append(frame)

            if i == 26:   cv2.imwrite(f"./te/{id}.jpg", frame)
        cap.release()
#
# #모든 프레임 저장
def trainCovertFrameJpg(ids):
    global black_canvas
    for id in ids:
        cap = cv2.VideoCapture('./train/'+id+'.mp4')
        frames = []
        #make_new_dir(f"train_jpg/{id}")
        black_canvas = np.zeros((128, 128, 3), dtype="uint8")
        for i in range(CFG['FPS']):
            ret, frame = cap.read()
            if ret == False:
                break
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame2)
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    black_canvas = np.zeros((128, 128, 3), dtype="uint8")
                    mp_drawing.draw_landmarks(black_canvas, res, None,
                                              mp_drawing.DrawingSpec(thickness=2,
                                                                     color=(254, 254, 254)))
                frame = cv2.resize(black_canvas, (128,128))
            frame = cv2.resize(frame, (128,128))
            frames.append(frame)

            cv2.imwrite(f"train_jpg/{id}/frame_{i:02d}.jpg", frame)
        cap.release()
    print("train 변환 완료")
#
#
# #모든 프레임 저장
# def TestCovertFrameJpg(ids):
#     global black_canvas
#     for id in ids:
#         cap = cv2.VideoCapture('./test/'+id+'.mp4')
#         frames = []
#         make_new_dir(f"test_jpg/{id}")
#         black_canvas = np.zeros((128, 128, 3), dtype="uint8")
#         for i in range(CFG['FPS']):
#             ret, frame = cap.read()
#             if ret == False:
#                 break
#             frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             result = hands.process(frame2)
#
#             if result.multi_hand_landmarks is not None:
#                 for res in result.multi_hand_landmarks:
#                     black_canvas = np.zeros((128, 128, 3), dtype="uint8")
#                     mp_drawing.draw_landmarks(black_canvas, res, None,
#                                               mp_drawing.DrawingSpec(thickness=2,
#                                                                      color=(254, 254, 254)))
#                 frame = cv2.resize(black_canvas, (128,128))
#             frame = cv2.resize(frame, (128,128))
#             frames.append(frame)
#
#             cv2.imwrite(f"test_jpg/{id}/frame_{i:02d}.jpg", frame)
#             print(f"test_jpg/{id}/frame_{i:02d}.jpg 저장 완료")
#         cap.release()
#     print("test 변환 완료")

#모든 프레임 저장
def trainCovertFrameJpg(ids):
    global black_canvas
    for id in ids:
        cap = cv2.VideoCapture('./train/'+id+'.mp4')
        frames = []
        make_new_dir(f"train_jpg/{id}")
        black_canvas = np.zeros((128, 128, 3), dtype="uint8")
        for i in range(CFG['FPS']):
            ret, frame = cap.read()
            if ret == False:
                break

            frame = cv2.resize(frame, (128,128))
            frames.append(frame)

            cv2.imwrite(f"train_jpg/{id}/frame_{i:02d}.jpg", frame)
            print(f"train_jpg/{id}/frame_{i:02d}.jpg 저장 완료")
        cap.release()
    print("train 변환 완료")


#모든 프레임 원본그대로 저장
def TestCovertFrameJpg(ids):
    global black_canvas
    for id in ids:
        cap = cv2.VideoCapture('./test/'+id+'.mp4')
        frames = []
        make_new_dir(f"test_jpg/{id}")
        black_canvas = np.zeros((128, 128, 3), dtype="uint8")
        for i in range(CFG['FPS']):
            ret, frame = cap.read()
            if ret == False:
                break

            frame = cv2.resize(frame, (128,128))
            frames.append(frame)

            cv2.imwrite(f"test_jpg/{id}/frame_{i:02d}.jpg", frame)
        cap.release()
    print("test 변환 완료")

def make_new_dir(path) :
    if os.path.isdir(path) == False:
        os.makedirs(path)

# trCovertFrameJpg(train['id'].values)
# trCovertFrameJpg(val['id'].values)
# teCovertFrameJpg(test['id'].values)

trainCovertFrameJpg(df['id'].values)
TestCovertFrameJpg(test['id'].values)

