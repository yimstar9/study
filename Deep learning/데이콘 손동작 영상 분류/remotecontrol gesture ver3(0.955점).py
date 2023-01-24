##mediapipe써서 영상에서 hand추출 후 cnn모델링
#f1 score 0.955
import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# import torchvision.models as models
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import mediapipe as mp
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
black_canvas = np.zeros((128, 128, 3), dtype="uint8")



os.chdir("E:\GoogleDrive\pycv\리모콘 제스쳐")
os.getcwd()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}

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

train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])


class CustomDataset(Dataset):

    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list
        #self.video_id = video_id
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        black_canvas = np.zeros((128, 128, 3), dtype="uint8")
        for i in range(CFG['FPS']):
            _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(black_canvas, res, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(thickness=1, color=(i*8,i*8,i*8)),mp_drawing.DrawingSpec(thickness=1, color=(i*8,i*8,i*8)))
                    black_canvas = np.zeros((128, 128, 3), dtype="uint8")
                    mp_drawing.draw_landmarks(black_canvas, res, None,
                                              mp_drawing.DrawingSpec(thickness=2,
                                                                     color=((8-i)*31,(8-i)*31,254)))
            img = cv2.resize(black_canvas, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            frames.append(img)

            # cv2.imshow('img', img)
            # if cv2.waitKey(1) == ord('q'):
            #     break
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)
#
# ####test####
# frames = []
# cap = cv2.VideoCapture('train/TRAIN_280.mp4')
# black_canvas = np.zeros((128, 128, 3), dtype="uint8")
# for i in range(CFG['FPS']):
#     _, img = cap.read()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img)
#     if result.multi_hand_landmarks is not None:
#         for res in result.multi_hand_landmarks:
#             #mp_drawing.draw_landmarks(black_canvas, res, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(thickness=1, color=(i*8,i*8,i*8)),mp_drawing.DrawingSpec(thickness=1, color=(i*8,i*8,i*8)))
#             mp_drawing.draw_landmarks(black_canvas, res, None,
#                                       mp_drawing.DrawingSpec(thickness=2, color=(255,(30-i) * 8, (30-i) * 8)))
# #    res[0]
#     #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img = cv2.resize(black_canvas, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
#     #img = img / 255.
#
#     frames.append(img)
#     # 무한반복
#     # if (cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#     #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#     cv2.imshow('img', img)
#
#     if cv2.waitKey(30) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()



train_dataset = CustomDataset(train['path'].values, train['label'].values)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)


val_dataset = CustomDataset(val['path'].values, val['label'].values)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


class BaseModel(nn.Module):
    def __init__(self, num_classes=5):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (3, 3, 3)),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (2, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((1, 7, 7)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(
            f'\n Epoch[{epoch}],TrainLoss:[{_train_loss:.3f}] ValLoss:[{_val_loss:.3f}] ValF1:[{_val_score:.3f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)

            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

model = BaseModel()
model = nn.DataParallel(model)
model.cuda()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['path'].values, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)

            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['label'] = preds
submit.head()

submit.to_csv('./baseline_submit.csv', index=False)

T= pd.read_csv('./answer.csv')
print(f1_score(list(T.label), preds, average='macro'))