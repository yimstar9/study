import random
import pandas as pd
import numpy as np
import os
import cv2
os.chdir("E:\GoogleDrive\pycv\리모콘 제스쳐")
os.getcwd()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':20,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

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

train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, random_state=CFG['SEED'])


class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, transforms=None):
        self.video_path_list = video_path_list
        self.label_list = label_list
        self.transforms = transforms
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
        for i in range(CFG['FPS']):
            _, img = cap.read()

            if self.transforms:
                #img=img.transpose(0,1,2)
                img = self.transforms(image=img)['image']
            #img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
            #img = img / 255.
            # if img.type == 'tensor':
            #     frames.append(img.numpy())
            # else:
            #     frames.append(img)
            try:
               frames.append(img.numpy())
            except Exception as e:
                print('예외가 발생했습니다.', e)
        return torch.FloatTensor(np.array(frames)).permute(1, 0, 2, 3)
        #return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)


train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'] , CFG['IMG_SIZE']),
    #A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    A.Rotate(limit=20, p=1),
    #A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9), rotate_limit=0,p=1),#스케일
    #A.ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=0, rotate_limit=20,p=1), #shift
    #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(0.5, 0.9), rotate_limit=30,p=1,border_mode=cv2.BORDER_REPLICATE),
    #A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.8), contrast_limit=0,p=1),
    A.Normalize(mean, std),
    A.pytorch.transforms.ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'] , CFG['IMG_SIZE']),
    #A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    A.Rotate(limit=20, p=1),
    #A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9), rotate_limit=0,p=1),#스케일
    #A.ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=0, rotate_limit=20,p=1), #shift
    #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(0.5, 0.9), rotate_limit=30,p=1,border_mode=cv2.BORDER_REPLICATE),
    #A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.8), contrast_limit=0,p=1),
    A.Normalize(mean, std),
    A.pytorch.transforms.ToTensorV2()
])

# frames = []
# cap = cv2.VideoCapture('train/TRAIN_277.mp4')
# for i in range(CFG['FPS']):
#     _, img = cap.read()
#
#     if train_transform is not None:
#         # img=img.transpose(0,1,2)
#         img = train_transform(image=img)['image']
#     # img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
#     # img = img / 255.
#
#     frames.append(img.numpy())
# z=torch.FloatTensor(np.array(frames)).permute(1, 0, 2,3)


# frames = []
# cap = cv2.VideoCapture('train/TRAIN_277.mp4')
# for _ in range(CFG['FPS']):
#     _, img = cap.read()
#     img = cv2.resize(img, (CFG['IMG_SIZE'], CFG['IMG_SIZE']))
#     img = img / 255.
#     frames.append(img)
# z=torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)

train_dataset = CustomDataset(train['path'].values, train['label'].values,train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val['path'].values, val['label'].values,train_transform)
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
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

test = pd.read_csv('./test.csv')

test_dataset = CustomDataset(test['path'].values, None,test_transform)
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

