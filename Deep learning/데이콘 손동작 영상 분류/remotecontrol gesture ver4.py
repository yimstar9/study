##마지막이미지로만 cnn
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

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
    'EPOCHS':30,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}
mean = [0.45, 0.45, 0.45]
std = [0.250, 0.250, 0.250]
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정

df = pd.read_csv('./tr.csv')
test = pd.read_csv('./te.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)

train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'] , CFG['IMG_SIZE']),
    #A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    A.Rotate(limit=20, p=0.5),
    #A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9), rotate_limit=0,p=1),#스케일
    #A.ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=0, rotate_limit=20,p=1), #shift
    #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(0.5, 0.9), rotate_limit=20,p=0.5,border_mode=cv2.BORDER_REPLICATE),
    #A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.8), contrast_limit=0,p=1),
    A.Normalize(mean, std),
    A.pytorch.transforms.ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'] , CFG['IMG_SIZE']),
    #A.HorizontalFlip(), # Same with transforms.RandomHorizontalFlip()
    #A.Rotate(limit=20, p=0.5),
    #A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.5, 0.9), rotate_limit=0,p=1),#스케일
    #A.ShiftScaleRotate(shift_limit=(0.2, 0.2), scale_limit=0, rotate_limit=20,p=1), #shift
    #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(0.5, 0.9), rotate_limit=20,p=0.5,border_mode=cv2.BORDER_REPLICATE),
    #A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.8), contrast_limit=0,p=1),
    A.Normalize(mean, std),
    A.pytorch.transforms.ToTensorV2()
])


train_dataset = CustomDataset(train['path'].values, train['label'].values,train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

# np.mean(train_dataset.__getitem__(3)[0].numpy(),axis=(1,2))
# np.std(train_dataset.__getitem__(2)[0].numpy(),axis=(1,2))

val_dataset = CustomDataset(val['path'].values, val['label'].values,test_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

#
# class BaseModel(nn.Module):
#     def __init__(self, num_classes=5):
#         super(BaseModel, self).__init__()
#         self.feature_extract = nn.Sequential(
#             nn.Conv3d(3, 8, (3, 3, 3)),
#             nn.ReLU(),
#             nn.BatchNorm3d(8),
#             nn.MaxPool3d(2),
#             nn.Conv3d(8, 32, (2, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(32),
#             nn.MaxPool3d(2),
#             nn.Conv3d(32, 64, (2, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(64),
#             nn.MaxPool3d(2),
#             nn.Conv3d(64, 128, (2, 2, 2)),
#             nn.ReLU(),
#             nn.BatchNorm3d(128),
#             nn.MaxPool3d((1, 7, 7)),
#         )
#         self.classifier = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = self.feature_extract(x)
#         x = x.view(batch_size, -1)
#         x = self.classifier(x)
#         return x
# layer = nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = 3, stride = 1).to(device)
# #Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1))
# layer.weight.shape
# #torch.Size([1, 3, 3, 3]) #batch_size, channels, height, width의 크기
# weight = layer.weight.detach().cpu().numpy()
# weight.shape

#input size 610개, (3,128,128)
class BaseModel(nn.Module):
    def __init__(self, num_classes=5):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 32, 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((7))
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



test_dataset = CustomDataset(test['path'].values,None,test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    preds = []

    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)

            logit = model(img)
            preds += logit.argmax(1).detach().cpu().numpy().tolist()

    print('Done.')
    return preds


preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['label'] = preds
submit.head()

submit.to_csv('./baseline_submit.csv', index=False)

T= pd.read_csv('./answer.csv')
print(f1_score(list(T.label), preds, average='macro'))