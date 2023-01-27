
#영상->프레임별 이미지로 저장
#원본이미지,mediapipe이미지 2개다 학습
#테스트셋은 mediapipe이미지로 변환(손검출안되면 원본이미지로 저장)
#30회 배치2 0.986663196703535
#100회 배치2 0.9134
import random
import pandas as pd
import numpy as np
import os
import cv2
os.chdir("E:\GoogleDrive\pycv\리모콘 제스쳐")
os.getcwd()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob

from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings(action='ignore')
# black_canvas = np.zeros((128, 128, 3), dtype="uint8")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

writer = SummaryWriter('ResNet')

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':1,
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

df = pd.read_csv('./train_jpg.csv')
test = pd.read_csv('./test_jpg.csv')
df2 = pd.read_csv('./train_jpg2.csv')

train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])
train2, val2, _, _ = train_test_split(df2, df2['label'], test_size=0.2, random_state=CFG['SEED'])

train_data = pd.concat([train,train2],ignore_index=True)
val_data = pd.concat([val,val2],ignore_index=True)


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
        folder_lists = glob(path + '\**')
        file_list = [f'frame_{i:02d}.jpg' for i in range(len(folder_lists))]
        for j in range(len(file_list)):
            image = cv2.imread(folder_lists[j])
            image = image / 255.
            frames.append(image)
        return torch.FloatTensor(np.array(frames)).permute(3,0, 1, 2)


train_dataset = CustomDataset(train_data['path'].values, train_data['label'].values)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val['path'].values, val['label'].values)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

#vgg16=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
#13(합성곱층) + 3(풀링층) = 16(전체 계층) = VGG16
class VGGNetModel(nn.Module):
    def __init__(self, num_classes=5):
        super(VGGNetModel, self).__init__()
        self.feature_extract = nn.Sequential(
            #block1
            nn.Conv3d(3, 64, 3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            #block2
            nn.Conv3d(64, 64, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d((1,2,2),stride=2),
            # block3
            nn.Conv3d(64, 128, 3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            # block4
            nn.Conv3d(128, 128, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((1,2,2),stride=2),
            # block5
            nn.Conv3d(128, 256, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            # block6
            nn.Conv3d(256, 256, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            # block7
            nn.Conv3d(256, 256, 3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            nn.MaxPool3d((1,2,2),stride=2),
            # block8
            nn.Conv3d(256, 512, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            # block9
            nn.Conv3d(512, 512, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            # block10
            nn.Conv3d(512, 512, 3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.MaxPool3d((1,2,2),stride=2),
            # block11
            nn.Conv3d(512, 512, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            # block12
            nn.Conv3d(512, 512, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            # block13
            nn.Conv3d(512, 512, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(2,stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1,8,8))
        self.classifier =  nn.Sequential(nn.Linear(32768, 4096),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(),
                                         nn.Linear(4096, 1000),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(),
                                         nn.Linear(1000, 5))
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extract(x)
        x=self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None
    running_loss = 0
    correct = 0
    total = 0

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

            #텐서보드 그래프용 통계
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

        accu = 100. * correct / total



        _val_loss, _val_score = validation(model, criterion, val_loader, device,epoch)
        _train_loss = np.mean(train_loss)
        print(
            f'\nEpoch[{epoch}],T_L:[{_train_loss:.3f}] V_L:[{_val_loss:.3f}] V_F1:[{_val_score:.3f}]')

        # 텐서보드 스칼라 작성
        writer.add_scalar('train loss', _train_loss,epoch)
        writer.add_scalar('train acc', accu,epoch)


        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            best_epoch = epoch
    #텐서보드 모델구조
    writer.add_graph(model, videos)
    writer.close()
    print(f'best epoch:{best_epoch}')
    return best_model


def validation(model, criterion, val_loader, device,epoch):
    model.eval()
    val_loss = []

    preds, trues = [], []
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)

            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            #텐서보드 통계
            running_loss += loss.item()
            _, predicted = logit.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        _val_loss = np.mean(val_loss)
    _val_score = f1_score(trues, preds, average='macro')
    #텐서보드
    testloss = running_loss / len(val_loader)
    accu = 100. * correct / total ################통계
    writer.add_scalar('val loss', testloss,epoch)
    writer.add_scalar('val acc', accu,epoch)

    return _val_loss, _val_score
#model=torch.load('./vggnet_mediapipe_model.pt')
model = VGGNetModel()
#savelr=7.3242e-08
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
#optimizer = torch.optim.Adam(params = model.parameters(), lr = savelr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
# torch.save(infer_model, f'./vggnet_mediapipe_model.pt')


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

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['label'] = preds
submit.head()

submit.to_csv('./vggnet_mediapipe_submit.csv', index=False)

T= pd.read_csv('./answer.csv')
print(f1_score(list(T.label), preds, average='macro'))

# tensorboard --logdir=runs
