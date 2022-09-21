import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
from tqdm import tqdm
import glob


# Import scaletorch and intialize
import scaletorch as st
st.init()


def get_train_transform():
    return t.Compose([
        t.RandomHorizontalFlip(p=0.5),
        t.RandomRotation(15),
        t.RandomCrop(204),
        t.ToTensor(),
        t.Normalize((0, 0, 0), (1, 1, 1))
    ])


def accuracy(preds, trues):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    acc = np.sum(acc) / len(preds)

    return acc * 100


class ImageDataset(Dataset):
    def __init__(self, transforms=None):
        print("Loading dataset")

        super().__init__()
        self.transforms = transforms
        self.imgs = glob.glob("/mnt/xray-dataset/**/*")
        self.imgs = self.imgs[:100] # Just for quick demonstration

    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        print(f'Downloading {image_name}')
        
        img = Image.open(st.file(image_name))
        img = img.resize((224, 224)).convert('RGB')

        # Preparing class label
        parts = image_name.split('/')
        label_name = parts[-2]
        label = 1 if label_name == 'NORMAL' else 0
        label = torch.tensor(label, dtype=torch.float32)

        # Apply Transforms on image
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def main(args):
    train_dataset = ImageDataset(transforms=get_train_transform())

    train_data_loader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        prefetch_factor=1
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = resnet50(pretrained=True)

    model.fc = nn.Sequential(
        nn.Linear(2048, 1, bias=True),
        nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCELoss()
    epochs = args.epochs
    model.to(device)

    print('Training has begun...')
    writer = SummaryWriter(f'{st.get_artifacts_dir()}/tensorboard')
    for epoch in range(epochs):

        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        for images, labels in tqdm(train_data_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.reshape((labels.shape[0], 1))  # [N, 1] - to match with preds shape

            # Reseting Gradients
            optimizer.zero_grad()

            # Forward
            preds = model(images)

            # Calculating Loss
            _loss = criterion(preds, labels)
            loss = _loss.item()

            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = accuracy(preds, labels)
            epoch_acc.append(acc)

            # Backward
            _loss.backward()
            optimizer.step()

            # Track metrics
            st.track(epoch=epoch, 
                    metrics={'loss' : loss, 'acc' : acc}, 
                    tuner_default='loss')
        
        
        end_time = time.time()
        total_time = end_time - start_time

        loss = np.mean(epoch_loss)
        acc = np.mean(epoch_acc)

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('acc', acc, epoch)

        st.torch.save(model.state_dict(), "model.pth", metadata={'epoch' : 5, 'loss': loss, 'acc' : acc})
        
        print(f"Epoch: {epoch + 1} | Loss: {loss} | Acc: {acc} | Time: {total_time} ")
        


    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, required=True)
    parser.add_argument('--batch_size', default=16, type=int, required=True)
    parser.add_argument('--lr', default=0.0001, type=float, required=True)

    parser.add_argument('--no_cuda', action='store_true', default=True,
                        help='disables CUDA training')

    args = parser.parse_args()
    print(args)
    main(args)
