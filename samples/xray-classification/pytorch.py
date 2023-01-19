import argparse
import time


import wandb
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as t
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from tqdm import tqdm
import glob


import scaletorch as st


def get_train_transform():
    return t.Compose(
        [
            t.RandomHorizontalFlip(p=0.5),
            t.RandomRotation(15),
            t.RandomCrop(204),
            t.ToTensor(),
            t.Normalize((0, 0, 0), (1, 1, 1)),
        ]
    )


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
        self.imgs = self.imgs * 1e2  # Just for quick demonstration
        print(f"Number of images: {len(self.imgs)}")

    def __getitem__(self, idx):
        image_name = self.imgs[idx]

        img = Image.open(image_name)
        img = img.resize((224, 224)).convert("RGB")

        # Preparing class label
        parts = image_name.split("/")
        label_name = parts[-2]
        label = 1 if label_name == "NORMAL" else 0
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
        prefetch_factor=4,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 1, bias=True), nn.Sigmoid())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    epochs = args.epochs
    model.to(device)

    print("Training has begun...")
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
            # st.track(epoch=epoch,
            #         metrics={'loss' : loss, 'acc' : acc},
            #         tuner_default='loss')

        end_time = time.time()
        total_time = end_time - start_time

        loss = np.mean(epoch_loss)
        acc = np.mean(epoch_acc)

        print(f"Epoch: {epoch + 1} | Loss: {loss} | Acc: {acc} | Time: {total_time} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int, required=True)
    parser.add_argument("--batch_size", default=64, type=int, required=True)
    parser.add_argument("--lr", default=0.0001, type=float, required=True)
    args = parser.parse_args()

    wandb.init(project="dapp-mlds", config=dict(args), name=st.get_trial_id())

    print(args)
    main(args)
