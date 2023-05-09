import argparse
import numpy as np
import time
from argparse import Namespace
import pandas as pd
import os 
import wandb

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    RandomHorizontalFlipVideo,
)

from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample, 
    Normalize,
    Permute,
)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int, required=False)
parser.add_argument('--batch_size', default=48, type=int, required=False)
parser.add_argument('--num_workers', default=4, type=int, required=False)
parser.add_argument('--dataset_sz', default=7985, type=int, required=False)
args = parser.parse_args()


GLOBAL_ARGS = {
    "dataset_sz": args.dataset_sz,
    "project_name": '3dbenchmark',
    "batch_size": args.batch_size,
    "prefetch_factor": 2,
    "num_workers": args.num_workers,
    "epochs": args.epochs,
}

opts = Namespace(**GLOBAL_ARGS)

device_name = 'cpu'
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name()
    run_name = f"DP-{opts.project_name}-bs-{opts.batch_size}-{torch.cuda.get_device_name()}x-{torch.cuda.device_count()}-pf-{opts.prefetch_factor}-num_workers-{opts.num_workers}"
else:
    run_name = f"DP-{opts.project_name}-bs-{opts.batch_size}-pf-{opts.prefetch_factor}-num_workers-{opts.num_workers}"

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 8
frames_per_second = 30

# Note that this transform is specific to the slow_R50 model.
transform=Compose(
    [
        UniformTemporalSubsample(num_frames),
        RandomHorizontalFlipVideo(),
        Lambda(lambda x: x/255.0),
        ShortSideScale(
            size=side_size
        ),
        CenterCropVideo(crop_size=(crop_size, crop_size)),
        Normalize(mean, std),
    ]
)

class Charades(Dataset):
    
    def __init__(self, opts, df, path):
        self.opts = opts
        self.df = df
        self.path = path

        self.c2i = {}
        self.i2c = {}
        self.categories = sorted(df['scene'].unique())

        for i, category in enumerate(self.categories):
          self.c2i[category]=i
          self.i2c[i]=category
            
            
        
    def __getitem__(self, index):
        index = index % (len(self.df))
        row = df.iloc[index]
        video_id = row['id']
        video_path = os.path.join(self.path, video_id + '.mp4')
        reader = torchvision.io.read_video(video_path, pts_unit='sec')
        video_data = reader[0].permute(3, 0, 1, 2)
        video_data = transform(video_data)
        
        label = self.c2i[row['scene']] #actions
        return video_data, label
    
    
    def __len__(self):
        return self.opts.dataset_sz
    
df = pd.read_csv('/mnt/Charades/Charades_v1_train.csv')

dataset = Charades(opts, df, '/mnt/video-dataset/Charades_v1_480')
data_loader = DataLoader(
    dataset, 
    batch_size=opts.batch_size,
    prefetch_factor=opts.prefetch_factor,
    num_workers=opts.num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f"torch cuda device count: {torch.cuda.device_count()}")
    print(f"torch cuda current device: {torch.cuda.current_device()}")
    print(f"torch num threads: {torch.get_num_threads()}")
    print()

print(f"@run: {run_name}")
print(f"@args: {opts}")
print()

print("@device", device)

model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
model = torch.nn.DataParallel(model).to(device)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lightweight optimizer
loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()


for i in range(opts.epochs):
    
    skip_factor = 5  # Only start counting after a couple of batches
    num_batches_process = 0
    num_elements_processed = 0

    epoch_time = time.time()
    batch_time = time.time()
    losses = []
    
    for idx, batch in enumerate(data_loader):
        
        if idx < skip_factor:
            print("----- skipping -----")
            continue

        # Set the start time
        if idx == skip_factor:
            start_time = time.time()
            
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            losses.append(loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        num_batches_process += 1
        num_elements_processed = num_batches_process * opts.batch_size
        
        if idx % 10 == 0:
            speed = num_elements_processed / (time.time() - start_time)
            print(f"# Processing at {round(speed, 2)} samples/sec")


        # print("Batch throughput: ", opts.batch_size / (time.time() - batch_time), "samples/sec")
        batch_time = time.time()


    epoch_time = time.time() - epoch_time
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    print(f"> Epoch - {i}")
    print("> Epoch time", epoch_time, "secs")
    print("> Average Throughput:", num_elements_processed / epoch_time, "samples/sec")
    print("> Average Loss:", mean_loss)
    print()