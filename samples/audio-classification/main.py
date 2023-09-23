import argparse
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from torchvision.models import resnet18, resnet34, resnet50, resnet101
import torch
import torch.nn as nn
import pandas as pd
import os
import time


class ESC50Data(Dataset):
  def __init__(self, df, opts):
    self.df = df
    self.c2i = {}
    self.i2c = {}
    self.categories = sorted(df['category'].unique())
    self.opts = opts

    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category


  def __len__(self):
    return self.opts.dataset_size


  def __getitem__(self, idx):
   
    idx = idx % len(self.df)
    row = self.df.iloc[idx]
    file_path = os.path.join(self.opts.base_dir + 'audio', row['filename'])
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Adding Noise
    noise = torch.randn(waveform.shape)
    waveform = waveform + noise * 2

    # Adding Effects
    effects = [
      ["lowpass", "-1", "300"], # apply single-pole lowpass filter
      ["speed", "0.8"],  # alter the speed
      ["reverb", "-w"],  # Reverbration 
    ]

    waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
      waveform, sample_rate, effects)

    # Extract Feature 
    spectro = self._get_mel_spectro(waveform2, sample_rate2) # (2, 8, 431)
    label = self.c2i[row['category']]
    return spectro, label

  
  def _get_mel_spectro(self, waveform, sample_rate):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 8

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(waveform)
    return melspec


def main(opts):
    
    df = pd.read_csv(opts.base_dir + 'meta/esc50.csv')
    dataset = ESC50Data(df, opts)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=opts.batch_size,
        prefetch_factor=opts.pf,
        num_workers=opts.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    # ---- sys info ----
    device_name = 'cpu'
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        
    run_name = f"DP-bs-{opts.batch_size}-{device_name}x-{torch.cuda.device_count()}-pf-{opts.pf}-num_workers-{opts.num_workers}"
    
    if "WANDB_API_KEY" in os.environ:
        import wandb
        wandb.init(
            project=f'simple-job-{os.environ.get("ST_JOB_ID")}',
            config=dict(opts),
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

    # -- model
    resnet_model = None
    if opts.resnet == 18:
        resnet_model = resnet18(pretrained=True)
    if opts.resnet == 34:
        resnet_model = resnet34(pretrained=True)
        resnet_model.fc = nn.Linear(512,50)
    if opts.resnet == 50:
        resnet_model = resnet50(pretrained=True)
    if opts.resnet == 101:
        resnet_model = resnet101(pretrained=True)
        resnet_model.fc = nn.Linear(2048, 50)

    resnet_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    if torch.cuda.device_count() > 1:
        resnet_model = torch.nn.DataParallel(resnet_model)
    
    resnet_model = resnet_model.to(device)

    # -- optim, criterion & scaler
    optimizer = torch.optim.SGD(resnet_model.parameters(), lr=0.01)  # lightweight optimizer
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # --- training
    for _ in range(opts.epochs):
        resnet_model.train()
        skip_factor = 0  # Only start counting after a couple of batches
        num_batches_process = 0
        num_elements_processed = 0

        epoch_time = time.time()
 
        for idx, batch in enumerate(data_loader):
            
            if idx < skip_factor:
                print("----- skipping -----")
                continue

            # Set the start time
            if idx == skip_factor:
                start_time = time.time()

            x, y = batch
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
 
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    y_hat = resnet_model(x)
            else:
                y_hat = resnet_model(x)

            loss = loss_fn(y_hat, y)                
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            num_batches_process += 1
            num_elements_processed = num_batches_process * opts.batch_size

            if idx % 20 == 0:
                speed = num_elements_processed / (time.time() - start_time)
                print(f"# Processing at {round(speed, 2)} samples/sec")


        epoch_time = time.time() - epoch_time
        print("> Epoch time", epoch_time, "secs")
        print("> Average Throughput:", num_elements_processed / epoch_time, "samples/sec")
        print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, required=False)
    parser.add_argument('--batch_size', default=1024, type=int, required=False)
    parser.add_argument('--num_workers', default=4, type=int, required=False)
    parser.add_argument('--pf', default=2, type=int, required=False)
    parser.add_argument('--dataset_size', default=4000000, type=int, required=False)
    parser.add_argument('--base_dir', default='/mnt/ESC-50/', type=str, required=False)
    parser.add_argument('--resnet', default=50, type=int, required=False)
    args = parser.parse_args()

    main(args)

