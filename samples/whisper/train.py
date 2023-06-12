import argparse
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from datasets import load_dataset, DatasetDict
from datasets import Audio

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch 
import wandb
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Union


class Data(Dataset):
  def __init__(self, ds, feature_extractor, tokenizer, opts):
    self.ds = ds
    self.feature_extractor = feature_extractor
    self.tokenizer = tokenizer
    self.opts = opts

  def __len__(self):
    return self.opts.dataset_sz


  def __getitem__(self, idx):
    if self.opts.dapp:
        batch = {}
        batch["input_features"] = torch.randn((80, 3000))
        batch["labels"] = torch.randint(1, 50000, (38,))
        return batch

    row = self.ds[idx % len(self.ds)]
    
    waveform = torch.Tensor(row['audio']['array'])
    features = feature_extractor(waveform, sampling_rate=row['audio']["sampling_rate"]).input_features[0]
    labels = tokenizer(row["sentence"]).input_ids
    batch = {}
    batch["input_features"] = features
    batch["labels"] = torch.tensor(labels)
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def main(opts):

    # Dataset prep
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")

    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation")
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")

    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.with_format('torch')
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    run_name = f"DP-bs/gpu-{opts.batch_size}-A100x-{torch.cuda.device_count()}-pf-{opts.prefetch_factor}-num_workers-{opts.num_workers}"
    if opts.dapp:
        run_name = f"{run_name}-dapp"

    train_data = Data(common_voice['train'], feature_extractor, tokenizer, opts)
   
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    # Model and Trainer
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-medium")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-hi",  # change to a repo name of your choice
        per_device_train_batch_size=opts.batch_size,
        dataloader_num_workers=opts.num_workers,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=opts.epochs,
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy="no",
        save_strategy='no',
        generation_max_length=225,
        logging_steps=25,
        report_to="none",
    )


    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_data,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    wandb.init(
        project="whisper-pulkit",
        config=GLOBAL_ARGS.copy(),
        name=run_name,
    )

    # Start Training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int, required=False)
    parser.add_argument('--batch_size', default=48, type=int, required=False)
    parser.add_argument('--num_workers', default=16, type=int, required=False)
    parser.add_argument('--dataset_sz', default=6540, type=int, required=False)
    parser.add_argument('--model_type', default="openai/whisper-small", type=str, required=False)

    opts = parser.parse_args()

    main(opts)