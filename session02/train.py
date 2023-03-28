import argparse, os
import wandb
from pathlib import Path
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

import params
from utils import t_or_f

default_config = SimpleNamespace(
    framework="fastai",
    img_size=(224, 224),
    batch_size=64,
    augment=True, # use data augmentation
    epochs=6, 
    arch="resnet18",
    lr=2e-3,
    pretrained=True,  # whether to use pretrained encoder
    mixed_precision=True,
    seed=42,
    log_preds=False
)

def parse_args():
    argparser = argparse.ArgumentParser(description = "process hyperparameters")
    argparser.add_argument('--img_size', type=int, default=default_config.img_size, help='img_size')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch_size')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--arch', type=str, default=default_config.arch, help='timm backbone architecture')
    argparser.add_argument('--augment', type=t_or_f, default=default_config.augment, help='Use Image Augmentation')
    argparser.add_argument('--seed', type=str, default=default_config.seed, help='Random Seed')
    argparser.add_argument('--log_preds', type=t_or_f, default=default_config.log_preds, help='Log model Predictions')
    argparser.add_argument('--pretrained', type=t_or_f, default=default_config.pretrained, help='Use pretrained model')
    argparser.add_argument('--mixed_precision', type=t_or_f, default=default_config.mixed_precision, help='use fp16')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def download_data():
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    if not is_test:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage == 'test'].reset_index(drop=True)
    
    ## Assign paths
    df["image_fname"] = [processed_dataset_dir/f'{f}' for f in df.File_Name.values]
    return df

def get_data(df, bs=64, img_size=(224, 224), augment=True):
    block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("Label"),
                  splitter=ColSplitter(),
                  item_tfms=Resize(img_size),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs)

def log_metrics(learn):
    scores = learn.validate()
    metric_names = ['final_loss', 'Accuracy', 'Error_rate']
    final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
    for k,v in final_results.items(): 
        wandb.summary[k] = v  

def train(config):
    set_seed(config.seed, reproducible=True)
    run = wandb.init(project=params.WANDB_PROJECT, job_type="training", config=config)

    config = wandb.config    
    
    processed_dataset_dir = download_data()
    df = get_df(processed_dataset_dir)

    dls = get_data(df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

    metrics=[accuracy, error_rate]
    learn = vision_learner(dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, metrics=metrics)

    cbs = [WandbCallback(log_preds=True, log_model=True), 
           SaveModelCallback(monitor='valid_loss')]
    cbs += ([MixedPrecision()] if config.mixed_precision else [])

    learn.fine_tune(config.epochs, config.lr, cbs=cbs)
    
    log_metrics(learn)

    wandb.finish()

if __name__ == '__main__':
    parse_args()
    train(default_config)