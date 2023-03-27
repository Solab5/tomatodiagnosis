import wandb
import timm
import argparse
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

config_defaults = SimpleNamespace(
    batch_size=64,
    epochs=5,
    learning_rate=2e-3,
    img_size=224,
    aug_size=128,
    resize_method="crop",
    model_name="convnext_tiny",
    seed=42,
    wandb_project='deepconc')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--img_size', type=int, default=config_defaults.img_size)
    parser.add_argument('--aug_size', type=int, default=config_defaults.aug_size)
    parser.add_argument('--resize_method', type=str, default=config_defaults.resize_method)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--seed', type=int, default=config_defaults.seed)
    parser.add_argument('--wandb_project', type=str, default='deepconc')
    return parser.parse_args()

def get_dataloader(batch_size, img_size, aug_size, seed=42, method="crop"):
    "Use fastai to get the Dataloader for the Oxford Pets dataset"
    Path.BASE_PATH = path = Path("/notebooks/tomatodiagnosis/tomato/data")
    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2,
                                       seed=seed, 
                                       bs=batch_size,
                                       item_tfms=Resize(img_size, method=method),
                                       batch_tfms=aug_transforms(size=aug_size, min_scale=0.75))
    return dls

def train(config=config_defaults):
    with wandb.init(project=config.wandb_project, config=config):
        config = wandb.config 
        dls = get_dataloader(config.batch_size, config.img_size,config.aug_size, config.seed, config.resize_method)
        learn = vision_learner(dls, 
                               config.model_name, 
                               metrics=[accuracy, error_rate], 
                               cbs=WandbCallback(log_preds=False)).to_fp16()
        learn.fine_tune(config.epochs, config.learning_rate)


if __name__ == "__main__":
    args = parse_args()
    train(config=args)