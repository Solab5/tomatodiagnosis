import wandb
import torchvision.models as tvmodels
import pandas as pd
from fastai.vision.all import *

import params
from utils import t_or_f

def download_data():
    """Grab dataset from artifact
    """
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir

def get_df(processed_dataset_dir, is_test=False):
    df = pd.read_csv(processed_dataset_dir / 'data_split.csv')
    if not is_test:
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
    else:
        df = df[df.Stage != 'train'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
        # when passed to datablock, this will return test at index 0 and valid at index 1
    
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

def count_by_class(arr, cidxs): 
    return [(arr == n).sum(axis=(1,2)).numpy() for n in cidxs]

def log_hist(c):
    _, bins, _ = plt.hist(target_counts[c],  bins=10, alpha=0.5, density=True, label='target')
    _ = plt.hist(pred_counts[c], bins=bins, alpha=0.5, density=True, label='pred')
    plt.legend(loc='upper right')
    plt.title(params.BDD_CLASSES[c])
    img_path = f'hist_val_{params.BDD_CLASSES[c]}'
    plt.savefig(img_path)
    plt.clf()
    im = plt.imread(f'{img_path}.png')
    wandb.log({img_path: wandb.Image(f'{img_path}.png', caption=img_path)})

run = wandb.init(project=params.WANDB_PROJECT, job_type="evaluation", tags=['staging'])

artifact = run.use_artifact('solab5/model-registry/Tomato Disease Diagnosis:v0', type='model')

artifact_dir = Path(artifact.download())

_model_pth = artifact_dir.ls()[0]
model_path = _model_pth.parent.absolute()/_model_pth.stem

producer_run = artifact.logged_by()
wandb.config.update(producer_run.config)
config = wandb.config

processed_dataset_dir = download_data()
test_valid_df = get_df(processed_dataset_dir, is_test=True)
test_valid_dls = get_data(test_valid_df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

metrics=[accuracy, error_rate]

cbs = [MixedPrecision()] if config.mixed_precision else []

learn = vision_learner(test_valid_dls, arch=getattr(tvmodels, config.arch), pretrained=config.pretrained, metrics=metrics)

learn.load(model_path);

val_metrics = learn.validate(ds_idx=1)
test_metrics = learn.validate(ds_idx=0)

val_metric_names = ['val_final_loss', 'val_Accuracy', 'val_Error_rate']
val_results = {val_metric_names[i] : val_metrics[i] for i in range(len(val_metric_names))}
for k,v in val_results.items(): 
    wandb.summary[k] = v 

test_metric_names = ['test_final_loss', 'test_Accuracy', 'test_Error_rate']
test_results = {test_metric_names[i] : test_metrics[i] for i in range(len(test_metric_names))}
for k,v in test_results.items(): 
    wandb.summary[k] = v

val_probs, val_targs = learn.get_preds(ds_idx=1)
val_preds = val_probs.argmax(dim=1)
class_idxs = params.BDD_CLASSES.keys()

val_targs

class_idxs

def display_diagnostics(learner, ds_idx=1, return_vals=False):
    """
    Display a confusion matrix for the learner.
    If `dls` is None it will get the validation set from the Learner
    
    You can create a test dataloader using the `test_dl()` method like so:
    >> dls = ... # You usually create this from the DataBlocks api, in this library it is get_data()
    >> tdls = dls.test_dl(test_dataframe, with_labels=True)
    
    See: https://docs.fast.ai/tutorial.pets.html#adding-a-test-dataloader-for-inference
    
    """
    probs, targs = learner.get_preds(ds_idx=ds_idx)
    preds = probs.argmax(dim=1)
    classes = list(params.BDD_CLASSES.values())
    y_true = targs.flatten().numpy()
    y_pred = preds.flatten().numpy()
    
    tdf, pdf = [pd.DataFrame(r).value_counts().to_frame(c) for r,c in zip((y_true, y_pred) , ['y_true', 'y_pred'])]
    countdf = tdf.join(pdf, how='outer').reset_index(drop=True).fillna(0).astype(int).rename(index=params.BDD_CLASSES)
    countdf.index = list(params.BDD_CLASSES.values())
    countdf = countdf/countdf.sum() 
    display(Markdown('###% Of Pixels In Each Class'))
    display(countdf.applymap('{:.1%}'.format))
    
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred,
                                                   display_labels=countdf.index,
                                                   normalize='pred')
    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10) 
    disp.ax_.set_title('Confusion Matrix (by Pixels)', fontdict={'fontsize': 32, 'fontweight': 'medium'})
    fig.show()
    fig.autofmt_xdate(rotation=45)

    if return_vals: return countdf, disp

from sklearn.metrics import ConfusionMatrixDisplay

from IPython.display import display, Markdown

val_count_df, val_disp = display_diagnostics(learner=learn, ds_idx=1, return_vals=True)

wandb.log({'val_confusion_matrix': val_disp.figure_})
val_ct_table = wandb.Table(dataframe=val_count_df)
wandb.log({'val_count_table': val_ct_table})

test_count_df, test_disp = display_diagnostics(learner=learn, ds_idx=0, return_vals=True)
wandb.log({'test_confusion_matrix': test_disp.figure_})
test_ct_table = wandb.Table(dataframe=test_count_df)
wandb.log({'test_count_table': test_ct_table})

run.finish()