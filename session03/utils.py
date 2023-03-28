import wandb
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display, Markdown
from fastai.vision.all import *

import params

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua): return True
    else: return False