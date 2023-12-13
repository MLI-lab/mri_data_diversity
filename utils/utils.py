import os
import time
import shutil
from torch.nn.utils import clip_grad_norm_
import re


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def empty_dir(dir, exclude_files=None):
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f not in exclude_files:
                os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def logging(save_dir, text):
    with open(os.path.join(save_dir, 'logs.txt'), 'a') as f:
        f.write(time.strftime('%X %x %Z')+', ')
        f.write(text+'\n')

def flatten(l):
    return [item for sublist in l for item in sublist]

