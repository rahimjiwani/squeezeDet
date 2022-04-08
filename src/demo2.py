import os
import glob
import tqdm

import numpy as np
import skimage.io
import torch
import torch.utils.data

from datasets.kitti import KITTI
from engine.detector import Detector
from model.squeezedet import SqueezeDet
from utils.config import Config
from utils.model import load_model

from utils.config import Config
from utils.misc import init_env


cfg = Config().parse()
cfg.load_model = '../models/squeezedet_kitti_epoch280.pth'
cfg.gpus = [-1]  # -1 to use CPU
cfg.debug = 2  # to visualize detection boxes
model = SqueezeDet(cfg)
model = load_model(model, cfg.load_model)
detector = Detector(model.to(cfg.device), cfg)

