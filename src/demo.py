import os
import glob
import tqdm

import numpy as np
import skimage.io
import torch
import torch.utils.data
import cv2

from datasets.kitti import KITTI
from engine.detector import Detector
from model.squeezedet import SqueezeDet
from utils.config import Config
from utils.model import load_model
import time
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.color import rgba2rgb

def demo(cfg):
    # prepare configurations
    cfg.load_model = '../models/squeezedet_kitti_epoch280.pth'
    cfg.gpus = [-1]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    dataset = KITTI('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)

    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset

    # prepare model & detector
    model = SqueezeDet(cfg)
    model = load_model(model, cfg.load_model)
    detector = Detector(model.to(cfg.device), cfg)

    # prepare images
    sample_images_dir = '../data/samples/kitti/testing/image_2'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.*'))
    sum_time = 0
    print(sample_image_paths)

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        image = cv2.imread(path).astype(np.float32)#skimage.io.imread(path).astype(np.float32)

        print(image.shape)
        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}
        start_time = time.time()

        image, image_meta, _ = preprocess_func(image, image_meta)

        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)

        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}


        _ = detector.detect(inp)
        sum_time = sum_time + (time.time() - start_time)
    print("--- %s seconds ---" % (sum_time))
