from utils.config import Config
from utils.misc import load_dataset
from utils.compute_dataset_seed_anchors import compute_dataset_anchors_seed


cfg = Config().parse('eval --dataset kitti'.split(' '))
dataset = load_dataset(cfg.dataset)('trainval', cfg)

anchors_seed = compute_dataset_anchors_seed(dataset)

print('Dataset\'s anchors seed: ', anchors_seed)
