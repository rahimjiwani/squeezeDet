from utils.config import Config
from utils.misc import load_dataset
from utils.compute_dataset_mean_and_std import compute_dataset_mean_and_std

cfg = Config().parse('eval --dataset kitti'.split(' '))
dataset = load_dataset(cfg.dataset)('trainval', cfg)

mean, std = compute_dataset_mean_and_std(dataset)

print('Dataset\'s RGB mean: ', mean)
print('Dataset\'s RGB std: ', std)