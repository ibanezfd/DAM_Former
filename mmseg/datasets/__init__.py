# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional datasets


from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .flair_source import FLAIRSDataset
from .flair_target import FLAIRTDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .uda_dataset import UDADataset
from .flair import Flair_Dataset

__all__ = [
    'CustomDataset',
    'Flair_Dataset',
    'build_dataloader',
    'ConcatDataset',
    'RepeatDataset',
    'DATASETS',
    'build_dataset',
    'PIPELINES',
    'CityscapesDataset',
    'GTADataset',
    'SynthiaDataset',
    'UDADataset',
    'ACDCDataset',
    'DarkZurichDataset',
    'FLAIRSDataset',
    'FLAIRTDataset',
]
