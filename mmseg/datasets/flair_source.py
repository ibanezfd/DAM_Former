# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset
import numpy as np


@DATASETS.register_module()
class FLAIRSDataset(CustomDataset):
    CLASSES  = ('building', 'pervious surface', 'impervious surface', 'bare soil',
               'water', 'coniferous', 'deciduous', 'brushwood',
               'vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land', 'others')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0]]


    def __init__(self, crop_pseudo_margins=None, **kwargs):

        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')

        if crop_pseudo_margins is not None:
            assert kwargs['pipeline'][-1]['type'] == 'Collect'
            print(kwargs['pipeline'])
            kwargs['pipeline'][-1]['keys'].append('valid_pseudo_mask')

        super(FLAIRSDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            split=None,
            **kwargs)

        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [512, 512]

    def pre_pipeline(self, results):
        super(FLAIRSDataset, self).pre_pipeline(results)
        if self.pseudo_margins is not None:
            results['valid_pseudo_mask'] = np.ones(
                self.valid_mask_size, dtype=np.uint8)
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            if self.pseudo_margins[0] > 0:
                results['valid_pseudo_mask'][:self.pseudo_margins[0], :] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[1] > 0:
                results['valid_pseudo_mask'][-self.pseudo_margins[1]:, :] = 0
            if self.pseudo_margins[2] > 0:
                results['valid_pseudo_mask'][:, :self.pseudo_margins[2]] = 0
            # Here, the if statement is absolutely necessary
            if self.pseudo_margins[3] > 0:
                results['valid_pseudo_mask'][:, -self.pseudo_margins[3]:] = 0
            results['seg_fields'].append('valid_pseudo_mask')
        #self.crop_pseudo_margins=crop_pseudo_margins
