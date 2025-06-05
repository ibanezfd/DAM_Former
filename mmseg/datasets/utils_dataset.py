import numpy as np
import yaml

import torch
import torch
from torch.nn import functional as F
import pdb

def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def filter_dates(mask, clouds:bool=2, area_threshold:float=0.5, proba_threshold:int=60):
    """ Mask : array T*2*H*W
        Clouds : 1 if filter on cloud cover, 0 if filter on snow cover, 2 if filter on both
        Area_threshold : threshold on the surface covered by the clouds / snow 
        Proba_threshold : threshold on the probability to consider the pixel covered (ex if proba of clouds of 30%, do we consider it in the covered surface or not)
        Return array of indexes to keep
    """
    dates_to_keep = []
    
    for t in range(mask.shape[0]):
        if clouds != 2:
            cover = np.count_nonzero(mask[t, clouds, :,:]>=proba_threshold)
        else:
            cover = np.count_nonzero((mask[t, 0, :,:]>=proba_threshold)) + np.count_nonzero((mask[t, 1, :,:]>=proba_threshold))
        cover /= mask.shape[2]*mask.shape[3]
        if cover < area_threshold:
            dates_to_keep.append(t)

    return dates_to_keep


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)



def pad_collate_train(dict, pad_value=0):
    
       
    _imgs   = [i['img'] for i in dict]    
    _gt     = [i['gt_semantic_seg'] for i in dict] 
    _dates  = [i['dates'] for i in dict]
    _sp_img = [i['sp_image'] for i in dict] 


    sizes = [e.shape[0] for e in _sp_img]
    m = max(sizes)
    padded_data, padded_dates = [],[]
    if not all(s == m for s in sizes):
        for data, date in zip(_sp_img, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:

        padded_data = _sp_img
        padded_dates = _dates
          
    batch = {
             "patch": torch.stack(_imgs, dim=0),
             "spatch": torch.stack(padded_data, dim=0),
             "dates": torch.stack(padded_dates, dim=0),
             "gt_semantic_seg": torch.stack(_gt, dim=0),
             }  
    return batch



def pad_collate_predict(dict, pad_value=0):
    
    _imgs   = [i['patch'] for i in dict]
    _sen    = [i['spatch'] for i in dict] 
    _dates  = [i['dates'] for i in dict]
    _mtd    = [i['mtd'] for i in dict]
    _ids   = [i['id'] for i in dict] 


    sizes = [e.shape[0] for e in _sen]
    m = max(sizes)
    padded_data, padded_dates = [],[]
    if not all(s == m for s in sizes):
        for data, date in zip(_sen, _dates):
            padded_data.append(pad_tensor(data, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date, m, pad_value=pad_value))
    else:
        padded_data = _sen
        padded_dates = _dates
          
    batch = {
             "patch": torch.stack(_imgs, dim=0),
             "spatch": torch.stack(padded_data, dim=0),
             "dates": torch.stack(padded_dates, dim=0),
             "mtd" : torch.stack(_mtd, dim=0),
             "id": _ids,
            }  
    return batch

# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import functools



def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data,
                 stack=False,
                 padding_value=0,
                 cpu_only=False,
                 pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()



def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)

'''
    _dates = []
    _sp_img = []
    
    for i in range(0, len(batch)):
        _dates.append(batch[i]['dates'])
        _sp_img.append(batch[i]['sp_image'])
        
    sizes = [e.shape[0] for e in _sp_img]

    m = max(sizes)
    
    padded_data, padded_dates = [],[]
    
    if not all(s == m for s in sizes):
        raise TypeError(sizes)
        
        for data, date in zip(_sp_img, _dates):
            data_t = torch.from_numpy(data)
            date_t = torch.from_numpy(date)
            padded_data.append(pad_tensor(data_t, m, pad_value=pad_value))
            padded_dates.append(pad_tensor(date_t, m, pad_value=pad_value))
            
        sizes = [e.shape[0] for e in padded_dates]
        raise TypeError(sizes)
        for i in range(0,len(batch)):
            batch[i]['dates'] = padded_data[i].numpy()
            #batch['sp_image'] = (torch.stack(padded_data, dim=0)).numpy()
            #batch['dates'] = (torch.stack(padded_dates, dim=0)).numpy()
'''