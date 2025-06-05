# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .builder import DATASETS
from .custom import CustomDataset
import numpy as np
from skimage import img_as_float
import torch

from pathlib import Path 
import os
import json
import os.path as osp
from collections import OrderedDict
from functools import reduce
import rasterio
import datetime
import pandas as pd
import mmcv
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .pipelines import Compose

from mmseg.core.evaluation.sklearn.metrics import confusion_matrix
# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Additional dataset location logging






@DATASETS.register_module()
class Flair_Dataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """
    CLASSES  = ('building', 'pervious surface', 'impervious surface', 'bare soil',
               'water', 'coniferous', 'deciduous', 'brushwood',
               'vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land', 'others')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0]]

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_sp_dir,
                 ann_dir,
                 list_sp_coords,
                 dataset_domains,
                 dataset_name,

                 img_suffix='.tif',
                 img_sp_suffix='.npy',
                 crop_pseudo_margins=None,
                 seg_map_suffix='.tif',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=CLASSES,
                 palette=PALETTE,
                 ref_year="2021",
                 ref_date="05-15",
                 sat_patch_size =40,
                 num_classes = 13,
                 filter_mask = True,
                 average_month = True,
                 **kwargs,
                 ):

        if crop_pseudo_margins is not None:
            assert pipeline[-1]['type'] == 'Collect'
            pipeline[-1]['keys'].append('valid_pseudo_mask')
        self.dataset_domains = dataset_domains
        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [512, 512]
        self.dataset_name = dataset_name
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir #["PATH_IMG"]
        self.img_sp_dir = img_sp_dir #["PATH_SP_DATA"]

        self.list_sp_coords = list_sp_coords #["SP_COORDS"])

        

        self.img_suffix = img_suffix
        self.img_sp_suffix = img_sp_suffix
        self.ann_dir = ann_dir  #["PATH_LABELS"])
        self.seg_map_suffix = seg_map_suffix


        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.ref_year = ref_year
        self.ref_date = ref_date
        self.sat_patch_size = sat_patch_size
        self.num_classes = num_classes
        self.filter_mask = filter_mask
        self.average_month = average_month

        # join paths if data_root is specifiedvscode-remote://ssh-remote%2Braiden.riken.jp/data/ggeoinfo/datasets/FLAIR2/flair_aerial_train
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.img_sp_dir is None or osp.isabs(self.img_sp_dir)):
                self.img_sp_dir = osp.join(self.data_root, self.img_sp_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)
            if not (self.list_sp_coords is None or osp.isabs(self.list_sp_coords)):
                self.list_sp_coords = osp.join(self.data_root, self.list_sp_coords)

        # load annotations
        with open(self.list_sp_coords, 'r') as file: 
            self.matching_dict = json.load(file)

        self.path_domains = Path(self.img_dir) 
        self.domains = os.listdir(self.path_domains) 
        #self.domains_use = [x for x in self.domains if not any( y in x for y in dataset_domains)]
        self.domains_use = []
        for d in dataset_domains:
            self.domains_use.append([domain for domain in self.domains if all(val in domain for val in d.split(' '))])
        #print(self.domains_use)
        self.img_infos, self.ann_infos, self.img_sp_infos = self.get_data_paths(self.domains_use, self.matching_dict) 

        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')

        


    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def convert_to_train_id(self, results):
    
        label = results['gt_semantic_seg']
                
        #pil_label = Image.open(file)
        id_to_trainid = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 8,
            9: 9,
            10: 10,
            11: 11,
            12: 12,
            13: 13
            }
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        sample_class_stats = {}
        for k, v in id_to_trainid.items():
            k_mask = label == k
            label_copy[k_mask] = v
            n = int(np.sum(k_mask))
            if n > 0:
                sample_class_stats[v] = n
        #new_file = file.replace('.png', '_labelTrainIds.png')
        #assert file != new_file
        #sample_class_stats['file'] = new_file
        results['sample_class_stats'] = sample_class_stats
        #Image.fromarray(label_copy, mode='L').save(new_file)
        return results
    

    def get_data_paths(self, path_domains, matching_dict: dict) -> dict: 
        #### return data paths 
        def list_items(path, filter): 
            for path in Path(path).rglob(filter): 
                yield path.resolve().as_posix()  
        ## data paths dict
        # if dom is xxx read

        img_infos = []
        img_sp_infos = []
        ann_infos = []

        for domain in path_domains: 
            domain = domain[0]
            for area in os.listdir(Path(self.img_dir, domain)): 
                
                aerial = sorted(list(list_items(Path(self.img_dir)/domain/Path(area), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
                sen2sp = sorted(list(list_items(Path(self.img_sp_dir)/domain/Path(area), '*data.npy'))) 
                sprods = sorted(list(list_items(Path(self.img_sp_dir)/domain/Path(area), '*products.txt')))
                smasks = sorted(list(list_items(Path(self.img_sp_dir)/domain/Path(area), '*masks.npy')))
                labels = sorted(list(list_items(Path(self.ann_dir)/domain/Path(area), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
                coords = [] 

                for k in aerial: 
                    coords.append(matching_dict[k.split('/')[-1]]) 

                for img, lab, coord in zip(aerial, labels, coords): 


                    
                    img_info = dict(filename=os.path.basename(img))
                    img_info['ann'] = dict(seg_map=os.path.basename(lab))
                    img_info['img_prefix'] = os.path.dirname(img)
                    ann_info = dict(seg_map=os.path.basename(lab))
                    ann_info['seg_prefix'] = os.path.dirname(lab)
                    ann_infos.append(ann_info)
                    img_infos.append(img_info)

                    img_sp_info = dict(filename_sp=os.path.basename(sen2sp[0]))
                    img_sp_info['sp_file'] = sen2sp[0]
                    img_sp_info['sprods'] = sprods[0]
                    img_sp_info['smasks'] = smasks[0]
                    img_sp_info['coords'] = coord
                    img_sp_infos.append(img_sp_info)

                    sp_patch = self.read_superarea_and_crop(sen2sp[0], coord)
                    sp_dates, sp_raw_dates = self.read_dates(sprods[0])
                    sp_masks2 = self.read_superarea_and_crop(smasks[0], coord)
              

        return img_infos, ann_infos, img_sp_infos

    ###################################################33

    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            img = src_img.read([3,2,1])
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1) # (w,h,c)
            #nir_e = src_img.read([5]) #nir
            nir_e = src_img.read([4,5])
            nir_e = np.swapaxes(nir_e, 0, 2)
            nir_e = np.swapaxes(nir_e, 0, 1)
            return img, nir_e
    
    def read_labels(self, raster_file: str, pix_tokeep:int = 500) -> np.ndarray:
        with rasterio.open(raster_file) as src_label:
            labels = src_label.read()[0]
            labels[labels > self.num_classes] = self.num_classes
            labels = labels-1
            return labels, labels

    def read_superarea_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode='r')
        subset_sp = data[:,:,idx_centroid[0]-int(self.sat_patch_size/2):idx_centroid[0]+int(self.sat_patch_size/2),idx_centroid[1]-int(self.sat_patch_size/2):idx_centroid[1]+int(self.sat_patch_size/2)]
        return subset_sp

    def read_dates(self, txt_file: str) -> np.array:
        with open(txt_file, 'r') as f:
            products= f.read().splitlines()
        diff_dates = []
        dates_arr = []
        for file in products:
            diff_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                              -datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:]))).days           
                             )
            dates_arr.append(datetime.datetime(int(self.ref_year), int(file[15:19][:2]), int(file[15:19][2:])))
        return np.array(diff_dates), np.array(dates_arr)

    def monthly_image(self, sp_patch, sp_raw_dates):
        average_patch, average_dates  = [], []
        month_range = pd.period_range(start=sp_raw_dates[0].strftime('%Y-%m-%d'),end=sp_raw_dates[-1].strftime('%Y-%m-%d'), freq='M')
        for m in month_range:
            month_dates = list(filter(lambda i: (sp_raw_dates[i].month == m.month) and (sp_raw_dates[i].year==m.year), range(len(sp_raw_dates))))
            if len(month_dates)!=0:
                average_patch.append(np.mean(sp_patch[month_dates], axis=0))
                average_dates.append((datetime.datetime(int(self.ref_year), int(self.ref_date.split('-')[0]), int(self.ref_date.split('-')[1])) 
                                     -datetime.datetime(int(self.ref_year), int(m.month), 15)).days           
                                    )
        return np.array(average_patch), np.array(average_dates)

    def __getitem__(self, index):

        # load of all items
        
        results = dict(img_info=self.img_infos[index], ann_info=self.ann_infos[index], img_sp_info=self.img_sp_infos[index])
        results['seg_fields'] = []
        ## aerial images ##
        results['num_images'] = len(self.img_infos)
        results['num_labels'] = len(self.ann_infos)
        results['num_sp'] = len(self.ann_infos)
        
        #image_file = osp.join(results['img_info']['img_prefix'],
         #                       results['img_info']['filename'])

        if results['img_info'].get('img_prefix') is not None:
            filename = osp.join(results['img_info']['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img, img_nir_e = self.read_img(raster_file=filename)  
        img = img.astype(np.float32)
        img_nir_e = img_nir_e.astype(np.float32)
        #img = img_as_float(img)
        #img_nir_e = img_as_float(img) #img = img.astype(np.float32)

        

        results['filename'] = filename
        results['img'] = img
        results['nir_e'] = img_nir_e
        results['ori_filename'] = results['img_info']['filename']
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        ## sp images ##

        sp_dir = results['img_sp_info']['sp_file']
        sp_file_coords = results['img_sp_info']['coords']
        sp_file_products = results['img_sp_info']['sprods']
        sp_file_mask = results['img_sp_info']['smasks']

        sp_patch = self.read_superarea_and_crop(sp_dir, sp_file_coords)
        sp_dates, sp_raw_dates = self.read_dates(sp_file_products)
        sp_mask = self.read_superarea_and_crop(sp_file_mask, sp_file_coords)
        sp_mask = sp_mask.astype(int)

        if self.filter_mask:
            dates_to_keep = filter_dates(sp_mask)
            sp_patch = sp_patch[dates_to_keep]
            sp_dates = sp_dates[dates_to_keep]
            sp_raw_dates = sp_raw_dates[dates_to_keep]

        if self.average_month:
            sp_patch, sp_dates = self.monthly_image(sp_patch, sp_raw_dates)
        
        #sp_patch = img_as_float(sp_patch)

        sp_patch = sp_patch.astype(np.float32)
        sp_patch = np.swapaxes(sp_patch, 1, 3)
        sp_patch = np.swapaxes(sp_patch, 1, 2)
        results['sp_image'] = sp_patch
        results['dates'] = sp_dates
        #"dates": torch.as_tensor(sp_dates, dtype=torch.float),
        #"labels": torch.as_tensor(labels, dtype=torch.float),
        #"slabels": torch.as_tensor(s_labels, dtype=torch.float),
        ## labels ##

        label_file = osp.join(results['ann_info']['seg_prefix'],
                                results['ann_info']['seg_map'])

        labels, s_labels = self.read_labels(raster_file=label_file)  

        results['gt_semantic_seg'] = labels
        results['gt_semantig_seg_sp'] = s_labels

        results = self.convert_to_train_id(results)

        if self.custom_classes:
            results['label_map'] = self.label_map

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
        
        #if type(results['sp_image']) == list:
        #    raise TypeError(type(results['sp_image']))
        
        results = self.pipeline(results)

        #if type(results['sp_image']) == list:
        #    raise TypeError(type(results['sp_image']))

        return results

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for ann_info in self.ann_infos:
            seg_map = osp.join(ann_info['seg_prefix'], ann_info['seg_map'])

            if efficient_test:
                gt_seg_map = seg_map
            else:
                seg_map = osp.join(ann_info['seg_prefix'], ann_info['seg_map'])
                                
                gt_seg_map, _ = self.read_labels(raster_file=seg_map) 
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        #label_file = osp.join(results['ann_info']['seg_prefix'],
        #                        results['ann_info']['seg_map'])
        #print(results[0]["img"][0].shape)
        #print(results)
        gt_seg_maps = self.get_gt_seg_maps(efficient_test) 
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        #raise KeyError(len(results))
        #results tiene el doble de len que el resto
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        #raise TypeError(type(gt_seg_maps[0]), type(results[0]), type(gt_seg_maps), type(results))
        n_im = len(results)
        for i in range(n_im):
            if i == 0:
                cm = confusion_matrix(np.concatenate(np.stack(gt_seg_maps[0],axis=0)).flatten(), np.concatenate(np.stack(results[0],axis=0)).flatten(), labels=list(range(num_classes)))
                #raise TypeError(np.concatenate(np.stack(results[0],axis=0)).flatten(), results[0], cm)
            else:
                cm = cm + confusion_matrix(np.concatenate(np.stack(gt_seg_maps[i],axis=0)).flatten(), np.concatenate(np.stack(results[i],axis=0)).flatten(), labels=list(range(num_classes)))

        return eval_results, cm

#########################################################################

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





         


