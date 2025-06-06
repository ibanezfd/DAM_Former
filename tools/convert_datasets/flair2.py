# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp
from pathlib import Path 
import os 
import mmcv
import numpy as np
from PIL import Image
import rasterio

def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
            
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
        13: 13,
        14: 13,
        15: 13,
        16: 13,
        17: 13,
        18: 13,
        19: 13
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
    sample_class_stats['file'] = file
    #Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert flair annotations to TrainIds')
    parser.add_argument('flair_path', help='flair data path')
    parser.add_argument('--gt-dir', default='ann_dir', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats, dataset_domains):
    with open(osp.join(out_dir, 'sample_class_stats'+dataset_domains+'.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict'+dataset_domains+'.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class'+dataset_domains+'.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def list_items(path, filter): 
            for path in Path(path).rglob(filter): 
                yield path.resolve().as_posix()  
                
def main():
    args = parse_args()
    flair_path = args.flair_path
    out_dir = args.out_dir if args.out_dir else flair_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = flair_path #osp.join(flair_path, args.gt_dir)

    poly_files = []
    dataset_domains = ["D081"] # Can use multiple domains like ["D080_","D060_", "D008_", "D051", "D077"] or ["D081_","D032_","D031_","D046_","D034_"] include lowbar to avoid missmatches
    dataset_name = "D081"
    domains = os.listdir(gt_dir) 
    domains_use = []
    for domain in domains:
        for val in dataset_domains:
            if val in domain:
                domains_use.append(domain)
    #domains_use = [domain for domain in domains if all(val in domain for val in dataset_domains)]
    print(domains_use)
    for domain in domains_use: 
            for area in os.listdir(Path(gt_dir, domain)): 
                labels = sorted(list(list_items(Path(gt_dir)/domain/Path(area), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 

                for lab in labels:
                    poly_file = osp.join(gt_dir, lab)
                    poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    #for poly in mmcv.scandir(
    #        gt_dir, suffix=tuple(f'{i}.tif' for i in range(10)),
    #        recursive=True):
    #    poly_file = osp.join(gt_dir, poly)
    #    poly_files.append(poly_file)
    #poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats_'+dataset_name+'.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats, dataset_name)


if __name__ == '__main__':
    main()
