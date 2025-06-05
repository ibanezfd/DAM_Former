# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..builder import build_loss
from mmcv.runner import force_fp32
from ..losses import accuracy

@HEADS.register_module()
class DAMFormerHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAMFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.loss_decode = build_loss(dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),)
        self.loss_decode_temp = build_loss(dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=1.0),)

        
    @force_fp32(apply_to=('seg_logit','seg_logit_temp' ))
    def losses(self, seg_logit, seg_logit_temp, seg_label, seg_weight=None, seg_weight_temp=None):
        """Compute segmentation loss."""
        loss = dict()
        #for temp
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)

        seg_logit_temp = resize(
            input=seg_logit_temp,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler_temp is not None:
            seg_weight_temp = self.sampler_temp.sample(seg_logit_temp, seg_label)

        seg_label = seg_label.squeeze(1)

        self.loss_decode.debug = self.debug
        self.loss_decode_temp.debug = self.debug
        
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        
        loss['loss_seg_temp'] = self.loss_decode_temp(
            seg_logit_temp,
            seg_label,
            weight=seg_weight_temp,
            ignore_index=self.ignore_index)
        
        loss['loss_seg'] = loss['loss_seg'] + loss['loss_seg_temp'] 

        if self.debug and hasattr(self.loss_decode, 'debug_output'):
            self.debug_output.update(self.loss_decode.debug_output)
        if self.debug and hasattr(self.loss_decode_temp, 'debug_output'):
            self.debug_output.update(self.loss_decode_temp.debug_output)
        return loss

    def forward(self, inputs):
        return inputs
    
    def forward_train(self,
                      inputs,
                      img_metas,
                      gt_semantic_seg,
                      train_cfg,
                      seg_weight=None,
                      return_logits=False):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.debug_output = {}

        seg_logits, seg_logits_temp = self.forward(inputs)
        losses = self.losses(seg_logits, seg_logits_temp, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
            losses['logits_temp'] = seg_logits_temp 
        return losses







