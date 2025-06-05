# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Support for seg_weight and forward_with_aux
# - Update debug output system

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from ..utils.dacs_transforms import get_mean_std
from ..utils.visualization import prepare_debug_out, subplotimg
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.automatic_debug = True
        self.debug = False
        self.debug_output = {}
        if train_cfg is not None and 'log_config' in train_cfg:
            self.debug_img_interval = train_cfg['log_config']['img_interval']
        self.local_iter = 0

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, sp_image=None, dates=None):
        """Extract features from images."""

        if sp_image==None:
            x = self.backbone(img)
        else: 
            #good
            x = self.backbone(img, sp_image=sp_image, dates=dates)
        if self.with_neck:
            x = self.neck(x)
        return x

    def generate_pseudo_label(self, img, img_metas, sp_image, dates):
        self.update_debug_state()
        if self.debug:
            self.debug_output = {
                'Image': img,
            }
        if sp_image==None:
            out = self.encode_decode(img, img_metas)
        else: 
            out = self.encode_decode(img, img_metas, sp_image, dates)
        
        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
            
            if sp_image!=None:
                self.debug_output['Pred'] = out[0].cpu().numpy()
                self.debug_output['Pred_temp'] = out[1].cpu().numpy()
            else:
                self.debug_output['Pred'] = out.cpu().numpy()
            
            ##test
            #self.debug_output['Pred'] = out.cpu().numpy()

        return out

    def encode_decode(self, img, img_metas, sp_image=None, dates=None, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        
        if sp_image==None:
            x = self.extract_feat(img)
        else: 
            x = self.extract_feat(img, sp_image, dates)
        
        out = self._decode_head_forward_test(x, img_metas)

        if sp_image!=None:
            out, out_t = out

        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        if sp_image!=None:
            out = [out, out_t]
        return out

    def forward_with_aux(self, img, img_metas, sp_image=None, dates=None):
        self.update_debug_state()

        ret = {}
        if sp_image==None:
            x = self.extract_feat(img)
        else: 
            x = self.extract_feat(img, sp_image, dates)

        out = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        ret['main'] = out

        if self.with_auxiliary_head:
            assert not isinstance(self.auxiliary_head, nn.ModuleList)
            out_aux = self.auxiliary_head.forward_test(x, img_metas, self.test_cfg)
            out_aux = resize(
                input=out_aux,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ret['aux'] = out_aux

        return ret

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_logits=False):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        #good
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg,
                                                     seg_weight, return_logits)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas, sp_image=None, dates=None):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self,
                                      x,
                                      img_metas,
                                      gt_semantic_seg,
                                      seg_weight=None):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg, seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""

        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def update_debug_state(self):
        self.debug_output = {}
        if self.automatic_debug:
            self.debug = (self.local_iter % self.debug_img_interval == 0)
        self.decode_head.debug = self.debug
        if self.with_auxiliary_head:
            self.auxiliary_head.debug = self.debug

    def forward_test(self, imgs, img_metas, sp_image=None, dates=None, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            if sp_image == None:
                return self.simple_test(imgs[0], img_metas[0], **kwargs)
            else:
                return self.simple_test(imgs[0], img_metas[0], sp_image[0],
                      dates[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)


    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      sp_image=None,
                      dates=None,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()

        if sp_image==None:
            x = self.extract_feat(img)
        else: 
            x = self.extract_feat(img, sp_image, dates)

        losses = dict()
        if return_feat:
            losses['features'] = x
 
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      return_logits)
        #here?
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        if self.debug:
            self.process_debug(img, img_metas)

        self.local_iter += 1
        return losses

    def process_debug(self, img, img_metas):
        self.debug_output = {
            'Image': img,
            **self.decode_head.debug_output,
        }
        if self.with_auxiliary_head:
            self.debug_output.update(
                add_prefix(self.auxiliary_head.debug_output, 'Aux'))
        if self.automatic_debug:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'encdec_debug')
            os.makedirs(out_dir, exist_ok=True)
            means, stds = get_mean_std(img_metas, img.device)
            for j in range(img.shape[0]):
                rows, cols = 1, len(self.debug_output)
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.92,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                for k, (n, v) in enumerate(self.debug_output.items()):
                    subplotimg(axs[k],
                               **prepare_debug_out(n, v[j], means, stds))
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
            del self.debug_output

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, sp_image, dates):
        """Inference by sliding-window with overlap.
        #todo
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta)
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale, sp_image, dates):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta, sp_image, dates)
        
        if sp_image!=None:
            if rescale:
                # support dynamic shape for onnx
                if torch.onnx.is_in_onnx_export():
                    size = img.shape[2:]
                else:
                    size = img_meta[0]['ori_shape'][:2]
                seg_logit[0] = resize(
                    seg_logit[0],
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                seg_logit[1] = resize(
                    seg_logit[1],
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
        else:

            if rescale:
                # support dynamic shape for onnx
                if torch.onnx.is_in_onnx_export():
                    size = img.shape[2:]
                else:
                    size = img_meta[0]['ori_shape'][:2]
                seg_logit = resize(
                    seg_logit,
                    size=size,
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
        
        # #test 
        # if rescale:
        #         # support dynamic shape for onnx
        #         if torch.onnx.is_in_onnx_export():
        #             size = img.shape[2:]
        #         else:
        #             size = img_meta[0]['ori_shape'][:2]
        #         seg_logit = resize(
        #             seg_logit,
        #             size=size,
        #             mode='bilinear',
        #             align_corners=self.align_corners,
        #             warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale, sp_image=None, dates=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale, sp_image, dates)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale, sp_image, dates)
        
        if sp_image!=None:
            if hasattr(self.decode_head, 'debug_output_attention') and \
                    self.decode_head.debug_output_attention:
                output = seg_logit
            else:
                output= list()
                output_a = F.softmax(seg_logit[0], dim=1)
                output.append(output_a)
                output_b = F.softmax(seg_logit[1], dim=1)
                output.append(output_b)
            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output[0] = output[0].flip(dims=(3, ))
                    output[1] = output[1].flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output[0] = output[0].flip(dims=(2, ))
                    output[1] = output[1].flip(dims=(2, ))
        
        else:
            if hasattr(self.decode_head, 'debug_output_attention') and \
                    self.decode_head.debug_output_attention:
                output = seg_logit
            else:
                output = F.softmax(seg_logit, dim=1)

            flip = img_meta[0]['flip']
            if flip:
                flip_direction = img_meta[0]['flip_direction']
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    output = output.flip(dims=(3, ))
                elif flip_direction == 'vertical':
                    output = output.flip(dims=(2, ))
        
        # #test
        # if hasattr(self.decode_head, 'debug_output_attention') and \
        #             self.decode_head.debug_output_attention:
        #         output = seg_logit
        # else:
        #     output = F.softmax(seg_logit, dim=1)

        # flip = img_meta[0]['flip']
        # if flip:
        #     flip_direction = img_meta[0]['flip_direction']
        #     assert flip_direction in ['horizontal', 'vertical']
        #     if flip_direction == 'horizontal':
        #         output = output.flip(dims=(3, ))
        #     elif flip_direction == 'vertical':
        #         output = output.flip(dims=(2, ))
        return output

    def simple_test(self, img, img_meta, sp_image=None, dates=None, rescale=True):
        """Simple test with single image."""
        #sp_image=None
        #dates=None
        seg_logit = self.inference(img, img_meta, rescale, sp_image, dates)
        
        if sp_image!=None:
            if hasattr(self.decode_head, 'debug_output_attention') and \
                    self.decode_head.debug_output_attention:
                seg_pred = seg_logit[0][:, 0]
                seg_pred_temp = seg_logit[1][:, 0]
            else:
                seg_pred = seg_logit[0].argmax(dim=1)
                seg_pred_temp = seg_logit[1].argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                seg_pred_temp = seg_pred_temp.unsqueeze(0)
                return [seg_pred, seg_pred_temp]
            seg_pred = seg_pred.cpu().numpy()
            seg_pred_temp = seg_pred_temp.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred) #seg_pred = list(temp) #AQUI SE CAMBIA PARA SACAR LAS IMAGENES TEMPORAL
            #seg_pred = [seg_pred, seg_pred_temp]
            #seg_pred.append(seg_pred_temp)
            #raise TypeError(seg_pred_temp[0].shape, seg_pred[0].shape)

            return seg_pred #seg_pred
        else:
            if hasattr(self.decode_head, 'debug_output_attention') and \
                    self.decode_head.debug_output_attention:
                seg_pred = seg_logit[:, 0]
            else:
                seg_pred = seg_logit.argmax(dim=1)
            if torch.onnx.is_in_onnx_export():
                # our inference backend only support 4D output
                seg_pred = seg_pred.unsqueeze(0)
                return seg_pred
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
        
        # #test 
        # if hasattr(self.decode_head, 'debug_output_attention') and \
        #             self.decode_head.debug_output_attention:
        #         seg_pred = seg_logit[:, 0]
        # else:
        #     seg_pred = seg_logit.argmax(dim=1)
        # if torch.onnx.is_in_onnx_export():
        #     # our inference backend only support 4D output
        #     seg_pred = seg_pred.unsqueeze(0)
        #     return seg_pred
        # seg_pred = seg_pred.cpu().numpy()
        # # unravel batch dim
        # seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
