import torch
import torch.nn as nn
import torchvision.transforms as T
#import sys                
import mmseg.models.segmentation_models_pytorch as smp  
import sys
sys.path.append("/home/damian/anaconda3/envs/MIC2/lib/python3.8/site-packages/segmentation_models_pytorch")
#import segmentation_models_pytorch as smp 
from mmseg.models.builder import BACKBONES
from mmseg.models.backbones.utae_model import UTAE
from mmseg.models.backbones.fusion_utils import *


@BACKBONES.register_module()
class DAMFormer(nn.Module):
    """ 
     U-Tae implementation for Sentinel-2 super-patc;
     U-Net smp implementation for aerial imagery;
     Added U-Tae feat maps as attention to encoder featmaps of unet.
    """      
    
    def __init__(self):
        
        super(DAMFormer, self).__init__()   

        self.arch_vhr = smp.create_model(
                                        arch="unet", 
                                        encoder_name = "mit_b4",
                                        classes = 13, 
                                        in_channels = 3, 
                                        )
        
        self.arch_hr  = UTAE(
                            input_dim=10,
                            encoder_widths=[64,64,128,128], 
                            decoder_widths=[64,64,128,128],
                            out_conv=[32, 13],
                            str_conv_k=4,
                            str_conv_s=2,
                            str_conv_p=1,
                            agg_mode="att_group", 
                            encoder_norm="group",
                            n_head=16, 
                            d_model=256, 
                            d_k=4,
                            encoder=True,
                            return_maps=True,
                            pad_value=0,
                            padding_mode="reflect",
                            )

        self.fm_utae_featmap_cropped = FM_cropped(self.arch_hr.encoder_widths[0], 
                                                    list(self.arch_vhr.encoder.out_channels[2:]),
                                                    )
        self.fm_utae_featmap_collapsed = FM_collapsed(self.arch_hr.encoder_widths[0], 
                                                        list(self.arch_vhr.encoder.out_channels[2:]),
                                                        ) 
        self.reshape_utae_output = nn.Sequential(nn.Upsample(size=(512,512), mode='nearest'),
                                                nn.Conv2d(self.arch_hr.encoder_widths[0], 13, 1) 
                                            )


        self.inconv = nn.Conv2d(5, 3, 1)
            
    def forward(self, x, sp_image, dates): 
        
        x = self.inconv(x)
        unet_fmaps_enc = self.arch_vhr.encoder(x)
        unet_out = self.arch_vhr.decoder(*unet_fmaps_enc) 
        utae_out, utae_fmaps_dec = self.arch_hr(sp_image, batch_positions=dates)
        
        transform = T.CenterCrop((10, 10))
        utae_last_fmaps_reshape_cropped = transform(utae_fmaps_dec[-1])    
        utae_last_fmaps_reshape_cropped = self.fm_utae_featmap_cropped(utae_last_fmaps_reshape_cropped, [i.size()[-1] for i in unet_fmaps_enc[2:]])    

        ### collapsed fusion module       
        utae_fmaps_dec_squeezed = torch.mean(utae_fmaps_dec[-1][0], dim=(-2,-1))
        utae_last_fmaps_reshape_collapsed = self.fm_utae_featmap_collapsed(utae_fmaps_dec_squeezed, [i.size()[-1] for i in unet_fmaps_enc[2:]])  ### reshape last feature map of utae to match feature maps enc. unet
        ### adding cropped/collasped
        utae_last_fmaps_reshape = [torch.add(i,j) for i,j in zip(utae_last_fmaps_reshape_cropped, utae_last_fmaps_reshape_collapsed)]

        ### modality dropout
        if torch.rand(1) > 0.0:
            unet_utae_fmaps = unet_fmaps_enc
            unet_utae_fmaps[-1] = torch.add(unet_fmaps_enc[-1], utae_last_fmaps_reshape[-1])  ### add utae mask to unet feats map
        else:
            unet_utae_fmaps = unet_fmaps_enc
        utae_out = self.reshape_utae_output(utae_out)
        ### unet decoder
        unet_out = self.arch_vhr.decoder(*unet_utae_fmaps) 
        unet_out = self.arch_vhr.segmentation_head(unet_out)
        
        return unet_out, utae_out
