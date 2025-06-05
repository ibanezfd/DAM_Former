# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional backbones


from .mix_transformer import (MixVisionTransformer,  mit_b4)

from .damformer import DAMFormer

__all__ = [
    'MixVisionTransformer',
    'mit_b4',
    'DAMFormer',
]
