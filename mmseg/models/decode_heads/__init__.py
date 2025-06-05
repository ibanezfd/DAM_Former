# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .daformer_head import DAFormerHead
from .segformer_head import SegFormerHead
from .damformer_head import DAMFormerHead

__all__ = [
    'SegFormerHead',
    'DAFormerHead',
    'DAMFormerHead',
]
