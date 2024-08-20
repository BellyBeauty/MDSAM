import torch
from torch import nn
from .mask_decoder import MaskDecoder
from .image_encoder import ImageEncoderViT 
from .transformer import TwoWayTransformer
from .common import LayerNorm2d

from typing import Any, Optional, Tuple, Type

from reprlib import recursive_repr
import numpy as np
from torch.nn import functional as F
from .block import MLFusion, DetailEnhancement


class MDSAM(nn.Module):
    def __init__(self, img_size = 512, norm = nn.BatchNorm2d, act = nn.ReLU):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(256 // 2)

        self.image_embedding_size = [img_size // 16, img_size // 16]
        self.img_size = img_size

        self.image_encoder = ImageEncoderViT(depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256)

        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth = 2,
                embedding_dim = 256,
                mlp_dim = 2048,
                num_heads = 8
            ),
            transformer_dim=256,
            norm = norm,
            act = act
        )
        self.deep_feautre_conv = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(256, 64, 3, padding = 1, bias = False),
            norm(64),
            act(),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(64, 32, 3, padding = 1, bias = False),
            norm(32),
            act(),
        )
        self.deep_out_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding = 1, bias = False),
            norm(16),
            act(),
            nn.Conv2d(16, 1, 1)
        )

        self.fusion_block = MLFusion(norm = norm, act = act)

        self.detail_enhance = DetailEnhancement(img_dim = 32, feature_dim = 32, norm = norm, act = act)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, img):
        #get different layer's features of the image encoder
        features_list = self.image_encoder(img)

        #get the output of the last layer of the encoder.
        deep_feature = self.deep_feautre_conv(features_list[-1].contiguous()) #256 * 32 * 32 -> 32 * 128 * 128

        img_feature = self.fusion_block(features_list)

        #extract intermediate outputs for deep supervision to prevent model overfitting on the detail enhancement module.
        img_pe = self.get_dense_pe()
        coarse_mask, feature= self.mask_decoder(img_feature, img_pe)

        coarse_mask = torch.nn.functional.interpolate(coarse_mask,[self.img_size,self.img_size], mode = 'bilinear', align_corners = False)

        mask = self.detail_enhance(img, feature, deep_feature)

        return mask, coarse_mask

    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
               self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args) # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds
