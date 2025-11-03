# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import numpy as np
from utils.pc_util import shift_scale_points


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
        self,
        temperature=10000,
        normalize=False,
        scale=None,
        pos_type="fourier",
        d_pos=None,
        d_in=None,
        query_with_normal=False,
        gauss_scale=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier": 
            assert d_pos is not None
            assert d_pos % 2 == 0
            if query_with_normal:
                d_in = 6
            else:
                d_in = 3
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos
            self.d_in = d_in
    

    
    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        d_out = num_channels // 2
        assert d_out <= max_d_out
        #print(d_in)
        #print(xyz.size())
        #print(xyz.shape[-1])
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize and self.d_in == 6:
            xyz[:, :, :3] = shift_scale_points(xyz[:, :, :3], src_range=input_range)
        
        elif self.normalize and self.d_in == 3:
            xyz = shift_scale_points(xyz, src_range=input_range)
            
        xyz *= 2 * np.pi
        xyz = xyz.contiguous()
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor) 
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                return self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                return self.get_fourier_embeddings(xyz, num_channels, input_range)
        else:
            raise ValueError(f"Unknown {self.pos_type}")

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += (
                f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
            )
        return st
