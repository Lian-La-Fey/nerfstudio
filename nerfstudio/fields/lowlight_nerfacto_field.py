from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.fields.nerfacto_field import NerfactoField

class LowLightNerfactoField(NerfactoField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.field_head_raw_rgb = RGBFieldHead()
        self.field_head_illumination = RGBFieldHead()
        self.field_head_gamma = RGBFieldHead()
        self.field_head_concealing = UncertaintyFieldHead()
        
        # illumination decomposition parameters
        self.mlp_illumination = MLP(
            in_dim=self.geo_feat_dim + self.direction_encoding.get_out_dim(),
            num_layers=2,
            layer_width=64,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Softplus(),
            implementation=self.config.implementation
        )
        
        # dynamic gamma correction parameters
        self.gamma_mlp = MLP(
            in_dim=self.geo_feat_dim,
            num_layers=1,
            layer_width=32,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=self.config.implementation
        )
        
        # concealing field (Aleth-NeRF)
        self.concealing_mlp = MLP(
            in_dim=3,  # spatial position
            num_layers=3,
            layer_width=64,
            out_dim=1,  # attenuation factor
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=self.config.implementation
        )

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = super().get_outputs(ray_samples, density_embedding)
        
        # base components
        directions = get_normalized_directions(ray_samples.frustums.directions)
        d_enc = self.direction_encoding(directions.view(-1, 3))
        positions = ray_samples.frustums.get_positions()

        # calculate low-light components
        illumination_input = torch.cat([
            density_embedding.view(-1, self.geo_feat_dim),
            d_enc
        ], dim=-1)
        
        illumination = self.mlp_illumination(illumination_input)
        gamma = self.mlp_gamma(density_embedding) * 3 + 0.5  # scaling to [0.5, 3.5]
        concealing = self.mlp_concealing(positions.view(-1, 3))

        # calculate final RGB
        raw_rgb = outputs[FieldHeadNames.RGB]
        enhanced_rgb = (raw_rgb ** gamma) * illumination
        final_rgb = enhanced_rgb * (1 - concealing)

        # create proper update dictionary with FieldHeadNames
        update_dict = {
            FieldHeadNames.RAW_RGB.value: raw_rgb,
            FieldHeadNames.ILLUMINATION.value: illumination,
            FieldHeadNames.GAMMA.value: gamma,
            FieldHeadNames.CONCEALING.value: concealing,
            FieldHeadNames.RGB.value: final_rgb
        }

        outputs.update(update_dict)
        return outputs