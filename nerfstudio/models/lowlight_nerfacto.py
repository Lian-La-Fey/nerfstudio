from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.fields.lowlight_nerfacto_field import LowLightNerfactoField
from nerfstudio.utils import colormaps


@dataclass
class LowLightNerfactoModelConfig(NerfactoModelConfig):
    # loss multipliers
    illum_smooth_mult: float = 0.1
    gamma_reg_mult: float = 0.01
    concealing_sparsity_mult: float = 0.05
    gradient_weight_mult: float = 0.2
    
    # architecture parameters
    use_illumination_mlp: bool = True
    use_gamma_correction: bool = True
    use_concealing_field: bool = True

class LowLightNerfactoModel(NerfactoModel):
    config: LowLightNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

        # nerfacto field with low-light version
        self.field = LowLightNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=self.field.spatial_distortion,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        
        # gradient-weighted reconstruction loss (RawNeRF)
        gt_rgb = batch["image"].to(self.device)
        raw_rgb = outputs[FieldHeadNames.RAW_RGB.value]
        weights = 1/(raw_rgb.detach() + 1e-6)  # dark region emphasis
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, outputs[FieldHeadNames.RGB], weights=weights)

        # illumination smoothness loss (LLNeRF)
        if self.config.use_illumination_mlp:
            illumination = outputs[FieldHeadNames.ILLUMINATION]
            loss_dict["illum_smooth"] = self.config.illum_smooth_mult * torch.mean(
                torch.abs(illumination[...,:-1,:] - illumination[...,1:,:])
            )

        # gamma regularization (LLNeRF)
        if self.config.use_gamma_correction:
            gamma = outputs[FieldHeadNames.GAMMA]
            loss_dict["gamma_reg"] = self.config.gamma_reg_mult * torch.mean((gamma - 1.0)**2)  # penalize deviation from γ=1

        # concealing field sparsity (Aleth-NeRF)
        if self.config.use_concealing_field:
            concealing = outputs[FieldHeadNames.CONCEALING]
            loss_dict["concealing_sparsity"] = self.config.concealing_sparsity_mult * torch.mean(concealing**2)

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        
        images_dict.update({
            "raw_rgb": colormaps.apply_colormap(outputs["raw_rgb"]),
            "illumination": colormaps.apply_colormap(outputs["illumination"]),
            "gamma": colormaps.apply_colormap(outputs["gamma"]),
            "concealing": colormaps.apply_colormap(outputs["concealing"])
        })

        return metrics_dict, images_dict
    
    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs(ray_bundle)
        
        if not self.training:
            outputs["rgb"] = self._enhance_output(
                outputs["raw_rgb"],
                outputs["illumination"],
                outputs["gamma"],
                outputs["concealing"]
            )
            
        return outputs

    def _enhance_output(self, raw_rgb, illumination, gamma, concealing):
        enhanced_rgb = (raw_rgb ** gamma) * illumination
        return enhanced_rgb * (1 - concealing)