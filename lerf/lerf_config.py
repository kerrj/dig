"""
LERF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from lerf.data.lerf_datamanager import LERFDataManagerConfig, DiGDataManagerConfig
from lerf.lerf import LERFModelConfig
from lerf.lerf_pipeline import LERFPipelineConfig
from lerf.dig import DiGModelConfig

from garfield.garfield_gaussian_pipeline import GarfieldGaussianPipelineConfig
"""
Swap out the network config to use OpenCLIP or CLIP here.
"""
from lerf.encoders.clip_encoder import CLIPNetworkConfig
from lerf.encoders.openclip_encoder import OpenCLIPNetworkConfig

lerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.3),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(12, 12),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=24,
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16",
                clip_model_pretrained="laion2b_s34b_b88k",
                clip_n_dims=512,
            ),
            #  You can swap the type of input encoder by specifying different NetworkConfigs, the one below uses OpenAI CLIP, the one above uses OpenCLIP
            # network=CLIPNetworkConfig(
            #     clip_model_type="ViT-B/16", clip_n_dims=512
            # )
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=30000
                ),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(
                    lr=1e-2, eps=1e-15, weight_decay=1e-9
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=4000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for LERF",
)
lerf_method_big = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-big",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                # NOTE: exceeding 16 layers per hashgrid causes a segfault within Tiny CUDA NN, so instead we compose multiple hashgrids together
                hashgrid_sizes=(19, 19),
                hashgrid_layers=(16, 16),
                hashgrid_resolutions=((16, 128), (128, 512)),
                num_lerf_samples=32,
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-L-14",
                clip_model_pretrained="laion2b_s32b_b82k",
                clip_n_dims=768,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=30000
                ),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(
                    lr=1e-2, eps=1e-15, weight_decay=1e-9
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=3000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A larger version of LERF with a higher memory footprint, bigger CLIP model, and more hashgrid capacity",
)

lerf_method_lite = MethodSpecification(
    config=TrainerConfig(
        method_name="lerf-lite",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=LERFPipelineConfig(
            datamanager=LERFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(train_split_fraction=0.99),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=LERFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hashgrid_sizes=(19,),
                hashgrid_layers=(16,),
                hashgrid_resolutions=((16, 512),),
                num_lerf_samples=12,
            ),
            network=OpenCLIPNetworkConfig(
                clip_model_type="ViT-B-16",
                clip_model_pretrained="laion2b_s34b_b88k",
                clip_n_dims=512,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=30000
                ),
            },
            "lerf": {
                "optimizer": RAdamOptimizerConfig(
                    lr=1e-2, eps=1e-15, weight_decay=1e-9
                ),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3, max_steps=7000
                ),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=5000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A lightweight version of LERF designed to work on smaller GPUs",
)
from pathlib import Path
dig_method = MethodSpecification(
    config=TrainerConfig(
        method_name="dig",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=6000,
        mixed_precision=False,
        pipeline=GarfieldGaussianPipelineConfig(#use this for overlaying dino on top of a garfield trained model
        # pipeline=VanillaPipelineConfig(#use this for JUST training DINO
            datamanager=DiGDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True,train_split_fraction=0.99),
            ),
            model=DiGModelConfig(),
            garfield_ckpt=Path("outputs/garfield_plushie/garfield/2024-02-29_165759/config.yml")
            # garfield_ckpt = Path("outputs/articulated_objects/garfield/2024-04-15_222909/config.yml")
            # garfield_ckpt=Path("outputs/buddha_balls/garfield/2024-04-08_155953/config.yml")
            # garfield_ckpt = Path("outputs/table_scan/garfield/2024-03-21_120025/config.yml")
            # garfield_ckpt = Path("outputs/tissue_scan/garfield/2024-03-21_135147/config.yml")
            # garfield_ckpt = Path("outputs/boops_mug/garfield/2024-03-18_180854/config.yml")
            # garfield_ckpt = Path("outputs/matt_hand/garfield/2024-05-02_164559/config.yml")
            # garfield_ckpt = Path("outputs/nerfgun2/garfield/2024-03-13_140635/config.yml")
            # garfield_ckpt = Path("outputs/nerfgun3/garfield/2024-05-03_165745/config.yml")
            # garfield_ckpt= Path("outputs/boops_poly/garfield/2024-05-05_185442/config.yml")
            # garfield_ckpt = Path("outputs/nerfgun4/garfield/2024-05-06_100908/config.yml")
            # garfield_ckpt = Path("outputs/bww_faucet/garfield/2024-05-07_140338/config.yml")
            # garfield_ckpt = Path("outputs/buddha_balls_poly/garfield/2024-05-08_115907/config.yml")
            # garfield_ckpt = Path("outputs/painter_sculpture/garfield/2024-05-10_130257/config.yml")
            # garfield_ckpt = Path("outputs/office_chair/garfield/2024-05-10_150230/config.yml")
            # garfield_ckpt = Path("outputs/cmk_tpose/garfield/2024-05-14_140318/config.yml")
            # garfield_ckpt = Path("outputs/mac_charger/garfield/2024-05-15_095130/config.yml")
            # garfield_ckpt = Path("outputs/cal_bear/garfield/2024-05-17_141946/config.yml")
            # garfield_ckpt = Path("outputs/mac_charger2/garfield/2024-05-17_151152/config.yml")
            # garfield_ckpt = Path("outputs/glue_gun/garfield/2024-05-17_160116/config.yml")
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-5,
                    max_steps=6000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "dino_feats": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=6000,
                ),
            },
            "nn_projection": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=6000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Base config for DiG",
)
