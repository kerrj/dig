from dataclasses import field
from typing import Dict, List, Type,Literal

from torch.nn import Parameter
from gsplat.sh import spherical_harmonics

from nerfstudio.viewer.viewer_elements import *
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
import math
from nerfstudio.models.splatfacto import projection_matrix
from nerfstudio.model_components import renderers
from nerfstudio.viewer.viewer_elements import *
from lerf.data.utils.dino_dataloader import get_img_resolution
from torchvision.transforms.functional import resize

@dataclass
class DiGModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DiGModel)
    dim: int = 64
    """dim of the thing"""
    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    dino_rescale_factor: int = 4
    """
    How much to upscale rendered dino for supervision
    """

class DiGModel(SplatfactoModel):
    config: DiGModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.gauss_params['dino_feats'] = torch.nn.Parameter(torch.randn((self.num_points, self.config.dim)))
        torch.inverse(torch.ones((1, 1), device="cuda:0"))# https://github.com/pytorch/pytorch/issues/90613
        self.viewer_control = ViewerControl()
        self.click_gaussian = ViewerButton(name="Click Gaussian", cb_hook=self._click_gaussian)
        self.click_location = None
        self.click_handle = None

    def _click_gaussian(self, button: ViewerButton):
        """Start listening for click-based 3D point specification.
        Refer to garfield_interaction.py for more details."""
        def del_handle_on_rayclick(click: ViewerClick):
            self._on_rayclick(click)
            self.click_gaussian.set_disabled(False)
            self.viewer_control.unregister_click_cb(del_handle_on_rayclick)

        self.click_gaussian.set_disabled(True)
        self.viewer_control.register_click_cb(del_handle_on_rayclick)

    def _on_rayclick(self, click: ViewerClick):
        """On click, calculate the 3D position of the click and visualize it.
        Refer to garfield_interaction.py for more details."""

        cam = self.viewer_control.get_camera(500, None, 0)
        cam2world = cam.camera_to_worlds[0, :3, :3]
        import viser.transforms as vtf

        x_pi = vtf.SO3.from_x_radians(np.pi).as_matrix().astype(np.float32)
        world2cam = (cam2world @ x_pi).inverse()
        # rotate the ray around into cam coordinates
        newdir = world2cam @ torch.tensor(click.direction).unsqueeze(-1)
        z_dir = newdir[2].item()
        # project it into coordinates with matrix
        K = cam.get_intrinsics_matrices()[0]
        coords = K @ newdir
        coords = coords / coords[2]
        pix_x, pix_y = int(coords[0]), int(coords[1])
        self.eval()
        outputs = self.get_outputs(cam.to(self.device))
        self.train()
        with torch.no_grad():
            depth = outputs["depth"][pix_y, pix_x].cpu().numpy()
            self.click_feat = outputs["dino"][pix_y, pix_x]

        self.click_location = np.array(click.origin) + np.array(click.direction) * (depth / z_dir)
        import trimesh
        from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
        sphere_mesh = trimesh.creation.icosphere(radius=0.2)
        sphere_mesh.visual.vertex_colors = (0.0, 1.0, 0.0, 1.0)  # type: ignore
        self.click_handle = self.viewer_control.viser_server.add_mesh_trimesh(
            name=f"/click",
            mesh=sphere_mesh,
            position=VISER_NERFSTUDIO_SCALE_RATIO * self.click_location,
        )

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        params = super().get_gaussian_param_groups()
        params['dino_feats'] = [self.gauss_params['dino_feats']]
        return params
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        # Important to allow xys grads to populate properly
        if self.training and self.xys.requires_grad:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        
        dino_feats = None
        DINO_BLOCK = 8
        p_size = 14
        downscale = 1.0 if not self.training else (self.config.dino_rescale_factor*840/max(H,W))/p_size
        h,w = get_img_resolution(H, W)
        if self.training:
            dino_h,dino_w = self.config.dino_rescale_factor*(h//p_size),self.config.dino_rescale_factor*(w//p_size)
        else:
            dino_h,dino_w = H,W
        with torch.no_grad():
            dino_xys, dino_depths, dino_radii, dino_conics, _, dino_num_tiles_hit, _ = project_gaussians(  # type: ignore
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                projmat.squeeze() @ viewmat.squeeze(),
                camera.fx.item()*downscale,
                camera.fy.item()*downscale,
                cx*downscale,
                cy*downscale,
                dino_h,
                dino_w,
                DINO_BLOCK,
            )  # type: ignore
        if crop_ids is not None:
            gauss_crops = self.gauss_params['dino_feats'][crop_ids]
        else:
            gauss_crops = self.gauss_params['dino_feats']
        dino_feats,dino_alpha = rasterize_gaussians(  # type: ignore
                dino_xys,
                dino_depths,
                dino_radii,
                dino_conics,
                dino_num_tiles_hit,  # type: ignore
                gauss_crops,
                opacities.detach(),
                dino_h,
                dino_w,
                DINO_BLOCK,
                background=torch.zeros(self.config.dim, device=self.device),
                return_alpha=True,
            )  # type: ignore
        if self.training:
            dino_feats = torch.where(dino_alpha[...,None] > 0, dino_feats / (dino_alpha[...,None].detach()), torch.zeros(self.config.dim, device=self.device))
        if not self.training:
            dino_feats[dino_alpha<.9]=0
        depth_im = None
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        
        out = {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": background,'dino':dino_feats,'dino_alpha':dino_alpha}
        if hasattr(self,'click_feat') and not self.training and dino_feats is not None:
            #compute similarity to click_feat across dino feats
            sim = (dino_feats - self.click_feat).pow(2).sum(dim=-1).sqrt()[...,None]
            out['click_similarity'] = sim
        return out   # type: ignore
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if outputs['dino'] is not None:
            gt = batch['dino']
            gt = resize(gt.permute(2,0,1), (outputs['dino'].shape[0],outputs['dino'].shape[1])).permute(1,2,0)
            loss_dict['dino_loss'] = torch.nn.functional.mse_loss(outputs['dino'],gt)
            if not hasattr(self,'nearest_ids') or self.num_points != self.nearest_ids.shape[0]:
                from cuml.neighbors import NearestNeighbors
                model = NearestNeighbors(n_neighbors=3)
                means = self.means.detach().cpu().numpy()
                model.fit(means)
                _, self.nearest_ids = model.kneighbors(means)
            # encourage the nearest neighbors to have similar dino feats
            loss_dict['dino_nn_loss'] = 20 * self.gauss_params['dino_feats'][self.nearest_ids].var(dim=1).mean()
        return loss_dict