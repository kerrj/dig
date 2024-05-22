from dataclasses import field
from typing import Dict, List, Type,Literal

from torch.nn import Parameter
# from gsplat.sh import spherical_harmonics

from nerfstudio.viewer.viewer_elements import *
from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel
# from gsplat.project_gaussians import project_gaussians
# from gsplat.rasterize import rasterize_gaussians
from gsplat.rendering import rasterization
from nerfstudio.model_components import renderers
from nerfstudio.viewer.viewer_elements import *
from lerf.data.utils.dino_dataloader import get_img_resolution
from torchvision.transforms.functional import resize
import tinycudann as tcnn
import contextlib
from collections import OrderedDict
@dataclass
class DiGModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DiGModel)
    dim: int = 64
    """Output dimension of the feature rendering"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    dino_rescale_factor: int = 6
    """
    How much to upscale rendered dino for supervision
    """
    num_downscales: int = 0
    gaussian_dim:int = 32
    """Dimension the gaussians actually store as features"""

class DiGModel(SplatfactoModel):
    config: DiGModelConfig

    def populate_modules(self):
        super().populate_modules()
        self.gauss_params['dino_feats'] = torch.nn.Parameter(torch.randn((self.num_points, self.config.gaussian_dim)))
        torch.inverse(torch.ones((1, 1), device="cuda:0"))# https://github.com/pytorch/pytorch/issues/90613
        self.viewer_control = ViewerControl()
        self.click_gaussian = ViewerButton(name="Click Gaussian", cb_hook=self._click_gaussian)
        self.click_location = None
        self.click_handle = None
        self.nn = tcnn.Network(
            n_input_dims=self.config.gaussian_dim,
            n_output_dims=self.config.dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        super().load_state_dict(dict, **kwargs)
        # here we need to do some hacky stuff....
        # Convert gauss_params from ParameterDict to a simple OrderedDict of Tensors
        # This is critical for allowing backprop through the gauss_params
        newdict = OrderedDict()
        for k, v in self.gauss_params.items():
            newdict[k] = torch.Tensor(v)
        del self.gauss_params
        self.gauss_params = newdict

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
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        gps = super().get_param_groups()
        gps['nn_projection'] = list(self.nn.parameters())
        return gps
    
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
        camera_scale_fac = 1.0 / self._get_downscale_factor()
        cx = camera.cx.item() * camera_scale_fac
        cy = camera.cy.item() * camera_scale_fac
        fx = camera.fx.item() * camera_scale_fac
        fy = camera.fy.item() * camera_scale_fac
        W, H = int(camera.width.item() * camera_scale_fac), int(camera.height.item() * camera_scale_fac)
        self.last_size = (H, W)

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
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            sh_degree_to_use = None
        
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        K = torch.tensor([[fx, 0, cx], [0, fy * camera_scale_fac, cy], [0, 0, 1]], device=self.device)
        render_mode = "RGB+ED"
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat[None, :, :],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            compute_means2d_absgrad=True,
            radius_clip=0 if self.training else 1,
            rasterize_mode=self.config.rasterize_mode,
        )
        if self.training and info['means2d'].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"] # [1, N, 2]
        self.radii = info["radii"][0] # [N]
            
        alpha = alpha[0, ...]
        rgb = render[0, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        if render_mode == "RGB+ED":
            depth_im = render[0, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max())
        else:
            depth_im = None

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        
        dino_feats = None
        p_size = 14
        downscale = 1.0 if not self.training else (self.config.dino_rescale_factor*1260/max(H,W))/p_size
        K = torch.tensor([[downscale*fx, 0, downscale*cx], [0, downscale*fy, downscale*cy], [0, 0, 1]], device=self.device)
        h,w = get_img_resolution(H, W)
        if self.training:
            dino_h,dino_w = self.config.dino_rescale_factor*(h//p_size),self.config.dino_rescale_factor*(w//p_size)
        else:
            dino_h,dino_w = H,W

        if crop_ids is not None:
            gauss_crops = self.gauss_params['dino_feats'][crop_ids]
        else:
            gauss_crops = self.gauss_params['dino_feats']
        dino_feats, dino_alpha, _ = rasterization(
            means=means_crop.detach() if self.training else means_crop,
            quats=quats_crop.detach() / quats_crop.detach().norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop.detach()),
            opacities=torch.sigmoid(opacities_crop.detach()).squeeze(-1),
            colors=gauss_crops,
            viewmats=viewmat[None, :, :],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            width=dino_w,
            height=dino_h,
            tile_size=16,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sparse_grad=False,
            backgrounds=torch.zeros((1,self.config.gaussian_dim), device=self.device),
            rasterize_mode=self.config.rasterize_mode,
        )
        alpha_cutoff = 0 if self.training else .8
        dino_feats = torch.where(dino_alpha>alpha_cutoff,dino_feats/dino_alpha.detach(),torch.zeros(1,device='cuda'))
        # dino_feats = torch.where(dino_alpha[...,None] > 0, dino_feats / (dino_alpha[...,None].detach()), torch.zeros(self.config.gaussian_dim, device=self.device))
        nn_inputs = dino_feats.view(-1,self.config.gaussian_dim)
        dino_feats = self.nn(nn_inputs.half()).float().view(dino_h,dino_w,-1)
        
        out = {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": background,'dino':dino_feats,'dino_alpha':dino_alpha[..., None]}
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
            if self.step>1000:
                loss_dict['dino_nn_loss'] = .01*self.gauss_params['dino_feats'][self.nearest_ids].var(dim=1).sum()
        return loss_dict