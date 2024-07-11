import typing

import torch
from lerf.data.utils.dino_extractor import ViTExtractor
from lerf.data.utils.feature_dataloader import FeatureDataloader
from tqdm import tqdm
from torchvision import transforms
from typing import Tuple
import numpy as np

#Denoisier
import os
import sys
from torch_kmeans import KMeans, CosineSimilarity
parent_dir_path = os.path.abspath(os.path.join(__file__, '../../../../Denoising-ViT'))
if parent_dir_path not in sys.path:
    sys.path.append(parent_dir_path)
import DenoisingViT
#usually 1260 max size, 1050 for vit-L with ROI
def get_img_resolution(H, W, max_size = 1050, p=14):
    if H<W:
        new_W = max_size
        new_H = (int((H/W)*max_size)//p)*p
    else:
        new_H = max_size
        new_W = (int((W/H)*max_size)//p)*p
    return new_H, new_W
class DinoV2DataLoader(FeatureDataloader):
    model_type = "dinov2_vits14"

    def __init__(
            self,
            cfg: dict,
            device: torch.device,
            image_list: torch.Tensor,
            cache_path: str = None,
            pca_dim = 128
    ):
        assert "image_shape" in cfg
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_type).cuda()
        super().__init__(cfg, device, image_list, cache_path)
        self.pca_dim = pca_dim
        data_shape = self.data.shape
        if self.pca_dim != self.data.shape[-1]:
            self.pca_matrix = torch.pca_lowrank(self.data.view(-1, data_shape[-1]), q=self.pca_dim)[2]
            self.data = torch.matmul(self.data.view(-1, data_shape[-1]), self.pca_matrix).reshape((*data_shape[:-1], self.pca_dim))
        print("Dino data shape", self.data.shape)

    def create(self, image_list):
        self.data = self.get_dino_feats(image_list)

    def get_pca_feats(self,image_list):
        feats = self.get_dino_feats(image_list)
        data_shape = feats.shape
        pca_feats = torch.matmul(feats.view(-1, data_shape[-1]), self.pca_matrix.to(feats)).reshape((*data_shape[:-1], self.pca_dim))
        return pca_feats

    def get_dino_feats(self, image_list):
        """
        image_list: BxCxHxW torch tensor
        returns: BxHxWxC torch tensor of features
        """
        h,w = get_img_resolution(image_list.shape[2], image_list.shape[3])
        preprocess = transforms.Compose([
                        transforms.Resize((h,w),antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
        preproc_image_lst = preprocess(image_list).to(self.device)
        dino_embeds = []
        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            with torch.no_grad():
                descriptors = self.model.get_intermediate_layers(image.unsqueeze(0),reshape=True)[0].squeeze().permute(1,2,0)
            dino_embeds.append(descriptors.cpu().detach())

        return torch.stack(dino_embeds, dim=0)
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

    def get_full_img_feats(self, img_ind) -> torch.Tensor:
        """
        returns BxHxWxC
        """
        return self.data[img_ind].to(self.device)
class DinoDataloader(FeatureDataloader):
    dino_model_type = "dinov2_vitl14"
    dino_stride = 14
    dino_layer = 11
    dino_facet = "key"
    dino_bin = False

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: str = None,
        pca_dim: int = 64,
        use_denoiser: bool = True,
    ):
        assert "image_shape" in cfg
        self.extractor = ViTExtractor(self.dino_model_type, self.dino_stride)
        self.use_denoiser = use_denoiser
        self.device = device
        if self.use_denoiser:
            #arg
            vit_type = "vit_base_patch14_dinov2.lvd142m"
            vit_stride = self.dino_stride
            noise_map_height = 37
            noise_map_width = 37
            enable_pe = False
            load_denoiser_from = parent_dir_path + '/better_DVT/denoiser_layernorm.pth'

            
            #load denoiser model
            vit = DenoisingViT.ViTWrapper(
            model_type=vit_type,
                stride=vit_stride,)
            vit = vit.to(self.device)
            self.denoise_model = DenoisingViT.Denoiser(
                noise_map_height=noise_map_height,
                noise_map_width=noise_map_width,
                feature_dim=vit.n_output_dims,
                vit=vit,
                enable_pe=enable_pe,
            ).to(self.device)
            if load_denoiser_from is not None:
                freevit_model_ckpt = torch.load(load_denoiser_from)["denoiser"]
                #modify the key to remove the trailing .0
                freevit_model_ckpt = {key.replace('.0', '') if '.0' in key else key: value for key, value in freevit_model_ckpt.items()}
                msg = self.denoise_model.load_state_dict(freevit_model_ckpt, strict=False)
            for k in self.denoise_model.state_dict().keys():
                if k in freevit_model_ckpt:
                    print(k, "loaded")
            for p in self.denoise_model.parameters():
                p.requires_grad = False
            self.denoise_model.eval()
            self.denoise_model.to(device)
        self.pca_dim = pca_dim
        super().__init__(cfg, device, image_list, cache_path)
        print("Dino data shape", self.data.shape)
        
        
           
    def create(self, image_list):
        self.data = self.get_dino_feats(image_list)
        data_shape = self.data.shape
        if self.pca_dim != self.data.shape[-1]:
            self.pca_matrix = torch.pca_lowrank(self.data.view(-1, data_shape[-1]), q=self.pca_dim,niter=20)[2]
            self.data = torch.matmul(self.data.view(-1, data_shape[-1]), self.pca_matrix).reshape((*data_shape[:-1], self.pca_dim))
        else:
            self.pca_matrix = torch.eye(data_shape[-1])

    def load(self):
        super().load()
        if self.use_denoiser:
            cache_pca_path = self.cache_path.parent / ("denoised_pca.npy")
        else:    
            cache_pca_path = self.cache_path.parent / ("pca.npy")
        self.pca_matrix = torch.from_numpy(np.load(cache_pca_path)).to(self.device)

    def save(self):
        super().save()
        if self.use_denoiser:
            cache_pca_path = self.cache_path.parent / ("denoised_pca.npy")
        else:
            cache_pca_path = self.cache_path.parent / ("pca.npy")
        np.save(cache_pca_path, self.pca_matrix.cpu().numpy())

    def get_dino_feats(self,image_list, keep_cuda=False):
        h,w = get_img_resolution(image_list.shape[2], image_list.shape[3])
        preprocess = transforms.Compose([
                        transforms.Resize((h,w),antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
        preproc_image_lst = preprocess(image_list).to(self.device)
        dino_embeds = []

        for image in tqdm(preproc_image_lst, desc="dino", total=len(image_list), leave=False):
            with torch.no_grad():
                # DVT
                if self.use_denoiser:
                    # denoiser
                    image=image.unsqueeze(0)
                    output_dict = self.denoise_model(image.to(self.device), return_dict=True)
                    denoised_features = output_dict["pred_denoised_feats"].squeeze(0)
                    denoised_features/=10
                    if keep_cuda:
                        dino_embeds.append(denoised_features)
                    else:
                        dino_embeds.append(denoised_features.cpu().detach())
                    
                else:
                    descriptors = self.extractor.model.get_intermediate_layers(image.unsqueeze(0),reshape=True)[0].squeeze().permute(1,2,0)/10
                    # descriptors = self.extractor.extract_descriptors(
                    #     image.unsqueeze(0),
                    #     [self.dino_layer],
                    #     self.dino_facet,
                    #     self.dino_bin, 
                    #     )             
                    # descriptors = descriptors.reshape(self.extractor.num_patches[0], self.extractor.num_patches[1], -1)
                    if keep_cuda:
                        dino_embeds.append(descriptors)
                    else:
                        dino_embeds.append(descriptors.cpu().detach())
                
         
                
        return torch.stack(dino_embeds, dim=0)
    
    def get_pca_feats(self,image_list, keep_cuda = False):
        feats = self.get_dino_feats(image_list, keep_cuda=keep_cuda)
        data_shape = feats.shape
        pca_feats = torch.matmul(feats.view(-1, data_shape[-1]), self.pca_matrix.to(feats)).reshape((*data_shape[:-1], self.pca_dim))
        return pca_feats
    
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.data.shape[1] / self.cfg["image_shape"][0],
            self.data.shape[2] / self.cfg["image_shape"][1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.data[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)

    def get_full_img_feats(self, img_ind) -> torch.Tensor:
        """
        returns BxHxWxC
        """
        return self.data[img_ind].to(self.device)