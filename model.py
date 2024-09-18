import torch
import timm
import numpy as np
import torch.nn as nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 output_dim=768,
                 num_layer=4,
                 num_head=3,
                 head_use = True,
                 head_latent_use = True
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        if head_use:
            self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2, bias=True)
        if head_latent_use:
            self.head_latent = torch.nn.Linear(emb_dim, output_dim ,bias=True)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.head_use = head_use
        self.head_latent_use = head_latent_use

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')

        if self.head_use:
            patches = self.head(features)
            feat_cls = patches[0]
            patches = patches[1:] 

            mask = torch.zeros_like(patches)
            mask[T-1:] = 1
            mask = take_indexes(mask, backward_indexes[1:] - 1)

            img =  self.patch2img(patches)
            mask = self.patch2img(mask)
        else:
            img = None
            mask = None

        if self.head_latent_use:
            features = self.head_latent(features)
            feat_cls = features[0]
            features = features[1:]

            features_mask = torch.zeros_like(features)
            features_mask[T-1:] = 1
            features_mask = take_indexes(features_mask, backward_indexes[1:] - 1)
        else:
            features = None
            features_mask = None
            
        return img, mask, features, features_mask, feat_cls


class PiLaMIM(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=768,
                 decoder_emb_dim=384,
                 latent_decoder_emb_dim=384,
                 encoder_layer=12,
                 encoder_head=12,
                 decoder_layer=6,
                 decoder_head=12,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.target_encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, 0.0)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.decoder_embed = nn.Linear(emb_dim, decoder_emb_dim, bias=True)
        self.decoder = MAE_Decoder(image_size, patch_size, decoder_emb_dim, emb_dim, decoder_layer, decoder_head, head_use=True, head_latent_use=False)

        self.decoder_latent_embed = nn.Linear(emb_dim, latent_decoder_emb_dim, bias=True)
        self.target_decoder = MAE_Decoder(image_size, patch_size, latent_decoder_emb_dim, emb_dim, decoder_layer, decoder_head, head_use=False, head_latent_use=True)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)

        features_img = self.decoder_embed(features)
        features_latent = self.decoder_latent_embed(features)

        target_features, target_backward_indexes = self.target_encoder(img) 
        
        target_cls = target_features[0,:,:]
        target_features = target_features[1:,:,:]

        predicted_img, mask, _ , _, cls_p = self.decoder(features_img,  backward_indexes)
        _, _, predicted_latent, mask_target, cls_l = self.target_decoder(features_latent,  backward_indexes)  

        predicted_latent = predicted_latent.permute(1,0,2)
        mask_target = mask_target.permute(1,0,2)
        target_features = target_features.permute(1,0,2)

        return predicted_img, mask, predicted_latent, mask_target, target_features, cls_p, cls_l, target_cls
    
    def update_target(self, m):
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm

        self.batch_norm = torch.nn.BatchNorm1d(self.pos_embedding.shape[-1], affine=False, eps=1e-6)
        self.linear = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)
        trunc_normal_(self.linear.weight, std=0.01)
        
        self.head = torch.nn.Sequential(self.batch_norm, self.linear)
        
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
    
        logits = self.head(features[0])

        return logits

