# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def alt_drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    another kind of drop path, to see whether it solves NAN issues.
    """
    keep_prob = 1 - drop_prob
    if drop_prob == 0.0 or not training:
        return x * keep_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x * random_tensor
    return output


class AltDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(AltDropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return alt_drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Conv(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        kernel_size=-1,
        padding=-1,
        groups=-1,
        dilation=1,
        padding_mode="zeros",
        conv_type="conv_2d",
    ):
        super().__init__()
        # self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

        if groups == -1:
            groups = dim

        self.type = conv_type
        self.ks = kernel_size
        self.padding = padding

        if conv_type == "conv2d":
            self.conv2d = nn.Conv2d(
                dim,
                output_dim,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                groups=groups,
                dilation=dilation,
            )
        elif conv_type == "conv2d_global":
            # kernel_size should equal image size, and padding should be half
            assert padding == kernel_size // 2
            self.conv2d = nn.Conv2d(
                dim,
                output_dim,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="circular",
                groups=groups,
            )
        elif conv_type == "avg_pool":
            self.conv2d = nn.AvgPool2d(
                kernel_size=kernel_size, stride=1, padding=padding
            )
        elif conv_type == "max_pool":
            self.conv2d = nn.MaxPool2d(
                kernel_size=kernel_size, stride=1, padding=padding
            )

        elif conv_type == "nonoverlap":
            self.layer = nn.Linear(
                kernel_size * kernel_size, kernel_size * kernel_size, bias=True
            )
            self.ps = kernel_size

    def forward(self, x, a, b):
        B, N, C = x.shape
        assert N == a * b

        x = x.view(B, a, b, C)

        if self.type == "nonoverlap":
            ps = self.ps
            x = torch.reshape(x, [B, a // ps, ps, b // ps, ps, C])
            x = x.permute(0, 1, 3, 5, 2, 4).reshape(
                B, a // ps, b // ps, C, ps * ps
            )  # N, H/ps, W/ps, C, ps * ps
            x = self.layer(x)
            x = x.reshape(B, a // ps, b // ps, C, ps, ps)
            x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, a, b, C)
            h, w = a, b
            x = x.view(B, N, C)
        else:
            x = x.permute(0, 3, 1, 2)  # B, C, a, b
            x = self.conv2d(x)  # B, C, a, b
            if self.type == "conv2d_global":
                x = x[:, :, :a, :a].roll(
                    shifts=(self.padding - 1, self.padding - 1), dims=(2, 3)
                )
            x = x.permute(0, 2, 3, 1)
            h, w = x.shape[1], x.shape[2]
            x = x.view(B, N, C)

        return x, (h, w)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        output_dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=None,
        pre_norm=True,
        kernel_size=-1,
        padding=-1,
        groups=-1,
        dilation=1,
        padding_mode="zeros",
        conv_type="conv_2d",
        dp_type="original",
    ):
        super().__init__()
        self.dim = dim
        if pre_norm:
            self.norm1 = norm_layer(dim)
        else:
            self.norm1 = nn.Identity()
        self.filter = Conv(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            dilation=dilation,
            padding_mode=padding_mode,
            conv_type=conv_type,
        )
        if dp_type == "original":
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        elif dp_type == "alt":
            self.drop_path = (
                AltDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            out_features=output_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop, dense=nn.Linear)

        if init_values is not None and init_values > 0.0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((output_dim)), requires_grad=True
            )
            # self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1 = None

    def forward(self, x, h, w):
        input = x
        x = self.norm1(x)
        x, (updated_h, updated_w) = self.filter(x, h, w)
        x = self.norm2(x)
        x = self.mlp(x)
        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        # x = input + self.drop_path(x)
        x = self.drop_path(x)
        x[..., : self.dim] = x[..., : self.dim] + input
        return x, (updated_h, updated_w)


class InvertedBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=None,
        pre_norm=True,
        kernel_size=-1,
        padding=-1,
        groups=-1,
        dilation=1,
        padding_mode="zeros",
        conv_type="conv_2d",
        dp_type="original",
    ):
        super().__init__()
        # if pre_norm:
        #     self.norm1 = norm_layer(dim)
        # else:
        #     self.norm1 = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = norm_layer(dim)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.act1 = act_layer()

        # self.filter = Conv(mlp_hidden_dim, kernel_size=kernel_size, padding=padding,
        # groups=groups, dilation=dilation, padding_mode=padding_mode,
        # conv_type=conv_type)

        self.filter = nn.Conv2d(
            mlp_hidden_dim,
            mlp_hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=mlp_hidden_dim,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.act2 = act_layer()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)

        if dp_type == "original":
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        elif dp_type == "alt":
            self.drop_path = (
                AltDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            # self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1 = None

    def forward(self, x):
        input = x

        # pdb.set_trace()
        # x = self.norm(x)
        x = self.fc1(x)
        x = self.act1(x)  # B, N, C

        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, a, b)  # .contiguous()
        # x = x.transpose(1, 2).view(B, C, a, b) #.contiguous()
        x = self.filter(x)
        x = x.view(B, C, N).transpose(1, 2)

        x = self.act2(x)
        x = self.fc2(x)

        if self.gamma_1 is not None:
            x = self.gamma_1 * x
        x = input + self.drop_path(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=None,
        pre_norm=True,
        kernel_size=-1,
        padding=-1,
        groups=-1,
        dilation=1,
        padding_mode="zeros",
        conv_type="conv_2d",
        dp_type="original",
    ):
        super().__init__()

        self.pwconv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=1, padding=0, groups=1, dilation=1
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio, eps=1e-6)

        self.act1 = nn.ReLU(inplace=True)

        if groups == -1:
            groups = dim
        self.dwconv = nn.Conv2d(
            dim * mlp_ratio,
            dim * mlp_ratio,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            padding_mode=padding_mode,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(dim * mlp_ratio, eps=1e-6)
        self.act2 = nn.ReLU(inplace=True)

        self.pwconv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=1, padding=0, groups=1, dilation=1
        )
        self.bn3 = nn.BatchNorm2d(dim, eps=1e-6)

        if dp_type == "original":
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        elif dp_type == "alt":
            self.drop_path = (
                AltDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1 = None

    def forward(self, x):
        input = x
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C).permute(0, 3, 1, 2)

        x = self.pwconv1(x)  # B, C, a, b
        x = self.bn1(x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = self.bn3(x)
        x = x.permute(0, 2, 3, 1).view(B, N, C)

        if self.gamma_1 is not None:
            x = self.gamma_1 * x

        x = input + self.drop_path(x)
        return x


class BasicLayer(nn.Module):
    """A basic ConvH layer for one stage."""

    def __init__(
        self,
        block_type,
        depth,
        dim,
        output_dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path_rate_lis=[],
        cur_depth_base=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values=None,
        pre_norm=True,
        kernel_size=-1,
        padding=-1,
        groups=-1,
        dilation=1,
        padding_mode="zeros",
        conv_type="conv_2d",
        dp_type="original",
    ):
        super().__init__()

        if block_type == "normal":
            block_fn = Block
        elif block_type == "MBConv":
            block_fn = MBConvBlock
        elif block_type == "inverted":
            block_fn = InvertedBlock

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    output_dim=(output_dim if i == depth - 1 else dim),
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path_rate_lis[cur_depth_base + i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    pre_norm=pre_norm,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=groups,
                    dilation=dilation,
                    padding_mode=padding_mode,
                    conv_type=conv_type,
                    dp_type=dp_type,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, h, w):
        for blk in self.blocks:
            x, (h, w) = blk(x, h, w)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class DownLayer(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, dim_in=64, dim_out=128, norm_layer=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm_layer(dim_out)

    def forward(self, x, h, w):
        B, N, C = x.size()
        assert h * w == N
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        if self.norm is not None:
            x = self.norm(x)
        return x, (Hp, Wp)


class ConvVisionTransformerH(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        fixup=True,
        pre_norm=True,
        kernel_size=-1,
        padding=-1,
        groups=-1,
        dilation=1,
        padding_mode="zeros",
        conv_type="conv2d",
        embed_dim=[64, 128, 256, 512],
        output_dim=512,
        depth=[2, 2, 10, 4],
        ln_type="normal",
        dp_type="original",
        block_type="normal",
        out_indices=[0, 1, 2, 3],
        use_ema=False,
        ds_norm=False,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # to keep architecture same with deit version for now
        assert use_mean_pooling == True
        assert use_rel_pos_bias == False
        assert use_shared_rel_pos_bias == False
        assert use_abs_pos_emb == False

        self.block_type = block_type

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.output_dim = output_dim

        if ln_type == "fp32":
            norm_layer = partial(Fp32LayerNorm, eps=1e-6)
        elif ln_type == "fused":
            norm_layer = partial(FusedLayerNorm, eps=1e-6)

        self.patch_embed = nn.ModuleList()
        patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0]
        )
        self.out_indices = out_indices

        self.patch_embed.append(patch_embed)

        self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        ds_norm_layer = norm_layer if ds_norm else None
        for i in range(3):
            patch_embed = DownLayer(embed_dim[i], embed_dim[i + 1], ds_norm_layer)
            self.patch_embed.append(patch_embed)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        self.use_rel_pos_bias = use_rel_pos_bias

        self.layers = nn.ModuleList()
        cur = 0
        for i in range(4):
            # print("using standard block")
            layer = BasicLayer(
                block_type=block_type,
                depth=depth[i],
                dim=embed_dim[i],
                output_dim=(output_dim if i == 3 else embed_dim[i]),
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path_rate_lis=dpr,
                cur_depth_base=cur,
                norm_layer=norm_layer,
                init_values=init_values,
                pre_norm=pre_norm,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                dilation=dilation,
                padding_mode=padding_mode,
                conv_type=conv_type,
                dp_type=dp_type,
            )
            self.layers.append(layer)
            cur += depth[i]

        self.apply(self._init_weights)

        if fixup:
            self.fix_init_weight()

        # for i_layer in range(4):
        #     layer = norm_layer(embed_dim[i_layer])
        #     layer_name = f"norm{i_layer}"
        #     self.add_module(layer_name, layer)
        layer = norm_layer(output_dim)
        layer_name = f"norm3"
        self.add_module(layer_name, layer)

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("N Params:", n_parameters)
        sum_parameters = sum(p.sum() for p in self.parameters() if p.requires_grad)
        print("Sum of Params:", sum_parameters)

        self.use_ema = use_ema

        self._out_features = ["convnext{}".format(i) for i in self.out_indices]
        self._out_feature_channels = {
            "convnext{}".format(i): self.embed_dim[i] for i in self.out_indices
        }
        self._out_feature_strides = {
            "convnext{}".format(i): 2 ** (i + 2) for i in self.out_indices
        }
        self._size_devisibility = 32
        print("self._out_features", self._out_features)
        print("self._out_feature_channels", self._out_feature_channels)
        print("self._out_feature_strides", self._out_feature_strides)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # def fix_init_weight(self):
    #     def rescale(param, layer_id):
    #         param.div_(math.sqrt(2.0 * layer_id))

    #     for layer_id, layer in enumerate(self.blocks):
    #         # pass
    #         # rescale(layer.attn.proj.weight.data, layer_id + 1)
    #         rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # if isinstance(pretrained, str):
        #     self.apply(_init_weights)
        #     print("load pretrained model...")
        #     load_checkpoint(self, pretrained, strict=False)
        #     # load_checkpoint(self, pretrained, strict=False, use_ema=self.use_ema)
        # elif pretrained is None:
        self.apply(_init_weights)
        # else:
        #     raise TypeError("pretrained must be a str or None")

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward_features(self, x):
        # x = self.patch_embed(x)
        # batch_size, seq_len, _ = x.size()

        B, C, H, W = x.shape

        sizes = [56, 28, 14, 7]
        sizes_h = []
        sizes_w = []
        for i in range(4):
            sizes_h.append(sizes[i] * H // 224)
            sizes_w.append(sizes[i] * W // 224)

        outs = {}
        for i in range(4):
            if i == 0:
                x, (Hp, Wp) = self.patch_embed[i](x)
            else:
                x, (Hp, Wp) = self.patch_embed[i](x, sizes_h[i - 1], sizes_w[i - 1])
            x = self.layers[i](x, Hp, Wp)

            # if i in self.out_indices:
            #     norm_layer = getattr(self, f"norm{i}")
            #     x_out = norm_layer(x)

            #     out = (
            #         x_out.view(-1, Hp, Wp, self.num_features[i])
            #         .permute(0, 3, 1, 2)
            #         .contiguous()
            #     )
            #     # print(f"conv{i}", out.shape)
            #     outs["convnext{}".format(i)] = out
        norm_layer = getattr(self, "norm3")
        x_out = norm_layer(x)

        out = x_out.view(-1, Hp, Wp, self.output_dim).permute(0, 3, 1, 2).contiguous()
        # print(f"conv{i}", out.shape)
        # outs["convnext{}".format(i)] = out

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # print(out.shape)
        return out

    def forward(self, x):
        x = self.forward_features(x)
        return x


"""
    patch_size=4,
    mlp_ratio=4,
    drop_path_rate=0.2,
    init_values=1e-6,
    use_abs_pos_emb=False,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    fixup=False,
    pre_norm=False,
    kernel_size=7, 
    padding=3, 
    groups=-1, 
    padding_mode='zeros', 
    conv_type='conv2d',
    embed_dim=[128, 256, 512, 1024], 
    depth=[3, 3, 27, 3], 
    ln_type='normal', 
    dp_type='original',
    block_type='normal', 
    out_indices=[0, 1, 2, 3],
    use_ema=False,
    ds_norm=False,
"""

size2config = {
    "T": {
        "kernel_size": 7,
        "padding": 3,
        "embed_dim": [96, 192, 384, 768],
        "depth": [3, 3, 9, 3],
        "init_values": 1.0,
        "drop_path_rate": 0.2,
        "drop_path_rate_22k": 0.1,
    },
    "S": {
        "kernel_size": 7,
        "padding": 3,
        "embed_dim": [96, 192, 384, 768],
        "depth": [3, 3, 27, 3],
        "init_values": 1.0,
        "drop_path_rate": 0.2,
        "drop_path_rate_22k": 0.1,
    },
    "B": {
        "kernel_size": 7,
        "padding": 3,
        "embed_dim": [128, 256, 512, 1024],
        "depth": [3, 3, 27, 3],
        "init_values": 1.0,
        "drop_path_rate": 0.3,
        "drop_path_rate_22k": 0.1,
    },
    "L": {
        "kernel_size": 7,
        "padding": 3,
        "embed_dim": [192, 384, 768, 1536],
        "depth": [3, 3, 27, 3],
        "init_values": 1.0,
        "drop_path_rate": 0.3,
        "drop_path_rate_22k": 0.1,
    },
    "H": {
        "kernel_size": 7,
        "padding": 3,
        "embed_dim": [256, 512, 1024, 2048],
        "depth": [3, 3, 27, 3],
        "init_values": 1.0,
        "drop_path_rate": 0.3,
        "drop_path_rate_22k": 0.2,
    },
}


def build_convnexttransformer_backbone(
    size="S", dataset="imagenet1k", drop_path=-1.0, dim=-1
):
    """ """
    config = size2config[size]
    if dim == -1:
        output_dim = config["embed_dim"][-1]
    else:
        output_dim = dim

    if drop_path >= 0.0:
        drop_path_rate = drop_path
    elif dataset == "imagenet1k":
        drop_path_rate = config["drop_path_rate"]
    elif dataset == "imagenet22k":
        drop_path_rate = config["drop_path_rate_22k"]
    else:
        drop_path_rate = 0.2
    print(f"Set drop path rate to {drop_path_rate}")

    model = ConvVisionTransformerH(
        embed_dim=config["embed_dim"],
        output_dim=output_dim,
        kernel_size=config["kernel_size"],
        padding=config["padding"],
        init_values=config["init_values"],
        # init_values=None,
        depth=config["depth"],
        # num_heads=config['num_heads'],
        drop_path_rate=drop_path_rate,
        # out_indices=out_indices,
        use_abs_pos_emb=False,
        fixup=False,
        pre_norm=False,
    )
    # print("Initializing", config["pretrained"])
    model.init_weights(None)
    return model, output_dim
