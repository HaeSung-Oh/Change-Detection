import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class AFM(nn.Module):
    def __init__(
        self,
        channels: int,
        mask_size=(256, 256),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.mask_size = mask_size
        self.eps = eps

        # Sigmoid masks cannot be initialized exactly to 1 with finite logits.
        # Use a small gap from identity so that AFM starts as a near-identity transform.
        identity_gap = 1e-2
        init_mask_value = 1.0 - identity_gap
        init_logit = math.log(init_mask_value / (1.0 - init_mask_value))

        self.mask_amplitude = nn.Parameter(
            torch.full((1, channels, mask_size[0], mask_size[1]), init_logit)
        )
        self.mask_phase = nn.Parameter(
            torch.full((1, channels, mask_size[0], mask_size[1]), init_logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape

        if C != self.channels:
            raise ValueError(f"AFM expects {self.channels} channels, but got {C}.")

        if (H, W) != self.mask_size:
            raise ValueError(
                f"AFM expects input size {self.mask_size}, but got {(H, W)}. "
                "Resize the input beforehand or set mask_size to match the input size."
            )

        # FFT with orthonormal normalization.
        freq = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))

        amp = torch.abs(freq) + self.eps
        phase = torch.angle(freq)

        mask_amp = torch.sigmoid(self.mask_amplitude)
        mask_phase = torch.sigmoid(self.mask_phase)

        adj_amp = mask_amp * amp
        adj_phase = mask_phase * phase

        adj_freq = torch.polar(adj_amp, adj_phase)
        out = torch.fft.ifft2(
            torch.fft.ifftshift(adj_freq),
            norm="ortho",
        ).real

        return out


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MultiScaleChangeDecoder(nn.Module):
    """
    ResNet-18-based multi-scale change decoder.
    Input: d1, d2, d3, d4 difference features.
    Output: change logits at the input resolution.
    """

    def __init__(
        self,
        encoder_channels=(64, 128, 256, 512),
        decoder_channels=(64, 128, 256, 256),
        num_classes: int = 1,
    ):
        super().__init__()

        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain four channel sizes: d1, d2, d3, d4.")
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain four channel sizes: dec1, dec2, dec3, dec4.")

        d1_ch, d2_ch, d3_ch, d4_ch = encoder_channels
        dec1_ch, dec2_ch, dec3_ch, dec4_ch = decoder_channels

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        self.dec4 = conv_block(d4_ch, dec4_ch)
        self.dec3 = conv_block(dec4_ch + d3_ch, dec3_ch)
        self.dec2 = conv_block(dec3_ch + d2_ch, dec2_ch)
        self.dec1 = conv_block(dec2_ch + d1_ch, dec1_ch)

        self.head = nn.Conv2d(dec1_ch, num_classes, kernel_size=1)

    def forward(self, d1, d2, d3, d4, input_size):
        x = self.dec4(d4)

        x = F.interpolate(x, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, d3], dim=1)
        x = self.dec3(x)

        x = F.interpolate(x, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, d2], dim=1)
        x = self.dec2(x)

        x = F.interpolate(x, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, d1], dim=1)
        x = self.dec1(x)

        logits = self.head(x)
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


class Proposed(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        input_size=(256, 256),
        encoder_channels=(64, 128, 256, 512),
        decoder_channels=(64, 128, 256, 256),
        pretrained_weights="IMAGENET1K_V1",
    ):
        super().__init__()

        # Pseudo-Siamese pretrained backbones.
        self.backbone1 = resnet18(weights=pretrained_weights)
        self.backbone2 = resnet18(weights=pretrained_weights)

        self.apm_stage0_t1 = AFM(
            channels=3,
            mask_size=input_size,
        )
        self.apm_stage0_t2 = AFM(
            channels=3,
            mask_size=input_size,
        )

        stage0_channels = encoder_channels[0]

        self.adapter = nn.Sequential(
            nn.Conv2d(3, stage0_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stage0_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.decoder = MultiScaleChangeDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def encode(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        s1 = backbone.layer1(x)
        s2 = backbone.layer2(s1)
        s3 = backbone.layer3(s2)
        s4 = backbone.layer4(s3)

        return s1, s2, s3, s4

    def forward(self, t1, t2):
        _, s2_1, s3_1, s4_1 = self.encode(self.backbone1, t1)
        _, s2_2, s3_2, s4_2 = self.encode(self.backbone2, t2)

        afm_img_1 = self.apm_stage0_t1(t1)
        afm_img_2 = self.apm_stage0_t2(t2)

        feat_0_1 = self.adapter(afm_img_1)
        feat_0_2 = self.adapter(afm_img_2)

        # Use Stage-0 AFM features as the finest-scale difference feature.
        diff1 = torch.abs(feat_0_1 - feat_0_2) + 1e-6
        diff2 = torch.abs(s2_1 - s2_2) + 1e-6
        diff3 = torch.abs(s3_1 - s3_2) + 1e-6
        diff4 = torch.abs(s4_1 - s4_2) + 1e-6

        input_size = t1.shape[-2:]
        change = self.decoder(diff1, diff2, diff3, diff4, input_size)

        return change