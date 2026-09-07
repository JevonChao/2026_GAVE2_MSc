import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, input_ch=3, output_ch=64, activf=nn.ReLU, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(output_ch, output_ch, 3, 1, 1, bias=bias)
        self.conv_block = nn.Sequential(
            self.conv1,
            activf(inplace=True),
            self.conv2,
            activf(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class UpConv(nn.Module):
    def __init__(self, input_ch=64, output_ch=32, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(input_ch, output_ch, 2, 2, bias=bias)
        self.conv_block = nn.Sequential(self.conv)

    def forward(self, x):
        return self.conv_block(x)


class UNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(input_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, 2* base_ch)
        self.conv3 = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4 = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5 = ConvBlock(8 * base_ch, 16 * base_ch)

        # Decoder
        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

    def forward(self, x):

        x1 = self.conv1(x)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4(x)
        x = F.max_pool2d(x4, 2, 2)

        x = self.conv5(x)
        x = self.upconv1(x)
        x = torch.cat((x4, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x3, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv9(x)
        x = self.outconv(x)

        return x


class UNet(UNetModule):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None):
        super().__init__(input_ch, output_ch, base_ch)

    def forward(self, x):
        return [super().forward(x)]


class WNet(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=None):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(output_ch, output_ch, base_ch)

    def forward(self, x):
        first_x = self.first_u(x)
        first_x_sig = torch.sigmoid(first_x)
        second_x = self.second_u(first_x_sig)
        return [first_x, second_x]


class RRUNet(nn.Module):
    """Mosinska et al. approach (without topology loss)"""

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__()
        self.unet_module = UNetModule(input_ch+output_ch, output_ch, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []
        x_size = x.size()
        zero_maps_size = (
            x_size[0],
            3,  # only refine AV3
            x_size[2],
            x_size[3],
        )
        zero_maps = torch.zeros(zero_maps_size).to(x.device)
        x_maps = torch.cat((x, zero_maps), dim=1)

        pred = self.unet_module(x_maps)
        predictions.append(pred)

        for _ in range(self.num_iterations):
            pred = torch.sigmoid(pred)
            x_maps = torch.cat((x, pred[:, :3, :, :]), dim=1)
            pred = self.unet_module(x_maps)
            predictions.append(pred[:, :3, :, :])

        return predictions


class RRWNetAll(nn.Module):
    """Network with all channels refined using a second recurrent
    UNet.
    """
    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__()
        self.first_u = UNetModule(input_ch, output_ch, base_ch)
        self.second_u = UNetModule(3, 3, base_ch)
        self.num_iterations = num_iterations

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        pred_1 = torch.sigmoid(pred_1)

        pred_2 = self.second_u(pred_1[:, :3, :, :])
        predictions.append(pred_2)

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = self.second_u(pred_2)
            predictions.append(pred_2)

        return predictions


class RRWNet(RRWNetAll):
    """RRWNetAll but refining only A/V maps.
    Proposed in the paper.
    """

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3, :, :]

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x, vis_fea):
        b, c, _, _ = vis_fea.size()
        y = self.avg_pool(vis_fea).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor
    
class ChannelAttentionConv(nn.Module):
    """ 
    Channel attention mechanism – 1D convolution version
    """
    def __init__(self, in_channel, gamma = 2, b = 1):
        """ 
            Initialization
            - channel: number of channels in the input feature map
            - gamma: two coefficients in the formula
            - b: two coefficients in the formula
        """
        super(ChannelAttentionConv, self).__init__()
        # Adaptively adjust kernel size based on the number of input channels
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))

        # Force the kernel size to be odd
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # pooling
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # 1D convolution
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size,
                              padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(in_channel, in_channel//2, kernel_size=1,  bias=False)

    def forward(self, X):
        """ 
        Forward propagation
        """
        # Global pooling [b,c,h,w]==>[b,c,1,1]
        avg_x = self.avg_pooling(X)
        max_x = self.max_pooling(X)
        # [b,c,1,1]==>[b,1,c] =1D Conv=> [b,1,c]==>[b,c,1,1]
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Normalise the attention weights
        v = self.sigmoid(avg_out + max_out)
        # Multiply the input feature map by the channel weights  [b,c,h,w]
        return  X * v
    
class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        # output_tensor = self.cSE(input_tensor) +  self.sSE(input_tensor)
        return output_tensor



class CrossModalAttention(nn.Module):
    """
    Cross-modal attention fusion.

    FFA_A and FFA_AV each produce an attention map that separately guides the
    CFP features, so arterial and venous information are reinforced along
    independent paths. This avoids the venous signal suppressing the arterial
    signal, as happens under simple addition.
    """
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.attn_a = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )
        self.attn_av = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, fea_rgb, fea_a, fea_av):
        attn_a = self.attn_a(fea_a)      # arterial-phase attention
        attn_av = self.attn_av(fea_av)   # arteriovenous-phase attention
        guided = fea_rgb * attn_a + fea_rgb * attn_av + fea_a + fea_av
        return self.proj(guided)

    
class NewUNetModule(nn.Module):
    def __init__(self, input_ch, output_ch, base_ch, fusion_mode='add'):
        super().__init__()
        self.fusion_mode = fusion_mode

        # Encoder
        # RGB encoder (3 input channels)
        self.conv1_rgb = ConvBlock(3, base_ch)
        self.conv2_rgb = ConvBlock(base_ch, 2 * base_ch)
        self.conv3_rgb = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4_rgb = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5_rgb = ConvBlock(8 * base_ch, 16 * base_ch)

        # FFA encoder (1 input channel)
        self.conv1_a = ConvBlock(1, base_ch)
        self.conv2_a = ConvBlock(base_ch, 2 * base_ch)
        self.conv3_a = ConvBlock(2 * base_ch, 4 * base_ch)
        self.conv4_a = ConvBlock(4 * base_ch, 8 * base_ch)
        self.conv5_a = ConvBlock(8 * base_ch, 16 * base_ch)

        # Decoder
        self.upconv1 = UpConv(16 * base_ch, 8 * base_ch)
        self.conv6 = ConvBlock(16 * base_ch, 8 * base_ch)
        self.upconv2 = UpConv(8 * base_ch, 4 * base_ch)
        self.conv7 = ConvBlock(8 * base_ch, 4 * base_ch)
        self.upconv3 = UpConv(4 * base_ch, 2 * base_ch)
        self.conv8 = ConvBlock(4 * base_ch, 2 * base_ch)
        self.upconv4 = UpConv(2 * base_ch, base_ch)
        self.conv9 = ConvBlock(2 * base_ch, base_ch)

        self.attention1 = SELayer(base_ch)
        self.attention2 = SELayer(2 * base_ch)
        self.attention3 = SELayer(4 * base_ch)
        self.attention4 = SELayer(8 * base_ch)
        self.attention5 = SELayer(16 * base_ch)

        self.ca1 = ChannelAttentionConv(base_ch)
        self.ca2 = ChannelAttentionConv(2 * base_ch)
        # self.ca1 = ChannelSpatialSELayer(base_ch)
        # self.ca2 = ChannelSpatialSELayer(2 * base_ch)

        # self.ca3 = ChannelAttentionConv(8 * base_ch)
        # self.ca4 = ChannelAttentionConv(16 * base_ch)
        self.ca3 = ChannelSpatialSELayer(8 * base_ch)
        self.ca4 = ChannelSpatialSELayer(16 * base_ch)

        self.outconv = nn.Conv2d(base_ch, output_ch, 1, bias=True)

        # ---- Learnable weighted fusion ----
        self.fusion_w1 = nn.Parameter(torch.ones(3))
        self.fusion_w2 = nn.Parameter(torch.ones(3))
        self.fusion_w3 = nn.Parameter(torch.ones(3))
        self.fusion_w4 = nn.Parameter(torch.ones(3))

        # ---- Cross-modal attention fusion ----
        self.cross_attn1 = CrossModalAttention(2 * base_ch)
        self.cross_attn2 = CrossModalAttention(4 * base_ch)
        self.cross_attn3 = CrossModalAttention(8 * base_ch)
        self.cross_attn4 = CrossModalAttention(16 * base_ch)


    def Asy_fusion1(self, fea_rgb, fea_a, fea_av):  # lower-resolution features first, higher-resolution last
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv2_rgb(fea_rgb)
        if self.fusion_mode == 'weighted':
            w = torch.softmax(self.fusion_w1, dim=0)
            add_fea = w[0] * fea_a + w[1] * fea_av + w[2] * fea_rgb
        elif self.fusion_mode == 'attention':
            add_fea = self.cross_attn1(fea_rgb, fea_a, fea_av)
        else:
            add_fea = fea_a + fea_av + fea_rgb
        res = self.ca1(add_fea)
        return res
    def Asy_fusion2(self, fea_rgb, fea_a, fea_av):  # lower-resolution features first, higher-resolution last
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv3_rgb(fea_rgb)
        if self.fusion_mode == 'weighted':
            w = torch.softmax(self.fusion_w2, dim=0)
            add_fea = w[0] * fea_a + w[1] * fea_av + w[2] * fea_rgb
        elif self.fusion_mode == 'attention':
            add_fea = self.cross_attn2(fea_rgb, fea_a, fea_av)
        else:
            add_fea = fea_a + fea_av + fea_rgb
        res = self.ca2(add_fea)
        return res
    def Asy_fusion3(self, fea_rgb, fea_a, fea_av):  # lower-resolution features first, higher-resolution last
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv4_rgb(fea_rgb)
        if self.fusion_mode == 'weighted':
            w = torch.softmax(self.fusion_w3, dim=0)
            add_fea = w[0] * fea_a + w[1] * fea_av + w[2] * fea_rgb
        elif self.fusion_mode == 'attention':
            add_fea = self.cross_attn3(fea_rgb, fea_a, fea_av)
        else:
            add_fea = fea_a + fea_av + fea_rgb
        res = self.ca3(add_fea)
        return res
    def Asy_fusion4(self, fea_rgb, fea_a, fea_av):  # lower-resolution features first, higher-resolution last
        fea_rgb = F.max_pool2d(fea_rgb, 2, 2)
        fea_rgb = self.conv5_rgb(fea_rgb)
        if self.fusion_mode == 'weighted':
            w = torch.softmax(self.fusion_w4, dim=0)
            add_fea = w[0] * fea_a + w[1] * fea_av + w[2] * fea_rgb
        elif self.fusion_mode == 'attention':
            add_fea = self.cross_attn4(fea_rgb, fea_a, fea_av)
        else:
            add_fea = fea_a + fea_av + fea_rgb
        res = self.ca4(add_fea)
        return res
    
    def forward_features(self, feats_rgb, feats_a, feats_av):
        """
        x_rgb: B x C x H x W  #ct
        x_e  #pet
        """
        f0_0 = self.Asy_fusion1(feats_rgb[0], feats_a[1], feats_av[1])  # CFP stage 0 -> FFA stage 1
        f1_0 = self.Asy_fusion2(feats_rgb[1], feats_a[2], feats_av[2])  # CFP stage 1 -> FFA stage 2
        f2_0 = self.Asy_fusion3(feats_rgb[2], feats_a[3], feats_av[3])  # CFP stage 2 -> FFA stage 3
        f3_0 = self.Asy_fusion4(feats_rgb[3], feats_a[4], feats_av[4])  # CFP stage 3 -> FFA stage 4

        return [f0_0, f1_0, f2_0, f3_0]
        return outs_fused
    
    def forward_encoder_rgb(self, rgb):
        x1 = self.conv1_rgb(rgb)
        x = F.max_pool2d(x1, 2, 2)

        x2 = self.conv2_rgb(x)
        x = F.max_pool2d(x2, 2, 2)

        x3 = self.conv3_rgb(x)
        x = F.max_pool2d(x3, 2, 2)

        x4 = self.conv4_rgb(x)
        x = F.max_pool2d(x4, 2, 2)

        x5 = self.conv5_rgb(x)
        return [x1, x2, x3, x4, x5]

    # ======================
    # Encoder: FFA forward pass
    # ======================
    def forward_encoder_a(self, a, feats_rgb):
        rgb_fea_x1, rgb_fea_x2, rgb_fea_x3, rgb_fea_x4, rgb_fea_x5 = feats_rgb
        x1 = self.conv1_a(a)
        x1 = self.attention1(x1, rgb_fea_x1)

        x2 = F.max_pool2d(x1, 2, 2)
        x2 = self.conv2_a(x2)
        x2 = self.attention2(x2, rgb_fea_x2)

        x3 = F.max_pool2d(x2, 2, 2)
        x3 = self.conv3_a(x3)
        x3 = self.attention3(x3, rgb_fea_x3)

        x4 = F.max_pool2d(x3, 2, 2)
        x4 = self.conv4_a(x4)
        x4 = self.attention4(x4, rgb_fea_x4)

        x5 = F.max_pool2d(x4, 2, 2)
        x5 = self.conv5_a(x5)
        x5 = self.attention5(x5, rgb_fea_x5)

        return [x1, x2, x3, x4, x5]

    
    def forward(self, x):
        rgb = x[:, 0:3, :, :]    # B x 3 x H x W
        a = x[:, 3:4, :, :]      # B x 1 x H x W  (keep the channel dim; do not squeeze to 3D)
        av = x[:, 4:5, :, :]     # B x 1 x H x W

        feats_rgb = self.forward_encoder_rgb(rgb)
        
        # 3. A Encoder
        feats_a = self.forward_encoder_a(a, feats_rgb)
        feats_av = self.forward_encoder_a(av, feats_rgb)
        
        # 4. Dual-modal feature fusion
        x1, x2, x3, x4= self.forward_features(feats_rgb, feats_a, feats_av)
        

        x = self.upconv1(x4)
        x = torch.cat((x3, x), dim=1)

        x = self.conv6(x)
        x = self.upconv2(x)
        x = torch.cat((x2, x), dim=1)

        x = self.conv7(x)
        x = self.upconv3(x)
        x = torch.cat((x1, x), dim=1)

        x = self.conv8(x)
        x = self.upconv4(x)
        # x = torch.cat((x1, x), dim=1)

        #x = self.conv9(x)
        x = self.outconv(x)

        return x


class CMRRWNet(RRWNet):
    """RRWNet with a cross-modal attention module in the first UNet.
    Proposed in the paper.
    """

    def __init__(self, input_ch, output_ch, base_ch, num_iterations=5, fusion_mode='add'):
        super().__init__(input_ch, output_ch, base_ch, num_iterations)
        self.first_u = NewUNetModule(input_ch, output_ch, base_ch, fusion_mode=fusion_mode)
        self.second_u = UNetModule(output_ch, 2, base_ch)

    def forward(self, x):
        predictions = []

        pred_1 = self.first_u(x)
        predictions.append(pred_1)
        bv_logits = pred_1[:, 2:3, :, :]
        pred_1 = torch.sigmoid(pred_1)
        bv = pred_1[:, 2:3, :, :]

        pred_2 = self.second_u(pred_1)
        predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        for _ in range(self.num_iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((pred_2, bv_logits), dim=1))

        return predictions
    
    def refine(self, x):
        predictions = []
        bv = x[:, 2:3, :, :]

        pred_2 = self.BHsecond_u(x)
        predictions.append(torch.cat((torch.sigmoid(pred_2), bv), dim=1))

        for _ in range(self.iterations):
            pred_2 = torch.sigmoid(pred_2)
            pred_2 = torch.cat((pred_2, bv), dim=1)
            pred_2 = self.second_u(pred_2)
            predictions.append(torch.cat((torch.sigmoid(pred_2), bv), dim=1))

        return predictions