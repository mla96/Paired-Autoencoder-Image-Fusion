from torchvision import models

from autoencoder_blocks import ConvBlock, DoubleConvBlock, DownBlock, UpBlock
import torch.nn as nn


class AutoEncoder_fundus(nn.Module):

    def __init__(self, n_channels, n_encoder_filters, n_decoder_filters, trainable=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_encoder_filters = n_encoder_filters
        self.n_decoder_filters = n_decoder_filters.insert(0, n_encoder_filters[-1])
        self.trainable = trainable

        self.double_conv_block = DoubleConvBlock(n_channels, n_encoder_filters[0])

        # [32, 64, 128, 256]
        down_blocks = [DownBlock(in_channels, out_channels)
                       for in_channels, out_channels in zip(n_encoder_filters, n_encoder_filters[1:])]
        self.down_blocks = nn.Sequential(*down_blocks)
        self.encoder = nn.Sequential(self.double_conv_block,
                                     self.down_blocks)

        # [128, 64, 32]
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        up_blocks[-1] = UpBlock(n_decoder_filters[-2], n_decoder_filters[-1], trainable=trainable, is_batch_norm=False)
        self.up_blocks = nn.Sequential(*up_blocks)

        # Uses tanh output layer to ensure -1 to 1
        # Potential parameters are kernel_size=1 and padding=0 OR kernel_size=3 and padding=1
        self.out_conv = ConvBlock(n_decoder_filters[-1], n_channels, kernel_size=3, padding=1, activation='tanh', is_batch_norm=False)
        self.decoder = nn.Sequential(self.up_blocks,
                                     self.out_conv)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


class AutoEncoder_FLIO(nn.Module):

    def __init__(self, n_channels, n_encoder_filters, n_decoder_filters, trainable=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_encoder_filters = n_encoder_filters
        self.n_decoder_filters = n_decoder_filters.insert(0, n_encoder_filters[-1])
        self.trainable = trainable

        self.instance_norm = nn.InstanceNorm2d(n_channels)
        self.double_conv_block = DoubleConvBlock(n_channels, n_encoder_filters[0])

        # [32, 64, 128, 256]
        down_blocks = [DownBlock(in_channels, out_channels)
                       for in_channels, out_channels in zip(n_encoder_filters, n_encoder_filters[1:])]
        self.down_blocks = nn.Sequential(*down_blocks)
        # Add self.instance_norm if needed
        self.encoder = nn.Sequential(self.double_conv_block,
                                     self.down_blocks)

        # [128, 64, 32]
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        up_blocks[-1] = UpBlock(n_decoder_filters[-2], n_decoder_filters[-1], trainable=trainable, is_batch_norm=False)
        self.up_blocks = nn.Sequential(*up_blocks)

        # Uses tanh output layer to ensure -1 to 1
        # Potential parameters are kernel_size=1 and padding=0 OR kernel_size=3 and padding=1
        self.out_conv = ConvBlock(n_decoder_filters[-1], n_channels, kernel_size=3, padding=1, activation='tanh')
        self.decoder = nn.Sequential(self.up_blocks,
                                     self.out_conv)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


# RGB fundus image dimensions: 1612, 1612, 3 resized to 512, 512, 3 cropped to 128, 128, 3
# Must be identical to model architecture for transfer learning
class AutoEncoder_ResEncoder_Fundus(nn.Module):
    def __init__(self, n_channels, n_decoder_filters, trainable=False):
        super().__init__()

        resnet = models.resnet34()
        resnet_layers = list(resnet.children())
        self.encoder = nn.Sequential(*resnet_layers[:8])  # Stops right before linear layer

        self.n_channels = n_channels
        self.n_decoder_filters = n_decoder_filters.insert(0, 512)  # Insert here the number of filters the resnet ends on
        self.trainable = trainable

        # [256, 128, 64, 32]
        # The first number is the output from the down blocks, and should be doubled if concatenation for skip connections is happening
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_conv = ConvBlock(n_decoder_filters[-1], n_channels, kernel_size=1, padding=0, activation='tanh')
        self.decoder = nn.Sequential(self.up_blocks,
                                     self.decoder)

    def forward(self, x):
        x = self.encoder(x)  # Dimensions:
        x = self.decoder(x)
        return x

    # Replace with load state dict
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


class AutoEncoder_ResEncoder_FLIO(nn.Module):  # FLIO parameter dimensions: 256, 256, 6 cropped to 128, 128, 3 (or 6?)
    def __init__(self, n_channels, n_decoder_filters, trainable=False):
        super().__init__()

        resnet = models.resnet34()
        resnet_layers = list(resnet.children())
        self.conv2d = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_block = nn.Sequential(*resnet_layers[1:8])  # Stops right before linear layer
        self.encoder = nn.Sequential(self.conv2d,
                                     self.resnet_block)

        self.n_channels = n_channels
        self.n_decoder_filters = n_decoder_filters.insert(0, 512)  # Insert here the number of filters the resnet ends on
        self.trainable = trainable

        # [256, 128, 64, 32]
        # The first number is the output from the down blocks, and should be doubled if concatenation for skip connections is happening
        up_blocks = [UpBlock(in_channels, out_channels, trainable=trainable)
                     for in_channels, out_channels in zip(n_decoder_filters, n_decoder_filters[1:])]
        self.up_blocks = nn.Sequential(*up_blocks)
        self.out_conv = ConvBlock(n_decoder_filters[-1], n_channels, kernel_size=1, padding=0, activation='tanh')
        self.decoder = nn.Sequential(self.up_blocks,
                                     self.out_conv)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
