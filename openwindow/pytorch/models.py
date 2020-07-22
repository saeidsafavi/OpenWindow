import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class CNN6(nn.Module):
    def __init__(self, n_classes, extractor=None):
        super().__init__()

        self.extractor = extractor

        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
            ConvBlock(in_channels=512, out_channels=512),
        )
        self.classifier = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x):
        if self.extractor:
            x = self.extractor(x)  # (N, C, F, T)
        x = self.conv_layers(x)  # (N, 512, F', T')
        x = x.mean(dim=[2, 3])  # (N, 512)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 pool_size=(2, 2), kernel_size=3, **args):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding, bias=False, **args)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool_size = pool_size

    def forward(self, x):
        x = self.bn(self.conv(x)).relu()
        x = F.max_pool2d(x, self.pool_size)
        return x


class LogmelExtractor(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, top_db=None):
        super().__init__()

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=top_db)

    def forward(self, x):
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        return x
