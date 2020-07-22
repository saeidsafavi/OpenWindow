import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class MappedDataLoader(DataLoader):
    def __init__(self, dataset, device=None, **args):
        super().__init__(dataset, **args)
        self.device = device

    def __iter__(self):
        def _to(data):
            if isinstance(data, (tuple, list)):
                return tuple(_to(item) for item in data)
            return data.to(self.device)

        return map(_to, super().__iter__())


class SegmentedDataset(Dataset):
    def __init__(self, subset, block_length=15, labeler=None, cache=True):
        self.subset = subset
        self.block_length = block_length
        self.labeler = labeler or subset.dataset.label

        self.index_map = []
        audio_lengths = [_padded_length(length, block_length)
                         for length in _audio_lengths(subset)]
        n_blocks = (np.array(audio_lengths) / block_length).astype(int)
        for i, n in enumerate(n_blocks):
            self.index_map += list(zip([i] * n, range(n)))

        self.cache = {} if cache else None

    @property
    def tags(self):
        if not hasattr(self, '_tags'):
            indexes = [file_index for file_index, _ in self.index_map]
            self._tags = self.subset.tags.iloc[indexes]
        return self._tags

    def __getitem__(self, index):
        if self.cache is not None and self.cache.get(index) is not None:
            return self.cache[index]

        file_index, block_index = self.index_map[index]

        # Load audio data
        path = self.subset.audio_paths[file_index]
        sample_rate = self.subset.dataset.sample_rate
        num_frames = self.block_length * sample_rate
        offset = block_index * num_frames
        x, _ = torchaudio.load(path, num_frames=num_frames, offset=offset)

        # Convert to mono
        x = x.mean(dim=0, keepdim=True)

        # Pad data to block length if necessary
        remainder = sample_rate * self.block_length - x.shape[1]
        if remainder > 0:
            x = torch.nn.functional.pad(x, (0, remainder))

        # Get corresponding target value
        y = self.labeler(self.subset, path.name)
        y = torch.as_tensor(y, dtype=torch.long)

        if self.cache is not None:
            self.cache[index] = (x, y)

        return x, y

    def __len__(self):
        return len(self.index_map)


def _audio_lengths(subset):
    bytes_per_sample = 4  # 2-channel 16-bit waveform
    factor = subset.dataset.sample_rate * bytes_per_sample
    return [path.stat().st_size / factor for path in subset.audio_paths]


def _padded_length(audio_length, block_length):
    q, r = divmod(audio_length, block_length)
    if q == 0:
        factor = int(r > 0)
    elif r > block_length * (2 / 3):
        factor = q + 1
    else:
        factor = q

    return factor * block_length
