import pandas as pd

from core.dataset import Dataset, DataSubset


LABEL_SET = ['O', 'C', 'OC', 'CO']


class OpenWindow(Dataset):
    def __init__(self, root_dir):
        super().__init__('OpenWindow', root_dir)

        self.label_set = LABEL_SET

        self.sample_rate = 44100

        # Read annotations from file
        annotations_path = self.root_dir / 'annotations.csv'
        tags = pd.read_csv(annotations_path, index_col=0)
        dtype = pd.CategoricalDtype(LABEL_SET, ordered=True)
        tags.label = tags.label.astype(dtype)

        # Create the relevant DataSubsets
        audio_dir = self.root_dir / 'audio'
        self.add_subset(DataSubset('root', self, tags, audio_dir))
        self.add_subset(self['root'].subset('training', tags.fold >= 3))
        self.add_subset(self['root'].subset('validation', tags.fold == 2))
        self.add_subset(self['root'].subset('test', tags.fold == 1))

    @staticmethod
    def label(subset, index=None):
        return label(subset, index)


def label(subset, index=None):
    labels = subset.tags.label.cat.codes
    if index is None:
        return labels
    return labels.loc[index]
