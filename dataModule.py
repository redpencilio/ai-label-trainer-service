import pandas as pd
from torch.utils.data import Dataset

from constants import *


class SequenceDataset(Dataset):
    def __init__(self, data, tokenizer, preprocessor):
        df = pd.DataFrame(data)

        # remove biases towards larger presence in dataset
        df = df.groupby('label')

        def select(x):
            size = len(x.index)
            return x.sample(min(size, MAX_CLASS_SIZE))

        df = df.apply(select).reset_index(drop=True)

        df = df.sample(frac=1).reset_index(drop=True)
        df["value"] = df["value"].apply(str)

        self._df = df
        self.id_dict = {}
        self._label_dict = {}
        self._build_label_to_id_dict()
        self._df_labels_to_ids()

        # calculate weights based on size of class in dataset
        weights = self._df["label"].value_counts(normalize=True).sort_index().tolist()
        self.class_weights = [1 - weight for weight in weights]

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer

        self.inputs = self._df.values

        self.label_count = len(self.id_dict.keys())

    def __len__(self):
        return len(self.inputs)

    def id_to_label(self, id):
        return self.id_dict[id]

    def _build_label_to_id_dict(self):
        values = self._df['label'].unique()
        for i, v in enumerate(values):
            self.id_dict[i] = v
            self._label_dict[v] = i

    def _df_labels_to_ids(self):
        self._df["label"] = self._df["label"].apply(lambda x: self._label_dict[x])

    def __getitem__(self, index):
        label, value = self.inputs[index]

        value = self.preprocessor(value)

        tokens = self.tokenizer(value, add_special_tokens=True, return_tensors="pt", max_length=512,
                                truncation=True, padding="max_length")

        for k, v in tokens.items():
            tokens[k] = torch.squeeze(v)

        # model_input = self.preprocessor.convert_to_model_input(tokens)

        return tokens, torch.tensor(label, dtype=torch.long, device=DEVICE)
