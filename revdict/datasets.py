"""
for pre-processing data
"""
from typing import List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer
from revdict.vocab import VOCAB_MONO_EN, VOCAB_CROSS


class Word2DefDataset(Dataset):
    def __init__(self,
                 word2def: List[Tuple[str, str]],
                 tokenizer: BertTokenizer,
                 k: int,
                 classes: List[str]):
        # (N, 3, L)
        self.X = self.build_X(word2def, tokenizer, k)
        # (N,)
        self.y = self.build_y(word2def, classes)

    @staticmethod
    def build_X(word2def: List[Tuple[str, str]], tokenizer: BertTokenizer, k: int):
        defs = [def_ for _, def_ in word2def]
        lefts = [" ".join(["[MASK]"] * k)] * len(defs)
        rights = defs
        encodings = tokenizer(text=lefts,
                              text_pair=rights,
                              return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True,
                              padding=True)

        return torch.stack([encodings['input_ids'],
                            # token type for the padded tokens? -> they are masked with the
                            # attention mask anyways
                            # https://github.com/google-research/bert/issues/635#issuecomment-491485521
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)

    @staticmethod
    def build_y(word2def: List[Tuple[str, str]], classes: List[str]):
        words = [word for word, _ in word2def]
        return Tensor([
            classes.index(word)
            for word in words
        ]).long()

    def upsample(self, repeat: int):
        """
        this is to try upsampling the batch by simply repeating the instances.
        :return:
        """
        self.X = self.X.repeat(repeat, 1, 1)  # (N, 3, L) -> (N * repeat, 3, L)
        self.y = self.y.repeat(repeat)  # (N,) -> (N * repeat, )

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]


class MonoENWord2Def(Word2DefDataset):
    """
    eng-eng
    """
    def __init__(self, revdict_dataset: List[List[str]], tokenizer: BertTokenizer, k: int):
        classes = VOCAB_MONO_EN
        word2def = self.to_word2def(revdict_dataset)
        super().__init__(word2def, tokenizer, k, classes)

    @staticmethod
    def to_word2def(revdict_dataset) -> List[Tuple[str, str]]:
        return [
            (row[0].strip(), en_def.strip())
            for row in revdict_dataset
            if row[2] == "en"
            for en_def in row[3:]
        ]


class CrossWord2Def(Word2DefDataset):
    """
    eng-kor
    kor-en
    """
    def __init__(self, revdict_dataset: List[List[str]], tokenizer: BertTokenizer, k: int):
        classes = VOCAB_CROSS
        word2def = self.to_word2def(revdict_dataset)
        super().__init__(word2def, tokenizer, k, classes)

    @staticmethod
    def to_word2def(revdict_dataset) -> List[Tuple[str, str]]:
        en2kor = [
            (row[0].strip(), def_.strip())
            for row in revdict_dataset
            for def_ in row[3:]
            if row[2] == "kr"
        ]  # just take all the definitions
        kr2en = [
            (row[1].strip(), def_.strip())
            for row in revdict_dataset
            for def_ in row[3:]
            if row[2] == "en"
        ]  # just take all the definitions
        return en2kor + kr2en
