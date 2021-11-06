"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from argparse import Namespace
from typing import Tuple, List
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from torch.nn import functional as F
from revdict.datasets import Word2DefDataset


class ReverseDict(pl.LightningModule):
    """
    A reverse-dictionary.
    """
    # revdict into these!
    def __init__(self, bert_mlm: BertForMaskedLM, word2subs: Tensor, k: int, lr: float):
        super().__init__()
        # -- the only network we need -- #
        self.bert_mlm = bert_mlm
        # -- to be used to compute S_word -- #
        self.word2subs = word2subs
        # -- hyper params --- #
        # should be saved to self.hparams
        self.save_hyperparameters(Namespace(k=k, lr=lr))

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 3, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask, the maximum length)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert_mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, 768)
        H_k = H_all[:, 1: self.hparams['k'] + 1]  # (N, L, 768) -> (N, K, 768)
        S_subword = self.bert_mlm.cls(H_k)  # (N, K, 768) ->  (N, K, |S|)
        return S_subword

    def S_word(self, S_subword: Tensor) -> Tensor:
       
        word2subs = self.word2subs.T.repeat(S_subword.shape[0], 1, 1)  # (|V|, K) -> (N, K, |V|)
        S_word = S_subword.gather(dim=-1, index=word2subs)  # (N, K, |S|) -> (N, K, |V|)
        S_word = S_word.sum(dim=1)  # (N, K, |V|) -> (N, |V|)
        return S_word

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        :param batch: A tuple of X, y, subword_ids; ((N, 3, L), (N,),
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        X, y = batch
        # load the batches on the device.
        X = X.to(self.device)
        y = y.to(self.device)
        S_subword = self.forward(X)  # (N, 3, L) -> (N, K, |S|)
        S_word = self.S_word(S_subword)  # (N, K, |S|) -> (N, |V|)
        loss = F.cross_entropy(S_word, y)  # (N, |V|) -> (N,)
        loss = loss.sum()  # (N,) -> scalar
        self.log('train_loss', loss.item())  # monitor this, while training.
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class Def2Word:

    def __init__(self, rd: ReverseDict, tokenizer: BertTokenizer, words: List[str]):
        self.rd = rd  # a trained reverse dictionary
        self.tokenizer = tokenizer
        self.words = words  # the classes

    def revdict(self, descriptions: List[str], *args, **kwargs) -> List[List[Tuple[str, float]]]:
        # get the X
        word2def = [("", desc) for desc in descriptions]
        X = Word2DefDataset.build_X(word2def, tokenizer=self.tokenizer, k=self.rd.hparams['k'])\
            .to(self.rd.device)
        # get S_subword for this.
        S_subword = self.rd.forward(X)
        S_word = self.rd.S_word(S_subword)  # (N, |V|).
        S_word_probs = F.softmax(S_word, dim=1)  # (N, |V|) -> (N, |V|) softmax along |V|
        results = list()
        for w_probs in S_word_probs.tolist():
            word2score = [
                (word, w_score)
                for word, w_score in zip(self.words, w_probs)
            ]
            # sort and append
            results.append(sorted(word2score, key=lambda x: x[1], reverse=True))
        return results

