"""
load a trained reverse dictionary model, and inference with it!
"""

import argparse
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

from revdict.models import ReverseDict, Def2Word
from revdict.paths import MONO_EN_CKPT, CROSS_CKPT
from revdict.configs import BERT_MODEL, MBERT_MODEL
from revdict.vocab import build_word2subs, VOCAB_MONO_EN, VOCAB_CROSS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd_mode", type=str,
                        default="mono_en")
    parser.add_argument("--desc", type=str,
                        default="The fruit that monkeys love")
    args = parser.parse_args()
    rd_mode: str = args.rd_mode
    desc: str = args.desc

    device = "cpu" ### RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rd_mode == "mono_en":
        bert_mlm = BertForMaskedLM.from_pretrained(BERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=3, mode=rd_mode)  # this is something I don't really like...
        rd = ReverseDict.load_from_checkpoint(MONO_EN_CKPT, bert_mlm=bert_mlm, word2subs=word2subs)
        rd.eval()  # this is necessary
        rd = rd.to(device)
        vocab = VOCAB_MONO_EN
    elif rd_mode == "cross":
        mbert_mlm = BertForMaskedLM.from_pretrained(MBERT_MODEL)
        tokenizer = BertTokenizer.from_pretrained(MBERT_MODEL)
        word2subs = build_word2subs(tokenizer, k=10, mode=rd_mode)  # this is something I don't really like...
        rd = ReverseDict.load_from_checkpoint(CROSS_CKPT, bert_mlm=mbert_mlm, word2subs=word2subs)
        rd.eval()  # this is necessary
        rd = rd.to(device)
        vocab = VOCAB_CROSS
    else:
        raise ValueError
    def2word = Def2Word(rd, tokenizer, vocab)
    print("[Question] \n>>> {}".format(desc))
    for results in def2word.revdict(descriptions=[desc]):
        print("[Answer]")
        for idx, res in enumerate(results):
            print(">>> {}:".format(idx + 1), res)


if __name__ == '__main__':
    main()
