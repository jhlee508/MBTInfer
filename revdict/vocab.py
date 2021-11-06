from torch import Tensor
from transformers import BertTokenizer

# setting class names
VOCAB_MONO_EN = ['apple', 'banana', 'orange', 'grape', 'pineapple', 'avocado', 'blueberry']
VOCAB_MONO_KR = ['사과', '바나나', '오렌지', '포도', '파인애플', '아보카도', '블루베리']
VOCAB_CROSS = VOCAB_MONO_EN + VOCAB_MONO_KR


def build_word2subs(tokenizer: BertTokenizer, k: int, mode: str) -> Tensor:
    if mode == "mono_en":
        vocab = VOCAB_MONO_EN
    elif mode == "cross":
        vocab = VOCAB_CROSS
    else:
        raise ValueError
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    encoded = tokenizer(text=vocab,
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=k,  # set to k
                        return_tensors="pt")
    input_ids = encoded['input_ids']
    input_ids[input_ids == pad_id] = mask_id  # replace them with masks
    return input_ids
