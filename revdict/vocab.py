from torch import Tensor
from transformers import BertTokenizer

# setting class names
VOCAB_MONO_EN = ['INTJ', 'INFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP', 'ENFP', 'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ESTP', 'ESFP']
VOCAB_MONO_KR = ['전략가', '사색사', '통솔가', '변론가', '옹호자', '중재자', '사회운동가', '활동가', '논리주의자', '수호자', '관리자', '외교관', '재주꾼', '예술가', '사업가', '연예인']
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
