from typing import List


class SpecialToken:
    MASK = '[MASK]'
    CLS = '[CLS]'
    SEP = '[SEP]'
    UNK = '[UNK]'
    PAD = '[PAD]'

    @classmethod
    def all(cls) -> List[str]:
        return [
        SpecialToken.MASK,
        SpecialToken.CLS,
        SpecialToken.SEP,
        SpecialToken.UNK,
        SpecialToken.PAD
    ]

WORD_PIECE_SUBWORD_PREFIX = '##'
