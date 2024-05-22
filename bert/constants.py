from typing import List


WORD_PIECE_SUBWORD_PREFIX = '##'

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
