"""
Prepare data for training BERT on masked language modeling and next sentence prediction tasks.
requires: python >= 3.10
"""

import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import TypeAlias

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tokenizers import Tokenizer

from tqdm import tqdm

from constants import SpecialToken, WORD_PIECE_SUBWORD_PREFIX
import opts
import utils


TokenType: TypeAlias = str
TokenList: TypeAlias = list[TokenType]

@dataclass
class TrainingInstance:
    tokens: TokenList
    segment_ids: list[int]
    is_next: bool
    masked_positions: list[int]
    masked_labels: list[TokenType]

def create_training_instances(
    data_files: list[str],
    tokenizer: Tokenizer,
    max_seq_length: int,
    num_rounds: int,
    mask_lm_prob: float,
    max_masked_tokens: int,
    whole_word_masking: bool = False,
    short_seq_prob: float = 0.1,
    is_next_prob: float = 0.5,
) -> list[TrainingInstance]:
    documents: list[list[TokenList]] = [[]]
    for data_file in data_files:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Reading data from {data_file}', unit='lines'):
                line = utils.clean_line(line)
                if not line:
                    documents.append([])
                    continue
                tokens = tokenizer.encode(line)
                if not tokens.tokens:  # a string contains only spaces will be encoded to an empty list!
                    continue
                documents[-1].append(tokens.tokens)

    # remove empty documents
    documents = [doc for doc in documents if doc]
    training_instances = []

    for _ in range(num_rounds):
        docs_iter = tqdm(range(len(documents)), desc='Creating training instances')
        for doc_idx in docs_iter:
            training_instances.extend(create_training_instances_from_doc(
                documents,
                doc_idx,
                max_seq_length,
                mask_lm_prob,
                max_masked_tokens,
                tokenizer,
                whole_word_masking=whole_word_masking,
                short_seq_prob=short_seq_prob,
                is_next_prob=is_next_prob,
            ))
    return training_instances

def create_training_instances_from_doc(
    documents: list[list[TokenList]],
    doc_idx: int,
    max_seq_length: int,
    mask_lm_prob: float,
    max_masked_tokens: int,
    tokenizer: Tokenizer,
    whole_word_masking: bool = True,
    short_seq_prob: float = 0.1,
    is_next_prob: float = 0.5,
) -> list[TrainingInstance]:
    doc = documents[doc_idx]
    doc_size = len(doc)
    max_num_tokens = max_seq_length - 3  # excluding [CLS], [SEP], [SEP]

    # 10% of the time, we will use shorter sequences to minimize mismatch between
    # pre-training and fine-tuning
    if random.random() < short_seq_prob:
        max_num_tokens = random.randint(2, max_num_tokens)

    training_instances: list[TrainingInstance] = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < doc_size:
        segment = doc[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == doc_size - 1 or current_length >= max_num_tokens:
            if not current_chunk:
                continue
            # a_size = the number of segments in A
            a_size = 1
            if len(current_chunk) > 1:
                a_size = random.randint(1, len(current_chunk) - 1)

            a_tokens: TokenList = []
            for j in range(a_size):
                a_tokens.extend(current_chunk[j])

            b_tokens: TokenList = []
            if a_size == len(current_chunk) or random.random() < is_next_prob:
                # isNotNext case, so we need to choose random segments for B
                is_next = False
                i -= len(current_chunk) - a_size
                max_b_num_tokens = max_num_tokens - len(a_tokens)

                # take a random document
                random_doc_idx = random.randint(0, len(documents) - 1)
                remaining_iter = 10
                while random_doc_idx == doc_idx and remaining_iter > 0:
                    random_doc_idx = random.randint(0, len(documents) - 1)
                    remaining_iter -= 1

                random_doc = documents[random_doc_idx]
                random_start = random.randint(0, len(random_doc) - 1)
                b_tokens_size = 0
                for j in range(random_start, len(random_doc)):
                    b_tokens.extend(random_doc[j])
                    b_tokens_size += len(random_doc[j])
                    if b_tokens_size >= max_b_num_tokens:
                        break
            else:
                # isNext case
                is_next = True
                for j in range(a_size, len(current_chunk)):
                    b_tokens.extend(current_chunk[j])

            truncate_pair(a_tokens, b_tokens, max_num_tokens)
            assert len(a_tokens) >= 1
            assert len(b_tokens) >= 1

            a_tokens.append(SpecialToken.SEP)
            b_tokens.append(SpecialToken.SEP)

            tokens: TokenList = [SpecialToken.CLS] + a_tokens + b_tokens
            segment_ids = [0] * (len(a_tokens) + 1) + [1] * len(b_tokens)

            masked_tokens, masked_positions, masked_labels = mask_input_tokens(
                tokens,
                tokenizer,
                mask_lm_prob,
                max_masked_tokens,
                whole_word_masking=whole_word_masking,
            )
            instance = TrainingInstance(
                tokens=masked_tokens,
                segment_ids=segment_ids,
                is_next=is_next,
                masked_positions=masked_positions,
                masked_labels=masked_labels,
            )
            training_instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return training_instances

def write_training_instances(
    output_file: str,
    format: str,
    training_instances: list[TrainingInstance],
    tokenizer: Tokenizer,
    max_seq_length: int,
    max_masked_tokens: int,
    save_tokens: bool = False,
) -> None:
    content = []
    headers = [
        'input_ids',
        'input_mask',
        'segment_ids',
        'masked_positions',
        'masked_label_ids',
        'masked_weights',
        'next_sentence_label',
    ]
    if save_tokens:
        headers.extend(['source_tokens', 'target_tokens'])

    pad_token_id = tokenizer.token_to_id(SpecialToken.PAD)
    for instance in training_instances:
        input_ids = [tokenizer.token_to_id(token) for token in instance.tokens]
        masked_label_ids = [tokenizer.token_to_id(token) for token in instance.masked_labels]
        input_mask = [1] * len(input_ids)
        segment_ids = instance.segment_ids
        masked_positions = instance.masked_positions
        is_next = instance.is_next
        masked_weights = [1.0] * len(masked_positions)  # true masked tokens have weight 1.0, padding tokens have weight 0.0

        # we will pad `input_ids`, `input_mask`, and `segment_ids` `to max_seq_length`
        # so they can be batched together
        assert len(input_ids) <= max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)  # I have no idea why we use id 0 for pad tokens here!

        # we do the same with `masked_positions`, `masked_label_ids`, and `masked_weights`
        # to `max_masked_tokens`
        assert len(masked_positions) <= max_masked_tokens
        while len(masked_positions) < max_masked_tokens:
            masked_positions.append(0)
            masked_label_ids.append(pad_token_id)
            masked_weights.append(0.0)

        content.append({
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': instance.segment_ids,
            'masked_positions': instance.masked_positions,
            'masked_label_ids': masked_label_ids,
            'masked_weights': masked_weights,
            'next_sentence_label': 1 if is_next else 0,
        })
        if save_tokens:
            content[-1]['source_tokens'] = instance.tokens
            content[-1]['target_tokens'] = instance.masked_labels

    df = pd.DataFrame(content, columns=headers)
    if format == 'csv':
        df.to_csv(output_file, index=False)
    elif format == 'parquet':
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
    else:
        raise ValueError(f'Unsupported format: {format}')

    print(f'Wrote {len(content)} training instances to {output_file}')

def mask_input_tokens(
    tokens: TokenList,
    tokenizer: Tokenizer,
    masked_lm_prob: float,
    max_masked_tokens: int,
    whole_word_masking: bool = True,
) -> tuple[TokenList, list[int], list[TokenType]]:
    cand_indices: list[list[int]] = []
    for i, token in enumerate(tokens):
        if token == SpecialToken.CLS or token == SpecialToken.SEP:
            continue
        if (
            whole_word_masking and
            len(cand_indices) >= 1 and
            token.startswith(WORD_PIECE_SUBWORD_PREFIX)
        ):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    random.shuffle(cand_indices)
    output_tokens = tokens[:]
    num_masked_tokens = min(
        max_masked_tokens,
        max(1, int(round(len(tokens) * masked_lm_prob))),
    )
    masked: list[tuple[int, TokenType]] = []
    for cand in cand_indices:
        if len(masked) >= num_masked_tokens:
            break
        if len(masked) + len(cand) > num_masked_tokens:
            continue
        for index in cand:
            masked_token = None

            if random.random() < 0.8:
                # replace with [MASK] token
                masked_token = SpecialToken.MASK
            else:
                if random.random() < 0.5:
                    # replace with random token
                    masked_token = tokenizer.id_to_token(random.randint(0, tokenizer.get_vocab_size() - 1))
                else:
                    # keep the original token
                    masked_token = tokens[index]
            output_tokens[index] = masked_token
            masked.append((index, tokens[index]))

    assert len(masked) <= num_masked_tokens
    masked = sorted(masked, key=lambda item: item[0])
    masked_positions = []
    masked_labels = []
    for pos, label in masked:
        masked_positions.append(pos)
        masked_labels.append(label)
    return output_tokens, masked_positions, masked_labels

def truncate_pair(a_tokens: TokenList, b_tokens: TokenList, max_num_tokens: int) -> None:
    total_len = len(a_tokens) + len(b_tokens)
    while total_len > max_num_tokens:
        list_to_pop = a_tokens if len(a_tokens) > len(b_tokens) else b_tokens
        assert len(list_to_pop) > 0
        if random.random() < 0.5:
            # delete token from the beginning
            list_to_pop.pop(0)
        else:
            # delete from the end
            list_to_pop.pop()
        total_len -= 1


def main():
    parser = argparse.ArgumentParser(
        description='Building tokenizer and preparing training instances',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    opts.prepare_training_data_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)

    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)
    tokenizer_save_path = os.path.join(checkpoints_dir, args.tokenizer_basename)
    tokenizer = Tokenizer.from_file(tokenizer_save_path)
    training_instances = create_training_instances(
        args.data_file,
        tokenizer,
        args.max_seq_length,
        args.num_rounds,
        args.masked_lm_prob,
        args.max_masked_tokens,
        whole_word_masking=args.whole_word_masking,
        short_seq_prob=args.short_seq_prob,
        is_next_prob=args.is_next_prob,
    )
    write_training_instances(
        args.output_file,
        args.format,
        training_instances,
        tokenizer,
        args.max_seq_length,
        args.max_masked_tokens,
        args.save_tokens,
    )


if __name__ == '__main__':
    main()
