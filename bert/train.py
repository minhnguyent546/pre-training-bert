"""
Pre-train BERT on masked language modeling and next sentence prediction task
requires: python >= 3.10
"""


import argparse
from contextlib import nullcontext
import os
from tqdm.autonotebook import tqdm

import datasets
from tokenizers import Tokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import (
    BertBase,
    BertConfig,
    BertForPretraining,
    LayerNorm,
)
import opts
import utils


def train_model(args: argparse.Namespace):
    data_save_path = utils.ensure_dir(args.data_save_path)
    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)

    # load trained tokenizer
    tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(checkpoints_dir, args.tokenizer_basename))

    # create data loaders
    saved_dataset: datasets.DatasetDict = datasets.load_from_disk(data_save_path).with_format('torch')
    train_data_loader = DataLoader(
        saved_dataset['train'],
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        saved_dataset['test'],
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    checkpoint_states = None
    if args.from_checkpoint is None:
        print('Starting training from scratch')
        bert_config = BertConfig(
            vocab_size=tokenizer.get_vocab_size(),
            type_vocab_size=args.type_vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            max_seq_length=args.seq_length,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            activation=args.activation,
        )
    else:
        print(f'Loading states from checkpoint {args.from_checkpoint}')

        checkpoint_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ['model', 'optimizer', 'lr_scheduler', 'config']
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        bert_config = checkpoint_states['config']

    # model, optimizer, lr_scheduler, criterion
    model = BertForPretraining(bert_config)
    model.to(device)
    print(f'Model has {model.num_params()} parameters')
    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(
        model,
        args.optim,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        exclude_module_list=(LayerNorm,),
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate * utils.noam_decay(
            step,
            args.hidden_size,
            args.warmup_steps
        ),
    )
    initial_global_step = 0
    accum_train_mlm_loss = 0.0
    accum_train_nsp_loss = 0.0
    accum_train_mlm_acc = 0.0
    accum_train_nsp_acc = 0.0
    if checkpoint_states is not None:
        model.load_state_dict(checkpoint_states['model'])
        optimizer.load_state_dict(checkpoint_states['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_states['lr_scheduler'])
        if 'global_step' in checkpoint_states:
            initial_global_step = checkpoint_states['global_step']
        if 'accum_train_mlm_loss' in checkpoint_states:
            accum_train_mlm_loss = checkpoint_states['accum_train_mlm_loss']
        if 'accum_train_nsp_loss' in checkpoint_states:
            accum_train_nsp_loss = checkpoint_states['accum_train_nsp_loss']
        if 'accum_train_mlm_acc' in checkpoint_states:
            accum_train_mlm_acc = checkpoint_states['accum_train_mlm_acc']
        if 'accum_train_nsp_acc' in checkpoint_states:
            accum_train_nsp_acc = checkpoint_states['accum_train_nsp_acc']

    # mixed precision training with fp16
    train_dtype = torch.float32
    autocast_context = nullcontext()
    if args.fp16 and torch.cuda.is_available() and device.type == 'cuda':
        train_dtype = torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=train_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(train_dtype == torch.float16))

    # tensorboard
    writer = SummaryWriter(args.expr_name)

    # training loop
    train_steps = args.train_steps
    valid_interval = args.valid_interval
    save_interval = args.save_interval
    model.train()

    train_progress_bar = tqdm(range(initial_global_step, train_steps), desc='Training model')
    global_step = initial_global_step
    while global_step < train_steps:
        torch.cuda.empty_cache()

        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            masked_positions = batch['masked_positions'].to(device)
            masked_label_ids = batch['masked_label_ids'].to(device)
            masked_weights = batch['masked_weights'].to(device)
            next_sentence_labels = batch['next_sentence_label'].to(device)
            optimizer.zero_grad()

            with autocast_context:
                masked_lm_logits, nsp_logits, masked_lm_loss, nsp_loss = model(
                    input_ids,
                    segment_ids=segment_ids,
                    input_mask=input_mask,
                    masked_positions=masked_positions,
                    masked_label_ids=masked_label_ids,
                    masked_weights=masked_weights,
                    next_sentence_labels=next_sentence_labels,
                )
                loss = masked_lm_loss + nsp_loss

            masked_lm_acc = utils.compute_mlm_acc(
                masked_lm_logits,
                masked_label_ids,
                masked_weights,
            )
            nsp_acc = utils.compute_nsp_acc(nsp_logits, next_sentence_labels)

            scaler.scale(loss).backward()

            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, global_step)

            lr_scheduler.step()

            train_progress_bar.set_postfix({
                'mlm_loss': masked_lm_loss.item(),
                'nsp_loss': nsp_loss.item(),
            })
            accum_train_mlm_loss += masked_lm_loss.item()
            accum_train_nsp_loss += nsp_loss.item()
            accum_train_mlm_acc += masked_lm_acc
            accum_train_nsp_acc += nsp_acc

            writer.add_scalar('loss/batch_total_loss', loss.item(), global_step)
            writer.add_scalar('loss/batch_mlm_loss', masked_lm_loss.item(), global_step)
            writer.add_scalar('loss/batch_nsp_loss', nsp_loss.item(), global_step)
            writer.add_scalar('accuracy/batch_mlm_acc', masked_lm_acc, global_step)
            writer.add_scalar('accuracy/batch_nsp_acc', nsp_acc, global_step)
            writer.flush()

            if (global_step + 1) % valid_interval == 0:
                valid_results = eval_model(model, test_data_loader, device)
                writer.add_scalars('loss/mlm_loss', {
                    'train': accum_train_mlm_loss / valid_interval,
                    'valid': valid_results['mlm_loss'],
                }, global_step + 1)
                writer.add_scalars('loss/nsp_loss', {
                    'train': accum_train_nsp_loss / valid_interval,
                    'valid': valid_results['nsp_loss'],
                }, global_step + 1)
                writer.add_scalars('loss/total_loss', {
                    'train': (accum_train_mlm_loss + accum_train_nsp_loss) / valid_interval,
                    'valid': valid_results['loss'],
                }, global_step + 1)
                writer.add_scalars('accuracy/mlm', {
                    'train': accum_train_mlm_acc / valid_interval,
                    'valid': valid_results['mlm_acc'],
                }, global_step + 1)
                writer.add_scalars('accuracy/nsp', {
                    'train': accum_train_nsp_acc / valid_interval,
                    'valid': valid_results['nsp_acc'],
                }, global_step + 1)
                writer.flush()
                accum_train_mlm_loss = 0.0
                accum_train_nsp_loss = 0.0
                accum_train_mlm_acc = 0.0
                accum_train_nsp_acc = 0.0

            if (global_step + 1) % save_interval == 0:
                checkpoint_dict = {
                    'global_step': global_step + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': bert_config,
                    'accum_train_mlm_loss': accum_train_mlm_loss,
                    'accum_train_nsp_loss': accum_train_nsp_loss,
                    'accum_train_mlm_acc': accum_train_mlm_acc,
                    'accum_train_nsp_acc': accum_train_nsp_acc,
                }
                model_save_path = os.path.join(checkpoints_dir, f'bert-{global_step + 1}.pt')
                torch.save(checkpoint_dict, model_save_path)

            global_step += 1
            train_progress_bar.update()
            if global_step >= train_steps:
                break

def eval_model(
    model: BertBase,
    eval_data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    is_training = model.training
    model.eval()

    accum_valid_mlm_loss = 0.0
    accum_valid_nsp_loss = 0.0
    accum_valid_mlm_acc = 0.0
    accum_valid_nsp_acc = 0.0
    batch_iter = tqdm(eval_data_loader, desc='Evaluating model')
    with torch.no_grad():
        for batch in batch_iter:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            masked_positions = batch['masked_positions'].to(device)
            masked_label_ids = batch['masked_label_ids'].to(device)
            masked_weights = batch['masked_weights'].to(device)
            next_sentence_labels = batch['next_sentence_label'].to(device)

            masked_lm_logits, nsp_logits, masked_lm_loss, nsp_loss = model(
                input_ids,
                segment_ids=segment_ids,
                input_mask=input_mask,
                masked_positions=masked_positions,
                masked_label_ids=masked_label_ids,
                masked_weights=masked_weights,
                next_sentence_labels=next_sentence_labels,
            )
            masked_lm_acc = utils.compute_mlm_acc(
                masked_lm_logits,
                masked_label_ids,
                masked_weights,
            )
            nsp_acc = utils.compute_nsp_acc(nsp_logits, next_sentence_labels)

            accum_valid_mlm_loss += masked_lm_loss.item()
            accum_valid_nsp_loss += nsp_loss.item()
            accum_valid_mlm_acc += masked_lm_acc
            accum_valid_nsp_acc += nsp_acc


            batch_iter.set_postfix({
                'mlm_loss': masked_lm_loss.item(),
                'nsp_loss': nsp_loss.item(),
            })

    model.train(is_training)

    num_iters = len(eval_data_loader)
    return {
        'loss': (accum_valid_mlm_loss + accum_valid_nsp_loss) / len(eval_data_loader),
        'mlm_loss': accum_valid_mlm_loss / num_iters,
        'nsp_loss': accum_valid_nsp_loss / num_iters,
        'mlm_acc': accum_valid_mlm_acc / num_iters,
        'nsp_acc': accum_valid_nsp_acc / num_iters,
    }

def main():
    parser = argparse.ArgumentParser(
        description='Training BERT model on MLM and NSP tasks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.train_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)
    train_model(args)


if __name__ == '__main__':
    main()
