import gzip
import numpy as np
import tqdm
import json
import time
import datetime

import torch
from torch import nn

from transformers.optimization import get_constant_schedule_with_warmup
from model.optimizer import get_optimizer
from model.utils import get_masks_and_count_tokens_trg

from torch.utils.data import DataLoader

from utils import TextSamplerDataset, MyCollate, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, remove_eos

from model.transformer import Transformer

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import evaluate

def train(finetuning):

    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    with open('dataset/nl/wmt17_en_de/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    EPOCHS = 400
    BATCH_SIZE = 10
    LEARNING_RATE = 5e-4
    GENERATE_EVERY  = 1
    MAX_LEN = 30
    WARMUP_STEP = 0

    # Step 2: Prepare the model (original transformer) and push to GPU
    model = Transformer(
        model_dimension=512,
        src_vocab_size=NUM_TOKENS,
        trg_vocab_size=NUM_TOKENS,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )

    # Step 3: Prepare other training related utilities
    ca = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    metric = evaluate.load('./sacrebleu')

    # optimizer
    optimizer = get_optimizer(model.parameters(), LEARNING_RATE, wd=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEP)

    print('number of parameters:', count_parameters(model))

    # with gzip.open('dataset/nl/wmt17_en_de/train.en.ids.gz', 'r') as file:
    #     X_train = file.read()
    #     X_train = X_train.decode(encoding='utf-8')
    #     X_train = X_train.split('\n')
    #     X_train = [np.array([int(x) for x in line.split()]) for line in X_train]
    #
    # with gzip.open('dataset/nl/wmt17_en_de/train.de.ids.gz', 'r') as file:
    #     Y_train = file.read()
    #     Y_train = Y_train.decode(encoding='utf-8')
    #     Y_train = Y_train.split('\n')
    #     Y_train = [np.array([int(x) for x in line.split()]) for line in Y_train]
    #
    # with gzip.open('dataset/nl/wmt17_en_de/valid.en.ids.gz', 'r') as file:
    #     X_dev = file.read()
    #     X_dev = X_dev.decode(encoding='utf-8')
    #     X_dev = X_dev.split('\n')
    #     X_dev = [np.array([int(x) for x in line.split()]) for line in X_dev]
    #
    # with gzip.open('dataset/nl/wmt17_en_de/valid.de.ids.gz', 'r') as file:
    #     Y_dev = file.read()
    #     Y_dev = Y_dev.decode(encoding='utf-8')
    #     Y_dev = Y_dev.split('\n')
    #     Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev]
    #
    #
    # train_dataset = TextSamplerDataset(X_train, Y_train, MAX_LEN)
    # train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=2, shuffle=True,
    #                        pin_memory=True, collate_fn=MyCollate(pad_idx=0))
    # dev_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    # dev_loader  = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=2, collate_fn=MyCollate(pad_idx=0))

    with gzip.open('dataset/nl/wmt17_en_de/valid.en.ids.gz', 'r') as file:
        X_dev = file.read()
        X_dev = X_dev.decode(encoding='utf-8')
        X_dev = X_dev.split('\n')
        X_dev = [np.array([int(x) for x in line.split()]) for line in X_dev]
        X_dev = X_dev[0:60]

    with gzip.open('dataset/nl/wmt17_en_de/valid.de.ids.gz', 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev]
        Y_dev = Y_dev[0:60]

    train_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=4, shuffle=True,
                           pin_memory=True, collate_fn=MyCollate(pad_idx=0))
    dev_dataset = TextSamplerDataset(X_dev, Y_dev, MAX_LEN)
    dev_loader  = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=MyCollate(pad_idx=0))

    model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)

    if finetuning:
        print('finetune')
        model.load_state_dict(
            torch.load(
                'output/model_seq2seq.pt',
            ),
        )

    best_bleu = 0

    # training
    for i in tqdm.tqdm(range(EPOCHS), desc='training'):
        start_time = time.time()
        model.train()

        countdown = 0

        for src_train, tgt_train in train_loader:

            src_mask = src_train != 0
            src_mask = src_mask[:, None, None, :]

            inp_tgt, out_tgt = remove_eos(tgt_train), tgt_train[:, 1:]

            tgt_mask = get_masks_and_count_tokens_trg(inp_tgt)

            countdown += 1

            predicted_log_distributions = model(src_train, inp_tgt, src_mask, tgt_mask)

            loss = ca(predicted_log_distributions.view(-1, NUM_TOKENS), out_tgt.contiguous().view(-1).type(torch.LongTensor))

            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print('loss =', loss.item())

        # torch.save(model.state_dict(),
        #            'output/model_seq2seq_each_epoch.pt'
        #            )

        if i != 0 and i % GENERATE_EVERY == 0:

            model.eval()
            target = []
            predicted = []
            for src_dev, tgt_dev in dev_loader:
                src_mask = src_dev != 0
                src_mask = src_mask[:, None, None, :]

                sample = model.generate_greedy(src_dev, src_mask, MAX_LEN)

                target.append([ids_to_tokens(tgt_dev.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])
                predicted.append([ids_to_tokens(sample.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])

            target_bleu = [BPE_to_eval(sentence) for sentence in target[0]]
            predicted_bleu = [BPE_to_eval(sentence) for sentence in predicted[0]]

            predicted_bleu = accelerator.gather(predicted_bleu)
            target_bleu = accelerator.gather(target_bleu)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            metric.add_batch(predictions=[predicted_bleu], references=[target_bleu])

            bleu = metric.compute()
            bleu = bleu['score']

            print('Epoch: {0} | Time: {1}m {2}s, bleu score = {3}'.format(i, epoch_mins, epoch_secs, bleu))

            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.state_dict(),
                           'output/model_seq2seq.pt'
                           )

                torch.save(optimizer.state_dict(), 'output/optim_seq2seq.bin')


if __name__ == '__main__':
    train(finetuning=False)
