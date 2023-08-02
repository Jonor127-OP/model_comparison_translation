import argparse
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

from torch.utils.data import DataLoader

from utils import TextSamplerDatasetLM, MyCollateLM, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, remove_eos, get_input_output_lm

from model.lm import LanguageModel

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import sacrebleu

def load_vocabulary(dataset_option):
    if dataset_option == 1:
        vocab_path = 'dataset/nl/lm/en2de/wmt17_en_de/vocabulary.json'
    elif dataset_option == 2:
        vocab_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/vocabulary.json'
    else:
        raise ValueError("Invalid dataset option. Choose 1 for dataset/nl/lm/en2de/wmt17_en_de or 2 for dataset/nl/lm/en2fr/wmt14_en_fr.")

    with open(vocab_path, 'r') as f:
        vocabulary = json.load(f)

    return vocabulary

def train(dataset_option, finetuning):

    print(torch.cuda.device_count())

    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    vocabulary = load_vocabulary(dataset_option)
    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    EPOCHS = 30
    BATCH_SIZE = 96
    LEARNING_RATE = 1e-3
    GENERATE_EVERY  = 1
    MAX_LEN = 250
    WARMUP_STEP = 30000
    WINDOW_TRAINING = 0

    # Step 2: Prepare the model (original transformer) and push to GPU
    model = LanguageModel(
        model_dimension=512,
        vocab_size=NUM_TOKENS,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )

    # Step 3: Prepare other training related utilities
    ca = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    # optimizer
    optimizer = get_optimizer(model.parameters(), LEARNING_RATE, wd=0.01)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEP)

    print('number of parameters:', count_parameters(model))

    if dataset_option == 1:
        train_data_path = 'dataset/nl/lm/en2de/wmt17_en_de/train.merge_en_de.ids.gz'
        valid_data_path = 'dataset/nl/lm/en2de/wmt17_en_de/valid.merge_en_de.ids.gz'
    elif dataset_option == 2:
        train_data_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/train.merge_en_fr.ids.gz'
        valid_data_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/valid.merge_en_fr.ids.gz'
    else:
        raise ValueError("Invalid dataset option. Choose 1 for dataset/nl/lm/en2de/wmt17_en_de or 2 for dataset/nl/lm/en2fr/wmt14_en_fr.")

    with gzip.open(train_data_path, 'r') as file:
        Y_train = file.read()
        Y_train = Y_train.decode(encoding='utf-8')
        Y_train = Y_train.split('\n')
        Y_train = [np.array([int(x) for x in line.split()]) for line in Y_train if line != '']

    with gzip.open(valid_data_path, 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev if line != '']

    train_dataset = TextSamplerDatasetLM(Y_train, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=True,
                           pin_memory=True, collate_fn=MyCollateLM(pad_idx=0))
    dev_dataset = TextSamplerDatasetLM(Y_dev, MAX_LEN)
    dev_loader  = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=8, collate_fn=MyCollateLM(pad_idx=0))

    with gzip.open(valid_data_path, 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev if line != '']
        Y_dev = Y_dev[0:500]
    
    train_dataset = TextSamplerDatasetLM(Y_dev, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=True,
                           pin_memory=True, collate_fn=MyCollateLM(pad_idx=0))
    dev_dataset = TextSamplerDatasetLM(Y_dev, MAX_LEN)
    dev_loader  = DataLoader(dev_dataset, batch_size=1, num_workers=8, collate_fn=MyCollateLM(pad_idx=0))

    model, optimizer, train_loader, dev_loader, scheduler= accelerator.prepare(model, optimizer, train_loader, dev_loader, scheduler)

    if finetuning:
        print('finetune')
        model.load_state_dict(
            torch.load(
                'output/model_lm_%.pt'.format(dataset_option),
            ),
        )

    best_bleu = 0
    breakaction = False
    # training
    for i in tqdm.tqdm(range(EPOCHS), desc='training'):
        start_time = time.time()
        model.train()
        if breakaction:
            break

        countdown = 0
        count_loss = 0

        for tgt_train in train_loader:

            inp_tgt, out_tgt = remove_eos(tgt_train, window=WINDOW_TRAINING), tgt_train[:, WINDOW_TRAINING + 1:]
            # print('inp_tgt', inp_tgt)
            # print('out_tgt', out_tgt)

            tgt_mask = model.get_masks_and_count_tokens_trg(inp_tgt)

            countdown += 1

            predicted_log_distributions = model(inp_tgt, tgt_mask)

            # print('predicted_log_distributions', predicted_log_distributions)

            loss = ca(predicted_log_distributions.view(-1, NUM_TOKENS), out_tgt.contiguous().view(-1).type(torch.LongTensor).cuda())

            count_loss += loss.item()

            # print(loss.item())

            # if torch.isnan(loss).item():
            #     breakaction = True
            #     break

            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print('loss =', count_loss)

        # torch.save(model.state_dict(),
        #            'output/model_seq2seq_each_epoch.pt'
        #            )

        if i != 0 and i % GENERATE_EVERY == 0:

            model.eval()
            target = []
            predicted = []
            for tgt_dev in dev_loader:

                tgt_dev_input, tgt_dev_output = get_input_output_lm(tgt_dev, window=WINDOW_TRAINING)

                sample = model.generate_greedy(tgt_dev_input, MAX_LEN, cuda=True)

                # print(sample)
                # print(sample.shape)

                target.append([ids_to_tokens(tgt_dev_output.tolist()[0][1:], vocabulary)])
                predicted.append([ids_to_tokens(sample.tolist()[0][1:], vocabulary)])

            target_bleu = [BPE_to_eval(sentence, lm=True) for sentence in target]
            predicted_bleu = [BPE_to_eval(sentence, lm=True) for sentence in predicted]

            print('target_bleu', target_bleu)
            print('predicted_bleu', predicted_bleu)

            bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])

            bleu = bleu.score

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print('Epoch: {0} | Time: {1}m {2}s, bleu score = {3}'.format(i, epoch_mins, epoch_secs, bleu))

            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.state_dict(),
                           'output/model_lm_%.pt'.format(dataset_option)
                           )

                torch.save(optimizer.state_dict(), 'output/optim_lm.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model Training')
    parser.add_argument('--dataset', type=int, choices=[1, 2], default=1, help='Dataset option: 1 for dataset/nl/lm/en2de/wmt17_en_de, 2 for dataset/nl/lm/en2fr/wmt14_en_fr')
    parser.add_argument('--finetuning', action='store_true', help='Whether to perform finetuning using the pre-trained model')
    args = parser.parse_args()

    train(args.dataset, args.finetuning)
