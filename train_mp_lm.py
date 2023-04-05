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

def train(finetuning):

    print(torch.cuda.device_count())

    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    with open('dataset/nl/lm/wmt17_en_de/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    GENERATE_EVERY  = 20
    MAX_LEN = 200
    WARMUP_STEP = 10
    WINDOW_TRAINING = 1

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

    # with gzip.open('dataset/nl/lm/wmt17_en_de/train.de.ids.gz', 'r') as file:
    #     Y_train = file.read()
    #     Y_train = Y_train.decode(encoding='utf-8')
    #     Y_train = Y_train.split('\n')
    #     Y_train = [np.array([int(x) for x in line.split()]) for line in Y_train if line != '']
    #
    # with gzip.open('dataset/nl/lm/wmt17_en_de/valid.merge_en_de.ids.gz', 'r') as file:
    #     Y_dev = file.read()
    #     Y_dev = Y_dev.decode(encoding='utf-8')
    #     Y_dev = Y_dev.split('\n')
    #     Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev if line != '']
    #
    #
    # train_dataset = TextSamplerDatasetLM(Y_train, MAX_LEN)
    # train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=True,
    #                        pin_memory=True, collate_fn=MyCollateLM(pad_idx=0))
    # dev_dataset = TextSamplerDatasetLM(Y_dev, MAX_LEN)
    # dev_loader  = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=8, collate_fn=MyCollateLM(pad_idx=0))

    with gzip.open('dataset/nl/lm/wmt17_en_de/valid.merge_en_de.ids.gz', 'r') as file:
        Y_dev = file.read()
        Y_dev = Y_dev.decode(encoding='utf-8')
        Y_dev = Y_dev.split('\n')
        Y_dev = [np.array([int(x) for x in line.split()]) for line in Y_dev if line != '']
        Y_dev =Y_dev[0:160]


    train_dataset = TextSamplerDatasetLM(Y_dev, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers=8, shuffle=True,
                           pin_memory=True, collate_fn=MyCollateLM(pad_idx=0))
    dev_dataset = TextSamplerDatasetLM(Y_dev, MAX_LEN)
    dev_loader  = DataLoader(dev_dataset, batch_size=1, num_workers=8, collate_fn=MyCollateLM(pad_idx=0))


    model, optimizer, train_loader, dev_loader = accelerator.prepare(model, optimizer, train_loader, dev_loader)

    if finetuning:
        print('finetune')
        model.load_state_dict(
            torch.load(
                'output/model_lm.pt',
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

            tgt_mask = model.module.get_masks_and_count_tokens_trg(inp_tgt)

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

                sample = model.module.generate_greedy(tgt_dev_input, MAX_LEN, cuda=True)

                target.append([ids_to_tokens(tgt_dev.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])
                predicted.append([ids_to_tokens(sample.tolist()[i], vocabulary) for i in range(tgt_dev.shape[0])])

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
                           'output/model_lm.pt'
                           )

                torch.save(optimizer.state_dict(), 'output/optim_lm.bin')


if __name__ == '__main__':
    train(finetuning=False)
