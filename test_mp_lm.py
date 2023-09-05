import gzip
import numpy as np
import tqdm
import json
import time
import datetime
import re
import csv
import argparse
import random

import torch
from torch import nn


from torch.utils.data import DataLoader

from utils import TextSamplerDatasetLM, MyCollateLM, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, remove_eos, get_input_output_lm

from model.lm import LanguageModel

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import sacrebleu

def load_vocabulary(dataset_option):
    if dataset_option == 1:
        vocab_path = 'dataset/nl/lm/en2de/wmt17_en_de/vocabulary.json'
        src_path = 'dataset/nl/lm/en2de/wmt17_en_de/test.merge_en_de.ids.gz'
        
    elif dataset_option == 2:
        vocab_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/vocabulary.json'
        src_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/test.merge_en_fr.ids.gz'
        
    else:
        raise ValueError("Invalid dataset option. Choose 1 for wmt17_en_de or 2 for wmt14_en_fr.")
    
    
    with open(vocab_path, 'r') as f:
        vocabulary = json.load(f)

    return vocabulary, src_path


def test(dataset_option):
    ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_2 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1, ddp_kwargs_2])

    vocabulary, src_path = load_vocabulary(dataset_option)
    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    BATCH_SIZE = 10
    MAX_LEN = 100

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Step 2: Prepare the model (original transformer) and push to GPU
    model = LanguageModel(
        model_dimension=512,
        vocab_size=NUM_TOKENS,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )
    
    if dataset_option == 1:
         src_path = 'dataset/nl/lm/en2de/wmt17_en_de/test.merge_en_de.ids.gz'
    elif dataset_option == 2:
         src_path = 'dataset/nl/lm/en2fr/wmt14_en_fr/test.merge_en_fr.ids.gz'
        

    with gzip.open(src_path, 'r') as file:
        Y_test = file.read()
        Y_test = Y_test.decode(encoding='utf-8')
        Y_test = Y_test.split('\n')
        Y_test = [np.array([int(x) for x in line.split()]) for line in Y_test]
        Y_test = Y_test[0:50]

    test_dataset = TextSamplerDatasetLM(Y_test, MAX_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=MyCollateLM(pad_idx=0))

    model, test_loader = accelerator.prepare(model, test_loader)
    

    if dataset_option == 1:
       folder_path = './output/lm/en2de/' + str(SEED) + str('/')
    elif dataset_option == 2:
       folder_path = './output/lm/en2fr/' + str(SEED) + str('/')
    else:
       raise ValueError("Invalid dataset option. Choose 1 for en2de or 2 for en2fr.")

    model.load_state_dict(
        torch.load(
            folder_path + 'model.pt', map_location='cuda:0'
            ),strict=False
        )

    model.eval()
    target = []
    predicted = []
    pairs = []

    for i, tgt_test in enumerate(test_loader):
        tgt_dev_input, tgt_dev_output = get_input_output_lm(tgt_test, window=0)

        if i == 0:
           tgt_dev = tgt_dev_input

        sample = model.generate_greedy(tgt_dev_input, MAX_LEN, cuda=True)   
        

        sample = accelerator.gather(sample)
        tgt_dev = accelerator.gather(tgt_dev)
        
         # Remove special characters using regex
        source_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(' '.join(ids_to_tokens(tgt_dev_input.tolist()[i], vocabulary)) for i in range(tgt_dev_input.shape[0])))
        target_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(' '.join(ids_to_tokens(tgt_dev_output.tolist()[i][1:], vocabulary)) for i in range(tgt_dev_output.shape[0])))
        predicted_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(' '.join(ids_to_tokens(sample.tolist()[i][1:], vocabulary)) for i in range(sample.shape[0])))

        
        pairs.append([source_str, target_str, predicted_str])

        target.append([ids_to_tokens(tgt_dev.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])
        predicted.append([ids_to_tokens(sample.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])

    target_bleu = [BPE_to_eval(sentence, lm=True) for sentence in target]
    predicted_bleu = [BPE_to_eval(sentence, lm=True) for sentence in predicted]
    print(target_bleu)
    print(predicted_bleu)
    bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])

    bleu = bleu.score

    print('BLEU test set', bleu)

    output_file = "comparisonresult_bleu={0}.csv".format(bleu)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', "Predicted" ])    

        for i, pair in enumerate(pairs):
           row = [pair[0], pair[1], pair[2]]
           writer.writerow(row)

    print('Comparison results exported to', output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Language Model Testing')
    parser.add_argument('--dataset', type=int, choices=[1, 2], default=1, help='Dataset option: 1 for dataset/nl/seq2seq/en2de/wmt17_en_de, 2 for dataset/nl/seq2seq/en2fr/wmt14_en_fr')
    args = parser.parse_args()

    test(args.dataset)