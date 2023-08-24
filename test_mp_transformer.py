import gzip
import numpy as np
import tqdm
import json
import time
import datetime
import re
import csv

import torch


from torch.utils.data import DataLoader

from utils import TextSamplerDatasetS2S, MyCollateS2S, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, remove_eos

from model.transformer import Transformer

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import sacrebleu


def test():
    # ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_1 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1])
    

    vocab_path = 'dataset/nl/seq2seq/en2de/wmt17_en_de/vocabulary.json'
    
    with open(vocab_path, 'r') as f:
        vocabulary = json.load(f)

    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    BATCH_SIZE = 32
    MAX_LEN = 100

    # Step 2: Prepare the model (original transformer) and push to GPU
    model = Transformer(
        model_dimension=512,
        src_vocab_size=NUM_TOKENS,
        trg_vocab_size=NUM_TOKENS,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )

    with gzip.open('dataset/nl/seq2seq/en2de/wmt17_en_de/test.en.ids.gz', 'r') as file:
        X_test = file.read()
        X_test = X_test.decode(encoding='utf-8')
        X_test = X_test.split('\n')
        X_test = [np.array([int(x) for x in line.split()]) for line in X_test]
        X_test = X_test[0:50]

    with gzip.open('dataset/nl/seq2seq/en2de/wmt17_en_de/test.de.ids.gz', 'r') as file:
        Y_test = file.read()
        Y_test = Y_test.decode(encoding='utf-8')
        Y_test = Y_test.split('\n')
        Y_test = [np.array([int(x) for x in line.split()]) for line in Y_test]
        Y_test = Y_test[0:50]

    test_dataset = TextSamplerDatasetS2S(X_test, Y_test, MAX_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=MyCollateS2S(pad_idx=0))

    model, test_loader = accelerator.prepare(model, test_loader)

    model.load_state_dict(
        torch.load(
            'output/model_lm.pt',
        ),
    )

    model.eval()
    target = []
    predicted = []
    pairs = []

    for src_dev, tgt_dev in test_loader:
        src_mask = src_dev != 0
        src_mask = src_mask[:, None, None, :]

        sample = model.module.generate_greedy(src_dev, src_mask, MAX_LEN)

        sample = accelerator.gather(sample)
        tgt_dev = accelerator.gather(tgt_dev)
        
         # Remove special characters using regex
        source_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(src_dev))
        target_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(tgt_dev))
        predicted_str = re.sub(r'(@@ )|(@@ ?$)', '', ' '.join(sample))
        
        pairs.append([source_str, target_str, predicted_str])

        target.append([ids_to_tokens(tgt_dev.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])
        predicted.append([ids_to_tokens(sample.tolist()[i][1:], vocabulary) for i in range(tgt_dev.shape[0])])

    target_bleu = [BPE_to_eval(sentence) for sentence in target[0]]
    predicted_bleu = [BPE_to_eval(sentence) for sentence in predicted[0]]

    bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])

    bleu = bleu.score

    print('BLEU test set', bleu)

    output_file = "comparisonresult_bleu={0}.csv".format(bleu)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', "Predicted" ])    

        for i, pair in enumerate(pairs):
            writer.writerow(pair[0], pair[1], pair[2])
    
    print('Comparison results exported to', output_file)


if __name__ == '__main__':
    test()