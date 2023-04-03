import gzip
import numpy as np
import tqdm
import json
import time
import datetime

import torch


from torch.utils.data import DataLoader

from utils import TextSamplerDatasetLM, MyCollateLM, ids_to_tokens, BPE_to_eval, epoch_time, count_parameters, remove_eos, get_input_output_lm

from model.lm import LanguageModel

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import sacrebleu


def test():
    # ddp_kwargs_1 = DistributedDataParallelKwargs(find_unused_parameters=True)
    ddp_kwargs_1 = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs_1])

    with open('dataset/nl/lm/wmt17_en_de/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)

    reverse_vocab = {id: token for token, id in vocabulary.items()}

    # Get the size of the JSON object
    NUM_TOKENS = len(reverse_vocab.keys())

    # constants

    BATCH_SIZE = 156
    MAX_LEN = 100

    # Step 2: Prepare the model (original transformer) and push to GPU
    model = LanguageModel(
        model_dimension=512,
        vocab_size=NUM_TOKENS,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )

    with gzip.open('dataset/nl/seq2seq/wmt17_en_de/test.de.ids.gz', 'r') as file:
        Y_test = file.read()
        Y_test = Y_test.decode(encoding='utf-8')
        Y_test = Y_test.split('\n')
        Y_test = [np.array([int(x) for x in line.split()]) for line in Y_test]
        Y_test = Y_test[0:20]

    test_dataset = TextSamplerDatasetLM(Y_test, MAX_LEN)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1, collate_fn=MyCollateLM(pad_idx=0))

    model, test_loader = accelerator.prepare(model, test_loader)

    model.load_state_dict(
        torch.load(
            'output/model_lm.pt',
        ),
    )

    model.eval()
    target = []
    predicted = []

    for tgt_test in test_loader:
        tgt_dev_input, tgt_dev_output = get_input_output_lm(tgt_test, window=1)

        sample = model.module.generate_greedy(tgt_dev_input, MAX_LEN)

        target.append([ids_to_tokens(tgt_test.tolist()[i][1:], vocabulary) for i in range(tgt_test.shape[0])])
        predicted.append([ids_to_tokens(sample.tolist()[i][1:], vocabulary) for i in range(tgt_test.shape[0])])

    target_bleu = [BPE_to_eval(sentence, lm=True) for sentence in target]
    predicted_bleu = [BPE_to_eval(sentence, lm=True) for sentence in predicted]

    bleu = sacrebleu.corpus_bleu(predicted_bleu, [target_bleu])

    bleu = bleu.score

    print('BLEU test set', bleu)


if __name__ == '__main__':
    test()